//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <vector>
//#include <strings>
#include <string>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <mutex>
#include <unordered_set>
#include <memory>

#ifndef NOCDUA
#include "cuda_testkernel.h"

#include "heatmap_cuda.h"
#include "heatmap_seq.cpp"
#endif
#include <stdlib.h>

std::mutex gridMutex;
std::unordered_set<long long> reservedPositions; // Stores occupied positions


void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation, int max_threads)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	if (implementation == Ped::CUDA) {
        cuda_test();
    }
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

    int N = agents.size();

    posX.resize(N);
    posY.resize(N);
    destX.resize(N);
    destY.resize(N);
    desiredX.resize(N);
    desiredY.resize(N);

    numWaypoints.resize(N);
    currentWaypointIndex.resize(N, 0);
    waypointsX.resize(N);
    waypointsY.resize(N);
    waypointR.resize(N);

    for (int i = 0; i < (int)agents.size(); ++i) {
        Ped::Tagent* agent = agents[i];        // get pointer to this agent
        int wpCount = agent->getNumWaypoints();
        for (int j = 0; j < wpCount && j < MAX_WAYPOINTS; ++j) {
            waypointsX[i][j] = (float)agent->getWaypoint(j)->getx();
            waypointsY[i][j] = (float)agent->getWaypoint(j)->gety();
            waypointR[i][j]  = (float)agent->getWaypoint(j)->getr();
        }
    }

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	if (max_threads > 0) this->setMaxThreads(max_threads);

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

    int regionWidth  = worldWidth  / numRegionsX;
    int regionHeight = worldHeight / numRegionsY;

    if (worldWidth <= 0 || worldHeight <= 0) {
        throw std::runtime_error("Invalid world dimensions");
    }

    regions.clear();
    regions = std::vector<Region>(numRegionsX * numRegionsY);

    int index = 0;
    for (int rx = 0; rx < numRegionsX; ++rx) {
        for (int ry = 0; ry < numRegionsY; ++ry) {

            Region& r = regions[index++];

            r.x0 = rx * regionWidth;
            r.y0 = ry * regionHeight;
            // Ensure last region doesn't exceed world size
            if (rx == numRegionsX - 1) r.x1 = worldWidth;
            if (ry == numRegionsY - 1) r.y1 = worldHeight;
        }
    }

    // Assign agents to regions
    for (Ped::Tagent* agent : agents) {
        int x = agent->getX();
        int y = agent->getY();
        for (Region& r : regions) {
            if (x >= r.x0 && x < r.x1 && y >= r.y0 && y < r.y1) {
                r.agents.push_back(agent);
                break;
            }
        }
    }

    cellLocks.clear();
    cellLocks.resize(worldWidth);

    for (int x = 0; x < worldWidth; ++x) {
        cellLocks[x].resize(worldHeight);
        for (int y = 0; y < worldHeight; ++y) {
            cellLocks[x][y] = std::make_unique<std::mutex>();
        }
    }

    // --- Initialize occupancy grid ---
    cellOccupied.clear();
    cellOccupied.resize(worldWidth);

    for (int x = 0; x < worldWidth; ++x) {
        cellOccupied[x].resize(worldHeight, false);
    }
}


/*static Ped::IMPLEMENTATION get_impl_env_or_fallback(Ped::IMPLEMENTATION fallback) {
    const char* env = std::getenv("PED_IMPL");
    if (!env || !*env) return fallback;

    std::string s(env);
    if (s == "seq" || s == "serial") return Ped::SEQ;
    if (s == "omp") return Ped::OMP;
    if (s == "threads" || s == "thr" || s == "pthread") return Ped::PTHREAD;

    return fallback;
}
*/

void Ped::Model::setMaxThreads(int maxThreads) {
    if (maxThreads < 0) maxThreads = 0;
    this->max_threads = maxThreads;
}

int Ped::Model::getMaxThreads() const {
    return this->max_threads;
}


static int get_max_threads_from_env() {
    // OpenMP standard variable
    const char* omp = std::getenv("OMP_NUM_THREADS");
    if (omp && *omp) {
        int v = std::atoi(omp);
        if (v > 0) return v;
    }

    // Your C++ threads variable
    const char* thr = std::getenv("PED_THREADS");
    if (thr && *thr) {
        int v = std::atoi(thr);
        if (v > 0) return v;
    }

    return 0; // default/runtime
}

/*
static unsigned int get_thread_count_env_or_hw(unsigned int fallback_hw) {
    const char* env = std::getenv("PED_THREADS");
    if (env && *env) {
        int v = std::atoi(env);
        if (v > 0) return (unsigned int)v;
    }
    return fallback_hw;
}
*/


static unsigned int get_thread_count(int model_max_threads, unsigned int fallback_hw) {
    if (model_max_threads > 0) return (unsigned int)model_max_threads;

    const char* env = std::getenv("PED_THREADS");
    if (env && *env) {
        int v = std::atoi(env);
        if (v > 0) return (unsigned int)v;
    }

    return fallback_hw;
}

void Ped::Model::moveAgent(Ped::Tagent* agent)
{
    move(agent);   // call private function internally
}


static void tick_seq_impl(const std::vector<Ped::Tagent*>& agents,
                          Ped::Model* model) {
    for (Ped::Tagent* a : agents) a->computeNextDesiredPosition();
    model->updateHeatmapCuda();
    for (Ped::Tagent* a : agents) {
        model->moveAgent(a);
    }
}
/*
static void tick_omp_impl(const std::vector<Ped::Tagent*>& agents) {
    int N = (int)agents.size();

    #pragma omp parallel for 
    for (int i = 0; i < N; ++i) agents[i]->computeNextDesiredPosition();

    #pragma omp parallel for 
    for (int i = 0; i < N; ++i) {
        agents[i]->setX(agents[i]->getDesiredX());
        agents[i]->setY(agents[i]->getDesiredY());
    }
}
*/
static void tick_omp_impl(const std::vector<Ped::Tagent*>& agents,
                          Ped::Model* model,
                          int max_threads) {
    int N = (int)agents.size();
    if (N == 0) return;

    if (max_threads > 0) {
        omp_set_num_threads(max_threads);
    }

    #pragma omp parallel
    {
        #pragma omp for 
        for (int i = 0; i < N; ++i)
            agents[i]->computeNextDesiredPosition();

        #pragma omp for 
        for (int i = 0; i < N; ++i){
            model->moveAgent(agents[i]);
        }
    }
    
}



static void tick_threads_impl(const std::vector<Ped::Tagent*>& agents, Ped::Model* model, int max_threads) {
    int N = (int)agents.size();
    if (N == 0) return;

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;

    unsigned int T = get_thread_count(max_threads, hw);
    if ((int)T > N) T = (unsigned int)N;
    if (T == 0) T = 1;

    int chunk = (N + (int)T - 1) / (int)T;

    auto range_compute = [&](int start, int end) {
        for (int i = start; i < end; ++i) agents[i]->computeNextDesiredPosition();
    };

    std::vector<std::thread> threads;
    threads.reserve(T);
    for (unsigned int t = 0; t < T; ++t) {
        int start = (int)t * chunk;
        int end = std::min(start + chunk, N);
        if (start >= N) break;
        threads.emplace_back(range_compute, start, end);
    }
    for (auto& th : threads) th.join();

    threads.clear();

    auto range_set = [&](int start, int end) {
        for (int i = 0; i < N; ++i){
            model->moveAgent(agents[i]);
        }
    };

    for (unsigned int t = 0; t < T; ++t) {
        int start = (int)t * chunk;
        int end = std::min(start + chunk, N);
        if (start >= N) break;
        threads.emplace_back(range_set, start, end);
    }
    for (auto& th : threads) th.join();
}

void Ped::Model::tick_region_impl() {
    // Phase 1: parallel per region
    #pragma omp parallel for
    for (int ri = 0; ri < (int)regions.size(); ++ri) {
        Region& r = regions[ri];
        std::vector<Ped::Tagent*> movedAgents;

        for (Ped::Tagent* agent : r.agents) {
            agent->computeNextDesiredPosition();
            int newX = agent->getDesiredX();
            int newY = agent->getDesiredY();

            // check if inside region
            if (newX >= r.x0 && newX < r.x1 && newY >= r.y0 && newY < r.y1) {
                agent->setX(newX);
                agent->setY(newY);
            } else {
                // Crossed border → store for transfer
                if (newX < 0 || newX >= worldWidth || newY < 0 || newY >= worldHeight) {
                    std::cerr << "Warning: agent out of bounds: " << newX << "," << newY << std::endl;
                    newX = std::max(0, std::min(newX, worldWidth-1));
                    newY = std::max(0, std::min(newY, worldHeight-1));
                }
                std::lock_guard<std::mutex> lock(*cellLocks[newX][newY]);
                r.agentsToTransfer.push_back(agent);
                agent->setX(newX);
                agent->setY(newY);
            }
        }
    }

    // Phase 2: serially transfer agents across regions
    for (Region& r : regions) {
        if (r.agentsToTransfer.empty()) continue;

        for (Ped::Tagent* agent : r.agentsToTransfer) {
            // remove from old region
            auto it = std::find(r.agents.begin(), r.agents.end(), agent);
            if (it != r.agents.end()) r.agents.erase(it);

            // add to new region
            Region* newR = getRegionFor(agent->getX(), agent->getY());
            if (newR != nullptr) {
                std::lock_guard<std::mutex> lock(newR->borderMutex);
                newR->agents.push_back(agent);
            }
        }
        r.agentsToTransfer.clear();
    }
}

Ped::Region* Ped::Model::getRegionFor(int x, int y) {
    for (Region& r : regions) {
        if (x >= r.x0 && x < r.x1 && y >= r.y0 && y < r.y1)
            return &r;
    }
    return nullptr; // fallback
}
#ifndef NOCDUA
namespace Ped {

void tick_cuda_impl(Model* model) {
    int N = (int)model->agents.size();
    if (N == 0) return;

    for (int i = 0; i < N; ++i) {
        model->posX[i] = (float)model->agents[i]->getX();
        model->posY[i] = (float)model->agents[i]->getY();

        Ped::Twaypoint* dest = model->agents[i]->getDestination();
        if (dest != nullptr) {
            model->destX[i] = dest->getx();
            model->destY[i] = dest->gety();
        } else {
            model->destX[i] = model->posX[i];
            model->destY[i] = model->posY[i];
        }
    }

    // Call the CUDA kernel in global namespace
    ::updateDesiredPositionsCuda(
        N,
        model->posX.data(),
        model->posY.data(),
        model->destX.data(),
        model->destY.data(),
        model->desiredX.data(),
        model->desiredY.data()
    );

    model->updateHeatmapCuda();

    for (int i = 0; i < N; ++i) {
        model->agents[i]->setX((int)(model->desiredX[i] + 0.5f));
        model->agents[i]->setY((int)(model->desiredY[i] + 0.5f));
    }
}

} // namespace Ped
#endif

void Ped::Model::tick()
{
    reservedPositions.clear();
    int mt = this->max_threads;
    if (mt == 0) mt = get_max_threads_from_env();

    int N = (int)agents.size();
    if (N == 0) return;

    switch (this->implementation) {
        case Ped::SEQ:
            tick_seq_impl(agents, this);
            break;
        case Ped::OMP:
            if (mt > N) mt = N; // don't spawn more threads than agents
            tick_omp_impl(agents, this, mt);
            break;
        case PTHREAD:
            tick_threads_impl(agents,this, mt);
            break;

        case Ped::REGION:
            tick_region_impl();
            break;

        case Ped::VECTOR:
        {
            int N = (int)agents.size();
            if (N == 0) return;

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                posX[i] = agents[i]->getX();
                posY[i] = agents[i]->getY();

                Ped::Twaypoint* dest = agents[i]->getDestination();

                if (dest != nullptr) {
                    destX[i] = dest->getx();
                    destY[i] = dest->gety();
                } else {
                    // If no destination, stay in place
                    destX[i] = posX[i];
                    destY[i] = posY[i];
                }
            }

     
            int i = 0;
            for (; i <= N - 4; i += 4) {

                __m128 x  = _mm_loadu_ps(&posX[i]);
                __m128 y  = _mm_loadu_ps(&posY[i]);
                __m128 dx = _mm_loadu_ps(&destX[i]);
                __m128 dy = _mm_loadu_ps(&destY[i]);

                __m128 diffX = _mm_sub_ps(dx, x);
                __m128 diffY = _mm_sub_ps(dy, y);

                __m128 len2 = _mm_add_ps(
                    _mm_mul_ps(diffX, diffX),
                    _mm_mul_ps(diffY, diffY)
                );

                __m128 len = _mm_sqrt_ps(len2);

                __m128 eps  = _mm_set1_ps(1e-9);
                __m128 mask = _mm_cmp_ps(len, eps, _CMP_GT_OQ);

                __m128 stepX = _mm_blendv_ps(
                    _mm_setzero_ps(),
                    _mm_div_ps(diffX, len),
                    mask
                );

                __m128 stepY = _mm_blendv_ps(
                    _mm_setzero_ps(),
                    _mm_div_ps(diffY, len),
                    mask
                );

                __m128 newX = _mm_add_ps(x, stepX);
                __m128 newY = _mm_add_ps(y, stepY);

                _mm_storeu_ps(&desiredX[i], newX);
                _mm_storeu_ps(&desiredY[i], newY);
            }

  
            for (; i < N; ++i) {
                float dx = destX[i] - posX[i];
                float dy = destY[i] - posY[i];
                float len = sqrt(dx*dx + dy*dy);

                if (len > 1e-9f) {
                    desiredX[i] = posX[i] + dx / len;
                    desiredY[i] = posY[i] + dy / len;
                } else {
                    desiredX[i] = posX[i];
                    desiredY[i] = posY[i];
                }
            }


            for (int i = 0; i < N; ++i) {
                agents[i]->setX((int)(desiredX[i] + 0.5));
                agents[i]->setY((int)(desiredY[i] + 0.5));
            }

            break;


        }
        #ifndef NOCDUA
        case Ped::CUDA:
            tick_cuda_impl(this);
            break;
        #endif
        default:
            tick_seq_impl(agents, this);
            break;
        
    }
}




/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.





long long encodePos(int x, int y) {
    return ((long long)x << 32) | (unsigned int)y; // Unique key for each cell
}

void Ped::Model::move(Ped::Tagent *agent)
{
	 int x0 = agent->getX();
    int y0 = agent->getY();
    int dx = agent->getDesiredX();
    int dy = agent->getDesiredY();

    std::vector<std::pair<int,int>> candidates = {
        {dx, dy},
        {x0+1, y0}, {x0-1, y0}, {x0, y0+1}, {x0, y0-1},
        {x0+1, y0+1}, {x0-1, y0-1}, {x0+1, y0-1}, {x0-1, y0+1}
    };

    for (auto& pos : candidates) {
        if (pos.first < 0 || pos.second < 0 ||
        pos.first >= worldWidth || pos.second >= worldHeight) continue;

        std::lock_guard<std::mutex> lock(*cellLocks[pos.first][pos.second]);
        if (!cellOccupied[pos.first][pos.second]) {
            cellOccupied[pos.first][pos.second] = true;
            agent->setX(pos.first);
            agent->setY(pos.second);
            return;
        }
    }

    // Stay in place
    agent->setX(x0);
    agent->setY(y0);
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist, const Ped::Tagent* self) const {

	set<const Ped::Tagent*> neighbors;

    for (Ped::Tagent* a : agents) {
        if (a == self) continue; // ignore self
        int dx = std::abs(a->getX() - x);
        int dy = std::abs(a->getY() - y);
        if (dx <= dist && dy <= dist) {
            neighbors.insert(a);
        }
    }

    return neighbors;
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}