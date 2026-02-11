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

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>


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
            waypointsX[i][j] = (double)agent->getWaypoint(j)->getx();
            waypointsY[i][j] = (double)agent->getWaypoint(j)->gety();
            waypointR[i][j]  = (double)agent->getWaypoint(j)->getr();
        }
    }

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	if (max_threads > 0) this->setMaxThreads(max_threads);

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
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


static void tick_seq_impl(const std::vector<Ped::Tagent*>& agents) {
    for (Ped::Tagent* a : agents) a->computeNextDesiredPosition();
    for (Ped::Tagent* a : agents) {
        a->setX(a->getDesiredX());
        a->setY(a->getDesiredY());
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
static void tick_omp_impl(const std::vector<Ped::Tagent*>& agents, int max_threads) {
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
        for (int i = 0; i < N; ++i) {
            agents[i]->setX(agents[i]->getDesiredX());
            agents[i]->setY(agents[i]->getDesiredY());
        }
    }
    
}



static void tick_threads_impl(const std::vector<Ped::Tagent*>& agents, int max_threads) {
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
        for (int i = start; i < end; ++i) {
            agents[i]->setX(agents[i]->getDesiredX());
            agents[i]->setY(agents[i]->getDesiredY());
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


void Ped::Model::tick()
{
    int mt = this->max_threads;
    if (mt == 0) mt = get_max_threads_from_env();

    int N = (int)agents.size();
    if (N == 0) return;

    switch (this->implementation) {
        case Ped::SEQ:
            tick_seq_impl(agents);
            break;
        case Ped::OMP:
            if (mt > N) mt = N; // don't spawn more threads than agents
            tick_omp_impl(agents, mt);
            break;
        case PTHREAD:
            tick_threads_impl(agents, mt);
            break;

        case Ped::VECTOR:
        {
            int N = (int)agents.size();
            if (N == 0) return;

            // -------------------------------
            // 1. Compute next destinations
            // -------------------------------
            for (int i = 0; i < N; ++i) {
                if (currentWaypointIndex[i] >= numWaypoints[i]) continue;

                double dx = waypointsX[i][currentWaypointIndex[i]] - posX[i];
                double dy = waypointsY[i][currentWaypointIndex[i]] - posY[i];
                double len2 = dx*dx + dy*dy;

                // Reached waypoint?
                if (len2 < waypointR[i][currentWaypointIndex[i]]*waypointR[i][currentWaypointIndex[i]]) {
                    currentWaypointIndex[i]++;
                }

                // Set destination
                if (currentWaypointIndex[i] < numWaypoints[i]) {
                    destX[i] = waypointsX[i][currentWaypointIndex[i]];
                    destY[i] = waypointsY[i][currentWaypointIndex[i]];
                } else {
                    destX[i] = posX[i];
                    destY[i] = posY[i];
                }
            }

            // -------------------------------
            // 2. Vectorized movement
            // -------------------------------
            int i = 0;
            for (; i <= N - 4; i += 4) {
                __m256d x  = _mm256_loadu_pd(&posX[i]);
                __m256d y  = _mm256_loadu_pd(&posY[i]);
                __m256d dx = _mm256_loadu_pd(&destX[i]);
                __m256d dy = _mm256_loadu_pd(&destY[i]);

                __m256d diffX = _mm256_sub_pd(dx, x);
                __m256d diffY = _mm256_sub_pd(dy, y);
                __m256d len2 = _mm256_add_pd(_mm256_mul_pd(diffX,diffX), _mm256_mul_pd(diffY,diffY));
                __m256d len  = _mm256_sqrt_pd(len2);

                // Avoid division by zero
                __m256d mask = _mm256_cmp_pd(len, _mm256_set1_pd(1e-9), _CMP_GT_OQ);
                __m256d stepX = _mm256_blendv_pd(_mm256_set1_pd(0.0), _mm256_div_pd(diffX, len), mask);
                __m256d stepY = _mm256_blendv_pd(_mm256_set1_pd(0.0), _mm256_div_pd(diffY, len), mask);

                _mm256_storeu_pd(&desiredX[i], _mm256_add_pd(x, stepX));
                _mm256_storeu_pd(&desiredY[i], _mm256_add_pd(y, stepY));
            }

            // Tail loop
            for (; i < N; ++i) {
                double dx = destX[i] - posX[i];
                double dy = destY[i] - posY[i];
                double len = sqrt(dx*dx + dy*dy);
                if (len > 1e-9) {
                    desiredX[i] = posX[i] + dx/len;
                    desiredY[i] = posY[i] + dy/len;
                } else {
                    desiredX[i] = posX[i];
                    desiredY[i] = posY[i];
                }
            }

            // -------------------------------
            // 3. Update main positions
            // -------------------------------
            for (int i = 0; i < N; ++i) {
                posX[i] = desiredX[i];
                posY[i] = desiredY[i];
            }

            // -------------------------------
            // 4. Write back to agents (optional)
            // -------------------------------
            for (int i = 0; i < N; ++i) {
                agents[i]->setX((int)round(posX[i]));
                agents[i]->setY((int)round(posY[i]));
            }
            break;
        }
        default:
            tick_seq_impl(agents);
            break;
        
    }
}




/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}