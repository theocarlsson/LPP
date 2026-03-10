//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <memory>

#include "ped_agent.h"

namespace Ped{
	class Tagent;

	struct Region {
		int x0, y0, x1, y1;
		std::vector<Tagent*> agents;
		std::mutex borderMutex;
		std::vector<Tagent*> agentsToTransfer;

		Region() = default;

		Region(const Region&) = delete;
		Region& operator=(const Region&) = delete;
		Region(Region&&) = delete;
		Region& operator=(Region&&) = delete;
	};

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, SIMD, REGION };

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation, int max_threads);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// Returns the agents of this scenario
		const std::vector<Tagent*>& getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		void setMaxThreads(int maxThreads);
        int getMaxThreads() const;
		void moveAgent(Ped::Tagent* agent);
		void tick_region_impl();
    	Region* getRegionFor(int x, int y);
		void updateHeatmapCuda();

		friend void tick_cuda_impl(Ped::Model* model);

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Current positions
        std::vector<float> posX;
        std::vector<float> posY;

        // Destination positions
        std::vector<float> destX;
        std::vector<float> destY;

        // Computed desired positions
        std::vector<float> desiredX;
        std::vector<float> desiredY;

		static constexpr int MAX_WAYPOINTS = 16;

		// Number of waypoints each agent has
		std::vector<int> numWaypoints;          // total waypoints for each agent
		std::vector<int> currentWaypointIndex;  // current waypoint index per agent

		// Waypoints coordinates per agent
		std::vector<std::array<float, MAX_WAYPOINTS>> waypointsX;
		std::vector<std::array<float, MAX_WAYPOINTS>> waypointsY;

		// Destination radius per waypoint
		std::vector<std::array<float, MAX_WAYPOINTS>> waypointR;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		std::vector<Region> regions;      // all regions
		int numRegionsX = 2;              // number of regions along X
		int numRegionsY = 2;              // number of regions along Y
		int worldWidth = 100;             // total width of the world
		int worldHeight = 100;

		std::vector<std::vector<std::unique_ptr<std::mutex>>> cellLocks;
		std::vector<std::vector<bool>> cellOccupied;

		int max_threads = 2; 

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist, const Ped::Tagent* self = nullptr) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
	};
}
#endif
