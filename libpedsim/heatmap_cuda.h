
#ifndef HEATMAP_CUDA_H
#define HEATMAP_CUDA_H

#include "ped_model.h"

namespace Ped
{
    class Model;
}

/*
 GPU implementation of the heatmap update.
 This replaces updateHeatmapSeq() when running the CUDA version.
*/
void updateHeatmapCuda(Ped::Model* model);

#endif