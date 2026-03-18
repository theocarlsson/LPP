#pragma once

#ifndef CUDA_TESTKERNEL_H
#define CUDA_TESTKERNEL_H

int cuda_test();

// Function called by Ped::Model::tick_cuda_impl
void updateDesiredPositionsCuda(
    int N,
    float* posX,
    float* posY,
    float* destX,
    float* destY,
    float* desiredX,
    float* desiredY
);

#endif
