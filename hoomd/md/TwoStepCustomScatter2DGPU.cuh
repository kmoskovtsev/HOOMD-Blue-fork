// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TwoStepCustomScatter2DGPU.cuh
    \brief Declares GPU kernel code for CustomScatter2D integration on the GPU. Used by TwoStepCustomScatter2DGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#ifndef __TWO_STEP_CUSTOMSCATTER2D_GPU_CUH__
#define __TWO_STEP_CUSTOMSCATTER2D_GPU_CUH__

//! Kernel driver for the first part of the CustomScatter2D update called by TwoStepNVEGPU
cudaError_t gpu_scatter2d_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxDim& box,
                             Scalar deltaT,
                             bool limit,
                             Scalar limit_val,
                             bool zero_force,
                             unsigned int block_size);

//! Kernel driver for the second part of the NVE update called by TwoStepNVEGPU
cudaError_t gpu_scatter2d_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             Scalar deltaT,
                             bool limit,
                             Scalar limit_val,
                             bool zero_force,
                             unsigned int block_size,
                             unsigned int *d_tag,
                             Scalar *d_wk,
                             Scalar *d_Wint,
                             unsigned int pitch,
                             unsigned int Nk,
                             unsigned int NW,
                             Scalar3 v_params,
                             unsigned int seed,
                             unsigned int timestep);

//! Kernel driver for the first part of the angular NVE update (NO_SQUISH) by TwoStepNVEPU
cudaError_t gpu_scatter2d_angular_step_one(Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar deltaT,
                             Scalar scale);

//! Kernel driver for the second part of the angular NVE update (NO_SQUISH) by TwoStepNVEPU
cudaError_t gpu_scatter2d_angular_step_two(const Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar deltaT,
                             Scalar scale);

#endif //__TWO_STEP_CUSTOMSCATTER2D_GPU_CUH__
