// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TablePotential2DGPU.cuh
    \brief Declares GPU kernel code for calculating the 2D-table pair forces. Used by TablePotential2DGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/HOOMDMath.h"

#ifndef __TABLEPOTENTIAL2DGPU_CUH__
#define __TABLEPOTENTIAL2DGPU_CUH__

//! Kernel driver that computes table forces on the GPU for TablePotentialGPU
cudaError_t gpu_compute_table_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim& box,
                                     const Scalar3 *d_tables,
                                     const unsigned int tables_pitch,
                                     const Scalar2 *d_params,
                                     const unsigned int table_width,
                                     const unsigned int table_height,
                                     const unsigned int block_size,
                                     const unsigned int compute_capability,
                                     const unsigned int max_tex1d_width);

#endif



// vim:syntax=cpp
