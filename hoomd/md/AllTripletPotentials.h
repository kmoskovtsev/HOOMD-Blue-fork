// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#ifndef __TRIPLET_POTENTIALS__H__
#define __TRIPLET_POTENTIALS__H__

#include "PotentialPair.h"
#include "PotentialTersoff.h"
#include "EvaluatorTersoff.h"

#ifdef ENABLE_CUDA
#include "PotentialTersoffGPU.h"
#include "DriverTersoffGPU.cuh"
#endif

/*! \file AllTripletPotentials.h
    \brief Handy list of typedefs for all of the templated three-body potentials in hoomd
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Three-body potential force compute for Tersoff forces
typedef PotentialTersoff< EvaluatorTersoff > PotentialTripletTersoff;

#ifdef ENABLE_CUDA
//! Three-body potential force compute for Tersoff forces on the GPU
typedef PotentialTersoffGPU< EvaluatorTersoff, gpu_compute_tersoff_forces > PotentialTripletTersoffGPU;

#endif // ENABLE_CUDA

#endif // __TRIPLET_POTENTIALS_H__
