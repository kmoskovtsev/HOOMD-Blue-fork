// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander / Anyone is free to add their own pair potentials here

#ifndef __SPECIAL_PAIR_POTENTIALS__H__
#define __SPECIAL_PAIR_POTENTIALS__H__

#include "PotentialSpecialPair.h"
#include "EvaluatorSpecialPairLJ.h"

#ifdef ENABLE_CUDA
#include "PotentialSpecialPairGPU.h"
#include "AllDriverPotentialSpecialPairGPU.cuh"
#endif

/*! \file AllSpecialPairPotentials.h
    \brief Handy list of typedefs for all of the templated special pair potentials in hoomd
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Special pair potential force compute for LJ forces
typedef PotentialSpecialPair<EvaluatorSpecialPairLJ> PotentialSpecialPairLJ;

#ifdef ENABLE_CUDA
//! Special potential force compute for LJ forces on the GPU
typedef PotentialSpecialPairGPU< EvaluatorSpecialPairLJ, gpu_compute_lj_forces > PotentialSpecialPairLJGPU;
#endif

#endif // __SPECIAL_PAIR_POTENTIALS_H__
