// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: kmoskovtsev

#include "TablePotential2D.h"
#include "TablePotential2DGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file TablePotentialGPU.h
    \brief Declares the TablePotentialGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __TABLEPOTENTIAL2DGPU_H__
#define __TABLEPOTENTIAL2DGPU_H__

//! Compute 2D table based potentials on the GPU
/*! Calculates exactly the same thing as TablePotential2D, but on the GPU

    The GPU kernel for calculating this can be found in TablePotential2DGPU.cu/
    \ingroup computes
*/
class TablePotential2DGPU : public TablePotential2D
    {
    public:
        //! Constructs the compute
        TablePotential2DGPU(std::shared_ptr<SystemDefinition> sysdef,
                       unsigned int table_width,
                       unsigned int table_height,
                       const std::string& log_suffix="");

        //! Destructor
        virtual ~TablePotential2DGPU() { }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TablePotential2D::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the TablePotentialGPU class to python
void export_TablePotential2DGPU(pybind11::module& m);

#endif
