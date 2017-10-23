// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: kmoskovtsev


#include "TablePotential2DGPU.h"

namespace py = pybind11;
#include <stdexcept>

/*! \file TablePotential2DGPU.cc
    \brief Defines the TablePotential2DGPU class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
TablePotential2DGPU::TablePotential2DGPU(std::shared_ptr<SystemDefinition> sysdef,
                                     unsigned int table_width,
                                     unsigned int table_height,
                                     const std::string& log_suffix)
    : TablePotential2D(sysdef, table_width, table_height, log_suffix)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TableForceCompute2DGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing TableForceCompute2DGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "pair_table2D", this->m_exec_conf));
    }

/*! \post The table based forces are computed for the given timestep. The neighborlist's
compute method is called to ensure that it is up to date.

\param timestep specifies the current time step of the simulation

Calls gpu_compute_table_forces to do the leg work
*/
void TablePotential2DGPU::computeForces(unsigned int timestep)
    {

    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "Table pair");


    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar3> d_tables(m_tables, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    unsigned int tables_pitch = d_tables.getPitch();

    // run the kernel on all GPUs in parallel
    m_tuner->begin();
    gpu_compute_table2D_forces(d_force.data,
                             d_virial.data,
                             m_virial.getPitch(),
                             m_pdata->getN(),
                             d_pos.data,
                             box,
                             d_tables.data,
                             tables_pitch,
                             d_params.data[0],
                             m_table_width,
                             m_table_height,
                             m_tuner->getParam(),
                             m_exec_conf->getComputeCapability(),
                             m_exec_conf->dev_prop.maxTexture1DLinear);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_TablePotentialGPU2D(py::module& m)
    {
    py::class_<TablePotential2DGPU, std::shared_ptr<TablePotential2DGPU> >(m, "TablePotential2DGPU", py::base<TablePotential2D>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                                unsigned int,
                                unsigned int,
                                const std::string& >())
                                ;
    }
