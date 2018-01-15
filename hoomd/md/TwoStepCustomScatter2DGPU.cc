// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander



#include "TwoStepCustomScatter2DGPU.h"
#include "TwoStepCustomScatter2DGPU.cuh"

namespace py = pybind11;
using namespace std;

/*! \file TwoStepCustomScatter2DGPU.h
    \brief Contains code for the TwoStepCustomScatter2DGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
*/
TwoStepCustomScatter2DGPU::TwoStepCustomScatter2DGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             unsigned int Nk,
                             unsigned int NW,
                             unsigned int seed)
    : TwoStepCustomScatter2D(sysdef, group, Nk, NW, seed)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepCustomScatter2DGPU when CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepCustomScatter2DGPU");
        }

    // initialize autotuner
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        valid_params.push_back(block_size);

    m_tuner_one.reset(new Autotuner(valid_params, 5, 100000, "custom_scatter2D_step_one", this->m_exec_conf));
    m_tuner_two.reset(new Autotuner(valid_params, 5, 100000, "custom_scatter2D_step_two", this->m_exec_conf));
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepCustomScatter2DGPU::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "CustomScatter2D step 1");

    // access all the needed data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the update on the GPU
    m_tuner_one->begin();
    gpu_scatter2d_step_one(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_image.data,
                     d_index_array.data,
                     group_size,
                     box,
                     m_deltaT,
                     m_limit,
                     m_limit_val,
                     m_zero_force,
                     m_tuner_one->getParam());
    m_tuner_one->end();

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        gpu_scatter2d_angular_step_one(d_orientation.data,
                                 d_angmom.data,
                                 d_inertia.data,
                                 d_net_torque.data,
                                 d_index_array.data,
                                 group_size,
                                 m_deltaT,
                                 1.0);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepCustomScatter2DGPU::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "CustomScatter2D step 2");

    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_wk(m_wk, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_Winv(m_Winv, access_location::device, access_mode::read);
    int pitch = m_Winv.getPitch();
    m_tuner_two->begin();
    // perform the update on the GPU
    gpu_scatter2d_step_two(d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     group_size,
                     d_net_force.data,
                     m_deltaT,
                     m_limit,
                     m_limit_val,
                     m_zero_force,
                     m_tuner_two->getParam(),
                     d_tag.data,
                     d_wk.data,
                     d_Winv.data,
                     pitch,
                     m_Nk,
                     m_NW,
                     m_params,
                     m_seed,
                     timestep);
    m_tuner_two->end();

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_aniso)
        {
        // second part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        gpu_scatter2d_angular_step_two(d_orientation.data,
                                 d_angmom.data,
                                 d_inertia.data,
                                 d_net_torque.data,
                                 d_index_array.data,
                                 group_size,
                                 m_deltaT,
                                 1.0);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_TwoStepCustomScatter2DGPU(py::module& m)
    {
    py::class_<TwoStepCustomScatter2DGPU, std::shared_ptr<TwoStepCustomScatter2DGPU> >(m, "TwoStepCustomScatter2DGPU", py::base<TwoStepCustomScatter2D>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, unsigned int, unsigned int, unsigned int >())
        ;
    }
