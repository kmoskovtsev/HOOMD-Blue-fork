// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: kmoskovtsev

#include "IntegrationMethodTwoStep.h"
//#include "hoomd/GPUArray.h"
//#include "hoomd/Index1D.h"

#ifndef __TWO_STEP_CUSTOMSCATTER_H__
#define __TWO_STEP_CUSTOMSCATTER_H__

/*! \file TwoStepCusomScatter.h
    \brief Declares the TwoStepCustomScatter2D class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Integrates part of the system forward in two steps in the NVE ensemble with custom momentum-dependent scattering.
/*! Implements velocity-verlet NVE with scattering integration through the IntegrationMethodTwoStep interface

    \ingroup updaters
*/
class TwoStepCustomScatter2D : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepCustomScatter2D(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group,
                   unsigned int Nk,
                   unsigned int NW,
                   unsigned int NY,
                   unsigned int seed,
                   bool skip_restart=false);
        virtual ~TwoStepCustomScatter2D();
        // Set scattering tables
        void setTables(const std::vector<Scalar> &wk,
                                     const std::vector<Scalar> &Winv,
                                     const std::vector<Scalar> &wik,
                                     const std::vector<Scalar> &Finv,
                                     const std::vector<Scalar> &Yinv,
                                     const Scalar vmin,
                                     const Scalar vmax);
 
        //! Sets the movement limit
        void setLimit(Scalar limit);

        //! Removes the limit
        void removeLimit();

        //! Sets the zero force option
        /*! \param zero_force Set to true to specify that the integration with a zero net force on each of the particles
                              in the group
        */
        void setZeroForce(bool zero_force)
            {
            m_zero_force = zero_force;
            }

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        bool m_limit;       //!< True if we should limit the distance a particle moves in one step
        Scalar m_limit_val; //!< The maximum distance a particle is to move in one step
        bool m_zero_force;  //!< True if the integration step should ignore computed forces
        unsigned int m_Nk; //!< Number of k-points for scattering rates
        unsigned int m_NW; //!< Number of theta points to sample inverse cumulative distribution Winv
        unsigned int m_NY; //!< number of points to sample angle distribution for inelastic scattering
        GPUArray<Scalar> m_wk; //!< total probability to scatter in a unit time vs k (1D array) for elastic scattering
        GPUArray<Scalar> m_Winv; //!< Inverse cumulative probability distribution to scatter into angle d\theta
        GPUArray<Scalar> m_wik; //!< total probability to scatter in a unit time vs k (1D array) for inelastic scattering
        GPUArray<Scalar> m_Finv; //!< Inverse cumulative probability to scatter to state with momentum magnitude k'
        GPUArray<Scalar> m_Yinv; //!< 3D array of inverse cumulative angular distribution for each value of k and k'
        Scalar3 m_params; //!< v_min, v_max, (v_max - v_min)/Nk - min and max velocities in scattering rate calculation
        unsigned int m_seed; //!< seed for random number generation
    };

//! Exports the TwoStepCustomScatter2D class to python
void export_TwoStepCustomScatter2D(pybind11::module& m);

#endif // #ifndef __TWO_STEP_CUSTOMSCATTERING_H__
