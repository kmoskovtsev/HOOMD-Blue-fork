// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __EVALUATOR_EXTERNAL_GAUSSIAN_H__
#define __EVALUATOR_EXTERNAL_GAUSSIAN_H__

#ifndef NVCC
#include <string>
#endif

#include <math.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

/*! \file EvaluatorExternalGaussian.h
    \brief Defines the external potential evaluator to induce a periodic ordered phase
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// SCALARASINT resolves to __scalar_as_int on the device and to __scalar_as_int on the host
#ifdef NVCC
#define SCALARASINT(x) __scalar_as_int(x)
#else
#define SCALARASINT(x) __scalar_as_int(x)
#endif

//! Class for evaluating sphere constraints
/*! <b>General Overview</b>
    EvaluatorExternalGaussian is an evaluator to induce a periodic modulation on the concentration profile
    in the system, e.g. to generate a periodic phase in a system of diblock copolymers.

    The external potential \f$V(\vec{r}) \f$ is implemented using the following formula:

    \f[
    V(\vec{r}) = A * \exp\left[ -(\vec{r} - \vec{r}_c)^2/(2\sigma**2) \right]
    \f]

    where \f$A\f$ is the potential strength, \f$\vec{r}_c\f$ is center of the potential, 
    \f$\sigma\f$ is the width of the potential.
*/
class EvaluatorExternalGaussian
    {
    public:

        //! type of parameters this external potential accepts
        typedef Scalar4 param_type;
        typedef struct field{}field_type;

        //! Constructs the constraint evaluator
        /*! \param X position of particle
            \param box box dimensions
            \param params per-type parameters of external potential
        */
        DEVICE EvaluatorExternalGaussian(Scalar3 X, const BoxDim& box, const param_type& params, const field_type& field)
            : m_pos(X)
            {
            m_x =  params.x;
            m_y =  params.y;
            m_prefactor = params.z;
            m_sigma_sq = params.w*params.w;
            }

        //! External Periodic doesn't need diameters
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter value
        /*! \param di Diameter of particle i
        */
        DEVICE void setDiameter(Scalar di) { }

        //! External Periodic doesn't need charges
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter value
        /*! \param qi Charge of particle i
        */
        DEVICE void setCharge(Scalar qi) { }

        //! Declares additional virial cotribututions are needed for the external field
        /*! No contributions
        */
        DEVICE static bool requestFieldVirialTerm() { return true; }

        //! Evaluate the force, energy and virial
        /*! \param F force vector
            \param energy value of the energy
            \param virial array of six scalars for the upper triangular virial tensor
        */
        DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar& energy, Scalar* virial)
            {

            F.x = Scalar(0.0);
            F.y = Scalar(0.0);
            F.z = Scalar(0.0);
            energy = Scalar(0.0);

            // For this potential, since it uses scaled positions, the virial is always zero.
            for (unsigned int i = 0; i < 6; i++)
                virial[i] = Scalar(0.0);

            Scalar dx = m_pos.x - m_x;
            Scalar dy = m_pos.y - m_y;
            Scalar expf = fast::exp(-(fast::pow(dx,2) +fast::pow(dy,2))*0.5/m_sigma_sq);
            F.x = m_prefactor/m_sigma_sq*dx*expf;
            F.y = m_prefactor/m_sigma_sq*dy*expf;
            energy = m_prefactor*expf;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("gaussian");
            }
        #endif

    protected:
        Scalar3 m_pos;                //!< particle position
        Scalar m_x; //!< x coordinate of the center of the potential field
        Scalar m_y; //!< y coordinate of the center of the potential field
        Scalar m_prefactor; //!< prefactor A of the gaussian
        Scalar m_sigma_sq; //!< sigma-squared of the potential
        
   };


#endif // __EVALUATOR_EXTERNAL_LAMELLAR_H__
