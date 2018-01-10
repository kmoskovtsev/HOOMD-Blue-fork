// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: kmoskovtsev

#ifndef __EVALUATOR_EXTERNAL_PERIODIC_TIME_H__
#define __EVALUATOR_EXTERNAL_PERIODIC_TIME_H__

#ifndef NVCC
#include <string>
#endif

#include <math.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

/*! \file EvaluatorExternalPeriodicTime.h
    \brief Defines the external potential evaluator to induce a periodic potential oscillating in time
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

//! Class for evaluating periodic cos(q*x)*cos(omega*t) potential
/*! <b>General Overview</b>
    EvaluatorExternalPeriodicTime is an evaluator to induce a periodic modulation on the concentration profile
    in the system, also oscillating in time.

    The external potential \f$V(\vec{r}) \f$ is implemented using the following formula:

    \f[
    V(\vec{r}, t) = A * \cos\left(p \vec{b}_i\cdot\vec{r}\right) * \cos\left(\omega t\right)
    \f]

    where \f$A\f$ is the ordering parameter, \f$\vec{b}_i\f$ is the reciprocal lattice vector direction
    \f$i=0..2\f$, \f$p\f$ the periodicity 

    The modulation is one-dimensional. It extends along the lattice vector \f$\mathbf{a}_i\f$ of the
    simulation cell.
*/
class EvaluatorExternalPeriodicTime
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
        DEVICE EvaluatorExternalPeriodicTime(Scalar3 X, const BoxDim& box, const param_type& params, const unsigned int timestep, const field_type& field)
            : m_pos(X),
              m_box(box)
            {
            m_index=  SCALARASINT(params.x);
            m_orderParameter = params.y;
            m_periodicity =SCALARASINT(params.z);
            m_omega = params.w;
            m_timestep = timestep;
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
            Scalar3 a2 = make_scalar3(0,0,0);
            Scalar3 a3 = make_scalar3(0,0,0);

            F.x = Scalar(0.0);
            F.y = Scalar(0.0);
            F.z = Scalar(0.0);
            energy = Scalar(0.0);

            // For this potential, since it uses scaled positions, the virial is always zero.
            for (unsigned int i = 0; i < 6; i++)
                virial[i] = Scalar(0.0);

            Scalar V_box = m_box.getVolume();
            // compute the vector pointing from P to V
            if (m_index == 0)
                {
                a2 = m_box.getLatticeVector(1);
                a3 = m_box.getLatticeVector(2);
                }
            else if (m_index == 1)
                {
                a2 = m_box.getLatticeVector(2);
                a3 = m_box.getLatticeVector(0);
                }
            else if (m_index == 2)
                {
                a2 = m_box.getLatticeVector(0);
                a3 = m_box.getLatticeVector(1);
                }

            Scalar3 b = Scalar(2.0*M_PI)*make_scalar3(a2.y*a3.z-a2.z*a3.y,
                                                      a2.z*a3.x-a2.x*a3.z,
                                                      a2.x*a3.y-a2.y*a3.x)/V_box;

            Scalar3 q = b*m_periodicity;
            Scalar arg = dot(m_pos,q);
            Scalar time_factor = fast::cos(m_omega*m_timestep);

            F = m_orderParameter*fast::sin(arg)*q*time_factor;
            energy = m_orderParameter*fast::cos(arg)*time_factor;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("periodic_time");
            }
        #endif

    protected:
        Scalar3 m_pos;                //!< particle position
        BoxDim m_box;                 //!< box dimensions
        unsigned int m_index;         //!< cartesian index of direction along which the lammellae should be orientied
        Scalar m_orderParameter;      //!< ordering parameter
        unsigned int m_periodicity;   //!< Number of periods per unit cell
        Scalar m_omega;               //!< Angular frequency of time oscillation, in terms of inverse timestep
        unsigned int m_timestep;      //!< timestep
   };


#endif // __EVALUATOR_EXTERNAL_LAMELLAR_H__
