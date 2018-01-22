// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: kmoskovtsev



#include "TwoStepCustomScatter2D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Saru.h"

using namespace std;
using namespace hoomd;
namespace py = pybind11;

/*! \file TwoStepCustomScatter2D.cc
    \brief Contains code for the TwoStepCustomScatter2D class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param skip_restart Skip initialization of the restart information
    \param N_k number of k-points for scattering rate sampling
    \param N_W number of theta points to sample cumulative scattering distribution W
*/
TwoStepCustomScatter2D::TwoStepCustomScatter2D(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       unsigned int Nk,
                       unsigned int NW,
                       unsigned int seed,
                       std::shared_ptr<Variant> T,
                       bool noiseless_t,
                       bool skip_restart)
    : IntegrationMethodTwoStep(sysdef, group), m_limit(false), m_limit_val(1.0), m_zero_force(false),
    m_Nk(Nk), m_NW(NW), m_T(T), m_noiseless_t(noiseless_t)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepCustomScatter2D" << endl;

    if (!skip_restart)
        {
        // set a named, but otherwise blank set of integrator variables
        IntegratorVariables v = getIntegratorVariables();

        if (!restartInfoTestValid(v, "custom_scattering", 0))
            {
            v.type = "custom_scattering";
            v.variable.resize(0);
            setValidRestart(false);
            }
        else
            setValidRestart(true);

        setIntegratorVariables(v);
        }
    GPUArray<Scalar> wk(Nk, m_exec_conf);
    GPUArray<Scalar> Winv(NW, Nk, m_exec_conf);
    m_wk.swap(wk);
    m_Winv.swap(Winv);

    assert(!m_wk.isNull());
    assert(!m_Winv.isNull());
    
    //send seed=0 to all MPI ranks
    #ifdef ENABLE_MPI
    if( this->m_pdata->getDomainDecomposition() )
        bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
    #endif
    //Allocate memory for the per-type gamma storage and initialize them to 1.0
    GPUVector<Scalar> gamma(m_pdata->getNTypes(), m_exec_conf);
    m_gamma.swap(gamma);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i=0; i < m_gamma.size(); i++)
        h_gamma.data[i] = Scalar(1.0);
    m_seed = m_seed*0x12345677 + 0x12345; m_seed^=(m_seed>>16); m_seed*=0x45679;
    //connect to the ParticleData to receive notifications when the maximum number of particles changes
    m_pdata->getNumTypesChangeSignal().connect<TwoStepCustomScatter2D, &TwoStepCustomScatter2D::slotNumTypesChange>(this);
    }

TwoStepCustomScatter2D::~TwoStepCustomScatter2D()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepCustomScatter2D" << endl;
    m_pdata->getNumTypesChangeSignal().disconnect<TwoStepCustomScatter2D, &TwoStepCustomScatter2D::slotNumTypesChange>(this);
    }

/*! \param limit Distance to limit particle movement each time step

    Once the limit is set, future calls to update() will never move a particle
    a distance larger than the limit in a single time step
*/
void TwoStepCustomScatter2D::setLimit(Scalar limit)
    {
    m_limit = true;
    m_limit_val = limit;
    }

/*! Disables the limit, allowing particles to move normally
*/
void TwoStepCustomScatter2D::removeLimit()
    {
    m_limit = false;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/

void TwoStepCustomScatter2D::slotNumTypesChange()
    {
    // skip the reallocation if the number of types does not change
    // this keeps old parameters when restoring a snapshot
    // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
    if (m_pdata->getNTypes() == m_gamma.size())
        return;

    // re-allocate memory for the per-type gamma storage and initialize them to 1.0
    unsigned int old_ntypes = m_gamma.size();
    m_gamma.resize(m_pdata->getNTypes());

    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);

    for (unsigned int i = old_ntypes; i < m_gamma.size(); i++)
        {
        h_gamma.data[i] = Scalar(1.0);
        }
    }


/*! \param typ Particle type to set gamma for
    \param gamma The gamma value to set
*/
void TwoStepCustomScatter2D::setGamma(unsigned int typ, Scalar gamma)
    {
    if (typ >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Trying to set gamma for a non existent type! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepLangevinBase");
        }

    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
    h_gamma.data[typ] = gamma;
    }



void TwoStepCustomScatter2D::setTables(const std::vector<Scalar> &wk,
                                     const std::vector<Scalar> &Winv,
                                     const Scalar vmin,
                                     const Scalar vmax)
    {
    //access the arrays
    ArrayHandle<Scalar> h_wk(m_wk, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_Winv(m_Winv, access_location::host, access_mode::readwrite);
    unsigned int pitch = m_Winv.getPitch();
    m_params.x = vmin;
    m_params.y = vmax;
    m_params.z = (vmax - vmin)/(Scalar(m_Nk) - 1);
    for (unsigned int i = 0; i < m_Nk; i++)
        {
        h_wk.data[i] = wk[i];
        for (unsigned int j = 0; j < m_NW; j++)
            {
            h_Winv.data[i*pitch + j] = Winv[i*m_NW + j];
            }
        }
    }

void TwoStepCustomScatter2D::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("NVE step 1");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        if (m_zero_force)
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;

        Scalar dx = h_vel.data[j].x*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT*m_deltaT;
        Scalar dy = h_vel.data[j].y*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT*m_deltaT;
        Scalar dz = h_vel.data[j].z*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT*m_deltaT;

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar len = sqrt(dx*dx + dy*dy + dz*dz);
            if (len > m_limit_val)
                {
                dx = dx / len * m_limit_val;
                dy = dy / len * m_limit_val;
                dz = dz / len * m_limit_val;
                }
            }

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;

        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        }

    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        box.wrap(h_pos.data[j], h_image.data[j]);
        }

    // Integration of angular degrees of freedom using sympletic and
    // time-reversal symmetric integration scheme of Miller et al.
    if (m_aniso)
        {
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            quat<Scalar> q(h_orientation.data[j]);
            quat<Scalar> p(h_angmom.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
            vec3<Scalar> I(h_inertia.data[j]);

            // rotate torque into principal frame
            t = rotate(conj(q),t);

            // check for zero moment of inertia
            bool x_zero, y_zero, z_zero;
            x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

            // ignore torque component along an axis for which the moment of inertia zero
            if (x_zero) t.x = 0;
            if (y_zero) t.y = 0;
            if (z_zero) t.z = 0;

            // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
            // using Trotter factorization of rotation Liouvillian
            p += m_deltaT*q*t;

            quat<Scalar> p1, p2, p3; // permutated quaternions
            quat<Scalar> q1, q2, q3;
            Scalar phi1, cphi1, sphi1;
            Scalar phi2, cphi2, sphi2;
            Scalar phi3, cphi3, sphi3;

            if (!z_zero)
                {
                p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
                q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
                phi3 = Scalar(1./4.)/I.z*dot(p,q3);
                cphi3 = slow::cos(Scalar(1./2.)*m_deltaT*phi3);
                sphi3 = slow::sin(Scalar(1./2.)*m_deltaT*phi3);

                p=cphi3*p+sphi3*p3;
                q=cphi3*q+sphi3*q3;
                }

            if (!y_zero)
                {
                p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
                q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
                phi2 = Scalar(1./4.)/I.y*dot(p,q2);
                cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
                sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

                p=cphi2*p+sphi2*p2;
                q=cphi2*q+sphi2*q2;
                }

            if (!x_zero)
                {
                p1 = quat<Scalar>(-p.v.x,vec3<Scalar>(p.s,p.v.z,-p.v.y));
                q1 = quat<Scalar>(-q.v.x,vec3<Scalar>(q.s,q.v.z,-q.v.y));
                phi1 = Scalar(1./4.)/I.x*dot(p,q1);
                cphi1 = slow::cos(m_deltaT*phi1);
                sphi1 = slow::sin(m_deltaT*phi1);

                p=cphi1*p+sphi1*p1;
                q=cphi1*q+sphi1*q1;
                }

            if (! y_zero)
                {
                p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
                q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
                phi2 = Scalar(1./4.)/I.y*dot(p,q2);
                cphi2 = slow::cos(Scalar(1./2.)*m_deltaT*phi2);
                sphi2 = slow::sin(Scalar(1./2.)*m_deltaT*phi2);

                p=cphi2*p+sphi2*p2;
                q=cphi2*q+sphi2*q2;
                }

            if (! z_zero)
                {
                p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
                q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
                phi3 = Scalar(1./4.)/I.z*dot(p,q3);
                cphi3 = slow::cos(Scalar(1./2.)*m_deltaT*phi3);
                sphi3 = slow::sin(Scalar(1./2.)*m_deltaT*phi3);

                p=cphi3*p+sphi3*p3;
                q=cphi3*q+sphi3*q3;
                }

            // renormalize (improves stability)
            q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));

            h_orientation.data[j] = quat_to_scalar4(q);
            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepCustomScatter2D::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("CustomScatter2D step 2");

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);

    ArrayHandle<Scalar> h_wk(m_wk, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_Winv(m_Winv, access_location::host, access_mode::read);
    int pitch = m_Winv.getPitch();
    Scalar dW = Scalar(1)/Scalar(m_NW - Scalar(1));

    const Scalar currentTemp = m_T->getValue(timestep);
    const unsigned int D = Scalar(m_sysdef->getNDimensions());
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];
        if (m_zero_force)
            {
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
            }
        else
            {
            // first, calculate the BD forces
            // Generate three random numbers
            hoomd::detail::Saru saru(ptag, timestep, m_seed);
            Scalar rx = gaussian_rng(saru, Scalar(1.0));
            Scalar ry = gaussian_rng(saru, Scalar(1.0));
            Scalar rz = gaussian_rng(saru, Scalar(1.0));

            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            Scalar gamma = h_gamma.data[type];

            // compute the bd force
            Scalar coeff = fast::sqrt(Scalar(2.0) *gamma*currentTemp/m_deltaT); //version for Gaussian rng

            if (m_noiseless_t)
                coeff = Scalar(0.0);
            Scalar bd_fx = rx*coeff - gamma*h_vel.data[j].x;
            Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
            Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;

            if (D < 3)
                bd_fz = Scalar(0.0);
            // then, calculate acceleration from the net force
            Scalar minv = Scalar(1.0) / h_vel.data[j].w;
            h_accel.data[j].x = (h_net_force.data[j].x + bd_fx)*minv;
            h_accel.data[j].y = (h_net_force.data[j].y + bd_fy)*minv;
            h_accel.data[j].z = (h_net_force.data[j].z + bd_fz)*minv;

            //Scatter
            Scalar total_rn = saru.s<Scalar>(0,1);
            Scalar W_rn = saru.s<Scalar>(0,1);

            //Determine scattering rate
            Scalar v = fast::sqrt(fast::pow(h_vel.data[j].x, 2) + fast::pow(h_vel.data[j].y, 2));
            Scalar f_k = (v - m_params.x)/m_params.z; // (v - v_min)/dv
            int i_k = (int)floor(f_k);
            //Check boundaries
            if (f_k >= m_Nk - 1)
                {
                i_k = m_Nk - 2;
                f_k = m_Nk - 1;
                }
            if (f_k < 0)
                {
                i_k = 0;
                f_k = 0;
                }
            Scalar alpha_k = f_k - i_k;
            Scalar w = h_wk.data[i_k] + (h_wk.data[i_k + 1] - h_wk.data[i_k])*alpha_k;
            if (total_rn < m_deltaT*w)
                {
                //Determine scattering angle using bilinear interpolation from Winv table
                int i_W = (int)floor(W_rn/dW);
                Scalar alpha_W = W_rn/dW - i_W;
                Scalar theta_ik = (1 - alpha_W)*h_Winv.data[i_k*pitch + i_W] + alpha_W*h_Winv.data[i_k*pitch + i_W + 1];
                Scalar theta_ikp = (1 - alpha_W)*h_Winv.data[(i_k + 1)*pitch + i_W] + alpha_W*h_Winv.data[(i_k + 1)*pitch + i_W + 1];
                Scalar theta = (1 - alpha_k)*theta_ik + alpha_k*theta_ikp;
                //Rotate 2D (xy) velocity by theta
                Scalar vx = h_vel.data[j].x;
                Scalar vy = h_vel.data[j].y;
                h_vel.data[j].x = vx*fast::cos(theta) - vy*fast::sin(theta);
                h_vel.data[j].y = vx*fast::sin(theta) + vy*fast::cos(theta);
                }
            }

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar vel = sqrt(h_vel.data[j].x*h_vel.data[j].x+h_vel.data[j].y*h_vel.data[j].y+h_vel.data[j].z*h_vel.data[j].z);
            if ( (vel*m_deltaT) > m_limit_val)
                {
                h_vel.data[j].x = h_vel.data[j].x / vel * m_limit_val / m_deltaT;
                h_vel.data[j].y = h_vel.data[j].y / vel * m_limit_val / m_deltaT;
                h_vel.data[j].z = h_vel.data[j].z / vel * m_limit_val / m_deltaT;
                }
            }
        }

    if (m_aniso)
        {
        // angular degrees of freedom
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            quat<Scalar> q(h_orientation.data[j]);
            quat<Scalar> p(h_angmom.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
            vec3<Scalar> I(h_inertia.data[j]);

            // rotate torque into principal frame
            t = rotate(conj(q),t);

            // check for zero moment of inertia
            bool x_zero, y_zero, z_zero;
            x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

            // ignore torque component along an axis for which the moment of inertia zero
            if (x_zero) t.x = 0;
            if (y_zero) t.y = 0;
            if (z_zero) t.z = 0;

            // advance p(t+deltaT/2)->p(t+deltaT)
            p += m_deltaT*q*t;

            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepCustomScatter2D(py::module& m)
    {
    py::class_<TwoStepCustomScatter2D, std::shared_ptr<TwoStepCustomScatter2D> >(m, "TwoStepCustomScatter2D", py::base<IntegrationMethodTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, unsigned int, unsigned int, unsigned int, std::shared_ptr<Variant>, bool, bool >())
        .def("setLimit", &TwoStepCustomScatter2D::setLimit)
        .def("removeLimit", &TwoStepCustomScatter2D::removeLimit)
        .def("setZeroForce", &TwoStepCustomScatter2D::setZeroForce)
        .def("setTables", &TwoStepCustomScatter2D::setTables)
        .def("setGamma", &TwoStepCustomScatter2D::setGamma)
        .def("setT", &TwoStepCustomScatter2D::setT)
        ;
    }
