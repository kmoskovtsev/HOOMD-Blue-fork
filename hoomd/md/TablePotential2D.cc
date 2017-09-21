// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
#include "TablePotential.h"

namespace py = pybind11;

#include <stdexcept>

/*! \file TablePotential.cc
    \brief Defines the TablePotential class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
TablePotential2D::TablePotential2D(std::shared_ptr<SystemDefinition> sysdef,
                               unsigned int table_width,
                               unsigned int table_height,
                               const std::string& log_suffix)
        : ForceCompute(sysdef), m_nlist(nlist), m_table_width(table_width), m_table_height(table_height)
    {
    m_exec_conf->msg->notice(5) << "Constructing TablePotential" << endl;

    // sanity checks
    assert(m_pdata);
    assert(m_nlist);

    const BoxDim& box = m_pdata->getBox();
    if (box.getTiltFactorXY() != 0)
        {
        m_exec_conf->msg->error() << "pair.table2D: only rectangular unit cell is allowed" << endl;
        throw runtime_error("Error initializing TablePotential2D");
        }

    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);

    // allocate storage for the tables and parameters
    //Index2DUpperTriangular table_index(m_ntypes); //type pairs table?
    
    /*allocate 2-D table with each entry being 3-component vector:
    First component is V and the other two are F_x and F_y.
    The mesh covers one-quarter of unit cell to the right and up from the particle.
    Unit cell is usually from -L/2 to L/2.
    Other three quarters will be obtained by mirror symmetry about x and y axes.
    The origin of the mesh is shifted by 1/2 of its periods from the origin (which is particle position),
    so that singular value at the origin is not included in the mesh.
    Edge points of the mesh are 1/2 period apart from the simulation box edges.
    x and y periods may be different, periods are h1 = a1/table_width and h2 = a2/table_height.*/

    GPUArray<Scalar3> tables(m_table_height, m_table_width, m_exec_conf);
    m_tables.swap(tables);
    GPUArray<Scalar2> params(1, m_exec_conf);
    m_params.swap(params);

    assert(!m_tables.isNull());
    assert(!m_params.isNull());

    m_log_name = std::string("pair_table_energy") + log_suffix;

    // connect to the ParticleData to receive notifications when the number of types changes
    //m_pdata->getNumTypesChangeSignal().connect<TablePotential, &TablePotential2D::slotNumTypesChange>(this);
    }

TablePotential2D::~TablePotential2D()
    {
    m_exec_conf->msg->notice(5) << "Destroying TablePotential" << endl;

    //m_pdata->getNumTypesChangeSignal().disconnect<TablePotential, &TablePotential2D::slotNumTypesChange>(this);
    }
/*
void TablePotential2D::slotNumTypesChange()
    {
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);

    // skip the reallocation if the number of types does not change
    // this keeps old parameters when restoring a snapshot
    // it will result in invalid coeficients if the snapshot has a different type id -> name mapping
    if (m_ntypes*(m_ntypes+1)/2 == m_params.getNumElements())
        return;

    // allocate storage for the tables and parameters
    Index2DUpperTriangular table_index(m_ntypes);
    GPUArray<Scalar2> tables(m_table_width, table_index.getNumElements(), m_exec_conf);
    m_tables.swap(tables);
    GPUArray<Scalar4> params(table_index.getNumElements(), m_exec_conf);
    m_params.swap(params);

    assert(!m_tables.isNull());
    assert(!m_params.isNull());
    }
    */
/*!
    \typ1 must be equal typ2 for simulation of electrons
    \param V Table for the potential V. V is a flat vector of size (table_width x table_height).
            Rows of mesh data are merged starting from upper left corner.
    \param F1 and F2 Tables for the components of force F (must be - dV / dr)
            
    \post Values from \a V and \a F are copied into the interal storage
*/
void TablePotential2D::setTable(const std::vector<Scalar> &V,
                              const std::vector<Scalar> &F1,
                              const std::vector<Scalar> &F2)
    {

    // access the arrays
    ArrayHandle<Scalar3> h_tables(m_tables, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    unsigned int pitch = m_tables.getPitch();
    //Check size:
    if (V.size() != m_table_width*m_table_height || F1.size() != m_table_width*m_table_height ||
            F2.size() != m_table_width*m_table_height)
        {
        m_exec_conf->msg->error() << "pair.table2D: table (F1, F2, or V) provided to setTable is not of the correct size" << endl;
        throw runtime_error("Error initializing TablePotential");
        }
    if (m_table_width <= 0 || m_table_height <= 0)
        {
        m_exec_conf->msg->error() << "pair.table2D: table width and height must be positive integers" << endl;
        throw runtime_error("Error initializing TablePotential2D");
        }
    const BoxDim& box = m_pdata->getBox();
    Scalar3 box_L = box.get_L;
    // fill out the parameters
    h_params.data[0].x = box_L.x/m_table_width/2; //mesh step along x
    h_params.data[0].y = box_L.y/m_table_height/2; //mesh step along y
    //h_params.data[cur_table_index].z = (rmax - rmin) / Scalar(m_table_width - 1);

    // fill out the table
    for (unsigned int i = 0; i < m_table_width; i++)
        {
        for (unsigned int j = 0; j < m_table_height; j++)
            {
            h_tables.data[j*pitch + i].x = V[j*m_table_width + i];
            h_tables.data[j*pitch + i].y = F1[j*m_table_width + i];
            h_tables.data[j*pitch + i].z = F2[j*m_table_width + i];
            }
        }
    }

/*! TablePotential provides
    - \c pair_table_energy
*/
std::vector< std::string > TablePotential2D::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

Scalar TablePotential2D::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "pair.table2D: " << quantity << " is not a valid log quantity for TablePotential2D" << endl;
        throw runtime_error("Error getting log value");
        }
    }
/* Calculate absolute value of Scalar3 component-wise:

   */
Scalar3 Scalar3Abs(Scalar3 r)
    {
    Scalar3 res = r;
    if (res.x < 0)
        {
            res.x = - res.x;
        }
    if (res.x < 0)
        {
            res.y = - res.y;
        }
    if (res.z < 0)
        {
            res.z = - res.z;
        }
    return res;
    }

/*! \post The table based forces are computed for the given timestep. The neighborlist's
compute method is called to ensure that it is up to date.

\param timestep specifies the current time step of the simulation
*/


void TablePotential2D::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    // No neighbor list in electron version. Simply loop over all pairs.
    //m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push("Table pair");

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = true;

    // access the neighbor list
    //ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    //ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    //ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);

    unsigned int pitch = m_tables.getPitch();

    // index calculation helpers
    //Index2DUpperTriangular table_index(m_ntypes);
    //Index2D table_value(m_table_width);

    // for each particle
    for (int i = 0; i < (int) m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        //unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        //const unsigned int head_i = h_head_list.data[i];
        // sanity check
        //assert(typei < m_pdata->getNTypes());

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0,0,0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all particles (all particles are neighbors for long range)
        //const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = i + 1; j < (int) m_pdata->getN(); j++)
            {
            // access the index of this neighbor
            unsigned int k = j;

            // calculate dr
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;

            // access the type of the neighbor particle
            //unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            //assert(typej < m_pdata->getNTypes());

            // apply periodic boundary conditions
            dx = box.minImage(dx);
            //Calculate force for absolute of dx first (component-wise),
            //restore direction from symmetry later
            Scalar3 dxa = Scalar3Abs(dx);
            // access needed parameters
            //unsigned int cur_table_index = table_index(typei, typej);
            Scalar4 params = h_params.data[0];
            Scalar h1 = params.x; //step along x
            Scalar h2 = params.y; //step along y

            // start computing the force
            Scalar rsq = dot(dx, dx);
            Scalar r = sqrt(rsq);

            Scalar value_f1 = (dxa.x - h1*Scalar(0.5)) / h1;
            Scalar value_f2 = (dxa.y - h2*Scalar(0.5)) / h2;

            // compute index into the table and read in values
            unsigned int value_i = (unsigned int)floor(value_f1);
            unsigned int value_j = (unsigned int)floor(value_f2);
            if (value_i == -1 && value_j == -1)
                {
                Scalar3 VF11 = h_tables.data[0];
                Scalar3 VF00 = VF11;
                VF00.y = - VF00.y; //invert both force components
                VF00.z = - VF00.z;
                Scalar3 VF01 = VF11;
                VF01.y = - VF01.y; //invert F_x
                Scalar3 VF10 = F11;
                VF10.z = - VF10.z; //invert F_y
                }
            else if (value_i == - 1 && value_j > -1 && value_j < m_table_height - 1)
                {
                Scalar3 VF10 = h_tables.data[value_j*pitch + value_i + 1];
                Scalar3 VF11 = h_tables.data[(value_j + 1)*pitch + value_i + 1];
                Scalar3 VF00 = VF10;
                VF00.y = - VF00.y; //invert F_x
                Scalar3 VF01 = V11;
                VF01.y = - VF01.y;//invert F_x
                }
            else if (value_i == -1 && value_j == m_table_height - 1)
                {
                Scalar3 VF10 = h_tables.data[value_j*pitch + value_i + 1];
                Scalar3 VF00 = VF10;
                VF00.y = - VF00.y; //invert F_x
                Scalar3 VF01 = VF00;
                VF01.z = - VF01.z; //invert F_x and F_y
                Scalar3 VF11 = VF10;
                VF11.z = - VF11.z; //invert F_y
                }
            else if (value_i > -1 && value_i < m_table_width - 1 && value_j == -1)
                {
                Scalar3 VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                Scalar3 VF11 = h_tables.data[(value_j + 1)*pitch + value_i + 1];
                Scalar3 VF00 = VF01;
                VF00.z = - VF00.z;
                Scalar3 VF10 = VF11;
                VF10.z = - VF10.z;
                }
            else if (value_i == m_table_width - 1 && value_j == m_table_height - 1)
                {
                Scalar3 VF00 = h_tables.data[value_j*pitch + value_i];
                Scalar3 VF10 = VF00;
                VF10.y = - VF10.y; //reflect F_x
                Scalar3 VF01 = VF00;
                VF01.z = - VF01.z; //reflect F_y
                Scalar3 VF11 = VF01;
                VF11.y = - VF11.y;
                }
            else if (value_i == m_table_width - 1 && value_j > -1 && value_j < m_table_height - 1)
                {
                Scalar3 VF00 = h_tables.data[value_j*pitch + value_i];
                Scalar3 VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                Scalar3 VF10 = VF00;
                VF10.y = - VF10.y;
                Scalar3 VF11 = VF01;
                VF11.y = - VF11.y;
                }
            else if (value_i == m_table_width - 1 && value_j == -1)
                {
                Scalar3 VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                Scalar3 VF00 = VF01;
                VF00.z = - VF00.z;
                Scalar3 VF10 = VF00;
                VF10.y = - VF10.y;
                Scalar3 VF11 = VF01;
                VF11.y = - VF11.y;
                }
            else if (value_i > -1 && value_i < m_table_width - 1 && j == -1)
                {
                Scalar3 VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                Scalar3 VF11 = h_tables.data[(value_j + 1)*pitch + value_i + 1];
                Scalar3 VF00 = VF01;
                VF00.z = - VF00.z;
                Scalar3 VF10 = VF11;
                VF10.z = - VF10.z;
                }
            else
                {
                Scalar3 VF00 = h_tables.data[value_j*pitch + value_i];
                Scalar3 VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                Scalar3 VF10 = h_tables.data[value_j*pitch + value_i + 1];
                Scalar3 VF11 = h_tables.data[(value_j + 1)*pitch + value_i + 1];
                }
            // unpack the data
            //Scalar V00 = VF00.x;
            //Scalar V01 = V
            //Scalar V1 = VF1.x;
            //Scalar F0 = VF0.y;
            //Scalar F1 = VF1.y;

            // compute the linear interpolation coefficient
            Scalar f1 = value_f1 - Scalar(value_i);
            Scalar f2 = value_f2 - Scalar(value_j);

            // interpolate to get V and F;
            //Scalar V = V0 + f * (V1 - V0);
            //Scalar F = F0 + f * (F1 - F0);
            //Bilinear interpolation:
            Scalar3 VF = VF00 + f1*(VF10 - VF00) + f2*(V01 - V00) + f1*f2*(V00 + V11 - V01 - V10);

            // convert to standard variables used by the other pair computes in HOOMD-blue
            Scalar pair_eng = Scalar(0.5) * VF.x;
            Scalar Fx_div2 = Scalar(0.5)*VF.y;
            Scalar Fy_div2 = Scalar(0.5)*VF.z;
            Scalar Fz_div2 = 0;
            // compute the virial
            //Scalar forcemag_div2r = Scalar(0.5) * forcemag_divr;
            virialxxi += Fx_div2*dx.x;
            virialxyi += Fx_div2*dx.y;
            virialxzi += Fx_div2*dx.z;
            virialyyi += Fy_div2*dx.y;
            virialyzi += Fy_div2*dx.z;
            virialzzi += Fz_div2*dx.z;

            // add the force, potential energy and virial to the particle i
            fi.x += VF.y;
            fi.y += VF.z;
            pei += pair_eng;

            // add the force to particle j if we are using the third law
            // only add force to local particles
            if (third_law && k < m_pdata->getN())
                {
                unsigned int mem_idx = k;
                h_force.data[mem_idx].x -= VF.y;
                h_force.data[mem_idx].y -= VF.z;
                //h_force.data[mem_idx].z -= dx.z*forcemag_divr;
                h_force.data[mem_idx].w += pair_eng;
                h_virial.data[0*m_virial_pitch+mem_idx] += Fx_div2 * dx.x;
                h_virial.data[1*m_virial_pitch+mem_idx] += Fx_div2 * dx.y;
                h_virial.data[2*m_virial_pitch+mem_idx] += Fx_div2 * dx.z;
                h_virial.data[3*m_virial_pitch+mem_idx] += Fy_div2 * dx.y;
                h_virial.data[4*m_virial_pitch+mem_idx] += Fy_div2 * dx.z;
                h_virial.data[5*m_virial_pitch+mem_idx] += Fz_div2 * dx.z;
                }
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        //h_force.data[mem_idx].z += fi.z;
        h_force.data[mem_idx].w += pei;
        h_virial.data[0*m_virial_pitch+mem_idx] += virialxxi;
        h_virial.data[1*m_virial_pitch+mem_idx] += virialxyi;
        h_virial.data[2*m_virial_pitch+mem_idx] += virialxzi;
        h_virial.data[3*m_virial_pitch+mem_idx] += virialyyi;
        h_virial.data[4*m_virial_pitch+mem_idx] += virialyzi;
        h_virial.data[5*m_virial_pitch+mem_idx] += virialzzi;
        }

    if (m_prof) m_prof->pop();
    }

//! Exports the TablePotential class to python
void export_TablePotential2D(py::module& m)
    {
    py::class_<TablePotential2D, std::shared_ptr<TablePotential2D> >(m, "TablePotential2D", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, unsigned int, const std::string& >())
    .def("setTable", &TablePotential2D::setTable)
    ;
    }
