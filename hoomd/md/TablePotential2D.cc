// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
#include "TablePotential2D.h"

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
        : ForceCompute(sysdef), m_table_width(table_width), m_table_height(table_height)
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

    GPUArray<Scalar4> tables(m_table_width, m_table_height, m_exec_conf);
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
    ArrayHandle<Scalar4> h_tables(m_tables, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    unsigned int pitch = m_tables.getPitch();
    //std::cout << "Pitch = " << pitch << '\n';
    //std::cout << "len(h_tables.data) = " << m_tables.getNumElements() << '\n';
    //std::cout << "len(V)" << V.size() << '\n';
    
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
    Scalar3 box_L = box.getL();
    // fill out the parameters
    h_params.data[0].x = box_L.x/Scalar(m_table_width)*Scalar(0.5); //mesh step along x
    h_params.data[0].y = box_L.y/Scalar(m_table_height)*Scalar(0.5); //mesh step along y
    //h_params.data[cur_table_index].z = (rmax - rmin) / Scalar(m_table_width - 1);

    // fill out the table
    for (unsigned int j = 0; j < m_table_height; j++)
        {
        for (unsigned int i = 0; i < m_table_width; i++)
            {
            //std::cout << "j*pitch + i = " << j*pitch + i << '\n';
            //std::cout << "j*m_table_width + i = " << j*m_table_width + i << '\n';
            h_tables.data[j*pitch + i].x = V[j*m_table_width + i];
            h_tables.data[j*pitch + i].y = F1[j*m_table_width + i];
            h_tables.data[j*pitch + i].z = F2[j*m_table_width + i];
            h_tables.data[j*pitch + i].w = 0;
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
/*! \post Calculate absolute value of Scalar3 component-wise:

   */
Scalar3 Scalar3Abs(Scalar3 r)
    {
    Scalar3 res = r;
    res.x =  std::abs(res.x);
    res.y =  std::abs(res.y);
    res.z =  std::abs(res.z);
    return res;
    }


/*! \post Restore the force direction using mirror symmetry.
    Originally, the force is computed for dx reflected into upper-right quarter of the unit cell.
    VF = (V, F_x, F_y)
    dx = vector pointing from particle i to particle k
*/
Scalar4 restoreForceDirection(Scalar4 VF, Scalar3 dx)
    {
    if (dx.x < 0)
        {
        VF.y = - VF.y;
        }
    if (dx.y < 0)
        {
        VF.z = - VF.z;
        }
    return VF;
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
    
    //to check that m_pdata is valid
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

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
    ArrayHandle<Scalar4> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);

    int pitch = m_tables.getPitch();

    // index calculation helpers
    //Index2DUpperTriangular table_index(m_ntypes);
    //Index2D table_value(m_table_width);

    Scalar2 params = h_params.data[0];
    Scalar h1 = params.x; //step along x
    Scalar h2 = params.y; //step along y

    //std::cout << "m_tables size = " << m_tables.getNumElements() << '\n';

    // for each particle
    for (int i = 0; i < (int) m_pdata->getN(); i++)
        {
        //std::cout << "===================================\n";
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        //std::cout << "pi = (" << pi.x << ", " << pi.y << ")\n";
        //std::cout << "charge(i) = " << h_charge.data[i] << '\n';

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
        for (unsigned int j = i + 1; j < (unsigned int) m_pdata->getN(); j++)
            {
            // access the index of this neighbor
            unsigned int k = j;

            //std::cout << "----------------------------------\n";
            // calculate dr
            Scalar3 pk = make_scalar3(h_pos.data[k].x, h_pos.data[k].y, h_pos.data[k].z);
            Scalar3 dx = pi - pk;
            //std::cout << "pk = (" << pk.x << ", " << pk.y << ")\n";
            //std::cout << "charge(k) = " << h_charge.data[k] << '\n';

            // access the type of the neighbor particle
            //unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            //assert(typej < m_pdata->getNTypes());

            //std::cout << "dx.x before minImage: " << dx.x << '\n';
            //std::cout << "dx.y before minImage: " << dx.y << '\n';
            // apply periodic boundary conditions
            dx = box.minImage(dx);
            //Calculate force for absolute of dx first (component-wise),
            //restore direction from symmetry later
            Scalar3 dxa = Scalar3Abs(dx);

            // start computing the force
            //Use bilinear interpolation. f1 and f2 are "locations" of the dxa on the grid
            Scalar value_f1 = (dxa.x - h1*Scalar(0.5)) / h1;
            Scalar value_f2 = (dxa.y - h2*Scalar(0.5)) / h2;

            // compute index into the table and read in values
            int value_i = (int)floor(value_f1);
            int value_j = (int)floor(value_f2);
            Scalar4 zeroScalar4 = make_scalar4(0, 0, 0, 0);
            //init potential-force values at the adjacent nodes
            Scalar4 VF00 = zeroScalar4;
            Scalar4 VF01 = zeroScalar4;
            Scalar4 VF10 = zeroScalar4;
            Scalar4 VF11 = zeroScalar4;
            /*
            std::cout << "i = " << i << ", j = " << j << '\n';
            std::cout << "dx.x = " << dx.x << '\n';
            std::cout << "dx.y = " << dx.y << '\n';
            std::cout << "dxa.x = " << dxa.x << '\n';
            std::cout << "dxa.y = " << dxa.y << '\n';
            std::cout << "value_f1 = " << value_f1 << '\n';
            std::cout << "value_f2 = " << value_f2 << '\n';
            std::cout << "value_i = " << value_i << '\n';
            std::cout << "value_j = " << value_j << '\n';
            std::cout << "pitch = " << pitch << '\n';
            std::cout << "value_j*pitch + value_i = " << value_j*pitch + value_i << '\n';
            */
            if (value_i == -1 && value_j == -1)
                {
                VF11 = h_tables.data[0];
                VF00 = VF11;
                VF00.y = - VF00.y; //invert both force components
                VF00.z = - VF00.z;
                VF01 = VF11;
                VF01.y = - VF01.y; //invert F_x
                VF10 = VF11;
                VF10.z = - VF10.z; //invert F_y
                }
            else if (value_i == - 1 && value_j > -1 && value_j < (int) m_table_height - 1)
                {
                VF10 = h_tables.data[value_j*pitch + value_i + 1];
                VF11 = h_tables.data[(value_j + 1)*pitch + value_i + 1];
                VF00 = VF10;
                VF00.y = - VF00.y; //invert F_x
                VF01 = VF11;
                VF01.y = - VF01.y;//invert F_x
                }
            else if (value_i == -1 && value_j == (int) m_table_height - 1)
                {
                VF10 = h_tables.data[value_j*pitch + value_i + 1];
                VF00 = VF10;
                VF00.y = - VF00.y; //invert F_x
                VF01 = VF00;
                VF01.z = - VF01.z; //invert F_x and F_y
                VF11 = VF10;
                VF11.z = - VF11.z; //invert F_y
                }
            else if (value_i > -1 && value_i < (int) m_table_width - 1 && value_j == (int) m_table_height - 1)
                {
                VF00 = h_tables.data[value_j*pitch + value_i];
                VF10 = h_tables.data[value_j*pitch + value_i + 1];
                VF01 = VF00;
                VF01.z = - VF01.z;
                VF11 = VF10;
                VF11.z = - VF11.z;
                }
            else if (value_i == (int) m_table_width - 1 && value_j == (int) m_table_height - 1)
                {
                VF00 = h_tables.data[value_j*pitch + value_i];
                VF10 = VF00;
                VF10.y = - VF10.y; //reflect F_x
                VF01 = VF00;
                VF01.z = - VF01.z; //reflect F_y
                VF11 = VF01;
                VF11.y = - VF11.y;
                }
            else if (value_i == (int) m_table_width - 1 && value_j > -1 && value_j < (int) m_table_height - 1)
                {
                VF00 = h_tables.data[value_j*pitch + value_i];
                VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                VF10 = VF00;
                VF10.y = - VF10.y;
                VF11 = VF01;
                VF11.y = - VF11.y;
                }
            else if (value_i == (int) m_table_width - 1 && value_j == -1)
                {
                VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                VF00 = VF01;
                VF00.z = - VF00.z;
                VF10 = VF00;
                VF10.y = - VF10.y;
                VF11 = VF01;
                VF11.y = - VF11.y;
                }
            else if (value_i > -1 && value_i < (int) m_table_width - 1 && value_j == -1)
                {
                VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                VF11 = h_tables.data[(value_j + 1)*pitch + value_i + 1];
                VF00 = VF01;
                VF00.z = - VF00.z;
                VF10 = VF11;
                VF10.z = - VF10.z;
                }
            else
                {
                VF00 = h_tables.data[value_j*pitch + value_i];
                VF01 = h_tables.data[(value_j + 1)*pitch + value_i];
                VF10 = h_tables.data[value_j*pitch + value_i + 1];
                VF11 = h_tables.data[(value_j + 1)*pitch + value_i + 1];
                }

            // compute the bilinear interpolation coefficient
            Scalar f1 = value_f1 - Scalar(value_i);
            Scalar f2 = value_f2 - Scalar(value_j);

            // interpolate to get V and F;
            //Bilinear interpolation:
            Scalar4 VF = VF00 + f1*(VF10 - VF00) + f2*(VF01 - VF00) + f1*f2*(VF00 + VF11 - VF01 - VF10);
            /*
            if (VF.y > 10000 || VF.z > 10000 || VF.y < -10000 || VF.z < -10000)
                {
                std::cout << "VF00.y = " << VF00.y << '\n';
                std::cout << "VF00.z = " << VF00.z << '\n';
                std::cout << "VF01.y = " << VF01.y << '\n';
                std::cout << "VF01.z = " << VF00.z << '\n';
                std::cout << "VF10.y = " << VF10.y << '\n';
                std::cout << "VF10.z = " << VF10.z << '\n';
                std::cout << "VF11.y = " << VF11.y << '\n';
                std::cout << "VF11.z = " << VF11.z << '\n';
                }
                */
            VF = restoreForceDirection(VF, dx);
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

            //std::cout << "F_i = (" << VF.y << ", " << VF.z << ")\n";
            //std::cout << "pair energy = " << pair_eng << '\n';

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

//! Exports the TablePotential2D class to python
void export_TablePotential2D(py::module& m)
    {
    py::class_<TablePotential2D, std::shared_ptr<TablePotential2D> >(m, "TablePotential2D", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, const std::string& >())
    .def("setTable", &TablePotential2D::setTable)
    ;
    }
