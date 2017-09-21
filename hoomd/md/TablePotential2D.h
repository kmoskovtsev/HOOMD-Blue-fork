// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Author: Kirill Moskovtsev

#include "hoomd/ForceCompute.h"
#include "NeighborList.h"
#include "hoomd/Index1D.h"
#include "hoomd/GPUArray.h"

#include <memory>

/*! \file TablePotential2D.h
    \brief Declares the TablePotential2D class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __TABLEPOTENTIAL_H__
#define __TABLEPOTENTIAL_H__

//! Computes the potential and force on each particle based on values given in a 2D table
/*! \b Overview
    Pair potentials and forces are evaluated for all particle pairs in the system (no cutoff).
    Both the potentials and forces** are provided the tables V(r) and F(r) at r mesh covering upper-right quarter 
    of a rectangular unit cell. The mesh is shifted from the origin by half of a period in each direction
    to avoid sampling singular value at the origin.
    Evaluations are performed by bilinear interpolation, thus why F(r) must be explicitly specified to
    avoid large errors resulting from the numerical derivative. Note that F(r) should store - dV/dr.

    \b Table memory layout

    V(r) and F(r) are the same for all particles, regardless of their types. The class is written for electrons
    on helium, where there is only one type of particles.

    To improve cache coherency even further, values for V and F will be interleaved like so: V1 F1 V2 F2 V3 F3 ... To
    accomplish this, tables are stored with a value type of Scalar3, elem.x will be V, elem.y will be F_x, and elem_z will be F_y.
    Since Fn, Vn+1 and Fn+1 are read right after Vn, these are likely to be cache hits. Furthermore, on the GPU a single Scalar2
    texture read can be used to access Vn and Fn.

    Two parameters need to be stored for the potential: h1 and h2, mesh steps along x and y directions.
    For simple access on the GPU, these will be stored in a Scalar2 where
    x is h1, y is h2.

    V(0,0) is the value of potential at the nearest upper-right mesh point from the particle. V(i,j) would be the value of 
    the potential at point ((i + 1/2)*h1, (j + 1/2)*h2) relative to the particle creating the field. Same is true for 
    F_x and F_y force components.

    \b Interpolation
    Values are interpolated bilinearly between four points that surround given r. For a given r, the lower i index of those four points
    can be calculated via i = floorf((r_x - h1/2) / h1). The upper i_high index would be i + 1. Analogous formula works for j indices.
    The fraction between r_xi and r_xi+1 can be calculated via
    f1 = (rx - h1/2) / h1 - Scalar(i). And the bilinear interpolation can then be performed via 
    V(r) ~= Vij + f1*(Vi+1j - Vij) + f2*(Vij+1 - Vij) + f1*f2*(Vij + Vi+1j+1 - Vij+1 - Vi+1j)
    The same approach is used for each force component.
    \ingroup computes
*/
class TablePotential2D : public ForceCompute
    {
    public:
        //! Constructs the compute
        TablePotential2D(std::shared_ptr<SystemDefinition> sysdef,,
                       unsigned int table_width,
                       unsigned int table_height,
                       const std::string& log_suffix="");

        //! Destructor
        virtual ~TablePotential2D();

        //! Set the table for a given type pair
        virtual void setTable(const std::vector<Scalar> &V,
                              const std::vector<Scalar> &F1,
                              const std::vector<Scalar> &F2);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

    protected:
        unsigned int m_table_width;                 //!< Width of the tables in memory
        unsigned int m_table_height;                 //!< Height of the tables in memory
        unsigned int m_ntypes;                      //!< Store the number of particle types
        GPUArray<Scalar3> m_tables;                  //!< Stored V and F tables
        GPUArray<Scalar2> m_params;                 //!< Parameters stored for each table
        std::string m_log_name;                     //!< Cached log name

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! Method to be called when number of types changes
        //virtual void slotNumTypesChange();
    };

//! Exports the TablePotential class to python
void export_TablePotential2D(pybind11::module& m);

#endif
