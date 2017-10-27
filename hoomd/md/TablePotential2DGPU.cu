// Copyright (c) 2009-2016 The Regents of the University of Michigan


// Author: Kirill Moskovtsev

#include "TablePotential2DGPU.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/Index1D.h"

#include <assert.h>

/*! \file TablePotential2DGPU.cu
    \brief Defines GPU kernel code for calculating the table pair forces. Used by TablePotential2DGPU.
*/

//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;

//! Texture for reading the neighborlist
//texture<unsigned int, 1, cudaReadModeElementType> nlist_tex;

//! Texture for reading table values
scalar4_tex_t tables_tex;

scalar2_tex_t params_tex;
/* \post Calculate absolute value of Scalar3 component-wise:

*/
__device__ Scalar3 d_Scalar3Abs(Scalar3 r)
    {
    Scalar3 res = r;
    if (res.x < 0)
        {
        res.x = - res.x;
        }
    if (res.y < 0)
        {
       res.y = - res.y;
        }
    if (res.z < 0)
        {
        res.z = - res.z;
        }
    return res;
    }


/*! \post Restore the force direction using mirror symmetry.
    Originally, the force is computed for dx reflected into upper-right quarter of the unit cell.
    VF = (V, F_x, F_y)
    dx = vector pointing from particle i to particle k
*/
__device__ Scalar4 d_restoreForceDirection(Scalar4 VF, Scalar3 dx)
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



/*!  This kernel is called to calculate the 2D-table pair forces on all N particles

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles in system
    \param d_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Indexer for reading \a d_nlist
    \param d_params Parameters for each table associated with a type pair
    \param ntypes Number of particle types in the system
    \param table_width Number of points in each table

    See TablePotential for information on the memory layout.

    \tparam use_gmem_nlist When non-zero, the neighbor list is read out of global memory. When zero, textures or __ldg
                           is used depending on architecture.

    \b Details:
    * Table entries are read from tables_tex. Note that currently this is bound to a 1D memory region. Performance tests
      at a later date may result in this changing.
*/
__global__ void gpu_compute_table2D_forces_kernel(Scalar4* d_force,
                                                Scalar* d_virial,
                                                const unsigned virial_pitch,
                                                const unsigned int N,
                                                const Scalar4 *d_pos,
                                                const BoxDim box,
                                                const Scalar4 *d_tables,
                                                const unsigned tables_pitch,
                                                const Scalar2 *d_params,
                                                const unsigned int table_width,
                                                const unsigned int table_height)
    {

    // read in params for easy and fast access in the kernel
    // params.x  h1 (grid step along x), params.y = h2 (step along y)
    Scalar2 params = texFetchScalar2(d_params, params_tex, 0);
    // access needed parameters
    Scalar h1 = params.x;
    Scalar h2 = params.y;


    //for (unsigned int cur_offset = 0; cur_offset < table_index.getNumElements(); cur_offset += blockDim.x)
    //    {
    //    if (cur_offset + threadIdx.x < table_index.getNumElements())
    //        s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
    //    }
    //__syncthreads();


    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    //printf("idx = %d\n", idx);
    // read in the position of our particle. Texture reads of Scalar4's are faster than global reads on compute 1.0 hardware
    Scalar4 postype = texFetchScalar4(d_pos, pdata_pos_tex, idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virialxx = Scalar(0.0);
    Scalar virialxy = Scalar(0.0);
    Scalar virialxz = Scalar(0.0);
    Scalar virialyy = Scalar(0.0);
    Scalar virialyz = Scalar(0.0);
    Scalar virialzz = Scalar(0.0);

    
    // loop over neighbors
    for (int cur_neigh = 0; cur_neigh < N; cur_neigh++)
        {
        if (cur_neigh == idx)
            {
            // skip the particle itself
            continue;
            }

        Scalar4 postype_n = texFetchScalar4(d_pos, pdata_pos_tex, cur_neigh);
        Scalar3 neigh_pos = make_scalar3(postype_n.x, postype_n.y, postype_n.z);
        
        //printf("neigh_pos = %f %f\n", neigh_pos.x, neigh_pos.y);

        // calculate dr (with periodic boundary conditions)
        Scalar3 dx = pos - neigh_pos;

        //printf("dx = %f %f", dx.x, dx.y);
        
        // apply periodic boundary conditions
        dx = box.minImage(dx);
        Scalar3 dxa = d_Scalar3Abs(dx);

        Scalar value_f1 = (dxa.x - h1*Scalar(0.5)) / h1;
        Scalar value_f2 = (dxa.y - h2*Scalar(0.5)) / h2;

        // compute index into the table and read in values
        int value_i = (int)floor(value_f1);
        int value_j = (int)floor(value_f2);

        //printf("value_i = %d, value_j = %d\n", value_i, value_j);

        Scalar4 zeroScalar4 = make_scalar4(0, 0, 0, 0);
        //init potential-force values at the adjacent nodes
        Scalar4 VF00 = zeroScalar4;
        Scalar4 VF01 = zeroScalar4;
        Scalar4 VF10 = zeroScalar4;
        Scalar4 VF11 = zeroScalar4;

        if (value_i == -1 && value_j == -1)
            {
            //printf("Case 1\n");
            VF11 = texFetchScalar4(d_tables, tables_tex, 0);
            VF00 = VF11;
            VF00.y = - VF00.y; //invert both force components
            VF00.z = - VF00.z;
            VF01 = VF11;
            VF01.y = - VF01.y; //invert F_x
            VF10 = VF11;
            VF10.z = - VF10.z; //invert F_y
            }
        else if (value_i == - 1 && value_j > -1 && value_j < (int) table_height - 1)
            {
            //printf("Case 2\n");
            VF10 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i + 1);
            VF11 = texFetchScalar4(d_tables, tables_tex, (value_j + 1)*tables_pitch + value_i + 1);
            VF00 = VF10;
            VF00.y = - VF00.y; //invert F_x
            VF01 = VF11;
            VF01.y = - VF01.y;//invert F_x
            }
        else if (value_i == -1 && value_j == (int) table_height - 1)
            {
            //printf("Case 3\n");
            VF10 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i + 1);
            VF00 = VF10;
            VF00.y = - VF00.y; //invert F_x
            VF01 = VF00;
            VF01.z = - VF01.z; //invert F_x and F_y
            VF11 = VF10;
            VF11.z = - VF11.z; //invert F_y
            }
        else if (value_i > -1 && value_i < (int) table_width - 1 && value_j == (int) table_height - 1)
            {
            //printf("Case 4\n");
            VF00 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i);
            VF10 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i + 1);
            VF01 = VF00;
            VF01.z = - VF01.z;
            VF11 = VF10;
            VF11.z = - VF11.z;
            }
        else if (value_i == (int) table_width - 1 && value_j == (int) table_height - 1)
            {
            //printf("Case 5\n");
            VF00 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i);
            VF10 = VF00;
            VF10.y = - VF10.y; //reflect F_x
            VF01 = VF00;
            VF01.z = - VF01.z; //reflect F_y
            VF11 = VF01;
            VF11.y = - VF11.y;
            }
        else if (value_i == (int) table_width - 1 && value_j > -1 && value_j < (int) table_height - 1)
            {
            //printf("Case 6\n");
            VF00 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i);
            VF01 = texFetchScalar4(d_tables, tables_tex, (value_j + 1)*tables_pitch + value_i);
            VF10 = VF00;
            VF10.y = - VF10.y;
            VF11 = VF01;
            VF11.y = - VF11.y;
            }
        else if (value_i == (int) table_width - 1 && value_j == -1)
            {
            //printf("Case 7\n");
            VF01 = texFetchScalar4(d_tables, tables_tex, (value_j + 1)*tables_pitch + value_i);
            VF00 = VF01;
            VF00.z = - VF00.z;
            VF10 = VF00;
            VF10.y = - VF10.y;
            VF11 = VF01;
            VF11.y = - VF11.y;
            }
        else if (value_i > -1 && value_i < (int) table_width - 1 && value_j == -1)
            {
            //printf("Case 8\n");
            VF01 = texFetchScalar4(d_tables, tables_tex, (value_j + 1)*tables_pitch + value_i);
            VF11 = texFetchScalar4(d_tables, tables_tex, (value_j + 1)*tables_pitch + value_i + 1);
            VF00 = VF01;
            VF00.z = - VF00.z;
            VF10 = VF11;
            VF10.z = - VF10.z;
            }
        else
            {
            //printf("Case 9\n");
            VF00 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i);
            VF01 = texFetchScalar4(d_tables, tables_tex, (value_j + 1)*tables_pitch + value_i);
            VF10 = texFetchScalar4(d_tables, tables_tex, value_j*tables_pitch + value_i + 1);
            VF11 = texFetchScalar4(d_tables, tables_tex, (value_j + 1)*tables_pitch + value_i + 1);
            }

        // compute the bilinear interpolation coefficient
        Scalar f1 = value_f1 - Scalar(value_i);
        Scalar f2 = value_f2 - Scalar(value_j);
        // interpolate to get V and F;
        //Bilinear interpolation:
        Scalar4 VF = VF00 + f1*(VF10 - VF00) + f2*(VF01 - VF00) + f1*f2*(VF00 + VF11 - VF01 - VF10);

        VF = d_restoreForceDirection(VF, dx);
        // convert to standard variables used by the other pair computes in HOOMD-blue
        Scalar pair_eng = VF.x;
        Scalar Fx_div2 = Scalar(0.5)*VF.y;
        Scalar Fy_div2 = Scalar(0.5)*VF.z;
        Scalar Fz_div2 = 0;


        // compute the virial
        //Scalar forcemag_div2r = Scalar(0.5) * forcemag_divr;
        virialxx += Fx_div2*dx.x;
        virialxy += Fx_div2*dx.y;
        virialxz += Fx_div2*dx.z;
        virialyy += Fy_div2*dx.y;
        virialyz += Fy_div2*dx.z;
        virialzz += Fz_div2*dx.z;
        // add the force, potential energy and virial to the particle i
        force.x += VF.y;
        force.y += VF.z;
        force.w += pair_eng;
        }


    //printf("Write force\n");
    // potential energy per particle must be halved
    force.w *= Scalar(0.5);
    // now that the force calculation is complete, write out the result
    d_force[idx] = force;

    //printf("write virial\n");
    d_virial[0*virial_pitch+idx] = virialxx;
    d_virial[1*virial_pitch+idx] = virialxy;
    d_virial[2*virial_pitch+idx] = virialxz;
    d_virial[3*virial_pitch+idx] = virialyy;
    d_virial[4*virial_pitch+idx] = virialyz;
    d_virial[5*virial_pitch+idx] = virialzz;
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_tables Tables of the potential and force
    \param tables_pitch Pitch of 2D array d_tables
    \param d_params Parameters h1 and h2
    \param table_width width of the table
    \param table_height Height of the table
    \param block_size Block size at which to run the kernel
    \param compute_capability Compute capability of the device (200, 300, 350)
    \param max_tex1d_width Maximum width of a linear 1d texture

    \note This is just a kernel driver. See gpu_compute_table2D_forces_kernel for full documentation.
*/
cudaError_t gpu_compute_table2D_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim& box,
                                     const Scalar4 *d_tables,
                                     const unsigned int tables_pitch,
                                     const Scalar2 *d_params,
                                     const unsigned int table_width,
                                     const unsigned int table_height,
                                     const unsigned int block_size,
                                     const unsigned int compute_capability,
                                     const unsigned int max_tex1d_width)
    {
    assert(d_tables);
    assert(table_width > 1);
    assert(table_height > 1);
    //printf("Entering table2D_forces");

    if (compute_capability < 350)
        {
        throw std::runtime_error("CUDA Compute below 3.5 not supported in TablePotential2DGPU.cu");
        }
    else
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_table2D_forces_kernel);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( N / run_block_size + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        //printf("before running table2D_forces_kernel");
        gpu_compute_table2D_forces_kernel<<<grid, threads>>>(d_force,
                                                                d_virial,
                                                                virial_pitch,
                                                                N,
                                                                d_pos,
                                                                box,
                                                                d_tables,
                                                                tables_pitch,
                                                                d_params,
                                                                table_width,
                                                                table_height);
        }

    return cudaSuccess;
    }
// vim:syntax=cpp
