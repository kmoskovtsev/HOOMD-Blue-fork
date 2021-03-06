// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "PPPMForceCompute.h"

#ifndef __PPPM_FORCE_COMPUTE_GPU_H__
#define __PPPM_FORCE_COMPUTE_GPU_H__

#ifdef ENABLE_CUDA

#include <cufft.h>
#include <sstream>

//#define USE_HOST_DFFT

#include "hoomd/Autotuner.h"

#ifdef ENABLE_MPI
#include "CommunicatorGridGPU.h"

#ifndef USE_HOST_DFFT
#include "hoomd/extern/dfftlib/src/dfft_cuda.h"
#else
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#endif
#endif

#define CHECK_CUFFT_ERROR(status) { handleCUFFTResult(status, __FILE__, __LINE__); }

/*! Order parameter evaluated using the particle mesh method
 */
class PPPMForceComputeGPU : public PPPMForceCompute
    {
    public:
        //! Constructor
        PPPMForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<NeighborList> nlist,
            std::shared_ptr<ParticleGroup> group);
        virtual ~PPPMForceComputeGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tuner_bin->setPeriod(period);
            m_tuner_assign->setPeriod(period);
            m_tuner_update->setPeriod(period);
            m_tuner_force->setPeriod(period);
            m_tuner_influence->setPeriod(period);

            m_tuner_bin->setEnabled(enable);
            m_tuner_assign->setEnabled(enable);
            m_tuner_update->setEnabled(enable);
            m_tuner_force->setEnabled(enable);
            m_tuner_influence->setEnabled(enable);
            }

    protected:
        //! Helper function to setup FFT and allocate the mesh arrays
        virtual void initializeFFT();

        //! Setup coefficient tables
        virtual void setupCoeffs();

        //! Helper function to assign particle coordinates to mesh
        virtual void assignParticles();

        //! Helper function to update the mesh arrays
        virtual void updateMeshes();

        //! Helper function to interpolate the forces
        virtual void interpolateForces();

        //! Compute the optimal influence function
        virtual void computeInfluenceFunction();

        //! Helper function to calculate value of collective variable
        virtual Scalar computePE();

        //! Helper function to compute the virial
        virtual void computeVirial();

        //! Helper function to correct forces on excluded particles
        virtual void fixExclusions();

        //! Check for CUFFT errors
        inline void handleCUFFTResult(cufftResult result, const char *file, unsigned int line) const
            {
            if (result != CUFFT_SUCCESS)
                {
                std::ostringstream oss;
                oss << "CUFFT returned error " << result << " in file " << file << " line " << line << std::endl;
                throw std::runtime_error(oss.str());
                }
            }

    private:
        std::unique_ptr<Autotuner> m_tuner_bin;  //!< Autotuner for binning particles
        std::unique_ptr<Autotuner> m_tuner_assign;//!< Autotuner for assigning binned charges to mesh
        std::unique_ptr<Autotuner> m_tuner_update;  //!< Autotuner for updating mesh values
        std::unique_ptr<Autotuner> m_tuner_force; //!< Autotuner for populating the force array
        std::unique_ptr<Autotuner> m_tuner_influence; //!< Autotuner for computing the influence function

        cufftHandle m_cufft_plan;          //!< The FFT plan
        bool m_local_fft;                  //!< True if we are only doing local FFTs (not distributed)

        #ifdef ENABLE_MPI
        typedef CommunicatorGridGPU<cufftComplex> CommunicatorGridGPUComplex;
        std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_forward; //!< Communicate mesh
        std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_reverse; //!< Communicate fourier mesh

        dfft_plan m_dfft_plan_forward;     //!< Forward distributed FFT
        dfft_plan m_dfft_plan_inverse;     //!< Forward distributed FFT
        #endif

        GPUArray<cufftComplex> m_mesh;                 //!< The particle density mesh
        GPUArray<cufftComplex> m_fourier_mesh;         //!< The fourier transformed mesh
        GPUArray<cufftComplex> m_fourier_mesh_G_x;       //!< Fourier transformed mesh times the influence function, x component
        GPUArray<cufftComplex> m_fourier_mesh_G_y;       //!< Fourier transformed mesh times the influence function, y component
        GPUArray<cufftComplex> m_fourier_mesh_G_z;       //!< Fourier transformed mesh times the influence function, z component
        GPUArray<cufftComplex> m_inv_fourier_mesh_x;     //!< The inverse-fourier transformed force mesh
        GPUArray<cufftComplex> m_inv_fourier_mesh_y;     //!< The inverse-fourier transformed force mesh
        GPUArray<cufftComplex> m_inv_fourier_mesh_z;     //!< The inverse-fourier transformed force mesh

        Index2D m_bin_idx;                         //!< Total number of bins
        GPUArray<Scalar4> m_particle_bins;         //!< Cell list for particle positions and modes
        GPUArray<Scalar> m_mesh_scratch;           //!< Mesh with scratch space for density reduction
        Index2D m_scratch_idx;                     //!< Indexer for scratch space
        GPUArray<unsigned int> m_n_cell;           //!< Number of particles per cell
        unsigned int m_cell_size;                  //!< Current max. number of particles per cell
        GPUFlags<unsigned int> m_cell_overflowed;  //!< Flag set to 1 if a cell overflows

        GPUFlags<Scalar> m_sum;                    //!< Sum over fourier mesh values
        GPUArray<Scalar> m_sum_partial;            //!< Partial sums over fourier mesh values
        GPUArray<Scalar> m_sum_virial_partial;     //!< Partial sums over virial mesh values
        GPUArray<Scalar> m_sum_virial;             //!< Final sum over virial mesh values
        unsigned int m_block_size;                 //!< Block size for fourier mesh reduction

        GPUFlags<Scalar4> m_gpu_q_max;             //!< Return value for maximum Fourier mode reduction
        GPUArray<Scalar4> m_max_partial;           //!< Scratch space for reduction of maximum Fourier amplitude
    };

void export_PPPMForceComputeGPU(pybind11::module& m);

#endif // ENABLE_CUDA
#endif // __PPPM_FORCE_COMPUTE_GPU_H__
