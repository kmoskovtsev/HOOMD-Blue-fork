/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

/*! \file NPTUpdaterGPU.cc
	\brief Defines the NPTUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "NPTUpdaterGPU.h"
#include "gpu_updaters.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
	\param tau Nose-Hoover period
	\param tauP barostat period
	\param T Temperature set point
	\param P Pressure set point
*/
NPTUpdaterGPU::NPTUpdaterGPU(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar tauP, Scalar T, Scalar P) : NPTUpdater(pdata, deltaT, tau, tauP, T, P)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	m_d_virial_data_ptrs.resize(exec_conf.gpu.size());
	// allocate and initialize force data pointers (if running on a GPU)
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_d_virial_data_ptrs[cur_gpu]), sizeof(float*)*32));
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_d_virial_data_ptrs[cur_gpu], 0, sizeof(float*)*32));
			}
		}

	// at least one GPU is needed
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a NPTUpdaterGPU with no GPU in the execution configuration." << endl << endl;
		throw std::runtime_error("Error initializing NPTUpdaterGPU.");
		}
	
	d_npt_data.resize(exec_conf.gpu.size());
	allocateNPTData(128);
	}

/*! NPTUpdaterGPU provides
        - \c npt_timestep
	- \c npt_temperature
	- \c npt_pressure
	- \c npt_volume
*/
std::vector< std::string > NPTUpdaterGPU::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("npt_timestep");
	list.push_back("npt_temperature");
	list.push_back("npt_pressure");
	list.push_back("npt_volume");
	list.push_back("npt_conserved");
	return list;
	}
	
Scalar NPTUpdaterGPU::getLogValue(const std::string& quantity)
	{
	if (quantity == string("npt_timestep"))
		{
		  return m_timestep;
		}
	else if (quantity == string("npt_temperature"))
		{
		  return computeTemperature();
		}
	else if (quantity == string("npt_pressure"))
	        {
	          return computePressure();
	        }
	else if (quantity == string("npt_volume"))
	        {
		  return m_V;
		}
	else if (quantity == string("npt_conserved"))
	        {
		  return 0.0; // not implemented yet!
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for NPTUpdater" << endl;
		throw runtime_error("Error getting log value");
		}
	}	


NPTUpdaterGPU::~NPTUpdaterGPU()
	{
	freeNPTData();
	}

/*! \param block_size block size to allocate data for
*/
void NPTUpdaterGPU::allocateNPTData(int block_size)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	int local_num;

	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		d_npt_data[cur_gpu].block_size = block_size;
		d_npt_data[cur_gpu].NBlocks = m_pdata->getLocalNum(cur_gpu) / block_size + 1;
		
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].partial_Ksum), d_npt_data[cur_gpu].NBlocks * sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].Ksum), sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].partial_Psum), d_npt_data[cur_gpu].NBlocks * sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].Psum), sizeof(float)));
		local_num = d_pdata[cur_gpu].local_num;
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].virial), local_num * sizeof(float)));

		}
	m_pdata->release();

	}

void NPTUpdaterGPU::freeNPTData()
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].partial_Ksum));
		d_npt_data[cur_gpu].partial_Ksum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].Ksum));
		d_npt_data[cur_gpu].Ksum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].partial_Psum));
		d_npt_data[cur_gpu].partial_Psum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].Psum));
		d_npt_data[cur_gpu].Psum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].virial));
		d_npt_data[cur_gpu].virial = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void *)m_d_virial_data_ptrs[cur_gpu]));
		}
	}

/*! \param fc ForceCompute to add	
     also add virial compute
*/
void NPTUpdaterGPU::addForceCompute(boost::shared_ptr<ForceCompute> fc)
       {
	 Integrator::addForceCompute(fc);
	 // add stuff for virials
         #ifdef USE_CUDA
	 const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	 if (!exec_conf.gpu.empty())
	        {
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			// reinitialize the memory on the device
		
			// fill out the memory on the host
			// this only needs to be done once since the output of acquireGPU is
			// guaranteed not to change later
			float *h_virial_data_ptrs[32];
			for (int i = 0; i < 32; i++)
				h_virial_data_ptrs[i] = NULL;
			
			for (unsigned int i = 0; i < m_forces.size(); i++)
				h_virial_data_ptrs[i] = m_forces[i]->acquireGPU()[cur_gpu].d_data.virial;
			
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, (void*)m_d_virial_data_ptrs[cur_gpu], (void*)h_virial_data_ptrs, sizeof(float*)*32, cudaMemcpyHostToDevice));
			}
		}
	#endif

       }

/*! Call removeForceComputes() to completely wipe out the list of force computes
	that the integrator uses to sum forces.
	Removes virial compute.
*/
void NPTUpdaterGPU::removeForceComputes()
       {
	 
	 #ifdef USE_CUDA
	 
	 const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	 if (!exec_conf.gpu.empty())
	        {
		  for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	                 {
			   exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		 
			   // reinitialize the memory on the device
			   float *h_virial_data_ptrs[32];
			   for (int i = 0; i < 32; i++)
			         h_virial_data_ptrs[i] = NULL;
			   
			   exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, (void*)m_d_virial_data_ptrs[cur_gpu], (void*)h_virial_data_ptrs, sizeof(float*)*32, cudaMemcpyHostToDevice));
			 }
		}
	
         #endif
	 Integrator::removeForceComputes();
       }


/*! \param timestep Current time step of the simulation
*/
void NPTUpdaterGPU::update(unsigned int timestep)
	{
	assert(m_pdata);
	int N = m_pdata->getN();
	m_timestep = timestep;
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeAccelerationsGPU(timestep, "NPT", true);
		m_curr_T = computeTemperature();  // Compute temperature and pressure for the first time step
		m_curr_P = computePressure();
		//cout << "m_curr_T = " << m_curr_T << endl;
		//cout << "m_curr_P = " << m_curr_P << endl;
		}

	if (m_prof) m_prof->push(exec_conf, "NPT");
		
	// access the particle data arrays
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();

	if (m_prof) m_prof->push(exec_conf, "Half-step 1");
		
	// advance thermostat(m_Xi) half a time step

	m_Xi += (1.0f/2.0f)/(m_tau*m_tau)*(m_curr_T/m_T - 1.0f)*m_deltaT;

	// advance barostat (m_Eta) half time step

	m_Eta += (1.0f/2.0f)/(m_tauP*m_tauP)*m_V/(N*m_T)*(m_curr_P - m_P)*m_deltaT;

		
	// advance volume

	//cout << "m_Eta = " << m_Eta << endl;
	//cout << "m_Xi = " << m_Xi << endl;
	//cout << "m_deltaT = " << m_deltaT << endl;	



	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->callAsync(bind(npt_pre_step, &d_pdata[cur_gpu], &box, &d_npt_data[cur_gpu], m_Xi, m_Eta, m_deltaT));
		}
	exec_conf.syncAll();
	
	if (m_prof) m_prof->pop(exec_conf, 36*m_pdata->getN(), 80 * m_pdata->getN());

	m_V *= exp(3.0f*m_Eta*m_deltaT);

	float box_len_scale = exp(m_Eta*m_deltaT);
	m_Lx *= box_len_scale;
	m_Ly *= box_len_scale;
	m_Lz *= box_len_scale;

	//cout << "m_Lx = " << m_Lx << endl;
	//cout << "m_Ly = " << m_Ly << endl;
	//cout << "m_Lz = " << m_Lz << endl;

	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();

	// rescale simulation box

	m_pdata->setBox(BoxDim(m_Lx, m_Ly, m_Lz));
	
	// communicate the updated positions among the GPUs
	m_pdata->communicatePosition();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof) m_prof->pop(exec_conf);
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerationsGPU(timestep+1, "NPT.GPU", false);
	// compute temperature for the next half time step
	m_curr_T = computeTemperature();
	// compute pressure for the next half time step
	m_curr_P = computePressure();
	
	//cout << "m_curr_T = " << m_curr_T << endl;
	//cout << "m_curr_P = " << m_curr_P << endl;

	if (m_prof) m_prof->push(exec_conf, "Half-step 2");

	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	  exec_conf.gpu[cur_gpu]->callAsync(bind(npt_step, &d_pdata[cur_gpu], &d_npt_data[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size(), m_Xi, m_Eta, m_deltaT));
	exec_conf.syncAll();
		
	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		m_prof->pop(exec_conf, 15 * m_pdata->getN(), m_pdata->getN() * 16 * (3 + m_forces.size()));
		m_prof->pop();
		}
	// Update m_Eta

	m_Eta += (1.0f/2.0f)/(m_tauP*m_tauP)*m_V/(N*m_T)*(m_curr_P - m_P)*m_deltaT;

	// Update m_Xi

	m_Xi += (1.0f/2.0f)/(m_tau*m_tau)*(m_curr_T/m_T - 1.0f)*m_deltaT;

	//cout << "m_Eta = " << m_Eta << endl;
	//cout << "m_Xi = " << m_Xi << endl;

	}

float NPTUpdaterGPU::computeTemperature()
        {
	  const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	  vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	  float g = 3.0f*m_pdata->getN();
	  
	  if (m_prof)
	    {
	      m_prof->push(exec_conf, "NPT");
	      m_prof->push(exec_conf, "Reducing Ksum");
	    }
	  
	  for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	    {
	      exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
	      exec_conf.gpu[cur_gpu]->callAsync(bind(npt_temperature, &d_pdata[cur_gpu], &d_npt_data[cur_gpu]));
	    }
	  exec_conf.syncAll();

	  // reduce the Ksum values on the GPU
	  for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	    {
	      exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
	      exec_conf.gpu[cur_gpu]->callAsync(bind(npt_reduce_ksum, &d_npt_data[cur_gpu]));
	    }
	  exec_conf.syncAll();
	  
	  // copy the values from the GPU to the CPU and complete the sum
	  float Ksum_total = 0.0f;
	  for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	    {
	      float Ksum_tmp;
	      exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, &Ksum_tmp, d_npt_data[cur_gpu].Ksum, sizeof(float), cudaMemcpyDeviceToHost));
	      Ksum_total += Ksum_tmp;
	    }

	  m_pdata->release();
	  
	  if (m_prof) m_prof->pop(exec_conf);
	  
	  return Ksum_total / g;
	}

float NPTUpdaterGPU::computePressure()
	{
	  if (m_prof)
		m_prof->push("Pressure");
	
	assert(m_pdata);
	
	
	if (m_prof)
		m_prof->push("Compute");

	// Number of particles
	unsigned int N = m_pdata->getN();


	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
        if (exec_conf.gpu.empty())
                {
                cerr << endl << "***Error! NPT computePressure() asked to compute GPU virial but there is no GPU in the execution configuration" <<
endl << endl;
                throw runtime_error("Error computing virials");
                }

	if (m_prof)
	  {
	    m_prof->push(exec_conf, "Sum virial");
	  }
	
	// acquire the particle data on the GPU and add the forces into the acceleration
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();

	// sum up all the forces
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	  {
	    exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
	    exec_conf.gpu[cur_gpu]->callAsync(bind(integrator_sum_virials, &d_pdata[cur_gpu], m_d_virial_data_ptrs[cur_gpu], (int)m_forces.size(),&d_npt_data[cur_gpu]));
	  }

	exec_conf.syncAll();

	// done
	m_pdata->release();

	 for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	    {
	      exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
	      exec_conf.gpu[cur_gpu]->callAsync(bind(npt_pressure, &d_pdata[cur_gpu], &d_npt_data[cur_gpu]));
	    }
	  exec_conf.syncAll();


	// reduce the Psum values on the GPU
	  for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	    {
	      exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
	      exec_conf.gpu[cur_gpu]->callAsync(bind(npt_reduce_psum, &d_npt_data[cur_gpu]));
	    }
	  exec_conf.syncAll();
	  
	  // copy the values from the GPU to the CPU and complete the sum
	  float Wsum_total = 0.0f;
	  for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	    {
	      float Wsum_tmp;
	      exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, &Wsum_tmp, d_npt_data[cur_gpu].Psum, sizeof(float), cudaMemcpyDeviceToHost));
	      Wsum_total += Wsum_tmp;
	    }

	  //cout << "Wsum_total = " << Wsum_total << endl;
        return (N * m_curr_T + Wsum_total)/m_V; 

	}
	
void export_NPTUpdaterGPU()
	{
	class_<NPTUpdaterGPU, boost::shared_ptr<NPTUpdaterGPU>, bases<NPTUpdater>, boost::noncopyable>
	  ("NPTUpdaterGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, Scalar, Scalar, Scalar >())
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
