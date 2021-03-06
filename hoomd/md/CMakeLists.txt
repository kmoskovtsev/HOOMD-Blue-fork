# Maintainer: joaander

set(_md_sources module-md.cc
                   ActiveForceCompute.cc
                   BondTablePotential.cc
                   CommunicatorGrid.cc
                   ConstExternalFieldDipoleForceCompute.cc
                   ConstraintEllipsoid.cc
                   ConstraintSphere.cc
                   OneDConstraint.cc
                   Enforce2DUpdater.cc
                   FIREEnergyMinimizer.cc
                   ForceComposite.cc
                   ForceDistanceConstraint.cc
                   HarmonicAngleForceCompute.cc
                   HarmonicDihedralForceCompute.cc
                   HarmonicImproperForceCompute.cc
                   IntegrationMethodTwoStep.cc
                   IntegratorTwoStep.cc
                   MolecularForceCompute.cc
                   NeighborListBinned.cc
                   NeighborList.cc
                   NeighborListStencil.cc
                   NeighborListTree.cc
                   OPLSDihedralForceCompute.cc
                   PPPMForceCompute.cc
                   TableAngleForceCompute.cc
                   TableDihedralForceCompute.cc
                   TablePotential.cc
                   TablePotential2D.cc
                   TempRescaleUpdater.cc
                   TwoStepBD.cc
                   TwoStepBerendsen.cc
                   TwoStepLangevinBase.cc
                   TwoStepLangevin.cc
                   TwoStepNPTMTK.cc
                   TwoStepNVE.cc
                   TwoStepCustomScatter2D.cc
                   TwoStepNVTMTK.cc
                   ZeroMomentumUpdater.cc
		   MuellerPlatheFlow.cc
                   )

set(_md_headers ActiveForceComputeGPU.h
                ActiveForceCompute.h
                AllAnisoPairPotentials.h
                AllBondPotentials.h
                AllExternalPotentials.h
                AllPairPotentials.h
                AllTripletPotentials.h
                AnisoPotentialPairGPU.cuh
                AnisoPotentialPairGPU.h
                AnisoPotentialPair.h
                BondTablePotentialGPU.h
                BondTablePotential.h
                CommunicatorGridGPU.h
                CommunicatorGrid.h
                ConstExternalFieldDipoleForceCompute.h
                ConstraintEllipsoidGPU.h
                ConstraintEllipsoid.h
                ConstraintSphereGPU.h
                ConstraintSphere.h
                Enforce2DUpdaterGPU.h
                Enforce2DUpdater.h
                EvaluatorBondFENE.h
                EvaluatorBondHarmonic.h
                EvaluatorSpecialPairLJ.h
                EvaluatorConstraintEllipsoid.h
                EvaluatorConstraint.h
                EvaluatorConstraintSphere.h
                EvaluatorExternalElectricField.h
                EvaluatorExternalPeriodic.h
                EvaluatorExternalPeriodicCos.h
                EvaluatorPairDipole.h
                EvaluatorPairDPDLJThermo.h
                EvaluatorPairDPDThermo.h
                EvaluatorPairEwald.h
                EvaluatorPairForceShiftedLJ.h
                EvaluatorPairGauss.h
                EvaluatorPairGB.h
                EvaluatorPairLJ.h
                EvaluatorPairMie.h
                EvaluatorPairMoliere.h
                EvaluatorPairMorse.h
                EvaluatorPairReactionField.h
                EvaluatorPairSLJ.h
                EvaluatorPairYukawa.h
                EvaluatorPairZBL.h
                EvaluatorTersoff.h
                EvaluatorWalls.h
                FIREEnergyMinimizerGPU.h
                FIREEnergyMinimizer.h
                ForceCompositeGPU.h
                ForceComposite.h
                ForceDistanceConstraintGPU.h
                ForceDistanceConstraint.h
                HarmonicAngleForceComputeGPU.h
                HarmonicAngleForceCompute.h
                HarmonicDihedralForceComputeGPU.h
                HarmonicDihedralForceCompute.h
                HarmonicImproperForceComputeGPU.h
                HarmonicImproperForceCompute.h
                IntegrationMethodTwoStep.h
                IntegratorTwoStep.h
                MolecularForceCompute.h
                NeighborListBinned.h
                NeighborListGPUBinned.h
                NeighborListGPU.h
                NeighborListGPUStencil.h
                NeighborListGPUTree.h
                NeighborList.h
                NeighborListStencil.h
                NeighborListTree.h
                OPLSDihedralForceComputeGPU.h
                OPLSDihedralForceCompute.h
                PotentialBondGPU.h
		PotentialBondGPU.cuh
                PotentialBond.h
                PotentialExternalGPU.h
                PotentialExternalGPU.cuh
                PotentialExternal.h
                PotentialPairDPDThermoGPU.h
		PotentialPairDPDThermoGPU.cuh
                PotentialPairDPDThermo.h
                PotentialPairGPU.h
		PotentialPairGPU.cuh
                PotentialPair.h
                PotentialTersoffGPU.h
                PotentialTersoff.h
                PPPMForceComputeGPU.h
                PPPMForceCompute.h
                QuaternionMath.h
                TableAngleForceComputeGPU.h
                TableAngleForceCompute.h
                TableDihedralForceComputeGPU.h
                TableDihedralForceCompute.h
                TablePotentialGPU.h
                TablePotential.h
                TablePotential2DGPU.h
                TablePotential2D.h
                TempRescaleUpdater.h
                TwoStepBDGPU.h
                TwoStepBD.h
                TwoStepBerendsenGPU.h
                TwoStepBerendsen.h
                TwoStepLangevinBase.h
                TwoStepLangevinGPU.h
                TwoStepLangevin.h
                TwoStepNPTMTKGPU.h
                TwoStepNPTMTK.h
                TwoStepNVEGPU.h
                TwoStepCustomScatter2DGPU.h
                TwoStepNVE.h
                TwoStepCustomScatter2D.h
                TwoStepNVTMTKGPU.h
                TwoStepNVTMTK.h
                WallData.h
                ZeroMomentumUpdater.h
                )

if (ENABLE_CUDA)
list(APPEND _md_sources ActiveForceComputeGPU.cc
                           BondTablePotentialGPU.cc
                           CommunicatorGridGPU.cc
                           ConstraintEllipsoidGPU.cc
                           ConstraintSphereGPU.cc
                           OneDConstraintGPU.cc
                           Enforce2DUpdaterGPU.cc
                           FIREEnergyMinimizerGPU.cc
                           ForceCompositeGPU.cc
                           ForceDistanceConstraintGPU.cc
                           HarmonicAngleForceComputeGPU.cc
                           HarmonicDihedralForceComputeGPU.cc
                           HarmonicImproperForceComputeGPU.cc
                           NeighborListGPU.cc
                           NeighborListGPUBinned.cc
                           NeighborListGPUStencil.cc
                           NeighborListGPUTree.cc
                           OPLSDihedralForceComputeGPU.cc
                           PPPMForceComputeGPU.cc
                           TableAngleForceComputeGPU.cc
                           TableDihedralForceComputeGPU.cc
                           TablePotentialGPU.cc
                           TablePotential2DGPU.cc
                           TwoStepBDGPU.cc
                           TwoStepBerendsenGPU.cc
                           TwoStepLangevinGPU.cc
                           TwoStepNPTMTKGPU.cc
                           TwoStepNVEGPU.cc
                           TwoStepCustomScatter2DGPU.cc
                           TwoStepNVTMTKGPU.cc
			   MuellerPlatheFlowGPU.cc
                           )
endif()

set(_md_cu_sources ActiveForceComputeGPU.cu
                      AllDriverAnisoPotentialPairGPU.cu
                      AllDriverPotentialBondGPU.cu
                      AllDriverPotentialSpecialPairGPU.cu
                      AllDriverPotentialPairGPU.cu
                      BondTablePotentialGPU.cu
                      CommunicatorGridGPU.cu
                      ConstraintEllipsoidGPU.cu
                      ConstraintSphereGPU.cu
                      OneDConstraintGPU.cu
                      DriverTersoffGPU.cu
                      Enforce2DUpdaterGPU.cu
                      FIREEnergyMinimizerGPU.cu
                      ForceCompositeGPU.cu
                      ForceDistanceConstraintGPU.cu
                      HarmonicAngleForceGPU.cu
                      HarmonicDihedralForceGPU.cu
                      HarmonicImproperForceGPU.cu
                      NeighborListGPUBinned.cu
                      NeighborListGPU.cu
                      NeighborListGPUStencil.cu
                      NeighborListGPUTree.cu
                      OPLSDihedralForceGPU.cu
                      PotentialExternalGPU.cu
                      PPPMForceComputeGPU.cu
                      TableAngleForceGPU.cu
                      TableDihedralForceGPU.cu
                      TablePotentialGPU.cu
                      TablePotential2DGPU.cu
                      TwoStepBDGPU.cu
                      TwoStepBerendsenGPU.cu
                      TwoStepLangevinGPU.cu
                      TwoStepNPTMTKGPU.cu
                      TwoStepNVEGPU.cu
                      TwoStepCustomScatter2DGPU.cu
                      TwoStepNVTMTKGPU.cu
		      MuellerPlatheFlowGPU.cu
                      )

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_md_sources} ${_md_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_md_cu_sources} ${DFFT_CU_SOURCES} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_md SHARED ${_md_sources} ${_CUDA_GENERATED_FILES} ${DFFT_SOURCES})

# link the library to its dependencies
if (CUSOLVER_AVAILABLE)
    if(${CUDA_VERSION} VERSION_LESS 8.0)
        target_link_libraries(_md _hoomd ${HOOMD_COMMON_LIBS} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})
    else()
        # CUDA 8.0 requires that we link in gomp
        target_link_libraries(_md _hoomd ${HOOMD_COMMON_LIBS} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} gomp)
    endif()
else()
    target_link_libraries(_md _hoomd ${HOOMD_COMMON_LIBS})
endif()

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_md PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_md PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_md PROPERTIES PREFIX "" OUTPUT_NAME "_md")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_md PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_md)

# install the library
install(TARGETS _md
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/md
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/md/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          angle.py
          bond.py
          charge.py
          constrain.py
          dihedral.py
          external.py
          force.py
          improper.py
          integrate.py
          nlist.py
          pair.py
          update.py
          wall.py
          special_pair.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/md
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_md ALL DEPENDS ${files})

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/../include/hoomd/md/${file}
        COMMENT    "Copy hoomd/include/hoomd/md/${file}"
    )
ENDMACRO(copy_header)

# install headers in installation target
install(FILES ${_md_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/md
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_md_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_md_headers ALL DEPENDS ${_md_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()
