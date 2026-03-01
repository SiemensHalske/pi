cmake_minimum_required(VERSION 3.26)

project(PowderCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF)

option(POWDERCPP_ENABLE_OPENMP "Enable OpenMP parallel backend" ON)
option(POWDERCPP_ENABLE_BENCHMARKS "Build benchmark targets" ON)
option(POWDERCPP_ENABLE_TESTS "Build test targets" ON)
option(POWDERCPP_ENABLE_LIBNUMA "Enable libnuma integration if available" ON)

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo;Release;ReleaseLTO;ASanUBSan" CACHE STRING "Build configs" FORCE)
else()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release" "ReleaseLTO" "ASanUBSan")
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(DetectSIMD)

add_library(powder_build_flags INTERFACE)
powder_apply_strict_warnings(powder_build_flags)
powder_apply_profile_flags(powder_build_flags)
powder_apply_simd_compile_definitions(powder_build_flags)

target_include_directories(powder_build_flags INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(powder_build_flags INTERFACE
  POWDERCPP_BUILD_CONFIG="$<CONFIG>"
)

if(POWDERCPP_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
endif()

if(POWDERCPP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(powder_build_flags INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(powder_build_flags INTERFACE POWDERCPP_WITH_OPENMP=1)
else()
  target_compile_definitions(powder_build_flags INTERFACE POWDERCPP_WITH_OPENMP=0)
endif()

if(POWDERCPP_ENABLE_LIBNUMA)
  find_library(NUMA_LIBRARY numa)
endif()

if(POWDERCPP_ENABLE_LIBNUMA AND NUMA_LIBRARY)
  target_link_libraries(powder_build_flags INTERFACE ${NUMA_LIBRARY})
  target_compile_definitions(powder_build_flags INTERFACE POWDERCPP_WITH_LIBNUMA=1)
else()
  target_compile_definitions(powder_build_flags INTERFACE POWDERCPP_WITH_LIBNUMA=0)
endif()

add_library(engine_core STATIC
  src/core/BuildInfo.cpp
  src/core/CpuFeatures.cpp
  src/core/DeterministicBootstrap.cpp
  src/core/Memory.cpp
  src/core/KernelRegistry.cpp
  src/core/RuntimePlacement.cpp
)
target_link_libraries(engine_core PUBLIC powder_build_flags)

add_library(engine_sim STATIC
  src/sim/SimStub.cpp
  src/sim/WorldState.cpp
  src/sim/MaterialDB.cpp
  src/sim/SimConfig.cpp
  src/sim/SubstepScheduler.cpp
  src/sim/Checkpoint.cpp
  src/sim/FluidPDE.cpp
  src/sim/ThermoChemStruct.cpp
  src/sim/CAGrid.cpp
  src/sim/Granular.cpp
  src/sim/DEM.cpp
  src/sim/SpatialHashDEM.cpp
  src/sim/ParticleGridCoupling.cpp
)
target_link_libraries(engine_sim PUBLIC powder_build_flags engine_core)

add_library(engine_render STATIC
  src/render/RenderStub.cpp
)
target_link_libraries(engine_render PUBLIC powder_build_flags engine_core)

add_library(engine_script STATIC
  src/script/ScriptStub.cpp
)
target_link_libraries(engine_script PUBLIC powder_build_flags engine_core)

add_library(engine_app STATIC
  src/app/App.cpp
  src/app/Cli.cpp
)
target_link_libraries(engine_app PUBLIC powder_build_flags engine_core engine_sim engine_render engine_script)

add_executable(engine_app_main
  src/main.cpp
)
target_link_libraries(engine_app_main PRIVATE engine_app)

if(POWDERCPP_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(POWDERCPP_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
