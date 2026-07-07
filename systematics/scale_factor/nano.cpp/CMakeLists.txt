cmake_minimum_required(VERSION 3.16)
project(NanoAODToolsCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ROOT REQUIRED COMPONENTS Core MathCore Physics RIO Tree TreePlayer)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_testing()

if(NANO_YAML_CPP_LIBRARY)
  if(NOT EXISTS "${NANO_YAML_CPP_LIBRARY}")
    message(FATAL_ERROR "NANO_YAML_CPP_LIBRARY does not exist: ${NANO_YAML_CPP_LIBRARY}")
  endif()
  if(NOT NANO_YAML_CPP_INCLUDE_DIR OR NOT EXISTS "${NANO_YAML_CPP_INCLUDE_DIR}/yaml-cpp/yaml.h")
    message(FATAL_ERROR "NANO_YAML_CPP_INCLUDE_DIR does not contain yaml-cpp/yaml.h: ${NANO_YAML_CPP_INCLUDE_DIR}")
  endif()
  message(STATUS "Using explicit yaml-cpp include: ${NANO_YAML_CPP_INCLUDE_DIR}")
  message(STATUS "Using explicit yaml-cpp library: ${NANO_YAML_CPP_LIBRARY}")
  add_library(nano_yaml_cpp UNKNOWN IMPORTED)
  set_target_properties(nano_yaml_cpp PROPERTIES IMPORTED_LOCATION "${NANO_YAML_CPP_LIBRARY}")
  set_target_properties(nano_yaml_cpp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NANO_YAML_CPP_INCLUDE_DIR}")
  set(NANO_YAML_CPP_TARGET nano_yaml_cpp)
else()
  find_package(yaml-cpp REQUIRED)
  if(TARGET yaml-cpp::yaml-cpp)
    set(NANO_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
  elseif(TARGET yaml-cpp)
    set(NANO_YAML_CPP_TARGET yaml-cpp)
  else()
    set(NANO_YAML_CPP_TARGET yaml-cpp)
  endif()
endif()

set(SKBUILD ON CACHE BOOL "Disable CMSJMECalculators python/dictionary build" FORCE)
add_subdirectory(external/CMSJMECalculators EXCLUDE_FROM_ALL)
unset(SKBUILD CACHE)

add_library(nanoaodtools_cpp
  src/core/BranchSchema.cpp
  src/io/NanoReader.cpp
  src/io/RootOutputFile.cpp
  src/core/Event.cpp
  src/core/Collection.cpp
  src/core/OutputModel.cpp
  src/core/Helpers.cpp
  src/helpers/FatjetGenMatching.cpp
  src/helpers/JetMETCorrector.cpp
  src/helpers/JmeVariation.cpp
  src/helpers/PuWeightProducer.cpp
  src/helpers/TopPtWeightProducer.cpp
  src/producers/HeavyFlavBaseProducer.cpp
  src/producers/HeavyFlavMinimalProducer.cpp
  src/producers/HeavyFlavMuonSampleProducer.cpp
  src/producers/HeavyFlavScoutingMuonSampleProducer.cpp
)

target_include_directories(nanoaodtools_cpp PUBLIC include)
target_link_libraries(nanoaodtools_cpp PUBLIC CMSJMECalculators ROOT::Core ROOT::MathCore ROOT::Physics ROOT::RIO ROOT::Tree ROOT::TreePlayer ${NANO_YAML_CPP_TARGET})

add_executable(nano_run app/nano_run.cpp)
target_link_libraries(nano_run PRIVATE nanoaodtools_cpp ${NANO_YAML_CPP_TARGET})

add_executable(nano_make_condor app/nano_make_condor.cpp)
target_link_libraries(nano_make_condor PRIVATE nanoaodtools_cpp ${NANO_YAML_CPP_TARGET})

add_executable(nano_merge app/nano_merge.cpp)
target_link_libraries(nano_merge PRIVATE nanoaodtools_cpp ${NANO_YAML_CPP_TARGET})

add_test(
  NAME muon_validation
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/muon_validation_test.py --source-dir ${CMAKE_SOURCE_DIR} --build-dir ${CMAKE_BINARY_DIR} --group muon --max-input-events 5000
)
set_tests_properties(muon_validation PROPERTIES TIMEOUT 7200 LABELS "muon;reference;systematics")
