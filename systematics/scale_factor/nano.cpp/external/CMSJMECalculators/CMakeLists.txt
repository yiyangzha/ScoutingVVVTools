cmake_minimum_required(VERSION 3.11)

if(NOT DEFINED CMSJMECALCULATORS_VERSION)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" CMSJMECALCULATORS_VERSION)
endif()
string(REPLACE "." ";" VERSION_SPLIT ${CMSJMECALCULATORS_VERSION})
list(GET VERSION_SPLIT 0 SPLIT_VERSION_MAJOR)
list(GET VERSION_SPLIT 1 SPLIT_VERSION_MINOR)

project(CMSJMECalculators VERSION ${SPLIT_VERSION_MAJOR}.${SPLIT_VERSION_MINOR} LANGUAGES CXX)

if(SKBUILD) # add python package name, for scikit-build
  set(PKG_LIB_INSTALL "CMSJMECalculators/lib")
  set(PKG_HDR_INSTALL "CMSJMECalculators/include")
  set(PKG_CMK_INSTALL "CMSJMECalculators/cmake")
else()
  set(PKG_LIB_INSTALL "lib")
  set(PKG_HDR_INSTALL "include")
  set(PKG_CMK_INSTALL "cmake")
endif()

find_package(ROOT REQUIRED COMPONENTS MathCore Hist ROOTVecOps Physics)

find_package(correctionlib REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
find_package(ZLIB)

install(DIRECTORY interface/ DESTINATION ${PKG_HDR_INSTALL})

add_library(CMSJMECalculators SHARED src/FatJetVariationsCalculator.cc src/JetMETVariationsCalculatorBase.cc src/JetVariationsCalculator.cc src/Type1METVariationsCalculator.cc src/EGammaVariationsCalculator.cc src/TauVariationsCalculator.cc src/MuonVariationsCalculator.cc)
target_include_directories(CMSJMECalculators PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface>
    $<INSTALL_INTERFACE:${PKG_HDR_INSTALL}>)
target_link_libraries(CMSJMECalculators correctionlib ROOT::Hist ROOT::ROOTVecOps ROOT::Physics)
install(TARGETS CMSJMECalculators EXPORT CMSJMECalculators-targets LIBRARY DESTINATION ${PKG_LIB_INSTALL})

if(NOT SKBUILD) # also build the dictionary and install python modules
  set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/interface")
  REFLEX_GENERATE_DICTIONARY(G__CMSJMECalculators ${CMAKE_CURRENT_SOURCE_DIR}/src/classes.h
    SELECTION ${CMAKE_CURRENT_SOURCE_DIR}/src/classes_def.xml
    DEPENDS CMSJMECalculators
    )
  add_library(CMSJMECalculatorsDict SHARED G__CMSJMECalculators.cxx)
  target_link_libraries(CMSJMECalculatorsDict CMSJMECalculators)
  install(TARGETS CMSJMECalculatorsDict EXPORT CMSJMECalculators-targets LIBRARY DESTINATION ${PKG_LIB_INSTALL})

  find_package(Python)
  if(PYTHON_FOUND)
    install(DIRECTORY "python/CMSJMECalculators" DESTINATION "${PKG_LIB_INSTALL}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
  endif()
endif()

install(EXPORT CMSJMECalculators-targets FILE CMSJMECalculators-targets.cmake DESTINATION ${PKG_CMK_INSTALL})
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/CMSJMECalculatorsConfig.cmake"
  INSTALL_DESTINATION "${PKG_CMK_INSTALL}"
  NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CMSJMECalculatorsConfigVersion.cmake"
  VERSION "${CMSJMECalculators_VERSION_MAJOR}.${CMSJMECalculators_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMSJMECalculatorsConfig.cmake DESTINATION ${PKG_CMK_INSTALL})
