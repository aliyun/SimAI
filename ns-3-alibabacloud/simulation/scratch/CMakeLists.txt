set(target_prefix scratch_)

# set(ASTRA_PATH /workspace/astra-sim/)
# set(ASTRA_LIB_PATH "/workspace/astra-sim/extern/network_backend/ns3-interface/simulation/scratch")
# message(STATUS "The value of MY_VARIABLE is: ${ASTRA_PATH}")
# include_directories(${ASTRA_PATH})
# link_libraries("${ASTRA_LIB_PATH}/libAstraSim.a")

# link astra-sim
set(ASTRA_PATH "../src/applications")
include_directories(${ASTRA_PATH})


function(create_scratch source_files)
  # Return early if no sources in the subdirectory
  list(LENGTH source_files number_sources)
  if(number_sources EQUAL 0)
    return()
  endif()

  # If the scratch has more than a source file, we need to find the source with
  # the main function
  set(scratch_src)
  foreach(source_file ${source_files})
    file(READ ${source_file} source_file_contents)
    string(REGEX MATCHALL "main[(| (]" main_position "${source_file_contents}")
    if(CMAKE_MATCH_0)
      set(scratch_src ${source_file})
    endif()
  endforeach()

  if(NOT scratch_src)
    return()
  endif()

  # Get parent directory name
  get_filename_component(scratch_dirname ${scratch_src} DIRECTORY)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" scratch_dirname
                 "${scratch_dirname}"
  )
  string(REPLACE "/" "_" scratch_dirname "${scratch_dirname}")

  # Get source name
  get_filename_component(scratch_name ${scratch_src} NAME_WE)

  set(target_prefix scratch_)
  if(scratch_dirname)
    # Join the names together if dirname is not the scratch folder
    set(target_prefix scratch${scratch_dirname}_)
  endif()

  # Get source absolute path and transform into relative path
  get_filename_component(scratch_src ${scratch_src} ABSOLUTE)
  get_filename_component(scratch_absolute_directory ${scratch_src} DIRECTORY)
  string(REPLACE "${PROJECT_SOURCE_DIR}" "${CMAKE_OUTPUT_DIRECTORY}"
                 scratch_directory ${scratch_absolute_directory}
  )
  add_executable(${target_prefix}${scratch_name} "${source_files}")
  if(${NS3_STATIC})
    target_link_libraries(
      ${target_prefix}${scratch_name} ${LIB_AS_NEEDED_PRE_STATIC}
      ${lib-ns3-static}
      ${ASTRA_LIB_PATH}
    )
  else()
    target_link_libraries(
      ${target_prefix}${scratch_name} "${ns3-libs}" "${ns3-contrib-libs}"
      "${ns3-external-libs}"
    )
  endif()
  set_runtime_outputdirectory(
    ${scratch_name} ${scratch_directory}/ ${target_prefix}
  )
endfunction()

# Scan *.cc files in ns-3-dev/scratch and build a target for each
file(GLOB single_source_file_scratches CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
foreach(scratch_src ${single_source_file_scratches})
  create_scratch(${scratch_src})
endforeach()

# Scan *.cc files in ns-3-dev/scratch subdirectories and build a target for each
# subdirectory
file(
  GLOB_RECURSE scratch_subdirectories
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES true
  ${CMAKE_CURRENT_SOURCE_DIR}/**
)
# Filter out files
foreach(entry ${scratch_subdirectories})
  if(NOT (IS_DIRECTORY ${entry}))
    list(REMOVE_ITEM scratch_subdirectories ${entry})
  endif()
endforeach()

foreach(subdir ${scratch_subdirectories})
  if(EXISTS ${subdir}/CMakeLists.txt)
    # If the subdirectory contains a CMakeLists.txt file
    # we let the CMake file manage the source files
    #
    # Use this if you want to link to external libraries
    # without creating a module
    add_subdirectory(${subdir})
  else()
    # Otherwise we pick all the files in the subdirectory
    # and create a scratch for them automatically
    file(GLOB scratch_sources CONFIGURE_DEPENDS ${subdir}/*.cc)
    create_scratch("${scratch_sources}")
  endif()
endforeach()

