function(ns3_export_file src dst)
    get_filename_component(dst_dir "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${dst_dir}")

    get_filename_component(dst_abs "${dst}" ABSOLUTE)

    if(EXISTS "${dst}" OR IS_SYMLINK "${dst_abs}")
        file(REMOVE "${dst}")
    endif()

    if(NS3_USE_RELATIVE_PATHS_SYMLINKS)
        file(RELATIVE_PATH src_path "${dst_dir}" "${src}")
        file(TO_CMAKE_PATH "${src_path}" src_path)
    else()
        set(src_path "${src}")
    endif()

    if(NS3_EXPORT_HEADERS_AS_STUBS)
        if("${src_path}" MATCHES "\\.(h|hh)$")
            file(GENERATE OUTPUT "${dst}" CONTENT "#include \"${src_path}\"\n")
        else()
            configure_file("${src}" "${dst}" COPYONLY)
        endif()
    else()
        file(CREATE_LINK "${src_path}" "${dst}" SYMBOLIC)
    endif()
endfunction()

file(GLOB_RECURSE ASTRA_SIM_FILES CONFIGURE_DEPENDS
        "${ASTRA_SIM_DIR}/*.cc"
        "${ASTRA_SIM_DIR}/*.hh"
        "${ASTRA_SIM_DIR}/*.h"
)
list(FILTER ASTRA_SIM_FILES EXCLUDE REGEX "/network_frontend/")
list(FILTER ASTRA_SIM_FILES EXCLUDE REGEX "/build/")

foreach(src_file IN LISTS ASTRA_SIM_FILES)
    string(REPLACE "${ASTRA_SIM_DIR}/" "astra-sim/" relative_path "${src_file}")
    ns3_export_file("${src_file}" "${NS3_SRC_DIR}/src/applications/${relative_path}")
endforeach()

file(GLOB NS3_FRONTEND_FILES CONFIGURE_DEPENDS
        "${ASTRA_SIM_DIR}/network_frontend/ns3/*.cc"
        "${ASTRA_SIM_DIR}/network_frontend/ns3/*.hh"
        "${ASTRA_SIM_DIR}/network_frontend/ns3/*.h"
)

foreach(src_file IN LISTS NS3_FRONTEND_FILES)
    get_filename_component(filename "${src_file}" NAME)
    ns3_export_file("${src_file}" "${NS3_SRC_DIR}/scratch/${filename}")
endforeach()

string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type_lower)
if(_build_type_lower STREQUAL "release")
    if(NOT DEFINED NS3_ASSERT)
        set(NS3_ASSERT OFF CACHE BOOL "Override NS3_ASSERT" FORCE)
    endif()
    if(NOT DEFINED NS3_LOG)
        set(NS3_LOG OFF CACHE BOOL "Override NS3_LOG" FORCE)
    endif()
endif()

if(NOT DEFINED NS3_WARNINGS_AS_ERRORS)
    set(NS3_WARNINGS_AS_ERRORS OFF CACHE BOOL "Override NS3_WARNINGS_AS_ERRORS" FORCE)
endif()

if(NOT DEFINED NS3_MTP)
    set(NS3_MTP ON CACHE BOOL "Override NS3_MTP" FORCE)
endif()

if(DEFINED NS3_NATIVE_OPTIMIZATIONS AND NS3_NATIVE_OPTIMIZATIONS)
    add_compile_options(-march=native)
endif()

get_filename_component(NS3_COPIED_PREFIX_1 "${NS3_SRC_DIR}/src/applications/astra-sim" REALPATH)
get_filename_component(NS3_ORIGINAL_PREFIX_1 "${ASTRA_SIM_DIR}" REALPATH)
get_filename_component(NS3_COPIED_PREFIX_2 "${NS3_SRC_DIR}/scratch" REALPATH)
get_filename_component(NS3_ORIGINAL_PREFIX_2 "${ASTRA_SIM_DIR}/network_frontend/ns3" REALPATH)

add_compile_options(
        -ffile-prefix-map=${NS3_COPIED_PREFIX_1}=${NS3_ORIGINAL_PREFIX_1}
        -ffile-prefix-map=${NS3_COPIED_PREFIX_2}=${NS3_ORIGINAL_PREFIX_2}
)

add_subdirectory("${NS3_SRC_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ns3")
