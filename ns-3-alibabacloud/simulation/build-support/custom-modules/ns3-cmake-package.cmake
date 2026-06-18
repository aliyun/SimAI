# Copyright (c) 2017-2021 Universidade de Brasília
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation;
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA  02111-1307 USA
#
# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>

function(build_required_and_libs_lists module_name visibility libraries
         all_ns3_libraries
)
  set(linked_libs_list)
  set(required_modules_list)
  foreach(lib ${libraries})
    if(${lib} IN_LIST all_ns3_libraries)
      get_target_property(lib_real_name ${lib} OUTPUT_NAME)
      remove_lib_prefix(${lib} required_module_name)
      set(required_modules_list
          "${required_modules_list} ns3-${required_module_name}"
      )
      set(lib_real_name "-l${lib_real_name}")
    else()
      if(IS_ABSOLUTE ${lib})
        set(lib_real_name ${lib})
      else()
        set(lib_real_name "-l${lib}")
      endif()
    endif()
    set(linked_libs_list "${linked_libs_list} ${lib_real_name}")
  endforeach()
  set(pkgconfig_${visibility}_libs ${linked_libs_list} PARENT_SCOPE)
  set(pkgconfig_${visibility}_required ${required_modules_list} PARENT_SCOPE)
endfunction()

function(pkgconfig_module libname)
  # Fetch all libraries that will be linked to module
  get_target_property(all_libs ${libname} LINK_LIBRARIES)

  # Then fetch public libraries
  get_target_property(interface_libs ${libname} INTERFACE_LINK_LIBRARIES)

  # Filter linking flags
  string(REPLACE "${LIB_AS_NEEDED_PRE}" "" all_libs "${all_libs}")
  string(REPLACE "${LIB_AS_NEEDED_POST}" "" all_libs "${all_libs}")
  string(REPLACE "${LIB_AS_NEEDED_PRE}" "" interface_libs "${interface_libs}")
  string(REPLACE "${LIB_AS_NEEDED_POST}" "" interface_libs "${interface_libs}")

  foreach(interface_lib ${interface_libs})
    list(REMOVE_ITEM all_libs ${interface_lib})
  endforeach()
  set(private_libs ${all_libs})

  # Create two lists of publicly and privately linked libraries to this module
  remove_lib_prefix(${libname} module_name)

  # These filter out ns and non-ns libraries into public and private libraries
  # linked against module_name
  get_target_property(pkgconfig_target_lib ${libname} OUTPUT_NAME)

  # pkgconfig_public_libs pkgconfig_public_required
  build_required_and_libs_lists(
    "${module_name}" public "${interface_libs}"
    "${ns3-libs};${ns3-contrib-libs}"
  )

  # pkgconfig_private_libs pkgconfig_private_required
  build_required_and_libs_lists(
    "${module_name}" private "${private_libs}"
    "${ns3-libs};${ns3-contrib-libs}"
  )

  set(pkgconfig_interface_include_directories)
  if(${NS3_REEXPORT_THIRD_PARTY_LIBRARIES})
    get_target_includes(${libname} pkgconfig_interface_include_directories)
    string(REPLACE "-I${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/include" ""
                   pkgconfig_interface_include_directories
                   "${pkgconfig_interface_include_directories}"
    )
    string(REPLACE ";-I" " -I" pkgconfig_interface_include_directories
                   "${pkgconfig_interface_include_directories}"
    )
  endif()

  # Configure pkgconfig file for the module using pkgconfig variables
  set(pkgconfig_file ${CMAKE_BINARY_DIR}/pkgconfig/ns3-${module_name}.pc)
  configure_file(
    ${PROJECT_SOURCE_DIR}/build-support/pkgconfig-template.pc.in
    ${pkgconfig_file} @ONLY
  )

  # Set file to be installed
  install(FILES ${pkgconfig_file} DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endfunction()

function(ns3_cmake_package)
  # Only create configuration to export if there is an module configured to be
  # built
  set(enabled_modules "${ns3-libs};${ns3-contrib-libs}")
  if(enabled_modules STREQUAL ";")
    message(
      STATUS
        "No modules were configured, so we cannot create installation artifacts"
    )
    return()
  endif()

  # CMake does not support '-' separated versions in config packages, so replace
  # them with dots
  string(REPLACE "-" "." ns3_version "${NS3_VER}")

  foreach(library ${ns3-libs}${ns3-contrib-libs})
    pkgconfig_module(${library})
  endforeach()

  install(
    EXPORT ns3ExportTargets
    NAMESPACE ns3::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ns3
    FILE ns3Targets.cmake
  )

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "build-support/Config.cmake.in" "ns3Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ns3
    PATH_VARS CMAKE_INSTALL_LIBDIR
  )

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ns3ConfigVersion.cmake VERSION ${ns3_version}
    COMPATIBILITY ExactVersion
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ns3Config.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/ns3ConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ns3
  )
endfunction()

# You will need administrative privileges to run this
add_custom_target(
  uninstall
  COMMAND
    rm `ls ${CMAKE_INSTALL_FULL_LIBDIR}/libns3*` && rm -R
    ${CMAKE_INSTALL_FULL_LIBDIR}/cmake/ns3 && rm -R
    ${CMAKE_INSTALL_FULL_INCLUDEDIR}/ns3
)
