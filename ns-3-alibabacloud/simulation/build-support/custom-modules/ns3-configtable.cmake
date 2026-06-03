# cmake-format: off
#
# A sample of what Waf produced

# ---- Summary of optional ns-3 features:
# Build profile                 : debug
# Build directory               :
# BRITE Integration             : not enabled (BRITE not enabled (see option --with-brite))
# DES Metrics event collection  : not enabled (defaults to disabled)
# DPDK NetDevice                : not enabled (libdpdk not found, $RTE_SDK and/or $RTE_TARGET environment variable not set or incorrect)
# Emulation FdNetDevice         : enabled
# Examples                      : not enabled (defaults to disabled)
# File descriptor NetDevice     : enabled
# GNU Scientific Library (GSL)  : enabled
# GtkConfigStore                : enabled
# MPI Support                   : not enabled (option --enable-mpi not selected)
# ns-3 Click Integration        : not enabled (nsclick not enabled (see option --with-nsclick))
# ns-3 OpenFlow Integration     : not enabled (OpenFlow not enabled (see option --with-openflow))
# Netmap emulation FdNetDevice  : not enabled (needs net/netmap_user.h)
# PyViz visualizer              : not enabled (Missing python modules: pygraphviz, gi.repository.GooCanvas)
# Python API Scanning Support   : not enabled (castxml too old)
# Python Bindings               : enabled
# Real Time Simulator           : enabled
# SQLite stats support          : enabled
# Tap Bridge                    : enabled
# Tap FdNetDevice               : enabled
# Tests                         : not enabled (defaults to disabled)
# Use sudo to set suid bit      : not enabled (option --enable-sudo not selected)
# XmlIo                         : enabled
#
#
# And now a sample after build
#
# Modules built:
# antenna                   aodv                      applications
# bridge                    buildings                 config-store
# core                      csma                      csma-layout
# dsdv                      dsr                       energy
# fd-net-device             flow-monitor              internet
# internet-apps             lr-wpan                   lte
# mesh                      mobility                  netanim
# network                   nix-vector-routing        olsr
# point-to-point            point-to-point-layout     propagation
# sixlowpan                 spectrum                  stats
# tap-bridge                test (no Python)          topology-read
# traffic-control           uan                       virtual-net-device
# wave                      wifi                      wimax
#
# Modules not built (see ns-3 tutorial for explanation):
# brite                     click                     dpdk-net-device
# mpi                       openflow                  visualizer
#
# cmake-format: on

# Now the CMake part

macro(check_on_or_off user_config_switch confirmation_flag)
  if(${user_config_switch})
    if(${confirmation_flag})
      string(APPEND out "${Green}ON${ColourReset}\n")
    else()
      string(APPEND out "${Red}OFF (missing dependency)${ColourReset}\n")
    endif()
  else()
    string(APPEND out "OFF (not requested)\n")
  endif()
endmacro()

function(print_formatted_table_with_modules table_name modules output)
  set(temp)
  string(APPEND temp "${table_name}:\n")
  set(count 0) # Variable to count number of columns
  set(width 26) # Variable with column width
  string(REPLACE ";lib" ";" modules_to_print ";${modules}")
  string(SUBSTRING "${modules_to_print}" 1 -1 modules_to_print)
  list(SORT modules_to_print) # Sort for nice output
  set(modules_with_large_names)
  foreach(module ${modules_to_print})
    # Get the size of the module string name
    string(LENGTH ${module} module_name_length)

    # Skip modules with names wider than 26 characters
    if(${module_name_length} GREATER_EQUAL ${width})
      list(APPEND modules_with_large_names ${module})
      continue()
    endif()

    # Calculate trailing spaces to fill the column
    math(EXPR num_trailing_spaces "${width} - ${module_name_length}")

    # Get a string with spaces
    string(RANDOM LENGTH ${num_trailing_spaces} ALPHABET " " trailing_spaces)

    # Append module name and spaces to output
    string(APPEND temp "${module}${trailing_spaces}")
    math(EXPR count "${count} + 1") # Count number of column

    # When counter hits the 3rd column, wrap to the nextline
    if(${count} EQUAL 3)
      string(APPEND temp "\n")
      set(count 0)
    endif()
  endforeach()

  # Print modules with large names one by one
  foreach(module ${modules_with_large_names})
    string(APPEND temp "${module}\n")
  endforeach()
  string(APPEND temp "\n")

  # Save the table outer scope out variable
  set(${output} ${${output}}${temp} PARENT_SCOPE)
endfunction()

macro(write_configtable)
  set(out "---- Summary of optional ns-3 features:\n")
  string(APPEND out "Build profile                 : ${build_profile}\n")
  string(APPEND out
         "Build directory               : ${CMAKE_OUTPUT_DIRECTORY}\n"
  )
  string(APPEND out "BRITE Integration             : ")
  check_on_or_off("ON" "${NS3_BRITE}")

  string(APPEND out "DES Metrics event collection  : ")
  check_on_or_off("${NS3_DES_METRICS}" "${NS3_DES_METRICS}")

  string(APPEND out "DPDK NetDevice                : ")
  check_on_or_off("${NS3_DPDK}" "${ENABLE_DPDKDEVNET}")

  string(APPEND out "Emulation FdNetDevice         : ")
  check_on_or_off("${ENABLE_EMU}" "${ENABLE_EMUNETDEV}")

  string(APPEND out "Examples                      : ")
  check_on_or_off("${ENABLE_EXAMPLES}" "${ENABLE_EXAMPLES}")

  string(APPEND out "File descriptor NetDevice     : ")
  check_on_or_off("ON" "${ENABLE_FDNETDEV}")

  string(APPEND out "GNU Scientific Library (GSL)  : ")
  check_on_or_off("${NS3_GSL}" "${GSL_FOUND}")

  string(APPEND out "GtkConfigStore                : ")
  check_on_or_off("${NS3_GTK3}" "${GTK3_FOUND}")

  string(APPEND out "LibXml2 support               : ")
  check_on_or_off("ON" "${LIBXML2_FOUND}")

  string(APPEND out "MPI Support                   : ")
  check_on_or_off("${NS3_MPI}" "${MPI_FOUND}")

  string(APPEND out "Multithreaded Simulation      : ")
  check_on_or_off("${NS3_MTP}" "ON")

  string(APPEND out "ns-3 Click Integration        : ")
  check_on_or_off("ON" "${NS3_CLICK}")

  string(APPEND out "ns-3 OpenFlow Integration     : ")
  check_on_or_off("ON" "${NS3_OPENFLOW}")

  string(APPEND out "Netmap emulation FdNetDevice  : ")
  check_on_or_off("${ENABLE_EMU}" "${ENABLE_NETMAP_EMU}")

  string(APPEND out "PyViz visualizer              : ")
  check_on_or_off("${NS3_VISUALIZER}" "${ENABLE_VISUALIZER}")

  string(APPEND out "Python API Scanning Support   : ")
  check_on_or_off(
    "${NS3_SCAN_PYTHON_BINDINGS}" "${ENABLE_SCAN_PYTHON_BINDINGS}"
  )

  string(APPEND out "Python Bindings               : ")
  check_on_or_off("${NS3_PYTHON_BINDINGS}" "${ENABLE_PYTHON_BINDINGS}")

  string(APPEND out "Real Time Simulator           : ")
  check_on_or_off("${NS3_REALTIME}" "${ENABLE_REALTIME}")

  string(APPEND out "SQLite support                : ")
  check_on_or_off("${NS3_SQLITE}" "${ENABLE_SQLITE}")

  string(APPEND out "Tap Bridge                    : ")
  check_on_or_off("${ENABLE_TAP}" "${ENABLE_TAP}")

  string(APPEND out "Tap FdNetDevice               : ")
  check_on_or_off("${ENABLE_TAP}" "${ENABLE_TAPNETDEV}")

  string(APPEND out "Tests                         : ")
  check_on_or_off("${ENABLE_TESTS}" "${ENABLE_TESTS}")

  # string(APPEND out "Use sudo to set suid bit      : not enabled (option
  # --enable-sudo not selected) string(APPEND out "XmlIo : enabled
  string(APPEND out "\n\n")

  set(really-enabled-modules ${ns3-libs};${ns3-contrib-libs})
  if(${ENABLE_TESTS})
    list(APPEND really-enabled-modules libtest) # test is an object library and
                                                # is treated differently
  endif()
  if(really-enabled-modules)
    print_formatted_table_with_modules(
      "Modules configured to be built" "${really-enabled-modules}" "out"
    )
    string(APPEND out "\n")
  endif()

  set(disabled-modules)
  foreach(module ${ns3-all-enabled-modules})
    if(NOT (lib${module} IN_LIST really-enabled-modules))
      list(APPEND disabled-modules ${module})
    endif()
  endforeach()

  if(disabled-modules)
    print_formatted_table_with_modules(
      "Modules that cannot be built" "${disabled-modules}" "out"
    )
    string(APPEND out "\n")
  endif()

  file(WRITE ${PROJECT_BINARY_DIR}/ns3config.txt ${out})
  message(STATUS ${out})
endmacro()
