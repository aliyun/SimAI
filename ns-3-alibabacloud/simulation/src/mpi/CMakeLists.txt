set(example_as_test_suite)
if(${ENABLE_EXAMPLES})
  set(example_as_test_suite
      test/mpi-test-suite.cc
  )
endif()

if(${ENABLE_MTP})
  build_lib(
    LIBNAME mpi
    SOURCE_FILES
      model/distributed-simulator-impl.cc
      model/granted-time-window-mpi-interface.cc
      model/hybrid-simulator-impl.cc
      model/mpi-interface.cc
      model/mpi-receiver.cc
      model/null-message-mpi-interface.cc
      model/null-message-simulator-impl.cc
      model/parallel-communication-interface.h
      model/remote-channel-bundle-manager.cc
      model/remote-channel-bundle.cc
    HEADER_FILES
      model/mpi-interface.h
      model/mpi-receiver.h
      model/parallel-communication-interface.h
    LIBRARIES_TO_LINK
      ${libcore}
      ${libnetwork}
      ${libmtp}
      ${MPI_CXX_LIBRARIES}
    TEST_SOURCES ${example_as_test_suite}
  )
else()
  build_lib(
    LIBNAME mpi
    SOURCE_FILES
      model/distributed-simulator-impl.cc
      model/granted-time-window-mpi-interface.cc
      model/mpi-interface.cc
      model/mpi-receiver.cc
      model/null-message-mpi-interface.cc
      model/null-message-simulator-impl.cc
      model/parallel-communication-interface.h
      model/remote-channel-bundle-manager.cc
      model/remote-channel-bundle.cc
    HEADER_FILES
      model/mpi-interface.h
      model/mpi-receiver.h
      model/parallel-communication-interface.h
    LIBRARIES_TO_LINK
      ${libcore}
      ${libnetwork}
      ${MPI_CXX_LIBRARIES}
    TEST_SOURCES ${example_as_test_suite}
  )
endif()
