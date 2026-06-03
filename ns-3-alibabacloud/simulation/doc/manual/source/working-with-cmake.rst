.. include:: replace.txt
.. highlight:: console

.. Section Separators:
   ----
   ****
   ++++
   ====
   ~~~~

.. _Working with CMake:

Working with CMake
------------------

The ns-3 project used Waf build system in the past, but it has moved to
CMake for the ns-3.36 release.

CMake is very verbose and commands can be very long for basic operations.

The wrapper script ``ns3`` hides most of verbosity from CMake and provide a 
Waf-like interface for command-line users. 

.. _CLion : https://www.jetbrains.com/clion/
.. _Visual Studio : https://visualstudio.microsoft.com/
.. _Code : https://code.visualstudio.com/Download
.. _Xcode : https://developer.apple.com/xcode/
.. _CodeBlocks : https://www.codeblocks.org/
.. _Eclipse : https://www.eclipse.org/cdt/
.. _CMake generated : https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html
.. _generator options : https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html

It is the recommended way to work on ns-3, except if you are using an 
IDE that supports projects that can be generated with CMake or CMake projects.

Here is a non-exhaustive list of IDEs that can be used:

* Support CMake projects:
  
  * JetBrains's `CLion`_
  * Microsoft `Visual Studio`_ and Visual Studio `Code`_

* Supported IDEs via `CMake generated`_ projects:
  
  * Apple's `XCode`_ : ``ns3 configure -G Xcode``
  * `CodeBlocks`_ : ``ns3 configure -G "CodeBlocks - Ninja"``
  * `Eclipse`_ CDT4 : ``ns3 configure -G "Eclipse CDT4 - Ninja"`` 

Note: Ninja was used for brevity. 
Both CodeBlocks and Eclipse have additional `generator options`_.

General instructions on how to setup and use IDEs are available
in the Tutorial and will not be detailed here.

Configuring the project
***********************

After getting the code, either cloning the ns-3-dev repository or 
downloading the release tarball, you will need to configure the 
project to work on it.

There are two ways to configure the project: the easiest way 
is using the ``ns3`` script and the other way directly with CMake.

Configuring the project with ns3
++++++++++++++++++++++++++++++++

Navigate to the ns-3-dev directory, then run ``./ns3 configure --help`` to 
print the configuration options:

.. sourcecode:: console

  ~$ cd ns-3-dev
  ~/ns-3-dev$ ./ns3 configure --help
  usage: ns3 configure [-h] [-d {debug,release,optimized}] [-G G]
                     [--cxx-standard CXX_STANDARD] [--enable-asserts]
                     [--disable-asserts] [--enable-examples]
                     [--disable-examples] [--enable-logs]
                     [--disable-logs] [--enable-tests] 
                     [--disable-tests] [--enable-verbose] 
                     [--disable-verbose]
                     ...

  positional arguments:
    configure

  optional arguments:
    -h, --help            show this help message and exit
    -d {debug,release,optimized}, --build-profile {debug,release,optimized}
                          Build profile
    -G G                  CMake generator (e.g.
                          https://cmake.org/cmake/help/latest/manual/cmake-
                          generators.7.html)
    ...

Note: the command output was trimmed to the most used options.

To configure ns-3 in release mode, while enabling examples and tests,
run ``./ns3 configure -d release --enable-examples --enable-tests``.
To check what underlying commands dare being executed, add the 
``--dry-run`` option:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 --dry-run configure -d release --enable-examples --enable-tests
  The following commands would be executed:
  mkdir cmake-cache
  cd cmake-cache; /usr/bin/cmake -DCMAKE_BUILD_TYPE=release -DNS3_NATIVE_OPTIMIZATIONS=OFF -DNS3_EXAMPLES=ON -DNS3_TESTS=ON -G Unix Makefiles .. ; cd ..

Now we run it for real:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 configure -d release --enable-examples --enable-tests
  -- CCache is enabled. Precompiled headers are disabled by default.
  -- The CXX compiler identification is GNU 11.2.0
  -- The C compiler identification is GNU 11.2.0
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Check for working CXX compiler: /usr/bin/c++ - skipped
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  ...
  -- Processing src/wifi
  -- Processing src/wimax
  -- ---- Summary of optional NS-3 features:
  Build profile                 : release
  Build directory               : /mnt/dev/tools/source/ns-3-dev/build
  ...
  Examples                      : ON
  ...
  Tests                         : ON
  Threading Primitives          : ON


  Modules configured to be built:
  antenna                   aodv                      applications              
  bridge                    buildings                 config-store              
  core                      csma                      csma-layout               
  ...
  wifi                      wimax                     

  Modules that cannot be built:
  brite                     click                     openflow                  
  visualizer               


  -- Configuring done
  -- Generating done
  -- Build files have been written to: /mnt/dev/tools/source/ns-3-dev/cmake-cache
  Finished executing the following commands:
  mkdir cmake-cache
  cd cmake-cache; /usr/bin/cmake -DCMAKE_BUILD_TYPE=release -DNS3_NATIVE_OPTIMIZATIONS=OFF -DNS3_EXAMPLES=ON -DNS3_TESTS=ON -G Unix Makefiles .. ; cd ..

Notice that CCache is automatically used (if installed) for your convenience.

The summary with enabled feature shows both the ``release`` build type, along with
enabled examples and tests.

Below is a list of enabled modules and modules that cannot be built.

At the end, notice we print the same commands from ``--dry-run``. This is done
to familiarize Waf users with CMake and how the options names changed.

The mapping of the ns3 build profiles into the CMake build types is the following:

+---------------------------+------------------------------------------------------------------------------------------+
| Equivalent build profiles                                                                                            |
+---------------------------+--------------------------------------------------------+---------------------------------+
| ns3                       | CMake                                                  | Equivalent GCC compiler flags   |  
|                           +------------------------+-------------------------------+---------------------------------+
|                           | CMAKE_BUILD_TYPE       | Additional flags              |                                 |
+===========================+========================+===============================+=================================+
| debug                     | debug                  |                               | -Og -g                          |
+---------------------------+------------------------+-------------------------------+---------------------------------+
| default                   | default|relwithdebinfo |                               | -O2 -g                          |
+---------------------------+------------------------+-------------------------------+---------------------------------+
| release                   | release                |                               | -O3                             |
+---------------------------+------------------------+-------------------------------+---------------------------------+
| optimized                 | release                | -DNS3_NATIVE_OPTIMIZATIONS=ON | -O3 -march=native -mtune=native |
+---------------------------+------------------------+-------------------------------+---------------------------------+

Configuring the project with CMake
++++++++++++++++++++++++++++++++++

.. _CMake: https://cmake.org/cmake/help/latest/manual/cmake.1.html
.. _CMAKE_BUILD_TYPE: https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
.. _Effects (g++): https://github.com/Kitware/CMake/blob/master/Modules/Compiler/GNU.cmake
.. _generator: https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html

Navigate to the ns-3-dev directory, create a CMake cache folder, 
navigate to it and run `CMake`_ pointing to the ns-3-dev folder.

.. sourcecode:: console

  ~$ cd ns-3-dev
  ~/ns-3-dev$ mkdir cmake-cache
  ~/ns-3-dev$ cd cmake-cache
  ~/ns-3-dev/cmake-cache$ cmake ..

You can pass additional arguments to the CMake command, to configure it. To change variable values,
you should use the -D option followed by the variable name.

As an example, the build type is stored in the variable named `CMAKE_BUILD_TYPE`_. Setting it to one 
of the CMake build types shown in the table below will change compiler settings associated with those
build types and output executable and libraries names, which will receive a suffix.

+------------------+-------------------+
| CMAKE_BUILD_TYPE | `Effects (g++)`_  |
+==================+===================+
| DEBUG            | -g                |
+------------------+-------------------+
| RELEASE          | -O3 -DNDEBUG      |
+------------------+-------------------+
| RELWITHDEBINFO   | -O2 -g -DNDEBUG   |
+------------------+-------------------+
| MINSIZEREL       | -Os -DNDEBUG      |
+------------------+-------------------+

You can set the build type with the following command, which assumes your terminal is inside the cache folder
created previously.

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ cmake -DCMAKE_BUILD_TYPE=DEBUG ..

Another common option to change is the `generator`_, which is the real underlying build system called by CMake.
There are many generators supported by CMake, including the ones listed in the table below.

+------------------------------------------------+
| Generators                                     |
+================================================+
| MinGW Makefiles                                |
+------------------------------------------------+
| Unix Makefiles                                 |
+------------------------------------------------+
| MSYS Makefiles                                 |
+------------------------------------------------+
| CodeBlocks - *one of the previous Makefiles*   |
+------------------------------------------------+
| Eclipse CDT4 - *one of the previous Makefiles* |
+------------------------------------------------+
| Ninja                                          |
+------------------------------------------------+
| Xcode                                          |
+------------------------------------------------+

To change the generator, you will need to pass one of these generators with the -G option. For example, if we 
prefer Ninja to Makefiles, which are the default, we need to run the following command:

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ cmake -G Ninja ..

This command may fail if there are different generator files in the same CMake cache folder. It is recommended to clean up
the CMake cache folder, then recreate it and reconfigure setting the generator in the first run.

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ cd ..
  ~/ns-3-dev$ rm -R cmake-cache && mkdir cmake-cache && cd cmake-cache
  ~/ns-3-dev/cmake-cache$ cmake -DCMAKE_BUILD_TYPE=release -G Ninja ..

After configuring for the first time, settings will be initialized to their
default values, and then you can use the ``ccmake`` command to manually change them:

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ ccmake .
  CMAKE_BUILD_TYPE                 release
  CMAKE_INSTALL_PREFIX             /usr/local
  NS3_ASSERT                       OFF
  ...
  NS3_EXAMPLES                     ON
  ...
  NS3_LOG                          OFF
  NS3_TESTS                        ON
  NS3_VERBOSE                      OFF
  ...

  CMAKE_BUILD_TYPE: Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...                                                       
  Keys: [enter] Edit an entry [d] Delete an entry                                                                                             CMake Version 3.22.1
        [l] Show log output   [c] Configure
        [h] Help              [q] Quit without generating
        [t] Toggle advanced mode (currently off)

After moving the cursor and setting the desired values, type ``c`` to configure CMake.

If you prefer doing everything with a non-interactive command, look at the main ``CMakeLists.txt``
file in the ns-3-dev directory. It contains most of the option flags and their default values.
To enable both examples and tests, run:

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ cmake -DNS3_EXAMPLES=ON -DNS3_TESTS=ON ..


.. _Manually refresh the CMake cache:

Manually refresh the CMake cache
********************************

After the project has been configured, calling ``CMake`` will 
:ref:`refresh the CMake cache<Manually refresh the CMake cache>`. 
The refresh is required to discover new targets: libraries, executables and/or modules
that were created since the last run.

The refresh is done by running the CMake command from the CMake cache folder. 

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ cmake ..

Previous settings stored in the CMakeCache.txt will be preserved, while new modules will be
scanned and targets will be added.

The cache can also be refreshed with the ns3 wrapper script:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 configure


Building the project
********************

There are three ways of building the project: 
using the ``ns3`` script, calling ``CMake`` and 
calling the underlying build system (e.g. Ninja) directly.
The last way is omitted, since each underlying build system 
has its own unique command-line syntax.

Building the project with ns3
+++++++++++++++++++++++++++++

The ns3 wrapper script makes life easier for command line users, accepting module names without
the ``lib`` prefix and scratch files without the ``scratch_`` prefix. The following command can 
be used to build the entire project:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 build 


To build specific targets, run:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 build target_name


Building the project with CMake
+++++++++++++++++++++++++++++++

The build process of targets (either libraries, executables or custom tasks) can be done
invoking CMake build. To build all the targets, run:

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ cmake --build .

Notice the single dot now refers to the ``cmake-cache`` directory, where the underlying
build system files are stored (referred inside CMake as ``PROJECT_BINARY_DIR`` or 
``CMAKE_BINARY_DIR``, which have slightly different uses if working with sub-projects).

.. _PROJECT_BINARY_DIR: https://cmake.org/cmake/help/latest/variable/PROJECT_BINARY_DIR.html
.. _CMAKE_BINARY_DIR: https://cmake.org/cmake/help/latest/variable/CMAKE_BINARY_DIR.html

To build specific targets, run:

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ cmake --build . --target target_name

Where target_name is a valid target name. Module libraries are prefixed with ``lib`` (e.g. libcore),
executables from the scratch folder are prefixed with ``scratch_`` (e.g. scratch_scratch-simulator).
Executables targets have their source file name without the ".cc" prefix 
(e.g. sample-simulator.cc => sample-simulator).


Adding a new module
*******************

Adding a module is the only case where 
:ref:`manually refreshing the CMake cache<Manually refresh the CMake cache>` is required. 

More information on how to create a new module are provided in :ref:`Adding a New Module to ns3`.

Migrating a Waf module to CMake
*******************************

If your module does not have external dependencies, porting is very easy.
Start by copying the module Wscript, rename them to CMakeLists.txt and then open it.

We are going to use the aodv module as an example:

.. sourcecode:: python3

  ## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

  def build(bld):
    module = bld.create_ns3_module('aodv', ['internet', 'wifi'])
    module.includes = '.'
    module.source = [
        'model/aodv-id-cache.cc',
        'model/aodv-dpd.cc',
        'model/aodv-rtable.cc',
        'model/aodv-rqueue.cc',
        'model/aodv-packet.cc',
        'model/aodv-neighbor.cc',
        'model/aodv-routing-protocol.cc',
        'helper/aodv-helper.cc',
        ]

    aodv_test = bld.create_ns3_module_test_library('aodv')
    aodv_test.source = [
        'test/aodv-id-cache-test-suite.cc',
        'test/aodv-test-suite.cc',
        'test/aodv-regression.cc',
        'test/bug-772.cc',
        'test/loopback.cc',
        ]

    # Tests encapsulating example programs should be listed here
    if (bld.env['ENABLE_EXAMPLES']):
        aodv_test.source.extend([
        #   'test/aodv-examples-test-suite.cc',
            ])
    
    headers = bld(features='ns3header')
    headers.module = 'aodv'
    headers.source = [
        'model/aodv-id-cache.h',
        'model/aodv-dpd.h',
        'model/aodv-rtable.h',
        'model/aodv-rqueue.h',
        'model/aodv-packet.h',
        'model/aodv-neighbor.h',
        'model/aodv-routing-protocol.h',
        'helper/aodv-helper.h',
        ]

    if bld.env['ENABLE_EXAMPLES']:
        bld.recurse('examples')

    bld.ns3_python_bindings()


We can see the module name is ``aodv`` and it depends on the ``internet`` and the ``wifi`` libraries, 
plus the lists of files (``module.source``, ``headers.source`` and ``module_test.source``).

This translates to the following CMake lines:

.. sourcecode:: cmake

  build_lib(
    LIBNAME aodv # aodv module, which can later be linked to examples and modules with ${libaodv}
    SOURCE_FILES # equivalent to module.source
      helper/aodv-helper.cc
      model/aodv-dpd.cc
      model/aodv-id-cache.cc
      model/aodv-neighbor.cc
      model/aodv-packet.cc
      model/aodv-routing-protocol.cc
      model/aodv-rqueue.cc
      model/aodv-rtable.cc
    HEADER_FILES # equivalent to headers.source
      helper/aodv-helper.h
      model/aodv-dpd.h
      model/aodv-id-cache.h
      model/aodv-neighbor.h
      model/aodv-packet.h
      model/aodv-routing-protocol.h
      model/aodv-rqueue.h
      model/aodv-rtable.h
    LIBRARIES_TO_LINK ${libinternet} # depends on internet and wifi,
                      ${libwifi}     # but both are prefixed with lib in CMake
    TEST_SOURCES # equivalent to module_test.source
      test/aodv-id-cache-test-suite.cc
      test/aodv-regression.cc
      test/aodv-test-suite.cc
      test/loopback.cc
      test/bug-772.cc
  )

If your module depends on external libraries, check the section 
`Linking third-party libraries`_.

Python bindings will be picked up if there is a subdirectory bindings 
and NS3_PYTHON_BINDINGS is enabled.

Next, we need to port the examples wscript. Repeat the copy, rename and open
steps. We should have something like the following:

.. sourcecode:: python3

  ## -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-

  def build(bld):
      obj = bld.create_ns3_program('aodv',
                                   ['wifi', 'internet', 'aodv', 'internet-apps'])
      obj.source = 'aodv.cc'

This means we create an example named ``aodv`` which depends on ``wifi``, ``internet``,
``aodv`` and ``internet-apps`` module, and has a single source file ``aodv.cc``.
This translates into the following CMake:

.. sourcecode:: cmake

  build_lib_example(
    NAME aodv # example named aodv
    SOURCE_FILES aodv.cc # single source file aodv.cc
    LIBRARIES_TO_LINK # depends on wifi, internet, aodv and internet-apps
      ${libwifi}
      ${libinternet}
      ${libaodv}
      ${libinternet-apps}
  )


Running programs
****************

Running programs with the ns3 wrapper script is pretty simple. To run the 
scratch program produced by ``scratch/scratch-simulator.cc``, you need the following:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 run scratch-simulator --no-build

Notice the ``--no-build`` indicates that the program should only be executed, and not built
before execution. 

To familiarize users with CMake, ns3 can also print the underlying CMake
and command line commands used by adding the ``--dry-run`` flag. 
Removing the ``--no-build`` flag and adding ``--dry-run`` to the same example, 
produces the following:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 --dry-run run scratch-simulator
  The following commands would be executed:
  cd cmake-cache; cmake --build . -j 15 --target scratch_scratch-simulator ; cd ..
  export PATH=$PATH:~/ns-3-dev/build/lib 
  export PYTHONPATH=~/ns-3-dev/build/bindings/python
  export LD_LIBRARY_PATH=~/ns-3-dev/build/lib 
  ./build/scratch/ns3-dev-scratch-simulator


In the CMake build command line, notice the scratch-simulator has a ``scratch_`` prefix.
That is true for all the CMake scratch targets. This is done to guarantee globally unique names.
Similarly, library-related targets have ``lib`` as a prefix (e.g. ``libcore``, ``libnetwork``).

.. _RPATH: https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_RPATH.html

The next few lines exporting variables guarantee the executable can find python dependencies
(``PYTHONPATH``) and linked libraries (``LD_LIBRARY_PATH`` and ``PATH`` on Unix-like, and 
``PATH`` on Windows). This is not necessary in platforms that support `RPATH`_.

Notice that when the scratch-simulator program is called on the last line, it has
a ns3-version prefix and could also have a build type suffix. 
This is valid for all libraries and executables, but ommited in ns-3 for simplicity.

Debugging can be done with GDB. Again, we have the two ways to run the program.
Using the ns-3 wrapper:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 run scratch-simulator --no-build --gdb

Or directly:

.. sourcecode:: console

  ~/ns-3-dev/cmake-cache$ export PATH=$PATH:~/ns-3-dev/build/lib
  ~/ns-3-dev/cmake-cache$ export PYTHONPATH=~/ns-3-dev/build/bindings/python
  ~/ns-3-dev/cmake-cache$ export LD_LIBRARY_PATH=~/ns-3-dev/build/lib
  ~/ns-3-dev/cmake-cache$ gdb ../build/scratch/ns3-dev-scratch-simulator


Modifying files
***************

As CMake is not a build system on itself, but a meta build system, it requires 
frequent refreshes, also known as reconfigurations. Those refreshes are triggered
automatically in the following cases:

* Changes in linked libraries
* Changes in the CMake code
* Header changes
* Header/source file name changes
* Module name changes

The following sections will detail some of these cases assuming a hypothetical module defined below.
Notice that the build_lib is the fundamental piece of every ns-3 module, while user-settable
options and external libraries checking are optional.

.. sourcecode:: cmake

    build_lib(
      LIBNAME hypothetical
      SOURCE_FILES  helper/hypothetical-helper.cc
                    model/hypothetical.cc
      HEADER_FILES 
        helper/hypothetical-helper.h
        model/hypothetical.h
        model/colliding-header.h
      LIBRARIES_TO_LINK ${libcore}
      )


Module name changes
+++++++++++++++++++

Changing the module name requires changing the value of ``LIBNAME``.
In the following example the name of the module seen previously is 
changed from ``hypothetical`` to ``new-hypothetical-name``:

.. sourcecode:: cmake

    build_lib(
      LIBNAME new-hypothetical-name
      # ... 
    )

If the module was already scanned, saving the changes and trying to build will trigger the
automatic CMake refresh. Otherwise, reconfigure the project to 
:ref:`manually refresh it<Manually refresh the CMake cache>`.


Header/source file name changes
+++++++++++++++++++++++++++++++

Assuming the hypothetical module defined previously has a header name that collides
with a header of a different module.

The name of the ``colliding-header.h`` can be changed via the filesystem to 
``non-colliding-header.h``, and the ``CMakeLists.txt`` path needs to be updated to match
the new name. Some IDEs can do this automatically through refactoring tools.

.. sourcecode:: cmake

    build_lib(
      LIBNAME new-hypothetical-name
      # ... 
      HEADER_FILES
          helper/hypothetical-helper.h
          model/hypothetical.h
          model/non-colliding-header.h
      # ... 
    )


Linking ns-3 modules
++++++++++++++++++++

Adding a dependency to another ns-3 module just requires adding ``${lib${modulename}}``
to the ``LIBRARIES_TO_LINK`` list, where modulename contains the value of the ns-3 
module which will be depended upon. 

Note: All ns-3 module libraries are prefixed with ``lib``,
as CMake requires unique global target names.

.. sourcecode:: cmake

    # now ${libnew-hypothetical-name} will depend on both core and internet modules
    build_lib(
      LIBNAME new-hypothetical-name
      # ... 
      LIBRARIES_TO_LINK ${libcore} 
                        ${libinternet}
      # ... 
    )


.. _Linking third-party libraries:

Linking third-party libraries
+++++++++++++++++++++++++++++

Depending on a third-party library is a bit more complicated as we have multiple
ways to handle that within CMake.

.. _Linking third-party libraries without CMake or PkgConfig support:

Linking third-party libraries without CMake or PkgConfig support
================================================================

When the third-party library you want to use do not export CMake files to use
``find_package`` or PkgConfig files to use ``pkg_check_modules``, we need to 
search for the headers and libraries manually. To simplify this process,
we include the macro ``find_external_library`` that searches for libraries and
header include directories, exporting results similarly to ``find_package``.

Here is how it works:

.. sourcecode:: cmake

  function(find_external_library)
    # Parse arguments
    set(options QUIET)
    set(oneValueArgs DEPENDENCY_NAME HEADER_NAME LIBRARY_NAME)
    set(multiValueArgs HEADER_NAMES LIBRARY_NAMES PATH_SUFFIXES SEARCH_PATHS)
    cmake_parse_arguments(
      "FIND_LIB" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
    )

    # Set the external package/dependency name
    set(name ${FIND_LIB_DEPENDENCY_NAME})

    # We process individual and list of headers and libraries by transforming them
    # into lists
    set(library_names "${FIND_LIB_LIBRARY_NAME};${FIND_LIB_LIBRARY_NAMES}")
    set(header_names "${FIND_LIB_HEADER_NAME};${FIND_LIB_HEADER_NAMES}")

    # Just changing the parsed argument name back to something shorter
    set(search_paths ${FIND_LIB_SEARCH_PATHS})
    set(path_suffixes "${FIND_LIB_PATH_SUFFIXES}")

    set(not_found_libraries)
    set(library_dirs)
    set(libraries)
    # Paths and suffixes where libraries will be searched on
    set(library_search_paths
            ${search_paths}
            ${CMAKE_OUTPUT_DIRECTORY} # Search for libraries in ns-3-dev/build
            ${CMAKE_INSTALL_PREFIX} # Search for libraries in the install directory (e.g. /usr/)
            $ENV{LD_LIBRARY_PATH} # Search for libraries in LD_LIBRARY_PATH directories
            $ENV{PATH} # Search for libraries in PATH directories
            )
    set(suffixes /build /lib /build/lib / /bin ${path_suffixes})

    # For each of the library names in LIBRARY_NAMES or LIBRARY_NAME
    foreach(library ${library_names})
      # We mark this value is advanced not to pollute the configuration with
      # ccmake with the cache variables used internally
      mark_as_advanced(${name}_library_internal_${library})

      # We search for the library named ${library} and store the results in
      # ${name}_library_internal_${library}
      find_library(
        ${name}_library_internal_${library} ${library}
        HINTS ${library_search_paths}
        PATH_SUFFIXES ${suffixes}
      )
      # cmake-format: off
      # Note: the PATH_SUFFIXES above apply to *ALL* PATHS and HINTS Which
      # translates to CMake searching on standard library directories
      # CMAKE_SYSTEM_PREFIX_PATH, user-settable CMAKE_PREFIX_PATH or
      # CMAKE_LIBRARY_PATH and the directories listed above
      #
      # e.g.  from Ubuntu 22.04 CMAKE_SYSTEM_PREFIX_PATH =
      # /usr/local;/usr;/;/usr/local;/usr/X11R6;/usr/pkg;/opt
      #
      # Searched directories without suffixes
      #
      # ${CMAKE_SYSTEM_PREFIX_PATH}[0] = /usr/local/
      # ${CMAKE_SYSTEM_PREFIX_PATH}[1] = /usr
      # ${CMAKE_SYSTEM_PREFIX_PATH}[2] = /
      # ...
      # ${CMAKE_SYSTEM_PREFIX_PATH}[6] = /opt
      # ${LD_LIBRARY_PATH}[0]
      # ...
      # ${LD_LIBRARY_PATH}[m]
      # ...
      #
      # Searched directories with suffixes include all of the directories above
      # plus all suffixes
      # PATH_SUFFIXES /build /lib /build/lib / /bin # ${path_suffixes}
      #
      # /usr/local/build
      # /usr/local/lib
      # /usr/local/build/lib
      # /usr/local/bin
      # ...
      #
      # cmake-format: on
      # Or enable NS3_VERBOSE to print the searched paths

      # Print tested paths to the searched library and if it was found
      if(${NS3_VERBOSE})
        log_find_searched_paths(
                TARGET_TYPE Library
                TARGET_NAME ${library}
                SEARCH_RESULT ${name}_library_internal_${library}
                SEARCH_PATHS ${library_search_paths}
                SEARCH_SUFFIXES ${suffixes}
        )
      endif()

      # After searching the library, the internal variable should have either the
      # absolute path to the library or the name of the variable appended with
      # -NOTFOUND
      if("${${name}_library_internal_${library}}" STREQUAL
         "${name}_library_internal_${library}-NOTFOUND"
      )
        # We keep track of libraries that were not found
        list(APPEND not_found_libraries ${library})
      else()
        # We get the name of the parent directory of the library and append the
        # library to a list of found libraries
        get_filename_component(
          ${name}_library_dir_internal ${${name}_library_internal_${library}}
          DIRECTORY
        ) # e.g. lib/openflow.(so|dll|dylib|a) -> lib
        list(APPEND library_dirs ${${name}_library_dir_internal})
        list(APPEND libraries ${${name}_library_internal_${library}})
      endif()
    endforeach()

    # For each library that was found (e.g. /usr/lib/pthread.so), get their parent
    # directory (/usr/lib) and its parent (/usr)
    set(parent_dirs)
    foreach(libdir ${library_dirs})
      get_filename_component(parent_libdir ${libdir} DIRECTORY)
      get_filename_component(parent_parent_libdir ${parent_libdir} DIRECTORY)
      list(APPEND parent_dirs ${libdir} ${parent_libdir} ${parent_parent_libdir})
    endforeach()

    # If we already found a library somewhere, limit the search paths for the header
    if(parent_dirs)
      set(header_search_paths ${parent_dirs})
      set(header_skip_system_prefix NO_CMAKE_SYSTEM_PATH)
    else()
      set(header_search_paths
              ${search_paths}
              ${CMAKE_OUTPUT_DIRECTORY} # Search for headers in ns-3-dev/build
              ${CMAKE_INSTALL_PREFIX} # Search for headers in the install
              )
    endif()

    set(not_found_headers)
    set(include_dirs)
    foreach(header ${header_names})
      # The same way with libraries, we mark the internal variable as advanced not
      # to pollute ccmake configuration with variables used internally
      mark_as_advanced(${name}_header_internal_${header})
      set(suffixes
              /build
              /include
              /build/include
              /build/include/${name}
              /include/${name}
              /${name}
              /
              ${path_suffixes}
              )
      # cmake-format: off
      # Here we search for the header file named ${header} and store the result in
      # ${name}_header_internal_${header}
      #
      # The same way we did with libraries, here we search on
      # CMAKE_SYSTEM_PREFIX_PATH, along with user-settable ${search_paths}, the
      # parent directories from the libraries, CMAKE_OUTPUT_DIRECTORY and
      # CMAKE_INSTALL_PREFIX
      #
      # And again, for each of them, for every suffix listed /usr/local/build
      # /usr/local/include
      # /usr/local/build/include
      # /usr/local/build/include/${name}
      # /usr/local/include/${name}
      # ...
      #
      # cmake-format: on
      # Or enable NS3_VERBOSE to get the searched paths printed while configuring

      find_file(
        ${name}_header_internal_${header} ${header}
        HINTS ${header_search_paths} # directory (e.g. /usr/)
        ${header_skip_system_prefix}
        PATH_SUFFIXES ${suffixes}
      )

      # Print tested paths to the searched header and if it was found
      if(${NS3_VERBOSE})
        log_find_searched_paths(
                TARGET_TYPE Header
                TARGET_NAME ${header}
                SEARCH_RESULT ${name}_header_internal_${header}
                SEARCH_PATHS ${header_search_paths}
                SEARCH_SUFFIXES ${suffixes}
                SEARCH_SYSTEM_PREFIX ${header_skip_system_prefix}
        )
      endif()

      # If the header file was not found, append to the not-found list
      if("${${name}_header_internal_${header}}" STREQUAL
         "${name}_header_internal_${header}-NOTFOUND"
      )
        list(APPEND not_found_headers ${header})
      else()
        # If the header file was found, get their directories and the parent of
        # their directories to add as include directories
        get_filename_component(
          header_include_dir ${${name}_header_internal_${header}} DIRECTORY
        ) # e.g. include/click/ (simclick.h) -> #include <simclick.h> should work
        get_filename_component(
          header_include_dir2 ${header_include_dir} DIRECTORY
        ) # e.g. include/(click) -> #include <click/simclick.h> should work
        list(APPEND include_dirs ${header_include_dir} ${header_include_dir2})
      endif()
    endforeach()

    # Remove duplicate include directories
    if(include_dirs)
      list(REMOVE_DUPLICATES include_dirs)
    endif()

    # If we find both library and header, we export their values
    if((NOT not_found_libraries}) AND (NOT not_found_headers))
      set(${name}_INCLUDE_DIRS "${include_dirs}" PARENT_SCOPE)
      set(${name}_LIBRARIES "${libraries}" PARENT_SCOPE)
      set(${name}_HEADER ${${name}_header_internal} PARENT_SCOPE)
      set(${name}_FOUND TRUE PARENT_SCOPE)
      set(status_message "find_external_library: ${name} was found.")
    else()
      set(${name}_INCLUDE_DIRS PARENT_SCOPE)
      set(${name}_LIBRARIES PARENT_SCOPE)
      set(${name}_HEADER PARENT_SCOPE)
      set(${name}_FOUND FALSE PARENT_SCOPE)
      set(status_message
          "find_external_library: ${name} was not found. Missing headers: \"${not_found_headers}\" and missing libraries: \"${not_found_libraries}\"."
      )
    endif()

    if(NOT ${FIND_LIB_QUIET})
      message(STATUS "${status_message}")
    endif()
  endfunction()

Debugging why a header or a library cannot be found is fairly tricky.
For ``find_external_library`` users, enabling the ``NS3_VERBOSE`` switch
will enable the logging of search path directories for both headers and
libraries.

.. _CMAKE_FIND_DEBUG_MODE: https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_DEBUG_MODE.html

Note: The logging provided by find_external_library is an alternative to
CMake's own ``CMAKE_FIND_DEBUG_MODE=true`` introduced in CMake 3.17,
which gets used by *ALL* ``find_file``, ``find_library``, ``find_header``,
``find_package`` and ``find_path`` calls throughout CMake and its modules.
If you are using a recent version of CMake, it is recommended to use
`CMAKE_FIND_DEBUG_MODE`_ instead.

A commented version of the Openflow module ``CMakeLists.txt`` has an 
example of ``find_external_library`` usage.

.. sourcecode:: cmake

    # Export a user option to specify the path to a custom
    # openflow build directory.
    set(NS3_WITH_OPENFLOW
        ""
        CACHE PATH
              "Build with Openflow support"
    )
    # We use this variable later in the ns-3-dev scope, but 
    # the value would be lost if we saved it to the 
    # parent scope ns-3-dev/src or ns-3-dev/contrib.
    # We set it as an INTERNAL CACHE variable to make it globally available.
    set(NS3_OPENFLOW
        "OFF"
        CACHE INTERNAL
              "ON if Openflow is found"
    )

    # This is the macro that searches for headers and libraries.
    # The DEPENDENCY_NAME is the equivalent of the find_package package name.
    # Resulting variables will be prefixed with DEPENDENCY_NAME.
    # - openflow_FOUND will be set to True if both headers and libraries
    #     were found and False otherwise
    # - openflow_LIBRARIES will contain a list of absolute paths to the 
    #     libraries named in LIBRARY_NAME|LIBRARY_NAMES
    # - openflow_INCLUDE_DIRS will contain a list of include directories that contain
    #     headers named in HEADER_NAME|HEADER_NAMES and directories that contain
    #     those directories.
    #     e.g. searching for core-module.h will return
    #     both ns-3-dev/build/include/ns3 and ns-3-dev/build/include,
    #     allowing users to include both <core-module.h> and <ns3/core-module.h>
    # If a user-settable variable was created, it can be searched too by
    # adding it to the SEARCH_PATHS
    find_external_library(
      DEPENDENCY_NAME openflow
      HEADER_NAME openflow.h
      LIBRARY_NAME openflow
      SEARCH_PATHS ${NS3_WITH_OPENFLOW} # user-settable search path, empty by default
    )

    # Before testing if the header and library were found ${openflow_FOUND},
    # test if openflow_FOUND was defined
    # If openflow_FOUND was not defined, the dependency name above doesn't match
    # the tested values below
    # If openflow_FOUND is set to FALSE, stop processing the module by returning
    # to the parent directory with return()
    if((NOT
        openflow_FOUND)
       AND (NOT
            ${openflow_FOUND})
    )
      message(STATUS "Openflow was not found")
      return()
    endif()

    # Check for the Boost header used by the openflow module
    check_include_file_cxx(
      boost/static_assert.hpp
      BOOST_STATIC_ASSERT
    )

    # Stop processing the module if it was not found
    if(NOT
      BOOST_STATIC_ASSERT
    )
      message(STATUS "Openflow requires Boost static_assert.hpp")
      return()
    endif()

    # Here we consume the include directories found by 
    # find_external_library
    #
    # This will make the following work:
    #  include<openflow/openflow.h>
    #  include<openflow.h>
    include_directories(${openflow_INCLUDE_DIRS})

    # Manually set definitions
    add_definitions(
      -DNS3_OPENFLOW
      -DENABLE_OPENFLOW
    )

    # Set the cache variable indicating Openflow is enabled as
    # all dependencies were met
    set(NS3_OPENFLOW
        "ON"
        CACHE INTERNAL
              "ON if Openflow is found in NS3_WITH_OPENFLOW"
    )

    # Additional compilation flag to ignore a specific warning
    add_compile_options(-Wno-stringop-truncation)

    # Call macro to create the module target
    build_lib(
      LIBNAME openflow
      SOURCE_FILES
        helper/openflow-switch-helper.cc
        model/openflow-interface.cc
        model/openflow-switch-net-device.cc
      HEADER_FILES
        helper/openflow-switch-helper.h
        model/openflow-interface.h
        model/openflow-switch-net-device.h
      LIBRARIES_TO_LINK ${libinternet}
                        # Here we consume the list of libraries 
                        # exported by find_external_library
                        ${openflow_LIBRARIES} 
      TEST_SOURCES test/openflow-switch-test-suite.cc
    )


Linking third-party libraries using CMake's find_package
========================================================

Assume we have a module with optional features that rely on a third-party library
that provides a FindThirdPartyPackage.cmake. This ``Find${Package}.cmake`` file can be distributed
by `CMake itself`_, via library/package managers (APT, Pacman, 
`VcPkg`_), or included to the project tree in the build-support/3rd-party directory.

.. _CMake itself: https://github.com/Kitware/CMake/tree/master/Modules
.. _Vcpkg: https://github.com/Microsoft/vcpkg#using-vcpkg-with-cmake

When ``find_package(${Package})`` is called, the ``Find${Package}.cmake`` file gets processed,
and multiple variables are set. There is no hard standard in the name of those variables, nor if
they should follow the modern CMake usage, where just linking to the library will include 
associated header directories, forward compile flags and so on. 

We assume the old CMake style is the one being used, which means we need to include the include
directories provided by the ``Find${Package}.cmake module``, usually exported as a variable 
``${Package}_INCLUDE_DIRS``, and get a list of libraries for that module so that they can be 
added to the list of libraries to link of the ns-3 modules. Libraries are usually exported as
the variable ``${Package}_LIBRARIES``.

As an example for the above, we use the Boost library 
(excerpt from ``macros-and-definitions.cmake`` and ``src/core/CMakeLists.txt``):

.. sourcecode:: cmake

  # https://cmake.org/cmake/help/v3.10/module/FindBoost.html?highlight=module%20find#module:FindBoost
  find_package(Boost)

  # It is recommended to create either an empty list that is conditionally filled
  # and later included in the LIBRARIES_TO_LINK list unconditionally
  set(boost_libraries)

  # If Boost is found, Boost_FOUND will be set to true, which we can then test
  if(${Boost_FOUND})
    # This will export Boost include directories to ALL subdirectories
    # of the current CMAKE_CURRENT_SOURCE_DIR
    #
    # If calling this from the top-level directory (ns-3-dev), it will
    # be used by all contrib/src modules, examples, etc
    include_directories(${Boost_INCLUDE_DIRS})

    # This is a trick for Boost
    # Sometimes you want to check if specific Boost headers are available,
    # but they would not be found if they're not in system include directories
    set(CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})

    # We get the list of Boost libraries and save them in the boost_libraries list
    set(boost_libraries ${Boost_LIBRARIES})
  endif()

  # If Boost was found earlier, we will be able to check if Boost headers are available
  check_include_file_cxx(
    "boost/units/quantity.hpp"
    HAVE_BOOST_UNITS_QUANTITY
  )
  check_include_file_cxx(
    "boost/units/systems/si.hpp"
    HAVE_BOOST_UNITS_SI
  )
  if(${HAVE_BOOST_UNITS_QUANTITY}
    AND ${HAVE_BOOST_UNITS_SI}
  )
    # Activate optional features that rely on Boost
    add_definitions(
      -DHAVE_BOOST
      -DHAVE_BOOST_UNITS
    )
    # In this case, the Boost libraries are header-only, 
    # but in case we needed real libraries, we could add
    # boost_libraries to either the auxiliary libraries_to_link list
    # or the build_lib's LIBRARIES_TO_LINK list
    message(STATUS "Boost Units have been found.")
  else()
    message(
      STATUS
        "Boost Units are an optional feature of length.cc."
    )
  endif()


If ``Find${Package}.cmake`` does not exist in your module path, CMake will warn you that is the case.
If ``${Package_FOUND}`` is set to False, other variables such as the ones related to libraries and 
include directories might not be set, and can result in CMake failures to configure if used.

In case the ``Find${Package}.cmake`` you need is not distributed by the upstream CMake project, 
you can create your own and add it to ``build-support/3rd-party``. This directory is included 
to the ``CMAKE_MODULE_PATH`` variable, making it available for calls without needing to include
the file with the absolute path to it. To add more directories to the ``CMAKE_MODULE_PATH``,
use the following:

.. sourcecode:: cmake

  # Excerpt from build-support/macros-and-definitions.cmake

  # Add ns-3 custom modules to the module path
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/build-support/custom-modules")

  # Add the 3rd-party modules to the module path
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/build-support/3rd-party")

  # Add your new modules directory to the module path 
  # (${PROJECT_SOURCE_DIR} is /path/to/ns-3-dev/)
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/build-support/new-modules")


One of the custom Find files currently shipped by ns-3 is the ``FindGTK3.cmake`` file. 
GTK3 requires Harfbuzz, which has its own ``FindHarfBuzz.cmake`` file. Both of them
are in the ``build-support/3rd-party`` directory.

.. sourcecode:: cmake

  # You don't need to keep adding this, this is just a demonstration
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/build-support/3rd-party")

  # If the user-settable NS3_GTK3 is set, look for HarfBuzz and GTK
  if(${NS3_GTK3})
    # Use FindHarfBuzz.cmake to find HarfBuzz
    find_package(HarfBuzz QUIET)

    # If HarfBuzz is not found
    if(NOT ${HarfBuzz_FOUND})
      message(STATUS "Harfbuzz is required by GTK3 and was not found.")
    else()
      # FindGTK3.cmake does some weird tricks and results in warnings,
      # that we can only suppress this way
      set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")

      # If HarfBuzz is found, search for GTK
      find_package(GTK3 QUIET)

      # Remove suppressions needed for quiet operations
      unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS CACHE)

      # If GTK3 is not found, inform the user
      if(NOT ${GTK3_FOUND})
        message(STATUS "GTK3 was not found. Continuing without it.")
      else()
        # If an incompatible version is found, set the GTK3_FOUND flag to false,
        # to make sure it won't be used later
        if(${GTK3_VERSION} VERSION_LESS 3.22)
          set(GTK3_FOUND FALSE)
          message(STATUS "GTK3 found with incompatible version ${GTK3_VERSION}")
        else()
          # A compatible GTK3 version was found
          message(STATUS "GTK3 was found.")
        endif()
      endif()
    endif()
  endif()

The Stats module can use the same ``find_package`` macro to search for SQLite3.

Note: we currently use a custom macro to find Python3 and SQLite3 since  
``FindPython3.cmake`` and ``FindSQLite3.cmake`` were included in CMake 3.12 and 3.14.
More details on how to use the macro are listed in 
`Linking third-party libraries without CMake or PkgConfig support`_.

.. sourcecode:: cmake

    # Set enable flag to false before checking
    set(ENABLE_SQLITE False)

    # In this case, SQLite presence is only checked if the user sets
    # NS3_SQLITE to ON, but your case may be different
    if(${NS3_SQLITE})
      # FindSQLite3.cmake is used by CMake to find SQLite3
      # QUIET flag silences most warnings from the module and let us write our own
      find_package(SQLite3 QUIET) # FindSQLite3.cmake was included in CMake 3.14

      # If SQLite3 was found, SQLite3_FOUND will be set to True, otherwise to False
      if(${SQLite3_FOUND})
        set(ENABLE_SQLITE True)
      else()
        message(STATUS "SQLite was not found")
      endif()
    endif()

    # Here we declare empty lists, that only hold values if ENABLE_SQLITE is set to ON
    set(sqlite_sources)
    set(sqlite_header)
    set(sqlite_libraries)
    if(${ENABLE_SQLITE})
      # If SQLite was found, add the optional source files to the lists
      set(sqlite_sources
          model/sqlite-data-output.cc
      )
      set(sqlite_headers
          model/sqlite-data-output.h
      )
      # Include the include directories containing the sqlite3.h header
      include_directories(${SQLite3_INCLUDE_DIRS})
      # Copy the list of sqlite3 libraries 
      set(sqlite_libraries
          ${SQLite3_LIBRARIES}
      )

      # If the semaphore header is also found, 
      # append additional optional source files to 
      # the sqlite sources and headers lists
      if(HAVE_SEMAPHORE_H)
        list(
          APPEND
          sqlite_sources
          model/sqlite-output.cc
        )
        list(
          APPEND
          sqlite_headers
          model/sqlite-output.h
        )
      endif()
    endif()

    # Sources and headers file lists for stats are quite long,
    # so we use these auxiliary lists 
    # The optional sqlite_sources and sqlite_headers can be empty or not
    set(source_files
        ${sqlite_sources}
        # ... 
        model/uinteger-8-probe.cc
    )

    set(header_files
        ${sqlite_headers}
        # ... 
        model/uinteger-8-probe.h
    )

    # Create the stats module consuming source files
    build_lib(
      LIBNAME stats
      SOURCE_FILES ${source_files}
      HEADER_FILES ${header_files}
      LIBRARIES_TO_LINK ${libcore}
                        # Here we either have an empty list or 
                        # a list with the sqlite library
                        ${sqlite_libraries} 
      TEST_SOURCES
        test/average-test-suite.cc
        test/basic-data-calculators-test-suite.cc
        test/double-probe-test-suite.cc
        test/histogram-test-suite.cc
    )



Linking third-party libraries with PkgConfig support
====================================================

Assume we have a module with optional features that rely on a third-party library
that uses PkgConfig. We can look for the PkgConfig module and add the optional
source files similarly to the previous cases, as shown in the example below:

.. sourcecode:: cmake

    # Include CMake script to use pkg-config
    include(FindPkgConfig)

    # If pkg-config was found, search for library you want
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(THIRD_PARTY libthird-party)
    endif()

    set(third_party_sources)
    set(third_party_libs)
    # Set cached variable if both pkg-config and libthird-party are found
    if(PKG_CONFIG_FOUND AND THIRD_PARTY)
      # Include third-party include directories for 
      # consumption of the current module and its examples
      include_directories(${THIRD_PARTY_INCLUDE_DIRS})

      # Use exported CFLAGS required by the third-party library
      add_compile_options(${THIRD_PARTY_CFLAGS})

      # Copy the list of third-party libraries
      set(third_party_libs ${THIRD_PARTY_LIBRARIES})

      # Add optional source files that depend on the third-party library
      set(third_party_sources model/optional-feature.cc)
    endif()

    # Create module using the optional source files and libraries
    build_lib(
      LIBNAME hypothetical
      SOURCE_FILES model/hypothetical.cc 
                   ${third_party_sources}
      HEADER_FILES model/hypothetical.h
      LIBRARIES_TO_LINK ${libcore}
                        # Here we either have an empty list or 
                        # a list with the third-party library
                        ${third_party_libs} 
      TEST_SOURCES
        test/hypothetical.cc
    )
    
    

Inclusion of options
++++++++++++++++++++

There are two ways of managing module options: option switches or cached variables.
Both are present in the main CMakeLists.txt in the ns-3-dev directory and the 
build-support/macros-and-definitions.cmake file.


.. sourcecode:: cmake

    # Here are examples of ON and OFF switches
    # option(
    #        NS3_SWITCH # option switch prefixed with NS3\_
    #        "followed by the description of what the option does" 
    #        ON # and the default value for that option
    #        )
    option(NS3_EXAMPLES "Enable examples to be built" OFF)
    option(NS3_TESTS "Enable tests to be built" OFF)

    # Now here is how to let the user indicate a path
    # set( # declares a value
    #     NS3_PREFIXED_VALUE # stores the option value
    #     "" # default value is empty in this case
    #     CACHE # stores that NS3_PREFIXED_VALUE in the CMakeCache.txt file
    #     STRING # type of the cached variable
    #     "description of what this value is used for"
    #     )
    set(NS3_OUTPUT_DIRECTORY "" CACHE PATH "Directory to store built artifacts")

    # The last case are options that can only assume predefined values
    # First we cache the default option
    set(NS3_INT64X64 "INT128" CACHE STRING "Int64x64 implementation")

    # Then set a cache property for the variable indicating it can assume 
    # specific values
    set_property(CACHE NS3_INT64X64 PROPERTY STRINGS INT128 CAIRO DOUBLE)


More details about these commands can be found in the following links:
`option`_, `set`_, `set_property`_.

.. _option: https://cmake.org/cmake/help/latest/command/option.html
.. _set: https://cmake.org/cmake/help/latest/command/set.html
.. _set_property: https://cmake.org/cmake/help/latest/command/set_property.html


Changes in CMake macros and functions
+++++++++++++++++++++++++++++++++++++

In order for CMake to feel more familiar to Waf users, a few macros and functions
were created. 

The most frequently used macros them can be found in 
``build-support/macros-and-definitions.cmake``. This file includes build type checking,
compiler family and version checking, enabling and disabling features based
on user options, checking for dependencies of enabled features, 
pre-compiling headers, filtering enabled/disabled modules and dependencies,
and more.

Module macros
=============

Module macros are located in ``build-support/custom-modules/ns3-module-macros.cmake``.
This file contains macros defining a library (``build_lib``), the associated test library,
examples (``build_lib_example``) and more. It also contains the macro that builds the 
module header (``write_module_header``) that includes all headers from the module
for user scripts.

These macros are responsible for easing the porting of modules from Waf to CMake.

Module macros: build_lib
~~~~~~~~~~~~~~~~~~~~~~~~

As ``build_lib`` is the most important of the macros, we detail what it does here,
block by block.

The first block declares the arguments received by the macro (in CMake, the
only difference is that a function has its own scope). Notice that there are
different types of arguments. Options that can only be set to true/false 
(``IGNORE_PCH``). 

One value arguments that receive a single value 
(usually a string) and in this case used to receive the module name (``LIBNAME``).

Multiple value arguments receive a list of values, which we use to parse lists
of source (for the module itself and for the module tests) and 
header files, plus libraries that should be linked and module features.

The call to ``cmake_parse_arguments`` will parse ``${ARGN}`` into these values.
The variables containing the parsing results will be prefixed with ``BLIB_`` 
(e.g. ``LIBNAME`` -> ``BLIB_LIBNAME``).

.. sourcecode:: cmake

  function(build_lib)
    # Argument parsing
    set(options IGNORE_PCH)
    set(oneValueArgs LIBNAME)
    set(multiValueArgs SOURCE_FILES HEADER_FILES LIBRARIES_TO_LINK TEST_SOURCES
                      DEPRECATED_HEADER_FILES MODULE_ENABLED_FEATURES
    )
    cmake_parse_arguments(
      "BLIB" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
    )
    # ... 
  endfunction()

In the following block, we add modules in the src folder to a list
and modules in the contrib folder to a different list.

.. sourcecode:: cmake

  function(build_lib)
    # ... 
    # Get path src/module or contrib/module
    string(REPLACE "${PROJECT_SOURCE_DIR}/" "" FOLDER
                  "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Add library to a global list of libraries
    if("${FOLDER}" MATCHES "src")
      set(ns3-libs "${lib${BLIB_LIBNAME}};${ns3-libs}"
          CACHE INTERNAL "list of processed upstream modules"
      )
    else()
      set(ns3-contrib-libs "${lib${BLIB_LIBNAME}};${ns3-contrib-libs}"
          CACHE INTERNAL "list of processed contrib modules"
      )
    endif()

In the following block, we check if we are working with Xcode, which does
not handle correctly CMake object libraries (.o files). 

In other platforms,
we build an object file ``add_library(${lib${BLIB_LIBNAME}-obj} OBJECT "${BLIB_SOURCE_FILES}...)``
and a shared library ``add_library(${lib${BLIB_LIBNAME}} SHARED ...)``.

The object library contains the actual source files (``${BLIB_SOURCE_FILES}``), 
but is not linked, which mean we can reuse the object to build the static version of the libraries.
Notice the shared library uses the object file as its source files 
``$<TARGET_OBJECTS:${lib${BLIB_LIBNAME}-obj}``.

Notice that we can also reuse precompiled headers created previously to speed up the 
parsing phase of the compilation.


.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    if(NOT ${XCODE})
      # Create object library with sources and headers, that will be used in
      # lib-ns3-static and the shared library
      add_library(
        ${lib${BLIB_LIBNAME}-obj} OBJECT "${BLIB_SOURCE_FILES}"
                                        "${BLIB_HEADER_FILES}"
      )

      if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${IGNORE_PCH}))
        target_precompile_headers(${lib${BLIB_LIBNAME}-obj} REUSE_FROM stdlib_pch)
      endif()

      # Create shared library with previously created object library (saving
      # compilation time for static libraries)
      add_library(
        ${lib${BLIB_LIBNAME}} SHARED $<TARGET_OBJECTS:${lib${BLIB_LIBNAME}-obj}>
      )
    else()
      # Xcode and CMake don't play well when using object libraries, so we have a
      # specific path for that
      add_library(${lib${BLIB_LIBNAME}} SHARED "${BLIB_SOURCE_FILES}")

      if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${IGNORE_PCH}))
        target_precompile_headers(${lib${BLIB_LIBNAME}} REUSE_FROM stdlib_pch)
      endif()
    endif()
    # ... 
  endfunction()

In the next code block, we create an alias to ``libmodule``, ``ns3::libmodule``,
which can later be used when importing ns-3 with CMake's ``find_package(ns3)``.

Then, we associate configured headers (``config-store-config``, ``core-config.h`` and
``version-defines.h``) to the core module.

And finally associate all of the public headers of the module to that library, 
to make sure CMake will be refreshed in case one of them changes.

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    add_library(ns3::${lib${BLIB_LIBNAME}} ALIAS ${lib${BLIB_LIBNAME}})

    # Associate public headers with library for installation purposes
    if("${BLIB_LIBNAME}" STREQUAL "core")
      set(config_headers ${CMAKE_HEADER_OUTPUT_DIRECTORY}/config-store-config.h
                        ${CMAKE_HEADER_OUTPUT_DIRECTORY}/core-config.h
      )
      if(${NS3_ENABLE_BUILD_VERSION})
        list(APPEND config_headers
            ${CMAKE_HEADER_OUTPUT_DIRECTORY}/version-defines.h
        )
      endif()
    endif()
    set_target_properties(
      ${lib${BLIB_LIBNAME}}
      PROPERTIES
        PUBLIC_HEADER
        "${BLIB_HEADER_FILES};${BLIB_DEPRECATED_HEADER_FILES};${config_headers};${CMAKE_HEADER_OUTPUT_DIRECTORY}/${BLIB_LIBNAME}-module.h"
    )
    # ... 
  endfunction()

In the next code block, we make the library a dependency to the ClangAnalyzer's time trace report,
which measures which step of compilation took most time and which files were responsible for that.

Then, the ns-3 libraries are separated from non-ns-3 libraries, that can be propagated or not
for libraries/executables linked to the current ns-3 module being processed.

The default is propagating these third-party libraries and their include directories, but this
can be turned off by setting ``NS3_REEXPORT_THIRD_PARTY_LIBRARIES=OFF`` 

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    if(${NS3_CLANG_TIMETRACE})
      add_dependencies(timeTraceReport ${lib${BLIB_LIBNAME}})
    endif()

    # Split ns and non-ns libraries to manage their propagation properly
    set(non_ns_libraries_to_link)
    set(ns_libraries_to_link)

    foreach(library ${BLIB_LIBRARIES_TO_LINK})
      remove_lib_prefix("${library}" module_name)

      # Check if the module exists in the ns-3 modules list
      # or if it is a 3rd-party library
      if(${module_name} IN_LIST ns3-all-enabled-modules)
        list(APPEND ns_libraries_to_link ${library})
      else()
        list(APPEND non_ns_libraries_to_link ${library})
      endif()
      unset(module_name)
    endforeach()

    if(NOT ${NS3_REEXPORT_THIRD_PARTY_LIBRARIES})
      # ns-3 libraries are linked publicly, to make sure other modules can find
      # each other without being directly linked
      set(exported_libraries PUBLIC ${LIB_AS_NEEDED_PRE} ${ns_libraries_to_link}
                            ${LIB_AS_NEEDED_POST}
      )

      # non-ns-3 libraries are linked privately, not propagating unnecessary
      # libraries such as pthread, librt, etc
      set(private_libraries PRIVATE ${LIB_AS_NEEDED_PRE}
                            ${non_ns_libraries_to_link} ${LIB_AS_NEEDED_POST}
      )

      # we don't re-export included libraries from 3rd-party modules
      set(exported_include_directories)
    else()
      # we export everything by default when NS3_REEXPORT_THIRD_PARTY_LIBRARIES=ON
      set(exported_libraries PUBLIC ${LIB_AS_NEEDED_PRE} ${ns_libraries_to_link}
                            ${non_ns_libraries_to_link} ${LIB_AS_NEEDED_POST}
      )
      set(private_libraries)

      # with NS3_REEXPORT_THIRD_PARTY_LIBRARIES, we export all 3rd-party library
      # include directories, allowing consumers of this module to include and link
      # the 3rd-party code with no additional setup
      get_target_includes(${lib${BLIB_LIBNAME}} exported_include_directories)
      string(REPLACE "-I" "" exported_include_directories
                    "${exported_include_directories}"
      )
      string(REPLACE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/include" ""
                    exported_include_directories
                    "${exported_include_directories}"
      )
    endif()
    # ... 
  endfunction()

After the lists of libraries to link that should be exported (``PUBLIC``) and 
not exported (``PRIVATE``) are built, we can link them with ``target_link_libraries``.

Next, we set the output name of the module library to n3version-modulename(optional build suffix).

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    target_link_libraries(
      ${lib${BLIB_LIBNAME}} ${exported_libraries} ${private_libraries}
    )

    # set output name of library
    set_target_properties(
      ${lib${BLIB_LIBNAME}}
      PROPERTIES OUTPUT_NAME ns${NS3_VER}-${BLIB_LIBNAME}${build_profile_suffix}
    )
    # ... 
  endfunction()

Next we export include directories, to let library consumers importing ns-3 via CMake
use them just by linking to one of the ns-3 modules.

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    # export include directories used by this library so that it can be used by
    # 3rd-party consumers of ns-3 using find_package(ns3) this will automatically
    # add the build/include path to them, so that they can ns-3 headers with
    # <ns3/something.h>
    target_include_directories(
      ${lib${BLIB_LIBNAME}}
      PUBLIC $<BUILD_INTERFACE:${CMAKE_OUTPUT_DIRECTORY}/include>
            $<INSTALL_INTERFACE:include>
      INTERFACE ${exported_include_directories}
    )
    # ... 
  endfunction()

We append the list of third-party/external libraries for each processed module,
and append a list of object libraries that can be later used for the static ns-3 build.

.. sourcecode:: cmake
    
  function(build_lib)
    # ...   
    set(ns3-external-libs "${non_ns_libraries_to_link};${ns3-external-libs}"
        CACHE INTERNAL
              "list of non-ns libraries to link to NS3_STATIC and NS3_MONOLIB"
    )
    if(${NS3_STATIC} OR ${NS3_MONOLIB})
      set(lib-ns3-static-objs
          "$<TARGET_OBJECTS:${lib${BLIB_LIBNAME}-obj}>;${lib-ns3-static-objs}"
          CACHE
            INTERNAL
            "list of object files from module used by NS3_STATIC and NS3_MONOLIB"
      )
    endif()
    # ... 
  endfunction()

The following block creates the ``${BLIB_LIBNAME}-module.h`` header for user scripts, 
and copies header files from src/module and contrib/module to the include/ns3 directory.

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    # Write a module header that includes all headers from that module
    write_module_header("${BLIB_LIBNAME}" "${BLIB_HEADER_FILES}")

    # Copy all header files to outputfolder/include before each build
    copy_headers_before_building_lib(
      ${BLIB_LIBNAME} ${CMAKE_HEADER_OUTPUT_DIRECTORY} "${BLIB_HEADER_FILES}"
      public
    )
    if(BLIB_DEPRECATED_HEADER_FILES)
      copy_headers_before_building_lib(
        ${BLIB_LIBNAME} ${CMAKE_HEADER_OUTPUT_DIRECTORY}
        "${BLIB_DEPRECATED_HEADER_FILES}" deprecated
      )
    endif()
    # ... 
  endfunction()

The following block creates the test library for the module currently being processed.

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    # Build tests if requested
    if(${ENABLE_TESTS})
      list(LENGTH BLIB_TEST_SOURCES test_source_len)
      if(${test_source_len} GREATER 0)
        # Create BLIB_LIBNAME of output library test of module
        set(test${BLIB_LIBNAME} lib${BLIB_LIBNAME}-test CACHE INTERNAL "")
        set(ns3-libs-tests "${test${BLIB_LIBNAME}};${ns3-libs-tests}"
            CACHE INTERNAL "list of test libraries"
        )

        # Create shared library containing tests of the module
        add_library(${test${BLIB_LIBNAME}} SHARED "${BLIB_TEST_SOURCES}")

        # Link test library to the module library
        if(${NS3_MONOLIB})
          target_link_libraries(
            ${test${BLIB_LIBNAME}} ${LIB_AS_NEEDED_PRE} ${lib-ns3-monolib}
            ${LIB_AS_NEEDED_POST}
          )
        else()
          target_link_libraries(
            ${test${BLIB_LIBNAME}} ${LIB_AS_NEEDED_PRE} ${lib${BLIB_LIBNAME}}
            "${BLIB_LIBRARIES_TO_LINK}" ${LIB_AS_NEEDED_POST}
          )
        endif()
        set_target_properties(
          ${test${BLIB_LIBNAME}}
          PROPERTIES OUTPUT_NAME
                    ns${NS3_VER}-${BLIB_LIBNAME}-test${build_profile_suffix}
        )

        target_compile_definitions(
          ${test${BLIB_LIBNAME}} PRIVATE NS_TEST_SOURCEDIR="${FOLDER}/test"
        )
        if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${IGNORE_PCH}))
          target_precompile_headers(${test${BLIB_LIBNAME}} REUSE_FROM stdlib_pch)
        endif()
      endif()
    endif()
    # ... 
  endfunction()

The following block checks for examples subdirectories and add them to parse their 
CMakeLists.txt file, creating the examples. It also scans for python examples. 

.. sourcecode:: cmake

  function(build_lib)
    # ... 
    # Build lib examples if requested
    if(${ENABLE_EXAMPLES})
      foreach(example_folder example;examples)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_folder})
          if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_folder}/CMakeLists.txt)
            add_subdirectory(${example_folder})
          endif()
          scan_python_examples(${CMAKE_CURRENT_SOURCE_DIR}/${example_folder})
        endif()
      endforeach()
    endif()
    # ... 
  endfunction()

The following block creates the ``lib${BLIB_LIBNAME}-apiscan`` CMake target.
When the target is built, it runs modulescan-modular to extract the python 
bindings for the module using pybindgen. 

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    # Get architecture pair for python bindings
    if((${CMAKE_SIZEOF_VOID_P} EQUAL 8) AND (NOT APPLE))
      set(arch gcc_LP64)
      set(arch_flags -m64)
    else()
      set(arch gcc_ILP32)
      set(arch_flags)
    endif()

    # Add target to scan python bindings
    if(${ENABLE_SCAN_PYTHON_BINDINGS}
      AND EXISTS ${CMAKE_HEADER_OUTPUT_DIRECTORY}/${BLIB_LIBNAME}-module.h
    )
      set(bindings_output_folder ${PROJECT_SOURCE_DIR}/${FOLDER}/bindings)
      file(MAKE_DIRECTORY ${bindings_output_folder})
      set(module_api_ILP32 ${bindings_output_folder}/modulegen__gcc_ILP32.py)
      set(module_api_LP64 ${bindings_output_folder}/modulegen__gcc_LP64.py)

      set(modulescan_modular_command
          ${Python_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/bindings/python/ns3modulescan-modular.py
      )

      set(header_map "")
      # We need a python map that takes header.h to module e.g. "ptr.h": "core"
      foreach(header ${BLIB_HEADER_FILES})
        # header is a relative path to the current working directory
        get_filename_component(
          header_name ${CMAKE_CURRENT_SOURCE_DIR}/${header} NAME
        )
        string(APPEND header_map "\"${header_name}\":\"${BLIB_LIBNAME}\",")
      endforeach()

      set(ns3-headers-to-module-map "${ns3-headers-to-module-map}${header_map}"
          CACHE INTERNAL "Map connecting headers to their modules"
      )

      # API scan needs the include directories to find a few headers (e.g. mpi.h)
      get_target_includes(${lib${BLIB_LIBNAME}} modulegen_include_dirs)

      set(module_to_generate_api ${module_api_ILP32})
      set(LP64toILP32)
      if("${arch}" STREQUAL "gcc_LP64")
        set(module_to_generate_api ${module_api_LP64})
        set(LP64toILP32
            ${Python_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/build-support/pybindings_LP64_to_ILP32.py
            ${module_api_LP64} ${module_api_ILP32}
        )
      endif()

      add_custom_target(
        ${lib${BLIB_LIBNAME}}-apiscan
        COMMAND
          ${modulescan_modular_command} ${CMAKE_OUTPUT_DIRECTORY} ${BLIB_LIBNAME}
          ${PROJECT_BINARY_DIR}/header_map.json ${module_to_generate_api}
          \"${arch_flags} ${modulegen_include_dirs}\"
        COMMAND ${LP64toILP32}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${lib${BLIB_LIBNAME}}
      )
      add_dependencies(apiscan-all ${lib${BLIB_LIBNAME}}-apiscan)
    endif()
    
    # ... 
  endfunction()

The targets can be built to execute the scanning using one of the following commands:

.. sourcecode:: console

  ~/cmake-cache$ cmake --build . --target libcore-apiscan
  ~/ns-3-dev/$ ./ns3 build core-apiscan
  
To re-scan all bindings, use ``./ns3 build apiscan-all``. 

The next block runs the ``modulegen-modular`` and consumes the results of the previous step.
However, this step runs at configuration time, while the previous runs at build time.
This means the output directory needs to be cleared after running ``apiscan``, to regenerate 
up-to-date bindings source files.

.. sourcecode:: cmake
    
  function(build_lib)
    # ... 
    # Build pybindings if requested and if bindings subfolder exists in
    # NS3/src/BLIB_LIBNAME
    if(${ENABLE_PYTHON_BINDINGS} AND EXISTS
                                    "${CMAKE_CURRENT_SOURCE_DIR}/bindings"
    )
      set(bindings_output_folder ${CMAKE_OUTPUT_DIRECTORY}/${FOLDER}/bindings)
      file(MAKE_DIRECTORY ${bindings_output_folder})
      set(module_src ${bindings_output_folder}/ns3module.cc)
      set(module_hdr ${bindings_output_folder}/ns3module.h)

      string(REPLACE "-" "_" BLIB_LIBNAME_sub ${BLIB_LIBNAME}) # '-' causes
                                                              # problems (e.g.
      # csma-layout), replace with '_' (e.g. csma_layout)

      # Set prefix of binding to _ if a ${BLIB_LIBNAME}.py exists, and copy the
      # ${BLIB_LIBNAME}.py to the output folder
      set(prefix)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bindings/${BLIB_LIBNAME}.py)
        set(prefix _)
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/${BLIB_LIBNAME}.py
            DESTINATION ${CMAKE_OUTPUT_DIRECTORY}/bindings/python/ns
        )
      endif()

      # Run modulegen-modular to generate the bindings sources
      if((NOT EXISTS ${module_hdr}) OR (NOT EXISTS ${module_src})) # OR TRUE) # to
                                                                  # force
                                                                  # reprocessing
        string(REPLACE ";" "," ENABLED_FEATURES
                      "${ns3-libs};${BLIB_MODULE_ENABLED_FEATURES}"
        )
        set(modulegen_modular_command
            GCC_RTTI_ABI_COMPLETE=True NS3_ENABLED_FEATURES="${ENABLED_FEATURES}"
            ${Python_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/bindings/python/ns3modulegen-modular.py
        )
        execute_process(
          COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_OUTPUT_DIRECTORY}:$ENV{PYTHONPATH}
            ${modulegen_modular_command} ${CMAKE_CURRENT_SOURCE_DIR} ${arch}
            ${prefix}${BLIB_LIBNAME_sub} ${module_src}
          TIMEOUT 60
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          OUTPUT_FILE ${module_hdr}
          ERROR_FILE ${bindings_output_folder}/ns3modulegen.log
          RESULT_VARIABLE error_code
        )
        if(${error_code} OR NOT (EXISTS ${module_hdr}))
          message(
            FATAL_ERROR
              "Something went wrong during processing of the python bindings of module ${BLIB_LIBNAME}."
              " Make sure you have the latest version of Pybindgen."
          )
          if(EXISTS ${module_src})
            file(REMOVE ${module_src})
          endif()
        endif()
      endif()

      # Add core module helper sources
      set(python_module_files ${module_hdr} ${module_src})
      if(${BLIB_LIBNAME} STREQUAL "core")
        list(APPEND python_module_files
            ${CMAKE_CURRENT_SOURCE_DIR}/bindings/module_helpers.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/bindings/scan-header.h
        )
      endif()
      set(bindings-name lib${BLIB_LIBNAME}-bindings)
      add_library(${bindings-name} SHARED "${python_module_files}")
      target_include_directories(
        ${bindings-name} PUBLIC ${Python_INCLUDE_DIRS} ${bindings_output_folder}
      )
      target_compile_options(${bindings-name} PRIVATE -Wno-error)

      # If there is any, remove the "lib" prefix of libraries (search for
      # "set(lib${BLIB_LIBNAME}")
      list(LENGTH ns_libraries_to_link num_libraries)
      if(num_libraries GREATER "0")
        string(REPLACE ";" "-bindings;" bindings_to_link
                      "${ns_libraries_to_link};"
        ) # add -bindings suffix to all lib${name}
      endif()
      target_link_libraries(
        ${bindings-name}
        PUBLIC ${LIB_AS_NEEDED_PRE} ${lib${BLIB_LIBNAME}} "${bindings_to_link}"
              "${BLIB_LIBRARIES_TO_LINK}" ${LIB_AS_NEEDED_POST}
        PRIVATE ${Python_LIBRARIES}
      )
      target_include_directories(
        ${bindings-name} PRIVATE ${PROJECT_SOURCE_DIR}/src/core/bindings
      )

      set(suffix)
      if(APPLE)
        # Python doesn't like Apple's .dylib and will refuse to load bindings
        # unless its an .so
        set(suffix SUFFIX .so)
      endif()

      # Set binding library name and output folder
      set_target_properties(
        ${bindings-name}
        PROPERTIES OUTPUT_NAME ${prefix}${BLIB_LIBNAME_sub}
                  PREFIX ""
                  ${suffix} LIBRARY_OUTPUT_DIRECTORY
                  ${CMAKE_OUTPUT_DIRECTORY}/bindings/python/ns
      )

      set(ns3-python-bindings-modules
          "${bindings-name};${ns3-python-bindings-modules}"
          CACHE INTERNAL "list of modules python bindings"
      )
      
      # Make sure all bindings are built before building the visualizer module
      # that makes use of them
      if(${ENABLE_VISUALIZER} AND (visualizer IN_LIST libs_to_build))
        if(NOT (${BLIB_LIBNAME} STREQUAL visualizer))
          add_dependencies(${libvisualizer} ${bindings-name})
        endif()
      endif()
    endif()
    # ... 
  endfunction()

The recommended steps to scan the bindings and then use them is the following:

.. sourcecode:: console

  ~/ns-3-dev$ ./ns3 clean
  ~/ns-3-dev$ ./ns3 configure -- -DNS3_SCAN_PYTHON_BINDINGS=ON
  ~/ns-3-dev$ ./ns3 build apiscan-all
  ~/ns-3-dev$ ./ns3 configure --enable-python-bindings
  ~/ns-3-dev$ ./ns3 build


In the next code block we add the library to the ns3ExportTargets, later used for installation.
We also print an additional message the folder just finished being processed if NS3_VERBOSE is set to ON.

.. sourcecode:: cmake
    
  function(build_lib)
    # ...   
    # Handle package export
    install(
      TARGETS ${lib${BLIB_LIBNAME}}
      EXPORT ns3ExportTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/
      PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ns3"
    )
    if(${NS3_VERBOSE})
      message(STATUS "Processed ${FOLDER}")
    endif()
  endfunction()

Module macros: build_lib_example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second most important macro from a module author perspective is the ``build_lib_example``, which
builds the examples for their module. As with ``build_lib`` we explain what it does block-by-block.

In the first block, arguments are parsed and we check wether the current module is in the contrib 
or the src folder.

.. sourcecode:: cmake

  function(build_lib_example)
    # Argument parsing
    set(options IGNORE_PCH)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCE_FILES HEADER_FILES LIBRARIES_TO_LINK)
    cmake_parse_arguments("BLIB_EXAMPLE" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Get path src/module or contrib/module
    string(REPLACE "${PROJECT_SOURCE_DIR}/" "" FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
    get_filename_component(FOLDER ${FOLDER} DIRECTORY)
    # ... 
  endfunction()

Then we check if the ns-3 modules required by the example are enabled to be built. 
If the list ``missing_dependencies`` is empty, we create the example. Otherwise, we skip it.
The example can be linked to the current module (``${lib${BLIB_EXAMPLE_LIBNAME}}``) and 
other libraries to link (``${BLIB_EXAMPLE_LIBRARIES_TO_LINK}``) and optionally to the visualizer
module (``${optional_visualizer_lib}``). 
If the visualizer module is not enabled, ``optional_visualizer_lib`` is empty.

The example can also be linked to a single ns-3 shared library (``lib-ns3-monolib``) or 
a single ns-3 static library (``lib-ns3-static``), if either ``NS3_MONOLIB=ON`` or ``NS3_STATIC=ON``.

.. sourcecode:: cmake

  function(build_lib_example)
    # ...
    check_for_missing_libraries(missing_dependencies "${BLIB_EXAMPLE_LIBRARIES_TO_LINK}")
    if(NOT missing_dependencies)
      # Create shared library with sources and headers
      add_executable(
        "${BLIB_EXAMPLE_NAME}" ${BLIB_EXAMPLE_SOURCE_FILES}
                              ${BLIB_EXAMPLE_HEADER_FILES}
      )

      if(${NS3_STATIC})
        target_link_libraries(
          ${BLIB_EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE_STATIC} ${lib-ns3-static}
        )
      elseif(${NS3_MONOLIB})
        target_link_libraries(
          ${BLIB_EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE} ${lib-ns3-monolib}
          ${LIB_AS_NEEDED_POST}
        )
      else()
        target_link_libraries(
          ${BLIB_EXAMPLE_NAME} ${LIB_AS_NEEDED_PRE} ${lib${BLIB_EXAMPLE_LIBNAME}}
          ${BLIB_EXAMPLE_LIBRARIES_TO_LINK} ${optional_visualizer_lib}
          ${LIB_AS_NEEDED_POST}
        )
      endif()
      # ... 
    endif()
  endfunction()

As with the module libraries, we can also reuse precompiled headers here to speed up 
the parsing step of compilation.

Finally, we call another macro ``set_runtime_outputdirectory``, which indicates the 
resulting folder where the example will end up after built (e.g. build/src/module/examples)
and adds the proper ns-3 version prefix and build type suffix to the executable.

.. sourcecode:: cmake

  function(build_lib_example)
    # ... 
    if(NOT missing_dependencies)
      # ...   
      if(${PRECOMPILE_HEADERS_ENABLED} AND (NOT ${BLIB_EXAMPLE_IGNORE_PCH}))
        target_precompile_headers(${BLIB_EXAMPLE_NAME} REUSE_FROM stdlib_pch_exec)
      endif()

      set_runtime_outputdirectory(
        ${BLIB_EXAMPLE_NAME}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FOLDER}/examples/ ""
      )
    endif()
  endfunction()

User options and header checking
================================

User-settable options should be prefixed with ``NS3_``, otherwise
they will not be preserved by ``./ns3 configure --force-refresh``.

After checking if the pre-requisites of the user-settable options
are met, set the same option now prefixed with ``ENABLE_``. The
following example demonstrates this pattern:

.. sourcecode:: cmake

  # Option() means the variable NS3_GSL will be set to ON/OFF
  # The second argument is a comment explaining what this option does
  # The last argument is the default value for the user-settable option
  option(NS3_GSL "Enable GSL related features" OFF)

  # Set the ENABLE\_ counterpart to FALSE by default
  set(ENABLE_GSL FALSE)
  if(${NS3_GSL})
    # If the user enabled GSL, check if GSL is available
    find_package(GSL)
    if(${GSL_FOUND})
      set(ENABLE_GSL TRUE)
      message(STATUS "GSL was requested by the user and was found")
    else()
      message(STATUS "GSL was not found and GSL features will continue disabled")
    endif()
  else()
    message(STATUS "GSL features were not requested by the user")
  endif()

  # Now the module can check for ENABLE_GSL before being processed 
  if(NOT ${ENABLE_GSL})
    return()
  endif()

  # Or to enable optional features
  set(gsl_sources)
  if(${ENABLE_GSL})
    set(gsl_sources model/gsl_features.cc)
  endif()

Here are examples of how to do the options and header checking, 
followed by a header configuration:

.. sourcecode:: cmake

  # We always set the ENABLE\_ counterpart of NS3\_ option to FALSE before checking
  #
  # If this variable is created inside your module, use
  # set(ENABLE_MPI FALSE CACHE INTERNAL "") 
  # instead, to make it globally available
  set(ENABLE_MPI FALSE)

  # If the user option switch is set to ON, we check
  if(${NS3_MPI})
    # Use find_package to look for MPI
    find_package(MPI QUIET)

    # If the package is optional, which is the case for MPI,
    # we can proceed if it is not found
    if(NOT ${MPI_FOUND})
      message(STATUS "MPI was not found. Continuing without it.")
    else()
      # If it is false, we add necessary C++ definitions (e.g. NS3_MPI)
      message(STATUS "MPI was found.")
      add_definitions(-DNS3_MPI)

      # Then set ENABLE_MPI to TRUE, which can be used to check 
      # if NS3_MPI is enabled AND MPI was found
      #
      # If this variable is created inside your module, use
      # set(ENABLE_MPI TRUE CACHE INTERNAL "") 
      # instead, to make it globally available
      set(ENABLE_MPI TRUE)
    endif()
  endif()

  # ... 

  # These two standard CMake modules allow for header and function checking
  include(CheckIncludeFileCXX)
  include(CheckFunctionExists)

  # Check for required headers and functions, 
  # set flags on the right argument if header in the first argument is found
  # if they are not found, a warning is emitted
  check_include_file_cxx("stdint.h" "HAVE_STDINT_H")
  check_include_file_cxx("inttypes.h" "HAVE_INTTYPES_H")
  check_include_file_cxx("sys/types.h" "HAVE_SYS_TYPES_H")
  check_include_file_cxx("stat.h" "HAVE_SYS_STAT_H")
  check_include_file_cxx("dirent.h" "HAVE_DIRENT_H")
  check_include_file_cxx("stdlib.h" "HAVE_STDLIB_H")
  check_include_file_cxx("signal.h" "HAVE_SIGNAL_H")
  check_include_file_cxx("netpacket/packet.h" "HAVE_PACKETH")
  check_function_exists("getenv" "HAVE_GETENV")

  # This is the CMake command to open up a file template (in this case a header
  # passed as the first argument), then fill its fields with values stored in 
  # CMake variables and save the resulting file to the target destination 
  # (in the second argument)
  configure_file(
    build-support/core-config-template.h
    ${CMAKE_HEADER_OUTPUT_DIRECTORY}/core-config.h
  )

The configure_file command is not very clear by itself, as you do not know which
values are being used. So we need to check the template.

.. sourcecode:: cpp

    #ifndef NS3_CORE_CONFIG_H
    #define NS3_CORE_CONFIG_H

    // Defined if HAVE_UINT128_T is defined in CMake
    #cmakedefine   HAVE_UINT128_T 
    // Set to 1 if HAVE__UINT128_T is defined in CMake, 0 otherwise
    #cmakedefine01 HAVE___UINT128_T 
    #cmakedefine   INT64X64_USE_128
    #cmakedefine   INT64X64_USE_DOUBLE
    #cmakedefine   INT64X64_USE_CAIRO
    #cmakedefine01 HAVE_STDINT_H
    #cmakedefine01 HAVE_INTTYPES_H
    #cmakedefine   HAVE_SYS_INT_TYPES_H
    #cmakedefine01 HAVE_SYS_TYPES_H
    #cmakedefine01 HAVE_SYS_STAT_H
    #cmakedefine01 HAVE_DIRENT_H
    #cmakedefine01 HAVE_STDLIB_H
    #cmakedefine01 HAVE_GETENV
    #cmakedefine01 HAVE_SIGNAL_H
    #cmakedefine   HAVE_PTHREAD_H
    #cmakedefine   HAVE_RT

    /*
    * #cmakedefine turns into:
    * //#define HAVE_FLAG // if HAVE_FLAG is not defined in CMake (e.g. unset(HAVE_FLAG))
    * #define HAVE_FLAG // if HAVE_FLAG is defined in CMake (e.g. set(HAVE_FLAG))
    *
    * #cmakedefine01 turns into:
    * #define HAVE_FLAG 0 // if HAVE_FLAG is not defined in CMake
    * #define HAVE_FLAG 1 // if HAVE_FLAG is defined in CMake
    */

    #endif //NS3_CORE_CONFIG_H

Custom targets
==============

Another common thing to do is implement custom targets that run specific commands and
manage dependencies. Here is an example for Doxygen:

.. sourcecode:: cmake

  # This command hides DOXYGEN from some CMake cache interfaces
  mark_as_advanced(DOXYGEN)

  # This custom macro checks for dependencies CMake find_package and program 
  # dependencies and return the missing dependencies in the third argument
  check_deps("" "doxygen;dot;dia" doxygen_docs_missing_deps)

  # If the variable contains missing dependencies, we stop processing doxygen targets
  if(doxygen_docs_missing_deps)
    message(
      STATUS
        "docs: doxygen documentation not enabled due to missing dependencies: ${doxygen_docs_missing_deps}"
    )
  else()
    # We checked this already exists, but we need the path to the executable
    find_package(Doxygen QUIET)

    # Get introspected doxygen
    add_custom_target(
      run-print-introspected-doxygen
      COMMAND
        ${CMAKE_OUTPUT_DIRECTORY}/utils/ns${NS3_VER}-print-introspected-doxygen${build_profile_suffix}
        > ${PROJECT_SOURCE_DIR}/doc/introspected-doxygen.h
      COMMAND
        ${CMAKE_OUTPUT_DIRECTORY}/utils/ns${NS3_VER}-print-introspected-doxygen${build_profile_suffix}
        --output-text > ${PROJECT_SOURCE_DIR}/doc/ns3-object.txt
      DEPENDS print-introspected-doxygen
    )

    # Run test.py with NS_COMMANDLINE_INTROSPECTION=.. to print examples 
    # introspected commandline
    add_custom_target(
      run-introspected-command-line
      COMMAND ${CMAKE_COMMAND} -E env NS_COMMANDLINE_INTROSPECTION=..
              ${Python_EXECUTABLE} ./test.py --no-build --constrain=example
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS all-test-targets # all-test-targets only exists if ENABLE_TESTS is
                               # set to ON
    )

    # This file header is written during configuration
    file(
      WRITE ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h
      "/* This file is automatically generated by
  CommandLine::PrintDoxygenUsage() from the CommandLine configuration
  in various example programs.  Do not edit this file!  Edit the
  CommandLine configuration in those files instead.
  */
  \n"
    )
    # After running test.py for the introspected commandline above,
    # merge outputs and concatenate to the header file created during
    # configuration
    add_custom_target(
      assemble-introspected-command-line
      # works on CMake 3.18 or newer > COMMAND ${CMAKE_COMMAND} -E cat
      # ${PROJECT_SOURCE_DIR}/testpy-output/*.command-line >
      # ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h
      COMMAND ${cat_command} ${PROJECT_SOURCE_DIR}/testpy-output/*.command-line
              > ${PROJECT_SOURCE_DIR}/doc/introspected-command-line.h 2> NULL
      DEPENDS run-introspected-command-line
    )

    # Create a target that updates the doxygen version
    add_custom_target(
      update_doxygen_version
      COMMAND ${PROJECT_SOURCE_DIR}/doc/ns3_html_theme/get_version.sh
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )

    # Create a doxygen target that builds the documentation and only runs
    # after the version target above was executed, the introspected doxygen 
    # and command line were extracted
    add_custom_target(
      doxygen
      COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/doxygen.conf
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS update_doxygen_version run-print-introspected-doxygen
              assemble-introspected-command-line
    )

    # Create a doxygen target that only needs to run the version target
    # which doesn't trigger compilation of examples neither the execution of test.py
    # nor print-introspected-doxygen
    add_custom_target(
      doxygen-no-build
      COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doc/doxygen.conf
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      DEPENDS update_doxygen_version
    )
  endif()


Project-wide compiler and linker flags
======================================

Different compilers and links accept different flags, which must be 
known during configuration time. Some of these flags are
handled directly by CMake:

.. sourcecode:: cmake

  # equivalent to -fPIC for libraries and -fPIE for executables
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # link-time optimization flags such as -flto and -flto=thin
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

  # C++ standard flag to use
  set(CMAKE_CXX_STANDARD_MINIMUM 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  add_library(static_lib STATIC) # equivalent to -static flag
  add_library(shared_lib SHARED) # equivalent to -shared flags

Other flags need to be handled manually and change based on 
the compiler used. The most commonly used are handled in 
``build-support/macros-and-definitions.cmake``.

.. sourcecode:: cmake

  set(LIB_AS_NEEDED_PRE)
  set(LIB_AS_NEEDED_POST)
  if(${GCC} AND NOT APPLE)
    # using GCC
    set(LIB_AS_NEEDED_PRE -Wl,--no-as-needed)
    set(LIB_AS_NEEDED_POST -Wl,--as-needed)
    set(LIB_AS_NEEDED_PRE_STATIC -Wl,--whole-archive,-Bstatic)
    set(LIB_AS_NEEDED_POST_STATIC -Wl,--no-whole-archive)
    set(LIB_AS_NEEDED_POST_STATIC_DYN -Wl,-Bdynamic,--no-whole-archive)
  endif()

The ``LIB_AS_NEEDED`` values are used to force linking of all symbols,
and not only those explicitly used by the applications, which is necessary
since simulation scripts don't directly use most of the symbols exported
by the modules. Their use can be seen in the ``utils/CMakeLists.txt``:

.. sourcecode:: cmake

  target_link_libraries(
      test-runner ${LIB_AS_NEEDED_PRE} ${ns3-libs-tests} ${LIB_AS_NEEDED_POST}
      ${ns3-libs} ${ns3-contrib-libs}
    )

This will ensure test-runner linking to ``ns3-libs-tests`` (list containing all
module test libraries) with all symbols, which will make it able to find and run
all tests. The other two lists ``ns3-libs`` (src modules) and ``ns3-contrib-libs``
(contrib modules) don't need to be completely linked since the tests libraries are
already linked to them.

Other project-wide compiler-dependent flags can be set during compiler checking.

.. sourcecode:: cmake

  # Check if the compiler is GCC
  set(GCC FALSE)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # Check if GCC is a supported version
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${GNU_MinVersion})
      message(
        FATAL_ERROR
          "GNU ${CMAKE_CXX_COMPILER_VERSION} ${below_minimum_msg} ${GNU_MinVersion}"
      )
    endif()
    # If GCC is up-to-date, set flag to true and continue
    set(GCC TRUE)

    # Disable semantic interposition
    add_definitions(-fno-semantic-interposition)
  endif()

.. _disables semantic interposition: https://gitlab.com/nsnam/ns-3-dev/-/merge_requests/777

The ``-fno-semantic-interposition`` flag `disables semantic interposition`_, which can 
reduce overhead of function calls in shared libraries built with ``-fPIC``.
This is the default behaviour for Clang. The inlined ns-3 calls will not be 
correctly interposed by the ``LD_PRELOAD`` trick, which is not know to be used by ns-3 users. 
To re-enable semantic interposition, comment out the line and reconfigure the project.

Note: the most common use of the ``LD_PRELOAD`` trick is to use custom memory allocators,
and this continues to work since the interposed symbols are from the standard libraries,
which are compiled with semantic interposition. 

Some modules may require special flags. The Openflow module for example 
may require ``-Wno-stringop-truncation`` flag to prevent an warning that 
is treated as error to prevent the compilation from proceeding. The flag
can be passed to the entire module with the following:

.. sourcecode:: cmake

  add_compile_options(-Wno-stringop-truncation)

  build_lib(
    LIBNAME openflow
    SOURCE_FILES
      helper/openflow-switch-helper.cc
      model/openflow-interface.cc
      model/openflow-switch-net-device.cc
    HEADER_FILES
      helper/openflow-switch-helper.h
      model/openflow-interface.h
      model/openflow-switch-net-device.h
    LIBRARIES_TO_LINK ${libinternet}
                      ${openflow_LIBRARIES}
    TEST_SOURCES test/openflow-switch-test-suite.cc
  )

If a flag prevents your compiler from compiling, wrap the flag inside a
compiler check. The currently checked compilers are ``GCC`` and ``CLANG`` 
(includes both upstream LLVM Clang and Apple Clang).

.. sourcecode:: cmake

  if(NOT ${FAILING_COMPILER})
    add_compile_options(-Wno-stringop-truncation)
  endif()

  # or 

  if(${ONLY_COMPILER_THAT_SUPPORTS_UNIQUE_FLAG})
    add_compile_options(-unique_flag)
  endif()

