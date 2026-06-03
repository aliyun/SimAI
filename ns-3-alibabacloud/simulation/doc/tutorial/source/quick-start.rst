.. include:: replace.txt
.. highlight:: bash

.. _Quick Start:

Quick Start
-----------

This section is optional, for readers who want to get |ns3| up and running
as quickly as possible.  Readers may skip forward to the :ref:`Introduction`
chapter, followed by the :ref:`Getting Started` chapter, for a lengthier
coverage of all of this material.

Brief Summary
*************
|ns3| is a discrete-event simulator typically run from the command line.
It is written directly in C++, not in a high-level modeling language;
simulation events are simply C++ function calls, organized by a scheduler.

An |ns3| user will obtain the |ns3| source code (see below), 
compile it into shared (or static) libraries, and link the libraries to 
`main()` programs that he or she authors.  The `main()` program is where
the specific simulation scenario configuration is performed and where the 
simulator is run and stopped.  Several example programs are provided, which
can be modified or copied to create new simulation scenarios.  Users also
often edit the |ns3| library code (and rebuild the libraries) to change
its behavior.

|ns3| has optional Python bindings for authoring scenario configuration
programs in Python (and using a Python-based workflow); this quick start
does not cover those aspects.

Prerequisites
*************
|ns3| has various optional extensions, but the main features just require
a C++ compiler (g++ or clang++), Python (version 3.6 or above), CMake and
a build-system (e.g. make, ninja, Xcode).
We focus in this chapter only on getting |ns3| up and running on a system
supported by a recent C++ compiler and Python runtime support.

For Linux, use either g++ or clang++ compilers.  For macOS, use clang++ 
(available in Xcode or Xcode Command Line Tools).  For Windows, we recommend
to either use a Linux virtual machine, or the Windows Subsystem for Linux. 

Downloading ns-3
****************

|ns3| is distributed in source code only (some binary packages exist but
they are not maintained by the open source project).  There are two
main ways to obtain the source code:  1) downloading the latest release
as a source code archive from the main |ns3| web site, or 2) cloning
the Git repository from GitLab.com.  These two options are described next;
either one or the other download option (but not both) should be followed.

Downloading the Latest Release
++++++++++++++++++++++++++++++

1) Download the latest release from https://www.nsnam.org/releases/latest

2) Unpack it in a working directory of your choice.

   ::

    $ tar xjf ns-allinone-3.36.tar.bz2

3) Change into the |ns3| directory directly; e.g.

   ::

    $ cd ns-allinone-3.36/ns-3.36

The ns-allinone directory has some additional components but we are skipping
over them here; one can work directly from the |ns3| source code directory.
The rest of the tutorial describes the additional components.

Cloning ns-3 from GitLab.com
++++++++++++++++++++++++++++

You can perform a Git clone in the usual way:

::

  $ git clone https://gitlab.com/nsnam/ns-3-dev.git

If you are content to work with the tip of the development tree; you need
only to `cd` into ns-3-dev; the `master` branch is checked out by default.

::

  $ cd ns-3-dev

If instead you want to try the most recent release (version 3.36 as of this
writing), you can checkout a branch corresponding to that git tag:

::

  $ git checkout -b ns-3.36-branch ns-3.36

Building and testing ns-3
*************************

Once you have obtained the source either by downloading a release or by
cloning a Git repository, the next step is to
configure the build using the *CMake* build system.  The below
commands make use of a Python wrapper around CMake, called ``ns3``, that
simplifies the command-line syntax, resembling *Waf* syntax.  There
are several options to control the build, but enabling the example programs
and the tests, for a default build profile (with asserts enabled and
and support for |ns3| logging) is what is usually done at first:

::

  $ ./ns3 configure --enable-examples --enable-tests

Then, use ``ns3`` to build |ns3|:

::

  $ ./ns3 build

Once complete, you can run the unit tests to check your build:

::

  $ ./test.py

All tests should either PASS or be SKIPped.  At this point, you have a
working |ns3| simulator.  From here, you can start to
run programs (look in the examples directory).  To run the first tutorial
program, whose source code is located at `examples/tutorial/first.cc`, 
use ns3 to run it (by doing so, the |ns3| shared libraries are found
automatically):

::

  $ ./ns3 run first

To view possible command-line options, specify the `--PrintHelp` argument:

::

  $ ./ns3 run 'first --PrintHelp'

To continue reading about the conceptual model and architecture of |ns3|,
the tutorial chapter :ref:`Conceptual Overview` would be the next natural place
to skip to, or you can learn more about the project and the
various build options by continuing directly with the :ref:`Introduction`
and :ref:`Getting Started` chapters.
