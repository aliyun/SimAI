.. include:: replace.txt
.. highlight:: cpp


Organization
------------

This chapter describes the overall |ns3| software organization and the
corresponding organization of this manual.

|ns3| is a discrete-event network simulator in which the simulation core
and models are implemented in C++. |ns3| is built as a library which may be
statically or dynamically linked to a C++ main program that defines the
simulation topology and starts the simulator. |ns3| also exports nearly all
of its API to Python, allowing Python programs to import an "ns3" module in
much the same way as the |ns3| library is linked by executables in C++.  

.. _software-organization:

.. figure:: figures/software-organization.*
   
   Software organization of |ns3|

The source code for |ns3| is mostly organized in the ``src`` directory and
can be described by the diagram in :ref:`software-organization`. We will
work our way from the bottom up; in general, modules only have dependencies
on modules beneath them in the figure.

We first describe the core of the simulator; those components that are 
common across all protocol, hardware, and environmental models. 
The simulation core is implemented in ``src/core``. Packets are 
fundamental objects in a network simulator
and are implemented in ``src/network``. These two simulation modules by
themselves are intended to comprise a generic simulation core that can be
used by different kinds of networks, not just Internet-based networks.  The
above modules of |ns3| are independent of specific network and device
models, which are covered in subsequent parts of this manual.

In addition to the above |ns3| core, we introduce, also in the initial 
portion of the manual, two other modules that supplement the core C++-based 
API.  |ns3| programs may access
all of the API directly or may make use of a so-called *helper API* that
provides convenient wrappers or encapsulation of low-level API calls. The
fact that |ns3| programs can be written to two APIs (or a combination
thereof) is a fundamental aspect of the simulator.
We also describe how Python is supported in |ns3| before moving onto
specific models of relevance to network simulation.

The remainder of the manual is focused on documenting the models and
supporting capabilities.  The next part focuses on two fundamental objects in
|ns3|:  the ``Node`` and ``NetDevice``. Two special NetDevice types are
designed to support network emulation use cases, and emulation is described
next.  The following chapter is devoted to Internet-related models, 
including the
sockets API used by Internet applications. The next chapter covers 
applications, and the following chapter describes additional support for 
simulation, such as animators and statistics.

The project maintains a separate manual devoted to testing and validation
of |ns3| code (see the `ns-3 Testing and Validation manual
<http://www.nsnam.org/tutorials.html>`_).
