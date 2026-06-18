.. include:: replace.txt

Organization
------------

This manual compiles documentation for |ns3| models and supporting
software that enable users to construct network simulations.
It is important to distinguish between **modules** and **models**:

* |ns3| software is organized into separate *modules* that are each
  built as a separate software library.  Individual ns-3 programs can link
  the modules (libraries) they need to conduct their simulation.

* |ns3| *models* are abstract representations of real-world objects,
  protocols, devices, etc.  

An |ns3| module may consist of more than one model (for instance, the
:mod:`internet` module contains models for both TCP and UDP).  In general,
ns-3 models do not span multiple software modules, however.  

This manual provides documentation about the models of |ns3|.  It 
complements two other sources of documentation concerning models:

* the model APIs are documented, from a programming perspective, using
  `Doxygen <http://www.doxygen.org>`_.  Doxygen for ns-3 models is available
  `on the project web server <http://www.nsnam.org/docs/doxygen/index.html>`_. 

* the |ns3| core is documented in the developer's manual.  |ns3| models make
  use of the facilities of the core, such as attributes, default values, 
  random numbers, test frameworks, etc.  Consult the 
  `main web site <http://www.nsnam.org>`_ to find copies of the manual.

Finally, additional documentation about various aspects of |ns3| may
exist on the `project wiki <http://www.nsnam.org/wiki>`_.

A sample outline of how to write model library documentation can be
found by executing the ``create-module.py`` program and looking at the
template created in the file ``new-module/doc/new-module.rst``. 

.. sourcecode:: bash

  $ cd src
  $ ./create-module.py new-module

The remainder of this document is organized alphabetically by module name.

If you are new to |ns3|, you might first want to read below about the network
module, which contains some fundamental models for the simulator.
The packet model, models for different address formats, and abstract 
base classes for objects such as nodes, net devices, channels, sockets, and 
applications are discussed there.


