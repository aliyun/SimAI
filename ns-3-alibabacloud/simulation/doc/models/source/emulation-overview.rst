.. include:: replace.txt

Emulation Overview
------------------

|ns3| has been designed for integration into testbed and virtual machine
environments. We have addressed this need by providing two kinds of net devices.
The first kind of device is a file descriptor net device (``FdNetDevice``),
which is a generic device type that can read and write from a file descriptor.
By associating this file descriptor with different things on the host
system, different capabilities can be provided.  For instance, the
FdNetDevice can be associated with an underlying packet socket to provide
emulation capabilities.  This allows |ns3| simulations
to send data on a "real" network. The second kind, called a ``TapBridge``
``NetDevice`` allows a "real" host to participate in an |ns3| simulation as if
it were one of the simulated nodes. An |ns3| simulation may be constructed with
any combination of simulated or emulated devices.

**Note:** Prior to ns-3.17, the emulation capability was provided by a
special device called an ``Emu`` NetDevice; the ``Emu`` NetDevice has
been replaced by the ``FdNetDevice``.

One of the use-cases we want to support is that of a testbed. A concrete example
of an environment of this kind is the ORBIT testbed. ORBIT is a laboratory
emulator/field trial network arranged as a two dimensional grid of 400 802.11
radio nodes. We integrate with ORBIT by using their "imaging" process to load
and run |ns3| simulations on the ORBIT array. We can use our 
``EmuFdNetDevice``
to drive the hardware in the testbed and we can accumulate results either using
the |ns3| tracing and logging functions, or the native ORBIT data gathering
techniques. See `<http://www.orbit-lab.org/>`_ for details on the ORBIT
testbed.

A simulation of this kind is shown in the following figure:

.. _testbed:

.. figure:: figures/testbed.*
   
   Example Implementation of Testbed Emulation.

You can see that there are separate hosts, each running a subset of a "global"
simulation. Instead of an |ns3| channel connecting the hosts, we use real
hardware provided by the testbed. This allows |ns3| applications and protocol
stacks attached to a simulation node to communicate over real hardware.

We expect the primary use for this configuration will be to generate repeatable
experimental results in a real-world network environment that includes all of
the |ns3| tracing, logging, visualization and statistics gathering tools.

In what can be viewed as essentially an inverse configuration, we allow "real"
machines running native applications and protocol stacks to integrate with an
|ns3| simulation. This allows for the simulation of large networks connected to
a real machine, and also enables virtualization. A simulation of this kind is
shown in the following figure:

.. _emulated-channel:

.. figure:: figures/emulated-channel.*

    Implementation overview of emulated channel.

Here, you will see that there is a single host with a number of virtual machines
running on it. An |ns3| simulation is shown running in the virtual machine shown
in the center of the figure. This simulation has a number of nodes with
associated |ns3| applications and protocol stacks that are talking to an |ns3|
channel through native simulated |ns3| net devices.

There are also two virtual machines shown at the far left and far right of the
figure. These VMs are running native (Linux) applications and protocol stacks.
The VM is connected into the simulation by a Linux Tap net device. The user-mode
handler for the Tap device is instantiated in the simulation and attached to a
proxy node that represents the native VM in the simulation. These handlers allow
the Tap devices on the native VMs to behave as if they were |ns3| net devices in
the simulation VM. This, in turn, allows the native software and protocol suites
in the native VMs to believe that they are connected to the simulated |ns3|
channel.

We expect the typical use case for this environment will be to analyze the
behavior of native applications and protocol suites in the presence of large
simulated |ns3| networks.

The basic testbed mode of emulation uses raw sockets.  Two other variants
(netmap-based and DPDK-based emulation) have been recently added; these
make use of more recent network interface cards that make use of
directly-mapped memory capabilities to improve packet processing efficiency.

For more details:

.. toctree::

   fd-net-device
   netmap-net-device
   dpdk-net-device
   tap
