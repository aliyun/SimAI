Netmap NetDevice
----------------
.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

The ``fd-net-device`` module provides the ``NetmapNetDevice`` class, a class derived
from the ``FdNetDevice`` which is able to read and write traffic using a netmap file descriptor.
This netmap file descriptor must be associated to a real ethernet device in the host machine.
The ``NetmapNetDeviceHelper`` class supports the configuration of a ``NetmapNetDevice``.

netmap is a fast packet processing capability that bypasses the
host networking stack and gains direct access to network device.
netmap was developed by Luigi Rizzo [Rizzo2012]_ and is maintained as
an open source project on GitHub at https://github.com/luigirizzo/netmap.

The ``NetmapNetDevice`` for |ns3| [Imputato2019]_ was developed by Pasquale Imputato in the 2017-19 timeframe.  The use of NetmapNetDevice requires that the
host system has netmap support (and for best performance, the drivers
must support netmap and must be using a netmap-enabled device driver).  Users
can expect that emulation support using Netmap will support higher packets
per second than emulation using FdNetDevice with raw sockets (which pass
through the Linux networking kernel).

.. [Rizzo2012] Luigi Rizzo, "netmap: A Novel Framework for Fast Packet I/O", Proceedings of 2012 USENIX Annual Techincal Conference, June 2012.

.. [Imputato2019] Pasquale Imputato, Stefano Avallone, Enhancing the fidelity of network emulation through direct access to device buffers, Journal of Network and Computer Applications, Volume 130, 2019, Pages 63-75, (http://www.sciencedirect.com/science/article/pii/S1084804519300220)

Model Description
*****************

Design
======

Because netmap uses file descriptor based communication to interact with the
real device, the straightforward approach to design a new ``NetDevice`` around
netmap is to have it inherit from the existing ``FdNetDevice`` and implement
a specialized version of the operations specific to netmap.
The operations that require a specialized implementation are the 
initialization, because the NIC has to be put in netmap mode, and the 
read/write methods, which have to make use of the netmap API to coordinate
the exchange of packets with the netmap rings.

In the initialization stage, the network device is switched to netmap mode,
so that |ns3| is able to send/receive packets to/from the
real network device by writing/reading them to/from the netmap rings.
Following the design of the ``FdNetDevice``, a separate reading thread is
started during the initialization. The task of the reading thread is 
to wait for new incoming packets in the netmap receiver rings, in order
to schedule the events of packet reception. In
the initialization of the ``NetmapNetDevice``, an additional thread, 
the sync thread, is started. The sync thread is required because, in order
to reduce the cost of the system calls, netmap does not automatically 
transfer a packet written to a slot of the netmap ring to the transmission
ring or to the installed qdisc. It is up to the user process to
periodically request a synchronization of the netmap ring. Therefore,
the purpose of the sync thread is to periodically make a TXSYNC ioctl
request, so that pending packets in the netmap ring are transferred to
the transmission ring, if in native mode, or to the installed qdisc, if in
generic mode. Also, as described further below, the
sync thread is exploited to perform flow control and notify the BQL library
about the
amount of bytes that have been transferred to the network device.

The read method is called by the reading thread to retrieve new incoming
packets stored in the netmap receiver ring and pass them to the appropriate
|ns3| protocol handler for further processing within the simulator’s network
stack. After retrieving packets, the reading thread also synchronizes
the netmap receiver ring, so that the retrieved packets can be removed
from the netmap receiver ring.

The ``NetmapNetDevice`` also specializes the write method, i.e., the method
used to transmit a packet received from the upper layer (the |ns3| traffic
control layer).  The write method uses the netmap API to write the packet to a
free slot in the netmap
transmission ring. After writing a packet, the write method checks whether 
there is enough room in the netmap transmission ring for another packet. 
If not, the ``NetmapNetDevice`` stops its queue so that the |ns3| traffic
control layer does not attempt to send a packet that could not be stored in 
the netmap transmission ring.

A stopped ``NetmapNetDevice`` queue needs to be restarted as soon as some 
room is made in the netmap transmission ring. The sync thread can be exploited
for this purpose, given that it periodically synchronizes the netmap
transmission ring. In particular, the sync thread also checks the number of
free slots in the netmap transmission ring in case the ``NetmapNetDevice``
queue is stopped.  If the number of free slots exceeds a configurable value,
the sync thread restarts the ``NetmapNetDevice``
queue and wakes the associated |ns3| qdisc. The NetmapNetDevice also supports
BQL: the write method notifies the BQL library of the amount of bytes that
have been written to the netmap transmission ring, while the sync thread
notifies the BQL library of the amount of bytes that have been removed from
the netmap transmission ring and transferred to the NIC since the previous 
notification.

Scope and Limitations
=====================
The main scope of ``NetmapNetDevice`` is to support the flow-control between
the physical device and the upper layer and using at best the computational
resources to process packets.  However, the (Linux) system and network
device must support netmap to make use of this feature.

Usage
*****

The installation of netmap itself on a host machine is out of scope for
this document.  Refer to the `netmap GitHub README <https://github.com/luigirizzo/netmap>`_ for instructions.

The |ns3| netmap code has only been tested on Linux; it is not clear whether
other operating systems can be supported.

If |ns3| is able to detect the presence of netmap on the system, it will
report that:

.. sourcecode:: text

  Netmap emulation FdNetDevice  : not enabled

If not, it will report:

.. sourcecode:: text 

  Netmap emulation FdNetDevice  : not enabled (needs net/netmap_user.h)

To run FdNetDevice-enabled simulations, one must pass the ``--enable-sudo``
option to ``./ns3 configure``, or else run the simulations with root
privileges.

Helpers
=======

|ns3| netmap support uses a ``NetMapNetDeviceHelper`` helper object to
install the ``NetmapNetDevice``.  In other respects, the API and use is similar
to that of the ``EmuFdNetDeviceHelper``.

Attributes
==========

There is one attribute specialized to ``NetmapNetDevice``, named
``SyncAndNotifyQueuePeriod``.  This value takes an integer number of
microseconds, and is used as the period of time after which the device
syncs the netmap ring and notifies queue status.  The value should be
close to the interrupt coalescence period of the real device.  Users
may want to tune this parameter for their own system; it should be 
a compromise between CPU usage and accuracy in the ring sync (if it is
too high, the device goes into starvation and lower throughput occurs).

Output
======

The ``NetmapNetDevice`` does not provide any specialized output, but
supports the ``FdNetDevice`` output and traces (such as a promiscuous sniffer
trace).

Examples
========

Several examples are provided:

* ``fd-emu-onoff.cc``: This example is aimed at measuring the throughput of the
  ``NetmapNetDevice`` when using the ``NetmapNetDeviceHelper`` to attach the
  simulated device to a real device in the host machine. This is achieved
  by saturating the channel with TCP or UDP traffic.
* ``fd-emu-ping.cc``: This example uses the ``NetmapNetDevice`` to send ICMP
  traffic over a real device.
* ``fd-emu-tc.cc``: This example configures a router on a machine with two
   interfaces in emulated mode through netmap. The aim is to explore different
   qdiscs behaviours on the backlog of a device emulated bottleneck side.
* ``fd-emu-send.cc``: This example builds a node with a device in 
  emulation mode through netmap.  The aim is to measure the maximum transmit
  rate in packets per second (pps) achievable with ``NetmapNetDevice`` on 
  a specific machine.

Note that all the examples run in emulation mode through netmap (with 
``NetmapNetDevice``) and raw socket (with ``FdNetDevice``).

