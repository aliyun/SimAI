.. include:: replace.txt

Node and NetDevices Overview
----------------------------

This chapter describes how |ns3| nodes are put together, and provides a
walk-through of how packets traverse an internet-based Node.

.. _node-architecture:

.. figure:: figures/node.*

    High-level node architecture

In |ns3|, nodes are instances of :cpp:class:`ns3::Node`. This class may be
subclassed, but instead, the conceptual model is that we *aggregate* or insert
objects to it rather than define subclasses.

One might think of a bare |ns3| node as a shell of a computer, to which one may
add NetDevices (cards) and other innards including the protocols and
applications. :ref:`node-architecture` illustrates that :cpp:class:`ns3::Node`
objects contain a list of :cpp:class:`ns3::Application` instances (initially,
the list is empty), a list of :cpp:class:`ns3::NetDevice` instances (initially,
the list is empty), a list of :cpp:class:`ns3::Node::ProtocolHandler` instances,
a unique integer ID, and a system ID (for distributed simulation).

The design tries to avoid putting too many dependencies on the class
:cpp:class:`ns3::Node`, :cpp:class:`ns3::Application`, or
:cpp:class:`ns3::NetDevice` for the following:

* IP version, or whether IP is at all even used in the :cpp:class:`ns3::Node`.
* implementation details of the IP stack.

From a software perspective, the lower interface of applications corresponds to
the C-based sockets API. The upper interface of :cpp:class:`ns3::NetDevice`
objects corresponds to the device independent sublayer of the Linux stack.
Everything in between can be aggregated and plumbed together as needed.

Let's look more closely at the protocol demultiplexer. We want incoming frames
at layer-2 to be delivered to the right layer-3 protocol such as IPv4. The
function of this demultiplexer is to register callbacks for receiving packets.
The callbacks are indexed based on the `EtherType
<http://en.wikipedia.org/wiki/EtherType>`_ in the layer-2 frame.   

Many different types of higher-layer protocols may be connected to the
NetDevice, such as IPv4, IPv6, ARP, MPLS, IEEE 802.1x, and packet sockets.
Therefore, the use of a callback-based demultiplexer avoids the need to use a
common base class for all of these protocols, which is problematic because of
the different types of objects (including packet sockets) expected to be
registered there.
