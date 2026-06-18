Traffic Control Layer
---------------------

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

The Traffic Control layer aims at introducing an equivalent of the Linux Traffic
Control infrastructure into ns-3. The Traffic Control layer sits in between
the NetDevices (L2) and any network protocol (e.g. IP). It is in charge of processing
packets and performing actions on them: scheduling, dropping, marking, policing, etc.


Introducing the Traffic Control Layer
*************************************

The Traffic Control layer intercepts both outgoing packets flowing downwards from
the network layer to the network device and incoming packets flowing in the opposite
direction. Currently, only outgoing packets are processed by the Traffic Control layer.
In particular, outgoing packets are enqueued in a queuing discipline, which can perform
multiple actions on them.

In the following, more details are given about how the Traffic Control layer intercepts
outgoing and incoming packets and, more in general, about how the packets traverse the
network stack.

Transmitting packets
====================

The IPv{4,6} interfaces uses the aggregated object TrafficControlLayer to send
down packets, instead of calling NetDevice::Send() directly. After the analysis
and the process of the packet, when the backpressure mechanism allows it,
TrafficControlLayer will call the Send() method on the right NetDevice.

Receiving packets
=================

The callback chain that (in the past) involved IPv{4,6}L3Protocol and NetDevices,
through ReceiveCallback, is extended to involve TrafficControlLayer. When an
IPv{4,6}Interface is added in the IPv{4,6}L3Protocol, the callback chain is
configured to have the following packet exchange:

NetDevice --> Node --> TrafficControlLayer --> IPv{4,6}L3Protocol

Brief description of old node/device/protocol interactions
**************************************************************

The main question that we would like to answer in the following paragraphs is:
how a ns-3 node can send/receive packets?

If we analyze any example out there, the ability of the node to receive/transmit
packets derives from the interaction of two helper:

* L2 Helper (something derived from NetDevice)
* L3 Helper (usually from Internet module)

L2 Helper main operations
=========================

Any good L2 Helper will do the following operations:

* Create n netdevices (n>1)
* Attach a channel between these devices
* Call Node::AddDevice ()

Obviously the last point is the most important.

Node::AddDevice (network/model/node.cc:128) assigns an interface index to the
device, calls NetDevice::SetNode, sets the receive callback of the device to
Node::NonPromiscReceiveFromDevice. Then, it schedules NetDevice::Initialize() method at
Seconds(0.0), then notify the registered DeviceAdditionListener handlers (not used BY ANYONE).

Node::NonPromiscReceiveFromDevice calls Node::ReceiveFromDevice.

Node::ReceiveFromDevice iterates through ProtocolHandlers, which are callbacks
which accept as signature:

ProtocolHandler (Ptr<NetDevice>, Ptr<const Packet>, protocol, from_addr, to_addr, packetType).

If device, protocol number and promiscuous flag corresponds, the handler is
invoked.

Who is responsible to set ProtocolHandler ? We will analyze that in the next
section.

L3 Helper
=========

We have only internet which provides network protocol (IP). That module splits
the operations between two helpers: InternetStackHelper and Ipv{4,6}AddressHelper.

InternetStackHelper::Install (internet/helper/internet-stack-helper.cc:423)
creates and aggregates protocols {ArpL3,Ipv4L3,Icmpv4}Protocol. It creates the
routing protocol, and if Ipv6 is enabled it adds {Ipv6L3,Icmpv6L4}Protocol. In
any case, it instantiates and aggregates an UdpL4Protocol object, along with a
PacketSocketFactory.
Ultimately, it creates the required objects and aggregates them to the node.

Let's assume an Ipv4 environment (things are the same for Ipv6).

Ipv4AddressHelper::Assign (src/internet/helper/ipv4-address-helper.cc:131)
registers the handlers. The process is a bit long. The method is called with
a list of NetDevice. For each of them, the node and Ipv4L3Protocol pointers are
retrieved; if an Ipv4Interface is already registered for the device, on that the
address is set. Otherwise, the method Ipv4L3Protocol::AddInterface is called,
before adding the address.

IP interfaces
=============

In Ipv4L3Protocol::AddInterface (src/internet/model/ipv4-l3-protocol.cc:300)
two protocol handlers are installed: one that react to ipv4 protocol number,
and one that react to arp protocol number (Ipv4L3Protocol::Receive and
ArpL3Protocol::Receive, respectively). The interface is then created,
initialized, and returned.

Ipv4L3Protocol::Receive (src/internet/model/ipv4-l3-protocol.cc:472) iterates
through the interface. Once it finds the Ipv4Interface which has the same device
as the one passed as argument, invokes the rxTrace callback. If the interface is
down, the packet is dropped. Then, it removes the header and trim any residual
frame padding. If checksum is not OK, it drops the packet. Otherwise, forward
the packet to the raw sockets (not used). Then, it ask the routing protocol what
is the destiny of that packet. The choices are: Ipv4L3Protocol::{IpForward,
IpMulticastForward,LocalDeliver,RouteInputError}.

