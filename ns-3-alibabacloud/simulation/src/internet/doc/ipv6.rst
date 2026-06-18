.. include:: replace.txt
.. highlight:: cpp

IPv6
----

This chapter describes the |ns3| IPv6 model capabilities and limitations along with its
usage and examples.

IPv6 model description
**********************

The IPv6 model is loosely patterned after the Linux implementation;
the implementation is not complete as some features of IPv6 are not of
much interest to simulation studies, and some features of IPv6 are simply
not modeled yet in |ns3|.

The base class :cpp:class:`Ipv6` 
defines a generic API, while the class :cpp:class:`Ipv6L3Protocol` is the actual class
implementing the protocol. The actual classes used by the IPv6 stack are located mainly 
in the directory ``src/internet``. 

The implementation of IPv6 is contained in the following files:

.. sourcecode:: text

    src/internet/model/icmpv6-header.{cc,h}
    src/internet/model/icmpv6-l4-protocol.{cc,h}
    src/internet/model/ipv6.{cc,h}
    src/internet/model/ipv6-address-generator.{cc,h}
    src/internet/model/ipv6-autoconfigured-prefix.{cc,h}
    src/internet/model/ipv6-end-point.{cc,h}
    src/internet/model/ipv6-end-point-demux.{cc,h}
    src/internet/model/ipv6-extension.{cc,h}
    src/internet/model/ipv6-extension-demux.{cc,h}
    src/internet/model/ipv6-extension-header.{cc,h}
    src/internet/model/ipv6-header.{cc,h}
    src/internet/model/ipv6-interface.{cc,h}
    src/internet/model/ipv6-interface-address.{cc,h}
    src/internet/model/ipv6-l3-protocol.{cc,h}
    src/internet/model/ipv6-list-routing.{cc,h}
    src/internet/model/ipv6-option.{cc,h}
    src/internet/model/ipv6-option-demux.{cc,h}
    src/internet/model/ipv6-option-header.{cc,h}
    src/internet/model/ipv6-packet-info-tag.{cc,h}
    src/internet/model/ipv6-pmtu-cache.{cc,h}
    src/internet/model/ipv6-raw-socket-factory.{cc,h}
    src/internet/model/ipv6-raw-socket-factory-impl.{cc,h}
    src/internet/model/ipv6-raw-socket-impl.{cc,h}
    src/internet/model/ipv6-route.{cc,h}
    src/internet/model/ipv6-routing-protocol.{cc,h}
    src/internet/model/ipv6-routing-table-entry.{cc,h}
    src/internet/model/ipv6-static-routing.{cc,h}
    src/internet/model/ndisc-cache.{cc,h}
    src/network/utils/inet6-socket-address.{cc,h}
    src/network/utils/ipv6-address.{cc,h}

Also some helpers are involved with IPv6:

.. sourcecode:: text

    src/internet/helper/internet-stack-helper.{cc,h}
    src/internet/helper/ipv6-address-helper.{cc,h}
    src/internet/helper/ipv6-interface-container.{cc,h}
    src/internet/helper/ipv6-list-routing-helper.{cc,h}
    src/internet/helper/ipv6-routing-helper.{cc,h}
    src/internet/helper/ipv6-static-routing-helper.{cc,h}

The model files can be roughly divided into:

* protocol models (e.g., ipv6, ipv6-l3-protocol, icmpv6-l4-protocol, etc.)
* routing models (i.e., anything with 'routing' in its name)
* sockets and interfaces (e.g., ipv6-raw-socket, ipv6-interface, ipv6-end-point, etc.)
* address-related things
* headers, option headers, extension headers, etc.
* accessory classes (e.g., ndisc-cache)


Usage
*****

The following description is based on using the typical helpers found in the example code.

IPv6 does not need to be activated in a node. it is automatically added to the 
available protocols once the Internet Stack is installed.

In order to *not* install IPv6 along with IPv4, it is possible to use
:cpp:class:`ns3::InternetStackHelper` method `SetIpv6StackInstall (bool enable)` 
before installing the InternetStack in the nodes.

Note that to have an IPv6-only network (i.e., to not install the IPv4 stack in a node)
one should use :cpp:class:`ns3::InternetStackHelper` method `SetIpv4StackInstall (bool enable)` 
before the stack installation.

As an example, in the following code node 0 will have both IPv4 and IPv6, node 1 only 
only IPv6 and node 2 only IPv4:

::

    NodeContainer n;
    n.Create (3);

    InternetStackHelper internet;
    InternetStackHelper internetV4only;
    InternetStackHelper internetV6only;

    internetV4only.SetIpv6StackInstall (false);
    internetV6only.SetIpv4StackInstall (false);
    
    internet.Install (n.Get (0));
    internetV6only.Install (n.Get (1));
    internetV4only.Install (n.Get (2));
   


IPv6 addresses assignment
=========================

In order to use IPv6 on a network, the first thing to do is assigning IPv6 addresses.

Any IPv6-enabled |ns3| node will have at least one NetDevice: the :cpp:class:`ns3::LoopbackNetDevice`.
The loopback device address is ``::1``.
All the other NetDevices will have one or more IPv6 addresses:

* One link-local address: ``fe80::interface ID``, where ``interface ID`` is derived from the NetDevice MAC address.
* Zero or more global addresses, e.g., ``2001:db8::1``.

Typically the first address on an interface will be the link-local one, with the global 
address(es) being the following ones.

IPv6 global addresses might be:

* manually assigned
* auto-generated

|ns3| can use both methods, and it's quite important to understand the implications of both.

Manually assigned IPv6 addresses
++++++++++++++++++++++++++++++++

This is probably the easiest and most used method. As an example:

::

    Ptr<Node> n0 = CreateObject<Node> ();
    Ptr<Node> n1 = CreateObject<Node> ();
    NodeContainer net (n0, n1);
    CsmaHelper csma;
    NetDeviceContainer ndc = csma.Install (net); 
    
    NS_LOG_INFO ("Assign IPv6 Addresses.");
    Ipv6AddressHelper ipv6;
    ipv6.SetBase (Ipv6Address ("2001:db8::"), Ipv6Prefix (64));
    Ipv6InterfaceContainer ic = ipv6.Assign (ndc);

This method will add two global IPv6 addresses to the nodes. Note that, as usual for IPv6,
all the nodes will also have a link-local address. Typically the first address on an 
interface will be the link-local one, with the global address(es) being the following ones.

Note that the global addresses will be derived from the MAC address. As a consequence, expect
to have addresses similar to ``2001:db8::200:ff:fe00:1``.

It is possible to repeat the above to assign more than one global address to a node.
However, due to the :cpp:class:`Ipv6AddressHelper` singleton nature, one should first assign all the
addresses of a network, then change the network base (``SetBase``), then do a new assignment.

Alternatively, it is possible to assign a specific address to a node:

::

    Ptr<Node> n0 = CreateObject<Node> ();
    NodeContainer net (n0);
    CsmaHelper csma;
    NetDeviceContainer ndc = csma.Install (net); 

    NS_LOG_INFO ("Specifically Assign an IPv6 Address.");
    Ipv6AddressHelper ipv6;
    Ptr<NetDevice> device = ndc.Get (0);
    Ptr<Node> node = device->GetNode ();
    Ptr<Ipv6> ipv6proto = node->GetObject<Ipv6> ();
    int32_t ifIndex = 0;
    ifIndex = ipv6proto->GetInterfaceForDevice (device);
    Ipv6InterfaceAddress ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:db8:f00d:cafe::42"), Ipv6Prefix (64));
    ipv6proto->AddAddress (ifIndex, ipv6Addr);


Auto-generated IPv6 addresses
+++++++++++++++++++++++++++++

This is accomplished by relying on the RADVD protocol, implemented by the class
:cpp:class:`Radvd`. A helper class is available, which can be used to ease the most
common tasks, e.g., setting up a prefix on an interface, if it is announced periodically,
and if the router is the default router for that interface.

A fine grain configuration is possible though the :cpp:class:`RadvdInterface` class, which
allows to setup every parameter of the announced router advertisement on a given interface.

It is worth mentioning that the configurations must be set up before installing the 
application in the node.

Upon using this method, the nodes will acquire dynamically (i.e., during the simulation)
one (or more) global address(es) according to the RADVD configuration. 
These addresses will be bases on the RADVD announced prefix and the node's EUI-64.

Examples of RADVD use are shown in ``examples/ipv6/radvd.cc`` 
and ``examples/ipv6/radvd-two-prefix.cc``.

Note that the router (i.e., the node with :cpp:class:`Radvd`) will have to have a global address,
while the nodes using the auto-generated addresses (SLAAC) will have to have a link-local address.
This is accomplished using :cpp:class:`Ipv6AddressHelper::AssignWithoutAddress`, e.g.:

::

    Ipv6AddressHelper ipv6;
    NetDeviceContainer tmp;
    tmp.Add (d1.Get (0)); /* n0 */
    Ipv6InterfaceContainer iic1 = ipv6.AssignWithoutAddress (tmp); /* n0 interface */


Random-generated IPv6 addresses
###############################

While IPv6 real nodes will use randomly generated addresses to protect privacy, |ns3|
does NOT have this capability. This feature haven't been so far considered as interesting
for simulation.


Networks with and without the onlink property
+++++++++++++++++++++++++++++++++++++++++++++

IPv6 adds the concept of "on-link" for addresses and prefixes. Simplifying the concept, a
network with the on-link property behaves roughly as IPv4: a node will assume that any address
belonging to the on-link prefix is reachable on the link, so it uses Neighbor Discovery Protocol (NDP)
to find the MAC address corresponding to the IPv6 address.
If the prefix is not marked as "on-link", then any packet is sent to the default router.

:cpp:class:`Radvd` can announce prefixes that have the on-link flag not set. Moreover, it is possible
to add an address to a node without setting the on-link property for the prefix used in the address.
The function to use is :cpp:class:`Ipv6AddressHelper::AssignWithoutOnLink`.


Duplicate Address Detection (DAD)
+++++++++++++++++++++++++++++++++

Nodes will perform DAD (it can be disabled using an :cpp:class:`Icmpv6L4Protocol` attribute).
Upon receiving a DAD, however, nodes will not react to it. As is: DAD reaction is 
incomplete so far. 
The main reason relies on the missing random-generated address capability. Moreover, 
since |ns3| nodes will usually be well-behaving, there shouldn't be any Duplicate Address.
This might be changed in the future, so as to avoid issues with real-world 
integrated simulations.

Explicit Congestion Notification (ECN) bits in IPv6
===================================================

- In IPv6, ECN bits are the last 2 bits of the Traffic class and occupy 10th and 11th bit
  in the header.

- The IPv6 header class defines an EcnType enum with all four ECN codepoints
  (ECN_NotECT, ECN_ECT1, ECN_ECT0, ECN_CE) mentioned
  in RFC 3168, and also a setter and getter method to handle ECN values in
  the Traffic Class field.

Ipv6QueueDiscItem
*****************

The traffic control sublayer in |ns3| handles objects of class
``QueueDiscItem`` which are used to hold an ns3::Packet and an ns3::Header.
This is done to facilitate the marking of packets for Explicit
Congestion Notification.  The ``Mark ()`` method is implemented in
Ipv6QueueDiscItem. It returns true if marking the packet is successful, i.e.,
it successfully sets the CE bit in the IPv6 header. The ``Mark ()`` method
will return false, however, if the IPv6 header indicates the ``ECN_NotECT``
codepoint.


Host and Router behaviour in IPv6 and |ns3|
===========================================

In IPv6 there is a clear distinction between *routers* and *hosts*. As one might expect,
routers can forward packets from an interface to another interface, while hosts drop
packets not directed to them.

Unfortunately, forwarding is not the only thing affected by this distinction, and forwarding
itself might be fine-tuned, e.g., to forward packets incoming from an interface and drop 
packets from another interface.

In |ns3| a node is configured to be an *host* by default. There are two main ways to change 
this behaviour:

* Using :cpp:class:`ns3::Ipv6InterfaceContainer` `SetForwarding(uint32_t i, bool router)` where ``i`` is the interface index in the container.
* Changing the :cpp:class:`ns3::Ipv6` attribute ``IpForward``.

Either one can be used during the simulation.

A fine-grained setup can be accomplished by using :cpp:class:`ns3::Ipv6Interface` `SetForwarding (bool forward);`
which allows to change the behaviour on a per-interface-basis.

Note that the node-wide configuration only serves as a convenient method to enable/disable the 
:cpp:class:`ns3::Ipv6Interface` specific setting. An Ipv6Interface added to a node
with forwarding enabled will be set to be forwarding as well.
This is really important when a node has interfaces added during the simulation. 

According to the :cpp:class:`ns3::Ipv6Interface` forwarding state, the following happens:

* Forwarding OFF

 * The node will NOT reply to Router Solicitation
 * The node will react to Router Advertisement
 * The node will periodically send Router Solicitation
 * Routing protocols MUST DROP packets not directed to the node

* Forwarding ON

 * The node will reply to Router Solicitation
 * The node will NOT react to Router Advertisement
 * The node will NOT send Router Solicitation
 * Routing protocols MUST forward packets

The behaviour is matching ip-sysctl.txt (http://www.kernel.org/doc/Documentation/networking/ip-sysctl.txt)
with the difference that it's not possible to override the behaviour using esoteric settings 
(e.g., forwarding but accept router advertisements, accept_ra=2, or forwarding but send router solicitations
forwarding=2).

Consider carefully the implications of packet forwarding. As an example, a node will NOT send
ICMPv6 PACKET_TOO_BIG messages from an interface with forwarding off. This is completely normal,
as the Routing protocol will drop the packet before attempting to forward it.


Helpers
=======

Typically the helpers used in IPv6 setup are:

* :cpp:class:`ns3::InternetStackHelper`
* :cpp:class:`ns3::Ipv6AddressHelper`
* :cpp:class:`ns3::Ipv6InterfaceContainer`

The use is almost identical to the corresponding IPv4 case, e.g.:

::

    NodeContainer n;
    n.Create (4);
    
    NS_LOG_INFO ("Create IPv6 Internet Stack");
    InternetStackHelper internetv6;
    internetv6.Install (n);
    
    NS_LOG_INFO ("Create channels.");
    CsmaHelper csma;
    NetDeviceContainer d = csma.Install (n);
    
    NS_LOG_INFO ("Create networks and assign IPv6 Addresses.");
    Ipv6AddressHelper ipv6;
    ipv6.SetBase (Ipv6Address ("2001:db8::"), Ipv6Prefix (64));
    Ipv6InterfaceContainer iic = ipv6.Assign (d);

Additionally, a common task is to enable forwarding on one of the nodes and to
setup a default route toward it in the other nodes, e.g.:

::

    iic.SetForwarding (0, true);
    iic.SetDefaultRouteInAllNodes (0);

This will enable forwarding on the node *0* and will setup a default route in 
:cpp:class:`ns3::Ipv6StaticRouting` on all the other nodes. Note that this 
requires that Ipv6StaticRouting is present in the nodes. 

The IPv6 routing helpers enable the user to perform specific tasks on the
particular routing algorithm and to print the routing tables. 


Attributes
==========

Many classes in the |ns3| IPv6 implementation contain attributes. The most
useful ones are:

* :cpp:class:`ns3::Ipv6`
 
 * `IpForward`, boolean, default false. Globally enable or disable IP forwarding for all current and future IPv6 devices.
 * `MtuDiscover`, boolean, default true. If disabled, every interface will have its MTU set to 1280 bytes.

* :cpp:class:`ns3::Ipv6L3Protocol`

 * `DefaultTtl`, uint8_t, default 64. The TTL value set by default on all outgoing packets generated on this node.
 * `SendIcmpv6Redirect`, boolean, default true. Send the ICMPv6 Redirect when appropriate.
 
* :cpp:class:`ns3::Icmpv6L4Protocol`

 * `DAD`, boolean, default true. Always do DAD (Duplicate Address Detection) check.
 
* :cpp:class:`ns3::NdiscCache`

 * `UnresolvedQueueSize`, uint32_t, default 3. Size of the queue for packets pending an NA reply.
 
Output
======

The IPv6 stack provides some useful trace sources:

* :cpp:class:`ns3::Ipv6L3Protocol`

 * `Tx`, Send IPv6 packet to outgoing interface.
 * `Rx`, Receive IPv6 packet from incoming interface.
 * `Drop`, Drop IPv6 packet.

* :cpp:class:`ns3::Ipv6Extension`

 * `Drop`, Drop IPv6 packet.

The latest trace source is generated when a packet contains an unknown option blocking its processing.

Mind that :cpp:class:`ns3::NdiscCache` could drop packets as well, and they are not logged 
in a trace source (yet). This might generate some confusion in the sent/received packets counters.

Advanced Usage
==============

IPv6 maximum transmission unit (MTU) and fragmentation
++++++++++++++++++++++++++++++++++++++++++++++++++++++

|ns3| NetDevices define the MTU according to the L2 simulated Device. IPv6 requires
that the minimum MTU is 1280 bytes, so all NetDevices are required to support at least
this MTU. This is the link-MTU.

In order to support different MTUs in a source-destination path, |ns3| IPv6 model can
perform fragmentation.
This can be either triggered by receiving a packet bigger than the link-MTU from the L4
protocols (UDP, TCP, etc.), or by receiving an ICMPv6 PACKET_TOO_BIG message.
The model mimics RFC 1981, with the following notable exceptions:

* L4 protocols are not informed of the Path MTU change
* TCP can not change its Segment Size according to the Path-MTU.

Both limitations are going to be removed in due time.

The Path-MTU cache is currently based on the source-destination IPv6 addresses. Further
classifications (e.g., flow label) are possible but not yet implemented.

The Path-MTU default validity time is 10 minutes. After the cache entry expiration, the
Path-MTU information is removed and the next packet will (eventually) trigger a new ICMPv6
PACKET_TOO_BIG message.
Note that 1) this is consistent with the RFC specification and 2) L4 protocols are 
responsible for retransmitting the packets.

Examples
========

The examples for IPv6 are in the directory ``examples/ipv6``. These examples focus on
the most interesting IPv6 peculiarities, such as fragmentation, redirect and so on.

Moreover, most TCP and UDP examples located in ``examples/udp``, ``examples/tcp``, etc. 
have a command-line option to use IPv6 instead of IPv4.

Troubleshooting
===============

There are just a few pitfalls to avoid while using |ns3| IPv6.

Routing loops
+++++++++++++

Since the only (so far) routing scheme available for IPv6 is :cpp:class:`ns3::Ipv6StaticRouting`,
default router have to be setup manually. When there are two or more routers in a network 
(e.g., node A and node B), avoid using the helper function `SetDefaultRouteInAllNodes` 
for more than one router.

The consequence would be to install a default route to B in A and a default route pointing to 
A in B, generating a loop. 

Global address leakage
++++++++++++++++++++++

Remember that addresses in IPv6 are *global* by definition. When using IPv6 with an emulation
|ns3| capability, avoid at all costs address leakage toward the global Internet. 
It is advisable to setup an external firewall to prevent leakage.

2001:DB8::/32 addresses
+++++++++++++++++++++++

IPv6 standard (RFC 3849) defines the ``2001:DB8::/32`` address class for the *documentation*.
This manual uses this convention. The addresses in this class are, however, only usable in
a document, and routers should discard them.

Validation
**********

The IPv6 protocols has not yet been extensively validated against real implementations.
The actual tests involve mainly performing checks of the .pcap trace files with Wireshark, 
and the results are positive.

