.. include:: replace.txt

Ad Hoc On-Demand Distance Vector (AODV)
---------------------------------------

This model implements the base specification of the Ad Hoc On-Demand 
Distance Vector (AODV) protocol. The implementation is based on 
:rfc:`3561`.

The model was written by Elena Buchatskaia and Pavel Boyko of ITTP RAS,
and is based on the ns-2 AODV model developed by the CMU/MONARCH group
and optimized and tuned by Samir Das and Mahesh Marina, University of
Cincinnati, and also on the AODV-UU implementation by Erik Nordström of 
Uppsala University.

Model Description
*****************

The source code for the AODV model lives in the directory `src/aodv`.

Design
++++++

Class ``ns3::aodv::RoutingProtocol`` implements all functionality of 
service packet exchange and inherits from ``ns3::Ipv4RoutingProtocol``.
The base class defines two virtual functions for packet routing and 
forwarding.  The first one, ``ns3::aodv::RouteOutput``, is used for 
locally originated packets, and the second one, ``ns3::aodv::RouteInput``,
is used for forwarding and/or delivering received packets.

Protocol operation depends on many adjustable parameters. Parameters for 
this functionality are attributes of ``ns3::aodv::RoutingProtocol``. 
Parameter default values are drawn from the RFC and allow the 
enabling/disabling protocol features, such as broadcasting HELLO messages, 
broadcasting data packets and so on.

AODV discovers routes on demand.  Therefore, the AODV model buffers all 
packets while a route request packet (RREQ) is disseminated. 
A packet queue is implemented in aodv-rqueue.cc. A smart pointer to 
the packet, ``ns3::Ipv4RoutingProtocol::ErrorCallback``,
``ns3::Ipv4RoutingProtocol::UnicastForwardCallback``, and the IP header 
are stored in this queue. The packet queue implements garbage collection 
of old packets and a queue size limit.

The routing table implementation supports garbage collection of 
old entries and state machine, defined in the standard.
It is implemented as a STL map container. The key is a destination IP address.

Some elements of protocol operation aren't described in the RFC. These 
elements generally concern cooperation of different OSI model layers.
The model uses the following heuristics:

* This AODV implementation can detect the presence of unidirectional 
  links and avoid them if necessary.  If the node the model receives an
  RREQ for is a neighbor, the cause may be a unidirectional link.
  This heuristic is taken from AODV-UU implementation and can be disabled.
* Protocol operation strongly depends on broken link detection mechanism. 
  The model implements two such heuristics.  First, this implementation 
  support HELLO messages. However HELLO messages are not a good way to 
  perform neighbor sensing in a wireless environment (at least not over 
  802.11). Therefore, one may experience bad performance when running over 
  wireless.  There are several reasons for this: 1) HELLO messages are 
  broadcasted. In 802.11, broadcasting is often done at a 
  lower bit rate than unicasting, thus HELLO messages can travel further 
  than unicast data. 2) HELLO messages are small, thus less prone to 
  bit errors than data transmissions, and 3) Broadcast transmissions are 
  not guaranteed to be bidirectional, unlike unicast transmissions.  
  Second, we use layer 2 feedback when possible. Link are considered to be 
  broken if frame transmission results in a transmission failure for all 
  retries. This mechanism is meant for active links and works faster than
  the first method.

The layer 2 feedback implementation relies on the ``TxErrHeader`` trace source, 
currently supported in AdhocWifiMac only.

Scope and Limitations
+++++++++++++++++++++

The model is for IPv4 only.  The following optional protocol optimizations 
are not implemented:

#. Local link repair.
#. RREP, RREQ and HELLO message extensions.

These techniques require direct access to IP header, which contradicts
the assertion from the AODV RFC that AODV works over UDP.  This model uses
UDP for simplicity, hindering the ability to implement certain protocol 
optimizations. The model doesn't use low layer raw sockets because they 
are not portable.

Future Work
+++++++++++

No announced plans.

..
  References
  ++++++++++

..
  Usage
  *****

..
  Examples
  ++++++++

..
  Helpers
  +++++++

..
  Attributes
  ++++++++++

..
  Tracing
  +++++++

..
  Logging
  +++++++

..
  Caveats
  +++++++

..
  Validation
  **********
  Unit tests
  ++++++++++
  Larger-scale performance tests
  ++++++++++++++++++++++++++++++

