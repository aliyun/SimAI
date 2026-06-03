.. include:: replace.txt
.. highlight:: cpp

Routing overview
----------------

|ns3| is intended to support traditional routing approaches and protocols,
support ports of open source routing implementations, and facilitate research
into unorthodox routing techniques. The overall routing architecture is
described below in :ref:`Routing-architecture`. Users who wish to just read
about how to configure global routing for wired topologies can read 
:ref:`Global-centralized-routing`. Unicast routing protocols are described in 
:ref:`Unicast-routing`.  Multicast routing is documented in 
:ref:`Multicast-routing`.

.. _Routing-architecture:

Routing architecture
********************

.. _fig-routing:

.. figure:: figures/routing.*

    Overview of routing


:ref:`fig-routing` shows the overall routing architecture for Ipv4. The key
objects are Ipv4L3Protocol, Ipv4RoutingProtocol(s) (a class to which all
routing/forwarding has been delegated from Ipv4L3Protocol), and Ipv4Route(s).

Ipv4L3Protocol must have at least one Ipv4RoutingProtocol added to it at
simulation setup time. This is done explicitly by calling
Ipv4::SetRoutingProtocol ().

The abstract base class Ipv4RoutingProtocol () declares a minimal interface,
consisting of two methods:  RouteOutput () and RouteInput ().  For packets
traveling outbound from a host, the transport protocol will query Ipv4 for the
Ipv4RoutingProtocol object interface, and will request a route via
Ipv4RoutingProtocol::RouteOutput ().  A Ptr to Ipv4Route object is returned.
This is analogous to a dst_cache entry in Linux. The Ipv4Route is carried down
to the Ipv4L3Protocol to avoid a second lookup there. However, some cases (e.g.
Ipv4 raw sockets) will require a call to RouteOutput()
directly from Ipv4L3Protocol.

For packets received inbound for forwarding or delivery, 
the following steps occur. Ipv4L3Protocol::Receive() calls
Ipv4RoutingProtocol::RouteInput(). This passes the packet ownership to the
Ipv4RoutingProtocol object. There are four callbacks associated with this call:

* LocalDeliver 
* UnicastForward
* MulticastForward
* Error

The Ipv4RoutingProtocol must eventually call one of these callbacks for each
packet that it takes responsibility for. This is basically how the input routing
process works in Linux.

.. _routing-specialization:

.. figure:: figures/routing-specialization.*

   Ipv4Routing specialization. 

This overall architecture is designed to support different routing approaches,
including (in the future) a Linux-like policy-based routing implementation,
proactive and on-demand routing protocols, and simple routing protocols for when
the simulation user does not really care about routing.

:ref:`routing-specialization` illustrates how multiple routing protocols derive
from this base class. A class Ipv4ListRouting (implementation class
Ipv4ListRoutingImpl) provides the existing list routing approach in |ns3|. Its
API is the same as base class Ipv4Routing except for the ability to add multiple
prioritized routing protocols (Ipv4ListRouting::AddRoutingProtocol(),
Ipv4ListRouting::GetRoutingProtocol()).

The details of these routing protocols are described below in
:ref:`Unicast-routing`.  For now, we will first start with a basic
unicast routing capability that is intended to globally build routing
tables at simulation time t=0 for simulation users who do not care
about dynamic routing.

.. _Unicast-routing:

Unicast routing
***************

The following unicast routing protocols are defined for IPv4 and IPv6:

* classes Ipv4ListRouting and Ipv6ListRouting (used to store a prioritized list of routing protocols)
* classes Ipv4StaticRouting and Ipv6StaticRouting (covering both unicast and multicast)
* class Ipv4GlobalRouting (used to store routes computed by the global route
  manager, if that is used)
* class Ipv4NixVectorRouting (a more efficient version of global routing that
  stores source routes in a packet header field)
* class Rip - the IPv4 RIPv2 protocol (:rfc:`2453`)
* class RipNg - the IPv6 RIPng protocol (:rfc:`2080`)
* IPv4 Optimized Link State Routing (OLSR) (a MANET protocol defined in 
  :rfc:`3626`)
* IPv4 Ad Hoc On Demand Distance Vector (AODV) (a MANET protocol defined in
  :rfc:`3561`)
* IPv4 Destination Sequenced Distance Vector (DSDV) (a MANET protocol)
* IPv4 Dynamic Source Routing (DSR) (a MANET protocol)

In the future, this architecture should also allow someone to implement a
Linux-like implementation with routing cache, or a Click modular router, but
those are out of scope for now.

Ipv[4,6]ListRouting
+++++++++++++++++++

This section describes the current default |ns3| Ipv[4,6]RoutingProtocol. Typically,
multiple routing protocols are supported in user space and coordinate to write a
single forwarding table in the kernel. Presently in |ns3|, the implementation
instead allows for multiple routing protocols to build/keep their own routing
state, and the IP implementation will query each one of these routing
protocols (in some order determined by the simulation author) until a route is
found.  

We chose this approach because it may better facilitate the integration of
disparate routing approaches that may be difficult to coordinate the writing to
a single table, approaches where more information than destination IP address
(e.g., source routing) is used to determine the next hop, and on-demand routing
approaches where packets must be cached.  

Ipv[4,6]ListRouting::AddRoutingProtocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes Ipv4ListRouting and Ipv6ListRouting provides a pure virtual function declaration
for the method that allows one to add a routing protocol::

  void AddRoutingProtocol (Ptr<Ipv4RoutingProtocol> routingProtocol,
                           int16_t priority);

  void AddRoutingProtocol (Ptr<Ipv6RoutingProtocol> routingProtocol,
                           int16_t priority);

These methods are implemented respectively by class Ipv4ListRoutingImpl and by class 
Ipv6ListRoutingImpl in the internet module.

The priority variable above governs the priority in which the routing protocols
are inserted. Notice that it is a signed int.  By default in |ns3|, the helper
classes will instantiate a Ipv[4,6]ListRoutingImpl object, and add to it an
Ipv[4,6]StaticRoutingImpl object at priority zero.  Internally, a list of
Ipv[4,6]RoutingProtocols is stored, and and the routing protocols are each consulted
in decreasing order of priority to see whether a match is found. Therefore, if
you want your Ipv4RoutingProtocol to have priority lower than the static
routing, insert it with priority less than 0; e.g.::

  Ptr<MyRoutingProtocol> myRoutingProto = CreateObject<MyRoutingProtocol> ();
  listRoutingPtr->AddRoutingProtocol (myRoutingProto, -10);

Upon calls to RouteOutput() or RouteInput(), the list routing object will search
the list of routing protocols, in priority order, until a route is found. Such
routing protocol will invoke the appropriate callback and no further routing
protocols will be searched.  

.. _Global-centralized-routing:

Global centralized routing
++++++++++++++++++++++++++

Global centralized routing is sometimes called "God" routing; it is a special
implementation that walks the simulation topology and runs a shortest path
algorithm, and populates each node's routing tables. No actual protocol overhead
(on the simulated links) is incurred with this approach. It does have a few
constraints:

* **Wired only:**  It is not intended for use in wireless networks.
* **Unicast only:** It does not do multicast.
* **Scalability:**  Some users of this on large topologies (e.g. 1000 nodes)
  have noticed that the current implementation is not very scalable. The global
  centralized routing will be modified in the future to reduce computations and
  runtime performance.

Presently, global centralized IPv4 unicast routing over both point-to-point and
shared (CSMA) links is supported.

By default, when using the |ns3| helper API and the default InternetStackHelper,
global routing capability will be added to the node, and global routing will be
inserted as a routing protocol with lower priority than the static routes (i.e.,
users can insert routes via Ipv4StaticRouting API and they will take precedence
over routes found by global routing).

Global Unicast Routing API
~~~~~~~~~~~~~~~~~~~~~~~~~~

The public API is very minimal. User scripts include the following::

    #include "ns3/internet-module.h"

If the default InternetStackHelper is used, then an instance of global routing
will be aggregated to each node.  After IP addresses are configured, the
following function call will cause all of the nodes that have an Ipv4 interface
to receive forwarding tables entered automatically by the GlobalRouteManager::

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

*Note:* A reminder that the wifi NetDevice will work but does not take any
wireless effects into account. For wireless, we recommend OLSR dynamic routing
described below.

It is possible to call this function again in the midst of a simulation using
the following additional public function::

  Ipv4GlobalRoutingHelper::RecomputeRoutingTables ();

which flushes the old tables, queries the nodes for new interface information,
and rebuilds the routes.

For instance, this scheduling call will cause the tables to be rebuilt
at time 5 seconds::

  Simulator::Schedule (Seconds (5),
                       &Ipv4GlobalRoutingHelper::RecomputeRoutingTables);


There are two attributes that govern the behavior. The first is
Ipv4GlobalRouting::RandomEcmpRouting. If set to true, packets are randomly
routed across equal-cost multipath routes. If set to false (default), only one
route is consistently used. The second is
Ipv4GlobalRouting::RespondToInterfaceEvents. If set to true, dynamically
recompute the global routes upon Interface notification events (up/down, or
add/remove address). If set to false (default), routing may break unless the
user manually calls RecomputeRoutingTables() after such events. The default is
set to false to preserve legacy |ns3| program behavior.

Global Routing Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section is for those readers who care about how this is implemented.  A
singleton object (GlobalRouteManager) is responsible for populating the static
routes on each node, using the public Ipv4 API of that node.  It queries each
node in the topology for a "globalRouter" interface.  If found, it uses the API
of that interface to obtain a "link state advertisement (LSA)" for the router.
Link State Advertisements are used in OSPF routing, and we follow their
formatting.

It is important to note that all of these computations are done before 
packets are flowing in the network.  In particular, there are no
overhead or control packets being exchanged when using this implementation.
Instead, this global route manager just walks the list of nodes to
build the necessary information and configure each node's routing table.

The GlobalRouteManager populates a link state database with LSAs gathered from
the entire topology. Then, for each router in the topology, the
GlobalRouteManager executes the OSPF shortest path first (SPF) computation on
the database, and populates the routing tables on each node.

The quagga (`<http://www.quagga.net>`_) OSPF implementation was used as the
basis for the routing computation logic. One benefit of following an existing
OSPF SPF implementation is that OSPF already has defined link state
advertisements for all common types of network links:

* point-to-point (serial links)
* point-to-multipoint (Frame Relay, ad hoc wireless)
* non-broadcast multiple access (ATM)
* broadcast (Ethernet)

Therefore, we think that enabling these other link types will be more
straightforward now that the underlying OSPF SPF framework is in place.

Presently, we can handle IPv4 point-to-point, numbered links, as well as shared
broadcast (CSMA) links.  Equal-cost multipath is also supported.  Although
wireless link types are supported by the implementation, note that due
to the nature of this implementation, any channel effects will not be
considered and the routing tables will assume that every node on the
same shared channel is reachable from every other node (i.e. it will
be treated like a broadcast CSMA link).

The GlobalRouteManager first walks the list of nodes and aggregates
a GlobalRouter interface to each one as follows::

  typedef std::vector < Ptr<Node> >::iterator Iterator;
  for (Iterator i = NodeList::Begin (); i != NodeList::End (); i++)
    {
      Ptr<Node> node = *i;
      Ptr<GlobalRouter> globalRouter = CreateObject<GlobalRouter> (node);
      node->AggregateObject (globalRouter);
    }

This interface is later queried and used to generate a Link State
Advertisement for each router, and this link state database is
fed into the OSPF shortest path computation logic. The Ipv4 API
is finally used to populate the routes themselves. 


RIP and RIPng
+++++++++++++

The RIPv2 protocol for IPv4 is described in the :rfc:`2453`, and it consolidates
a number of improvements over the base protocol defined in :rfc:`1058`.

This IPv6 routing protocol (:rfc:`2080`) is the evolution of the well-known 
RIPv1 (see :rfc:`1058` and :rfc:`1723`) routing protocol for IPv4.

The protocols are very simple, and are normally suitable for flat, simple 
network topologies.

RIPv1, RIPv2, and RIPng have the very same goals and limitations. 
In particular, RIP considers any route with a metric equal or greater 
than 16 as unreachable. As a consequence, the maximum number of hops is the
network must be less than 15 (the number of routers is not set).
Users are encouraged to read :rfc:`2080` and :rfc:`1058` to fully understand
RIP behaviour and limitations.


Routing convergence
~~~~~~~~~~~~~~~~~~~

RIP uses a Distance-Vector algorithm, and routes are updated according to
the Bellman-Ford algorithm (sometimes known as Ford-Fulkerson algorithm).
The algorithm has a convergence time of O(\|V\|*\|E\|) where \|V\| and \|E\| 
are the number of vertices (routers) and edges (links) respectively.
It should be stressed that the convergence time is the number of steps in
the algorithm, and each step is triggered by a message.
Since Triggered Updates (i.e., when a route is changed) have a 1-5 seconds 
cooldown, the topology can require some time to be stabilized.

Users should be aware that, during routing tables construction, the routers 
might drop packets. Data traffic should be sent only after a time long
enough to allow RIP to build the network topology.
Usually 80 seconds should be enough to have a suboptimal (but working)
routing setup. This includes the time needed to propagate the routes to the
most distant router (16 hops) with Triggered Updates.

If the network topology is changed (e.g., a link is broken), the recovery 
time might be quite high, and it might be even higher than the initial 
setup time. Moreover, the network topology recovery is affected by
the Split Horizoning strategy.

The examples ``examples/routing/ripng-simple-network.cc`` and 
``examples/routing/rip-simple-network.cc``
shows both the network setup and network recovery phases.

Split Horizoning
~~~~~~~~~~~~~~~~

Split Horizon is a strategy to prevent routing instability.
Three options are possible:

* No Split Horizon
* Split Horizon
* Poison Reverse

In the first case, routes are advertised on all the router's interfaces.
In the second case, routers will not advertise a route on the interface
from which it was learned.
Poison Reverse will advertise the route on the interface from which it 
was learned, but with a metric of 16 (infinity).
For a full analysis of the three techniques, see :rfc:`1058`, section 2.2.

The examples are based on the network topology
described in the RFC, but it does not show the effect described there.

The reason are the Triggered Updates, together with the fact that when a 
router invalidates a route, it will immediately propagate the route 
unreachability, thus preventing most of the issues described in the RFC.

However, with complex topologies, it is still possible to have route 
instability phenomena similar to the one described in the RFC after a 
link failure. As a consequence, all the considerations about Split Horizon
remains valid.


Default routes
~~~~~~~~~~~~~~

RIP protocol should be installed *only* on routers. As a consequence, 
nodes will not know what is the default router.

To overcome this limitation, users should either install the default route
manually (e.g., by resorting to Ipv4StaticRouting or Ipv6StaticRouting), or 
by using RADVd (in case of IPv6).
RADVd is available in |ns3| in the Applications module, and it is strongly 
suggested.
 
Protocol parameters and options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RIP |ns3| implementations allow to change all the timers associated 
with route updates and routes lifetime.

Moreover, users can change the interface metrics on a per-node basis.

The type of Split Horizoning (to avoid routes back-propagation) can be 
selected on a per-node basis, with the choices being "no split horizon", 
"split horizon" and "poison reverse". See :rfc:`2080` for further details,
and :rfc:`1058` for a complete discussion on the split horizoning strategies.

Moreover, it is possible to use a non-standard value for Link Down Value (i.e.,
the value after which a link is considered down). The default is value is 16. 

Limitations
~~~~~~~~~~~

There is no support for the Next Hop option (:rfc:`2080`, Section 2.1.1).
The Next Hop option is useful when RIP is not being run on all of the 
routers on a network.
Support for this option may be considered in the future.

There is no support for CIDR prefix aggregation. As a result, both routing 
tables and route advertisements may be larger than necessary. 
Prefix aggregation may be added in the future.


Other routing protocols
+++++++++++++++++++++++

Other routing protocols documentation can be found under the respective
modules sections, e.g.:

* AODV
* Click
* DSDV
* DSR
* NixVectorRouting
* OLSR
* etc.


.. _Multicast-routing:

Multicast routing
*****************

The following function is used to add a static multicast route
to a node::

    void 
    Ipv4StaticRouting::AddMulticastRoute (Ipv4Address origin,
                                          Ipv4Address group,
                                          uint32_t inputInterface,
                                          std::vector<uint32_t> outputInterfaces);

A multicast route must specify an origin IP address, a multicast group and an
input network interface index as conditions and provide a vector of output
network interface indices over which packets matching the conditions are sent.

Typically there are two main types of multicast routes: 

* Routes used during forwarding, and
* Routes used in the originator node.

In the first case all the conditions must be explicitly
provided. 

In the second case, the route is equivalent to a unicast route, and must be added
through `Ipv4StaticRouting::AddHostRouteTo`.

Another command sets the default multicast route::

    void 
    Ipv4StaticRouting::SetDefaultMulticastRoute (uint32_t outputInterface);

This is the multicast equivalent of the unicast version SetDefaultRoute. We
tell the routing system what to do in the case where a specific route to a
destination multicast group is not found. The system forwards packets out the
specified interface in the hope that "something out there" knows better how to
route the packet. This method is only used in initially sending packets off of a
host. The default multicast route is not consulted during forwarding -- exact
routes must be specified using AddMulticastRoute for that case.

Since we're basically sending packets to some entity we think may know better
what to do, we don't pay attention to "subtleties" like origin address, nor do
we worry about forwarding out multiple  interfaces. If the default multicast
route is set, it is returned as the selected route from LookupStatic
irrespective of origin or multicast group if another specific route is not
found.

Finally, a number of additional functions are provided to fetch and remove
multicast routes::

  uint32_t GetNMulticastRoutes (void) const;

  Ipv4MulticastRoute *GetMulticastRoute (uint32_t i) const;

  Ipv4MulticastRoute *GetDefaultMulticastRoute (void) const;

  bool RemoveMulticastRoute (Ipv4Address origin,
                             Ipv4Address group,
                             uint32_t inputInterface);

  void RemoveMulticastRoute (uint32_t index);
