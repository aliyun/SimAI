.. include:: replace.txt

Optimized Link State Routing (OLSR)
-----------------------------------

This model implements the base specification of the Optimized
Link State Routing (OLSR) protocol, which is a dynamic mobile ad hoc
unicast routing protocol.  It has been developed at the
University of Murcia (Spain) by Francisco J. Ros for NS-2, and was
ported to NS-3 by Gustavo Carneiro at INESC Porto (Portugal).

The implementation is based on OLSR Version 1 (:rfc:`3626` [rfc3626]_) and
it is *not* compliant with OLSR Version 2 (:rfc:`7181` [rfc7181]_) or any 
of the Version 2 extensions.
 
Model Description
*****************

The source code for the OLSR model lives in the directory `src/olsr`.
As stated before, the model is based on :rfc:`3626` ([rfc3626]_). Moreover, many
design choices are based on the previous ns2 model.

Scope and Limitations
+++++++++++++++++++++

The model is for IPv4 only.  

* Mostly compliant with OLSR as documented in :rfc:`3626` ([rfc3626]_), 
* The use of multiple interfaces was not supported by the NS-2 version, but is supported in NS-3;
* OLSR does not respond to the routing event notifications corresponding to dynamic interface up and down (``ns3::RoutingProtocol::NotifyInterfaceUp`` and ``ns3::RoutingProtocol::NotifyInterfaceDown``) or address insertion/removal ``ns3::RoutingProtocol::NotifyAddAddress`` and ``ns3::RoutingProtocol::NotifyRemoveAddress``).
* Unlike the NS-2 version, does not yet support MAC layer feedback as described in :rfc:`3626` ([rfc3626]_);

Host Network Association (HNA) is supported in this implementation
of OLSR. Refer to ``examples/olsr-hna.cc`` to see how the API
is used.

References
++++++++++

.. [rfc3626] :rfc:`3626` *Optimized Link State Routing*
.. [rfc7181] :rfc:`7181` *The Optimized Link State Routing Protocol Version 2*

Usage
*****

The usage pattern is the one of all the Internet routing protocols.
Since OLSR is not installed by default in the Internet stack, it is necessary to
set it in the Internet Stack helper by using ``InternetStackHelper::SetRoutingHelper``

Typically, OLSR is enabled in a main program by use of an OlsrHelper class that
installs OLSR into an Ipv4ListRoutingProtocol object. The following sample
commands will enable OLSR in a simulation using this helper class along with
some other routing helper objects. The setting of priority value 10, ahead of
the staticRouting priority of 0, means that OLSR will be consulted for a route
before the node's static routing table.::

  NodeContainer c:
  ...
  // Enable OLSR
  NS_LOG_INFO ("Enabling OLSR Routing.");
  OlsrHelper olsr;

  Ipv4StaticRoutingHelper staticRouting;

  Ipv4ListRoutingHelper list;
  list.Add (staticRouting, 0);
  list.Add (olsr, 10);

  InternetStackHelper internet;
  internet.SetRoutingHelper (list);
  internet.Install (c);

Once installed,the OLSR "main interface" can be set with the SetMainInterface()
command. If the user does not specify a main address, the protocol will select
the first primary IP address that it finds, starting first the loopback
interface and then the next non-loopback interface found, in order of Ipv4
interface index. The loopback address of 127.0.0.1 is not selected. In addition,
a number of protocol constants are defined in olsr-routing-protocol.cc.

Olsr is started at time zero of the simulation, based on a call to
Object::Start() that eventually calls OlsrRoutingProtocol::DoStart(). Note:  a
patch to allow the user to start and stop the protocol at other times would be
welcome.

Examples
++++++++

The examples are in the ``src/olsr/examples/`` directory. However, many other examples esists in the
general examples directory, e.g., ``examples/routing/manet-routing-compare.cc``.

For specific examples of the HNA feature, see the examples in ``src/olsr/examples/``.

Helpers
+++++++

A helper class for OLSR has been written.  After an IPv4 topology
has been created and unique IP addresses assigned to each node, the
simulation script writer can call one of three overloaded functions
with different scope to enable OLSR: ``ns3::OlsrHelper::Install
(NodeContainer container)``; ``ns3::OlsrHelper::Install (Ptr<Node>
node)``; or ``ns3::OlsrHelper::InstallAll (void)``

Attributes
++++++++++

In addition, the behavior of OLSR can be modified by changing certain
attributes.  The method ``ns3::OlsrHelper::Set ()`` can be used
to set OLSR attributes.  These include HelloInterval, TcInterval,
MidInterval, Willingness.  Other parameters are defined as macros
in ``olsr-routing-protocol.cc``.

The list of configurabel attributes is:

* HelloInterval (time, default 2s), HELLO messages emission interval.
* TcInterval (time, default 5s), TC messages emission interval.
* MidInterval (time, default 5s), MID messages emission interval.
* HnaInterval (time, default 5s), HNA messages emission interval.
* Willingness (enum, default OLSR_WILL_DEFAULT), Willingness of a node to carry and forward traffic for other nodes.

Tracing
+++++++

The available traces are:

* Rx: Receive OLSR packet.
* Tx: Send OLSR packet.
* RoutingTableChanged: The OLSR routing table has changed.

Caveats
+++++++

Presently, OLSR is limited to use with an Ipv4ListRouting object, and does not
respond to dynamic changes to a device's IP address or link up/down
notifications; i.e. the topology changes are due to loss/gain of connectivity
over a wireless channel.

The code does not present any known issue.

Validation
**********

The code validationhas been done through Wireshark message compliance and unit testings.
