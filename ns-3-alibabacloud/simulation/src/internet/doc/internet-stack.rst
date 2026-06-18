.. include:: replace.txt
.. highlight:: cpp

Internet Stack
--------------

Internet stack aggregation
**************************

A bare class :cpp:class:`Node` is not very useful as-is; other objects must be
aggregated to it to provide useful node functionality.

The |ns3| source code directory ``src/internet`` provides implementation
of TCP/IPv4- and IPv6-related components. These include IPv4, ARP, UDP, TCP,
IPv6, Neighbor Discovery, and other related protocols.

Internet Nodes are not subclasses of class Node; they are simply Nodes that have
had a bunch of IP-related objects aggregated to them. They can be put together
by hand, or via a helper function :cpp:func:`InternetStackHelper::Install ()`
which does the following to all nodes passed in as arguments::

    void
    InternetStackHelper::Install (Ptr<Node> node) const
    {
      if (m_ipv4Enabled)
        {
          /* IPv4 stack */
          if (node->GetObject<Ipv4> () != 0)
            {
              NS_FATAL_ERROR ("InternetStackHelper::Install (): Aggregating " 
                              "an InternetStack to a node with an existing Ipv4 object");
              return;
            }

          CreateAndAggregateObjectFromTypeId (node, "ns3::ArpL3Protocol");
          CreateAndAggregateObjectFromTypeId (node, "ns3::Ipv4L3Protocol");
          CreateAndAggregateObjectFromTypeId (node, "ns3::Icmpv4L4Protocol");
          // Set routing
          Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
          Ptr<Ipv4RoutingProtocol> ipv4Routing = m_routing->Create (node);
          ipv4->SetRoutingProtocol (ipv4Routing);
        }

      if (m_ipv6Enabled)
        {
          /* IPv6 stack */
          if (node->GetObject<Ipv6> () != 0)
            {
              NS_FATAL_ERROR ("InternetStackHelper::Install (): Aggregating " 
                              "an InternetStack to a node with an existing Ipv6 object");
              return;
            }

          CreateAndAggregateObjectFromTypeId (node, "ns3::Ipv6L3Protocol");
          CreateAndAggregateObjectFromTypeId (node, "ns3::Icmpv6L4Protocol");
          // Set routing
          Ptr<Ipv6> ipv6 = node->GetObject<Ipv6> ();
          Ptr<Ipv6RoutingProtocol> ipv6Routing = m_routingv6->Create (node);
          ipv6->SetRoutingProtocol (ipv6Routing);

          /* register IPv6 extensions and options */
          ipv6->RegisterExtensions ();
          ipv6->RegisterOptions ();
        }

      if (m_ipv4Enabled || m_ipv6Enabled)
        {
          /* UDP and TCP stacks */
          CreateAndAggregateObjectFromTypeId (node, "ns3::UdpL4Protocol");
          node->AggregateObject (m_tcpFactory.Create<Object> ());
          Ptr<PacketSocketFactory> factory = CreateObject<PacketSocketFactory> ();
          node->AggregateObject (factory);
        }
    }

Where multiple implementations exist in |ns3| (TCP, IP routing), these objects
are added by a factory object (TCP) or by a routing helper (m_routing).

Note that the routing protocol is configured and set outside this
function. By default, the following protocols are added::

    void InternetStackHelper::Initialize ()
    {
      SetTcp ("ns3::TcpL4Protocol");
      Ipv4StaticRoutingHelper staticRouting;
      Ipv4GlobalRoutingHelper globalRouting;
      Ipv4ListRoutingHelper listRouting;
      Ipv6ListRoutingHelper listRoutingv6;
      Ipv6StaticRoutingHelper staticRoutingv6;
      listRouting.Add (staticRouting, 0);
      listRouting.Add (globalRouting, -10);
      listRoutingv6.Add (staticRoutingv6, 0);
      SetRoutingHelper (listRouting);
      SetRoutingHelper (listRoutingv6);
    }

By default, IPv4 and IPv6 are enabled.

Internet Node structure
+++++++++++++++++++++++

An IP-capable Node (an |ns3| Node augmented by aggregation to have one or more
IP stacks) has the following internal structure.

Layer-3 protocols
~~~~~~~~~~~~~~~~~

At the lowest layer, sitting above the NetDevices, are the "layer 3" protocols,
including IPv4, IPv6, ARP and so on. The class
:cpp:class:`Ipv4L3Protocol` is an implementation class whose public interface is
typically class :cpp:class:`Ipv4`, but the
Ipv4L3Protocol public API is also used internally at present.

In class Ipv4L3Protocol, one method described below is ``Receive ()``::

     /**
       * Lower layer calls this method after calling L3Demux::Lookup
       * The ARP subclass needs to know from which NetDevice this
       * packet is coming to:
       *    - implement a per-NetDevice ARP cache
       *    - send back arp replies on the right device
       */
     void Receive( Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, 
     const Address &from, const Address &to, NetDevice::PacketType packetType);

First, note that the ``Receive ()`` function has a matching signature to the
ReceiveCallback in the class :cpp:class:`Node`. This function pointer is
inserted into the Node's protocol handler when ``AddInterface ()`` is called.
The actual registration is done
with a statement such as follows::

     RegisterProtocolHandler ( MakeCallback (&Ipv4Protocol::Receive, ipv4),
                               Ipv4L3Protocol::PROT_NUMBER, 0);

The Ipv4L3Protocol object is aggregated to the Node; there is only one such
Ipv4L3Protocol object. Higher-layer protocols that have a packet to send down to
the Ipv4L3Protocol object can call ``GetObject<Ipv4L3Protocol> ()`` to obtain a
pointer, as follows::

  Ptr<Ipv4L3Protocol> ipv4 = m_node->GetObject<Ipv4L3Protocol> ();
  if (ipv4 != 0)
    {
      ipv4->Send (packet, saddr, daddr, PROT_NUMBER);
    }

This class nicely demonstrates two techniques we exploit in |ns3| to bind
objects together:  callbacks, and object aggregation.

Once IPv4 routing has determined that a packet is for the local node, it
forwards it up the stack.  This is done with the following function::

    void
    Ipv4L3Protocol::LocalDeliver (Ptr<const Packet> packet, Ipv4Header const&ip, uint32_t iif)

The first step is to find the right Ipv4L4Protocol object, based on IP protocol
number. For instance, TCP is registered in the demux as protocol number 6.
Finally, the ``Receive()`` function on the Ipv4L4Protocol (such as
``TcpL4Protocol::Receive`` is called.

We have not yet introduced the class Ipv4Interface. Basically, each NetDevice is
paired with an IPv4 representation of such device. In Linux, this class
:cpp:class:`Ipv4Interface` roughly corresponds to the ``struct in_device``; the
main purpose is to provide address-family specific information (addresses) about
an interface.

All the classes have appropriate traces in order to track sent, received and lost packets.
The users is encouraged to use them so to find out if (and where) a packet is dropped. A
common mistake is to forget the effects of local queues when sending packets, e.g., the ARP
queue. This can be particularly puzzling when sending jumbo packets or packet bursts using UDP.
The ARP cache pending queue is limited (3 datagrams) and IP packets might be fragmented, easily
overfilling the ARP cache queue size. In those cases it is useful to increase the ARP cache
pending size to a proper value, e.g.::

    Config::SetDefault ("ns3::ArpCache::PendingQueueSize", UintegerValue (MAX_BURST_SIZE/L2MTU*3));

The IPv6 implementation follows a similar architecture.  Dual-stacked nodes (one with
support for both IPv4 and IPv6) will allow an IPv6 socket to receive IPv4 connections
as a standard dual-stacked system does.  A socket bound and listening to an IPv6 endpoint
can receive an IPv4 connection and will return the remote address as an IPv4-mapped address.
Support for the IPV6_V6ONLY socket option does not currently exist.


Layer-4 protocols and sockets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We next describe how the transport protocols, sockets, and applications tie
together. In summary, each transport protocol implementation is a socket
factory. An application that needs a new socket 

For instance, to create a UDP socket, an application would use a code snippet
such as the following::

      Ptr<Udp> udpSocketFactory = GetNode ()->GetObject<Udp> ();
      Ptr<Socket> m_socket = socketFactory->CreateSocket ();
      m_socket->Bind (m_local_address);
      ...

The above will query the node to get a pointer to its UDP socket factory, will
create one such socket, and will use the socket with an API similar to the
C-based sockets API, such as ``Connect ()`` and ``Send ()``.  The address passed
to the ``Bind ()``, ``Connect ()``, or ``Send ()`` functions may be a
:cpp:class:`Ipv4Address`, :cpp:class:`Ipv6Address`, or :cpp:class:`Address`.
If a :cpp:class:`Address` is passed in and contains anything other than
a :cpp:class:`Ipv4Address` or :cpp:class:`Ipv6Address`, these functions will
return an error.  The ``Bind (void)`` and ``Bind6 (void)`` functions bind to
"0.0.0.0" and "::" respectively.

The socket can also be bound to a specific NetDevice though the 
``BindToNetDevice (Ptr<NetDevice> netdevice)`` function.
``BindToNetDevice (Ptr<NetDevice> netdevice)`` will bind the socket
to "0.0.0.0" and "::" (equivalent to calling ``Bind ()`` and ``Bind6 ()``,
unless the socket has been already bound to a specific address.
Summarizing, the correct sequence is::

      Ptr<Udp> udpSocketFactory = GetNode ()->GetObject<Udp> ();
      Ptr<Socket> m_socket = socketFactory->CreateSocket ();
      m_socket->BindToNetDevice (n_netDevice);
     ...

or::

      Ptr<Udp> udpSocketFactory = GetNode ()->GetObject<Udp> ();
      Ptr<Socket> m_socket = socketFactory->CreateSocket ();
      m_socket->Bind (m_local_address);
      m_socket->BindToNetDevice (n_netDevice);
      ...

The following raises an error::

      Ptr<Udp> udpSocketFactory = GetNode ()->GetObject<Udp> ();
      Ptr<Socket> m_socket = socketFactory->CreateSocket ();
      m_socket->BindToNetDevice (n_netDevice);
      m_socket->Bind (m_local_address);
      ...

See the chapter on |ns3| sockets for more information.  

We have described so far a socket factory (e.g. ``class Udp``) and a socket,
which may be specialized (e.g., class :cpp:class:`UdpSocket`).  There are a few
more key objects that relate to the specialized task of demultiplexing a packet
to one or more receiving sockets.  The key object in this task is class
:cpp:class:`Ipv4EndPointDemux`.  This demultiplexer stores objects of class
:cpp:class:`Ipv4EndPoint`.  This class holds the addressing/port tuple (local
port, local address, destination port, destination address) associated with the
socket, and a receive callback. This receive callback has a receive function
registered by the socket. The ``Lookup ()`` function to Ipv4EndPointDemux
returns a list of Ipv4EndPoint objects (there may be a list since more than one
socket may match the packet). The layer-4 protocol copies the packet to each
Ipv4EndPoint and calls its ``ForwardUp ()`` method, which then calls the
``Receive ()`` function registered by the socket.

An issue that arises when working with the sockets API on real
systems is the need to manage the reading from a socket, using 
some type of I/O (e.g., blocking, non-blocking, asynchronous, ...).
|ns3| implements an asynchronous model for socket I/O; the application
sets a callback to be notified of received data ready to be read, and the 
callback is invoked by the transport protocol when data is available.
This callback is specified as follows::

  void Socket::SetRecvCallback (Callback<void, Ptr<Socket>, 
                                Ptr<Packet>,
                                const Address&> receivedData);

The data being received is conveyed in the Packet data buffer.  An example
usage is in class :cpp:class:`PacketSink`::

  m_socket->SetRecvCallback (MakeCallback(&PacketSink::HandleRead, this));

To summarize, internally, the UDP implementation is organized as follows:

* a ``UdpImpl`` class that implements the UDP socket factory functionality

* a ``UdpL4Protocol`` class that implements the protocol logic that is
  socket-independent 
* a ``UdpSocketImpl`` class that implements socket-specific aspects of UDP
* a class called ``Ipv4EndPoint`` that stores the addressing tuple (local port,
  local address, destination port, destination address) associated with the
  socket, and a receive callback for the socket.

IP-capable node interfaces
++++++++++++++++++++++++++

Many of the implementation details, or internal objects themselves, of
IP-capable Node objects are not exposed at the simulator public API. This
allows for different implementations; for instance, replacing the native |ns3|
models with ported TCP/IP stack code. 
 
The C++ public APIs of all of these objects is found in the ``src/network``
directory, including principally:

* ``address.h``
* ``socket.h``
* ``node.h``
* ``packet.h``

These are typically base class objects that implement the default values used in
the implementation, implement access methods to get/set state variables, host
attributes, and implement publicly-available methods exposed to clients such as
``CreateSocket``.

Example path of a packet
++++++++++++++++++++++++

These two figures show an example stack trace of how packets flow through the
Internet Node objects.

.. _internet-node-send:

.. figure:: figures/internet-node-send.*
   
   Send path of a packet.

.. _internet-node-recv:

.. figure:: figures/internet-node-recv.*
   
   Receive path of a packet.
