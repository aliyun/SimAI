.. include:: replace.txt
.. highlight:: cpp

.. _Sockets-APIs:

Sockets APIs
------------

The `sockets API <http://en.wikipedia.org/wiki/Berkeley_sockets>`_
is a long-standing API used by user-space applications to access
network services in the kernel.  A *socket* is an abstraction, like
a Unix file handle, that allows applications to connect to other
Internet hosts and exchange reliable byte streams and unreliable
datagrams, among other services.   

|ns3| provides two types of sockets APIs, and it is important to
understand the differences between them.  The first is a *native*
|ns3| API, while the second uses the services of the native API to 
provide a `POSIX-like <http://en.wikipedia.org/wiki/POSIX>`_
API as part of an overall application process.  Both APIs strive
to be close to the typical sockets API that application writers
on Unix systems are accustomed to, but the POSIX variant is much
closer to a real system's sockets API.

ns-3 sockets API
*****************

The native sockets API for ns-3 provides an interface to various
types of transport protocols (TCP, UDP) as well as to packet sockets
and, in the future, Netlink-like sockets.  However, users are cautioned
to understand that the semantics are *not* the exact same as 
one finds in a real system (for an API which is very much aligned
to real systems, see the next section).

:cpp:class:`ns3::Socket` is defined in ``src/network/model/socket.h``.
Readers will note that many public member functions are aligned
with real sockets function calls, and all other things being equal,
we have tried to align with a Posix sockets API.  However, note that:

* ns-3 applications handle a smart pointer to a Socket object, not
  a file descriptor;

* there is no notion of synchronous API or a *blocking* API;
  in fact, the model for interaction between application and socket is
  one of asynchronous I/O, which is not typically found in real systems
  (more on this below);

* the C-style socket address structures are not used;

* the API is not a complete sockets API, such as supporting
  all socket options or all function variants; 

* many calls use :cpp:class:`ns3::Packet` class to transfer data
  between application and socket.  This may seem peculiar to
  pass *Packets* across a stream socket API, but think
  of these packets as just fancy byte buffers at this level (more
  on this also below).

Basic operation and calls
=========================

.. _sockets-overview:

.. figure:: figures/sockets-overview.*

   Implementation overview of native sockets API

Creating sockets
================

An application that wants to use sockets must first create one.
On real systems using a C-based API, this is accomplished by calling :c:func:`socket()` ::

     int socket(int domain, int type, int protocol);

which creates a socket in the system and returns an integer descriptor.

In ns-3, we have no equivalent of a system call at the lower layers,
so we adopt the following model.  There are certain *factory*
objects that can create sockets.  Each factory is capable of creating
one type of socket, and if sockets of a particular type are able to
be created on a given node, then a factory that can create such sockets
must be aggregated to the Node::

  static Ptr<Socket> CreateSocket (Ptr<Node> node, TypeId tid);

Examples of TypeIds to pass to this method are :cpp:class:`ns3::TcpSocketFactory`,
:cpp:class:`ns3::PacketSocketFactory`, and :cpp:class:`ns3::UdpSocketFactory`.

This method returns a smart pointer to a Socket object.  Here is an
example::

  Ptr<Node> n0;
  // Do some stuff to build up the Node's internet stack
  Ptr<Socket> localSocket = 
     Socket::CreateSocket (n0, TcpSocketFactory::GetTypeId ());

In some ns-3 code, sockets will not be explicitly created by user's
main programs, if an ns-3 application does it.  For instance, for
:cpp:class:`ns3::OnOffApplication`, the function :cpp:func:`ns3::OnOffApplication::StartApplication`
performs the socket creation, and the application holds the socket
pointer.

Using sockets
=============

Below is a typical sequence of socket calls for a TCP client in a
real implementation::

    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    bind(sock, ...);
    connect(sock, ...);
    send(sock, ...);
    recv(sock, ...);
    close(sock);

There are analogs to all of these calls in ns-3, but we will focus on  
two aspects here.  First, most usage of sockets in real systems 
requires a way to manage I/O between the application and kernel.  
These models include *blocking sockets*, *signal-based I/O*,
and *non-blocking sockets* with polling.  In ns-3, we make use
of the callback mechanisms to support a fourth mode, which is 
analogous to POSIX *asynchronous I/O*.

In this model, on the sending side, if the :c:func:`send()` call were to
fail because of insufficient buffers, the application suspends the
sending of more data until a function registered at the 
:cpp:func:`ns3::Socket::SetSendCallback` callback is invoked.  
An application can also ask the socket how much space is available 
by calling :cpp:func:`ns3::Socket::GetTxAvailable`.  A typical sequence 
of events for sending data (ignoring connection setup) might be::

    SetSendCallback (MakeCallback(&HandleSendCallback));
    Send ();
    Send ();
    ...
    // Send fails because buffer is full
    // Wait until HandleSendCallback is called
    // HandleSendCallback is called by socket, since space now available
    Send (); // Start sending again

Similarly, on the receive side, the socket user does not block on
a call to ``recv()``.  Instead, the application sets a callback
with :cpp:func:`ns3::Socket::SetRecvCallback` in which the socket will notify the
application when (and how much) there is data to be read, and
the application then calls :cpp:func:`ns3::Socket::Recv` to read the data until
no more can be read.


Packet vs. buffer variants
**************************

There are two basic variants of ``Send()`` and ``Recv()`` supported::

  virtual int Send (Ptr<Packet> p) = 0;
  int Send (const uint8_t* buf, uint32_t size);

  Ptr<Packet> Recv (void);
  int Recv (uint8_t* buf, uint32_t size);

The non-Packet variants are provided for legacy API reasons.  When calling
the raw buffer variant of :cpp:func:`ns3::Socket::Send`, the buffer is immediately
written into a Packet and the packet variant is invoked.

Users may find it semantically odd to pass a Packet to a stream socket
such as TCP.  However, do not let the name bother you; think of 
:cpp:class:`ns3::Packet` to be a fancy byte buffer.  There are a few reasons why
the Packet variants are more likely to be preferred in ns-3:

* Users can use the Tags facility of packets to, for example, encode
  a flow ID or other helper data at the application layer.

* Users can exploit the copy-on-write implementation to avoid
  memory copies (on the receive side, the conversion back to a 
  ``uint8_t* buf`` may sometimes incur an additional copy).

* Use of Packet is more aligned with the rest of the ns-3 API

Sending dummy data
******************

Sometimes, users want the simulator to just pretend that there is an
actual data payload in the packet (e.g. to calculate transmission delay)
but do not want to actually produce or consume the data.  This is
straightforward to support in ns-3; have applications call
``Create<Packet> (size);`` instead of ``Create<Packet> (buffer, size);``.
Similarly, passing in a zero to the pointer argument in the raw buffer
variants has the same effect.  Note that, if some subsequent code tries
to read the Packet data buffer, the fake buffer will be converted to
a real (zeroed) buffer on the spot, and the efficiency will be lost there.

.. _Socket-options:

Use of Send() vs. SendTo()
**************************

There are two variants of methods used to send data to the socket::

  virtual int Send (Ptr<Packet> p, uint32_t flags) = 0;

  virtual int SendTo (Ptr<Packet> p, uint32_t flags,
                      const Address &toAddress) = 0;

The first method is used if the socket has already been connected
(``Socket::Connect()``) to a peer address.  In the case of stream-based
sockets like TCP, the connect call is required to bind the socket to
a peer address, and thereafter, ``Send()`` is typically used.  In the case of
datagram-based sockets like UDP, the socket is not required to
be connected to a peer address before sending, and the socket may be used to
send data to different destination addresses; in this case, the
``SendTo()`` method is used to specify the destination address for
the datagram.

Socket options
**************

.. _Type-of-service:

ToS (Type of Service)
======================

The native sockets API for ns-3 provides two public methods
(of the Socket base class)::

    void SetIpTos (uint8_t ipTos);
    uint8_t GetIpTos (void) const;

to set and get, respectively, the type of service associated with the socket.
These methods are equivalent to using the IP_TOS option of BSD sockets.
Clearly, setting the type of service only applies to sockets using the IPv4 protocol.
However, users typically do not set the type of service associated with a socket
through :cpp:func:`ns3::Socket::SetIpTos` because sockets are normally created
by application helpers and users cannot get a pointer to the sockets.
Instead, users can create an address of type :cpp:class:`ns3::InetSocketAddress`
with the desired type of service value and pass it to the application helpers::

    InetSocketAddress destAddress (ipv4Address, udpPort);
    destAddress.SetTos (tos);
    OnOffHelper onoff ("ns3::UdpSocketFactory", destAddress);

For this to work, the application must eventually call the
:cpp:func:`ns3::Socket::Connect()` method to connect to the provided
destAddress and the Connect method of the particular socket type must
support setting the type of service associated with a socket (by using
the :cpp:func:`ns3::Socket::SetIpTos()` method). Currently, the socket
types that support setting the type of service in such a way are
:cpp:class:`ns3::UdpSocketImpl` and :cpp:class:`ns3::TcpSocketBase`.

The type of service associated with a socket is then used to determine the value
of the Type of Service field (renamed as Differentiated Services field by RFC
2474) of the IPv4 header of the packets sent through that socket, as detailed
in the next sections.


Setting the ToS with UDP sockets
#################################

For IPv4 packets, the ToS field is set according to the following rules:

* If the socket is connected, the ToS field is set to the ToS value associated
  with the socket.

* If the socket is not connected, the ToS field is set to the value specified
  in the destination address (of type :cpp:class:`ns3::InetSocketAddress`) passed
  to :cpp:func:`ns3::Socket::SendTo`, and the ToS value associated with the
  socket is ignored.

Setting the ToS with TCP sockets
#################################

For IPv4 packets, the ToS field is set to the ToS value associated with the
socket.

Priority
=========

The native sockets API for ns-3 provides two public methods
(of the Socket base class)::

    void SetPriority (uint8_t priority);
    uint8_t GetPriority (void) const;

to set and get, respectively, the priority associated with the socket.
These methods are equivalent to using the SO_PRIORITY option of BSD sockets.
Only values in the range 0..6 can be set through the above method.

Note that setting the type of service associated with a socket (by calling
:cpp:func:`ns3::Socket::SetIpTos()`) also sets the priority for the socket
to the value that the :cpp:func:`ns3::Socket::IpTos2Priority()` function
returns when it is passed the type of service value. This function
is implemented after the Linux rt_tos2priority function, which takes
an 8-bit value as input and returns a value which is a function of bits 3-6
(where bit 0 is the most significant bit) of the input value:

=========  ====================
Bits 3-6   Priority
=========  ====================
 0 to 3    0 (Best Effort)
 4 to 7    2 (Bulk)
 8 to 11   6 (Interactive)
 12 to 15  4 (Interactive Bulk)
=========  ====================

The rationale is that bits 3-6 of the Type of Service field were interpreted
as the TOS subfield by (the obsolete) RFC 1349. Readers can refer to the
doxygen documentation of :cpp:func:`ns3::Socket::IpTos2Priority()`
for more information, including how DSCP values map onto priority values.

The priority set for a socket (as described above) is then used to determine
the priority of the packets sent through that socket, as detailed in the next
sections. Currently, the socket types that support setting the packet priority
are :cpp:class:`ns3::UdpSocketImpl`, :cpp:class:`ns3::TcpSocketBase` and
:cpp:class:`ns3::PacketSocket`. The packet priority is used, e.g., by queuing
disciplines such as the default PfifoFastQueueDisc to classify packets into
distinct queues.

Setting the priority with UDP sockets
######################################

If the packet is an IPv4 packet and the value to be inserted in the ToS field
is not null, then the packet is assigned a priority based on such ToS value
(according to the :cpp:func:`ns3::Socket::IpTos2Priority` function). Otherwise,
the priority associated with the socket is assigned to the packet.

Setting the priority with TCP sockets
######################################

Every packet is assigned a priority equal to the priority associated with the
socket.

Setting the priority with packet sockets
#########################################

Every packet is assigned a priority equal to the priority associated with the
socket.

Socket errno
************

*to be completed*

Example programs
****************

*to be completed*

POSIX-like sockets API
**********************
