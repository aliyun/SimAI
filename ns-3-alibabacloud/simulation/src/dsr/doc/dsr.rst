.. include:: replace.txt

DSR Routing
-----------

Dynamic Source Routing (DSR) protocol is a reactive routing protocol designed specifically 
for use in multi-hop wireless ad hoc networks of mobile nodes.

This model was developed by 
`the ResiliNets research group <http://www.ittc.ku.edu/resilinets>`_
at the University of Kansas.  

DSR Routing Overview
********************
 
This model implements the base specification of the Dynamic Source Routing 
(DSR) protocol. Implementation is based on :rfc:`4728`, with some extensions
and modifications to the RFC specifications.

DSR operates on a on-demand behavior. Therefore, our DSR model buffers all 
packets while a route request packet (RREQ) is disseminated. We implement 
a packet buffer in dsr-rsendbuff.cc. The packet queue implements 
garbage collection of old packets and a queue size limit. When the packet 
is sent out from the send buffer, it will be queued in maintenance buffer 
for next hop acknowledgment.

The maintenance buffer then buffers the already sent out packets and waits 
for the notification of packet delivery.  Protocol operation strongly 
depends on broken link detection mechanism. We implement the three heuristics 
recommended based the RFC as follows:

First, we use link layer feedback when possible, which is also the fastest 
mechanism of these three to detect link errors. A link is considered to be 
broken if frame transmission results in a transmission failure for all 
retries. This mechanism is meant for active links and works much faster 
than in its absence.  DSR is able to 
detect the link layer transmission failure and notify that as broken.  
Recalculation of routes will be triggered
when needed.  If user does not want to use link layer acknowledgment, 
it can be tuned by setting "LinkAcknowledgment" attribute to false in 
"dsr-routing.cc".

Second, passive acknowledgment should be used whenever possible. The node 
turns on "promiscuous" receive mode, in which it can receive packets not 
destined for itself, and when the node assures the delivery of that data 
packet to its destination, it cancels the passive acknowledgment timer.

Last, we use a network layer acknowledge scheme to notify the receipt of 
a packet. Route request packet will not be acknowledged or retransmitted.

The Route Cache implementation support garbage collection of old entries 
and state machine, as defined in the 
standard.  It implements as a STL map container. The key is the 
destination IP address.

DSR operates with direct access to IP header, and operates between network 
and transport layer.  When packet is sent out from transport layer, it 
passes itself to DSR and DSR header is appended.

We have two caching mechanisms: path cache and link cache.  The path cache 
saves the whole path in the cache.  The paths are sorted based on the 
hop count, and whenever one path is not able to be used, we change to the 
next path.  The link cache is a slightly better design in the sense that it 
uses different subpaths and uses Implemented Link Cache using 
Dijkstra algorithm, and this part is implemented by 
Song Luan <lsuper@mail.ustc.edu.cn>. 

The following optional protocol optimizations aren't implemented:

* Flow state
* First Hop External (F), Last Hop External (L) flags
* Handling unknown DSR options
* Two types of error headers:
     1. flow state not supported option
     2. unsupported option (not going to happen in simulation)

DSR update in ns-3.17
=====================

We originally used "TxErrHeader" in Ptr<WifiMac> to indicate the transmission 
error of a specific packet in link layer, however, it was not working 
quite correctly since even when the packet was dropped, this header was 
not recorded in the trace file.  We used to a different path on implementing 
the link layer notification mechanism.  We look into the trace file by 
finding packet receive event.  If we find one receive event for the 
data packet, we count that as the indicator for successful data delivery.

Useful parameters
=================

:: 

   +------------------------- +------------------------------------+-------------+
   | Parameter                | Description                        | Default     |
   +==========================+====================================+=============+
   | MaxSendBuffLen           | Maximum number of packets that can | 64          |
   |                          | be stored in send buffer           |             | 
   +------------------------- +------------------------------------+-------------+
   | MaxSendBuffTime          | Maximum time packets can be queued | Seconds(30) |
   |                          | in the send buffer                 |             |
   +------------------------- +------------------------------------+-------------+
   | MaxMaintLen              | Maximum number of packets that can | 50          |
   |                          | be stored in maintenance buffer    |             |
   +------------------------- +------------------------------------+-------------+
   | MaxMaintTime             | Maximum time packets can be queued | Seconds(30) |
   |                          | in maintenance buffer              |             |
   +------------------------- +------------------------------------+-------------+
   | MaxCacheLen              | Maximum number of route entries    | 64          |
   |                          | that can be stored in route cache  |             |
   +------------------------- +------------------------------------+-------------+
   | RouteCacheTimeout        | Maximum time the route cache can   | Seconds(300)|
   |                          | be queued in route cache           |             |
   +------------------------- +------------------------------------+-------------+
   | RreqRetries              | Maximum number of retransmissions  | 16          |
   |                          | for request discovery of a route   |             |
   +------------------------- +------------------------------------+-------------+
   | CacheType                | Use Link Cache or use Path Cache   | "LinkCache" |
   |                          |                                    |             |
   +------------------------- +------------------------------------+-------------+
   | LinkAcknowledgment       | Enable Link layer acknowledgment   | True        |
   |                          | mechanism                          |             | 
   +------------------------- +------------------------------------+-------------+

Implementation modification
===========================

* The DsrFsHeader has added 3 fields: message type, source id, destination id, and these changes only for post-processing
	1. Message type is used to identify the data packet from control packet
	2. source id is used to identify the real source of the data packet since we have to deliver the packet hop-by-hop and the Ipv4Header is not carrying the real source and destination ip address as needed
	3. destination id is for same reason of above
* Route Reply header is not word-aligned in DSR RFC, change it to word-aligned in implementation
* DSR works as a shim header between transport and network protocol, it needs its own forwarding mechanism, we are changing the packet transmission to hop-by-hop delivery, so we added two fields in dsr fixed header to notify packet delivery


Current Route Cache implementation
==================================

This implementation used "path cache", which is simple to implement and ensures loop-free paths:

* the path cache has automatic expire policy
* the cache saves multiple route entries for a certain destination and sort the entries based on hop counts
* the MaxEntriesEachDst can be tuned to change the maximum entries saved for a single destination
* when adding multiple routes for one destination, the route is compared based on hop-count and expire time, the one with less hop count or relatively new route is favored
* Future implementation may include "link cache" as another possibility

DSR Instructions
****************

The following should be kept in mind when running DSR as routing protocol:

* NodeTraversalTime is the time it takes to traverse two neighboring nodes and should be chosen to fit the transmission range
* PassiveAckTimeout is the time a packet in maintenance buffer wait for passive acknowledgment, normally set as two times of NodeTraversalTime
* RouteCacheTimeout should be set smaller value when the nodes' velocity become higher. The default value is 300s.
   
Helper
******

To have a node run DSR, the easiest way would be to use the DsrHelper
and DsrMainHelpers in your simulation script. For instance:

.. sourcecode:: cpp

  DsrHelper dsr;
  DsrMainHelper dsrMain;
  dsrMain.Install (dsr, adhocNodes);

The example scripts inside ``src/dsr/examples/`` demonstrate the use of DSR based nodes in different scenarios. 
The helper source can be found inside ``src/dsr/helper/dsr-main-helper.{h,cc}`` 
and ``src/dsr/helper/dsr-helper.{h,cc}``

Examples
********

The example can be found in ``src/dsr/examples/``:

* dsr.cc use DSR as routing protocol within a traditional MANETs environment[3].

DSR is also built in the routing comparison case in ``examples/routing/``:

* ``manet-routing-compare.cc`` is a comparison case with built in MANET routing protocols and can generate its own results.

Validation
**********

This model has been tested as follows:

* Unit tests have been written to verify the internals of DSR. This can be found in ``src/dsr/test/dsr-test-suite.cc``. These tests verify whether the methods inside DSR module which deal with packet buffer, headers work correctly.
* Simulation cases similar to [3] have been tested and have comparable results.
* manet-routing-compare.cc has been used to compare DSR with three of other routing protocols.

A paper was presented on these results at the Workshop on ns-3 in 2011.

Limitations
***********

The model is not fully compliant with :rfc:`4728`. As an example, Dsr fixed size header
has been extended and it is four octets longer then the RFC specification.
As a consequence, the DSR headers can not be correctly decoded by Wireshark.

The model full compliance with the RFC is planned for the future.

References
**********

[1] Original paper: http://www.monarch.cs.rice.edu/monarch-papers/dsr-chapter00.pdf

[2] RFC 4728 http://www6.ietf.org/rfc/rfc4728.txt

[3] Broch's comparison paper: http://www.monarch.cs.rice.edu/monarch-papers/mobicom98.ps
