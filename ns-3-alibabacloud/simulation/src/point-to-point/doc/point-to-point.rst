.. include:: replace.txt
.. highlight:: cpp

PointToPoint NetDevice
----------------------

This is the introduction to PointToPoint NetDevice chapter, to complement the
PointToPoint model doxygen.

Overview of the PointToPoint model
**********************************

The |ns3| point-to-point model is of a very simple point to point data link
connecting exactly two PointToPointNetDevice devices over an
PointToPointChannel. This can be viewed as equivalent to a full duplex RS-232 or
RS-422 link with null modem and no handshaking.

Data is encapsulated in the Point-to-Point Protocol (PPP -- RFC 1661), however
the Link Control Protocol (LCP) and associated state machine is not implemented.
The PPP link is assumed to be established and authenticated at all times.

Data is not framed, therefore Address and Control fields will not be found.
Since the data is not framed, there is no need to provide Flag Sequence and
Control Escape octets, nor is a Frame Check Sequence appended. All that is
required to implement non-framed PPP is to prepend the PPP protocol number for
IP Version 4 which is the sixteen-bit number 0x21 (see
`<http://www.iana.org/assignments/ppp-numbers>`_).

The PointToPointNetDevice provides following Attributes:

* Address:  The ns3::Mac48Address of the device (if desired);
* DataRate:  The data rate (ns3::DataRate) of the device;
* TxQueue:  The transmit queue (ns3::Queue) used by the device;
* InterframeGap:  The optional ns3::Time to wait between "frames";
* Rx:  A trace source for received packets;
* Drop:  A trace source for dropped packets.

The PointToPointNetDevice models a transmitter section that puts bits on a
corresponding channel "wire." The DataRate attribute specifies the number of
bits per second that the device will simulate sending over the channel. In
reality no bits are sent, but an event is scheduled for an elapsed time
consistent with the number of bits in each packet and the specified DataRate.
The implication here is that the receiving device models a receiver section that
can receive any any data rate. Therefore there is no need, nor way to set a
receive data rate in this model. By setting the DataRate on the transmitter of
both devices connected to a given PointToPointChannel one can model a symmetric
channel; or by setting different DataRates one can model an asymmetric channel
(e.g., ADSL).

The PointToPointNetDevice supports the assignment of a "receive error model."
This is an ErrorModel object that is used to simulate data corruption on the
link.

Point-to-Point Channel Model
****************************

The point to point net devices are connected via an PointToPointChannel. This
channel models two wires transmitting bits at the data rate specified by the
source net device. There is no overhead beyond the eight bits per byte of the
packet sent. That is, we do not model Flag Sequences, Frame Check Sequences nor
do we "escape" any data.

The PointToPointChannel provides following Attributes:


* Delay:  An ns3::Time specifying the propagation delay for the channel.

Using the PointToPointNetDevice
*******************************

The PointToPoint net devices and channels are typically created and configured
using the associated ``PointToPointHelper`` object. The various ns3 device
helpers generally work in a similar way, and their use is seen in many of our
example programs and is also covered in the |ns3| tutorial.

The conceptual model of interest is that of a bare computer "husk" into which
you plug net devices. The bare computers are created using a ``NodeContainer``
helper. You just ask this helper to create as many computers (we call them
``Nodes``) as you need on your network::

  NodeContainer nodes;
  nodes.Create (2);

Once you have your nodes, you need to instantiate a ``PointToPointHelper`` and
set any attributes you may want to change. Note that since this is a
point-to-point (as compared to a point-to-multipoint) there may only be two
nodes with associated net devices connected by a PointToPointChannel.::

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
 
Once the attributes are set, all that remains is to create the devices and
install them on the required nodes, and to connect the devices together using a
PointToPoint channel. When we create the net devices, we add them to a container
to allow you to use them in the future. This all takes just one line of code.::

  NetDeviceContainer devices = pointToPoint.Install (nodes);

PointToPoint Tracing
********************

Like all |ns3| devices, the Point-to-Point Model provides a number of trace
sources. These trace sources can be hooked using your own custom trace code, or
you can use our helper functions to arrange for tracing to be enabled on devices
you specify.

Upper-Level (MAC) Hooks
+++++++++++++++++++++++

From the point of view of tracing in the net device, there are several
interesting points to insert trace hooks. A convention inherited from other
simulators is that packets destined for transmission onto attached networks pass
through a single "transmit queue" in the net device. We provide trace hooks at
this point in packet flow, which corresponds (abstractly) only to a transition
from the network to data link layer, and call them collectively
the device MAC hooks.

When a packet is sent to the Point-to-Point net device for transmission it
always passes through the transmit queue. The transmit queue in the
PointToPointNetDevice inherits from Queue, and therefore inherits three trace
sources:

* An Enqueue operation source (see ns3::Queue::m_traceEnqueue);
* A Dequeue operation source (see ns3::Queue::m_traceDequeue);
* A Drop operation source (see ns3::Queue::m_traceDrop).

The upper-level (MAC) trace hooks for the PointToPointNetDevice are, in fact, 
exactly these three trace sources on the single transmit queue of the device.  

The m_traceEnqueue event is triggered when a packet is placed on the transmit
queue. This happens at the time that ns3::PointtoPointNetDevice::Send or 
ns3::PointToPointNetDevice::SendFrom is called by a higher layer to queue a 
packet for transmission. An Enqueue trace event firing should be interpreted
as only indicating that a higher level protocol has sent a packet to the device.

The m_traceDequeue event is triggered when a packet is removed from the transmit
queue. Dequeues from the transmit queue can happen in two situations:  1) If the
underlying channel is idle when PointToPointNetDevice::Send is called, a packet
is dequeued from the transmit queue and immediately transmitted;  2) a packet
may be dequeued and immediately transmitted in an internal TransmitCompleteEvent
that functions much  like a transmit complete interrupt service routine. An
Dequeue trace event firing may be viewed as indicating that the
PointToPointNetDevice has begun transmitting a packet.

Lower-Level (PHY) Hooks
+++++++++++++++++++++++

Similar to the upper level trace hooks, there are trace hooks available at the
lower levels of the net device. We call these the PHY hooks. These events fire
from the device methods that talk directly to the 
PointToPointChannel.

The trace source m_dropTrace is called to indicate a packet that is dropped by
the device. This happens when a packet is discarded as corrupt due to a receive
error model indication (see ns3::ErrorModel and the associated attribute
"ReceiveErrorModel").

The other low-level trace source fires on reception of a packet (see
ns3::PointToPointNetDevice::m_rxTrace) from the PointToPointChannel.
