Queues
------

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

This section documents the queue object, which is typically used by NetDevices
and QueueDiscs to store packets.

Packets stored in a queue can be managed according to different policies.
Currently, only the DropTail policy is available.

Model Description
*****************

The source code for the new module lives in the directory ``src/network/utils``.

ns3::Queue has been redesigned as a template class object to allow us to
instantiate queues storing different types of items. The unique template
type parameter specifies the type of items stored in the queue.
The only requirement on the item type is that it must provide a GetSize ()
method which returns the size of the packet included in the item.
Currently, queue items can be objects of the following classes:

* Packet
* QueueItem and subclasses (e.g., QueueDiscItem)
* WifiMacQueueItem

The internal queues of the queue discs are of type Queue<QueueDiscItem>
(an alias of which being InternalQueue). A number of network devices
(SimpleNetDevice, PointToPointNetDevice, CsmaNetDevice) use a Queue<Packet>
to store packets to be transmitted. WifiNetDevices use instead queues of
type WifiMacQueue, which is a subclass of Queue storing objects of
type WifiMacQueueItem. Other devices, such as WiMax and LTE, use specialized
queues.

Design
======

The Queue class derives from the QueueBase class, which is a non-template
class providing all the methods that are independent of the type of the items
stored in the queue. The Queue class provides instead all the operations that
depend on the item type, such as enqueue, dequeue, peek and remove. The Queue
class also provides the ability to trace certain queue operations such as
enqueuing, dequeuing, and dropping.

Queue is an abstract base class and is subclassed for specific scheduling and
drop policies. Subclasses need to define the following public methods:

* ``bool Enqueue (Ptr<Item> item)``:  Enqueue a packet
* ``Ptr<Item> Dequeue (void)``:  Dequeue a packet
* ``Ptr<Item> Remove (void)``:  Remove a packet
* ``Ptr<const Item> Peek (void)``:  Peek a packet

The Enqueue method does not allow to store a packet if the queue capacity is exceeded.
Subclasses may also define specialized public methods. For instance, the
WifiMacQueue class provides a method to dequeue a packet based on its tid
and MAC address.

There are five trace sources that may be hooked:

* ``Enqueue``
* ``Dequeue``
* ``Drop``
* ``DropBeforeEnqueue``
* ``DropAfterDequeue``

Also, the QueueBase class defines two additional trace sources:

* ``PacketsInQueue``
* ``BytesInQueue``

DropTail
########

This is a basic first-in-first-out (FIFO) queue that performs a tail drop
when the queue is full.

The DropTailQueue class defines one attribute:

* ``MaxSize``: the maximum queue size

Usage
*****

Helpers
=======

A typical usage pattern is to create a device helper and to configure
the queue type and attributes from the helper, such as this example:

.. sourcecode:: cpp

  PointToPointHelper p2p;

  p2p.SetQueue ("ns3::DropTailQueue");
  p2p.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("2ms"));
  NetDeviceContainer devn0n2 = p2p.Install (n0n2);

  p2p.SetQueue ("ns3::DropTailQueue");
  p2p.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("3ms"));
  NetDeviceContainer devn1n2 = p2p.Install (n1n2);

  p2p.SetQueue ("ns3::DropTailQueue",
                "MaxSize", StringValue ("50p"));
  p2p.SetDeviceAttribute ("DataRate", StringValue (linkDataRate));
  p2p.SetChannelAttribute ("Delay", StringValue (linkDelay));
  NetDeviceContainer devn2n3 = p2p.Install (n2n3);

Please note that the SetQueue method of the PointToPointHelper class allows
to specify "ns3::DropTailQueue" instead of "ns3::DropTailQueue<Packet>". The
same holds for CsmaHelper, SimpleNetDeviceHelper and TrafficControlHelper.

Output
======

The ns-3 ascii trace helpers used by many of the NetDevices will hook
the Enqueue, Dequeue, and Drop traces of these queues and print out 
trace statements, such as the following from ``examples/udp/udp-echo.cc``:

.. sourcecode:: text

  + 2 /NodeList/0/DeviceList/1/$ns3::CsmaNetDevice/TxQueue/Enqueue ns3::EthernetHeader 
  ( length/type=0x806, source=00:00:00:00:00:01, destination=ff:ff:ff:ff:ff:ff) 
  ns3::ArpHeader (request source mac: 00-06-00:00:00:00:00:01 source ipv4: 10.1.1.1 
  dest ipv4: 10.1.1.2) Payload (size=18) ns3::EthernetTrailer (fcs=0)
  - 2 /NodeList/0/DeviceList/1/$ns3::CsmaNetDevice/TxQueue/Dequeue ns3::EthernetHeader 
  ( length/type=0x806, source=00:00:00:00:00:01, destination=ff:ff:ff:ff:ff:ff) 
  ns3::ArpHeader (request source mac: 00-06-00:00:00:00:00:01 source ipv4: 10.1.1.1 
  dest ipv4: 10.1.1.2) Payload (size=18) ns3::EthernetTrailer (fcs=0)

which shows an enqueue "+" and dequeue "-" event at time 2 seconds.

Users are, of course, free to define and hook their own trace sinks to
these trace sources.

Examples
========

The drop-tail queue is used in several examples, such as 
``examples/udp/udp-echo.cc``.

