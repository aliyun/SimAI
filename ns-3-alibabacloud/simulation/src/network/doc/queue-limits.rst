Queue limits
------------

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

This section documents the queue limits model, which is used by the traffic control
to limit the NetDevices queueing delay. It operates on the transmission path of
the network node.

The reduction of the NetDevices queueing delay is essential to improve the effectiveness of
Active Queue Management (AQM) algorithms.
Careful assessment of the queueing delay includes a byte-based measure of the NetDevices
queue length. In this design, traffic control can use different byte-based schemes to
limit the queueing delay. Currently the only available scheme is DynamicQueueLimits, which is
modelled after the dynamic queue limit library of Linux.

Model Description
*****************

The source code for the model lives in the directory ``src/network/utils``.

The model allows a byte-based measure of the netdevice queue. The byte-based measure
more accurately approximates the time required to empty the queue than a packet-based measure.

To inform the upper layers about the transmission of packets, NetDevices can call a couple
of functions:

* ``void NotifyQueuedBytes (uint32_t bytes)``: Report the number of bytes queued to the device queue
* ``void NotifyTransmittedBytes (uint32_t bytes)``: Report the number of bytes transmitted by device

Based on this information, the QueueLimits object can stop the transmission queue.

In case of multiqueue NetDevices this mechanism is available for each queue.

The QueueLimits model can be used on any NetDevice modelled in ns-3.

Design
======

An abstract base class, class QueueLimits, is subclassed for specific
byte-based limiting strategies.

Common operations provided by the base class QueueLimits include:

* ``void Reset ()``:  Reset queue limits state
* ``void Completed (uint32_t count)``:  Record the number of completed bytes and recalculate the limit
* ``int32_t Available () const``:  Return how many bytes can be queued
* ``void Queued (uint32_t count)``:  Record number of bytes queued

DynamicQueueLimits
##################

Dynamic queue limits (DQL) is a basic library implemented in the Linux kernel to limit the Ethernet
queueing delay. DQL is a general purpose queue length controller. The goal of DQL is to calculate
the limit as the minimum number of bytes needed to prevent starvation.

Three attributes are defined in the DynamicQueueLimits class:

* ``HoldTime``: The DQL algorithm hold time
* ``MaxLimit``: Maximum limit
* ``MinLimit``: Minimum limit

The DQL algorithm hold time is 1 s. Reducing the HoldTime increases the responsiveness of
DQL with consequent greater number of limit variation events. Conversely, increasing the HoldTime
decreases the responsiveness of DQL with a minor number of limit variation events.
The limit calculated by DQL is in the range from MinLimit to MaxLimit.
The default values are respectively 0 and DQL_MAX_LIMIT.
Increasing the MinLimit is recommended in case of higher NetDevice transmission rate (e.g. 1 Gbps)
while reducing the MaxLimit is recommended in case of lower NetDevice transmission rate (e.g. 500 Kbps).

There is one trace source in DynamicQueueLimits class that may be hooked:

* ``Limit``: Limit value calculated by DQL

Usage
*****

Helpers
=======

A typical usage pattern is to create a traffic control helper and configure
the queue limits type and attributes from the helper, such as this example:

.. sourcecode:: cpp

  TrafficControlHelper tch;
  uint32_t handle = tch.SetRootQueueDisc ("ns3::PfifoFastQueueDisc", "Limit", UintegerValue (1000));

  tch.SetQueueLimits ("ns3::DynamicQueueLimits", "HoldTime", StringValue ("4ms"));

then install the configuration on a NetDevices container

.. sourcecode:: cpp

  tch.Install (devices);
