.. include:: replace.txt
.. highlight:: cpp
.. highlight:: bash

Mq queue disc
------------------

This chapter describes the mq queue disc implementation in |ns3|.

mq is a classful multiqueue dummy scheduler developed to best fit the multiqueue
traffic control API in Linux. The mq scheduler presents device transmission queues as
classes, allowing to attach different queue discs to them, which are grafted to the
device transmission queues.

Mq is installed by default on multi-queue devices (such as Wifi) with as many FqCodel
child queue discs as the number of device queues.

Model Description
*****************

mq is a multi-queue aware queue disc, meaning that it has as many child queue discs as
the number of device transmission queues. Each child queue disc maps to a distinct
device transmission queue. Every packet is enqueued into the child queue disc which
maps to the device transmission queue in which the device will enqueue
the packet.

In |ns3|, :cpp:class:`MqQueueDisc` has a wake mode of WAKE_CHILD, which means that the
traffic control layer enqueues packets directly into one of the child queue discs
(multi-queue devices can provide a callback to inform the traffic control layer of
the device transmission queue that will be selected for a given packet). Therefore,
``MqQueueDisc::DoEnqueue ()`` shall never be called (in fact, it raises a fatal error).
Given that dequeuing packets is triggered by enqueuing a packet in the queue disc or
by the device invoking the wake callback, it turns out that ``MqQueueDisc::DoDequeue ()``
is never called as well (in fact, it raises a fatal error, too).

The mq queue disc does not require packet filters, does not admit internal queues
and must have as many child queue discs as the number of device transmission queues.

Examples
========

A typical usage pattern is to create a traffic control helper used to add the required number of
queue disc classes, attach child queue discs to the classes and (if needed) add packet filters to the
child queue discs. The following code shows how to install an mq queue disc having FqCodel child queue
discs:

.. sourcecode:: cpp

  TrafficControlHelper tch;
  uint16_t handle = tch.SetRootQueueDisc ("ns3::MqQueueDisc");
  TrafficControlHelper::ClassIdList cls = tch.AddQueueDiscClasses (handle, numTxQueues, "ns3::QueueDiscClass");
  tch.AddChildQueueDiscs (handle, cls, "ns3::FqCoDelQueueDisc");
  QueueDiscContainer qdiscs = tch.Install (devices);

Note that the child queue discs attached to the classes do not necessarily have to be of the same type.

Validation
**********

The mq model is tested using :cpp:class:`WifiAcMappingTestSuite` class defined in
`src/test/wifi-ac-mapping-test-suite.cc`. The suite considers a node with a QoS-enabled
wifi device (which has 4 transmission queues) and includes 4 test cases:

* Test 1: EF-marked packets are enqueued in the queue disc which maps to the AC_VI queue
* Test 2: AF11-marked packets are enqueued in the queue disc which maps to the AC_BK queue
* Test 3: AF32-marked packets are enqueued in the queue disc which maps to the AC_BE queue
* Test 4: CS7-marked packets are enqueued in the queue disc which maps to the AC_VO queue

The test suite can be run using the following commands:

::

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s ns3-wifi-ac-mapping

or

::

  $ NS_LOG="WifiAcMappingTest" ./ns3 run "test-runner --suite=ns3-wifi-ac-mapping"

