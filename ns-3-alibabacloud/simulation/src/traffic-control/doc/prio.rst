.. include:: replace.txt
.. highlight:: cpp

Prio queue disc
---------------------

Model Description
*****************

PrioQueueDisc implements a strict priority policy, where packets are dequeued from
a band only if higher priority bands are all empty. PrioQueueDisc is a classful
queue disc and can have an arbitrary number of bands, each of which is handled by a
queue disc of any kind. The capacity of PrioQueueDisc is not limited; packets can
only be dropped by child queue discs (which may have a limited capacity).
If no packet filter is installed or able to classify a packet, then the
packet is enqueued into a priority band based on its priority (modulo 16), which
is used as an index into an array called priomap. Users can read :ref:`Socket-options`
for details on how to set the packet priority. If a packet is classified
by an installed packet filter and the returned value ``i`` is non-negative and less than the
number of priority bands, then the packet is enqueued into the ``i``-th priority band.
Otherwise, the packet is enqueued into the priority band specified by the first element
of the priomap array.

If no queue disc class is added by the user before the queue disc is initialized,
three child queue discs of type FifoQueueDisc are automatically added. It has to
be noted that PrioQueueDisc needs at least two child queue discs.


Attributes
==========

The PrioQueueDisc class holds the following attribute:

* ``Priomap:`` The priority to band mapping. The default value is the same mapping as the (fixed) one used by PfifoFastQueueDisc.

Examples
========

An example of how to configure PrioQueueDisc with custom child queue discs and priomap
is provided by `queue-discs-benchmark.cc` located in ``examples/traffic-control``::

  TrafficControlHelper tch;
  uint16_t handle = tch.SetRootQueueDisc ("ns3::PrioQueueDisc", "Priomap", StringValue ("0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1"));
  TrafficControlHelper::ClassIdList cid = tch.AddQueueDiscClasses (handle, 2, "ns3::QueueDiscClass");
  tch.AddChildQueueDisc (handle, cid[0], "ns3::FifoQueueDisc");
  tch.AddChildQueueDisc (handle, cid[1], "ns3::RedQueueDisc");

The code above adds two classes (bands) to a PrioQueueDisc. The highest priority one
is a FifoQueueDisc, the other one is a RedQueueDisc. The attribute Priomap is set to
an array containing only 0 and 1 (since PrioQueueDisc only has two bands).


Validation
**********

PrioQueueDisc is tested using :cpp:class:`PrioQueueDiscTestSuite` class defined
in ``src/traffic-control/test/prio-queue-disc-test-suite.cc``. The test aims to
check that: i) packets are enqueued in the correct band based on their priority and
the priomap or according to the value returned by the installed packet filter;
ii) packets are dequeued in the correct order.

The test suite can be run using the following commands:

.. sourcecode:: bash

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s prio-queue-disc

or

.. sourcecode:: bash

  $ NS_LOG="PrioQueueDisc" ./ns3 run "test-runner --suite=prio-queue-disc"
