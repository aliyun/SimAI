.. include:: replace.txt
.. highlight:: cpp

Fifo queue disc
---------------------

Model Description
*****************

FifoQueueDisc implements the FIFO (First-In First-Out) policy.
Packets are enqueued in the unique internal queue, which is implemented
as a DropTail queue. The queue disc capacity can be specified in terms of
either packets or bytes, depending on the value of the Mode attribute.

User is allowed to provide an internal queue before the queue disc is initialized.
If no internal queue is provided, one DropTail queue having the same capacity
of the queue disc is created by default. No packet filter can be added to a
FifoQueueDisc.


Attributes
==========

The FifoQueueDisc class holds the following attribute:

* ``MaxSize:`` The maximum number of packets/bytes the queue disc can hold. The default value is 1000 packets.


Validation
**********

The fifo model is tested using :cpp:class:`FifoQueueDiscTestSuite` class defined
in ``src/traffic-control/test/fifo-queue-disc-test-suite.cc``. The test aims to
check that the capacity of the queue disc is not exceeded and packets are dequeued
in the correct order.
