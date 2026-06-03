.. include:: replace.txt
.. highlight:: cpp
.. highlight:: bash

FQ-PIE queue disc
-----------------

This chapter describes the FQ-PIE ([Ram19]_) queue disc implementation in |ns3|.

The FQ-PIE queue disc combines the Proportional Integral Controller Enhanced
(PIE) AQM algorithm with the FlowQueue scheduler that is part of FQ-CoDel
(also available in |ns3|). FQ-PIE was introduced to Linux kernel version 5.6. 

Model Description
*****************

The source code for the ``FqPieQueueDisc`` is located in the directory
``src/traffic-control/model`` and consists of 2 files `fq-pie-queue-disc.h`
and `fq-pie-queue-disc.cc` defining a FqPieQueueDisc class and a helper
FqPieFlow class. The code was ported to |ns3| based on Linux kernel code
implemented by Mohit P. Tahiliani.

This model calculates drop probability independently in each flow queue.
One difficulty, as pointed out by [CableLabs14]_, is that PIE calculates
drop probability based on the departure rate of a (flow) queue, which may
be more highly variable than the aggregate queue.  An alternative, which 
CableLabs has called SFQ-PIE, is to calculate an overall drop probability
for the entire queue structure, and then scale this drop probability based
on the ratio of the queue depth of each flow queue compared with the depth
of the current largest queue.  This ns-3 model does not implement the
SFQ-PIE variant described by CableLabs.

References
==========

.. [Ram19] G. Ramakrishnan, M. Bhasi, V. Saicharan, L. Monis, S. D. Patil and M. P. Tahiliani, "FQ-PIE Queue Discipline in the Linux Kernel: Design, Implementation and Challenges," 2019 IEEE 44th LCN Symposium on Emerging Topics in Networking (LCN Symposium), Osnabrueck, Germany, 2019, pp. 117-124,

.. [CableLabs14] G. White, Active Queue Management in DOCSIS 3.X Cable Modems, CableLabs white paper, May 2014.

Attributes
==========

The key attributes that the FqPieQueue class holds include the following.
First, there are PIE-specific attributes that are copied into the individual
PIE flow queues:

* ``UseEcn:`` Whether to use ECN marking
* ``MarkEcnThreshold:`` ECN marking threshold (RFC 8033 suggests 0.1 (i.e., 10%) default).
* ``UseL4s:`` Whether to use L4S (only mark ECT1 packets at CE threshold)
* ``MeanPktSize:`` Constant used to roughly convert bytes to packets
* ``A:`` Alpha value in PIE algorithm drop probability calculation
* ``B:`` Beta value in PIE algorithm drop probability calculation
* ``Tupdate:`` Time period to calculate drop probability
* ``Supdate:`` Start time of the update timer
* ``DequeueThreshold:`` Minimum queue size in bytes before dequeue rate is measured
* ``QueueDelayReference:`` AQM latency target
* ``MaxBurstAllowance:`` AQM max burst allowance before random drop
* ``UseDequeueRateEstimator:`` Enable/Disable usage of Dequeue Rate Estimator
* ``UseCapDropAdjustment:`` Enable/Disable Cap Drop Adjustment feature mentioned in RFC 8033
* ``UseDerandomization:`` Enable/Disable Derandomization feature mentioned in RFC 8033

Second, there are QueueDisc level, or FQ-specific attributes::
* ``MaxSize:`` Maximum number of packets in the queue disc
* ``Flows:`` Maximum number of flow queues
* ``DropBatchSize:`` Maximum number of packets dropped from the fat flow
* ``Perturbation:`` Salt value used as hash input when classifying flows
* ``EnableSetAssociativeHash:`` Enable or disable set associative hash
* ``SetWays:`` Size of a set of queues in set associative hash

Examples
========

A typical usage pattern is to create a traffic control helper and to configure 
the type and attributes of queue disc and filters from the helper. For example, FqPIE
can be configured as follows:

.. sourcecode:: cpp

  TrafficControlHelper tch;
  tch.SetRootQueueDisc ("ns3::FqPieQueueDisc", "DropBatchSize", UintegerValue (1)
                                                 "Perturbation", UintegerValue (256));
  QueueDiscContainer qdiscs = tch.Install (devices);

Validation
**********

The FqPie model is tested using :cpp:class:`FqPieQueueDiscTestSuite` class defined in `src/test/ns3tc/fq-pie-queue-test-suite.cc`.

The tests are similar to the ones for FqCoDel queue disc mentioned in first section of this document.
The test suite can be run using the following commands::

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s fq-pie-queue-disc

or::

  $ NS_LOG="FqPieQueueDisc" ./ns3 run "test-runner --suite=fq-pie-queue-disc"
