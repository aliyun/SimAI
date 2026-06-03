.. include:: replace.txt


Cobalt queue disc
-----------------

This chapter describes the COBALT (CoDel BLUE Alternate) ([Cake16]_) queue disc 
implementation in |ns3|.

COBALT queue disc is an integral component of CAKE smart queue management system.
It is a combination of the CoDel ([Kath17]_) and BLUE ([BLUE02]_) Active Queue
Management algorithms.


Model Description
*****************

The source code for the COBALT model is located in the directory
``src/traffic-control/model`` and consists of 2 files: `cobalt-queue-disc.h` and
`cobalt-queue-disc.cc` defining a CobaltQueueDisc class and a helper
CobaltTimestampTag class. The code was ported to |ns3| by Vignesh Kanan,
Harsh Lara, Shefali Gupta, Jendaipou Palmei and Mohit P. Tahiliani based on 
the Linux kernel code.

Stefano Avallone and Pasquale Imputato helped in verifying the correctness of 
COBALT model in |ns3| by comparing the results obtained from it to those obtained
from the Linux model of COBALT. A detailed comparison of ns-3 model of COBALT with
Linux model of COBALT is provided in ([Cobalt19]_).

* class :cpp:class:`CobaltQueueDisc`: This class implements the main
  Cobalt algorithm:
* ``CobaltQueueDisc::DoEnqueue ()``: This routine tags a packet with the
  current time before pushing it into the queue.  The timestamp tag is used by
  ``CobaltQueue::DoDequeue()`` to compute the packet's sojourn time.  If the
  queue is full upon the packet arrival, this routine will drop the packet and
  record the number of drops due to queue overflow, which is stored in
  `m_stats.qLimDrop`.
* ``CobaltQueueDisc::ShouldDrop ()``: This routine is
  ``CobaltQueueDisc::DoDequeue()``'s helper routine that determines whether a
  packet should be dropped or not based on its sojourn time. If L4S mode is 
  enabled then if the packet is ECT1 is checked and if delay is greater than
  CE threshold then the packet is marked and returns ``false``.
  If the sojourn time goes above `m_target` and remains above continuously
  for at least `m_interval`, the routine returns ``true`` indicating that it
  is OK to drop the packet. ``Otherwise, it returns ``false``.  If L4S mode
  is turned off and CE threshold marking is enabled, then if the delay is
  greater than CE threshold, packet is marked. This routine
  decides if a packet should be dropped based on the dropping state of
  CoDel and drop probability of BLUE.  The idea is to have both algorithms
  running in parallel and their effectiveness is decided by their respective
  parameters (Pdrop of BLUE and dropping state of CoDel). If either of them
  decide to drop the packet, the packet is dropped.
* ``CobaltQueueDisc::DoDequeue ()``: This routine performs the actual packet
  ``drop based on ``CobaltQueueDisc::ShouldDrop ()``'s return value and
  schedules the next drop. Cobalt will decrease BLUE's drop probability
  if the queue is empty. This will ensure that the queue does not underflow.
  Otherwise Cobalt will take the next packet from the queue and calculate
  its drop state by running CoDel and BLUE in parallel till there are none
  left to drop.



References
==========

.. [Cake16] Linux implementation of Cobalt as a part of the cake framework.
   Available online at
   `<https://github.com/dtaht/sch_cake/blob/master/cobalt.c>`_.

.. [Kath17] Controlled Delay Active Queue Management
   (draft-ietf-aqm-fq-codel-07)
   Available online at
   `<https://tools.ietf.org/html/draft-ietf-aqm-codel-07>`_.

.. [BLUE02] Feng, W. C., Shin, K. G., Kandlur, D. D., & Saha, D. (2002).
   The BLUE
   Active Queue Management Algorithms. IEEE/ACM Transactions on Networking
   (ToN), 10(4), 513-528.

.. [Cobalt19] Jendaipou Palmei, Shefali Gupta, Pasquale Imputato, Jonathan
   Morton,  Mohit P. Tahiliani, Stefano Avallone and Dave Taht (2019).
   Design and Evaluation of COBALT Queue Discipline. IEEE International
   Symposium on Local and Metropolitan Area Networks (LANMAN), July 2019.


Attributes
==========

The key attributes that the CobaltQueue Disc class holds include the following:

* ``MaxSize:`` The maximum number of packets/bytes accepted by this queue disc.
* ``Interval:`` The sliding-minimum window. The default value is 100 ms.
* ``Target:`` The Cobalt algorithm target queue delay. The default value is 5 ms.
* ``Pdrop:`` Value of drop probability.
* ``Increment:`` Increment value of drop probability. Default value is 1./256 .
* ``Decrement:`` Decrement value of drop probability. Default value is 1./4096 .
* ``CeThreshold:`` The CoDel CE threshold for marking packets.
* ``UseL4s:`` True to use L4S (only ECT1 packets are marked at CE threshold).
* ``Count:`` Cobalt count.
* ``DropState:`` Dropping state of Cobalt. Default value is false.
* ``Sojourn:`` Per packet time spent in the queue.
* ``DropNext:`` Time until next packet drop.

Examples
========

An example program named `cobalt-vs-codel.cc` is located in
``src/traffic-control/examples``. Use the following command to run the program.

::

   $ ./ns3 run cobalt-vs-codel


Validation
**********

The COBALT model is tested using :cpp:class:`CobaltQueueDiscTestSuite` class
defined in `src/traffic-control/test/cobalt-queue-test-suite.cc`.
The suite includes 2 test cases:

* Test 1: Simple enqueue/dequeue with no drops.
* Test 2: Change of BLUE's drop probability upon queue full
  (Activation of Blue).
* Test 3: This test verfies ECN marking.
* Test 4: CE threshold marking test.

The test suite can be run using the following commands:

::

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s cobalt-queue-disc

or

::

  $ NS_LOG="CobaltQueueDisc" ./ns3 run "test-runner --suite=cobalt-queue-disc"

