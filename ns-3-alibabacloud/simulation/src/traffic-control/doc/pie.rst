.. include:: replace.txt
.. highlight:: cpp

PIE queue disc
----------------

This chapter describes the PIE ([Pan13]_, [Pan16]_) queue disc implementation 
in |ns3|. 

Proportional Integral controller Enhanced (PIE) is a queuing discipline that aims to
solve the bufferbloat [Buf14]_ problem. The model in ns-3 is a port of Preethi
Natarajan's ns-2 PIE model.


Model Description
*****************

The source code for the PIE model is located in the directory ``src/traffic-control/model``
and consists of 2 files `pie-queue-disc.h` and `pie-queue-disc.cc` defining a PieQueueDisc
class. The code was ported to |ns3| by Mohit P. Tahiliani, Shravya K. S. and Smriti Murali
based on ns-2 code implemented by Preethi Natarajan, Rong Pan, Chiara Piglione, Greg White
and Takashi Hayakawa. The implementation was aligned with RFC 8033 by Vivek Jain and Mohit
P. Tahiliani for the ns-3.32 release, with additional unit test cases contributed by Bhaskar Kataria.

* class :cpp:class:`PieQueueDisc`: This class implements the main PIE algorithm:

  * ``PieQueueDisc::DoEnqueue ()``: This routine checks whether the queue is full, and if so, drops the packets and records the number of drops due to queue overflow. If queue is not full then if ActiveThreshold is set then it checks if queue delay is higher than ActiveThreshold and if it is then, this routine calls ``PieQueueDisc::DropEarly()``, and depending on the value returned, the incoming packet is either enqueued or dropped.

  * ``PieQueueDisc::DropEarly ()``: The decision to enqueue or drop the packet is taken by invoking this routine, which returns a boolean value; false indicates enqueue and true indicates drop.

  * ``PieQueueDisc::CalculateP ()``: This routine is called at a regular interval of `m_tUpdate` and updates the drop probability, which is required by ``PieQueueDisc::DropEarly()``

  * ``PieQueueDisc::DoDequeue ()``: This routine calculates queue delay using timestamps (by default) or, optionally with the `UseDequeRateEstimator` attribute enabled, calculates the average departure rate to estimate queue delay. A queue delay estimate required for updating the drop probability in ``PieQueueDisc::CalculateP ()``. Starting with the ns-3.32 release, the default approach to calculate queue delay has been changed to use timestamps.

References
==========

.. [Pan13] Pan, R., Natarajan, P., Piglione, C., Prabhu, M. S., Subramanian, V., Baker, F., & VerSteeg, B. (2013, July). PIE: A lightweight control scheme to address the bufferbloat problem. In High Performance Switching and Routing (HPSR), 2013 IEEE 14th International Conference on (pp. 148-155). IEEE.  Available online at `<https://www.ietf.org/mail-archive/web/iccrg/current/pdfB57AZSheOH.pdf>`_.

.. [Pan16] R. Pan, P. Natarajan, F. Baker, G. White, B. VerSteeg, M.S. Prabhu, C. Piglione, V. Subramanian, Internet-Draft: PIE: A lightweight control scheme to address the bufferbloat problem, April 2016.  Available online at `<https://tools.ietf.org/html/draft-ietf-aqm-pie-07>`_.

.. comment
   This ref defined in codel.rst:
   [Buf14] Bufferbloat.net.  Available online at `<http://www.bufferbloat.net/>`_.

Attributes
==========

The key attributes that the PieQueue class holds include the following: 

* ``MaxSize:`` The maximum number of bytes or packets the queue can hold.
* ``MeanPktSize:`` Mean packet size in bytes. The default value is 1000 bytes.
* ``Tupdate:`` Time period to calculate drop probability. The default value is 30 ms. 
* ``Supdate:`` Start time of the update timer. The default value is 0 ms. 
* ``DequeueThreshold:`` Minimum queue size in bytes before dequeue rate is measured. The default value is 10000 bytes. 
* ``QueueDelayReference:`` Desired queue delay. The default value is 20 ms. 
* ``MaxBurstAllowance:`` Current max burst allowance in seconds before random drop. The default value is 0.1 seconds.
* ``A:`` Value of alpha. The default value is 0.125.
* ``B:`` Value of beta. The default value is 1.25.
* ``UseDequeueRateEstimator:`` Enable/Disable usage of Dequeue Rate Estimator (Default: false).
* ``UseEcn:`` True to use ECN. Packets are marked instead of being dropped (Default: false).
* ``MarkEcnThreshold:`` ECN marking threshold (Default: 10% as suggested in RFC 8033).
* ``UseDerandomization:`` Enable/Disable Derandomization feature mentioned in RFC 8033 (Default: false).
* ``UseCapDropAdjustment:`` Enable/Disable Cap Drop Adjustment feature mentioned in RFC 8033 (Default: true).
* ``ActiveThreshold:`` Threshold for activating PIE (disabled by default).

Examples
========

The example for PIE is `pie-example.cc` located in ``src/traffic-control/examples``.  To run the file (the first invocation below shows the available
command-line options):

.. sourcecode:: bash

   $ ./ns3 run "pie-example --PrintHelp"
   $ ./ns3 run "pie-example --writePcap=1"

The expected output from the previous commands are ten .pcap files.

Validation
**********

The PIE model is tested using :cpp:class:`PieQueueDiscTestSuite` class defined in `src/traffic-control/test/pie-queue-test-suite.cc`. The suite includes the following test cases:

* Test 1: simple enqueue/dequeue with defaults, no drops
* Test 2: more data with defaults, unforced drops but no forced drops
* Test 3: same as test 2, but with higher QueueDelayReference
* Test 4: same as test 2, but with reduced dequeue rate
* Test 5: same dequeue rate as test 4, but with higher Tupdate
* Test 6: same as test 2, but with UseDequeueRateEstimator enabled
* Test 7: test with CapDropAdjustment disabled
* Test 8: test with CapDropAdjustment enabled
* Test 9: PIE queue disc is ECN enabled, but packets are not ECN capable
* Test 10: Packets are ECN capable, but PIE queue disc is not ECN enabled
* Test 11: Packets and PIE queue disc both are ECN capable
* Test 12: test with Derandomization enabled
* Test 13: same as test 11 but with accumulated drop probability set below the low threshold
* Test 14: same as test 12 but with accumulated drop probability set above the high threshold
* Test 15: Tests Active/Inactive feature, ActiveThreshold set to a high value so PIE never starts.
* Test 16: Tests Active/Inactive feature, ActiveThreshold set to a low value so PIE starts early.

The test suite can be run using the following commands: 

.. sourcecode:: bash

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s pie-queue-disc

or alternatively (to see logging statements in a debug build):  

.. sourcecode:: bash

  $ NS_LOG="PieQueueDisc" ./ns3 run "test-runner --suite=pie-queue-disc"

