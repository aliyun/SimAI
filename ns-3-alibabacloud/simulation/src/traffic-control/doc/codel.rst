.. include:: replace.txt
.. highlight:: cpp

CoDel queue disc
----------------

This chapter describes the CoDel ([Nic12]_, [Nic14]_) queue disc implementation 
in |ns3|. 

Developed by Kathleen Nichols and Van Jacobson as a solution to the 
bufferbloat [Buf14]_ problem, CoDel (Controlled Delay Management) is a queuing 
discipline that uses a packet's sojourn time (time in queue) to make 
decisions on packet drops.

Note that, starting from ns-3.25, CoDel is no longer a queue variant and
cannot be installed as a NetDevice queue. Instead, CoDel is a queue disc
and must be installed in the context of the traffic control (see the
examples mentioned below).


Model Description
*****************

The source code for the CoDel model is located in the directory ``src/traffic-control/model``
and consists of 2 files `codel-queue-disc.h` and `codel-queue-disc.cc` defining a CoDelQueueDisc
class and a helper CoDelTimestampTag class. The code was ported to |ns3| by
Andrew McGregor based on Linux kernel code implemented by Dave Täht and Eric Dumazet. 

* class :cpp:class:`CoDelQueueDisc`: This class implements the main CoDel algorithm:

  * ``CoDelQueueDisc::DoEnqueue ()``: This routine tags a packet with the current time before pushing it into the queue.  The timestamp tag is used by ``CoDelQueue::DoDequeue()`` to compute the packet's sojourn time.  If the queue is full upon the packet arrival, this routine will drop the packet and record the number of drops due to queue overflow, which is stored in `m_dropOverLimit`.

  * ``CoDelQueueDisc::ShouldDrop ()``: This routine is ``CoDelQueueDisc::DoDequeue()``'s helper routine that determines whether a packet should be dropped or not based on its sojourn time.  If the sojourn time goes above `m_target` and remains above continuously for at least `m_interval`, the routine returns ``true`` indicating that it is OK to drop the packet. Otherwise, it returns ``false``. 

  * ``CoDelQueueDisc::DoDequeue ()``: This routine performs the actual packet drop based on ``CoDelQueueDisc::ShouldDrop ()``'s return value and schedules the next drop/mark.
* class :cpp:class:`CoDelTimestampTag`: This class implements the timestamp tagging for a packet.  This tag is used to compute the packet's sojourn time (the difference between the time the packet is dequeued and the time it is pushed into the queue).

There are 2 branches to ``CoDelQueueDisc::DoDequeue ()``: 

1. If the queue is currently in the dropping state, which means the sojourn time has remained above `m_target` for more than `m_interval`, the routine determines if it's OK to leave the dropping state or it's time for the next drop/mark. When ``CoDelQueueDisc::ShouldDrop ()`` returns ``false``, the queue can move out of the dropping state (set `m_dropping` to ``false``).  Otherwise, the queue continuously drops/marks packets and updates the time for next drop (`m_dropNext`) until one of the following conditions is met:

    1. The queue is empty, upon which the queue leaves the dropping state and exits ``CoDelQueueDisc::ShouldDrop ()`` routine; 
    2. ``CoDelQueueDisc::ShouldDrop ()`` returns ``false`` (meaning the sojourn time goes below `m_target`) upon which the queue leaves the dropping state; 
    3. It is not yet time for next drop/mark (`m_dropNext` is less than current time) upon which the queue waits for the next packet dequeue to check the condition again.

2. If the queue is not in the dropping state, the routine enters the dropping state and drop/mark the first packet if ``CoDelQueueDisc::ShouldDrop ()`` returns ``true`` (meaning the sojourn time has gone above `m_target` for at least `m_interval` for the first time or it has gone above again after the queue leaves the dropping state).

The CoDel queue disc does not require packet filters, does not admit
child queue discs and uses a single internal queue. If not provided by
the user, a DropTail queue operating in the same mode (packet or byte)
as the queue disc and having a size equal to the CoDel MaxSize attribute
is created. Otherwise, the capacity of the queue disc is determined by
the capacity of the internal queue provided by the user.


References
==========

.. [Nic12] K. Nichols and V. Jacobson, Controlling Queue Delay, ACM Queue, Vol. 10 No. 5, May 2012.  Available online at `<http://queue.acm.org/detail.cfm?id=2209336>`_.

.. [Nic14] K. Nichols and V. Jacobson, Internet-Draft:  Controlled Delay Active Queue Management, March 2014.  Available online at `<http://tools.ietf.org/html/draft-nichols-tsvwg-codel-02>`_.

.. [Buf14] Bufferbloat.net.  Available online at `<http://www.bufferbloat.net/>`_.

Attributes
==========

The key attributes that the CoDelQueue class holds include the following: 

* ``MaxSize:`` The maximum number of packets/bytes the queue can hold. The default value is 1500 * DEFAULT_CODEL_LIMIT, which is 1500 * 1000 bytes.
* ``MinBytes:`` The CoDel algorithm minbytes parameter. The default value is 1500 bytes. 
* ``Interval:`` The sliding-minimum window. The default value is 100 ms. 
* ``Target:`` The CoDel algorithm target queue delay. The default value is 5 ms. 
* ``UseEcn:`` True to use ECN (packets are marked instead of being dropped). The default value is false.
* ``CeThreshold:`` The CoDel CE threshold for marking packets. Disabled by default.

Examples
========

The first example is `codel-vs-pfifo-basic-test.cc` located in ``src/traffic-control/examples``.  To run the file (the first invocation below shows the available
command-line options):

.. sourcecode:: bash

   $ ./ns3 run "codel-vs-pfifo-basic-test --PrintHelp"
   $ ./ns3 run "codel-vs-pfifo-basic-test --queueType=CoDel --pcapFileName=codel.pcap --cwndTrFileName=cwndCodel.tr"

The expected output from the previous commands are two files: `codel.pcap` file and `cwndCoDel.tr` (ASCII trace) file The .pcap file can be analyzed using 
wireshark or tcptrace:

.. sourcecode:: bash

   $ tcptrace -l -r -n -W codel.pcap

The second example is `codel-vs-pfifo-asymmetric.cc` located in ``src/traffic-control/examples``.  This example is intended to model a typical cable modem
deployment scenario.  To run the file:

.. sourcecode:: bash

   $ ./ns3 run "codel-vs-pfifo-asymmetric --PrintHelp"
   $ ./ns3 run codel-vs-pfifo-asymmetric

The expected output from the previous commands is six pcap files:

* codel-vs-pfifo-asymmetric-CoDel-server-lan.pcap
* codel-vs-pfifo-asymmetric-CoDel-router-wan.pcap
* codel-vs-pfifo-asymmetric-CoDel-router-lan.pcap
* codel-vs-pfifo-asymmetric-CoDel-cmts-wan.pcap
* codel-vs-pfifo-asymmetric-CoDel-cmts-lan.pcap
* codel-vs-pfifo-asymmetric-CoDel-host-lan.pcap

One attribute file:

* codel-vs-pfifo-asymmetric-CoDel.attr 

Five ASCII trace files:

* codel-vs-pfifo-asymmetric-CoDel-drop.tr
* codel-vs-pfifo-asymmetric-CoDel-drop-state.tr
* codel-vs-pfifo-asymmetric-CoDel-sojourn.tr
* codel-vs-pfifo-asymmetric-CoDel-length.tr
* codel-vs-pfifo-asymmetric-CoDel-cwnd.tr 

Validation
**********

The CoDel model is tested using :cpp:class:`CoDelQueueDiscTestSuite` class defined in `src/traffic-control/test/codel-queue-test-suite.cc`.  The suite includes 5 test cases:

* Test 1: The first test checks the enqueue/dequeue with no drops and makes sure that CoDel attributes can be set correctly.
* Test 2: The second test checks the enqueue with drops due to queue overflow.
* Test 3: The third test checks the NewtonStep() arithmetic against explicit port of Linux implementation
* Test 4: The fourth test checks the ControlLaw() against explicit port of Linux implementation
* Test 5: The fifth test checks the enqueue/dequeue with drops according to CoDel algorithm
* Test 6: The sixth test checks the enqueue/dequeue with marks according to CoDel algorithm

The test suite can be run using the following commands: 

.. sourcecode:: bash

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s codel-queue-disc

or  

.. sourcecode:: bash

  $ NS_LOG="CoDelQueueDisc" ./ns3 run "test-runner --suite=codel-queue-disc"

