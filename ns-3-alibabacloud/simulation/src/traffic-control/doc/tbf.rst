.. include:: replace.txt
.. highlight:: cpp

TBF queue disc
----------------

This chapter describes the TBF ([Ref1]_) queue disc implementation
in |ns3|. The TBF model in ns-3 is ported based on Linux kernel code implemented by
A. Kuznetsov and D. Torokhov.

TBF is a qdisc that allows controlling the bandwidth of the output according
to a set rate with the possibility of managing burst conditions also. The TBF implementation
consists of a bucket (buffer) having a limited capacity into which tokens (normally representing a
unit of bytes or a single packet of predetermined size) are added at a fixed rate 'r' called the
token rate. Whenever a packet arrives into the tx queue (fifo by default), the bucket is checked
to see if there are appropriate number of tokens that is equivalent to the length of the packet in
bytes. If yes, then the tokens are removed and the packet is passed for transmission. If no, then
packets will have to wait until there are sufficient tokens in the bucket. This data conformance
can be thus put into three possible scenarios [Ref3]_:

1. Data rate = Token rate : Packets pass without delay.
2. Data rate < Token rate : The tokens might accumulate and the bucket might become
   full. Then, the next packets to enter TBF will be transmitted right away without
   having any limit applied to them, until the bucket is empty. This is called a burst
   condition and in TBF the burst parameter defines the size of the bucket. In order
   to overcome this problem and provide better control over the bursts, TBF
   implements a second bucket which is smaller and generally the same size as the
   MTU. This second bucket cannot store large amount of tokens, but its
   replenishing rate will be a lot faster than the one of the big bucket. This second
   rate is called 'peakRate' and it will determine the maximum rate of a burst.
3. Data rate > Token rate : This causes the TBF algorithm to throttle itself for a while as
   soon as the bucket gets empty. This is called an 'overlimit situation' [Ref2]_. In this situation,
   some of the packets will be blocked until enough tokens are available at which time a schedule for
   the waking of the queue will be done. If packets keep coming in, at a larger rate, then the
   packets will start to get dropped when the total number of bytes exceeds the QueueLimit.

Model Description
*****************

The TBF queue disc does not require packet filters, does not admit internal queues
and uses a single child queue disc. If the user does not provide a child queue disc,
a Fifo queue disc operating in the same mode (packet or byte) as the TBF queue disc
and having a size equal to the TBF QueueLimit attribute is created. Otherwise, the
capacity of the TBF queue disc is determined by the capacity of the child queue disc.

There are two token buckets: first bucket and second bucket. The size of the
first bucket called 'Burst' should always be greater than the size of the second
bucket called the Mtu (which is usually the size of a single packet). But the
'PeakRate' which is the second bucket's token rate should be always greater than
the 'Rate' which is the first bucket's token rate.

If the PeakRate is zero, then the second bucket does not exist. In order to activate
the second bucket, both the Mtu and PeakRate values have to be greater than zero. If
the Mtu value is zero at initialization time, then if a NetDevice exits, the Mtu's
value will be equal to the Mtu of the NetDevice. But if no NetDevice exists, then
the QueueDisc will complain thus prompting the user to set the Mtu value.

The source code for the TBF model is located in the directory ``src/traffic-control/model``
and consists of 2 files `tbf-queue-disc.h` and `tbf-queue-disc.cc` defining a TbfQueueDisc
class.

* class :cpp:class:`TbfQueueDisc`: This class implements the main TBF algorithm:

  * ``TbfQueueDisc::DoEnqueue ()``: This routine enqueue's the incoming packet if the queue is not full and drops the packet otherwise.

  * ``TbfQueueDisc::DoPeek ()``: This routine peeks for the top item in the queue and if the queue is not empty, it returns the topmost item.

  * ``TbfQueueDisc::DoDequeue ()``: This routine performs the dequeuing of packets according to the following logic:

    * It calls ``TbfQueueDisc::Peek ()`` and calculates the size of the packet to be dequeued in bytes.
    * Then it calculates the time difference 'delta', which is the time elapsed since the last update of tokens in the buckets.
    * If the second bucket exists, the number of tokens are updated according to the 'PeakRate' and 'delta'.
    * From this second bucket a number of tokens equal to the size of the packet to be dequeued is subtracted.
    * Now the number of tokens in the first bucket are updated according to 'Rate' and 'delta'.
    * From this first bucket a number of tokens equal to the size of the packet to be dequeued is subtracted.
    * If after this, both the first and second buckets have tokens greater than zero, then the packet is dequeued.
    * Else, an event to ``QueueDisc::Run ()`` is scheduled after a time period when enough tokens will be present for the dequeue operation.

References
==========

.. [Ref1] A. Kuznetsov and D. Torokhov; Linux Cross Reference Source Code; Available online at `<http://lxr.free-electrons.com/source/net/sched/sch_tbf.c>`_.

.. [Ref2] J. Vehent; Journey to the Center of the Linux Kernel: Traffic Control, Shaping and QoS; Available online at `<http://wiki.linuxwall.info/doku.php/en:resources:dossiers:networking:traffic_control#tbf_-_token_bucket_filter>`_.

.. [Ref3] Practical IP Network QoS: TBF queuing discipline; Available online at `<http://web.opalsoft.net/qos/default.php?p=ds-24>`_.

Attributes
==========

The key attributes that the TbfQueueDisc class holds include the following:

* ``MaxSize:`` The maximum number of packets/bytes the queue disc can hold. The default value is 1000 packets.
* ``Burst:`` Size of the first bucket, in bytes. The default value is 125000 bytes.
* ``Mtu:`` Size of second bucket defaults to the MTU of the attached NetDevice, if any, or 0 otherwise.
* ``Rate:`` Rate at which tokens enter the first bucket. The default value is 125KB/s.
* ``PeakRate:`` Rate at which tokens enter the second bucket. The default value is 0KB/s, which means that there is no second bucket.

TraceSources
============

The TbfQueueDisc class provides the following trace sources:

* ``TokensInFirstBucket:`` Number of First Bucket Tokens in bytes
* ``TokensInSecondBucket:`` Number of Second Bucket Tokens in bytes

Examples
========

The example for TBF is `tbf-example.cc` located in ``examples/traffic-control/``.  The command to run the file (the invocation below shows the available command-line options) is:

::

.. sourcecode:: bash

   $ ./ns3 run "tbf-example --PrintHelp"
   $ ./ns3 run "tbf-example --burst=125000 --rate=1Mbps --peakRate=1.5Mbps"

The expected output from the previous commands are traced value changes in the number of tokens in the first and second buckets.

Validation
**********

The TBF model is tested using :cpp:class:`TbfQueueDiscTestSuite` class defined in `src/traffic-control/test/tbf-queue-disc-test-suite.cc`. The suite includes 4 test cases:

* Test 1: Simple Enqueue/Dequeue with verification of attribute setting and subtraction of tokens from the buckets.
* Test 2: When DataRate == FirstBucketTokenRate; packets should pass smoothly.
* Test 3: When DataRate >>> FirstBucketTokenRate; some packets should get blocked and waking of queue should get scheduled.
* Test 4: When DataRate < FirstBucketTokenRate; burst condition, peakRate is set so that bursts are controlled.

The test suite can be run using the following commands:

::

.. sourcecode:: bash

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s tbf-queue-disc

or

::

.. sourcecode:: bash

  $ NS_LOG="TbfQueueDisc" ./ns3 run "test-runner --suite=tbf-queue-disc"
