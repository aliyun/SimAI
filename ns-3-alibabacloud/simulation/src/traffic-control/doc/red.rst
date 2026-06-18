.. include:: replace.txt
.. highlight:: cpp

RED queue disc
---------------------

Model Description
*****************

Random Early Detection (RED) is a queue discipline that aims to provide
early signals to transport protocol congestion control (e.g. TCP) that
congestion is imminent, so that they back off their rate gracefully
rather than with a bunch of tail-drop losses (possibly incurring
TCP timeout).  The model in ns-3 is a port of Sally Floyd's ns-2
RED model.

Note that, starting from ns-3.25, RED is no longer a queue variant and
cannot be installed as a NetDevice queue. Instead, RED is a queue disc
and must be installed in the context of the traffic control (see the
examples mentioned below).

The RED queue disc does not require packet filters, does not admit
child queue discs and uses a single internal queue. If not provided by
the user, a DropTail queue operating in the same mode (packet or byte)
as the queue disc and having a size equal to the RED MaxSize attribute
is created. Otherwise, the capacity of the queue disc is determined by
the capacity of the internal queue provided by the user.

Adaptive Random Early Detection (ARED)
======================================
ARED is a variant of RED with two main features: (i) automatically sets Queue
weight, MinTh and MaxTh and (ii) adapts maximum drop probability. The model
in ns-3 contains implementation of both the features, and is a port of Sally
Floyd's ns-2 ARED model. Note that the user is allowed to choose and explicitly
configure the simulation by selecting feature (i) or feature (ii), or both.

Feng's Adaptive RED
===================
Feng's Adaptive RED is a variant of RED that adapts the maximum drop
probability. The model in ns-3 contains implementation of this feature, and is a
port of ns-2 Feng's Adaptive RED model.

Nonlinear Random Early Detection (NLRED)
========================================
NLRED is a variant of RED in which the linear packet dropping function of
RED is replaced by a nonlinear quadratic function. This approach makes packet
dropping gentler for light traffic load and aggressive for heavy traffic load.

Explicit Congestion Notification (ECN)
======================================
This RED model supports an ECN mode of operation to notify endpoints of
congestion that may be developing in a bottleneck queue, without resorting
to packet drops. Such a mode is enabled by setting the UseEcn attribute to
true (it is false by default) and only affects incoming packets with the
ECT bit set in their header. When the average queue length is between the
minimum and maximum thresholds, an incoming packet is marked instead of being
dropped. When the average queue length is above the maximum threshold, an
incoming packet is marked (instead of being dropped) only if the UseHardDrop
attribute is set to false (it is true by default).

The implementation of support for ECN marking is done in such a way as
to not impose an internet module dependency on the traffic control module.
The RED model does not directly set ECN bits on the header, but delegates
that job to the QueueDiscItem class.  As a result, it is possible to
use RED queues for other non-IP QueueDiscItems that may or may not support
the ``Mark ()`` method.

References
==========

The RED queue disc aims to be close to the results cited in:
S.Floyd, K.Fall http://icir.org/floyd/papers/redsims.ps

ARED queue implementation is based on the algorithm provided in:
S. Floyd et al, http://www.icir.org/floyd/papers/adaptiveRed.pdf

Feng's Adaptive RED queue implementation is based on the algorithm
provided in:
W. C. Feng et al, http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=752150

NLRED queue implementation is based on the algorithm provided in:
Kaiyu Zhou et al, http://www.sciencedirect.com/science/article/pii/S1389128606000879

The addition of explicit congestion notification (ECN) to IP:
K. K. Ramakrishnan et al, https://tools.ietf.org/html/rfc3168

Attributes
==========

The RED queue contains a number of attributes that control the RED
policies:

* MaxSize
* MeanPktSize
* IdlePktSize
* Wait (time)
* Gentle mode
* MinTh, MaxTh
* Queue weight
* LInterm
* LinkBandwidth
* LinkDelay
* UseEcn
* UseHardDrop

In addition to RED attributes, ARED queue requires following attributes:

* ARED (Boolean attribute. Default: false)
* AdaptMaxP (Boolean attribute to adapt m_curMaxP. Default: false)
* Target Delay (time)
* Interval (time)
* LastSet (time)
* Top (upper limit of m_curMaxP)
* Bottom (lower limit of m_curMaxP)
* Alpha (increment parameter for m_curMaxP)
* Beta (decrement parameter for m_curMaxP)
* RTT

In addition to RED attributes, Feng's Adaptive RED queue requires following
attributes:

* FengAdaptive  (Boolean attribute, Default: false)
* Status        (status of current queue length, Default: Above)
* FengAlpha     (increment parameter for m_curMaxP, Default: 3)
* FengBeta      (decrement parameter for m_curMaxP, Default: 2)

The following attribute should be turned on to simulate NLRED queue disc:

* NLRED (Boolean attribute. Default: false)

Consult the ns-3 documentation for explanation of these attributes.

Simulating ARED
===============

To switch on ARED algorithm, the attribute ARED must be set to true,
as done in ``src/traffic-control/examples/adaptive-red-tests.cc``:

.. sourcecode:: cpp

  Config::SetDefault ("ns3::RedQueueDisc::ARED", BooleanValue (true));

Setting ARED to true implicitly configures both: (i) automatic setting
of Queue weight, MinTh and MaxTh and (ii) adapting m_curMaxP.

NOTE: To explicitly configure (i) or (ii), set ARED attribute to false
and follow the procedure described next:

To configure (i); Queue weight, MinTh and MaxTh, all must be set to 0,
as done in ``src/traffic-control/examples/adaptive-red-tests.cc``:

.. sourcecode:: cpp

  Config::SetDefault ("ns3::RedQueueDisc::QW", DoubleValue (0.0));
  Config::SetDefault ("ns3::RedQueueDisc::MinTh", DoubleValue (0));
  Config::SetDefault ("ns3::RedQueueDisc::MaxTh", DoubleValue (0));

To configure (ii); AdaptMaxP must be set to true, as done in
``src/traffic-control/examples/adaptive-red-tests.cc``:

.. sourcecode:: cpp

  Config::SetDefault ("ns3::RedQueueDisc::AdaptMaxP", BooleanValue (true));

Simulating Feng's Adaptive RED
==============================

To switch on Feng's Adaptive RED algorithm, the attribute FengAdaptive must be
set to true, as done in ``examples/traffic-control/red-vs-fengadaptive.cc``:

.. sourcecode:: cpp

  Config::SetDefault ("ns3::RedQueueDisc::FengAdaptive", BooleanValue (true));

Simulating NLRED
================

To switch on NLRED algorithm, the attribute NLRED must be set to true,
as shown below:

.. sourcecode:: cpp

  Config::SetDefault ("ns3::RedQueueDisc::NLRED", BooleanValue (true));

Examples
========

The RED queue example is found at ``src/traffic-control/examples/red-tests.cc``.

ARED queue examples can be found at:
``src/traffic-control/examples/adaptive-red-tests.cc`` and 
``src/traffic-control/examples/red-vs-ared.cc``

Feng's Adaptive RED example can be found at:
``examples/traffic-control/red-vs-fengadaptive.cc``

NLRED queue example can be found at:
``examples/traffic-control/red-vs-nlred.cc``

Validation
**********

The RED model has been validated and the report is currently stored
at: https://github.com/downloads/talau/ns-3-tcp-red/report-red-ns3.pdf 
