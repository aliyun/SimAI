.. include:: replace.txt
.. highlight:: cpp
.. highlight:: bash

.. _sec-fq-cobalt:

FqCobalt queue disc
-------------------

This chapter describes the FqCobalt ([Pal19]_) queue disc implementation in |ns3|.

The FlowQueue-Cobalt (FQ-Cobalt) algorithm is similar to FlowQueue-CoDel (FQ-CoDel) algorithm available in ::ref:`sec-fq-codel`.
The documentation for Cobalt is available in
``ns-3-dev/src/traffic-control/doc/cobalt.rst``.

FqCobalt is one of the key components of the CAKE smart queue management framework  ([Hoe18]_).  The COBALT AQM is preferred to the CoDel AQM for CAKE because
it adds a heuristic called BLUE to cover cases in which the CoDel control
law is too sluggish to respond to queue growth.

Model Description
*****************

The source code for the FqCobalt queue disc is located in the directory
``src/traffic-control/model`` and consists of 2 files `fq-cobalt-queue-disc.h`
and `fq-cobalt-queue-disc.cc` defining a FqCobaltQueueDisc class and a helper
FqCobaltFlow class. The code was ported to |ns3| based on Linux kernel code
implemented by Jonathan Morton
(https://github.com/torvalds/linux/blob/master/net/sched/sch_cake.c).

The Model Description is similar to the FqCoDel documentation mentioned above.

References
==========

.. [Pal19]  J. Palmei, S. Gupta, P. Imputato, J. Morton, M. Tahiliani, S. Avallone, and D. Taht, Design and Evaluation of COBALT Queue Discipline, 2019 IEEE International Symposium on Local and Metropolitan Area Networks (LANMAN), Paris, France, 2019.

.. [Hoe18] T. Hoiland-Jørgensen, D. Taht and J. Morton, "Piece of CAKE: A Comprehensive Queue Management Solution for Home Gateways," 2018 IEEE International Symposium on Local and Metropolitan Area Networks (LANMAN), Washington, DC, USA, 2018.

Attributes
==========

Most of the key attributes are similar to the FqCoDel implementation mentioned above.  One difference is the absence of the ``MinBytes`` parameter.

Some additional parameters implemented as attributes are:

* ``Pdrop:`` Value of drop probability.
* ``Increment:`` Increment value of drop probability. Default value is 1./256 .
* ``Decrement:`` Decrement value of drop probability. Default value is 1./4096 .
* ``BlueThreshold:`` The threshold after which Blue is enabled. Default value is 400ms.

Note that if the user wants to disable Blue Enhancement then the user can set
it to a large value; for example, to `Time::Max ()`.

Examples
========

A typical usage pattern is to create a traffic control helper and to configure
the type
and attributes of the queue disc and filters from the helper.
For example, `FqCobalt` can be configured as follows:

.. sourcecode:: cpp

  TrafficControlHelper tch;
  tch.SetRootQueueDisc ("ns3::FqCobaltQueueDisc", "DropBatchSize", UintegerValue (1)
                                                 "Perturbation", UintegerValue (256));
  QueueDiscContainer qdiscs = tch.Install (devices);

Validation
**********

The FqCobalt model is tested using :cpp:class:`FqCobaltQueueDiscTestSuite` class defined in `src/test/ns3tc/codel-queue-test-suite.cc`.

The tests are similar to the ones for FqCoDel queue disc mentioned in first section of this document.
The test suite can be run using the following commands::

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3 build
  $ ./test.py -s fq-cobalt-queue-disc

or::

  $ NS_LOG="FqCobaltQueueDisc" ./ns3 run "test-runner --suite=fq-cobalt-queue-disc"
