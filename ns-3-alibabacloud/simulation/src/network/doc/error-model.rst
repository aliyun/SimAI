Error Model
-----------
.. include:: replace.txt

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

This section documents a few error model objects, typically associated with
NetDevice models, that are maintained as part of the ``network`` module:

* RateErrorModel
* ListErrorModel
* ReceiveListErrorModel
* BurstErrorModel

Error models are used to indicate that a packet should be considered to
be errored, according to the underlying (possibly stochastic or 
empirical) error model.  

Model Description
*****************

The source code for error models live in the directory ``src/packet/utils``.

Two types of error models are generally provided.  The first are stochastic
models.  In this case, packets are errored according to underlying 
random variable distributions.  An example of this is the ``RateErrorModel``.
The other type of model is a deterministic or empirical model, in which
packets are errored according to a particular prescribed pattern.
An example is the ``ListErrorModel`` that allows users to specify
the list of packets to be errored, by listing the specific packet UIDs.

The ``ns3::RateErrorModel`` errors packets according to an underlying
random variable distribution, which is by default a UniformRandomVariable
distributed between 0.0 and 1.0.  The error rate and error units (bit,
byte, or packet) are set by the user.  For instance, by setting ErrorRate
to 0.1 and ErrorUnit to "Packet", in the long run, around 10% of the
packets will be lost.


Design
======

Error models are |ns3| objects and can be created using the typical
pattern of ``CreateObject<>()``.  They have configuration attributes.

An ErrorModel can be applied anywhere, but are commonly deployed on
NetDevice models so that artificial losses (mimicking channel losses) 
can be induced.

Scope and Limitations
=====================

No known limitations.  There are no existing models that try to modify
the packet contents (e.g. apply bit or byte errors to the byte buffers).
This type of operation will likely be performance-expensive, and existing
Packet APIs may not easily support it.

The |ns3| spectrum model and devices that derive from it (e.g. LTE) have
their own error model base class, found in 

References
==========

The initial |ns3| error models were ported from ns-2 (queue/errmodel.{cc,h})

Usage
*****

The base class API is as follows:

* ``bool ErrorModel::IsCorrupt (Ptr<Packet> pkt)``:  Evaluate the packet and
  return true or false whether the packet should be considered errored or not.
  Some models could potentially alter the contents of the packet bit buffer.
* ``void ErrorModel::Reset (void)``:  Reset any state.
* ``void ErrorModel::Enable (void)``:  Enable the model
* ``void ErrorModel::Disble (void)``:  Disable the model; IsCorrupt() will 
  always return false.
* ``bool ErrorModel::IsEnabled (void) const``:  Return the enabled state


Many |ns3| NetDevices contain attributes holding pointers to error
models.  The error model is applied in the notional physical layer 
processing chain of the device, and drops should show up on the ``PhyRxDrop``
trace source of the device.  The following are known to include an attribute 
with a pointer available to hold this type of error model:

* ``SimpleNetDevice``
* ``PointToPointNetDevice``
* ``CsmaNetDevice``
* ``VirtualNetDevice``

However, the ErrorModel could be used anywhere where packets are used

Helpers
=======

This model is typically not used with helpers.

Attributes
==========

The ``RateErrorModel`` contains the following attributes:


Output
======

What kind of data does the model generate?  What are the key trace
sources?   What kind of logging output can be enabled?

Examples
========

Error models are used in the tutorial ``fifth`` and ``sixth`` programs.

The directory ``examples/error-model/`` contains an example 
``simple-error-model.cc`` that exercises the Rate and List error models.

The TCP example ``examples/tcp/tcp-nsc-lfn.cc`` uses the Rate error model.

Troubleshooting
===============

No known issues.

Validation
**********

The ``error-model`` unit test suite provides a single test case of 
of a particular combination of ErrorRate and ErrorUnit for the 
``RateErrorModel`` applied to a ``SimpleNetDevice``. 

Acknowledgements
****************

The basic ErrorModel, RateErrorModel, and ListErrorModel classes were ported
from |ns2| to |ns3| in 2007.  The ReceiveListErrorModel was added at that
time.

The burst error model is due to Truc Anh N. Nguyen at the University of
Kansas (James P.G. Sterbenz <jpgs@ittc.ku.edu>, director, ResiliNets 
Research Group (http://wiki.ittc.ku.edu/resilinets), Information and 
Telecommunication Technology Center (ITTC) and Department of Electrical 
Engineering and Computer Science, The University of Kansas Lawrence, KS USA).
Work supported in part by NSF FIND (Future Internet Design) Program
under grant CNS-0626918 (Postmodern Internet Architecture),
NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and 
Experimentation on GENI), US Department of Defense (DoD), and ITTC at 
The University of Kansas.
