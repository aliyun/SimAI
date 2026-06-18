.. include:: replace.txt
.. highlight:: cpp

Creating a new |ns3| model
--------------------------

This chapter walks through the design process of an |ns3| model.  In many
research cases, users will not be satisfied to merely adapt existing models, but
may want to extend the core of the simulator in a novel way. We will use the
example of adding an ErrorModel to a simple |ns3| link as a motivating example
of how one might approach this problem and proceed through a design and
implementation.

.. note:: Documentation

   Here we focus on the process of creating new models
   and new modules, and some of the design choices involved.
   For the sake of clarity, we defer discussion of the
   *mechanics* of documenting models and source code to the
   :doc:`Documentation <documentation>` chapter.


Design Approach
***************

Consider how you want it to work; what should it do. Think about these things:

* *functionality:*  What functionality should it have?  What attributes or
  configuration is exposed to the user?
* *reusability:*  How much should others be able to reuse my design?  Can I
  reuse code from |ns2| to get started?  How does a user integrate the model
  with the rest of another simulation?
* *dependencies:*  How can I reduce the introduction of outside dependencies on
  my new code as much as possible (to make it more modular)?  For instance,
  should I avoid any dependence on IPv4 if I want it to also be used by IPv6?
  Should I avoid any dependency on IP at all?

Do not be hesitant to contact the `ns-3-users` or `ns-developers` list if you have
questions. In particular, it is important to think about the public API of your
new model and ask for feedback. It also helps to let others know of your work in
case you are interested in collaborators.

Example: `ErrorModel`
+++++++++++++++++++++

An error model exists in |ns2|. It allows packets to be passed to a stateful
object that determines, based on a random variable, whether the packet is
corrupted.  The caller can then decide what to do with the packet (drop it,
etc.).

The main API of the error model is a function to pass a packet to, and the
return value of this function is a boolean that tells the caller whether any
corruption occurred.  Note that depending on the error model, the packet data
buffer may or may not be corrupted.  Let's call this function "IsCorrupt()".

So far, in our design, we have::

    class ErrorModel
    {
    public:
     /**
      * \returns true if the Packet is to be considered as errored/corrupted
      * \param pkt Packet to apply error model to
      */
      bool IsCorrupt (Ptr<Packet> pkt);
    };

Note that we do not pass a const pointer, thereby allowing the function to
modify the packet if IsCorrupt() returns true. Not all error models will
actually modify the packet; whether or not the packet data buffer is corrupted
should be documented.

We may also want specialized versions of this, such as in |ns2|, so although it
is not the only design choice for polymorphism, we assume that we will subclass
a base class ErrorModel for specialized classes, such as RateErrorModel,
ListErrorModel, etc, such as is done in |ns2|.

You may be thinking at this point, "Why not make IsCorrupt() a virtual method?".
That is one approach; the other is to make the public non-virtual function
indirect through a private virtual function (this in C++ is known as the non
virtual interface idiom and is adopted in the |ns3| ErrorModel class).  

Next, should this device have any dependencies on IP or other protocols?  We do
not want to create dependencies on Internet protocols (the error model should be
applicable to non-Internet protocols too), so we'll keep that in mind later.

Another consideration is how objects will include this error model.  We envision
putting an explicit setter in certain NetDevice implementations, for example.::

  /**
   * Attach a receive ErrorModel to the PointToPointNetDevice.
   *
   * The PointToPointNetDevice may optionally include an ErrorModel in
   * the packet receive chain.
   *
   * @see ErrorModel
   * @param em Ptr to the ErrorModel.
   */
  void PointToPointNetDevice::SetReceiveErrorModel(Ptr<ErrorModel> em);

Again, this is not the only choice we have (error models could be aggregated to
lots of other objects), but it satisfies our primary use case, which is to allow
a user to force errors on otherwise successful packet transmissions, at the
NetDevice level.

After some thinking and looking at existing |ns2| code, here is a sample API of
a base class and first subclass that could be posted for initial review::

    class ErrorModel 
    {
    public:
      ErrorModel ();
      virtual ~ErrorModel ();
      bool IsCorrupt (Ptr<Packet> pkt);
      void Reset (void);
      void Enable (void);
      void Disable (void);
      bool IsEnabled (void) const;
    private:
      virtual bool DoCorrupt (Ptr<Packet> pkt) = 0;
      virtual void DoReset (void) = 0;
    };

    enum ErrorUnit
      {
        EU_BIT,
        EU_BYTE,
        EU_PKT
      };

    // Determine which packets are errored corresponding to an underlying
    // random variable distribution, an error rate, and unit for the rate.
    class RateErrorModel : public ErrorModel
    {
    public:
      RateErrorModel ();
      virtual ~RateErrorModel ();
      enum ErrorUnit GetUnit (void) const;
      void SetUnit (enum ErrorUnit error_unit);
      double GetRate (void) const;
      void SetRate (double rate);
      void SetRandomVariable (const RandomVariable &ranvar);
    private:
      virtual bool DoCorrupt (Ptr<Packet> pkt);
      virtual void DoReset (void);
    };


Scaffolding
***********

Let's say that you are ready to start implementing; you have a fairly clear
picture of what you want to build, and you may have solicited some initial
review or suggestions from the list.  One way to approach the next step
(implementation) is to create scaffolding and fill in the details as the design
matures.  

This section walks through many of the steps you should consider to define
scaffolding, or a non-functional skeleton of what your model will eventually
implement. It is usually good practice to not wait to get these details
integrated at the end, but instead to plumb a skeleton of your model into the
system early and then add functions later once the API and integration seems
about right.  

Note that you will want to modify a few things in the below presentation for
your model since if you follow the error model verbatim, the code you produce
will collide with the existing error model. The below is just an outline of how
ErrorModel was built that you can adapt to other models.

Review the |ns3| Coding Style Document
++++++++++++++++++++++++++++++++++++++

At this point, you may want to pause and read the |ns3| coding style document,
especially if you are considering to contribute your code back to the project.
The coding style document is linked off the main project page: `ns-3 coding
style <http://www.nsnam.org/developers/contributing-code/coding-style/>`_.

Decide Where in the Source Tree the Model Should Reside
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

All of the |ns3| model source code is in the directory ``src/``.  You will need
to choose which subdirectory it resides in. If it is new model code of some
sort, it makes sense to put it into the ``src/`` directory somewhere,
particularly for ease of integrating with the build system.

In the case of the error model, it is very related to the packet class, so it
makes sense to implement this in the ``src/network/`` module where |ns3|
packets are implemented.  

`cmake` and `CMakeLists.txt`
++++++++++++++++++++++++++++

|ns3| uses the `CMake <https://cmake.org/>`_ build system.
You will want to integrate your new |ns3| uses the CMake build system. You will
want to integrate your new source files into this system. This requires that you
add your files to the ``CMakeLists.txt`` file found in each directory.

Let's start with empty files error-model.h and error-model.cc, and add this to
``src/network/CMakeLists.txt``. It is really just a matter of adding the .cc file to the
rest of the source files, and the .h file to the list of the header files. 

Now, pop up to the top level directory and type "./test.py".  You
shouldn't have broken anything by this operation.

Include Guards
++++++++++++++

Next, let's add some `include guards
<http://en.wikipedia.org/wiki/Include_guard>`_ in our header file.::

    #ifndef ERROR_MODEL_H
    #define ERROR_MODEL_H
    ...
    #endif

`namespace ns3`
+++++++++++++++

|ns3| uses the |ns3| `namespace
<http://en.wikipedia.org/wiki/Namespace_(computer_science)#Use_in_common_languages>`_
to isolate its symbols from other namespaces. Typically, a user will next put
an |ns3| namespace block in both the cc and h file.::

    namespace ns3 {
    ...
    }

At this point, we have some skeletal files in which we can start defining
our new classes. The header file looks like this::

    #ifndef ERROR_MODEL_H
    #define ERROR_MODEL_H

    namespace ns3 {

    } // namespace ns3
    #endif

while the ``error-model.cc`` file simply looks like this::

    #include "error-model.h"

    namespace ns3 {

    } // namespace ns3

These files should compile since they don't really have any contents.  We're now
ready to start adding classes.

Initial Implementation
**********************

At this point, we're still working on some scaffolding, but we can begin to
define our classes, with the functionality to be added later.

Inherit from the `Object` Class?
++++++++++++++++++++++++++++++++

This is an important design step; whether to use class :cpp:class:`Object` as a
base class for your new classes.

As described in the chapter on the |ns3| :ref:`Object-model`, classes that
inherit from class :cpp:class:`Object` get special properties:

* the |ns3| type and attribute system (see :ref:`Attributes`)
* an object aggregation system
* a smart-pointer reference counting system (class Ptr)

Classes that derive from class :cpp:class:`ObjectBase`} get the first two
properties above, but do not get smart pointers. Classes that derive from class
:cpp:class:`RefCountBase` get only the smart-pointer reference counting system.

In practice, class :cpp:class:`Object` is the variant of the three above that
the |ns3| developer will most commonly encounter.

In our case, we want to make use of the attribute system, and we will be passing
instances of this object across the |ns3| public API, so class
:cpp:class:`Object` is appropriate for us.

Initial Classes
+++++++++++++++

One way to proceed is to start by defining the bare minimum functions and see if
they will compile. Let's review what all is needed to implement when we derive
from class Object.::

    #ifndef ERROR_MODEL_H
    #define ERROR_MODEL_H

    #include "ns3/object.h"

    namespace ns3 {
      
    class ErrorModel : public Object
    {
    public:
      static TypeId GetTypeId (void);

      ErrorModel ();
      virtual ~ErrorModel ();
    };

    class RateErrorModel : public ErrorModel
    {
    public:
      static TypeId GetTypeId (void);
     
      RateErrorModel ();
      virtual ~RateErrorModel ();
    };
    #endif


A few things to note here. We need to include ``object.h``. The convention in
|ns3| is that if the header file is co-located in the same directory, it may be
included without any path prefix. Therefore, if we were implementing ErrorModel
in ``src/core/model`` directory, we could have just said "``#include "object.h"``".
But we are in ``src/network/model``, so we must include it as "``#include
"ns3/object.h"``". Note also that this goes outside the namespace declaration.
 
Second, each class must implement a static public member function called
``GetTypeId (void)``.  

Third, it is a good idea to implement constructors and destructors rather than
to let the compiler generate them, and to make the destructor virtual. In C++,
note also that copy assignment operator and copy constructors are auto-generated
if they are not defined, so if you do not want those, you should implement those
as private members. This aspect of C++ is discussed in Scott Meyers' Effective
C++ book. item 45. 

Let's now look at some corresponding skeletal implementation code in the .cc 
file.::

    #include "error-model.h"

    namespace ns3 {

    NS_OBJECT_ENSURE_REGISTERED (ErrorModel);

    TypeId ErrorModel::GetTypeId (void)
    {
      static TypeId tid = TypeId ("ns3::ErrorModel")
        .SetParent<Object> ()
        .SetGroupName ("Network")
        ;
      return tid;
    }

    ErrorModel::ErrorModel ()
    { 
    }

    ErrorModel::~ErrorModel ()
    { 
    }

    NS_OBJECT_ENSURE_REGISTERED (RateErrorModel);

    TypeId RateErrorModel::GetTypeId (void)
    {
      static TypeId tid = TypeId ("ns3::RateErrorModel")
        .SetParent<ErrorModel> ()
        .SetGroupName ("Network")
        .AddConstructor<RateErrorModel> ()
        ;
      return tid;
    }

    RateErrorModel::RateErrorModel ()
    {
    } 

    RateErrorModel::~RateErrorModel ()
    {
    }

What is the ``GetTypeId (void)`` function? This function does a few things.  It
registers a unique string into the TypeId system. It establishes  the hierarchy
of objects in the attribute system (via ``SetParent``). It also declares that
certain objects can be created via the object creation framework
(``AddConstructor``).  

The macro ``NS_OBJECT_ENSURE_REGISTERED (classname)`` is needed also once for
every class that defines a new GetTypeId method, and it does the actual
registration of the class into the system.  The :ref:`Object-model` chapter
discusses this in more detail.

Including External Files
++++++++++++++++++++++++

Logging Support
+++++++++++++++

*Here, write a bit about adding |ns3| logging macros. Note that
LOG_COMPONENT_DEFINE is done outside the namespace ns3*

Constructor, Empty Function Prototypes
++++++++++++++++++++++++++++++++++++++

Key Variables (Default Values, Attributes)
++++++++++++++++++++++++++++++++++++++++++

Test Program 1
++++++++++++++

Object Framework
++++++++++++++++

Adding a Sample Script
**********************

At this point, one may want to try to take the basic scaffolding defined above
and add it into the system. Performing this step now allows one to use a simpler
model when plumbing into the system and may also reveal whether any design or
API modifications need to be made. Once this is done, we will return to building
out the functionality of the ErrorModels themselves.

Add Basic Support in the Class
++++++++++++++++++++++++++++++

::

    /* point-to-point-net-device.h */
      class ErrorModel;
      
      /**
       * Error model for receive packet events
       */
      Ptr<ErrorModel> m_receiveErrorModel;
      
Add Accessor
++++++++++++

::

    void
    PointToPointNetDevice::SetReceiveErrorModel (Ptr<ErrorModel> em)
    {
      NS_LOG_FUNCTION (this << em);
      m_receiveErrorModel = em;
    } 

       .AddAttribute ("ReceiveErrorModel",
                       "The receiver error model used to simulate packet loss",
                       PointerValue (),
                       MakePointerAccessor (&PointToPointNetDevice::m_receiveErrorModel),
                       MakePointerChecker<ErrorModel> ())

Plumb Into the System
+++++++++++++++++++++

::

    void PointToPointNetDevice::Receive (Ptr<Packet> packet)
    {
      NS_LOG_FUNCTION (this << packet);
      uint16_t protocol = 0;
      
      if (m_receiveErrorModel && m_receiveErrorModel->IsCorrupt (packet) )
        {
    //  
    // If we have an error model and it indicates that it is time to lose a
    // corrupted packet, don't forward this packet up, let it go. 
    // 
          m_dropTrace (packet);
        } 
      else
        {
    //  
    // Hit the receive trace hook, strip off the point-to-point protocol header
    // and forward this packet up the protocol stack.
    //     
          m_rxTrace (packet);
          ProcessHeader(packet, protocol);
          m_rxCallback (this, packet, protocol, GetRemote ());
          if (!m_promiscCallback.IsNull ())
            {           m_promiscCallback (this, packet, protocol, GetRemote (), 
                          GetAddress (), NetDevice::PACKET_HOST);
            }
        }
    }

Create Null Functional Script
+++++++++++++++++++++++++++++

::

    /* simple-error-model.cc */

      // Error model
      // We want to add an error model to node 3's NetDevice
      // We can obtain a handle to the NetDevice via the channel and node
      // pointers
      Ptr<PointToPointNetDevice> nd3 = PointToPointTopology::GetNetDevice
        (n3, channel2);
      Ptr<ErrorModel> em = Create<ErrorModel> ();
      nd3->SetReceiveErrorModel (em);


    bool
    ErrorModel::DoCorrupt (Packet& p)
    {
      NS_LOG_FUNCTION;
      NS_LOG_UNCOND("Corrupt!");
      return false;
    }

At this point, we can run the program with our trivial ErrorModel plumbed into
the receive path of the PointToPointNetDevice. It prints out the string
"Corrupt!" for each packet received at node n3. Next, we return to the error
model to add in a subclass that performs more interesting error modeling.

Add a Subclass
**************

The trivial base class ErrorModel does not do anything interesting, but it
provides a useful base class interface (Corrupt () and Reset ()), forwarded to
virtual functions that can be subclassed. Let's next consider what we call a
BasicErrorModel which is based on the |ns2| ErrorModel class (in
``ns-2/queue/errmodel.{cc,h}``).

What properties do we want this to have, from a user interface perspective? We
would like for the user to be able to trivially swap out the type of ErrorModel
used in the NetDevice. We would also like the capability to set configurable
parameters.

Here are a few simple requirements we will consider:

* Ability to set the random variable that governs the losses (default is
  UniformVariable)
* Ability to set the unit (bit, byte, packet, time) of granularity over which
  errors are applied.
* Ability to set the rate of errors (e.g. 10^-3) corresponding to the above unit
  of granularity.
* Ability to enable/disable (default is enabled)

How to Subclass
+++++++++++++++

We declare BasicErrorModel to be a subclass of ErrorModel as follows,::

    class BasicErrorModel : public ErrorModel
    {
    public:
      static TypeId GetTypeId (void);
      ...
    private:
      // Implement base class pure virtual functions
      virtual bool DoCorrupt (Ptr<Packet> p);
      virtual bool DoReset (void);
      ...
    }

and configure the subclass GetTypeId function by setting a unique TypeId string
and setting the Parent to ErrorModel::

    TypeId RateErrorModel::GetTypeId (void)
    {
      static TypeId tid = TypeId ("ns3::RateErrorModel")
        .SetParent<ErrorModel> ()
        .SetGroupName ("Network")
        .AddConstructor<RateErrorModel> ()
      ...

Build Core Functions and Unit Tests
***********************************

Assert Macros
+++++++++++++

Writing Unit Tests
++++++++++++++++++

