.. include:: replace.txt
.. highlight:: cpp

.. Section Separators:
   ----
   ****
   ++++
   ====
   ~~~~

.. _Attributes:

Configuration and Attributes
----------------------------

In |ns3| simulations, there are two main aspects to configuration:

* The simulation topology and how objects are connected.
* The values used by the models instantiated in the topology.

This chapter focuses on the second item above: how the many values in use in
|ns3| are organized, documented, and modifiable by |ns3| users. The |ns3|
attribute system is also the underpinning of how traces and statistics are
gathered in the simulator.

In the course of this chapter we will discuss the various ways to set or
modify the values used by |ns3| model objects.  In increasing order of
specificity, these are:

+-------------------------------------+------------------------------------+
| Method                              | Scope                              |
+=====================================+====================================+
| Default Attribute values set when   | Affect all instances of the class. |
| Attributes are defined in           |                                    |
| :cpp:func:`GetTypeId ()`.           |                                    |
+-------------------------------------+------------------------------------+
| :cpp:class:`CommandLine`            | Affect all future instances.       |
| :cpp:func:`Config::SetDefault()`    |                                    |
| :cpp:class:`ConfigStore`            |                                    |
+-------------------------------------+------------------------------------+
| :cpp:class:`ObjectFactory`          | Affects all instances created with |
|                                     | the factory.                       |
+-------------------------------------+------------------------------------+
| Helper methods with (string/        | Affects all instances created by   |
| AttributeValue) parameter pairs     | the helper.                        |
+-------------------------------------+------------------------------------+
| :cpp:func:`MyClass::SetX ()`        | Alters this particular instance.   |
| :cpp:func:`Object::SetAttribute ()` | Generally this is the only form    |
| :cpp:func:`Config::Set()`           | which can be scheduled to alter    |
|                                     | an instance once the simulation    |
|                                     | is running.                        |
+-------------------------------------+------------------------------------+

By "specificity" we mean that methods in later rows in the table
override the values set by, and typically affect fewer instances than,
earlier methods.

Before delving into details of the attribute value system, it will help to
review some basic properties of class :cpp:class:`Object`.

Object Overview
***************

|ns3| is fundamentally a C++ object-based system. By this we mean that new C++
classes (types) can be declared, defined, and subclassed as usual.

Many |ns3| objects inherit from the :cpp:class:`Object` base class.  These
objects have some additional properties that we exploit for organizing the
system and improving the memory management of our objects:

* "Metadata" system that links the class name to a lot of meta-information
  about the object, including:

  * The base class of the subclass,
  * The set of accessible constructors in the subclass,
  * The set of "attributes" of the subclass,
  * Whether each attribute can be set, or is read-only,
  * The allowed range of values for each attribute.

* Reference counting smart pointer implementation, for memory management.

|ns3| objects that use the attribute system derive from either
:cpp:class:`Object` or :cpp:class:`ObjectBase`. Most |ns3| objects we
will discuss derive from :cpp:class:`Object`, but a few that are outside
the smart pointer memory management framework derive from
:cpp:class:`ObjectBase`.

Let's review a couple of properties of these objects.

Smart Pointers
++++++++++++++

As introduced in the |ns3| tutorial, |ns3| objects are memory managed by a
`reference counting smart pointer implementation
<http://en.wikipedia.org/wiki/Smart_pointer>`_, class :cpp:class:`Ptr`.

Smart pointers are used extensively in the |ns3| APIs, to avoid passing
references to heap-allocated objects that may cause memory leaks.
For most basic usage (syntax), treat a smart pointer like a regular pointer::

  Ptr<WifiNetDevice> nd = ...;
  nd->CallSomeFunction ();
  // etc.

So how do you get a smart pointer to an object, as in the first line
of this example?

CreateObject
============

As we discussed above in :ref:`Memory-management-and-class-Ptr`, at the
lowest-level API, objects of type :cpp:class:`Object` are not instantiated
using ``operator new`` as usual but instead by a templated function called
:cpp:func:`CreateObject ()`.

A typical way to create such an object is as follows::

  Ptr<WifiNetDevice> nd = CreateObject<WifiNetDevice> ();

You can think of this as being functionally equivalent to::

  WifiNetDevice* nd = new WifiNetDevice ();

Objects that derive from :cpp:class:`Object` must be allocated on the heap
using :cpp:func:`CreateObject ()`. Those deriving from :cpp:class:`ObjectBase`,
such as |ns3| helper functions and packet headers and trailers,
can be allocated on the stack.

In some scripts, you may not see a lot of :cpp:func:`CreateObject ()` calls
in the code; this is because there are some helper objects in effect
that are doing the :cpp:func:`CreateObject ()` calls for you.

TypeId
++++++

|ns3| classes that derive from class :cpp:class:`Object` can include
a metadata class called :cpp:class:`TypeId` that records meta-information
about the class, for use in the object aggregation and component manager
systems:

* A unique string identifying the class.
* The base class of the subclass, within the metadata system.
* The set of accessible constructors in the subclass.
* A list of publicly accessible properties ("attributes") of the class.

Object Summary
++++++++++++++

Putting all of these concepts together, let's look at a specific
example: class :cpp:class:`Node`.

The public header file ``node.h`` has a declaration that includes
a static :cpp:func:`GetTypeId ()` function call::

    class Node : public Object
    {
    public:
      static TypeId GetTypeId (void);
      ...

This is defined in the ``node.cc`` file as follows::

    TypeId
    Node::GetTypeId (void)
    {
      static TypeId tid = TypeId ("ns3::Node")
        .SetParent<Object> ()
        .SetGroupName ("Network")
        .AddConstructor<Node> ()
        .AddAttribute ("DeviceList",
	               "The list of devices associated to this Node.",
                       ObjectVectorValue (),
                       MakeObjectVectorAccessor (&Node::m_devices),
                       MakeObjectVectorChecker<NetDevice> ())
        .AddAttribute ("ApplicationList",
	               "The list of applications associated to this Node.",
                       ObjectVectorValue (),
                       MakeObjectVectorAccessor (&Node::m_applications),
                       MakeObjectVectorChecker<Application> ())
        .AddAttribute ("Id",
	               "The id (unique integer) of this Node.",
                       TypeId::ATTR_GET, // allow only getting it.
                       UintegerValue (0),
                       MakeUintegerAccessor (&Node::m_id),
                       MakeUintegerChecker<uint32_t> ())
        ;
      return tid;
    }

Consider the :cpp:class:`TypeId` of the |ns3| :cpp:class:`Object` class
as an extended form of run time type information (RTTI). The C++ language
includes a simple kind of RTTI in order to support ``dynamic_cast`` and
``typeid`` operators.

The :cpp:func:`SetParent<Object> ()` call in the definition above is used in
conjunction with our object aggregation mechanisms to allow safe up- and
down-casting in inheritance trees during :cpp:func:`GetObject ()`.
It also enables subclasses to inherit the Attributes of their parent class.

The :cpp:func:`AddConstructor<Node> ()` call is used in conjunction
with our abstract object factory mechanisms to allow us to construct
C++ objects without forcing a user to know the concrete class of
the object she is building.

The three calls to :cpp:func:`AddAttribute ()` associate a given string
with a strongly typed value in the class. Notice that you must provide
a help string which may be displayed, for example, *via* command line
processors. Each :cpp:class:`Attribute` is associated with mechanisms
for accessing the underlying member variable in the object (for example,
:cpp:func:`MakeUintegerAccessor ()` tells the generic :cpp:class:`Attribute`
code how to get to the node ID above). There are also "Checker" methods which
are used to validate values against range limitations, such as maximum
and minimum allowed values.

When users want to create Nodes, they will usually call some form of
:cpp:func:`CreateObject ()`,::

    Ptr<Node> n = CreateObject<Node> ();

or more abstractly, using an object factory, you can create a
:cpp:class:`Node` object without even knowing the concrete C++ type::

    ObjectFactory factory;
    const std::string typeId = "ns3::Node'';
    factory.SetTypeId (typeId);
    Ptr<Object> node = factory.Create <Object> ();

Both of these methods result in fully initialized attributes being available
in the resulting :cpp:class:`Object` instances.

We next discuss how attributes (values associated with member variables or
functions of the class) are plumbed into the above :cpp:class:`TypeId`.

Attributes
**********

The goal of the attribute system is to organize the access of
internal member objects of a simulation. This goal arises because,
typically in simulation, users will cut and paste/modify existing
simulation scripts, or will use higher-level simulation constructs,
but often will be interested in studying or tracing particular
internal variables.  For instance, use cases such as:

* *"I want to trace the packets on the wireless interface only on the first
  access point."*
* *"I want to trace the value of the TCP congestion window (every time it
  changes) on a particular TCP socket."*
* *"I want a dump of all values that were used in my simulation."*

Similarly, users may want fine-grained access to internal variables in the
simulation, or may want to broadly change the initial value used for a
particular parameter in all subsequently created objects. Finally, users may
wish to know what variables are settable and retrievable in a simulation
configuration. This is not just for direct simulation interaction on the command
line; consider also a (future) graphical user interface that would like to be
able to provide a feature whereby a user might right-click on an node on the
canvas and see a hierarchical, organized list of parameters that are settable on
the node and its constituent member objects, and help text and default values
for each parameter.

Available AttributeValue Types
++++++++++++++++++++++++++++++

* AddressValue

* AttributeContainerValue

* BooleanValue

* BoxValue

* CallbackValue

* DataRateValue

* DoubleValue

* EmptyAttributeValue

* EnumValue

* IntegerValue

* Ipv4AddressValue

* Ipv4MaskValue

* Ipv6AddressValue

* Ipv6PrefixValue

* LengthValue

* Mac16AddressValue

* Mac48AddressValue

* Mac64AddressValue

* ObjectFactoryValue

* ObjectPtrContainerValue

* PairValue<A, B>

* PointerValue

* PriomapValue

* QueueSizeValue

* RectangleValue

* SsidValue

* TimeValue

* TupleValue<Args...>

  A TupleValue is capable of storing values of different types, hence it is suitable for
  structured data. A prominent example is the ChannelSettings attribute of WifiPhy, which
  consists of channel number, channel width, PHY band and primary 20 MHz channel index.
  In this case the values have to be mutually consistent, which makes it difficult to set them
  as individual Attributes. Capturing them in a TupleValue simplifies this problem, see
  ``src/wifi/model/wifi-phy.cc``.

  Values stored in a TupleValue object can be set/get through a std::tuple object or
  can be serialized to/deserialized from a string containing a comma-separated sequence of
  the values enclosed in a pair of curly braces (e.g., "{36, 20, BAND_5GHZ, 0}").

  The usage of the TupleValue attribute is illustrated in
  ``src/core/test/tuple-value-test-suite.cc``.

* TypeIdValue

* UanModesListValue

* UintegerValue

* Vector2DValue

* Vector3DValue

* WaypointValue

* WifiModeValue


Defining Attributes
+++++++++++++++++++

We provide a way for users to access values deep in the system, without having
to plumb accessors (pointers) through the system and walk pointer chains to get
to them. Consider a class :cpp:class:`QueueBase` that has a member variable
:cpp:member:`m_maxSize` controlling the depth of the queue.

If we look at the declaration of :cpp:class:`QueueBase`, we see
the following::

    class QueueBase : public Object {
    public:
      static TypeId GetTypeId (void);
      ...

    private:
      ...
      QueueSize m_maxSize;                //!< max queue size
      ...
    };

:cpp:class:`QueueSize` is a special type in |ns3| that allows size
to be represented in different units::

    enum QueueSizeUnit
    {
      PACKETS,     /**< Use number of packets for queue size */
      BYTES,       /**< Use number of bytes for queue size */
    };

    class QueueSize
    {
      ...
    private:
      ...
      QueueSizeUnit m_unit; //!< unit
      uint32_t m_value;     //!< queue size [bytes or packets]
    };

Finally, the class :cpp:class:`DropTailQueue` inherits from this base
class and provides the semantics that packets that are submitted to
a full queue will be dropped from the back of the queue ("drop tail").

::

    /**
     * \ingroup queue
     *
     * \brief A FIFO packet queue that drops tail-end packets on overflow
     */
    template <typename Item>
    class DropTailQueue : public Queue<Item>

Let's consider things that a user may want to do with the value of
:cpp:member:`m_maxSize`:

* Set a default value for the system, such that whenever a new
  :cpp:class:`DropTailQueue` is created, this member is initialized
  to that default.
* Set or get the value on an already instantiated queue.

The above things typically require providing ``Set ()`` and ``Get ()``
functions, and some type of global default value.

In the |ns3| attribute system, these value definitions and accessor function
registrations are moved into the :cpp:class:`TypeId` class; *e.g*.::

    NS_OBJECT_ENSURE_REGISTERED (QueueBase);

    TypeId
    QueueBase::GetTypeId (void)
    {
      static TypeId tid = TypeId ("ns3::DropTailQueue")
        .SetParent<Queue> ()
        .SetGroupName ("Network")
        ...
        .AddAttribute ("MaxSize",
                       "The max queue size",
                       QueueSizeValue (QueueSize ("100p")),
                       MakeQueueSizeAccessor (&QueueBase::SetMaxSize,
                                              &QueueBase::GetMaxSize),
                       MakeQueueSizeChecker ())
        ...
        ;

      return tid;
    }

The :cpp:func:`AddAttribute ()` method is performing a number of things for the
:cpp:member:`m_maxSize` value:

* Binding the (usually private) member variable :cpp:member:`m_maxSize`
  to a public string ``"MaxSize"``.
* Providing a default value (0 packets).
* Providing some help text defining the meaning of the value.
* Providing a "Checker" (not used in this example) that can be used to set
  bounds on the allowable range of values.

The key point is that now the value of this variable and its default value are
accessible in the attribute namespace, which is based on strings such as
``"MaxSize"`` and :cpp:class:`TypeId` name strings. In the next section,
we will provide an example script that shows how users may manipulate
these values.

Note that initialization of the attribute relies on the macro
``NS_OBJECT_ENSURE_REGISTERED (QueueBase)`` being called; if you leave this
out of your new class implementation, your attributes will not be initialized
correctly.

While we have described how to create attributes, we still haven't described how
to access and manage these values. For instance, there is no ``globals.h``
header file where these are stored; attributes are stored with their classes.
Questions that naturally arise are how do users easily learn about all of the
attributes of their models, and how does a user access these attributes, or
document their values as part of the record of their simulation?

Detailed documentation of the actual attributes defined for a type,
and a global list of all defined attributes, are available in
the API documentation.  For the rest of this document we are going
to demonstrate the various ways of getting and setting attribute values.

Setting Default Values
++++++++++++++++++++++

Config::SetDefault and CommandLine
==================================

Let's look at how a user script might access a specific attribute value.
We're going to use the
``src/point-to-point/examples/main-attribute-value.cc``
script for illustration, with some details stripped out.  The ``main``
function begins::

    // This is a basic example of how to use the attribute system to
    // set and get a value in the underlying system; namely, the maximum
    // size of the FIFO queue in the PointToPointNetDevice
    //

    int
    main (int argc, char *argv[])
    {

      // Queues in ns-3 are objects that hold items (other objects) in
      // a queue structure.  The C++ implementation uses templates to
      // allow queues to hold various types of items, but the most
      // common is a pointer to a packet (Ptr<Packet>).
      //
      // The maximum queue size can either be enforced in bytes ('b') or
      // packets ('p').  A special type called the ns3::QueueSize can
      // hold queue size values in either unit (bytes or packets).  The
      // queue base class ns3::QueueBase has a MaxSize attribute that can
      // be set to a QueueSize.

      // By default, the MaxSize attribute has a value of 100 packets ('100p')
      // (this default can be observed in the function QueueBase::GetTypeId)
      //
      // Here, we set it to 80 packets.  We could use one of two value types:
      // a string-based value or a QueueSizeValue value
      Config::SetDefault ("ns3::QueueBase::MaxSize", StringValue ("80p"));
      // The below function call is redundant
      Config::SetDefault ("ns3::QueueBase::MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, 80)));

The main thing to notice in the above are the two equivalent calls to
:cpp:func:`Config::SetDefault ()`.  This is how we set the default value
for all subsequently instantiated :cpp:class:`DropTailQueue`\s.  We illustrate
that two types of ``Value`` classes, a :cpp:class:`StringValue` and
a :cpp:class:`QueueSizeValue` class, can be used to assign the value
to the attribute named by "ns3::QueueBase::MaxSize".

It is also possible to manipulate Attributes using the :cpp:class:`CommandLine`;
we saw some examples early in the |ns3| Tutorial.  In particular, it is
straightforward to add a shorthand argument name, such as ``--maxSize``,
for an Attribute that is particular relevant for your model, in this case
``"ns3::QueueBase::MaxSize"``.  This has the additional feature that
the help string for the Attribute will be printed as part of the usage
message for the script.  For more information see
the :cpp:class:`CommandLine` API documentation.

::

      // Allow the user to override any of the defaults and the above
      // SetDefaults() at run-time, via command-line arguments
      // For example, via "--ns3::QueueBase::MaxSize=80p"
      CommandLine cmd;
      // This provides yet another way to set the value from the command line:
      cmd.AddValue ("maxSize", "ns3::QueueBase::MaxSize");
      cmd.Parse (argc, argv);

Now, we will create a few objects using the low-level API.  Our
newly created queues will not have :cpp:member:`m_maxSize` initialized to
0 packets, as defined in the :cpp:func:`QueueBase::GetTypeId ()`
function, but to 80 packets, because of what we did above with
default values.::

    Ptr<Node> n0 = CreateObject<Node> ();

    Ptr<PointToPointNetDevice> net0 = CreateObject<PointToPointNetDevice> ();
    n0->AddDevice (net0);

    Ptr<Queue<Packet> > q = CreateObject<DropTailQueue<Packet> > ();
    net0->AddQueue(q);

At this point, we have created a single :cpp:class:`Node` (``n0``)
and a single :cpp:class:`PointToPointNetDevice` (``net0``),
added a :cpp:class:`DropTailQueue` (``q``) to ``net0``,
which will be configured with a queue size limit of 80 packets.

As a final note, the `Config::Set...()` functions
will throw an error if the targeted Attribute does not exist at the path
given.  There are also "fail-safe" versions,
`Config::Set...FailSafe()`, if you can't be sure the Attribute
exists.  The fail-safe versions return `true` if at least one instance
could be set.

Constructors, Helpers and ObjectFactory
=======================================

Arbitrary combinations of attributes can be set and fetched from
the helper and low-level APIs; either from the constructors themselves::

    Ptr<GridPositionAllocator> p =
      CreateObjectWithAttributes<GridPositionAllocator>
        ("MinX", DoubleValue (-100.0),
         "MinY", DoubleValue (-100.0),
         "DeltaX", DoubleValue (5.0),
         "DeltaY", DoubleValue (20.0),
         "GridWidth", UintegerValue (20),
         "LayoutType", StringValue ("RowFirst"));

or from the higher-level helper APIs, such as::

    mobility.SetPositionAllocator
        ("ns3::GridPositionAllocator",
         "MinX", DoubleValue (-100.0),
         "MinY", DoubleValue (-100.0),
         "DeltaX", DoubleValue (5.0),
         "DeltaY", DoubleValue (20.0),
         "GridWidth", UintegerValue (20),
         "LayoutType", StringValue ("RowFirst"));

We don't illustrate it here, but you can also configure an
:cpp:class:`ObjectFactory` with new values for specific attributes.
Instances created by the :cpp:class:`ObjectFactory` will have those
attributes set during construction.  This is very similar to using
one of the helper APIs for the class.

To review, there are several ways to set values for attributes for
class instances *to be created in the future:*

* :cpp:func:`Config::SetDefault ()`
* :cpp:func:`CommandLine::AddValue ()`
* :cpp:func:`CreateObjectWithAttributes<> ()`
* Various helper APIs

But what if you've already created an instance, and you want
to change the value of the attribute?  In this example, how can we
manipulate the :cpp:member:`m_maxSize` value of the already instantiated
:cpp:class:`DropTailQueue`?  Here are various ways to do that.

Changing Values
+++++++++++++++

SmartPointer
============

Assume that a smart pointer (:cpp:class:`Ptr`) to a relevant network device
is in hand; in the current example, it is the ``net0`` pointer.

One way to change the value is to access a pointer to the underlying queue and
modify its attribute.

First, we observe that we can get a pointer to the (base class)
:cpp:class:`Queue` *via* the
:cpp:class:`PointToPointNetDevice` attributes, where it is called
``"TxQueue"``::

    PointerValue ptr;
    net0->GetAttribute ("TxQueue", ptr);
    Ptr<Queue<Packet> > txQueue = ptr.Get<Queue<Packet> > ();

Using the :cpp:func:`GetObject ()` function, we can perform a safe downcast
to a :cpp:class:`DropTailQueue`.  The `NS_ASSERT` checks that the pointer is
valid.

::

    Ptr<DropTailQueue<Packet> > dtq = txQueue->GetObject <DropTailQueue<Packet> > ();
    NS_ASSERT (dtq != 0);

Next, we can get the value of an attribute on this queue.  We have introduced
wrapper ``Value`` classes for the underlying data types, similar
to Java wrappers around these types, since the attribute system stores values
serialized to strings, and not disparate types.  Here, the attribute value
is assigned to a :cpp:class:`QueueSizeValue`, and the :cpp:func:`Get ()`
method on this value produces the (unwrapped) ``QueueSize``.  That is,
the variable `limit` is written into by the GetAttribute method.::

    QueueSizeValue limit;
    dtq->GetAttribute ("MaxSize", limit);
    NS_LOG_INFO ("1.  dtq limit: " << limit.Get ());

Note that the above downcast is not really needed; we could have gotten
the attribute value directly from ``txQueue``::

    txQueue->GetAttribute ("MaxSize", limit);
    NS_LOG_INFO ("2.  txQueue limit: " << limit.Get ());

Now, let's set it to another value (60 packets).  Let's also make
use of the StringValue shorthand notation to set the size by
passing in a string (the string must be a positive integer suffixed
by either the `p` or `b` character).

::

    txQueue->SetAttribute ("MaxSize", StringValue ("60p"));
    txQueue->GetAttribute ("MaxSize", limit);
    NS_LOG_INFO ("3.  txQueue limit changed: " << limit.Get ());

Config Namespace Path
=====================

An alternative way to get at the attribute is to use the configuration
namespace.  Here, this attribute resides on a known path in this namespace; this
approach is useful if one doesn't have access to the underlying pointers and
would like to configure a specific attribute with a single statement.

::

    Config::Set ("/NodeList/0/DeviceList/0/TxQueue/MaxSize",
                 StringValue ("25p"));
    txQueue->GetAttribute ("MaxSize", limit);
    NS_LOG_INFO ("4.  txQueue limit changed through namespace: "
                 << limit.Get ());

The configuration path often has the form of
``".../<container name>/<index>/.../<attribute>/<attribute>"``
to refer to a specific instance by index of an object in the container.
In this case the first container is the list of all :cpp:class:`Node`\s;
the second container is the list of all :cpp:class:`NetDevice`\s on
the chosen :cpp:class:`Node`.  Finally, the configuration path usually
ends with a succession of member attributes, in this case the ``"MaxSize"``
attribute of the ``"TxQueue"`` of the chosen :cpp:class:`NetDevice`.

We could have also used wildcards to set this value for all nodes and all net
devices (which in this simple example has the same effect as the previous
:cpp:func:`Config::Set ()`)::

    Config::Set ("/NodeList/*/DeviceList/*/TxQueue/MaxSize",
                 StringValue ("15p"));
    txQueue->GetAttribute ("MaxSize", limit);
    NS_LOG_INFO ("5.  txQueue limit changed through wildcarded namespace: "
                 << limit.Get ());

If you run this program from the command line, you should see the following
output corresponding to the steps we took above:

.. sourcecode:: bash

    $ ./ns3 run main-attribute-value
    1.  dtq limit: 80p
    2.  txQueue limit: 80p
    3.  txQueue limit changed: 60p
    4.  txQueue limit changed through namespace: 25p
    5.  txQueue limit changed through wildcarded namespace: 15p

Object Name Service
===================

Another way to get at the attribute is to use the object name service facility.
The object name service allows us to add items to the configuration
namespace under the ``"/Names/"`` path with a user-defined name string.
This approach is useful if one doesn't have access to the underlying
pointers and it is difficult to determine the required concrete configuration
namespace path.

::

    Names::Add ("server", n0);
    Names::Add ("server/eth0", net0);

    ...

    Config::Set ("/Names/server/eth0/TxQueue/MaxPackets", UintegerValue (25));

Here we've added the path elements ``"server"`` and ``"eth0"`` under
the ``"/Names/"`` namespace, then used the resulting configuration path
to set the attribute.

See :ref:`Object-names` for a fuller treatment of the |ns3| configuration namespace.

Implementation Details
**********************

Value Classes
+++++++++++++

Readers will note the ``TypeValue`` classes which are subclasses of the
:cpp:class:`AttributeValue` base class. These can be thought of as
intermediate classes which are used to convert from raw types to the
:cpp:class:`AttributeValue`\s that are used by the attribute system.
Recall that this database is holding objects of many types serialized
to strings. Conversions to this type can either be done using
an intermediate class (such as :cpp:class:`IntegerValue`,
or :cpp:class:`DoubleValue` for floating point numbers)
or *via* strings. Direct implicit conversion of types to
:cpp:class:`AttributeValue` is not really practical.
So in the above, users have a choice of using strings or values::

    p->Set ("cwnd", StringValue ("100")); // string-based setter
    p->Set ("cwnd", IntegerValue (100)); // integer-based setter

The system provides some macros that help users declare and define
new AttributeValue subclasses for new types that they want to introduce into
the attribute system:

* ``ATTRIBUTE_HELPER_HEADER``
* ``ATTRIBUTE_HELPER_CPP``

See the API documentation for these constructs for more information.

Initialization Order
++++++++++++++++++++

Attributes in the system must not depend on the state of any other Attribute in
this system. This is because an ordering of Attribute initialization is not
specified, nor enforced, by the system. A specific example of this can be seen
in automated configuration programs such as :cpp:class:`ConfigStore`.
Although a given model may arrange it so that Attributes are initialized in a
particular order, another automatic configurator may decide independently to
change Attributes in, for example, alphabetic order.

Because of this non-specific ordering, no Attribute in the system may have any
dependence on any other Attribute. As a corollary, Attribute setters must never
fail due to the state of another Attribute. No Attribute setter may change (set)
any other Attribute value as a result of changing its value.

This is a very strong restriction and there are cases where Attributes must set
consistently to allow correct operation. To this end we do allow for consistency
checking *when the attribute is used* (*cf*. ``NS_ASSERT_MSG``
or ``NS_ABORT_MSG``).

In general, the attribute code to assign values to the underlying class member
variables is executed after an object is constructed. But what if you need the
values assigned before the constructor body executes, because you need them in
the logic of the constructor? There is a way to do this, used for example in the
class :cpp:class:`ConfigStore`: call :cpp:func:`ObjectBase::ConstructSelf ()` as
follows::

    ConfigStore::ConfigStore ()
    {
      ObjectBase::ConstructSelf (AttributeConstructionList ());
      // continue on with constructor.
    }

Beware that the object and all its derived classes must also implement
a :cpp:func:`GetInstanceTypeId ()` method. Otherwise
the :cpp:func:`ObjectBase::ConstructSelf ()` will not be able to read
the attributes.

Adding Attributes
+++++++++++++++++

The |ns3| system will place a number of internal values under the attribute
system, but undoubtedly users will want to extend this to pick up ones we have
missed, or to add their own classes to the system.

There are three typical use cases:

* Making an existing class data member accessible as an Attribute,
  when it isn't already.
* Making a new class able to expose some data members as Attributes
  by giving it a TypeId.
* Creating an :cpp:class:`AttributeValue` subclass for a new class
  so that it can be accessed as an Attribute.

Existing Member Variable
========================

Consider this variable in :cpp:class:`TcpSocket`::

    uint32_t m_cWnd;   // Congestion window

Suppose that someone working with TCP wanted to get or set the value of that
variable using the metadata system. If it were not already provided by |ns3|,
the user could declare the following addition in the runtime metadata system (to
the :cpp:func:`GetTypeId` definition for :cpp:class:`TcpSocket`)::

    .AddAttribute ("Congestion window",
                   "Tcp congestion window (bytes)",
                   UintegerValue (1),
                   MakeUintegerAccessor (&TcpSocket::m_cWnd),
                   MakeUintegerChecker<uint16_t> ())

Now, the user with a pointer to a :cpp:class:`TcpSocket` instance
can perform operations such as
setting and getting the value, without having to add these functions explicitly.
Furthermore, access controls can be applied, such as allowing the parameter to
be read and not written, or bounds checking on the permissible values can be
applied.

New Class TypeId
================

Here, we discuss the impact on a user who wants to add a new class to |ns3|.
What additional things must be done to enable it to hold attributes?

Let's assume our new class, called :cpp:class:`ns3::MyMobility`,
is a type of mobility model.  First, the class should inherit from
its parent class, :cpp:class:`ns3::MobilityModel`.
In the ``my-mobility.h`` header file::

    namespace ns3 {

    class MyMobility : public MobilityModel
    {

This requires we declare the :cpp:func:`GetTypeId ()` function.
This is a one-line public function declaration::

    public:
      /**
       *  Register this type.
       *  \return The object TypeId.
       */
      static TypeId GetTypeId (void);

We've already introduced what a :cpp:class:`TypeId` definition will look like
in the ``my-mobility.cc`` implementation file::

    NS_OBJECT_ENSURE_REGISTERED (MyMobility);

    TypeId
    MyMobility::GetTypeId (void)
    {
      static TypeId tid = TypeId ("ns3::MyMobility")
        .SetParent<MobilityModel> ()
        .SetGroupName ("Mobility")
        .AddConstructor<MyMobility> ()
        .AddAttribute ("Bounds",
                       "Bounds of the area to cruise.",
                       RectangleValue (Rectangle (0.0, 0.0, 100.0, 100.0)),
                       MakeRectangleAccessor (&MyMobility::m_bounds),
                       MakeRectangleChecker ())
        .AddAttribute ("Time",
                       "Change current direction and speed after moving for this delay.",
                       TimeValue (Seconds (1.0)),
                       MakeTimeAccessor (&MyMobility::m_modeTime),
                       MakeTimeChecker ())
        // etc (more parameters).
        ;
      return tid;
    }

If we don't want to subclass from an existing class, in the header file
we just inherit from :cpp:class:`ns3::Object`, and in the object file
we set the parent class to :cpp:class:`ns3::Object` with
``.SetParent<Object> ()``.

Typical mistakes here involve:

* Not calling ``NS_OBJECT_ENSURE_REGISTERED ()``
* Not calling the :cpp:func:`SetParent ()` method,
  or calling it with the wrong type.
* Not calling the :cpp:func:`AddConstructor ()` method,
  or calling it with the wrong type.
* Introducing a typographical error in the name of the :cpp:class:`TypeId`
  in its constructor.
* Not using the fully-qualified C++ typename of the enclosing C++ class as the
  name of the :cpp:class:`TypeId`.  Note that ``"ns3::"`` is required.

None of these mistakes can be detected by the |ns3| codebase, so users
are advised to check carefully multiple times that they got these right.

New AttributeValue Type
=======================

From the perspective of the user who writes a new class in the system and wants
it to be accessible as an attribute, there is mainly the matter of writing the
conversions to/from strings and attribute values.  Most of this can be
copy/pasted with macro-ized code.  For instance, consider a class declaration
for :cpp:class:`Rectangle` in the ``src/mobility/model`` directory:

Header File
~~~~~~~~~~~

::

    /**
     * \brief a 2d rectangle
     */
    class Rectangle
    {
      ...

      double xMin;
      double xMax;
      double yMin;
      double yMax;
    };

One macro call and two operators, must be added below the class declaration in
order to turn a Rectangle into a value usable by the ``Attribute`` system::

    std::ostream &operator << (std::ostream &os, const Rectangle &rectangle);
    std::istream &operator >> (std::istream &is, Rectangle &rectangle);

    ATTRIBUTE_HELPER_HEADER (Rectangle);

Implementation File
~~~~~~~~~~~~~~~~~~~

In the class definition (``.cc`` file), the code looks like this::

    ATTRIBUTE_HELPER_CPP (Rectangle);

    std::ostream &
    operator << (std::ostream &os, const Rectangle &rectangle)
    {
      os << rectangle.xMin << "|" << rectangle.xMax << "|" << rectangle.yMin << "|"
         << rectangle.yMax;
      return os;
    }
    std::istream &
    operator >> (std::istream &is, Rectangle &rectangle)
     {
      char c1, c2, c3;
      is >> rectangle.xMin >> c1 >> rectangle.xMax >> c2 >> rectangle.yMin >> c3
         >> rectangle.yMax;
      if (c1 != '|' ||
          c2 != '|' ||
          c3 != '|')
        {
          is.setstate (std::ios_base::failbit);
        }
      return is;
    }

These stream operators simply convert from a string representation of the
Rectangle (``"xMin|xMax|yMin|yMax"``) to the underlying Rectangle.  The modeler
must specify these operators and the string syntactical representation of an
instance of the new class.

ConfigStore
***********

Values for |ns3| attributes can be stored in an ASCII or XML text file
and loaded into a future simulation run.  This feature is known as the
|ns3| ConfigStore.  The :cpp:class:`ConfigStore` is a specialized database for attribute values and default values.

Although it is a separately maintained module in the
``src/config-store/`` directory, we document it here because of its
sole dependency on |ns3| core module and attributes.

We can explore this system by using an example from
``src/config-store/examples/config-store-save.cc``.

First, all users of the :cpp:class:`ConfigStore` must include
the following statement::

    #include "ns3/config-store-module.h"

Next, this program adds a sample object :cpp:class:`ConfigExample`
to show how the system is extended::

    class ConfigExample : public Object
    {
    public:
      static TypeId GetTypeId (void) {
        static TypeId tid = TypeId ("ns3::A")
          .SetParent<Object> ()
          .AddAttribute ("TestInt16", "help text",
                         IntegerValue (-2),
                         MakeIntegerAccessor (&A::m_int16),
                         MakeIntegerChecker<int16_t> ())
          ;
          return tid;
        }
      int16_t m_int16;
    };

    NS_OBJECT_ENSURE_REGISTERED (ConfigExample);

Next, we use the Config subsystem to override the defaults in a couple of
ways::

      Config::SetDefault ("ns3::ConfigExample::TestInt16", IntegerValue (-5));

      Ptr<ConfigExample> a_obj = CreateObject<ConfigExample> ();
      NS_ABORT_MSG_UNLESS (a_obj->m_int16 == -5,
                           "Cannot set ConfigExample's integer attribute via Config::SetDefault");

      Ptr<ConfigExample> a2_obj = CreateObject<ConfigExample> ();
      a2_obj->SetAttribute ("TestInt16", IntegerValue (-3));
      IntegerValue iv;
      a2_obj->GetAttribute ("TestInt16", iv);
      NS_ABORT_MSG_UNLESS (iv.Get () == -3,
                           "Cannot set ConfigExample's integer attribute via SetAttribute");

The next statement is necessary to make sure that (one of) the objects
created is rooted in the configuration namespace as an object instance.
This normally happens when you aggregate objects to a :cpp:class:`ns3::Node`
or :cpp:class:`ns3::Channel` instance,
but here, since we are working at the core level, we need to create a
new root namespace object::

      Config::RegisterRootNamespaceObject (a2_obj);

Writing
+++++++

Next, we want to output the configuration store.  The examples show how
to do it in two formats, XML and raw text.  In practice, one should perform
this step just before calling :cpp:func:`Simulator::Run ()` to save the
final configuration just before running the simulation.

There are three Attributes that govern the behavior of the ConfigStore:
``"Mode"``, ``"Filename"``, and ``"FileFormat"``.  The Mode (default ``"None"``)
configures whether |ns3| should load configuration from a previously saved file
(specify ``"Mode=Load"``) or save it to a file (specify ``"Mode=Save"``).
The Filename (default ``""``) is where the ConfigStore should read or write
its data.  The FileFormat (default ``"RawText"``) governs whether
the ConfigStore format is plain text or Xml (``"FileFormat=Xml"``)

The example shows::

      Config::SetDefault ("ns3::ConfigStore::Filename", StringValue ("output-attributes.xml"));
      Config::SetDefault ("ns3::ConfigStore::FileFormat", StringValue ("Xml"));
      Config::SetDefault ("ns3::ConfigStore::Mode", StringValue ("Save"));
      ConfigStore outputConfig;
      outputConfig.ConfigureDefaults ();
      outputConfig.ConfigureAttributes ();

      // Output config store to txt format
      Config::SetDefault ("ns3::ConfigStore::Filename", StringValue ("output-attributes.txt"));
      Config::SetDefault ("ns3::ConfigStore::FileFormat", StringValue ("RawText"));
      Config::SetDefault ("ns3::ConfigStore::Mode", StringValue ("Save"));
      ConfigStore outputConfig2;
      outputConfig2.ConfigureDefaults ();
      outputConfig2.ConfigureAttributes ();

      Simulator::Run ();

      Simulator::Destroy ();

Note the placement of these statements just prior to the
:cpp:func:`Simulator::Run ()` statement.  This output logs all of the
values in place just prior to starting the simulation (*i.e*. after
all of the configuration has taken place).

After running, you can open the ``output-attributes.txt`` file and see:

.. sourcecode:: text

    ...
    default ns3::ErrorModel::IsEnabled "true"
    default ns3::RateErrorModel::ErrorUnit "ERROR_UNIT_BYTE"
    default ns3::RateErrorModel::ErrorRate "0"
    default ns3::RateErrorModel::RanVar "ns3::UniformRandomVariable[Min=0.0|Max=1.0]"
    default ns3::BurstErrorModel::ErrorRate "0"
    default ns3::BurstErrorModel::BurstStart "ns3::UniformRandomVariable[Min=0.0|Max=1.0]"
    default ns3::BurstErrorModel::BurstSize "ns3::UniformRandomVariable[Min=1|Max=4]"
    default ns3::PacketSocket::RcvBufSize "131072"
    default ns3::PcapFileWrapper::CaptureSize "65535"
    default ns3::PcapFileWrapper::NanosecMode "false"
    default ns3::SimpleNetDevice::PointToPointMode "false"
    default ns3::SimpleNetDevice::TxQueue "ns3::DropTailQueue<Packet>"
    default ns3::SimpleNetDevice::DataRate "0bps"
    default ns3::PacketSocketClient::MaxPackets "100"
    default ns3::PacketSocketClient::Interval "+1000000000.0ns"
    default ns3::PacketSocketClient::PacketSize "1024"
    default ns3::PacketSocketClient::Priority "0"
    default ns3::ConfigStore::Mode "Save"
    default ns3::ConfigStore::Filename "output-attributes.txt"
    default ns3::ConfigStore::FileFormat "RawText"
    default ns3::ConfigExample::TestInt16 "-5"
    global SimulatorImplementationType "ns3::DefaultSimulatorImpl"
    global SchedulerType "ns3::MapScheduler"
    global RngSeed "1"
    global RngRun "1"
    global ChecksumEnabled "false"
    value /$ns3::ConfigExample/TestInt16 "-3"

In the above, several of the default values for attributes for the core
and network modules are shown.  Then, all the values for the |ns3| global values
are recorded.  Finally, the value of the instance of :cpp:class:`ConfigExample`
that was rooted in the configuration namespace is shown.  In a real
|ns3| program, many more models, attributes, and defaults would be shown.

An XML version also exists in ``output-attributes.xml``:

.. sourcecode:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <ns3>
     <default name="ns3::ErrorModel::IsEnabled" value="true"/>
     <default name="ns3::RateErrorModel::ErrorUnit" value="ERROR_UNIT_BYTE"/>
     <default name="ns3::RateErrorModel::ErrorRate" value="0"/>
     <default name="ns3::RateErrorModel::RanVar" value="ns3::UniformRandomVariable[Min=0.0|Max=1.0]"/>
     <default name="ns3::BurstErrorModel::ErrorRate" value="0"/>
     <default name="ns3::BurstErrorModel::BurstStart" value="ns3::UniformRandomVariable[Min=0.0|Max=1.0]"/>
     <default name="ns3::BurstErrorModel::BurstSize" value="ns3::UniformRandomVariable[Min=1|Max=4]"/>
     <default name="ns3::PacketSocket::RcvBufSize" value="131072"/>
     <default name="ns3::PcapFileWrapper::CaptureSize" value="65535"/>
     <default name="ns3::PcapFileWrapper::NanosecMode" value="false"/>
     <default name="ns3::SimpleNetDevice::PointToPointMode" value="false"/>
     <default name="ns3::SimpleNetDevice::TxQueue" value="ns3::DropTailQueue&lt;Packet&gt;"/>
     <default name="ns3::SimpleNetDevice::DataRate" value="0bps"/>
     <default name="ns3::PacketSocketClient::MaxPackets" value="100"/>
     <default name="ns3::PacketSocketClient::Interval" value="+1000000000.0ns"/>
     <default name="ns3::PacketSocketClient::PacketSize" value="1024"/>
     <default name="ns3::PacketSocketClient::Priority" value="0"/>
     <default name="ns3::ConfigStore::Mode" value="Save"/>
     <default name="ns3::ConfigStore::Filename" value="output-attributes.xml"/>
     <default name="ns3::ConfigStore::FileFormat" value="Xml"/>
     <default name="ns3::ConfigExample::TestInt16" value="-5"/>
     <global name="SimulatorImplementationType" value="ns3::DefaultSimulatorImpl"/>
     <global name="SchedulerType" value="ns3::MapScheduler"/>
     <global name="RngSeed" value="1"/>
     <global name="RngRun" value="1"/>
     <global name="ChecksumEnabled" value="false"/>
     <value path="/$ns3::ConfigExample/TestInt16" value="-3"/>
    </ns3>

This file can be archived with your simulation script and output data.

Reading
+++++++

Next, we discuss configuring simulations *via* a stored input
configuration file.  There are a couple of key differences
compared to writing the final simulation configuration.  First, we
need to place statements such as these at the beginning of the program,
before simulation configuration statements are written (so the values
are registered before being used in object construction).

::

      Config::SetDefault ("ns3::ConfigStore::Filename", StringValue ("input-defaults.xml"));
      Config::SetDefault ("ns3::ConfigStore::Mode", StringValue ("Load"));
      Config::SetDefault ("ns3::ConfigStore::FileFormat", StringValue ("Xml"));
      ConfigStore inputConfig;
      inputConfig.ConfigureDefaults ();

Next, note that loading of input configuration data is limited to Attribute
default (*i.e*. not instance) values, and global values.  Attribute instance
values are not supported because at this stage of the simulation, before
any objects are constructed, there are no such object instances around.
(Note, future enhancements to the config store may change this behavior).

Second, while the output of :cpp:class:`ConfigStore` state
will list everything in the database, the input file need only contain
the specific values to be overridden.  So, one way to use this class
for input file configuration is to generate an initial configuration
using the output (``"Save"``) ``"Mode"`` described above, extract from
that configuration file only the elements one wishes to change,
and move these minimal elements to a new configuration file
which can then safely be edited and loaded in a subsequent simulation run.

When the :cpp:class:`ConfigStore` object is instantiated, its attributes
``"Filename"``, ``"Mode"``, and ``"FileFormat"`` must be set,
either *via* command-line or *via* program statements.

Reading/Writing Example
+++++++++++++++++++++++

As a more complicated example, let's assume that we want to read in a
configuration of defaults from an input file named ``input-defaults.xml``, and
write out the resulting attributes to a separate file called
``output-attributes.xml``.::

    #include "ns3/config-store-module.h"
    ...
    int main (...)
    {

      Config::SetDefault ("ns3::ConfigStore::Filename", StringValue ("input-defaults.xml"));
      Config::SetDefault ("ns3::ConfigStore::Mode", StringValue ("Load"));
      Config::SetDefault ("ns3::ConfigStore::FileFormat", StringValue ("Xml"));
      ConfigStore inputConfig;
      inputConfig.ConfigureDefaults ();

      //
      // Allow the user to override any of the defaults and the above Bind () at
      // run-time, viacommand-line arguments
      //
      CommandLine cmd;
      cmd.Parse (argc, argv);

      // setup topology
      ...

      // Invoke just before entering Simulator::Run ()
      Config::SetDefault ("ns3::ConfigStore::Filename", StringValue ("output-attributes.xml"));
      Config::SetDefault ("ns3::ConfigStore::Mode", StringValue ("Save"));
      ConfigStore outputConfig;
      outputConfig.ConfigureAttributes ();
      Simulator::Run ();
    }

ConfigStore use cases (pre- and post-simulation)
++++++++++++++++++++++++++++++++++++++++++++++++

It is worth stressing that ConfigStore can be used for different purposes, and this is
reflected in where in the script ConfigStore is invoked.

The typical use-cases are:

* Change an Object default attributes
* Inspect/change a *specific* Object attributes
* Inspect the simulation Objects and their attributes

As a matter of fact, some Objects might be created when the simulation starts.
Hence, ConfigStore will not "report" their attributes if invoked earlier in the code.

A typical workflow might involve running the simulation, calling ConfigStore 
at the end of the simulation (after ``Simulator::Run ()`` and before ``Simulator::Destroy ()``)
This will show all the attributes in the Objects, both those with default values, and those
with values changed during the simulation execution.

To change these values, you'll need to either change the default (class-wide) attribute values
(in this case call ConfigStore before the Object creation), or  specific object attribute
(in this case call ConfigStore after the Object creation, typically just before ``Simulator::Run ()``.


ConfigStore GUI
+++++++++++++++

There is a GTK-based front end for the ConfigStore.  This allows users to use a
GUI to access and change variables.

Some screenshots are presented here. They are the result of using GtkConfig on 
``src/lte/examples/lena-dual-stripe.cc`` after ``Simulator::Run ()``.

.. _GtkConfig:

.. figure:: figures/gtk-config-lena-dual-stripe-device-view.*

.. figure:: figures/gtk-config-lena-dual-stripe-eNB-tx-power.*


To use this feature, one must install ``libgtk-3-dev``; an example
Ubuntu installation command is:

.. sourcecode:: bash

  $ sudo apt-get install libgtk-3-dev

On a MacOS it is possible to install GTK-3 using `Homebrew <https://brew.sh>`_.
The installation command is:

.. sourcecode:: bash

  $ brew install gtk+3 adwaita-icon-theme


To check whether it is configured or not, check the output of the step:

.. sourcecode:: bash

  $ ./ns3 configure --enable-examples --enable-tests

  ---- Summary of optional NS-3 features:
  Python Bindings               : enabled
  Python API Scanning Support   : enabled
  NS-3 Click Integration        : enabled
  GtkConfigStore                : not enabled (library 'gtk+-3.0 >= 3.0' not found)

In the above example, it was not enabled, so it cannot be used until a suitable
version is installed and:

.. sourcecode:: bash

  $ ./ns3 configure --enable-examples --enable-tests
  $ ./ns3

is rerun.

Usage is almost the same as the non-GTK-based version, but there
are no :cpp:class:`ConfigStore` attributes involved::

  // Invoke just before entering Simulator::Run ()
  GtkConfigStore config;
  config.ConfigureDefaults ();
  config.ConfigureAttributes ();

Now, when you run the script, a GUI should pop up, allowing you to open menus of
attributes on different nodes/objects, and then launch the simulation execution
when you are done.

Note that "launch the simulation" means to proceed with the simulation script.
If GtkConfigStore has been called after ``Simulator::Run ()`` the simulation will
not be started again - it will just end.


