.. include:: replace.txt
.. highlight:: cpp

.. _Object-model:

Object model
------------

|ns3| is fundamentally a C++ object system. Objects can be declared and
instantiated as usual, per C++ rules. |ns3| also adds some features to
traditional C++ objects, as described below, to provide greater functionality
and features. This manual chapter is intended to introduce the reader to the
|ns3| object model.

This section describes the C++ class design for |ns3| objects. In brief,
several design patterns in use include classic object-oriented design
(polymorphic interfaces and implementations), separation of interface and
implementation, the non-virtual public interface design pattern, an object
aggregation facility, and reference counting for memory management. Those
familiar with component models such as COM or Bonobo will recognize elements of
the design in the |ns3| object aggregation model, although the |ns3| design is
not strictly in accordance with either.

Object-oriented behavior
************************

C++ objects, in general, provide common object-oriented capabilities 
(abstraction, encapsulation, inheritance, and polymorphism) that are part 
of classic object-oriented design. |ns3| objects make use of these 
properties; for instance::

    class Address
    {
    public:
      Address ();
      Address (uint8_t type, const uint8_t *buffer, uint8_t len);
      Address (const Address & address);
      Address &operator = (const Address &address);
      ...
    private:
      uint8_t m_type;
      uint8_t m_len;
      ...
    };

Object base classes
*******************

There are three special base classes used in |ns3|. Classes that inherit
from these base classes can instantiate objects with special properties.
These base classes are:

* class :cpp:class:`Object`
* class :cpp:class:`ObjectBase`
* class :cpp:class:`SimpleRefCount`

It is not required that |ns3| objects inherit from these class, but 
those that do get special properties. Classes deriving from 
class :cpp:class:`Object` get the following properties.

* the |ns3| type and attribute system (see :ref:`Attributes`)
* an object aggregation system
* a smart-pointer reference counting system (class Ptr)

Classes that derive from class :cpp:class:`ObjectBase` get the first two
properties above, but do not get smart pointers. Classes that derive from class
:cpp:class:`SimpleRefCount`: get only the smart-pointer reference counting
system.

In practice, class :cpp:class:`Object` is the variant of the three above that
the |ns3| developer will most commonly encounter.

.. _Memory-management-and-class-Ptr:

Memory management and class Ptr
*******************************

Memory management in a C++ program is a complex process, and is often done
incorrectly or inconsistently. We have settled on a reference counting design
described as follows.

All objects using reference counting maintain an internal reference count to
determine when an object can safely delete itself. Each time that a pointer is
obtained to an interface, the object's reference count is incremented by calling
``Ref()``. It is the obligation of the user of the pointer to explicitly
``Unref()`` the pointer when done. When the reference count falls to zero, the
object is deleted.

* When the client code obtains a pointer from the object itself through object
  creation, or via GetObject, it does not have to increment the reference count.   
* When client code obtains a pointer from another source (e.g., copying a
  pointer) it must call ``Ref()`` to increment the reference count.
* All users of the object pointer must call ``Unref()`` to release the
  reference.

The burden for calling :cpp:func:`Unref()` is somewhat relieved by the use of
the reference counting smart pointer class described below. 

Users using a low-level API who wish to explicitly allocate
non-reference-counted objects on the heap, using operator new, are responsible
for deleting such objects.

Reference counting smart pointer (Ptr)
++++++++++++++++++++++++++++++++++++++

Calling ``Ref()`` and ``Unref()`` all the time would be cumbersome, so |ns3|
provides a smart pointer class :cpp:class:`Ptr` similar to
:cpp:class:`Boost::intrusive_ptr`. This smart-pointer class assumes that the
underlying type provides a pair of ``Ref`` and ``Unref`` methods that are
expected to increment and decrement the internal refcount of the object
instance.  

This implementation allows you to manipulate the smart pointer as if it was a
normal pointer: you can compare it with zero, compare it against other pointers,
assign zero to it, etc.

It is possible to extract the raw pointer from this smart pointer with the
:cpp:func:`GetPointer` and :cpp:func:`PeekPointer` methods.

If you want to store a newed object into a smart pointer, we recommend you to
use the CreateObject template functions to create the object and store it in a
smart pointer to avoid memory leaks. These functions are really small
convenience functions and their goal is just to save you a small bit of typing.

CreateObject and Create
***********************

Objects in C++ may be statically, dynamically, or automatically created.  This
holds true for |ns3| also, but some objects in the system have some additional
frameworks available. Specifically, reference counted objects are usually
allocated using a templated Create or CreateObject method, as follows.

For objects deriving from class :cpp:class:`Object`::

    Ptr<WifiNetDevice> device = CreateObject<WifiNetDevice> ();

Please do not create such objects using ``operator new``; create them using
:cpp:func:`CreateObject()` instead.

For objects deriving from class :cpp:class:`SimpleRefCount`, or other objects
that support usage of the smart pointer class, a templated helper function is
available and recommended to be used::

    Ptr<B> b = Create<B> ();

This is simply a wrapper around operator new that correctly handles the
reference counting system.

In summary, use ``Create<B>`` if B is not an object but just uses reference
counting (e.g. :cpp:class:`Packet`), and use ``CreateObject<B>`` if B derives
from :cpp:class:`ns3::Object`.

Aggregation
***********

The |ns3| object aggregation system is motivated in strong part by a recognition
that a common use case for |ns2| has been the use of inheritance and
polymorphism to extend protocol models. For instance, specialized versions of
TCP such as RenoTcpAgent derive from (and override functions from) class
TcpAgent.  

However, two problems that have arisen in the |ns2| model are downcasts and
"weak base class." Downcasting refers to the procedure of using a base class
pointer to an object and querying it at run time to find out type information,
used to explicitly cast the pointer to a subclass pointer so that the subclass
API can be used. Weak base class refers to the problems that arise when a class
cannot be effectively reused (derived from) because it lacks necessary
functionality, leading the developer to have to modify the base class and
causing proliferation of base class API calls, some of which may not be
semantically correct for all subclasses.

|ns3| is using a version of the query interface design pattern to avoid these
problems. This design is based on elements of the `Component Object Model
<http://en.wikipedia.org/wiki/Component_Object_Model>`_ and `GNOME Bonobo
<http://en.wikipedia.org/wiki/Bonobo_(component_model)>`_ although full
binary-level compatibility of replaceable components is not supported and we
have tried to simplify the syntax and impact on model developers.  

Examples
********

Aggregation example
+++++++++++++++++++

:cpp:class:`Node` is a good example of the use of aggregation in |ns3|.  Note
that there are not derived classes of Nodes in |ns3| such as class
:cpp:class:`InternetNode`.  Instead, components (protocols) are aggregated to a
node. Let's look at how some Ipv4 protocols are added to a node.::

    static void
    AddIpv4Stack(Ptr<Node> node)
    {
      Ptr<Ipv4L3Protocol> ipv4 = CreateObject<Ipv4L3Protocol> ();
      ipv4->SetNode (node);
      node->AggregateObject (ipv4);
      Ptr<Ipv4Impl> ipv4Impl = CreateObject<Ipv4Impl> ();
      ipv4Impl->SetIpv4 (ipv4);
      node->AggregateObject (ipv4Impl);
    }

Note that the Ipv4 protocols are created using :cpp:func:`CreateObject()`.
Then, they are aggregated to the node. In this manner, the Node base class does
not need to be edited to allow users with a base class Node pointer to access
the Ipv4 interface; users may ask the node for a pointer to its Ipv4 interface
at runtime. How the user asks the node is described in the next subsection.

Note that it is a programming error to aggregate more than one object of the
same type to an :cpp:class:`ns3::Object`. So, for instance, aggregation is not
an option for storing all of the active sockets of a node.

GetObject example
+++++++++++++++++

GetObject is a type-safe way to achieve a safe downcasting and to allow
interfaces to be found on an object.  

Consider a node pointer ``m_node`` that points to a Node object that has an
implementation of IPv4 previously aggregated to it. The client code wishes to
configure a default route. To do so, it must access an object within the node
that has an interface to the IP forwarding configuration. It performs the
following::

    Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4> ();

If the node in fact does not have an Ipv4 object aggregated to it, then the
method will return null. Therefore, it is good practice to check the return
value from such a function call. If successful, the user can now use the Ptr to
the Ipv4 object that was previously aggregated to the node.

Another example of how one might use aggregation is to add optional models to
objects. For instance, an existing Node object may have an "Energy Model" object
aggregated to it at run time (without modifying and recompiling the node class).
An existing model (such as a wireless net device) can then later "GetObject" for
the energy model and act appropriately if the interface has been either built in
to the underlying Node object or aggregated to it at run time.  However, other
nodes need not know anything about energy models.

We hope that this mode of programming will require much less need for developers
to modify the base classes.

Object factories
****************

A common use case is to create lots of similarly configured objects. One can
repeatedly call :cpp:func:`CreateObject` but there is also a factory design
pattern in use in the |ns3| system. It is heavily used in the "helper" API. 

Class :cpp:class:`ObjectFactory` can be used to instantiate objects and to
configure the attributes on those objects::

    void SetTypeId (TypeId tid);
    void Set (std::string name, const AttributeValue &value);
    Ptr<T> Create (void) const;

The first method allows one to use the |ns3| TypeId system to specify the type
of objects created. The second allows one to set attributes on the objects to be
created, and the third allows one to create the objects themselves. 

For example: ::

    ObjectFactory factory;
    // Make this factory create objects of type FriisPropagationLossModel
    factory.SetTypeId ("ns3::FriisPropagationLossModel")
    // Make this factory object change a default value of an attribute, for
    // subsequently created objects
    factory.Set ("SystemLoss", DoubleValue (2.0));
    // Create one such object
    Ptr<Object> object = factory.Create (); 
    factory.Set ("SystemLoss", DoubleValue (3.0));
    // Create another object with a different SystemLoss
    Ptr<Object> object = factory.Create (); 

Downcasting
***********

A question that has arisen several times is, "If I have a base class pointer
(Ptr) to an object and I want the derived class pointer, should I downcast (via
C++ dynamic cast) to get the derived pointer, or should I use the object
aggregation system to :cpp:func:`GetObject\<> ()` to find a Ptr to the interface
to the subclass API?"

The answer to this is that in many situations, both techniques will work.
|ns3| provides a templated function for making the syntax of Object
dynamic casting much more user friendly::

    template <typename T1, typename T2>
    Ptr<T1>
    DynamicCast (Ptr<T2> const&p)
    {
      return Ptr<T1> (dynamic_cast<T1 *> (PeekPointer (p)));
    }

DynamicCast works when the programmer has a base type pointer and is testing
against a subclass pointer. GetObject works when looking for different objects
aggregated, but also works with subclasses, in the same way as DynamicCast. If
unsure, the programmer should use GetObject, as it works in all cases. If the
programmer knows the class hierarchy of the object under consideration, it is
more direct to just use DynamicCast.
