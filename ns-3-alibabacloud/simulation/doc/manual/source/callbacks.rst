.. include:: replace.txt
.. highlight:: cpp

Callbacks
---------

Some new users to |ns3| are unfamiliar with an extensively used programming
idiom used throughout the code: the *ns-3 callback*. This chapter provides some
motivation on the callback, guidance on how to use it, and details on its
implementation.

Callbacks Motivation
********************

Consider that you have two simulation models A and B, and you wish to have them
pass information between them during the simulation. One way that you can do
that is that you can make A and B each explicitly knowledgeable about the other,
so that they can invoke methods on each other::

  class A {
  public:
    void ReceiveInput ( // parameters );
    ...
  }

  (in another source file:)

  class B {
  public:
    void DoSomething (void);
    ...

  private:
    A* a_instance; // pointer to an A
  }

  void
  B::DoSomething()
  {
    // Tell a_instance that something happened
    a_instance->ReceiveInput ( // parameters);
    ...
  }

This certainly works, but it has the drawback that it introduces a dependency on
A and B to know about the other at compile time (this makes it harder to have
independent compilation units in the simulator) and is not generalized; if in a
later usage scenario, B needs to talk to a completely different C object, the
source code for B needs to be changed to add a ``c_instance`` and so forth. It
is easy to see that this is a brute force mechanism of communication that can
lead to programming cruft in the models.  

This is not to say that objects should not know about one another if there is a
hard dependency between them, but that often the model can be made more flexible
if its interactions are less constrained at compile time.

This is not an abstract problem for network simulation research, but rather it
has been a source of problems in previous simulators, when researchers want to
extend or modify the system to do different things (as they are apt to do in
research). Consider, for example, a user who wants to add an IPsec security
protocol sublayer between TCP and IP::

  ------------                   -----------
  |   TCP    |                   |  TCP    |
  ------------                   -----------
       |           becomes ->        |
  -----------                    -----------
  |   IP    |                    | IPsec   |
  -----------                    -----------
                                     |
                                 -----------
                                 |   IP    |
                                 -----------


If the simulator has made assumptions, and hard coded into the code, that IP
always talks to a transport protocol above, the user may be forced to hack the
system to get the desired interconnections. This is clearly not an optimal way
to design a generic simulator.

Callbacks Background
********************

.. note:: Readers familiar with programming callbacks may skip this tutorial
    section.

The basic mechanism that allows one to address the problem above is known as a
*callback*. The ultimate goal is to allow one piece of code to call a function
(or method in C++) without any specific inter-module dependency.

This ultimately means you need some kind of indirection -- you treat the address
of the called function as a variable.  This variable is called a
pointer-to-function variable. The relationship between function and
pointer-to-function pointer is really no different that that of object and
pointer-to-object.

In C the canonical example of a pointer-to-function is a
pointer-to-function-returning-integer (PFI). For a PFI taking one int parameter,
this could be declared like,::

  int (*pfi)(int arg) = 0;

What you get from this is a variable named simply ``pfi`` that is initialized to
the value 0. If you want to initialize this pointer to something meaningful, you
have to have a function with a matching signature. In this case::

  int MyFunction (int arg) {}

If you have this target, you can initialize the variable to point to your
function like::

  pfi = MyFunction;

You can then call MyFunction indirectly using the more suggestive form of the
call::

  int result = (*pfi) (1234);

This is suggestive since it looks like you are dereferencing the function
pointer just like you would dereference any pointer. Typically, however, people
take advantage of the fact that the compiler knows what is going on and will
just use a shorter form::

  int result = pfi (1234);

Notice that the function pointer obeys value semantics, so you can pass it
around like any other value. Typically, when you use an asynchronous interface
you will pass some entity like this to a function which will perform an action
and *call back* to let you know it completed. It calls back by following the
indirection and executing the provided function.

In C++ you have the added complexity of objects. The analogy with the PFI above
means you have a pointer to a member function returning an int (PMI) instead of
the pointer to function returning an int (PFI).

The declaration of the variable providing the indirection looks only slightly
different::

  int (MyClass::*pmi) (int arg) = 0;

This declares a variable named ``pmi`` just as the previous example declared a
variable named ``pfi``. Since the will be to call a method of an instance of a
particular class, one must declare that method in a class::

  class MyClass {
  public:
    int MyMethod (int arg);
  };

Given this class declaration, one would then initialize that variable like
this::

  pmi = &MyClass::MyMethod;

This assigns the address of the code implementing the method to the variable,
completing the indirection. In order to call a method, the code needs a ``this``
pointer. This, in turn, means there must be an object of MyClass to refer to. A
simplistic example of this is just calling a method indirectly (think virtual
function)::

  int (MyClass::*pmi) (int arg) = 0;  // Declare a PMI
  pmi = &MyClass::MyMethod;           // Point at the implementation code

  MyClass myClass;                    // Need an instance of the class
  (myClass.*pmi) (1234);              // Call the method with an object ptr

Just like in the C example, you can use this in an asynchronous call to another
module which will *call back* using a method and an object pointer. The
straightforward extension one might consider is to pass a pointer to the object
and the PMI variable. The module would just do::

  (*objectPtr.*pmi) (1234);

to execute the callback on the desired object.

One might ask at this time, *what's the point*? The called module will have to
understand the concrete type of the calling object in order to properly make the
callback. Why not just accept this, pass the correctly typed object pointer and
do ``object->Method(1234)`` in the code instead of the callback?  This is
precisely the problem described above. What is needed is a way to decouple the
calling function from the called class completely. This requirement led to the
development of the *Functor*.

A functor is the outgrowth of something invented in the 1960s called a closure.
It is basically just a packaged-up function call, possibly with some state.  

A functor has two parts, a specific part and a generic part, related through
inheritance. The calling code (the code that executes the callback) will execute
a generic overloaded ``operator ()`` of a generic functor to cause the callback
to be called. The called code (the code that wants to be called back) will have
to provide a specialized implementation of the ``operator ()`` that performs the
class-specific work that caused the close-coupling problem above.  

With the specific functor and its overloaded ``operator ()`` created, the called
code then gives the specialized code to the module that will execute the
callback (the calling code).

The calling code will take a generic functor as a parameter, so an implicit cast
is done in the function call to convert the specific functor to a generic
functor.  This means that the calling module just needs to understand the
generic functor type. It is decoupled from the calling code completely.

The information one needs to make a specific functor is the object pointer and
the pointer-to-method address. 

The essence of what needs to happen is that the system declares a generic part
of the functor::

  template <typename T>
  class Functor
  {
  public:
    virtual int operator() (T arg) = 0;
  };

The caller defines a specific part of the functor that really is just there to 
implement the specific ``operator()`` method::

  template <typename T, typename ARG>
  class SpecificFunctor : public Functor<ARG>
  {
  public:
    SpecificFunctor(T* p, int (T::*_pmi)(ARG arg))
    {
      m_p = p;
      m_pmi = _pmi;
    }

    virtual int operator() (ARG arg)
    {
      (*m_p.*m_pmi)(arg);
    }
  private:
    int (T::*m_pmi)(ARG arg);
    T* m_p;
  };

Here is an example of the usage::

  class A
  {
  public:
  A (int a0) : a (a0) {}
  int Hello (int b0)
  {
    std::cout << "Hello from A, a = " << a << " b0 = " << b0 << std::endl;
  }
  int a;
  };

  int main()
  {
    A a(10);
    SpecificFunctor<A, int> sf(&a, &A::Hello);
    sf(5);
  }

.. note:: The previous code is not real ns-3 code.  It is simplistic example
   code used only to illustrate the concepts involved and to help you understand 
   the system more.  Do not expect to find this code anywhere in the ns-3 tree.

Notice that there are two variables defined in the class above.  The m_p 
variable is the object pointer and m_pmi is the variable containing the 
address of the function to execute.

Notice that when ``operator()`` is called, it in turn calls the method provided
with the object pointer using the C++ PMI syntax.

To use this, one could then declare some model code that takes a generic functor
as a parameter::

  void LibraryFunction (Functor functor);

The code that will talk to the model would build a specific functor and pass it to ``LibraryFunction``:: 

  MyClass myClass;
  SpecificFunctor<MyClass, int> functor (&myclass, MyClass::MyMethod);

When ``LibraryFunction`` is done, it executes the callback using the 
``operator()`` on the generic functor it was passed, and in this particular
case, provides the integer argument::

  void 
  LibraryFunction (Functor functor)
  {
    // Execute the library function
    functor(1234);
  }

Notice that ``LibraryFunction`` is completely decoupled from the specific
type of the client.  The connection is made through the Functor polymorphism.

The Callback API in |ns3| implements object-oriented callbacks using
the functor mechanism.  This callback API, being based on C++ templates, is 
type-safe; that is, it performs static type checks to enforce proper signature
compatibility between callers and callees.  It is therefore more type-safe to 
use than traditional function pointers, but the syntax may look imposing at 
first.  This section is designed to walk you through the Callback system so 
that you can be comfortable using it in |ns3|.

Using the Callback API
**********************

The Callback API is fairly minimal, providing only two services:

1. callback type declaration: a way to declare a type of callback
with a given signature, and,

2. callback instantiation: a way to instantiate a
template-generated forwarding callback which can forward any calls
to another C++ class member method or C++ function.

This is best observed via walking through an example, based on
``samples/main-callback.cc``.

Using the Callback API with static functions
++++++++++++++++++++++++++++++++++++++++++++

Consider a function::

  static double
  CbOne (double a, double b)
  {
    std::cout << "invoke cbOne a=" << a << ", b=" << b << std::endl;
    return a;
  }

Consider also the following main program snippet::

  int main (int argc, char *argv[])
  {
    // return type: double
    // first arg type: double
    // second arg type: double
    Callback<double, double, double> one;
  }

This is an example of a C-style callback -- one which does not include or need
a ``this`` pointer.  The function template ``Callback`` is essentially the
declaration of the variable containing the pointer-to-function.  In the example
above, we explicitly showed a pointer to a function that returned an integer and
took a single integer as a parameter,  The ``Callback`` template function is
a generic version of that -- it is used to declare the type of a callback.  

.. note:: Readers unfamiliar with C++ templates may consult `<http://www.cplusplus.com/doc/tutorial/templates/>`_.

The ``Callback`` template requires one mandatory argument (the return type 
of the function to be assigned to this callback) and up to five optional 
arguments, which each specify the type of the arguments (if your particular
callback function has more than five arguments, then this can be handled
by extending the callback implementation).

So in the above example, we have a declared a callback named "one" that will
eventually hold a function pointer.  The signature of the function that it will
hold must return double and must support two double arguments.  If one tries 
to pass a function whose signature does not match the declared callback, 
a compilation error will occur.  Also, if one tries to assign to a callback
an incompatible one, compilation will succeed but a run-time 
NS_FATAL_ERROR will be raised.  The sample program 
``src/core/examples/main-callback.cc`` demonstrates both of these error cases
at the end of the ``main()`` program.

Now, we need to tie together this callback instance and the actual target function
(CbOne).  Notice above that CbOne has the same function signature types as the 
callback-- this is important.  We can pass in any such properly-typed function 
to this callback.  Let's look at this more closely::

  static   double CbOne (double a, double b) {}
             ^             ^         ^
             |             |         |
             |             |         | 
  Callback<double,       double,   double> one;

You can only bind a function to a callback if they have the matching signature.
The first template argument is the return type, and the additional template 
arguments are the types of the arguments of the function signature.

Now, let's bind our callback "one" to the function that matches its signature::

  // build callback instance which points to cbOne function
  one = MakeCallback (&CbOne);

This call to ``MakeCallback`` is, in essence, creating one of the specialized
functors mentioned above.  The variable declared using the ``Callback`` 
template function is going to be playing the part of the generic functor.  The
assignment ``one = MakeCallback (&CbOne)`` is the cast that converts the 
specialized functor known to the callee to a generic functor known to the caller.

Then, later in the program, if the callback is needed, it can be used as follows::

  NS_ASSERT (!one.IsNull ());

  // invoke cbOne function through callback instance
  double retOne;
  retOne = one (10.0, 20.0);

The check for ``IsNull()`` ensures that the callback is not null -- that there 
is a function to call behind this callback.  Then, ``one()`` executes the
generic ``operator()`` which is really overloaded with a specific implementation
of ``operator()`` and returns the same result as if ``CbOne()`` had been 
called directly.

Using the Callback API with member functions
++++++++++++++++++++++++++++++++++++++++++++

Generally, you will not be calling static functions but instead public member 
functions of an object.  In this case, an extra argument is needed to the 
MakeCallback function, to tell the system on which object the function should be 
invoked.  Consider this example, also from main-callback.cc::

  class MyCb {
  public:
    int CbTwo (double a) {
        std::cout << "invoke cbTwo a=" << a << std::endl;
        return -5;
    }
  };

  int main ()
  {
    ...
    // return type: int
    // first arg type: double
    Callback<int, double> two;
    MyCb cb;
    // build callback instance which points to MyCb::cbTwo
    two = MakeCallback (&MyCb::CbTwo, &cb);
    ...
  }

Here, we pass an additional object pointer to the ``MakeCallback<>`` function.
Recall from the background section above that ``Operator()`` will use the pointer to 
member syntax when it executes on an object::

      virtual int operator() (ARG arg)
      {
        (*m_p.*m_pmi)(arg);
      }

And so we needed to provide the two variables (``m_p`` and ``m_pmi``) when
we made the specific functor.  The line::

    two = MakeCallback (&MyCb::CbTwo, &cb);

does precisely that.  In this case, when ``two ()`` is invoked::

  int result = two (1.0);

will result in a call to the ``CbTwo`` member function (method) on the object 
pointed to by ``&cb``.   

Building Null Callbacks
+++++++++++++++++++++++

It is possible for callbacks to be null; hence it may be wise to
check before using them.  There is a special construct for a null
callback, which is preferable to simply passing "0" as an argument;
it is the ``MakeNullCallback<>`` construct::

  two = MakeNullCallback<int, double> ();
  NS_ASSERT (two.IsNull ());

Invoking a null callback is just like invoking a null function pointer: it will
crash at runtime.

Bound Callbacks
***************

A very useful extension to the functor concept is that of a Bound Callback.  
Previously it was mentioned that closures were originally function calls 
packaged up for later execution.  Notice that in all of the Callback 
descriptions above, there is no way to package up any parameters for use 
later -- when the ``Callback`` is called via ``operator()``.  All of 
the parameters are provided by the calling function.  

What if it is desired to allow the client function (the one that provides the
callback) to provide some of the parameters?  `Alexandrescu <http://erdani.com/book/main.html>`_ calls the process of
allowing a client to specify one of the parameters *"binding"*.  One of the 
parameters of ``operator()`` has been bound (fixed) by the client.

Some of our pcap tracing code provides a nice example of this.  There is a
function that needs to be called whenever a packet is received.  This function
calls an object that actually writes the packet to disk in the pcap file 
format.  The signature of one of these functions will be::

  static void DefaultSink (Ptr<PcapFileWrapper> file, Ptr<const Packet> p);

The static keyword means this is a static function which does not need a
``this`` pointer, so it will be using C-style callbacks.  We don't want the
calling code to have to know about anything but the Packet.  What we want in
the calling code is just a call that looks like::

  m_promiscSnifferTrace (m_currentPkt);

What we want to do is to *bind* the ``Ptr<PcapFileWriter> file`` to the 
specific callback implementation when it is created and arrange for the 
``operator()`` of the Callback to provide that parameter for free.

We provide the ``MakeBoundCallback`` template function for that purpose.  It
takes the same parameters as the ``MakeCallback`` template function but also
takes the parameters to be bound.  In the case of the example above::

    MakeBoundCallback (&DefaultSink, file);

will create a specific callback implementation that knows to add in the extra
bound arguments.  Conceptually, it extends the specific functor described above
with one or more bound arguments::

  template <typename T, typename ARG, typename BOUND_ARG>
  class SpecificFunctor : public Functor
   {
   public:
      SpecificFunctor(T* p, int (T::*_pmi)(ARG arg), BOUND_ARG boundArg)
      {
        m_p = p;
        m_pmi = pmi;
        m_boundArg = boundArg;
      }

      virtual int operator() (ARG arg)
      {
        (*m_p.*m_pmi)(m_boundArg, arg);
      }
  private:
      void (T::*m_pmi)(ARG arg);
      T* m_p;
      BOUND_ARG m_boundArg;
   };

You can see that when the specific functor is created, the bound argument is saved
in the functor / callback object itself.  When the ``operator()`` is invoked with
the single parameter, as in::

  m_promiscSnifferTrace (m_currentPkt);

the implementation of ``operator()`` adds the bound parameter into the actual
function call::

  (*m_p.*m_pmi)(m_boundArg, arg);

It's possible to bind two or three arguments as well.  Say we have a function with
signature::

  static void NotifyEvent (Ptr<A> a, Ptr<B> b, MyEventType e);

One can create bound callback binding first two arguments like::

  MakeBoundCallback (&NotifyEvent, a1, b1);

assuming `a1` and `b1` are objects of type `A` and `B` respectively.  Similarly for
three arguments one would have function with a signature::

  static void NotifyEvent (Ptr<A> a, Ptr<B> b, MyEventType e);

Binding three arguments in done with::

  MakeBoundCallback (&NotifyEvent, a1, b1, c1);

again assuming `a1`, `b1` and `c1` are objects of type `A`, `B` and `C` respectively.

This kind of binding can be used for exchanging information between objects in
simulation; specifically, bound callbacks can be used as traced callbacks, which will
be described in the next section.

Traced Callbacks
****************
*Placeholder subsection*

Callback locations in ns-3
**************************

Where are callbacks frequently used in |ns3|?  Here are some of the
more visible ones to typical users:

* Socket API
* Layer-2/Layer-3 API
* Tracing subsystem
* API between IP and routing subsystems

Implementation details
**********************

The code snippets above are simplistic and only designed to illustrate the mechanism
itself.  The actual Callback code is quite complicated and very template-intense and
a deep understanding of the code is not required.  If interested, expert users may
find the following useful.

The code was originally written based on the techniques described in 
`<http://www.codeproject.com/cpp/TTLFunction.asp>`_.
It was subsequently rewritten to follow the architecture outlined in 
`Modern C++ Design, Generic Programming and Design Patterns Applied, Alexandrescu, chapter 5, Generalized Functors <http://www.moderncppdesign.com/book/main.html>`_.

This code uses:

* default template parameters to saves users from having to
  specify empty parameters when the number of parameters
  is smaller than the maximum supported number
* the pimpl idiom: the Callback class is passed around by
  value and delegates the crux of the work to its pimpl pointer.
* two pimpl implementations which derive from CallbackImpl
  FunctorCallbackImpl can be used with any functor-type
  while MemPtrCallbackImpl can be used with pointers to
  member functions.
* a reference list implementation to implement the Callback's
  value semantics.

This code most notably departs from the Alexandrescu implementation in that it
does not use type lists to specify and pass around the types of the callback 
arguments. Of course, it also does not use copy-destruction semantics and 
relies on a reference list rather than autoPtr to hold the pointer.
