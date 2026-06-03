.. include:: replace.txt
.. highlight:: cpp

Helpers
-------

The above chapters introduced you to various |ns3| programming concepts such as
smart pointers for reference-counted memory management, attributes, namespaces,
callbacks, etc. Users who work at this low-level API can interconnect |ns3|
objects with fine granularity. However, a simulation program written entirely
using the low-level API would be quite long and tedious to code. For this
reason, a separate so-called "helper API" has been overlaid on the core |ns3|
API. If you have read the |ns3| tutorial, you will already be familiar with the
helper API, since it is the API that new users are typically introduced to
first.  In this chapter, we introduce the design philosophy of the helper API
and contrast it to the low-level API. If you become a heavy user of |ns3|, you
will likely move back and forth between these APIs even in the same program.

The helper API has a few goals:

#. the rest of ``src/`` has no dependencies on the helper API; anything that can
   be done with the helper API can be coded also at the low-level API
#. **Containers:** Often simulations will need to do a number of identical
   actions to groups of objects. The helper API makes heavy use of containers of
   similar objects to which similar or identical operations can be performed.
#. The helper API is not generic; it does not strive to maximize code reuse. So,
   programming constructs such as polymorphism and templates that achieve code
   reuse are not as prevalent. For instance, there are separate CsmaNetDevice
   helpers and PointToPointNetDevice helpers but they do not derive from a
   common NetDevice base class.
#. The helper API typically works with stack-allocated (vs. heap-allocated)
   objects. For some programs, |ns3| users may not need to worry about any low
   level Object Create or Ptr handling; they can make do with containers of
   objects and stack-allocated helpers that operate on them.

The helper API is really all about making |ns3| programs easier to write and
read, without taking away the power of the low-level interface. The rest of this
chapter provides some examples of the programming conventions of the helper API.
