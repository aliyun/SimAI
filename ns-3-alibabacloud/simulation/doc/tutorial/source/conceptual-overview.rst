.. include:: replace.txt
.. highlight:: cpp

.. _Conceptual Overview:

Conceptual Overview
-------------------

The first thing we need to do before actually starting to look at or write
|ns3| code is to explain a few core concepts and abstractions in the
system.  Much of this may appear transparently obvious to some, but we
recommend taking the time to read through this section just to ensure you
are starting on a firm foundation.

Key Abstractions
****************

In this section, we'll review some terms that are commonly used in
networking, but have a specific meaning in |ns3|.

Node
++++
In Internet jargon, a computing device that connects to a network is called
a *host* or sometimes an *end system*.  Because |ns3| is a 
*network* simulator, not specifically an *Internet* simulator, we 
intentionally do not use the term host since it is closely associated with
the Internet and its protocols.  Instead, we use a more generic term also
used by other simulators that originates in Graph Theory --- the *node*.

In |ns3| the basic computing device abstraction is called the 
node.  This abstraction is represented in C++ by the class ``Node``.  The 
``Node`` class provides methods for managing the representations of 
computing devices in simulations.

You should think of a ``Node`` as a computer to which you will add 
functionality.  One adds things like applications, protocol stacks and
peripheral cards with their associated drivers to enable the computer to do
useful work.  We use the same basic model in |ns3|.

Application
+++++++++++
Typically, computer software is divided into two broad classes.  *System
Software* organizes various computer resources such as memory, processor
cycles, disk, network, etc., according to some computing model.  System
software usually does not use those resources to complete tasks that directly
benefit a user.  A user would typically run an *application* that acquires
and uses the resources controlled by the system software to accomplish some
goal.  

Often, the line of separation between system and application software is made
at the privilege level change that happens in operating system traps.
In |ns3| there is no real concept of operating system and especially
no concept of privilege levels or system calls.  We do, however, have the
idea of an application.  Just as software applications run on computers to
perform tasks in the "real world," |ns3| applications run on
|ns3| ``Nodes`` to drive simulations in the simulated world.

In |ns3| the basic abstraction for a user program that generates some
activity to be simulated is the application.  This abstraction is represented 
in C++ by the class ``Application``.  The ``Application`` class provides 
methods for managing the representations of our version of user-level 
applications in simulations.  Developers are expected to specialize the
``Application`` class in the object-oriented programming sense to create new
applications.  In this tutorial, we will use specializations of class 
``Application`` called ``UdpEchoClientApplication`` and 
``UdpEchoServerApplication``.  As you might expect, these applications 
compose a client/server application set used to generate and echo simulated 
network packets 

Channel
+++++++

In the real world, one can connect a computer to a network.  Often the media
over which data flows in these networks are called *channels*.  When
you connect your Ethernet cable to the plug in the wall, you are connecting 
your computer to an Ethernet communication channel.  In the simulated world
of |ns3|, one connects a ``Node`` to an object representing a
communication channel.  Here the basic communication subnetwork abstraction 
is called the channel and is represented in C++ by the class ``Channel``.  

The ``Channel`` class provides methods for managing communication 
subnetwork objects and connecting nodes to them.  ``Channels`` may also be
specialized by developers in the object oriented programming sense.  A 
``Channel`` specialization may model something as simple as a wire.  The 
specialized  ``Channel`` can also model things as complicated as a large 
Ethernet switch, or three-dimensional space full of obstructions in the case 
of wireless networks.

We will use specialized versions of the ``Channel`` called
``CsmaChannel``, ``PointToPointChannel`` and ``WifiChannel`` in this
tutorial.  The ``CsmaChannel``, for example, models a version of a 
communication subnetwork that implements a *carrier sense multiple 
access* communication medium.  This gives us Ethernet-like functionality.  

Net Device
++++++++++
It used to be the case that if you wanted to connect a computer to a network,
you had to buy a specific kind of network cable and a hardware device called
(in PC terminology) a *peripheral card* that needed to be installed in
your computer.  If the peripheral card implemented some networking function,
they were called Network Interface Cards, or *NICs*.  Today most 
computers come with the network interface hardware built in and users don't 
see these building blocks.

A NIC will not work without a software driver to control the hardware.  In 
Unix (or Linux), a piece of peripheral hardware is classified as a 
*device*.  Devices are controlled using *device drivers*, and network
devices (NICs) are controlled using *network device drivers*
collectively known as *net devices*.  In Unix and Linux you refer
to these net devices by names such as *eth0*.

In |ns3| the *net device* abstraction covers both the software 
driver and the simulated hardware.  A net device is "installed" in a 
``Node`` in order to enable the ``Node`` to communicate with other 
``Nodes`` in the simulation via ``Channels``.  Just as in a real
computer, a ``Node`` may be connected to more than one ``Channel`` via
multiple ``NetDevices``.

The net device abstraction is represented in C++ by the class ``NetDevice``.
The ``NetDevice`` class provides methods for managing connections to 
``Node`` and ``Channel`` objects; and may be specialized by developers
in the object-oriented programming sense.  We will use the several specialized
versions of the ``NetDevice`` called ``CsmaNetDevice``,
``PointToPointNetDevice``, and ``WifiNetDevice`` in this tutorial.
Just as an Ethernet NIC is designed to work with an Ethernet network, the
``CsmaNetDevice`` is designed to work with a ``CsmaChannel``; the
``PointToPointNetDevice`` is designed to work with a 
``PointToPointChannel`` and a ``WifiNetNevice`` is designed to work with
a ``WifiChannel``.

Topology Helpers
++++++++++++++++
In a real network, you will find host computers with added (or built-in)
NICs.  In |ns3| we would say that you will find ``Nodes`` with 
attached ``NetDevices``.  In a large simulated network you will need to 
arrange many connections between ``Nodes``, ``NetDevices`` and 
``Channels``.

Since connecting ``NetDevices`` to ``Nodes``, ``NetDevices``
to ``Channels``, assigning IP addresses,  etc., are such common tasks
in |ns3|, we provide what we call *topology helpers* to make 
this as easy as possible.  For example, it may take many distinct 
|ns3| core operations to create a NetDevice, add a MAC address, 
install that net device on a ``Node``, configure the node's protocol stack,
and then connect the ``NetDevice`` to a ``Channel``.  Even more
operations would be required to connect multiple devices onto multipoint 
channels and then to connect individual networks together into internetworks.
We provide topology helper objects that combine those many distinct operations
into an easy to use model for your convenience.

A First ns-3 Script
*******************
If you downloaded the system as was suggested above, you will have a release
of |ns3| in a directory called ``repos`` under your home 
directory.  Change into that release directory, and you should find a 
directory structure something like the following:

.. sourcecode:: bash

  AUTHORS          doc        RELEASE_NOTES.md  utils
  bindings         examples   scratch           utils.py
  CHANGES.html     LICENSE    src               VERSION
  contrib          Makefile   test.py           
  CONTRIBUTING.md  README.md  testpy.supp

Change into the ``examples/tutorial`` directory.  You should see a file named 
``first.cc`` located there.  This is a script that will create a simple
point-to-point link between two nodes and echo a single packet between the
nodes.  Let's take a look at that script line by line, so go ahead and open
``first.cc`` in your favorite editor.

Boilerplate
+++++++++++
The first line in the file is an emacs mode line.  This tells emacs about the
formatting conventions (coding style) we use in our source code.  

::

  /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

This is always a somewhat controversial subject, so we might as well get it
out of the way immediately.  The |ns3| project, like most large 
projects, has adopted a coding style to which all contributed code must 
adhere.  If you want to contribute your code to the project, you will 
eventually have to conform to the |ns3| coding standard as described 
in the file ``doc/codingstd.txt`` or shown on the project web page
`here
<http://www.nsnam.org/developers/contributing-code/coding-style/>`_.

We recommend that you, well, just get used to the look and feel of |ns3|
code and adopt this standard whenever you are working with our code.  All of 
the development team and contributors have done so with various amounts of 
grumbling.  The emacs mode line above makes it easier to get the formatting 
correct if you use the emacs editor.

The |ns3| simulator is licensed using the GNU General Public 
License.  You will see the appropriate GNU legalese at the head of every file 
in the |ns3| distribution.  Often you will see a copyright notice for
one of the institutions involved in the |ns3| project above the GPL
text and an author listed below.

::

  /*
   * This program is free software; you can redistribute it and/or modify
   * it under the terms of the GNU General Public License version 2 as
   * published by the Free Software Foundation;
   *
   * This program is distributed in the hope that it will be useful,
   * but WITHOUT ANY WARRANTY; without even the implied warranty of
   * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   * GNU General Public License for more details.
   *
   * You should have received a copy of the GNU General Public License
   * along with this program; if not, write to the Free Software
   * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   */

Module Includes
+++++++++++++++
The code proper starts with a number of include statements.  

::

  #include "ns3/core-module.h"
  #include "ns3/network-module.h"
  #include "ns3/internet-module.h"
  #include "ns3/point-to-point-module.h"
  #include "ns3/applications-module.h"

To help our high-level script users deal with the large number of include 
files present in the system, we group includes according to relatively large 
modules.  We provide a single include file that will recursively load all of 
the include files used in each module.  Rather than having to look up exactly
what header you need, and possibly have to get a number of dependencies right,
we give you the ability to load a group of files at a large granularity.  This
is not the most efficient approach but it certainly makes writing scripts much
easier.

Each of the |ns3| include files is placed in a directory called 
``ns3`` (under the build directory) during the build process to help avoid
include file name collisions.  The ``ns3/core-module.h`` file corresponds 
to the ns-3 module you will find in the directory ``src/core`` in your 
downloaded release distribution.  If you list this directory you will find a
large number of header files.  When you do a build, ns3 will place public
header files in an ``ns3`` directory under the appropriate 
``build/debug`` or ``build/optimized`` directory depending on your 
configuration.  CMake will also automatically generate a module include file to
load all of the public header files.

Since you are, of course, following this tutorial religiously, you will 
already have done a

.. sourcecode:: bash

  $ ./ns3 configure -d debug --enable-examples --enable-tests

in order to configure the project to perform debug builds that include 
examples and tests.  You will also have done a

.. sourcecode:: bash

  $ ./ns3 build

to build the project.  So now if you look in the directory 
``../../build/include/ns3`` you will find the four module include files shown
above.  You can take a look at the contents of these files and find that they
do include all of the public include files in their respective modules.

Ns3 Namespace
+++++++++++++
The next line in the ``first.cc`` script is a namespace declaration.

::

  using namespace ns3;

The |ns3| project is implemented in a C++ namespace called 
``ns3``.  This groups all |ns3|-related declarations in a scope
outside the global namespace, which we hope will help with integration with 
other code.  The C++ ``using`` statement introduces the |ns3|
namespace into the current (global) declarative region.  This is a fancy way
of saying that after this declaration, you will not have to type ``ns3::``
scope resolution operator before all of the |ns3| code in order to use
it.  If you are unfamiliar with namespaces, please consult almost any C++ 
tutorial and compare the ``ns3`` namespace and usage here with instances of
the ``std`` namespace and the ``using namespace std;`` statements you 
will often find in discussions of ``cout`` and streams.

Logging
+++++++
The next line of the script is the following,

::

  NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");

We will use this statement as a convenient place to talk about our Doxygen
documentation system.  If you look at the project web site, 
`ns-3 project
<http://www.nsnam.org>`_, you will find a link to "Documentation" in the navigation bar.  If you select this link, you will be
taken to our documentation page. There 
is a link to "Latest Release" that will take you to the documentation
for the latest stable release of |ns3|.
If you select the "API Documentation" link, you will be
taken to the |ns3| API documentation page.

Along the left side, you will find a graphical representation of the structure
of the documentation.  A good place to start is the ``NS-3 Modules``
"book" in the |ns3| navigation tree.  If you expand ``Modules`` 
you will see a list of |ns3| module documentation.  The concept of 
module here ties directly into the module include files discussed above.  The |ns3| logging subsystem is discussed in the :ref:`UsingLogging` section, so
we'll get to it later in this tutorial, but you can find out about the above
statement by looking at the ``Core`` module, then expanding the 
``Debugging tools`` book, and then selecting the ``Logging`` page.  Click
on ``Logging``.

You should now be looking at the Doxygen documentation for the Logging module.
In the list of ``Macros``'s at the top of the page you will see the entry
for ``NS_LOG_COMPONENT_DEFINE``.  Before jumping in, it would probably be 
good to look for the "Detailed Description" of the logging module to get a 
feel for the overall operation.  You can either scroll down or select the
"More..." link under the collaboration diagram to do this.

Once you have a general idea of what is going on, go ahead and take a look at
the specific ``NS_LOG_COMPONENT_DEFINE`` documentation.  I won't duplicate
the documentation here, but to summarize, this line declares a logging 
component called ``FirstScriptExample`` that allows you to enable and 
disable console message logging by reference to the name.

Main Function
+++++++++++++
The next lines of the script you will find are,

::

  int
  main (int argc, char *argv[])
  {

This is just the declaration of the main function of your program (script).
Just as in any C++ program, you need to define a main function that will be 
the first function run.  There is nothing at all special here.  Your 
|ns3| script is just a C++ program.

The next line sets the time resolution to one nanosecond, which happens
to be the default value:

::

    Time::SetResolution (Time::NS);

The resolution is the smallest time value that can be represented (as well as
the smallest representable difference between two time values).
You can change the resolution exactly once.  The mechanism enabling this
flexibility is somewhat memory hungry, so once the resolution has been
set explicitly we release the memory, preventing further updates.   (If
you don't set the resolution explicitly, it will default to one nanosecond,
and the memory will be released when the simulation starts.)

The next two lines of the script are used to enable two logging components that
are built into the Echo Client and Echo Server applications:

::

    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

If you have read over the Logging component documentation you will have seen
that there are a number of levels of logging verbosity/detail that you can 
enable on each component.  These two lines of code enable debug logging at the
INFO level for echo clients and servers.  This will result in the application
printing out messages as packets are sent and received during the simulation.

Now we will get directly to the business of creating a topology and running 
a simulation.  We use the topology helper objects to make this job as
easy as possible.

Topology Helpers
++++++++++++++++
NodeContainer
~~~~~~~~~~~~~
The next two lines of code in our script will actually create the 
|ns3| ``Node`` objects that will represent the computers in the 
simulation.  

::

    NodeContainer nodes;
    nodes.Create (2);

Let's find the documentation for the ``NodeContainer`` class before we
continue.  Another way to get into the documentation for a given class is via 
the ``Classes`` tab in the Doxygen pages.  If you still have the Doxygen 
handy, just scroll up to the top of the page and select the ``Classes`` 
tab.  You should see a new set of tabs appear, one of which is 
``Class List``.  Under that tab you will see a list of all of the 
|ns3| classes.  Scroll down, looking for ``ns3::NodeContainer``.
When you find the class, go ahead and select it to go to the documentation for
the class.

You may recall that one of our key abstractions is the ``Node``.  This
represents a computer to which we are going to add things like protocol stacks,
applications and peripheral cards.  The ``NodeContainer`` topology helper
provides a convenient way to create, manage and access any ``Node`` objects
that we create in order to run a simulation.  The first line above just 
declares a NodeContainer which we call ``nodes``.  The second line calls the
``Create`` method on the ``nodes`` object and asks the container to 
create two nodes.  As described in the Doxygen, the container calls down into
the |ns3| system proper to create two ``Node`` objects and stores
pointers to those objects internally.

The nodes as they stand in the script do nothing.  The next step in 
constructing a topology is to connect our nodes together into a network.
The simplest form of network we support is a single point-to-point link 
between two nodes.  We'll construct one of those links here.

PointToPointHelper
~~~~~~~~~~~~~~~~~~
We are constructing a point to point link, and, in a pattern which will become
quite familiar to you, we use a topology helper object to do the low-level
work required to put the link together.  Recall that two of our key 
abstractions are the ``NetDevice`` and the ``Channel``.  In the real
world, these terms correspond roughly to peripheral cards and network cables.  
Typically these two things are intimately tied together and one cannot expect
to interchange, for example, Ethernet devices and wireless channels.  Our 
Topology Helpers follow this intimate coupling and therefore you will use a
single ``PointToPointHelper`` to configure and connect |ns3|
``PointToPointNetDevice`` and ``PointToPointChannel`` objects in this 
script.

The next three lines in the script are,

::

    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
    pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

The first line,

::

    PointToPointHelper pointToPoint;

instantiates a ``PointToPointHelper`` object on the stack.  From a 
high-level perspective the next line,

::

    pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));

tells the ``PointToPointHelper`` object to use the value "5Mbps"
(five megabits per second) as the "DataRate" when it creates a 
``PointToPointNetDevice`` object.

From a more detailed perspective, the string "DataRate" corresponds
to what we call an ``Attribute`` of the ``PointToPointNetDevice``.
If you look at the Doxygen for class ``ns3::PointToPointNetDevice`` and 
find the documentation for the ``GetTypeId`` method, you will find a list
of  ``Attributes`` defined for the device.  Among these is the "DataRate"
``Attribute``.  Most user-visible |ns3| objects have similar lists of 
``Attributes``.  We use this mechanism to easily configure simulations without
recompiling as you will see in a following section.

Similar to the "DataRate" on the ``PointToPointNetDevice`` you will find a
"Delay" ``Attribute`` associated with the ``PointToPointChannel``.  The 
final line,

::

    pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

tells the ``PointToPointHelper`` to use the value "2ms" (two milliseconds)
as the value of the propagation delay of every point to point channel it 
subsequently creates.

NetDeviceContainer
~~~~~~~~~~~~~~~~~~
At this point in the script, we have a ``NodeContainer`` that contains
two nodes.  We have a ``PointToPointHelper`` that is primed and ready to 
make ``PointToPointNetDevices`` and wire ``PointToPointChannel`` objects
between them.  Just as we used the ``NodeContainer`` topology helper object
to create the ``Nodes`` for our simulation, we will ask the 
``PointToPointHelper`` to do the work involved in creating, configuring and
installing our devices for us.  We will need to have a list of all of the 
NetDevice objects that are created, so we use a NetDeviceContainer to hold 
them just as we used a NodeContainer to hold the nodes we created.  The 
following two lines of code,

::

    NetDeviceContainer devices;
    devices = pointToPoint.Install (nodes);

will finish configuring the devices and channel.  The first line declares the 
device container mentioned above and the second does the heavy lifting.  The 
``Install`` method of the ``PointToPointHelper`` takes a 
``NodeContainer`` as a parameter.  Internally, a ``NetDeviceContainer`` 
is created.  For each node in the ``NodeContainer`` (there must be exactly 
two for a point-to-point link) a ``PointToPointNetDevice`` is created and 
saved in the device container.  A ``PointToPointChannel`` is created and 
the two ``PointToPointNetDevices`` are attached.  When objects are created
by the ``PointToPointHelper``, the ``Attributes`` previously set in the 
helper are used to initialize the corresponding ``Attributes`` in the 
created objects.

After executing the ``pointToPoint.Install (nodes)`` call we will have
two nodes, each with an installed point-to-point net device and a single
point-to-point channel between them.  Both devices will be configured to 
transmit data at five megabits per second over the channel which has a two 
millisecond transmission delay.

InternetStackHelper
~~~~~~~~~~~~~~~~~~~
We now have nodes and devices configured, but we don't have any protocol stacks
installed on our nodes.  The next two lines of code will take care of that.

::

    InternetStackHelper stack;
    stack.Install (nodes);

The ``InternetStackHelper`` is a topology helper that is to internet stacks
what the ``PointToPointHelper`` is to point-to-point net devices.  The
``Install`` method takes a ``NodeContainer`` as a parameter.  When it is
executed, it will install an Internet Stack (TCP, UDP, IP, etc.) on each of
the nodes in the node container.

Ipv4AddressHelper
~~~~~~~~~~~~~~~~~
Next we need to associate the devices on our nodes with IP addresses.  We 
provide a topology helper to manage the allocation of IP addresses.  The only
user-visible API is to set the base IP address and network mask to use when
performing the actual address allocation (which is done at a lower level 
inside the helper).

The next two lines of code in our example script, ``first.cc``,

::

    Ipv4AddressHelper address;
    address.SetBase ("10.1.1.0", "255.255.255.0");

declare an address helper object and tell it that it should begin allocating IP
addresses from the network 10.1.1.0 using the mask 255.255.255.0 to define 
the allocatable bits.  By default the addresses allocated will start at one
and increase monotonically, so the first address allocated from this base will
be 10.1.1.1, followed by 10.1.1.2, etc.  The low level |ns3| system
actually remembers all of the IP addresses allocated and will generate a
fatal error if you accidentally cause the same address to be generated twice 
(which is a very hard to debug error, by the way).

The next line of code,

::

    Ipv4InterfaceContainer interfaces = address.Assign (devices);

performs the actual address assignment.  In |ns3| we make the
association between an IP address and a device using an ``Ipv4Interface``
object.  Just as we sometimes need a list of net devices created by a helper 
for future reference we sometimes need a list of ``Ipv4Interface`` objects.
The ``Ipv4InterfaceContainer`` provides this functionality.

Now we have a point-to-point network built, with stacks installed and IP 
addresses assigned.  What we need at this point are applications to generate
traffic.

Applications
++++++++++++
Another one of the core abstractions of the ns-3 system is the 
``Application``.  In this script we use two specializations of the core
|ns3| class ``Application`` called ``UdpEchoServerApplication``
and ``UdpEchoClientApplication``.  Just as we have in our previous 
explanations,  we use helper objects to help configure and manage the 
underlying objects.  Here, we use ``UdpEchoServerHelper`` and
``UdpEchoClientHelper`` objects to make our lives easier.

UdpEchoServerHelper
~~~~~~~~~~~~~~~~~~~
The following lines of code in our example script, ``first.cc``, are used
to set up a UDP echo server application on one of the nodes we have previously
created.

::

    UdpEchoServerHelper echoServer (9);

    ApplicationContainer serverApps = echoServer.Install (nodes.Get (1));
    serverApps.Start (Seconds (1.0));
    serverApps.Stop (Seconds (10.0));

The first line of code in the above snippet declares the 
``UdpEchoServerHelper``.  As usual, this isn't the application itself, it
is an object used to help us create the actual applications.  One of our 
conventions is to place *required* ``Attributes`` in the helper constructor.
In this case, the helper can't do anything useful unless it is provided with
a port number that the client also knows about.  Rather than just picking one 
and hoping it all works out, we require the port number as a parameter to the 
constructor.  The constructor, in turn, simply does a ``SetAttribute``
with the passed value.  If you want, you can set the "Port" ``Attribute``
to another value later using ``SetAttribute``.

Similar to many other helper objects, the ``UdpEchoServerHelper`` object 
has an ``Install`` method.  It is the execution of this method that actually
causes the underlying echo server application to be instantiated and attached
to a node.  Interestingly, the ``Install`` method takes a
``NodeContainter`` as a parameter just as the other ``Install`` methods
we have seen.  This is actually what is passed to the method even though it 
doesn't look so in this case.  There is a C++ *implicit conversion* at
work here that takes the result of ``nodes.Get (1)`` (which returns a smart
pointer to a node object --- ``Ptr<Node>``) and uses that in a constructor
for an unnamed ``NodeContainer`` that is then passed to ``Install``.
If you are ever at a loss to find a particular method signature in C++ code
that compiles and runs just fine, look for these kinds of implicit conversions.  

We now see that ``echoServer.Install`` is going to install a
``UdpEchoServerApplication`` on the node found at index number one of the
``NodeContainer`` we used to manage our nodes.  ``Install`` will return
a container that holds pointers to all of the applications (one in this case 
since we passed a ``NodeContainer`` containing one node) created by the 
helper.

Applications require a time to "start" generating traffic and may take an
optional time to "stop".  We provide both.  These times are set using  the
``ApplicationContainer`` methods ``Start`` and ``Stop``.  These 
methods take ``Time`` parameters.  In this case, we use an *explicit*
C++ conversion sequence to take the C++ double 1.0 and convert it to an 
|ns3| ``Time`` object using a ``Seconds`` cast.  Be aware that
the conversion rules may be controlled by the model author, and C++ has its
own rules, so you can't always just assume that parameters will be happily 
converted for you.  The two lines,

::

    serverApps.Start (Seconds (1.0));
    serverApps.Stop (Seconds (10.0));

will cause the echo server application to ``Start`` (enable itself) at one
second into the simulation and to ``Stop`` (disable itself) at ten seconds
into the simulation.  By virtue of the fact that we have declared a simulation
event (the application stop event) to be executed at ten seconds, the simulation
will last *at least* ten seconds.

UdpEchoClientHelper
~~~~~~~~~~~~~~~~~~~

The echo client application is set up in a method substantially similar to
that for the server.  There is an underlying ``UdpEchoClientApplication``
that is managed by an ``UdpEchoClientHelper``.

::

    UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
    echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
    echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
    echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

    ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
    clientApps.Start (Seconds (2.0));
    clientApps.Stop (Seconds (10.0));

For the echo client, however, we need to set five different ``Attributes``.
The first two ``Attributes`` are set during construction of the 
``UdpEchoClientHelper``.  We pass parameters that are used (internally to
the helper) to set the "RemoteAddress" and "RemotePort" ``Attributes``
in accordance with our convention to make required ``Attributes`` parameters
in the helper constructors.  

Recall that we used an ``Ipv4InterfaceContainer`` to keep track of the IP 
addresses we assigned to our devices.  The zeroth interface in the 
``interfaces`` container is going to correspond to the IP address of the 
zeroth node in the ``nodes`` container.  The first interface in the 
``interfaces`` container corresponds to the IP address of the first node 
in the ``nodes`` container.  So, in the first line of code (from above), we
are creating the helper and telling it so set the remote address of the client
to be  the IP address assigned to the node on which the server resides.  We 
also tell it to arrange to send packets to port nine.

The "MaxPackets" ``Attribute`` tells the client the maximum number of 
packets we allow it to send during the simulation.  The "Interval" 
``Attribute`` tells the client how long to wait between packets, and the
"PacketSize" ``Attribute`` tells the client how large its packet payloads
should be.  With this particular combination of ``Attributes``, we are 
telling the client to send one 1024-byte packet.

Just as in the case of the echo server, we tell the echo client to ``Start``
and ``Stop``, but here we start the client one second after the server is
enabled (at two seconds into the simulation).

Simulator
+++++++++
What we need to do at this point is to actually run the simulation.  This is 
done using the global function ``Simulator::Run``.

::

    Simulator::Run ();

When we previously called the methods,

::

    serverApps.Start (Seconds (1.0));
    serverApps.Stop (Seconds (10.0));
    ...
    clientApps.Start (Seconds (2.0));
    clientApps.Stop (Seconds (10.0));

we actually scheduled events in the simulator at 1.0 seconds, 2.0 seconds and
two events at 10.0 seconds.  When ``Simulator::Run`` is called, the system 
will begin looking through the list of scheduled events and executing them.  
First it will run the event at 1.0 seconds, which will enable the echo server 
application (this event may, in turn, schedule many other events).  Then it 
will run the event scheduled for t=2.0 seconds which will start the echo client
application.  Again, this event may schedule many more events.  The start event
implementation in the echo client application will begin the data transfer phase
of the simulation by sending a packet to the server.

The act of sending the packet to the server will trigger a chain of events
that will be automatically scheduled behind the scenes and which will perform 
the mechanics of the packet echo according to the various timing parameters 
that we have set in the script.

Eventually, since we only send one packet (recall the ``MaxPackets`` 
``Attribute`` was set to one), the chain of events triggered by 
that single client echo request will taper off and the simulation will go 
idle.  Once this happens, the remaining events will be the ``Stop`` events
for the server and the client.  When these events are executed, there are
no further events to process and ``Simulator::Run`` returns.  The simulation
is then complete.

All that remains is to clean up.  This is done by calling the global function 
``Simulator::Destroy``.  As the helper functions (or low level 
|ns3| code) executed, they arranged it so that hooks were inserted in
the simulator to destroy all of the objects that were created.  You did not 
have to keep track of any of these objects yourself --- all you had to do 
was to call ``Simulator::Destroy`` and exit.  The |ns3| system
took care of the hard part for you.  The remaining lines of our first 
|ns3| script, ``first.cc``, do just that:

::

    Simulator::Destroy ();
    return 0;
  }

When the simulator will stop?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|ns3| is a Discrete Event (DE) simulator. In such a simulator, each event is
associated with its execution time, and the simulation proceeds by executing
events in the temporal order of simulation time.  Events may cause future
events to be scheduled (for example, a timer may reschedule itself to
expire at the next interval).

The initial events are usually triggered by each object, e.g., IPv6 will 
schedule Router Advertisements, Neighbor Solicitations, etc., 
an Application schedule the first packet sending event, etc.

When an event is processed, it may generate zero, one or more events.
As a simulation executes, events are consumed, but more events may (or may
not) be generated.
The simulation will stop automatically when no further events are in the 
event queue, or when a special Stop event is found. The Stop event is 
created through the 
``Simulator::Stop (stopTime);`` function.

There is a typical case where ``Simulator::Stop`` is absolutely necessary 
to stop the simulation: when there is a self-sustaining event.
Self-sustaining (or recurring) events are events that always reschedule 
themselves. As a consequence, they always keep the event queue non-empty.

There are many protocols and modules containing recurring events, e.g.:

* FlowMonitor - periodic check for lost packets
* RIPng - periodic broadcast of routing tables update
* etc.

In these cases, ``Simulator::Stop`` is necessary to gracefully stop the 
simulation.  In addition, when |ns3| is in emulation mode, the 
``RealtimeSimulator`` is used to keep the simulation clock aligned with 
the machine clock, and ``Simulator::Stop`` is necessary to stop the 
process.  

Many of the simulation programs in the tutorial do not explicitly call
``Simulator::Stop``, since the event queue will automatically run out
of events.  However, these programs will also accept a call to 
``Simulator::Stop``.  For example, the following additional statement
in the first example program will schedule an explicit stop at 11 seconds: 

::

  +  Simulator::Stop (Seconds (11.0));
     Simulator::Run ();
     Simulator::Destroy ();
     return 0;
   }

The above will not actually change the behavior of this program, since
this particular simulation naturally ends after 10 seconds.  But if you 
were to change the stop time in the above statement from 11 seconds to 1 
second, you would notice that the simulation stops before any output is 
printed to the screen (since the output occurs around time 2 seconds of 
simulation time).

It is important to call ``Simulator::Stop`` *before* calling 
``Simulator::Run``; otherwise, ``Simulator::Run`` may never return control
to the main program to execute the stop!

Building Your Script
++++++++++++++++++++
We have made it trivial to build your simple scripts.  All you have to do is 
to drop your script into the scratch directory and it will automatically be 
built if you run ns3.  Let's try it.  Copy ``examples/tutorial/first.cc`` into
the ``scratch`` directory after changing back into the top level directory.

.. sourcecode:: bash

  $ cd ../..
  $ cp examples/tutorial/first.cc scratch/myfirst.cc

Now build your first example script using ns3:

.. sourcecode:: bash

  $ ./ns3

You should see messages reporting that your ``myfirst`` example was built
successfully.

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  [614/708] cxx: scratch/myfirst.cc -> build/debug/scratch/myfirst_3.o
  [706/708] cxx_link: build/debug/scratch/myfirst_3.o -> build/debug/scratch/myfirst
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (2.357s)

You can now run the example (note that if you build your program in the scratch
directory you must run it out of the scratch directory):

.. sourcecode:: bash

  $ ./ns3 run scratch/myfirst

You should see some output:

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.418s)
  Sent 1024 bytes to 10.1.1.2
  Received 1024 bytes from 10.1.1.1
  Received 1024 bytes from 10.1.1.2

Here you see that the build system checks to make sure that the file has been
build and then runs it.  You see the logging component on the echo client 
indicate that it has sent one 1024 byte packet to the Echo Server on 
10.1.1.2.  You also see the logging component on the echo server say that
it has received the 1024 bytes from 10.1.1.1.  The echo server silently 
echoes the packet and you see the echo client log that it has received its 
packet back from the server.

Ns-3 Source Code
****************

Now that you have used some of the |ns3| helpers you may want to 
have a look at some of the source code that implements that functionality.
The most recent code can be browsed on our web server at the following link:
https://gitlab.com/nsnam/ns-3-dev.git.  There, you will see the Git/GitLab
summary page for our |ns3| development tree.

At the top of the page, you will see a number of links,

.. sourcecode:: text

  summary | shortlog | changelog | graph | tags | files 

Go ahead and select the ``files`` link.  This is what the top-level of
most of our *repositories* will look:

.. sourcecode:: text

  drwxr-xr-x                               [up]
  -rwxrwxrwx  12507 nov 23 23:12 AUTHORS
  drwxrwxrwx      0 dez 30  2020 bindings
  drwxrwxrwx   4096 nov 28 11:25 build-support
  -rwxrwxrwx 189870 nov 28 10:13 CHANGES.html
  -rwxrwxrwx   1490 nov 22 10:56 .clang-format
  -rwxrwxrwx   5079 nov 28 10:50 CMakeLists.txt
  drwxrwxrwx      0 nov 28 00:56 contrib
  -rwxrwxrwx  18336 nov 23 23:12 CONTRIBUTING.md
  drwxrwxrwx   4096 nov 28 11:30 doc
  -rwxrwxrwx    564 nov 22 10:56 .editorconfig
  drwxrwxrwx   4096 nov 28 00:28 examples
  drwxrwxrwx   4096 nov 28 11:15 .git
  -rwxrwxrwx     42 nov 22 10:56 .gitattributes
  -rwxrwxrwx    395 nov 28 11:25 .gitignore
  drwxrwxrwx   4096 nov 28 11:37 .idea
  -rwxrwxrwx  17987 nov 22 10:56 LICENSE
  -rwxrwxrwx   6841 nov 22 10:56 .mailmap
  drwxrwxrwx      0 jul 13 20:15 .nompi
  -rwxrwxrwx  31942 nov 28 11:16 ns3
  -rwxrwxrwx   4210 nov 28 11:25 README.md
  -rwxrwxrwx 185874 nov 28 10:13 RELEASE_NOTES.md
  drwxrwxrwx   4096 nov 28 00:28 scratch
  drwxrwxrwx      0 out 21 11:42 .settings
  drwxrwxrwx   8192 nov 28 00:56 src
  -rwxrwxrwx  83149 nov 28 11:28 test.py
  -rwxrwxrwx    131 nov 22 10:56 testpy.supp
  drwxrwxrwx   4096 nov 28 11:25 utils
  -rwxrwxrwx   4227 nov 28 11:25 utils.py
  -rwxrwxrwx      6 nov 22 10:56 VERSION

Our example scripts are in the ``examples`` directory.  If you click on ``examples``
you will see a list of subdirectories.  One of the files in ``tutorial`` subdirectory is ``first.cc``.  If
you click on ``first.cc`` you will find the code you just walked through.

The source code is mainly in the ``src`` directory.  You can view source
code either by clicking on the directory name or by clicking on the ``files``
link to the right of the directory name.  If you click on the ``src``
directory, you will be taken to the listing of the ``src`` subdirectories.  If you 
then click on ``core`` subdirectory, you will find a list of files.  The first file
you will find (as of this writing) is ``abort.h``.  If you click on the 
``abort.h`` link, you will be sent to the source file for ``abort.h`` which 
contains useful macros for exiting scripts if abnormal conditions are detected.

The source code for the helpers we have used in this chapter can be found in the 
``src/applications/helper`` directory.  Feel free to poke around in the directory tree to
get a feel for what is there and the style of |ns3| programs.
