.. highlight:: bash

Click Modular Router Integration
--------------------------------

Click is a software architecture for building configurable routers.
By using different combinations of packet processing units called elements,
a Click router can be made to perform a specific kind of functionality.
This flexibility provides a good platform for testing and experimenting with
different protocols. 

Model Description
*****************

The source code for the Click model lives in the directory ``src/click``.

Design
======

ns-3's design is well suited for an integration with Click due to the following reasons:

* Packets in ns-3 are serialised/deserialised as they move up/down the stack. This allows ns-3 packets to be passed to and from Click as they are.
* This also means that any kind of ns-3 traffic generator and transport should work easily on top of Click.
* By striving to implement click as an Ipv4RoutingProtocol instance, we can avoid significant changes to the LL and MAC layer of the ns-3 code. 

The design goal was to make the ns-3-click public API simple enough such that the user needs to merely add an Ipv4ClickRouting instance to the node, and inform each Click node of the Click configuration file (.click file) that it is to use. 

This model implements the interface to the Click Modular Router and
provides the Ipv4ClickRouting class to allow a node to use Click
for external routing. Unlike normal Ipv4RoutingProtocol sub types,
Ipv4ClickRouting doesn't use a RouteInput() method, but instead,
receives a packet on the appropriate interface and processes it
accordingly. Note that you need to have a routing table type element
in your Click graph to use Click for external routing. This is needed
by the RouteOutput() function inherited from Ipv4RoutingProtocol.
Furthermore, a Click based node uses a different kind of L3 in the
form of Ipv4L3ClickProtocol, which is a trimmed down version of
Ipv4L3Protocol. Ipv4L3ClickProtocol passes on packets passing through
the stack to Ipv4ClickRouting for processing.


Developing a Simulator API to allow ns-3 to interact with Click
###############################################################

Much of the API is already well defined, which allows Click to probe for information from the simulator (like a Node's ID, an Interface ID and so forth). By retaining most of the methods, it should be possible to write new implementations specific to ns-3 for the same functionality.

Hence, for the Click integration with ns-3, a class named Ipv4ClickRouting will handle the interaction with Click. The code for the same can be found in ``src/click/model/ipv4-click-routing.{cc,h}``.

Packet hand off between ns-3 and Click
######################################

There are four kinds of packet hand-offs that can occur between ns-3 and Click. 

* L4 to L3
* L3 to L4
* L3 to L2
* L2 to L3

To overcome this, we implement Ipv4L3ClickProtocol, a stripped down version of Ipv4L3Protocol. Ipv4L3ClickProtocol passes packets to and from Ipv4ClickRouting appropriately to perform routing.

Scope and Limitations
=====================

* In its current state, the NS-3 Click Integration is limited to use only with L3, leaving NS-3 to handle L2. We are currently working on adding Click MAC support as well. See the usage section to make sure that you design your Click graphs accordingly.
* Furthermore, ns-3-click will work only with userlevel elements. The complete list of elements are available at http://read.cs.ucla.edu/click/elements. Elements that have 'all', 'userlevel' or 'ns' mentioned beside them may be used.
* As of now, the ns-3 interface to Click is Ipv4 only. We will be adding Ipv6 support in the future.

References
==========

* Eddie Kohler, Robert Morris, Benjie Chen, John Jannotti, and M. Frans Kaashoek. The click modular router. ACM Transactions on Computer Systems 18(3), August 2000, pages 263-297.
* Lalith Suresh P., and Ruben Merz. Ns-3-click: click modular router integration for ns-3. In Proc. of 3rd International ICST Workshop on NS-3 (WNS3), Barcelona, Spain. March, 2011.
* Michael Neufeld, Ashish Jain, and Dirk Grunwald. Nsclick: bridging network simulation and deployment. MSWiM '02: Proceedings of the 5th ACM international workshop on Modeling analysis and simulation of wireless and mobile systems, 2002, Atlanta, Georgia, USA. http://doi.acm.org/10.1145/570758.570772

Usage
*****

Building Click
==============

The first step is to clone Click from the github repository and build it::

  $ git clone https://github.com/kohler/click
  $ cd click/
  $ ./configure --disable-linuxmodule --enable-nsclick --enable-wifi
  $ make

The --enable-wifi flag may be skipped if you don't intend on using Click with Wifi.
* Note: You don't need to do a 'make install'. 

Once Click has been built successfully, change into the ns-3 directory and 
configure ns-3 with Click Integration support::

  $ ./ns3 configure --enable-examples --enable-tests --with-nsclick=/path/to/click/source

Hint:  If you have click installed one directory above ns-3 (such as in the
ns-3-allinone directory), and the name of the directory is 'click' (or
a symbolic link to the directory is named 'click'), then the --with-nsclick
specifier is not necessary; the ns-3 build system will successfully find
the directory.

If it says 'enabled' beside 'NS-3 Click Integration Support', then you're good to go. Note: If running modular ns-3, the minimum set of modules required to run all ns-3-click examples is wifi, csma and config-store.

Next, try running one of the examples::

  $ ./ns3 run nsclick-simple-lan

You may then view the resulting .pcap traces, which are named nsclick-simple-lan-0-0.pcap and nsclick-simple-lan-0-1.pcap.

Click Graph Instructions
========================

The following should be kept in mind when making your Click graph:

* Only userlevel elements can be used.
* You will need to replace FromDevice and ToDevice elements with FromSimDevice and ToSimDevice elements.
* Packets to the kernel are sent up using ToSimDevice(tap0,IP).
* For any node, the device which sends/receives packets to/from the kernel, is named 'tap0'. The remaining interfaces should be named eth0, eth1 and so forth (even if you're using wifi). Please note that the device numbering should begin from 0. In future, this will be made flexible so that users can name devices in their Click file as they wish.
* A routing table element is a mandatory. The OUTports of the routing table element should correspond to the interface number of the device through which the packet will ultimately be sent out. Violating this rule will lead to really weird packet traces. This routing table element's name should then be passed to the Ipv4ClickRouting protocol object as a simulation parameter. See the Click examples for details.
* The current implementation leaves Click with mainly L3 functionality, with ns-3 handling L2. We will soon begin working to support the use of MAC protocols on Click as well. This means that as of now, Click's Wifi specific elements cannot be used with ns-3.

Debugging Packet Flows from Click
=================================

From any point within a Click graph, you may use the Print (http://read.cs.ucla.edu/click/elements/print) element and its variants for pretty printing of packet contents. Furthermore, you may generate pcap traces of packets flowing through a Click graph by using the ToDump (http://read.cs.ucla.edu/click/elements/todump) element as well. For instance:

.. sourcecode:: cpp

  myarpquerier
   -> Print(fromarpquery,64)
   -> ToDump(out_arpquery,PER_NODE 1)
   -> ethout;

and ...will print the contents of packets that flow out of the ArpQuerier, then generate a pcap trace file which will have a suffix 'out_arpquery', for each node using the Click file, before pushing packets onto 'ethout'.

Helper
======

To have a node run Click, the easiest way would be to use the ClickInternetStackHelper
class in your simulation script. For instance:

.. sourcecode:: cpp

  ClickInternetStackHelper click;
  click.SetClickFile (myNodeContainer, "nsclick-simple-lan.click");
  click.SetRoutingTableElement (myNodeContainer, "u/rt");
  click.Install (myNodeContainer);

The example scripts inside ``src/click/examples/`` demonstrate the use of Click based nodes
in different scenarios. The helper source can be found inside ``src/click/helper/click-internet-stack-helper.{h,cc}``

Examples
========

The following examples have been written, which can be found in ``src/click/examples/``:

* nsclick-simple-lan.cc and nsclick-raw-wlan.cc: A Click based node communicating with a normal ns-3 node without Click, using Csma and Wifi respectively. It also demonstrates the use of TCP on top of Click, something which the original nsclick implementation for NS-2 couldn't achieve.

* nsclick-udp-client-server-csma.cc and nsclick-udp-client-server-wifi.cc: A 3 node LAN (Csma and Wifi respectively) wherein 2 Click based nodes run a UDP client, that sends packets to a third Click based node running a UDP server.

* nsclick-routing.cc: One Click based node communicates to another via a third node that acts as an IP router (using the IP router Click configuration). This demonstrates routing using Click. 

Scripts are available within ``<click-dir>/conf/`` that allow you to generate Click files for some common scenarios. The IP Router used in ``nsclick-routing.cc`` was generated from the make-ip-conf.pl file and slightly adapted to work with ns-3-click.

Validation
**********

This model has been tested as follows:

* Unit tests have been written to verify the internals of Ipv4ClickRouting. This can be found in ``src/click/ipv4-click-routing-test.cc``. These tests verify whether the methods inside Ipv4ClickRouting which deal with Device name to ID, IP Address from device name and Mac Address from device name bindings work as expected.
* The examples have been used to test Click with actual simulation scenarios. These can be found in ``src/click/examples/``. These tests cover the following: the use of different kinds of transports on top of Click, TCP/UDP, whether Click nodes can communicate with non-Click based nodes, whether Click nodes can communicate with each other, using Click to route packets using static routing.
* Click has been tested with Csma, Wifi and Point-to-Point devices. Usage instructions are available in the preceding section.
