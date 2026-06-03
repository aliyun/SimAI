.. include:: replace.txt
.. toctree::

Animation
---------

Animation is an important tool for network simulation. While |ns3| does not
contain a default graphical animation tool, we currently have two ways to provide
animation, namely using the PyViz method or the NetAnim method.
The PyViz method is described in http://www.nsnam.org/wiki/PyViz.

We will describe the NetAnim method briefly here.

NetAnim
*******

NetAnim is a standalone, Qt4-based software executable that uses a trace file generated during 
an |ns3| simulation to display the topology and animate the packet flow between nodes.

.. figure:: figures/NetAnim_3_105.*
   :align: center
   :width: 500px
   :height: 400px

   An example of packet animation on wired-links

In addition, NetAnim also provides useful features such as tables to display meta-data of packets like the image below

.. figure:: figures/PacketStatistics.*
   :align: center
   :width: 500px

   An example of tables for packet meta-data with protocol filters

A way to visualize the trajectory of a mobile node

.. figure:: figures/Trajectory.*
   :align: center
   :width: 500px

   An example of the trajectory of a mobile node

A way to display the routing-tables of multiple nodes at various points in time

.. figure:: figures/RoutingTables.*
   :align: center
   :width: 500px

A way to display counters associated with multiple nodes as a chart or a table

.. figure:: figures/NodeCountersChart.*
   :align: center
   :width: 500px

.. figure:: figures/NodeCountersTable.*
   :align: center
   :width: 500px

A way to view the timeline of packet transmit and receive events

.. figure:: figures/PacketTimeline.*
   :align: center
   :width: 500px

Methodology
===========
The class ns3::AnimationInterface is responsible for the creation the trace XML file.
AnimationInterface uses the tracing infrastructure to track packet flows between nodes.
AnimationInterface registers itself as a trace hook for tx and rx events before the simulation
begins. When a packet is scheduled for transmission or reception, the corresponding tx and rx 
trace hooks in AnimationInterface are called. When the rx hooks are called, AnimationInterface will be aware of the two endpoints between which a packet has flowed, and adds this information
to the trace file, in XML format along with the corresponding tx and rx timestamps. The XML format 
will be discussed in a later section. It is important to note that AnimationInterface records a 
packet only if the rx trace hooks are called. Every tx event must be matched by an rx event.

Downloading NetAnim
===================
If NetAnim is not already available in the |ns3| package you downloaded, you can do the following:

Please ensure that you have installed mercurial.
The latest version of NetAnim can be downloaded using mercurial with the following command:

.. sourcecode:: bash

  $ hg clone http://code.nsnam.org/netanim

Building NetAnim
================
Prerequisites
~~~~~~~~~~~~~
Qt5 (5.4 and over) is required to build NetAnim. This can be obtained using the following ways:

For Ubuntu Linux distributions:

.. sourcecode:: bash

  $ apt-get install qt5-default

For Red Hat/Fedora based distribution:

.. sourcecode:: bash

  $ yum install qt5
  $ yum install qt5-devel

For Mac/OSX, see http://qt.nokia.com/downloads/

Build steps
~~~~~~~~~~~

To build NetAnim use the following commands:

.. sourcecode:: bash

  $ cd netanim
  $ make clean
  $ qmake NetAnim.pro
  $ make


Note: qmake could be "qmake-qt5" in some systems

This should create an executable named "NetAnim" in the same directory:

.. sourcecode:: bash

  $ ls -l NetAnim
 -rwxr-xr-x 1 john john 390395 2012-05-22 08:32 NetAnim

Usage
=====
Using NetAnim is a two-step process

Step 1:Generate the animation XML trace file during simulation using "ns3::AnimationInterface" in the |ns3| code base.

Step 2:Load the XML trace file generated in Step 1 with the offline Qt4-based animator named NetAnim.

Step 1: Generate XML animation trace file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The class "AnimationInterface" under "src/netanim" uses underlying |ns3| trace sources 
to construct a timestamped ASCII file in XML format.

Examples are found under src/netanim/examples
Example:

.. sourcecode:: bash

  $ ./ns3 configure -d debug --enable-examples
  $ ./ns3 run "dumbbell-animation"

The above will create an XML file dumbbell-animation.xml

Mandatory
#########

1. Ensure that your program's CMakeLists.txt includes the "netanim" module. An example of such a CMakeLists.txt is at src/netanim/examples/CMakeLists.txt.
2. Include the header [#include "ns3/netanim-module.h"] in your test program
3. Add the statement

.. sourcecode:: cpp

  AnimationInterface anim ("animation.xml");  // where "animation.xml" is any arbitrary filename

[for versions before ns-3.13 you also have to use the line "anim.SetXMLOutput() to set the XML mode and also use anim.StartAnimation();]


Optional
########

.. highlight:: cpp

The following are optional but useful steps::

  // Step 1
  anim.SetMobilityPollInterval (Seconds (1));

AnimationInterface records the position of all nodes every 250 ms by default. The statement above sets 
the periodic interval at which AnimationInterface records the position of all nodes. If the nodes are 
expected to move very little, it is useful to set a high mobility poll interval to avoid large XML files.

::

  // Step 2
  anim.SetConstantPosition (Ptr< Node > n, double x, double y);

AnimationInterface requires that the position of all nodes be set. In |ns3| this is done by setting an associated MobilityModel. "SetConstantPosition" is a quick way to set the x-y coordinates of a node which is stationary.

::

  // Step 3
  anim.SetStartTime (Seconds(150)); and anim.SetStopTime (Seconds(150));

AnimationInterface can generate large XML files. The above statements restricts the window between which AnimationInterface does tracing. Restricting the window serves to focus only on relevant portions of the simulation and creating manageably small XML files

::

  // Step 4
  AnimationInterface anim ("animation.xml", 50000);

Using the above constructor ensures that each animation XML trace file has only 50000 packets. For example, if AnimationInterface captures 150000 packets, using the above constructor splits the capture into 3 files

* animation.xml - containing the packet range 1-50000

* animation.xml-1 - containing the packet range 50001-100000 

* animation.xml-2 - containing the packet range 100001-150000 

::

  // Step 5
  anim.EnablePacketMetadata (true);

With the above statement, AnimationInterface records the meta-data of each packet in the xml trace file. Metadata can be used by NetAnim to provide better statistics and filter, along with providing some brief information about the packet such as TCP sequence number or source & destination IP address during packet animation.

CAUTION: Enabling this feature will result in larger XML trace files.
Please do NOT enable this feature when using Wimax links.

::

  // Step 6
  anim.UpdateNodeDescription (5, "Access-point");

With the above statement, AnimationInterface assigns the text "Access-point" to node 5.

::

  // Step 7
  anim.UpdateNodeSize (6, 1.5, 1.5);

With the above statement, AnimationInterface sets the node size to scale by 1.5. NetAnim automatically scales the graphics view to fit the oboundaries of the topology. This means that NetAnim, can abnormally scale a node's size too high or too low. Using AnimationInterface::UpdateNodeSize allows you to overwrite the default scaling in NetAnim and use your own custom scale.

:: 

  // Step 8
  anim.UpdateNodeCounter (89, 7, 3.4);

With the above statement, AnimationInterface sets the counter with Id == 89, associated with Node 7 with the value 3.4.
The counter with Id 89 is obtained using AnimationInterface::AddNodeCounter. An example usage for this is in src/netanim/examples/resource-counters.cc.


Step 2: Loading the XML in NetAnim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Assuming NetAnim was built, use the command "./NetAnim" to launch NetAnim. Please review the section "Building NetAnim" if NetAnim is not available.
2. When NetAnim is opened, click on the File open button at the top-left corner, select the XML file generated during Step 1.
3. Hit the green play button to begin animation.

Here is a video illustrating this
http://www.youtube.com/watch?v=tz_hUuNwFDs

Wiki
====
For detailed instructions on installing "NetAnim", F.A.Qs and loading the XML trace file 
(mentioned earlier) using NetAnim please refer:
http://www.nsnam.org/wiki/NetAnim
