.. include:: replace.txt
.. highlight:: cpp
 
Point to Point Dumbbell Topology Helper
---------------------------------------

This is an introduction to dumbbell topology helper to complement the
``PointToPointDumbbellHelper`` doxygen.

Model Description
*****************

The |ns3| :cpp:class:`PointToPointDumbbellHelper` class is used to create nodes, add
point-to-point connections between them and position them in a dumbbell type
layout. It provides several functions that give complete control over the node
creation, IPv4/IPv6 address assignment and node positioning (defining bounding
box).

It accepts the number of leaf nodes that are to be created on the left, number
of leaf nodes to be created on the right, and the ``PointToPointHelper``
objects for the left side, right side and router nodes. These
``PointToPointHelper`` objects define Device and Channel attributes for the
nodes in the respective ``NodeContainer`` for left, right and router nodes. It
then connects all the left leaf nodes to the left router and all the right
leaf nodes to the right router. Thus, all the left leaf nodes are connected
to the left router, which is connected to the right router, and the right
router is connected to all the right leaf nodes.

It provides a feature to position the nodes in a dumbbell type layout to help
visualize the topology. This is achieved through a function called
``BoundingBox`` which requires coordinate limits as parameters, and then
positions all the nodes within the limits specified to it in a dumbbell layout
which is described later in this documentation.

It also provides a function ``InstallStack`` which takes a
``InternetStackHelper`` object for installing the ``InternetStack`` on the
nodes, function ``AssignIpv4Addresses`` which takes ``Ipv4AddressHelper``
objects for the left, right and router nodes for assigning IPv4 address to
each node. It provides a function ``AssignIpv6Addresses`` which takes a
``Ipv6Address`` address base and a ``Ipv6Prefix`` prefix for assigning
IPv6Adddress to every node.

Node Positioning
================

The algorithm used by the ``BoundingBox`` function to position the leaf nodes
and router nodes uses properties of a circle to correctly place the nodes even
when the user specified coordinate constraints are not enough to place nodes
accurately.

It accepts the minimum upper left X-axis coordinate, minimum upper left Y-axis
coordinate, maximum lower right X-axis coordinate and maximum lower right Y-
axis coordinate. These limits are used by the algorithm to position the nodes
in a dumbbell type layout. It automatically adjusts the nodes according to the
parameters passed to it, and if the parameters specified are too less than
required, it positions them in a straight line.

This feature is useful if the user wants to visualize the topology using any
supported visualization tool such as PyViz or NetAnim.

Using PointToPointDumbbellHelper
================================

A dumbbell topology with point-to-point links can be configured by using the
``PointToPointDumbbellHelper``. The first task is to create two
``PointToPointHelper`` objects: one for the leaf nodes and the other for the
router nodes as shown below::

  PointToPointHelper p2pLeaf;
  p2pLeaf.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  p2pLeaf.SetChannelAttribute ("Delay", StringValue ("2ms"));

  PointToPointHelper p2pRouter;
  p2pRouter.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  p2pRouter.SetChannelAttribute ("Delay", StringValue ("10ms"));

After creating the ``PointToPointHelper`` objects, we need to instantiate the
``PointToPointDumbbellHelper``::

  PointToPointDumbbellHelper dumbbell (nLeftLeaf, p2pLeaf, nRightLeaf,
                                       p2pLeaf, p2pRouter);

It is also possible to create two separate ``PointToPointHelper`` objects for
left leaf nodes and right leaf nodes. In this case, a total of three
``PointToPointHelper`` objects should be created: one for the left leaf nodes,
one for right leaf nodes and one for the router nodes as shown below::

  PointToPointHelper p2pLeft;
  p2pLeft.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  p2pLeft.SetChannelAttribute ("Delay", StringValue ("2ms"));

  PointToPointHelper p2pRight;
  p2pRight.SetDeviceAttribute ("DataRate", StringValue ("15Mbps"));
  p2pRight.SetChannelAttribute ("Delay", StringValue ("5ms"));

  PointToPointHelper p2pRouter;
  p2pRouter.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  p2pRouter.SetChannelAttribute ("Delay", StringValue ("10ms"));

In this case, we instantiate the ``PointToPointDumbbellHelper`` as shown below::

  PointToPointDumbbellHelper dumbbell (nLeftLeaf, p2pLeft, nRightLeaf,
                                       p2pRight, p2pRouter);

After this, we can tell the helper to position the nodes in a dumbbell layout.
This is done to improve the visualization of the topology and can be achieved
by calling the ``BoundingBox`` function::

  // ulx -> Minimum Upper Left X coordinate
  // uly -> Minimum Upper Left Y coordinate
  // lrx -> Maximum Lower Right X coordinate
  // lry -> Maximum Lower Right Y coordinate
  dumbbell.BoundingBox (ulx, uly, lrx, lry);

Next, we need to install an Internet Stack on the nodes. We have to
instantiate a ``InternetStackHelper`` object for this, and pass it as a
parameter to ``InstallStack`` function::

  InternetStackHelper stack;
  dumbbell.InstallStack (stack);

Subsequently, we have to assign a IPv4/IPv6 address to the nodes as shown
below::

  // IPv4
  Ipv4AddressHelper leftIp, rightIp, routerIp;
  leftIp.SetBase ("10.1.1.0", "255.255.255.0");
  rightIp.SetBase ("10.2.1.0", "255.255.255.0");
  routerIp.SetBase ("10.3.1.0", "255.255.255.0");

  dumbbell.AssignIpv4Addresses (leftIp, rightIp, routerIp);

For IPv6 address::

  // IPv6
  Ipv6Address addrBase ("2001:1::");
  Ipv6Prefix prefix (64)

  dumbbell.AssignIpv6Addresses (addrBase, prefix);

Example
*******

The example for this helper is `dumbbell-animation.cc` located in
``src/netanim/examples``. The following command shows the available command-
line options for this example::

   $ ./ns3 run "dumbbell-animation --PrintHelp"

The following command sets up a dumbbell topology with the default
configuration::

   $ ./ns3 run dumbbell-animation

The following command sets up a dumbbell topology with 6 left leaf nodes and 9
right leaf nodes::

   $ ./ns3 run "dumbbell-animation --nLeftLeaf=6 --nRightLeaf=9"

The following command sets up a dumbbell topology with 10 left leaf nodes and
10 right leaf nodes::

   $ ./ns3 run "dumbbell-animation --nLeaf=10"

The expected output from the previous commands is a XML file to playback the
animation using NetAnim.
