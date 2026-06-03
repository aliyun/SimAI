.. include:: replace.txt


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  User Documentation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

How to use buildings in a simulation
====================================

In this section we explain the basic usage of the buildings model within a
simulation program.

Include the headers
*******************

Add this at the beginning of your simulation program::

   #include <ns3/buildings-module.h>


Create a building
*****************

As an example, let's create a residential 10 x 20 x 10 building::

    double x_min = 0.0;
    double x_max = 10.0;
    double y_min = 0.0;
    double y_max = 20.0;
    double z_min = 0.0;
    double z_max = 10.0;
    Ptr<Building> b = CreateObject <Building> ();
    b->SetBoundaries (Box (x_min, x_max, y_min, y_max, z_min, z_max));
    b->SetBuildingType (Building::Residential);
    b->SetExtWallsType (Building::ConcreteWithWindows);
    b->SetNFloors (3);
    b->SetNRoomsX (3);
    b->SetNRoomsY (2);

This building has three floors and an internal 3 x 2  grid of rooms of equal size.

The helper class GridBuildingAllocator is also available to easily
create a set of buildings with identical characteristics placed on a
rectangular grid. Here's an example of how to use it::

  Ptr<GridBuildingAllocator>  gridBuildingAllocator;
  gridBuildingAllocator = CreateObject<GridBuildingAllocator> ();
  gridBuildingAllocator->SetAttribute ("GridWidth", UintegerValue (3));
  gridBuildingAllocator->SetAttribute ("LengthX", DoubleValue (7));
  gridBuildingAllocator->SetAttribute ("LengthY", DoubleValue (13));
  gridBuildingAllocator->SetAttribute ("DeltaX", DoubleValue (3));
  gridBuildingAllocator->SetAttribute ("DeltaY", DoubleValue (3));
  gridBuildingAllocator->SetAttribute ("Height", DoubleValue (6));
  gridBuildingAllocator->SetBuildingAttribute ("NRoomsX", UintegerValue (2));
  gridBuildingAllocator->SetBuildingAttribute ("NRoomsY", UintegerValue (4));
  gridBuildingAllocator->SetBuildingAttribute ("NFloors", UintegerValue (2));
  gridBuildingAllocator->SetAttribute ("MinX", DoubleValue (0));
  gridBuildingAllocator->SetAttribute ("MinY", DoubleValue (0));
  gridBuildingAllocator->Create (6);


This will create a 3x2 grid of 6 buildings, each 7 x 13 x 6 m with 2 x
4 rooms inside and 2 foors; the buildings are spaced by 3 m on both
the x and the y axis.


Setup nodes and mobility models
*******************************

Nodes and mobility models are configured as usual, however in order to
use them with the buildings model you need an additional call to
``BuildingsHelper::Install()``, so as to let the mobility model include
the information on their position w.r.t. the buildings. Here is an example::

    MobilityHelper mobility;
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    ueNodes.Create (2);
    mobility.Install (ueNodes);
    BuildingsHelper::Install (ueNodes);

It is to be noted that any mobility model can be used. However, the
user is advised to make sure that the behavior of the mobility model
being used is consistent with the presence of Buildings. For example,
using a simple random mobility over the whole simulation area in
presence of buildings might easily results in node moving in and out
of buildings, regardless of the presence of walls.

One dedicated buildings-aware mobility model is the
``RandomWalk2dOutdoorMobilityModel``.  This class is similar to the
``RandomWalk2dMobilityModel`` but avoids placing the trajectory
on a path that would intersect a building wall.  If a boundary
is encountered (either the bounding box or a building wall), the
model rebounds with a random direction and speed that ensures that
the trajectory stays outside the buildings.  An example program
that demonstrates the use of this model is the
``src/buildings/examples/outdoor-random-walk-example.cc`` which
has an associated shell script to plot the traces generated.
Another example program demonstrates how this outdoor mobility
model can be used as the basis of a group mobility model, with
the outdoor buildings-aware model serving as the parent or
reference mobility model, and with additional nodes defining a
child mobility model providing the offset from the reference
mobility model.  This example,
``src/buildings/example/outdoor-group-mobility-example.cc``,
also has an associated shell script
(``outdoor-group-mobility-animate.sh``) that can be used to generate
an animated GIF of the group's movement.

Place some nodes
****************

You can place nodes in your simulation using several methods, which
are described in the following.

Legacy positioning methods
--------------------------

Any legacy ns-3 positioning method can be used to place node in the
simulation. The important additional step is to For example, you can
place nodes manually like this::

    Ptr<ConstantPositionMobilityModel> mm0 = enbNodes.Get (0)->GetObject<ConstantPositionMobilityModel> ();
    Ptr<ConstantPositionMobilityModel> mm1 = enbNodes.Get (1)->GetObject<ConstantPositionMobilityModel> ();
    mm0->SetPosition (Vector (5.0, 5.0, 1.5));
    mm1->SetPosition (Vector (30.0, 40.0, 1.5));

    MobilityHelper mobility;
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    ueNodes.Create (2);
    mobility.Install (ueNodes);
    BuildingsHelper::Install (ueNodes);
    mm0->SetPosition (Vector (5.0, 5.0, 1.5));
    mm1->SetPosition (Vector (30.0, 40.0, 1.5));

Alternatively, you could use any existing PositionAllocator
class. The coordinates of the node will determine whether it is placed
outdoor or indoor and, if indoor, in which building and room it is placed.


Building-specific positioning methods
-------------------------------------

The following position allocator classes are available to place node
in special positions with respect to buildings:

 - ``RandomBuildingPositionAllocator``: Allocate each position by
   randomly choosing a building from the list of all buildings, and
   then randomly choosing a position inside the building.

 - ``RandomRoomPositionAllocator``: Allocate each position by randomly
   choosing a room from the list of rooms in all buildings, and then
   randomly choosing a position inside the room.

 - ``SameRoomPositionAllocator``: Walks a given NodeContainer
   sequentially, and for each node allocate a new position randomly in
   the same room of that node.

 - ``FixedRoomPositionAllocator``: Generate a random position
   uniformly distributed in the volume of a chosen room inside a
   chosen building.



Making the Mobility Model Consistent for a node
***********************************************

Initially, a mobility model of a node is made consistent when a node is
initialized, which eventually triggers a call to the ``DoInitialize``
method of the `MobilityBuildingInfo`` class. In particular, it calls the
``MakeMobilityModelConsistent`` method, which goes through the lists of
all buildings, determine if the node is indoor or outdoor, and if indoor
it also determines the building in which the node is located and the
corresponding floor number inside the building. Moreover, this method also
caches the position of the node, which is used to make the mobility model
consistent for a moving node whenever the ``IsInside`` method of
``MobilityBuildingInfo`` class is called.



Building-aware pathloss model
*****************************

After you placed buildings and nodes in a simulation, you can use a
building-aware pathloss model in a simulation exactly in the same way
you would use any regular path loss model. How to do this is specific
for the wireless module that you are considering (lte, wifi, wimax,
etc.), so please refer to the documentation of that model for specific
instructions.

Building-aware channel condition models
***************************************

The class BuildingsChannelConditionModel implements a `channel condition model <propagation.html#channelconditionmodel>`_
which determines the LOS/NLOS channel state based on the buildings deployed in
the scenario.

The classes ``ThreeGppV2vUrbanChannelConditionModel`` and 
``ThreeGppV2vHighwayChannelConditionModel`` implement hybrid channel condition 
models, specifically designed to model vehicular environments. 
More information can be found in the :ref:`documentation 
of the propagation module <sec-3gpp-v2v-ch-cond>`.

Main configurable attributes
============================

The ``Building`` class has the following configurable parameters:

* building type: Residential, Office and Commercial.
* external walls type: Wood, ConcreteWithWindows, ConcreteWithoutWindows and StoneBlocks.
* building bounds: a ``Box`` class with the building bounds.
* number of floors.
* number of rooms in x-axis and y-axis (rooms can be placed only in a grid way).

The ``BuildingMobilityLossModel`` parameter configurable with the ns3 attribute system is represented by the bound (string ``Bounds``) of the simulation area by providing a ``Box`` class with the area bounds. Moreover, by means of its methods the following parameters can be configured:

* the number of floor the node is placed (default 0).
* the position in the rooms grid.


The ``BuildingPropagationLossModel`` class has the following configurable parameters configurable with the attribute system:

* ``Frequency``: reference frequency (default 2160 MHz), note that by setting the frequency the wavelength is set accordingly automatically and viceversa).
* ``Lambda``: the wavelength (0.139 meters, considering the above frequency).
* ``ShadowSigmaOutdoor``: the standard deviation of the shadowing for outdoor nodes (defaul 7.0).
* ``ShadowSigmaIndoor``: the standard deviation of the shadowing for indoor nodes (default 8.0).
* ``ShadowSigmaExtWalls``: the standard deviation of the shadowing due to external walls penetration for outdoor to indoor communications (default 5.0).
* ``RooftopLevel``: the level of the rooftop of the building in meters (default 20 meters).
* ``Los2NlosThr``: the value of distance of the switching point between line-of-sigth and non-line-of-sight propagation model in meters (default 200 meters).
* ``ITU1411DistanceThr``: the value of distance of the switching point between short range (ITU 1211) communications and long range (Okumura Hata) in meters (default 200 meters).
* ``MinDistance``: the minimum distance in meters between two nodes for evaluating the pathloss (considered neglictible before this threshold) (default 0.5 meters).
* ``Environment``: the environment scenario among Urban, SubUrban and OpenAreas (default Urban).
* ``CitySize``: the dimension of the city among Small, Medium, Large (default Large).

In order to use the hybrid mode, the class to be used is the ``HybridBuildingMobilityLossModel``, which allows the selection of the proper pathloss model according to the pathloss logic presented in the design chapter. However, this solution has the problem that the pathloss model switching points might present discontinuities due to the different characteristics of the model. This implies that according to the specific scenario, the threshold used for switching have to be properly tuned.
The simple ``OhBuildingMobilityLossModel`` overcome this problem by using only the Okumura Hata model and the wall penetration losses.
