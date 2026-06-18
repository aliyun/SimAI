/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Institute for the Wireless Internet of Things, Northeastern University, Boston, MA
 * Copyright (c) 2021 University of Washington: for HierarchicalMobilityModel
 *
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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Michele Polese <michele.polese@gmail.com>
 * Heavily edited by Tom Henderson (to reuse HierachicalMobilityModel)
 */

/**
 * This example shows how to use the ns3::HierarchicalMobilityModel
 * to construct a Reference Point Group Mobility model (RPGM) model
 * as described in "A survey of mobility models for ad hoc network
 * research" by Tracy Camp, Jeff Boleng, and Vanessa Davies, Wireless
 * Communications and Mobile Computing, 2002: vol. 2, pp. 2483-502.
 *
 * The HierarchicalMobilityModel is composed of two mobility models;
 * a parent and a child.  The position of the child is expressed
 * in reference to the position of the parent.  For group mobility,
 * each node in the group installs the same parent mobility model
 * and different child mobility models.
 *
 * There is no node associated with the parent (reference) model.
 * Instead, all nodes are associated with a hiearchical mobility model
 * containing both the parent and child models, and the position of
 * the node is the vector sum of these parent and child positions.
 *
 * Standard ns-3 mobility model course change output is traced in
 * 'reference-point-course-change.mob' file.  This file only traces
 * position when there is a course change.  A second trace is produced
 * by this example: a time-series of node positions sampled every second.
 * This file is 'reference-point-time-series.mob' and can be plotted
 * with the 'reference-point-group-mobility-animation.sh' program.
 *
 * There is a bit of randomness in the child mobility models (random
 * walk within a 10m x 10m box surrounding the parent mobility position);
 * slightly different output can be rendered by changing the ns-3 'RunNumber'
 * value (see the documentation on ns-3 random variables).
 *
 * There is one program option:  'useHelper'.  This is simply for code
 * demonstration purposes; it selects the branch of code that is used
 * to either configure the mobility using a helper object, or
 * to directly configure using CreateObject<> () and handling of pointers.
 * The traces generated should be the same.
 */

#include <iostream>
#include "ns3/core-module.h"
#include <ns3/mobility-module.h>
#include "ns3/network-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ReferencePointGroupMobilityExample");

std::ofstream g_timeSeries;

void
PrintPosition (Ptr<Node> node)
{
  if (node == nullptr) return;
  Ptr<MobilityModel> model = node->GetObject<MobilityModel> ();
  if (model == nullptr) return;
  NS_LOG_LOGIC ("Node: " << node->GetId () << " Position: " << model->GetPosition ());
  g_timeSeries << Simulator::Now ().GetSeconds () << " " << node->GetId () << " " << model->GetPosition () << std::endl;
  
}

int
main (int argc, char *argv[])
{
  Time simTime = Seconds (800);
  uint32_t numPrints = 800;
  bool useHelper = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("useHelper", "Whether to use helper code", useHelper);
  cmd.Parse (argc, argv);

  g_timeSeries.open ("reference-point-time-series.mob");
  
  NodeContainer n;
  n.Create (3);

  // The primary mobility model is the WaypointMobilityModel defined within
  // this bounding box:
  //
  // (0,50)                   (100,50)
  //   +-------------------------+
  //   |  .(10,40)    (90,40).   |
  //   |                         |
  //   |                         |
  //   |  .(10,10)    (90,10).   |
  //   |                         |
  //   +-------------------------+
  // (0,0)                     (100,0)
  //

  // The reference (parent) mobility model starts at coordinate (10,10
  // and walks clockwise to each waypoint, making two laps.  The time
  // to travel between each waypoint is 100s, so the velocity alternates
  // between two values due to the rectangular path. 
  // No actual node is represented by the position of this mobility
  // model; it forms the reference point from which the node's child
  // mobility model position is offset.
  //
  Ptr<WaypointMobilityModel> waypointMm = CreateObject<WaypointMobilityModel> ();
  waypointMm->AddWaypoint (Waypoint (Seconds (0), Vector (10, 10, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (100), Vector (10, 40, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (200), Vector (90, 40, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (300), Vector (90, 10, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (400), Vector (10, 10, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (500), Vector (10, 40, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (600), Vector (90, 40, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (700), Vector (90, 10, 0)));
  waypointMm->AddWaypoint (Waypoint (Seconds (800), Vector (10, 10, 0)));

  // Each HierachicalMobilityModel contains the above model as the Parent,
  // and a user defined model as the Child.  Two MobilityModel objects are
  // instantiated per node (one hierarchical, and one child model), and
  // a single parent model is reused across all nodes.

  // The program now branches into two:  one using the low-level API, and
  // one using the GroupMobilityHelper.  Both branches result in equivalent
  // configuration.

  int64_t streamIndex = 1;
  if (useHelper == false)
    {
      // Assign random variable stream numbers on the parent and each child
      streamIndex += waypointMm->AssignStreams (streamIndex);

      // Mobility model for the first node (node 0)
      Ptr<HierarchicalMobilityModel> hierarchical0 = CreateObject<HierarchicalMobilityModel> ();
      hierarchical0->SetParent (waypointMm);
    
      // Child Mobility model for the first node (node 0).  This can be any
      // other mobility model type; for this example, we reuse the random walk
      // but with a small 10m x 10m bounding box.
      Ptr<RandomWalk2dMobilityModel> childRandomWalk0 = CreateObject<RandomWalk2dMobilityModel> ();
      // Position in reference to the original random walk
      childRandomWalk0->SetAttribute ("Bounds", RectangleValue (Rectangle (-5, 5, -5, 5)));
      childRandomWalk0->SetAttribute ("Speed", StringValue ("ns3::ConstantRandomVariable[Constant=0.1]"));
      streamIndex += childRandomWalk0->AssignStreams (streamIndex);
      hierarchical0->SetChild (childRandomWalk0);
      n.Get (0)->AggregateObject (hierarchical0);
      // Repeat for other two nodes
      Ptr<HierarchicalMobilityModel> hierarchical1 = CreateObject<HierarchicalMobilityModel> ();
      hierarchical1->SetParent (waypointMm); // Same parent as before
      Ptr<RandomWalk2dMobilityModel> childRandomWalk1 = CreateObject<RandomWalk2dMobilityModel> ();
      childRandomWalk1->SetAttribute ("Bounds", RectangleValue (Rectangle (-5, 5, -5, 5)));
      childRandomWalk1->SetAttribute ("Speed", StringValue ("ns3::ConstantRandomVariable[Constant=0.1]"));
      streamIndex += childRandomWalk1->AssignStreams (streamIndex);
      hierarchical1->SetChild (childRandomWalk1);
      n.Get (1)->AggregateObject (hierarchical1);
      Ptr<HierarchicalMobilityModel> hierarchical2 = CreateObject<HierarchicalMobilityModel> ();
      hierarchical2->SetParent (waypointMm); // Same parent as before
      Ptr<RandomWalk2dMobilityModel> childRandomWalk2 = CreateObject<RandomWalk2dMobilityModel> ();
      childRandomWalk2->SetAttribute ("Bounds", RectangleValue (Rectangle (-5, 5, -5, 5)));
      childRandomWalk2->SetAttribute ("Speed", StringValue ("ns3::ConstantRandomVariable[Constant=0.1]"));
      streamIndex += childRandomWalk2->AssignStreams (streamIndex);
      hierarchical2->SetChild (childRandomWalk2);
      n.Get (2)->AggregateObject (hierarchical2);
    }
  else
    {
      // This branch demonstrates an equivalent set of commands but using
      // the GroupMobilityHelper
      GroupMobilityHelper group;

      // The helper provides a method to set the reference mobility model
      // for construction by an object factory, but in this case, since we
      // are using the WaypointMobilityModel, which requires us to add
      // waypoints directly on the object, we will just pass in the pointer.
      group.SetReferenceMobilityModel (waypointMm);

      // The WaypointMobilityModel does not need a position allocator
      // (it can use its first waypoint as such), but in general, the
      // GroupMobilityHelper can be configured to accept configuration for
      // a PositionAllocator for the reference model.  We skip that here.
  
      // Next, configure the member mobility model
      group.SetMemberMobilityModel ("ns3::RandomWalk2dMobilityModel",
        "Bounds", RectangleValue (Rectangle (-5, 5, -5, 5)),
        "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=0.1]"));

      // Again, we could call 'SetMemberPositionAllocator' and provide a
      // position allocator here for the member nodes, but none is provided
      // in this example, so they will start at time zero with the same
      // position as the reference node.

      // Install to all three nodes
      group.Install (n);

      // After installation, use the helper to make the equivalent
      // stream assignments as above 
      group.AssignStreams (n, streamIndex);
    }

  // Note:  The tracing methods are static methods declared on the
  // MobilityHelper class, not on the GroupMobilityHelper class
  AsciiTraceHelper ascii;
  MobilityHelper::EnableAsciiAll (ascii.CreateFileStream ("reference-point-course-change.mob"));

  // Use a logging PrintPosition() to record time-series position
  for (unsigned int i = 0; i < numPrints; i++)
    {
      for (auto nodeIt = n.Begin (); nodeIt != n.End (); ++nodeIt)
        {
          Simulator::Schedule (NanoSeconds (i * simTime.GetNanoSeconds () / numPrints), &PrintPosition, (*nodeIt));
        }
    }

  Simulator::Stop (simTime);
  Simulator::Run ();
  g_timeSeries.close ();
  Simulator::Destroy ();
}
