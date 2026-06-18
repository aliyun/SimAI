/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Institute for the Wireless Internet of Things, Northeastern University, Boston, MA
 * Copyright (c) 2021, University of Washington: refactor for hierarchical model
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
 * Heavily edited by Tom Henderson to use HierarchicalMobilityModel
 */

/**
 * This example shows how to use the ns3::HierarchicalMobilityModel
 * to construct a Reference Point Group Mobility model (RPGM) model
 * as described in "A survey of mobility models for ad hoc network
 * research" by Tracy Camp, Jeff Boleng, and Vanessa Davies, Wireless
 * Communications and Mobile Computing, 2002: vol. 2, pp. 2483-502.
 * This example is closely related to the group mobility example
 * `src/mobility/group/reference-point-group-mobility-example.cc`
 * except that this version is buildings-aware.
 *
 * The HierarchicalMobilityModel is composed of two mobility models;
 * a parent and a child.  The position of the child is expressed
 * in reference to the position of the parent.  For group mobility,
 * each node in the group can install the same parent mobility model
 * and different child mobility models.  The parent mobility model
 * in this case, the "RandomWalk2dOutdoorMobilityModel", is
 * buildings-aware.  The child mobility models are not, but the
 * "Tolerance" attribute is defined on the parent model such that the
 * child mobility models do not result in movement through a building
 * wall.
 *
 * Standard ns-3 mobility model course change output is traced in
 * outdoor-group-mobility-course-change.mob.  Time series data of position
 * is traced in outdoor-group-mobility-time-series.mob.  This time series
 * data can be animated with a corresponding Bash script.  Another program
 * output is the list of buildings ("outdoor-group-mobility-buildings.txt").
 *
 * There is one program option:  'useHelper'.  This is simply for code
 * demonstration purposes; it selects the branch of code that is used
 * to either configure the mobility using a helper object, or
 * to directly configure using CreateObject<> () and handling of pointers.
 * The traces generated should be the same.
 *
 * Randomness in this program is due to the parent mobility model which
 * executes a random walk, while the child mobility models are set as
 * constant position offsets from the parent.  Slightly different motion  
 * can be obtained by changing the ns-3 'RunNumber' value (see the
 * documentation on ns-3 random variables).
 */

#include <iostream>
#include "ns3/core-module.h"
#include <ns3/mobility-module.h>
#include "ns3/network-module.h"
#include "ns3/buildings-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OutdoorGroupMobilityExample");

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

void
PrintGnuplottableBuildingListToFile (std::string filename)
{
  std::ofstream outFile;
  outFile.open (filename.c_str (), std::ios_base::out | std::ios_base::trunc);
  if (!outFile.is_open ())
    {
      NS_LOG_ERROR ("Can't open file " << filename);
      return;
    }
  uint32_t index = 1;
  for (BuildingList::Iterator it = BuildingList::Begin (); it != BuildingList::End (); ++it)
    {
      ++index;
      Box box = (*it)->GetBoundaries ();
      outFile << "set object " << index
              << " rect from " << box.xMin  << "," << box.yMin
              << " to "   << box.xMax  << "," << box.yMax
              << std::endl;
    }
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

  g_timeSeries.open ("outdoor-group-mobility-time-series.mob");
  
  NodeContainer n;
  n.Create (3);

  // The primary mobility model is the RandomWalk2dOutdoorMobilityModel
  // defined within this bounding box (along with four buildings, not shown):
  //
  // (0,50)                   (100,50)
  //   +-------------------------+
  //   |                         |
  //   |                         |
  //   |                         |
  //   |                         |
  //   |                         |
  //   +-------------------------+
  // (0,0)                     (100,0)
  //
  //
  // There are two buildings centered at (50,10) and (50,40), and two
  // additional small buildings centered at (20,25) and (80,25) that
  // create obstacles for the nodes to avoid.

  std::vector<Ptr<Building> > buildingVector;
  Ptr<Building> building = CreateObject<Building> ();
  building->SetBoundaries (Box (45, 55, 5, 15, 0, 10));
  buildingVector.push_back (building);
  building = CreateObject<Building> ();
  building->SetBoundaries (Box (45, 55, 35, 45, 0, 10));
  buildingVector.push_back (building);
  building = CreateObject<Building> ();
  building->SetBoundaries (Box (17.5, 22.5, 22.5, 27.5, 0, 10));
  buildingVector.push_back (building);
  building = CreateObject<Building> ();
  building->SetBoundaries (Box (77.5, 82.5, 22.5, 27.5, 0, 10));
  buildingVector.push_back (building);

  // print the list of buildings to file
  PrintGnuplottableBuildingListToFile ("outdoor-group-mobility-buildings.txt");

  // The program now branches into two:  one using the low-level API, and
  // one using the GroupMobilityHelper.  Both branches result in equivalent
  // configuration.

  int64_t streamIndex = 1;
  if (useHelper == false)
    {
      // The reference (parent) mobility model starts at coordinate (10, 10, 0)
      // and performs a buildings-aware random walk.
      //
      Ptr<RandomWalk2dOutdoorMobilityModel> outdoorMm = CreateObject<RandomWalk2dOutdoorMobilityModel> ();
      outdoorMm->SetAttribute ("Bounds", RectangleValue (Rectangle (0, 100, 0, 50)));
      // The tolerance value is used to prevent the child mobility models from
      // crossing building walls.  The child mobility models are defined as
      // offsets from the parent but are not buildings-aware, so it could be
      // the case that the parent mobility model was just outside of a wall
      // but the child mobility model was inside of a wall.  The tolerance
      // value (in meters) prevents the composite position from passing
      // through a building wall.
      outdoorMm->SetAttribute ("Tolerance", DoubleValue (2));
      // The initial position can be directly set
      outdoorMm->SetPosition (Vector (10, 10, 0));
      streamIndex += outdoorMm->AssignStreams (streamIndex);

      // Each HierachicalMobilityModel contains the above model as the Parent,
      // and a user defined model as the Child.  Two MobilityModel objects are
      // instantiated per node, and each node also shares the same parent model.

      // Mobility model for the first node (node 0)
      Ptr<HierarchicalMobilityModel> hierarchical0 = CreateObject<HierarchicalMobilityModel> ();
      hierarchical0->SetParent (outdoorMm);

      // Child model for the first node (node 0)
      Ptr<ConstantPositionMobilityModel> child0 = CreateObject<ConstantPositionMobilityModel> ();
      child0->SetPosition (Vector (1, 0, 0)); // 1 m offset from parent position
      // There is no need to AssignStreams() to child0 because the position
      // is constant and deterministic
      hierarchical0->SetChild (child0);
      n.Get (0)->AggregateObject (hierarchical0);

      // Repeat for other two nodes
      Ptr<HierarchicalMobilityModel> hierarchical1 = CreateObject<HierarchicalMobilityModel> ();
      hierarchical1->SetParent (outdoorMm); // Same parent as before
      Ptr<ConstantPositionMobilityModel> child1 = CreateObject<ConstantPositionMobilityModel> ();
      child1->SetPosition (Vector (-1, 0, 0));
      hierarchical1->SetChild (child1);
      n.Get (1)->AggregateObject (hierarchical1);
      Ptr<HierarchicalMobilityModel> hierarchical2 = CreateObject<HierarchicalMobilityModel> ();
      hierarchical2->SetParent (outdoorMm); // Same parent as before
      Ptr<ConstantPositionMobilityModel> child2 = CreateObject<ConstantPositionMobilityModel> ();
      child2->SetPosition (Vector (0, 1, 0));
      hierarchical2->SetChild (child2);
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
      group.SetReferenceMobilityModel ("ns3::RandomWalk2dOutdoorMobilityModel", 
        "Bounds", RectangleValue (Rectangle (0, 100, 0, 50)),
        "Tolerance", DoubleValue (2));
      Ptr<ListPositionAllocator> listPosition = CreateObject<ListPositionAllocator> ();
      listPosition->Add (Vector (10, 10, 0));
      group.SetReferencePositionAllocator (listPosition);
    
      group.SetMemberMobilityModel ("ns3::ConstantPositionMobilityModel");
      listPosition = CreateObject<ListPositionAllocator> ();
      listPosition->Add (Vector (1, 0, 0)); // first child
      listPosition->Add (Vector (-1, 0, 0)); // second child
      listPosition->Add (Vector (0, 1, 0)); // second child
      group.SetMemberPositionAllocator (listPosition);
     
      // Install to all three nodes
      group.Install (n);

      // After installation, use the helper to make the equivalent
      // stream assignments as above
      group.AssignStreams (n, streamIndex);
    }

  AsciiTraceHelper ascii;
  MobilityHelper::EnableAsciiAll (ascii.CreateFileStream ("outdoor-group-mobility-course-change.mob"));

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
