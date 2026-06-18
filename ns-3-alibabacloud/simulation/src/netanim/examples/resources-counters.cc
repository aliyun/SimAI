/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: John Abraham <john.abraham.in@gmail.com>
 */

#include <iostream>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-layout-module.h"

using namespace ns3;

AnimationInterface * pAnim = 0;

/// RGB struture
struct rgb {
  uint8_t r; ///< red
  uint8_t g; ///< green
  uint8_t b; ///< blue
};

struct rgb colors [] = {
                        { 255, 0, 0 }, // Red
                        { 0, 255, 0 }, // Blue
                        { 0, 0, 255 }  // Green
                        };

uint32_t resourceId1;
uint32_t resourceId2;
uint32_t nodeCounterIdUint32;
uint32_t nodeCounterIdDouble1;
uint32_t nodeCounterIdDouble2;

void modify ()
{
  std::ostringstream oss;
  oss << "Update:" << Simulator::Now ().GetSeconds ();
  pAnim->UpdateLinkDescription (0, 1, oss.str ());
  pAnim->UpdateLinkDescription (0, 2, oss.str ());
  pAnim->UpdateLinkDescription (0, 3, oss.str ());
  pAnim->UpdateLinkDescription (0, 4, oss.str ());
  pAnim->UpdateLinkDescription (0, 5, oss.str ());
  pAnim->UpdateLinkDescription (0, 6, oss.str ());
  pAnim->UpdateLinkDescription (1, 7, oss.str ());
  pAnim->UpdateLinkDescription (1, 8, oss.str ());
  pAnim->UpdateLinkDescription (1, 9, oss.str ());
  pAnim->UpdateLinkDescription (1, 10, oss.str ());
  pAnim->UpdateLinkDescription (1, 11, oss.str ());
  
  // Every update change the node description for node 2
  std::ostringstream node0Oss;
  node0Oss << "-----Node:" << Simulator::Now ().GetSeconds ();
  pAnim->UpdateNodeDescription (2, node0Oss.str ());
  static double size = 2;
  static uint32_t currentResourceId = resourceId1;
  pAnim->UpdateNodeSize (2, size, size);
  pAnim->UpdateNodeImage (3, currentResourceId);
  size *= 1.1;
  if (size > 20)
    size = 1;
  pAnim->UpdateNodeSize (3, 10, 10);
  if (currentResourceId == resourceId1)
    currentResourceId = resourceId2;
  else
    currentResourceId = resourceId1;    

  // Every update change the color for node 4
  static uint32_t index = 0;
  index++;
  if (index == 3) 
    index = 0;
  struct rgb color = colors[index];
  for (uint32_t nodeId = 4; nodeId < 12; ++nodeId)
    pAnim->UpdateNodeColor (nodeId, color.r, color.g, color.b); 

  // Update Node Counter for node 0 and node 5, use some random number between 0 to 1000 for value
  Ptr <UniformRandomVariable> rv = CreateObject<UniformRandomVariable> ();
  pAnim->UpdateNodeCounter (nodeCounterIdUint32, 0, rv->GetValue (0, 1000));
  pAnim->UpdateNodeCounter (nodeCounterIdDouble1, 0, rv->GetValue (100.0, 200.0));
  pAnim->UpdateNodeCounter (nodeCounterIdDouble2, 0, rv->GetValue (300.0, 400.0));
  pAnim->UpdateNodeCounter (nodeCounterIdUint32, 5, rv->GetValue (0, 1000));
  pAnim->UpdateNodeCounter (nodeCounterIdDouble1, 5, rv->GetValue (100.0, 200.0));
  pAnim->UpdateNodeCounter (nodeCounterIdDouble2, 5, rv->GetValue (300.0, 400.0));

  if (Simulator::Now ().GetSeconds () < 10) // This is important or the simulation
    // will run endlessly
    Simulator::Schedule (Seconds (0.1), modify);

}

int main (int argc, char *argv[])
{
  Config::SetDefault ("ns3::OnOffApplication::PacketSize", UintegerValue (512));
  Config::SetDefault ("ns3::OnOffApplication::DataRate", StringValue ("500kb/s"));

  uint32_t    nLeftLeaf = 5;
  uint32_t    nRightLeaf = 5;
  uint32_t    nLeaf = 0; // If non-zero, number of both left and right
  std::string animFile = "resources_demo.xml" ;  // Name of file for animation output

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nLeftLeaf", "Number of left side leaf nodes", nLeftLeaf);
  cmd.AddValue ("nRightLeaf","Number of right side leaf nodes", nRightLeaf);
  cmd.AddValue ("nLeaf",     "Number of left and right side leaf nodes", nLeaf);
  cmd.AddValue ("animFile",  "File Name for Animation Output", animFile);

  cmd.Parse (argc,argv);
  if (nLeaf > 0)
    {
      nLeftLeaf = nLeaf;
      nRightLeaf = nLeaf;
    }

  // Create the point-to-point link helpers
  PointToPointHelper pointToPointRouter;
  pointToPointRouter.SetDeviceAttribute  ("DataRate", StringValue ("10Mbps"));
  pointToPointRouter.SetChannelAttribute ("Delay", StringValue ("1ms"));
  PointToPointHelper pointToPointLeaf;
  pointToPointLeaf.SetDeviceAttribute    ("DataRate", StringValue ("10Mbps"));
  pointToPointLeaf.SetChannelAttribute   ("Delay", StringValue ("1ms"));

  PointToPointDumbbellHelper d (nLeftLeaf, pointToPointLeaf,
                                nRightLeaf, pointToPointLeaf,
                                pointToPointRouter);

  // Install Stack
  InternetStackHelper stack;
  d.InstallStack (stack);

  // Assign IP Addresses
  d.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.2.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.3.1.0", "255.255.255.0"));

  d.BoundingBox (1, 1, 100, 100);
  // Install on/off app on all right side nodes
  OnOffHelper clientHelper ("ns3::UdpSocketFactory", Address ());
  clientHelper.SetAttribute 
    ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=0.|Max=1.]"));
  clientHelper.SetAttribute 
    ("OffTime", StringValue ("ns3::UniformRandomVariable[Min=0.|Max=1.]"));
  ApplicationContainer clientApps;

  for (uint32_t i = 0; i < d.RightCount (); ++i)
    {
      // Create an on/off app sending packets to the same leaf right side
      AddressValue remoteAddress (InetSocketAddress (d.GetLeftIpv4Address (i), 1000));
      clientHelper.SetAttribute ("Remote", remoteAddress);
      clientApps.Add (clientHelper.Install (d.GetRight (i)));
    }

  clientApps.Start (Seconds (0.0));
  clientApps.Stop (Seconds (5.0));

  // Set the bounding box for animation


  // Create the animation object and configure for specified output
  pAnim = new AnimationInterface (animFile); 
  // Provide the absolute path to the resource
  resourceId1 = pAnim->AddResource ("/Users/john/ns3/netanim-3.105/ns-3-logo1.png");
  resourceId2 = pAnim->AddResource ("/Users/john/ns3/netanim-3.105/ns-3-logo2.png");
  pAnim->SetBackgroundImage ("/Users/john/ns3/netanim-3.105/ns-3-background.png", 0, 0, 0.2, 0.2, 0.1);


  // Add a node counter
  nodeCounterIdUint32 = pAnim->AddNodeCounter ("Uint32 Counter", AnimationInterface::UINT32_COUNTER);
  nodeCounterIdDouble1 = pAnim->AddNodeCounter ("Double Counter 1", AnimationInterface::DOUBLE_COUNTER);
  nodeCounterIdDouble2 = pAnim->AddNodeCounter ("Double Counter 2", AnimationInterface::DOUBLE_COUNTER);

  Simulator::Schedule (Seconds (0.1), modify);
  
  // Set up the actual simulation
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  Simulator::Run ();
  std::cout << "Animation Trace file created:" << animFile.c_str ()<< std::endl;
  Simulator::Destroy ();
  delete pAnim;
  return 0;
}

