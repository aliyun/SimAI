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
 * Author: George F. Riley<riley@ece.gatech.edu>
 */

// Implement an object to create a dumbbell topology.

#include <cmath>
#include <iostream>
#include <sstream>

// ns3 includes
#include "ns3/log.h"
#include "ns3/point-to-point-dumbbell.h"
#include "ns3/constant-position-mobility-model.h"

#include "ns3/node-list.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/vector.h"
#include "ns3/ipv6-address-generator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PointToPointDumbbellHelper");

PointToPointDumbbellHelper::PointToPointDumbbellHelper (uint32_t nLeftLeaf,
                                                        PointToPointHelper leftHelper,
                                                        uint32_t nRightLeaf,
                                                        PointToPointHelper rightHelper,
                                                        PointToPointHelper bottleneckHelper)
{
  // Create the bottleneck routers
  m_routers.Create (2);
  // Create the leaf nodes
  m_leftLeaf.Create (nLeftLeaf);
  m_rightLeaf.Create (nRightLeaf);

  // Add the link connecting routers
  m_routerDevices = bottleneckHelper.Install (m_routers);
  // Add the left side links
  for (uint32_t i = 0; i < nLeftLeaf; ++i)
    {
      NetDeviceContainer c = leftHelper.Install (m_routers.Get (0),
                                                 m_leftLeaf.Get (i));
      m_leftRouterDevices.Add (c.Get (0));
      m_leftLeafDevices.Add (c.Get (1));
    }
  // Add the right side links
  for (uint32_t i = 0; i < nRightLeaf; ++i)
    {
      NetDeviceContainer c = rightHelper.Install (m_routers.Get (1),
                                                  m_rightLeaf.Get (i));
      m_rightRouterDevices.Add (c.Get (0));
      m_rightLeafDevices.Add (c.Get (1));
    }
}

PointToPointDumbbellHelper::~PointToPointDumbbellHelper ()
{
}

Ptr<Node> PointToPointDumbbellHelper::GetLeft () const
{ // Get the left side bottleneck router
  return m_routers.Get (0);
}

Ptr<Node> PointToPointDumbbellHelper::GetLeft (uint32_t i) const
{ // Get the i'th left side leaf
  return m_leftLeaf.Get (i);
}

Ptr<Node> PointToPointDumbbellHelper::GetRight () const
{ // Get the right side bottleneck router
  return m_routers.Get (1);
}

Ptr<Node> PointToPointDumbbellHelper::GetRight (uint32_t i) const
{ // Get the i'th right side leaf
  return m_rightLeaf.Get (i);
}

Ipv4Address PointToPointDumbbellHelper::GetLeftIpv4Address (uint32_t i) const
{
  return m_leftLeafInterfaces.GetAddress (i);
}

Ipv4Address PointToPointDumbbellHelper::GetRightIpv4Address (uint32_t i) const
{
  return m_rightLeafInterfaces.GetAddress (i);
}

Ipv6Address PointToPointDumbbellHelper::GetLeftIpv6Address (uint32_t i) const
{
  return m_leftLeafInterfaces6.GetAddress (i, 1);
}

Ipv6Address PointToPointDumbbellHelper::GetRightIpv6Address (uint32_t i) const
{
  return m_rightLeafInterfaces6.GetAddress (i, 1);
}

uint32_t  PointToPointDumbbellHelper::LeftCount () const
{ // Number of left side nodes
  return m_leftLeaf.GetN ();
}

uint32_t  PointToPointDumbbellHelper::RightCount () const
{ // Number of right side nodes
  return m_rightLeaf.GetN ();
}

void PointToPointDumbbellHelper::InstallStack (InternetStackHelper stack)
{
  stack.Install (m_routers);
  stack.Install (m_leftLeaf);
  stack.Install (m_rightLeaf);
}

void PointToPointDumbbellHelper::AssignIpv4Addresses (Ipv4AddressHelper leftIp,
                                                      Ipv4AddressHelper rightIp,
                                                      Ipv4AddressHelper routerIp)
{
  // Assign the router network
  m_routerInterfaces = routerIp.Assign (m_routerDevices);
  // Assign to left side 
  for (uint32_t i = 0; i < LeftCount (); ++i)
    {
      NetDeviceContainer ndc;
      ndc.Add (m_leftLeafDevices.Get (i));
      ndc.Add (m_leftRouterDevices.Get (i));
      Ipv4InterfaceContainer ifc = leftIp.Assign (ndc);
      m_leftLeafInterfaces.Add (ifc.Get (0));
      m_leftRouterInterfaces.Add (ifc.Get (1));
      leftIp.NewNetwork ();
    }
  // Assign to right side
  for (uint32_t i = 0; i < RightCount (); ++i)
    {
      NetDeviceContainer ndc;
      ndc.Add (m_rightLeafDevices.Get (i));
      ndc.Add (m_rightRouterDevices.Get (i));
      Ipv4InterfaceContainer ifc = rightIp.Assign (ndc);
      m_rightLeafInterfaces.Add (ifc.Get (0));
      m_rightRouterInterfaces.Add (ifc.Get (1));
      rightIp.NewNetwork ();
    }
}

void PointToPointDumbbellHelper::AssignIpv6Addresses (Ipv6Address addrBase, Ipv6Prefix prefix)
{
  // Assign the router network
  Ipv6AddressGenerator::Init (addrBase, prefix);
  Ipv6Address v6network;
  Ipv6AddressHelper addressHelper;
  
  v6network = Ipv6AddressGenerator::GetNetwork (prefix);
  addressHelper.SetBase (v6network, prefix);
  m_routerInterfaces6 = addressHelper.Assign (m_routerDevices);
  Ipv6AddressGenerator::NextNetwork (prefix);

  // Assign to left side
  for (uint32_t i = 0; i < LeftCount (); ++i)
    {
      v6network = Ipv6AddressGenerator::GetNetwork (prefix);
      addressHelper.SetBase (v6network, prefix);

      NetDeviceContainer ndc;
      ndc.Add (m_leftLeafDevices.Get (i));
      ndc.Add (m_leftRouterDevices.Get (i));
      Ipv6InterfaceContainer ifc = addressHelper.Assign (ndc);
      Ipv6InterfaceContainer::Iterator it = ifc.Begin ();
      m_leftLeafInterfaces6.Add ((*it).first, (*it).second);
      it++;
      m_leftRouterInterfaces6.Add ((*it).first, (*it).second);
      Ipv6AddressGenerator::NextNetwork (prefix);
    }
  // Assign to right side
  for (uint32_t i = 0; i < RightCount (); ++i)
    {
      v6network = Ipv6AddressGenerator::GetNetwork (prefix);
      addressHelper.SetBase (v6network, prefix);

      NetDeviceContainer ndc;
      ndc.Add (m_rightLeafDevices.Get (i));
      ndc.Add (m_rightRouterDevices.Get (i));
      Ipv6InterfaceContainer ifc = addressHelper.Assign (ndc);
      Ipv6InterfaceContainer::Iterator it = ifc.Begin ();
      m_rightLeafInterfaces6.Add ((*it).first, (*it).second);
      it++;
      m_rightRouterInterfaces6.Add ((*it).first, (*it).second);
      Ipv6AddressGenerator::NextNetwork (prefix);
    }
}


void PointToPointDumbbellHelper::BoundingBox (double ulx, double uly, // Upper left x/y
                                              double lrx, double lry) // Lower right x/y
{
  double xDist;
  double yDist;
  if (lrx > ulx)
    {
      xDist = lrx - ulx;
    }
  else
    {
      xDist = ulx - lrx;
    }
  if (lry > uly)
    {
      yDist = lry - uly;
    }
  else
    {
      yDist = uly - lry;
    }

  double xAdder = xDist / 3.0;
  double  thetaL = M_PI / (LeftCount () + 1.0);
  double  thetaR = M_PI / (RightCount () + 1.0);

  // Place the left router
  Ptr<Node> lr = GetLeft ();
  Ptr<ConstantPositionMobilityModel> loc = lr->GetObject<ConstantPositionMobilityModel> ();
  if (loc == 0)
    {
      loc = CreateObject<ConstantPositionMobilityModel> ();
      lr->AggregateObject (loc);
    }
  Vector lrl (ulx + xAdder, uly + yDist/2.0, 0);
  loc->SetPosition (lrl);

  // Place the right router
  Ptr<Node> rr = GetRight ();
  loc = rr->GetObject<ConstantPositionMobilityModel> ();
  if (loc == 0)
    {
      loc = CreateObject<ConstantPositionMobilityModel> ();
      rr->AggregateObject (loc);
    }
  Vector rrl (ulx + xAdder * 2, uly + yDist/2.0, 0); // Right router location
  loc->SetPosition (rrl);

  // Place the left leaf nodes
  double theta = -M_PI_2 + thetaL;
  for (uint32_t l = 0; l < LeftCount (); ++l)
    {
      // Make them in a circular pattern to make all line lengths the same
      // Special case when theta = 0, to be sure we get a straight line
      if ((LeftCount () % 2) == 1)
        { // Count is odd, see if we are in middle
          if (l == (LeftCount () / 2))
            {
              theta = 0.0;
            }
        }
      Ptr<Node> ln = GetLeft (l);
      loc = ln->GetObject<ConstantPositionMobilityModel> ();
      if (loc == 0)
        {
          loc = CreateObject<ConstantPositionMobilityModel> ();
          ln->AggregateObject (loc);
        }
      Vector lnl (lrl.x - std::cos (theta) * xAdder,
                  lrl.y + std::sin (theta) * xAdder, 0);   // Left Node Location
      // Insure did not exceed bounding box
      if (lnl.y < uly) 
        {
          lnl.y = uly; // Set to upper left y
        }
      if (lnl.y > lry) 
        {
          lnl.y = lry; // Set to lower right y
        }
      loc->SetPosition (lnl);
      theta += thetaL;
    }
  // Place the right nodes
  theta = -M_PI_2 + thetaR;
  for (uint32_t r = 0; r < RightCount (); ++r)
    {
      // Special case when theta = 0, to be sure we get a straight line
      if ((RightCount () % 2) == 1)
        { // Count is odd, see if we are in middle
          if (r == (RightCount () / 2))
            {
              theta = 0.0;
            }
        }
      Ptr<Node> rn = GetRight (r);
      loc = rn->GetObject<ConstantPositionMobilityModel> ();
      if (loc == 0)
        {
          loc = CreateObject<ConstantPositionMobilityModel> ();
          rn->AggregateObject (loc);
        }
      Vector rnl (rrl.x + std::cos (theta) * xAdder, // Right node location
                  rrl.y + std::sin (theta) * xAdder, 0);
      // Insure did not exceed bounding box
      if (rnl.y < uly) 
        {
          rnl.y = uly; // Set to upper left y
        }
      if (rnl.y > lry) 
        {
          rnl.y = lry; // Set to lower right y
        }
      loc->SetPosition (rnl);
      theta += thetaR;
    }
}

} // namespace ns3
