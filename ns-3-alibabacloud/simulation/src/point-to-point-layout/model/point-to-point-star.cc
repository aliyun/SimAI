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
 */

// Implement an object to create a star topology.

#include <cmath>
#include <iostream>
#include <sstream>

// ns3 includes
#include "ns3/log.h"
#include "ns3/point-to-point-star.h"
#include "ns3/constant-position-mobility-model.h"

#include "ns3/node-list.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/vector.h"
#include "ns3/ipv6-address-generator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PointToPointStarHelper");

PointToPointStarHelper::PointToPointStarHelper (uint32_t numSpokes,
                                                PointToPointHelper p2pHelper)
{
  m_hub.Create (1);
  m_spokes.Create (numSpokes);

  for (uint32_t i = 0; i < m_spokes.GetN (); ++i)
    {
      NetDeviceContainer nd = p2pHelper.Install (m_hub.Get (0), m_spokes.Get (i));
      m_hubDevices.Add (nd.Get (0));
      m_spokeDevices.Add (nd.Get (1));
    }
}

PointToPointStarHelper::~PointToPointStarHelper ()
{
}

Ptr<Node>
PointToPointStarHelper::GetHub () const
{
  return m_hub.Get (0);
}

Ptr<Node>
PointToPointStarHelper::GetSpokeNode (uint32_t i) const
{
  return m_spokes.Get (i);
}

Ipv4Address
PointToPointStarHelper::GetHubIpv4Address (uint32_t i) const
{
  return m_hubInterfaces.GetAddress (i);
}

Ipv4Address
PointToPointStarHelper::GetSpokeIpv4Address (uint32_t i) const
{
  return m_spokeInterfaces.GetAddress (i);
}

Ipv6Address
PointToPointStarHelper::GetHubIpv6Address (uint32_t i) const
{
  return m_hubInterfaces6.GetAddress (i, 1);
}

Ipv6Address
PointToPointStarHelper::GetSpokeIpv6Address (uint32_t i) const
{
  return m_spokeInterfaces6.GetAddress (i, 1);
}

uint32_t
PointToPointStarHelper::SpokeCount () const
{
  return m_spokes.GetN ();
}

void 
PointToPointStarHelper::InstallStack (InternetStackHelper stack)
{
  stack.Install (m_hub);
  stack.Install (m_spokes);
}

void 
PointToPointStarHelper::AssignIpv4Addresses (Ipv4AddressHelper address)
{
  for (uint32_t i = 0; i < m_spokes.GetN (); ++i)
    {
      m_hubInterfaces.Add (address.Assign (m_hubDevices.Get (i)));
      m_spokeInterfaces.Add (address.Assign (m_spokeDevices.Get (i)));
      address.NewNetwork ();
    }
}

void 
PointToPointStarHelper::AssignIpv6Addresses (Ipv6Address addrBase, Ipv6Prefix prefix)
{
  Ipv6AddressGenerator::Init (addrBase, prefix);
  Ipv6Address v6network;
  Ipv6AddressHelper addressHelper;

  for (uint32_t i = 0; i < m_spokes.GetN (); ++i)
    {
      v6network = Ipv6AddressGenerator::GetNetwork (prefix);
      addressHelper.SetBase (v6network, prefix);

      Ipv6InterfaceContainer ic = addressHelper.Assign (m_hubDevices.Get (i));
      m_hubInterfaces6.Add (ic);
      ic = addressHelper.Assign (m_spokeDevices.Get (i));
      m_spokeInterfaces6.Add (ic);

      Ipv6AddressGenerator::NextNetwork (prefix);
    }
}

void 
PointToPointStarHelper::BoundingBox (double ulx, double uly,
                                     double lrx, double lry)
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

  // Place the hub
  Ptr<Node> hub = m_hub.Get (0);
  Ptr<ConstantPositionMobilityModel> hubLoc =  hub->GetObject<ConstantPositionMobilityModel> ();
  if (hubLoc == 0)
    {
      hubLoc = CreateObject<ConstantPositionMobilityModel> ();
      hub->AggregateObject (hubLoc);
    }
  Vector hubVec (ulx + xDist/2.0, uly + yDist/2.0, 0);
  hubLoc->SetPosition (hubVec);

  double spokeDist;
  if (xDist > yDist)
    {
      spokeDist = yDist/4.0;
    }
  else
    {
      spokeDist = xDist/4.0;
    }

  double theta = 2*M_PI/m_spokes.GetN ();
  for (uint32_t i = 0; i < m_spokes.GetN (); ++i)
    {
      Ptr<Node> spokeNode = m_spokes.Get (i);
      Ptr<ConstantPositionMobilityModel> spokeLoc = spokeNode->GetObject<ConstantPositionMobilityModel> ();
      if (spokeLoc == 0)
        {
          spokeLoc = CreateObject<ConstantPositionMobilityModel> ();
          spokeNode->AggregateObject (spokeLoc);
        }
      Vector spokeVec (hubVec.x + std::cos (theta*i) * spokeDist,
                       hubVec.y + std::sin (theta*i) * spokeDist,
                       0);
      spokeLoc->SetPosition (spokeVec);
    }
}

} // namespace ns3
