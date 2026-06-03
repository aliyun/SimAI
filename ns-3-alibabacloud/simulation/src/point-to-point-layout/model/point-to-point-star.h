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

// Define an object to create a star topology.

#ifndef POINT_TO_POINT_STAR_HELPER_H
#define POINT_TO_POINT_STAR_HELPER_H

#include <string>

#include "ns3/point-to-point-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/ipv6-interface-container.h"

namespace ns3 {

/**
 * \defgroup point-to-point-layout Point-to-Point Layout Helpers
 *
 */

/**
 * \ingroup point-to-point-layout
 *
 * \brief A helper to make it easier to create a star topology
 * with PointToPoint links
 */
class PointToPointStarHelper
{
public:
  /**
   * Create a PointToPointStarHelper in order to easily create
   * star topologies using p2p links
   *
   * \param numSpokes the number of links attached to 
   *        the hub node, creating a total of 
   *        numSpokes + 1 nodes
   *
   * \param p2pHelper the link helper for p2p links, 
   *        used to link nodes together
   */
  PointToPointStarHelper (uint32_t numSpokes, 
                          PointToPointHelper p2pHelper);

  ~PointToPointStarHelper ();

public:
  /**
   * \returns a node pointer to the hub node in the
   *          star, i.e., the center node
   */
  Ptr<Node> GetHub () const;

  /**
   * \param i an index into the spokes of the star
   *
   * \returns a node pointer to the node at the indexed spoke
   */
  Ptr<Node> GetSpokeNode (uint32_t i) const;

  /**
   * \param i index into the hub interfaces
   *
   * \returns Ipv4Address according to indexed hub interface
   */
  Ipv4Address GetHubIpv4Address (uint32_t i) const;

  /**
   * \param i index into the spoke interfaces
   *
   * \returns Ipv4Address according to indexed spoke interface
   */
  Ipv4Address GetSpokeIpv4Address (uint32_t i) const;

  /**
   * \param i index into the hub interfaces
   *
   * \returns Ipv6Address according to indexed hub interface
   */
  Ipv6Address GetHubIpv6Address (uint32_t i) const;

  /**
   * \param i index into the spoke interfaces
   *
   * \returns Ipv6Address according to indexed spoke interface
   */
  Ipv6Address GetSpokeIpv6Address (uint32_t i) const;

  /**
   * \returns the total number of spokes in the star
   */
  uint32_t SpokeCount () const;

  /**
   * \param stack an InternetStackHelper which is used to install 
   *              on every node in the star
   */
  void InstallStack (InternetStackHelper stack);

  /**
   * \param address an Ipv4AddressHelper which is used to install 
   *                Ipv4 addresses on all the node interfaces in 
   *                the star
   */
  void AssignIpv4Addresses (Ipv4AddressHelper address);

  /**
   * \param network an IPv6 address representing the network portion
   *                of the IPv6 Address
   * \param prefix the prefix length
   */
  void AssignIpv6Addresses (Ipv6Address network, Ipv6Prefix prefix);

  /**
   * Sets up the node canvas locations for every node in the star. 
   * This is needed for use with the animation interface
   *
   * \param ulx upper left x value
   * \param uly upper left y value
   * \param lrx lower right x value
   * \param lry lower right y value
   */
  void BoundingBox (double ulx, double uly, double lrx, double lry);

private:
  NodeContainer m_hub;              //!< Hub node
  NetDeviceContainer m_hubDevices;  //!< Hub node NetDevices
  NodeContainer m_spokes;                     //!< Spoke nodes
  NetDeviceContainer m_spokeDevices;          //!< Spoke nodes NetDevices
  Ipv4InterfaceContainer m_hubInterfaces;     //!< IPv4 hub interfaces
  Ipv4InterfaceContainer m_spokeInterfaces;   //!< IPv4 spoke nodes interfaces
  Ipv6InterfaceContainer m_hubInterfaces6;    //!< IPv6 hub interfaces
  Ipv6InterfaceContainer m_spokeInterfaces6;  //!< IPv6 spoke nodes interfaces
};

} // namespace ns3

#endif /* POINT_TO_POINT_STAR_HELPER_H */
