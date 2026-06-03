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

// Define an object to create a dumbbell topology.

#ifndef POINT_TO_POINT_DUMBBELL_HELPER_H
#define POINT_TO_POINT_DUMBBELL_HELPER_H

#include <string>

#include "ns3/point-to-point-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/ipv6-interface-container.h"

namespace ns3 {

/**
 * \ingroup point-to-point-layout
 *
 * \brief A helper to make it easier to create a dumbbell topology
 * with p2p links
 */
class PointToPointDumbbellHelper
{
public:
  /**
   * Create a PointToPointDumbbellHelper in order to easily create
   * dumbbell topologies using p2p links
   *
   * \param nLeftLeaf number of left side leaf nodes in the dumbbell
   *
   * \param leftHelper PointToPointHelper used to install the links 
   *                   between the left leaf nodes and the left-most 
   *                   router
   *
   * \param nRightLeaf number of right side leaf nodes in the dumbbell
   *
   * \param rightHelper PointToPointHelper used to install the links 
   *                    between the right leaf nodes and the right-most 
   *                    router
   *
   * \param bottleneckHelper PointToPointHelper used to install the link 
   *                         between the inner-routers, usually known as 
   *                         the bottleneck link
   */
  PointToPointDumbbellHelper (uint32_t nLeftLeaf,
                              PointToPointHelper leftHelper,
                              uint32_t nRightLeaf,
                              PointToPointHelper rightHelper,
                              PointToPointHelper bottleneckHelper);

  ~PointToPointDumbbellHelper ();

public:
  /**
   * \returns pointer to the node of the left side bottleneck
   *          router
   */
  Ptr<Node> GetLeft () const;

  /**
   * \returns pointer to the i'th left side leaf node
   * \param i node number
   */
  Ptr<Node> GetLeft (uint32_t i) const;

  /**
   * \returns pointer to the node of the right side bottleneck
   *          router
   */
  Ptr<Node> GetRight () const;

  /**
   * \returns pointer to the i'th right side leaf node
   * \param i node number
   */
  Ptr<Node> GetRight (uint32_t i) const;

  /**
   * \returns an Ipv4Address of the i'th left leaf
   * \param i node number
   */
  Ipv4Address GetLeftIpv4Address (uint32_t i ) const; // Get left leaf address

  /**
   * \returns an Ipv4Address of the i'th right leaf
   * \param i node number
   */
  Ipv4Address GetRightIpv4Address (uint32_t i) const; // Get right leaf address

  /**
   * \returns an Ipv6Address of the i'th left leaf
   * \param i node number
   */
  Ipv6Address GetLeftIpv6Address (uint32_t i ) const; // Get left leaf address

  /**
   * \returns an Ipv6Address of the i'th right leaf
   * \param i node number
   */
  Ipv6Address GetRightIpv6Address (uint32_t i) const; // Get right leaf address

  /**
   * \returns total number of left side leaf nodes
   */
  uint32_t  LeftCount () const;

  /**
   * \returns total number of right side leaf nodes
   */
  uint32_t  RightCount () const;

  /**
   * \param stack an InternetStackHelper which is used to install 
   *              on every node in the dumbbell
   */
  void      InstallStack (InternetStackHelper stack);

  /**
   * \param leftIp Ipv4AddressHelper to assign Ipv4 addresses to the
   *               interfaces on the left side of the dumbbell
   *
   * \param rightIp Ipv4AddressHelper to assign Ipv4 addresses to the
   *                interfaces on the right side of the dumbbell
   *
   * \param routerIp Ipv4AddressHelper to assign Ipv4 addresses to the 
   *                 interfaces on the bottleneck link
   */
  void      AssignIpv4Addresses (Ipv4AddressHelper leftIp,
                                 Ipv4AddressHelper rightIp,
                                 Ipv4AddressHelper routerIp);

  /**
   * \param network an IPv6 address representing the network portion
   *                of the IPv6 Address
   * \param prefix the prefix length
   */
  void      AssignIpv6Addresses (Ipv6Address network, Ipv6Prefix prefix);

  /**
   * Sets up the node canvas locations for every node in the dumbbell.
   * This is needed for use with the animation interface
   *
   * \param ulx upper left x value
   * \param uly upper left y value
   * \param lrx lower right x value
   * \param lry lower right y value
   */
  void      BoundingBox (double ulx, double uly, double lrx, double lry);

private:
  NodeContainer          m_leftLeaf;            //!< Left Leaf nodes
  NetDeviceContainer     m_leftLeafDevices;     //!< Left Leaf NetDevices
  NodeContainer          m_rightLeaf;           //!< Right Leaf nodes
  NetDeviceContainer     m_rightLeafDevices;    //!< Right Leaf NetDevices
  NodeContainer          m_routers;             //!< Routers
  NetDeviceContainer     m_routerDevices;       //!< Routers NetDevices
  NetDeviceContainer     m_leftRouterDevices;     //!< Left router NetDevices
  NetDeviceContainer     m_rightRouterDevices;    //!< Right router NetDevices
  Ipv4InterfaceContainer m_leftLeafInterfaces;    //!< Left Leaf interfaces (IPv4)
  Ipv4InterfaceContainer m_leftRouterInterfaces;  //!< Left router interfaces (IPv4)
  Ipv4InterfaceContainer m_rightLeafInterfaces;   //!< Right Leaf interfaces (IPv4)
  Ipv4InterfaceContainer m_rightRouterInterfaces; //!< Right router interfaces (IPv4)
  Ipv4InterfaceContainer m_routerInterfaces;      //!< Router interfaces (IPv4)
  Ipv6InterfaceContainer m_leftLeafInterfaces6;   //!< Left Leaf interfaces (IPv6)
  Ipv6InterfaceContainer m_leftRouterInterfaces6; //!< Left router interfaces (IPv6)
  Ipv6InterfaceContainer m_rightLeafInterfaces6;  //!< Right Leaf interfaces (IPv6)
  Ipv6InterfaceContainer m_rightRouterInterfaces6;  //!< Right router interfaces (IPv6)
  Ipv6InterfaceContainer m_routerInterfaces6;     //!< Router interfaces (IPv6)
};

} // namespace ns3

#endif /* POINT_TO_POINT_DUMBBELL_HELPER_H */
