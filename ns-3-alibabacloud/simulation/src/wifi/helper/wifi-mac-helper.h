/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef WIFI_MAC_HELPER_H
#define WIFI_MAC_HELPER_H

#include "ns3/object-factory.h"
#include "ns3/wifi-standards.h"

namespace ns3 {

class WifiMac;
class WifiNetDevice;

/**
 * \brief create MAC layers for a ns3::WifiNetDevice.
 *
 * This class can create MACs of type ns3::ApWifiMac, ns3::StaWifiMac and ns3::AdhocWifiMac.
 * Its purpose is to allow a WifiHelper to configure and install WifiMac objects on a collection
 * of nodes. The WifiMac objects themselves are mainly composed of TxMiddle, RxMiddle, ChannelAccessManager,
 * FrameExchangeManager, WifiRemoteStationManager, MpduAggregator and MsduAggregartor objects, so this helper
 * offers the opportunity to configure attribute values away from their default values, on a
 * per-NodeContainer basis. By default, it creates an Adhoc MAC layer without QoS. Typically,
 * it is used to set type and attribute values, then hand this object over to the WifiHelper that
 * finishes the job of installing.
 *
 * This class may be further subclassed (WaveMacHelper is an example of this).
 *
 */
class WifiMacHelper
{
public:
  /**
   * Create a WifiMacHelper to make life easier for people who want to
   * work with Wifi MAC layers.
   */
  WifiMacHelper ();
  /**
   * Destroy a WifiMacHelper.
   */
  virtual ~WifiMacHelper ();

  /**
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param type the type of ns3::WifiMac to create.
   * \param args A sequence of name-value pairs of the attributes to set.
   *
   * All the attributes specified in this method should exist
   * in the requested MAC.
   */
  template <typename... Args>
  void SetType (std::string type, Args&&... args);

  /**
   * Helper function used to set the Protection Manager.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param type the type of Protection Manager
   * \param args A sequence of name-value pairs of the attributes to set.
   */
  template <typename... Args>
  void SetProtectionManager (std::string type, Args&&... args);

  /**
   * Helper function used to set the Acknowledgment Manager.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param type the type of Acknowledgment Manager
   * \param args A sequence of name-value pairs of the attributes to set.
   */
  template <typename... Args>
  void SetAckManager (std::string type, Args&&... args);

  /**
   * Helper function used to set the Multi User Scheduler that can be aggregated
   * to an HE AP's MAC.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param type the type of Multi User Scheduler
   * \param args A sequence of name-value pairs of the attributes to set.
   */
  template <typename... Args>
  void SetMultiUserScheduler (std::string type, Args&&... args);

  /**
   * \param device the device within which the MAC object will reside
   * \param standard the standard to configure during installation
   * \returns a new MAC object.
   *
   * This allows the ns3::WifiHelper class to create MAC objects from ns3::WifiHelper::Install.
   */
  virtual Ptr<WifiMac> Create (Ptr<WifiNetDevice> device, WifiStandard standard) const;


protected:
  ObjectFactory m_mac;                ///< MAC object factory
  ObjectFactory m_protectionManager;  ///< Factory to create a protection manager
  ObjectFactory m_ackManager;         ///< Factory to create an acknowledgment manager
  ObjectFactory m_muScheduler;        ///< Multi-user Scheduler object factory
};

} // namespace ns3


/***************************************************************
 *  Implementation of the templates declared above.
 ***************************************************************/

namespace ns3 {

template <typename... Args>
void
WifiMacHelper::SetType (std::string type, Args&&... args)
{
  m_mac.SetTypeId (type);
  m_mac.Set (args...);
}

template <typename... Args>
void
WifiMacHelper::SetProtectionManager (std::string type, Args&&... args)
{
  m_protectionManager.SetTypeId (type);
  m_protectionManager.Set (args...);
}

template <typename... Args>
void
WifiMacHelper::SetAckManager (std::string type, Args&&... args)
{
  m_ackManager.SetTypeId (type);
  m_ackManager.Set (args...);
}

template <typename... Args>
void
WifiMacHelper::SetMultiUserScheduler (std::string type, Args&&... args)
{
  m_muScheduler.SetTypeId (type);
  m_muScheduler.Set (args...);
}

} // namespace ns3

#endif /* WIFI_MAC_HELPER_H */
