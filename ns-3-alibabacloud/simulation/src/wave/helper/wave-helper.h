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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */

#ifndef WAVE_HELPER_H
#define WAVE_HELPER_H

#include <string>
#include "ns3/attribute.h"
#include "ns3/object-factory.h"
#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/trace-helper.h"
#include "ns3/yans-wifi-helper.h"

namespace ns3 {

class WaveNetDevice;
class Node;

/**
 * \ingroup wave
 * To trace WaveNetDevice, we have to overwrite the trace functions of class YansWifiPhyHelper.
 * The source code is very similar with YansWifiPhy, only with some adaptation.
 */
class YansWavePhyHelper : public YansWifiPhyHelper
{
public:
  /**
   * Create a phy helper in a default working state.
   * \return A phy helper
   */
  static YansWavePhyHelper Default (void);

private:
  /**
   * @brief Enable pcap output the indicated net device.
   *
   * NetDevice-specific implementation mechanism for hooking the trace and
   * writing to the trace file.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param nd Net device for which you want to enable tracing.
   * @param promiscuous If true capture all possible packets available at the device.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnablePcapInternal (std::string prefix,
                                   Ptr<NetDevice> nd,
                                   bool promiscuous,
                                   bool explicitFilename);

  /**
   * \brief Enable ascii trace output on the indicated net device.
   * \internal
   *
   * NetDevice-specific implementation mechanism for hooking the trace and
   * writing to the trace file.
   *
   * \param stream The output stream object to use when logging ascii traces.
   * \param prefix Filename prefix to use for ascii trace files.
   * \param nd Net device for which you want to enable tracing.
   * \param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnableAsciiInternal (Ptr<OutputStreamWrapper> stream,
                                    std::string prefix,
                                    Ptr<NetDevice> nd,
                                    bool explicitFilename);
};

/**
 * \ingroup wave
 * \brief helps to create WaveNetDevice objects
 *
 * This class can help to create a large set of similar
 * WaveNetDevice objects and to configure a large set of
 * their attributes during creation.
 *
 * Generally WaveHelper with default configuration will
 * create devices with 7 MAC entities,1 PHY entity and a
 * multiple-channel scheduler for to deal with these entities.
 *
 * If users can make sure on which channel this WAVE device
 * will work, they can set specific channel numbers to save
 * resources of unused channels as below:
 * WaveHelper helper = WaveHelper::Default ();
 * uint32_t channels[] = {CCH, SCH1};
 * std::vector<uint32_t> channelsVector (channels, channels + 2);
 * helper.helper.CreateMacForChannel (channelsVector);
 *
 * If users can create other channel scheduler such as "AnotherScheduler"
 * which can assign channel access in the context of  more PHY entities,
 * they can use this helper to create WAVE devices as below:
 * WaveHelper helper = WaveHelper::Default ();
 * helper.helper.CreateMacForChannel (ChannelManager::GetWaveChannels ());
 * helper.CreatePhys (2);        // or other number which should be less than 7
 * helper.SetChannelScheduler ("ns3::AnotherScheduler");
 * helper.SetRemoteStationManager ("ns3::ConstantRateWifiManager");  // or other  rate control algorithms
 */
class WaveHelper
{
public:
  WaveHelper ();
  virtual ~WaveHelper ();

  /**
   * \returns a new WaveHelper in a default state
   *
   * The default state is defined as being seven OcbWifiMac MAC entities
   * with an ConstantRate rate control algorithm, one WifiPhy entity and
   * a default multiple-channel scheduler DefaultChannelScheduler for
   * assigning channel access with these created entities.
   */
  static WaveHelper Default (void);

  /**
   * \param channelNumbers the MAC entities will be created to support these channels for multiple channel operation.
   */
  void CreateMacForChannel (std::vector<uint32_t>  channelNumbers);
  /**
   * \param phys the number of PHY entity which will be created for multiple channel operation.
   */
  void CreatePhys (uint32_t phys);

  /**
   * \param type the type of ns3::WifiRemoteStationManager to create.
   * \param n0 the name of the attribute to set
   * \param v0 the value of the attribute to set
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   * \param n2 the name of the attribute to set
   * \param v2 the value of the attribute to set
   * \param n3 the name of the attribute to set
   * \param v3 the value of the attribute to set
   * \param n4 the name of the attribute to set
   * \param v4 the value of the attribute to set
   * \param n5 the name of the attribute to set
   * \param v5 the value of the attribute to set
   * \param n6 the name of the attribute to set
   * \param v6 the value of the attribute to set
   * \param n7 the name of the attribute to set
   * \param v7 the value of the attribute to set
   *
   * All the attributes specified in this method should exist
   * in the requested station manager.
   */
  void SetRemoteStationManager (std::string type,
                                std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                                std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                                std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                                std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                                std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                                std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                                std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                                std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue ());
  /**
   * \param type the type of ns3::ChannelScheduler to create.
   * \param n0 the name of the attribute to set
   * \param v0 the value of the attribute to set
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   * \param n2 the name of the attribute to set
   * \param v2 the value of the attribute to set
   * \param n3 the name of the attribute to set
   * \param v3 the value of the attribute to set
   * \param n4 the name of the attribute to set
   * \param v4 the value of the attribute to set
   * \param n5 the name of the attribute to set
   * \param v5 the value of the attribute to set
   * \param n6 the name of the attribute to set
   * \param v6 the value of the attribute to set
   * \param n7 the name of the attribute to set
   * \param v7 the value of the attribute to set
   *
   * All the attributes specified in this method should exist
   * in the requested channel scheduler.
   */
  void SetChannelScheduler (std::string type,
                            std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                            std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                            std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                            std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                            std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                            std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                            std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                            std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue ());

  /**
   * \param phy the PHY helper to create PHY objects
   * \param mac the MAC helper to create MAC objects
   * \param c the set of nodes on which a wifi device must be created
   * \returns a device container which contains all the devices created by this method.
   */
  virtual NetDeviceContainer Install (const WifiPhyHelper &phy,
                                      const WifiMacHelper &mac, NodeContainer c) const;
  /**
   * \param phy the PHY helper to create PHY objects
   * \param mac the MAC helper to create MAC objects
   * \param node the node on which a wifi device must be created
   * \returns a device container which contains all the devices created by this method.
   */
  virtual NetDeviceContainer Install (const WifiPhyHelper &phy,
                                      const WifiMacHelper &mac,   Ptr<Node> node) const;
  /**
   * \param phy the PHY helper to create PHY objects
   * \param mac the MAC helper to create MAC objects
   * \param nodeName the name of node on which a wifi device must be created
   * \returns a device container which contains all the devices created by this method.
   */
  virtual NetDeviceContainer Install (const WifiPhyHelper &phy,
                                      const WifiMacHelper &mac, std::string nodeName) const;

  /**
   * Helper to enable all WaveNetDevice log components with one statement
   */
  static void EnableLogComponents (void);

  /**
  * Assign a fixed random variable stream number to the random variables
  * used by the Phy and Mac aspects of the WAVE models.  Each device in
  * container c has fixed stream numbers assigned to its random variables.
  * The Wifi channel (e.g. propagation loss model) is excluded.
  * Return the number of streams (possibly zero) that
  * have been assigned. The Install() method should have previously been
  * called by the user.
  *
  * \param c NetDeviceContainer of the set of net devices for which the
  *          WaveNetDevice should be modified to use fixed streams
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this helper
  */
  int64_t AssignStreams (NetDeviceContainer c, int64_t stream);

protected:
  ObjectFactory m_stationManager; ///< station manager
  ObjectFactory m_channelScheduler; ///< channel scheduler
  std::vector<uint32_t> m_macsForChannelNumber; ///< MACs for channel number
  uint32_t m_physNumber; ///< Phy number
};
}
#endif /* WAVE_HELPER_H */
