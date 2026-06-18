/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *         Junling Bu <linlinjavaer@gmail.com>
 */
#include <algorithm>
#include "ns3/node.h"
#include "ns3/wifi-phy.h"
#include "ns3/llc-snap-header.h"
#include "ns3/channel.h"
#include "ns3/log.h"
#include "ns3/socket.h"
#include "ns3/object-map.h"
#include "ns3/object-vector.h"
#include "wave-net-device.h"
#include "higher-tx-tag.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WaveNetDevice");

NS_OBJECT_ENSURE_REGISTERED (WaveNetDevice);

TypeId
WaveNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WaveNetDevice")
    .SetParent<NetDevice> ()
    .SetGroupName ("Wave")
    .AddConstructor<WaveNetDevice> ()
    .AddAttribute ("Mtu", "The MAC-level Maximum Transmission Unit",
                   UintegerValue (MAX_MSDU_SIZE - LLC_SNAP_HEADER_LENGTH),
                   MakeUintegerAccessor (&WaveNetDevice::SetMtu,
                                         &WaveNetDevice::GetMtu),
                   MakeUintegerChecker<uint16_t> (1,MAX_MSDU_SIZE - LLC_SNAP_HEADER_LENGTH))
    .AddAttribute ("Channel", "The channel attached to this device",
                   PointerValue (),
                   MakePointerAccessor (&WaveNetDevice::GetChannel),
                   MakePointerChecker<Channel> ())
    .AddAttribute ("PhyEntities", "The PHY entities attached to this device.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&WaveNetDevice::m_phyEntities),
                   MakeObjectVectorChecker<WifiPhy> ())
    .AddAttribute ("MacEntities", "The MAC layer attached to this device.",
                   ObjectMapValue (),
                   MakeObjectMapAccessor (&WaveNetDevice::m_macEntities),
                   MakeObjectMapChecker<OcbWifiMac> ())
    .AddAttribute ("ChannelScheduler", "The channel scheduler attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WaveNetDevice::SetChannelScheduler,
                                        &WaveNetDevice::GetChannelScheduler),
                   MakePointerChecker<ChannelScheduler> ())
    .AddAttribute ("ChannelManager", "The channel manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WaveNetDevice::SetChannelManager,
                                        &WaveNetDevice::GetChannelManager),
                   MakePointerChecker<ChannelManager> ())
    .AddAttribute ("ChannelCoordinator", "The channel coordinator attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WaveNetDevice::SetChannelCoordinator,
                                        &WaveNetDevice::GetChannelCoordinator),
                   MakePointerChecker<ChannelCoordinator> ())
    .AddAttribute ("VsaManager", "The VSA manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WaveNetDevice::SetVsaManager,
                                        &WaveNetDevice::GetVsaManager),
                   MakePointerChecker<VsaManager> ())
  ;
  return tid;
}

WaveNetDevice::WaveNetDevice (void)
  : m_txProfile (0)
{
  NS_LOG_FUNCTION (this);
}

WaveNetDevice::~WaveNetDevice (void)
{
  NS_LOG_FUNCTION (this);
}

void
WaveNetDevice::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  if (m_txProfile != 0)
    {
      delete m_txProfile;
      m_txProfile = 0;
    }
  for (PhyEntitiesI i = m_phyEntities.begin (); i != m_phyEntities.end (); ++i)
    {
      Ptr<WifiPhy> phy = (*i);
      phy->Dispose ();
    }
  m_phyEntities.clear ();
  for (MacEntitiesI i = m_macEntities.begin (); i != m_macEntities.end (); ++i)
    {
      Ptr<OcbWifiMac> mac = i->second;
      Ptr<WifiRemoteStationManager> stationManager = mac->GetWifiRemoteStationManager ();
      stationManager->Dispose ();
      mac->Dispose ();
    }
  m_macEntities.clear ();
  m_phyEntities.clear ();
  m_channelCoordinator->Dispose ();
  m_channelManager->Dispose ();
  m_channelScheduler->Dispose ();
  m_vsaManager->Dispose ();
  m_channelCoordinator = 0;
  m_channelManager = 0;
  m_channelScheduler = 0;
  m_vsaManager = 0;
  // chain up.
  WifiNetDevice::DoDispose ();
}

void
WaveNetDevice::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  if (m_phyEntities.size () == 0)
    {
      NS_FATAL_ERROR ("there is no PHY entity in this WAVE device");
    }
  for (PhyEntitiesI i = m_phyEntities.begin (); i != m_phyEntities.end (); ++i)
    {
      Ptr<WifiPhy> phy = (*i);
      phy->Initialize ();
    }
  if (m_macEntities.size () == 0)
    {
      NS_FATAL_ERROR ("there is no MAC entity in this WAVE device");
    }
  for (MacEntitiesI i = m_macEntities.begin (); i != m_macEntities.end (); ++i)
    {
      Ptr<OcbWifiMac> mac = i->second;
      mac->SetForwardUpCallback (MakeCallback (&WaveNetDevice::ForwardUp, this));
      // Make each MAC entity in sleep mode.
      mac->Suspend ();
      mac->Initialize ();

      Ptr<WifiRemoteStationManager> stationManager = mac->GetWifiRemoteStationManager ();
      // Currently PHY is not attached to MAC and will be dynamically attached and unattached to MAC latter,
      // however WifiRemoteStationManager in the MAC shall know something  in the PHY such as supported data rates.
      // Since these information can be treated as same when same PHY devices are added, here we force
      // all of WifiRemoteStationManagers in multiple MAC entities only associate with single PHY device even there may
      // be multiple PHY entities. This approach may be strange but can work fine.
      stationManager->SetupPhy (m_phyEntities[0]);
      stationManager->Initialize ();
    }
  m_channelScheduler->SetWaveNetDevice (this);
  m_vsaManager->SetWaveNetDevice (this);
  m_channelScheduler->Initialize ();
  m_channelCoordinator->Initialize ();
  m_channelManager->Initialize ();
  m_vsaManager->Initialize ();
  NetDevice::DoInitialize ();
}

void
WaveNetDevice::AddMac (uint32_t channelNumber, Ptr<OcbWifiMac> mac)
{
  NS_LOG_FUNCTION (this << channelNumber << mac);
  if (!ChannelManager::IsWaveChannel (channelNumber))
    {
      NS_FATAL_ERROR ("The channel " << channelNumber << " is not a valid WAVE channel number");
    }
  if (m_macEntities.find (channelNumber) != m_macEntities.end ())
    {
      NS_FATAL_ERROR ("The MAC entity for channel " << channelNumber << " already exists.");
    }
  m_macEntities.insert (std::make_pair (channelNumber, mac));
}
Ptr<OcbWifiMac>
WaveNetDevice::GetMac (uint32_t channelNumber) const
{
  NS_LOG_FUNCTION (this << channelNumber);
  MacEntitiesI i = m_macEntities.find (channelNumber);
  if (i == m_macEntities.end ())
    {
      NS_FATAL_ERROR ("there is no available MAC entity for channel " << channelNumber);
    }
  return i->second;
}

std::map<uint32_t, Ptr<OcbWifiMac> >
WaveNetDevice::GetMacs (void) const
{
  NS_LOG_FUNCTION (this);
  return m_macEntities;
}

void
WaveNetDevice::AddPhy (Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  if (std::find (m_phyEntities.begin (), m_phyEntities.end (), phy) != m_phyEntities.end ())
    {
      NS_FATAL_ERROR ("This PHY entity is already inserted");
    }
  m_phyEntities.push_back (phy);
}
Ptr<WifiPhy>
WaveNetDevice::GetPhy (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);
  return m_phyEntities.at (index);
}

std::vector<Ptr<WifiPhy> >
WaveNetDevice::GetPhys (void) const
{
  NS_LOG_FUNCTION (this);
  return m_phyEntities;
}

bool
WaveNetDevice::StartVsa (const VsaInfo & vsaInfo)
{
  NS_LOG_FUNCTION (this << &vsaInfo);
  if (!IsAvailableChannel ( vsaInfo.channelNumber))
    {
      return false;
    }
  if (!m_channelScheduler->IsChannelAccessAssigned (vsaInfo.channelNumber))
    {
      NS_LOG_DEBUG ("there is no channel access assigned for channel " << vsaInfo.channelNumber);
      return false;
    }
  if (vsaInfo.vsc == 0)
    {
      NS_LOG_DEBUG ("vendor specific information shall not be null");
      return false;
    }
  if (vsaInfo.oi.IsNull () && vsaInfo.managementId >= 16)
    {
      NS_LOG_DEBUG ("when organization identifier is not set, management ID "
                    "shall be in range from 0 to 15");
      return false;
    }

  m_vsaManager->SendVsa (vsaInfo);
  return true;
}


bool
WaveNetDevice::StopVsa (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  if (!IsAvailableChannel (channelNumber))
    {
      return false;
    }
  m_vsaManager->RemoveByChannel (channelNumber);
  return true;
}

void
WaveNetDevice::SetWaveVsaCallback (WaveVsaCallback vsaCallback)
{
  NS_LOG_FUNCTION (this);
  m_vsaManager->SetWaveVsaCallback (vsaCallback);
}

bool
WaveNetDevice::StartSch (const SchInfo & schInfo)
{
  NS_LOG_FUNCTION (this << &schInfo);
  if (!IsAvailableChannel (schInfo.channelNumber))
    {
      return false;
    }
  return m_channelScheduler->StartSch (schInfo);
}

bool
WaveNetDevice::StopSch (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  if (!IsAvailableChannel (channelNumber))
    {
      return false;
    }
  return m_channelScheduler->StopSch (channelNumber);
}

bool
WaveNetDevice::RegisterTxProfile (const TxProfile & txprofile)
{
  NS_LOG_FUNCTION (this << &txprofile);
  if (m_txProfile != 0)
    {
      return false;
    }
  if (!IsAvailableChannel (txprofile.channelNumber))
    {
      return false;
    }
  if (txprofile.txPowerLevel > 8)
    {
      return false;
    }
  // IP-based packets is not allowed to send in the CCH.
  if (txprofile.channelNumber == CCH)
    {
      NS_LOG_DEBUG ("IP-based packets shall not be transmitted on the CCH");
      return false;
    }
  if  (txprofile.dataRate == WifiMode () || txprofile.txPowerLevel == 8)
    {
      // let MAC layer itself determine tx parameters.
      NS_LOG_DEBUG ("High layer does not want to control tx parameters.");
    }
  else
    {
      // if current PHY devices do not support data rate of the tx profile
      for (PhyEntitiesI i = m_phyEntities.begin (); i != m_phyEntities.end (); ++i)
        {
          if (!((*i)->IsModeSupported (txprofile.dataRate)))
            {
              NS_LOG_DEBUG ("This data rate " << txprofile.dataRate.GetUniqueName () << " is not supported by current PHY device");
              return false;
            }
        }
    }

  m_txProfile = new TxProfile ();
  *m_txProfile = txprofile;
  return true;
}

bool
WaveNetDevice::DeleteTxProfile (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  if (!IsAvailableChannel (channelNumber))
    {
      return false;
    }
  if (m_txProfile == 0)
    {
      return false;
    }
  if (m_txProfile->channelNumber != channelNumber)
    {
      return false;
    }

  delete m_txProfile;
  m_txProfile = 0;
  return true;
}

bool
WaveNetDevice::SendX (Ptr<Packet> packet, const Address & dest, uint32_t protocol, const TxInfo & txInfo)
{
  NS_LOG_FUNCTION (this << packet << dest << protocol << &txInfo);
  if (!IsAvailableChannel (txInfo.channelNumber))
    {
      return false;
    }
  if (!m_channelScheduler->IsChannelAccessAssigned (txInfo.channelNumber))
    {
      NS_LOG_DEBUG ("there is no channel access assigned for channel " << txInfo.channelNumber);
      return false;
    }
  if ((txInfo.channelNumber == CCH) && (protocol == IPv4_PROT_NUMBER || protocol == IPv6_PROT_NUMBER))
    {
      NS_LOG_DEBUG ("IP-based packets shall not be transmitted on the CCH");
      return false;
    }
  if ((txInfo.priority > 7) || txInfo.txPowerLevel > 8)
    {
      NS_LOG_DEBUG ("invalid transmit parameters.");
      return false;
    }

  if ((txInfo.dataRate == WifiMode ()) ||  (txInfo.txPowerLevel == 8))
    {
      // let MAC layer itself determine tx parameters.
      NS_LOG_DEBUG ("High layer does not want to control tx parameters.");
    }
  else
    {
      // if current PHY devices do not support data rate of the  tx profile
      for (PhyEntitiesI i = m_phyEntities.begin (); i != m_phyEntities.end (); ++i)
        {
          if ( !((*i)->IsModeSupported (txInfo.dataRate)))
            {
              return false;
            }
        }
      WifiTxVector txVector;
      txVector.SetChannelWidth (10);
      txVector.SetTxPowerLevel (txInfo.txPowerLevel);
      txVector.SetMode (txInfo.dataRate);
      txVector.SetPreambleType (txInfo.preamble);
      HigherLayerTxVectorTag tag = HigherLayerTxVectorTag (txVector, false);
      packet->AddPacketTag (tag);
    }

  LlcSnapHeader llc;
  llc.SetType (protocol);
  packet->AddHeader (llc);

  // according to channel number and priority,
  // route the packet to a proper queue.
  SocketPriorityTag prio;
  prio.SetPriority (txInfo.priority);
  packet->ReplacePacketTag (prio);
  Ptr<WifiMac> mac = GetMac (txInfo.channelNumber);
  Mac48Address realTo = Mac48Address::ConvertFrom (dest);
  mac->NotifyTx (packet);
  mac->Enqueue (packet, realTo);
  return true;
}

void
WaveNetDevice::ChangeAddress (Address newAddress)
{
  NS_LOG_FUNCTION (this << newAddress);
  Address oldAddress = GetAddress ();
  if (newAddress == oldAddress)
    {
      return;
    }
  SetAddress (newAddress);
  // Since MAC address is changed, the MAC layer including multiple MAC entities should be reset
  // and internal MAC queues will be flushed.
  for (MacEntitiesI i = m_macEntities.begin (); i != m_macEntities.end (); ++i)
    {
      i->second->Reset ();
    }
  m_addressChange (oldAddress, newAddress);
}

void
WaveNetDevice::CancelTx (uint32_t channelNumber, enum AcIndex ac)
{
  if (IsAvailableChannel (channelNumber))
    {
      return;
    }
  Ptr<OcbWifiMac> mac = GetMac (channelNumber);
  mac->CancleTx (ac);
}

void
WaveNetDevice::SetChannelManager (Ptr<ChannelManager> channelManager)
{
  m_channelManager = channelManager;
}
Ptr<ChannelManager>
WaveNetDevice::GetChannelManager (void) const
{
  return m_channelManager;
}
void
WaveNetDevice::SetChannelScheduler (Ptr<ChannelScheduler> channelScheduler)
{
  m_channelScheduler = channelScheduler;
}
Ptr<ChannelScheduler>
WaveNetDevice::GetChannelScheduler (void) const
{
  return m_channelScheduler;
}
void
WaveNetDevice::SetChannelCoordinator (Ptr<ChannelCoordinator> channelCoordinator)
{
  m_channelCoordinator = channelCoordinator;
}
Ptr<ChannelCoordinator>
WaveNetDevice::GetChannelCoordinator (void) const
{
  return m_channelCoordinator;
}
void
WaveNetDevice::SetVsaManager (Ptr<VsaManager> vsaManager)
{
  m_vsaManager = vsaManager;
}
Ptr<VsaManager>
WaveNetDevice::GetVsaManager (void) const
{
  return m_vsaManager;
}

void
WaveNetDevice::SetIfIndex (const uint32_t index)
{
  m_ifIndex = index;
}
uint32_t
WaveNetDevice::GetIfIndex (void) const
{
  return m_ifIndex;
}
Ptr<Channel>
WaveNetDevice::GetChannel (void) const
{
  NS_ASSERT (!m_phyEntities.empty ());
  return GetPhy (0)->GetChannel ();
}
void
WaveNetDevice::SetAddress (Address address)
{
  for (MacEntitiesI i = m_macEntities.begin (); i != m_macEntities.end (); ++i)
    {
      i->second->SetAddress (Mac48Address::ConvertFrom (address));
    }
}
Address
WaveNetDevice::GetAddress (void) const
{
  return (GetMac (CCH))->GetAddress ();
}
bool
WaveNetDevice::SetMtu (const uint16_t mtu)
{
  if (mtu > MAX_MSDU_SIZE - LLC_SNAP_HEADER_LENGTH)
    {
      return false;
    }
  m_mtu = mtu;
  return true;
}
uint16_t
WaveNetDevice::GetMtu (void) const
{
  return m_mtu;
}
bool
WaveNetDevice::IsLinkUp (void) const
{
  // Different from WifiNetDevice::IsLinkUp, a WaveNetDevice device
  // is always link up so the m_linkup variable is true forever.
  // Even the device is in channel switch state, packets can still be queued.
  return true;
}
void
WaveNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
  NS_LOG_WARN ("WaveNetDevice is linkup forever, so this callback will be never called");
}
bool
WaveNetDevice::IsBroadcast (void) const
{
  return true;
}
Address
WaveNetDevice::GetBroadcast (void) const
{
  return Mac48Address::GetBroadcast ();
}
bool
WaveNetDevice::IsMulticast (void) const
{
  return true;
}
Address
WaveNetDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  return Mac48Address::GetMulticast (multicastGroup);
}
Address WaveNetDevice::GetMulticast (Ipv6Address addr) const
{
  return Mac48Address::GetMulticast (addr);
}
bool
WaveNetDevice::IsPointToPoint (void) const
{
  return false;
}
bool
WaveNetDevice::IsBridge (void) const
{
  return false;
}

bool
WaveNetDevice::Send (Ptr<Packet> packet, const Address& dest, uint16_t protocol)
{
  NS_LOG_FUNCTION (this << packet << dest << protocol);
  if (m_txProfile == 0)
    {
      NS_LOG_DEBUG ("there is no tx profile registered for transmission");
      return false;
    }
  if (!m_channelScheduler->IsChannelAccessAssigned (m_txProfile->channelNumber))
    {
      NS_LOG_DEBUG ("there is no channel access assigned for channel " << m_txProfile->channelNumber);
      return false;
    }
  if (m_txProfile->dataRate == WifiMode () || m_txProfile->txPowerLevel == 8)
    {
      // let MAC layer itself determine tx parameters.
      NS_LOG_DEBUG ("High layer does not want to control tx parameters.");
    }
  else
    {
      WifiTxVector txVector;
      txVector.SetTxPowerLevel (m_txProfile->txPowerLevel);
      txVector.SetMode (m_txProfile->dataRate);
      txVector.SetPreambleType (m_txProfile->preamble);
      HigherLayerTxVectorTag tag = HigherLayerTxVectorTag (txVector, m_txProfile->adaptable);
      packet->AddPacketTag (tag);
    }

  LlcSnapHeader llc;
  llc.SetType (protocol);
  packet->AddHeader (llc);

  // qos tag is already inserted into the packet by high layer  or with default value 7 if high layer forget it.
  Ptr<WifiMac> mac = GetMac (m_txProfile->channelNumber);
  Mac48Address realTo = Mac48Address::ConvertFrom (dest);
  mac->NotifyTx (packet);
  mac->Enqueue (packet, realTo);
  return true;
}

bool
WaveNetDevice::NeedsArp (void) const
{
  // Whether NeedsArp or not?
  // For IP-based packets , yes; For WSMP packets, no;
  // so return true always.
  return true;
}
void
WaveNetDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  m_forwardUp = cb;
}

bool
WaveNetDevice::IsAvailableChannel (uint32_t channelNumber) const
{
  if (!ChannelManager::IsWaveChannel (channelNumber))
    {
      NS_LOG_DEBUG ("this is no a valid WAVE channel for channel " << channelNumber);
      return false;
    }
  if (m_macEntities.find (channelNumber) == m_macEntities.end ())
    {
      NS_LOG_DEBUG ("this is no available WAVE entity  for channel " << channelNumber);
      return false;
    }
  return true;
}

void
WaveNetDevice::ForwardUp (Ptr<const Packet> packet, Mac48Address from, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << from << to);
  Ptr<Packet> copy = packet->Copy ();
  LlcSnapHeader llc;
  copy->RemoveHeader (llc);
  enum NetDevice::PacketType type;
  if (to.IsBroadcast ())
    {
      type = NetDevice::PACKET_BROADCAST;
    }
  else if (to.IsGroup ())
    {
      type = NetDevice::PACKET_MULTICAST;
    }
  else if (to == GetAddress ())
    {
      type = NetDevice::PACKET_HOST;
    }
  else
    {
      type = NetDevice::PACKET_OTHERHOST;
    }

  if (type != NetDevice::PACKET_OTHERHOST)
    {
      // currently we cannot know from which MAC entity the packet is received,
      // so we use the MAC entity for CCH as it receives this packet.
      Ptr<OcbWifiMac> mac = GetMac (CCH);
      mac->NotifyRx (copy);
      m_forwardUp (this, copy, llc.GetType (), from);
    }

  if (!m_promiscRx.IsNull ())
    {
      // currently we cannot know from which MAC entity the packet is received,
      // so we use the MAC entity for CCH as it receives this packet.
      Ptr<OcbWifiMac> mac = GetMac (CCH);
      mac->NotifyPromiscRx (copy);
      m_promiscRx (this, copy, llc.GetType (), from, to, type);
    }
}

bool
WaveNetDevice::SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocol)
{
  NS_LOG_FUNCTION (this << packet << source << dest << protocol);
  return false;
}

void
WaveNetDevice::SetPromiscReceiveCallback (PromiscReceiveCallback cb)
{
  m_promiscRx = cb;
  for (MacEntitiesI i = m_macEntities.begin (); i != m_macEntities.end (); ++i)
    {
      i->second->SetPromisc ();
    }
}

bool
WaveNetDevice::SupportsSendFrom (void) const
{
  return (GetMac (CCH))->SupportsSendFrom ();
}

} // namespace ns3
