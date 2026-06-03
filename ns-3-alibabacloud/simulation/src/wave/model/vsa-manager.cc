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
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/wifi-phy.h"
#include "vsa-manager.h"
#include "higher-tx-tag.h"
#include "wave-net-device.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("VsaManager");

NS_OBJECT_ENSURE_REGISTERED (VsaManager);

/// OI bytes for use in organization identifier
const static uint8_t oi_bytes_1609[5] = {0x00, 0x50, 0xC2, 0x4A, 0x40};
/// OI for IEEE standard 1609
const static OrganizationIdentifier oi_1609 = OrganizationIdentifier (oi_bytes_1609, 5);

TypeId
VsaManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::VsaManager")
    .SetParent<Object> ()
    .SetGroupName ("Wave")
    .AddConstructor<VsaManager> ()
  ;
  return tid;
}

VsaManager::VsaManager (void)
  : m_device (0)
{
  m_vsaReceived = MakeNullCallback<bool, Ptr<const Packet>,const Address &, uint32_t, uint32_t> ();
}

VsaManager::~VsaManager (void)
{

}

void
VsaManager::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  RemoveAll ();
  m_device = 0;
}

void
VsaManager::DoInitialize (void)
{
  std::map<uint32_t, Ptr<OcbWifiMac> > macs = m_device->GetMacs ();
  for (std::map<uint32_t, Ptr<OcbWifiMac> >::iterator i = macs.begin (); i != macs.end (); ++i)
    {
      i->second->AddReceiveVscCallback (oi_1609, MakeCallback (&VsaManager::ReceiveVsc, this));
    }
}

void
VsaManager::SetWaveNetDevice (Ptr<WaveNetDevice> device)
{
  NS_LOG_FUNCTION (this << device);
  m_device = device;
}

void
VsaManager::SendVsa (const VsaInfo & vsaInfo)
{
  NS_LOG_FUNCTION (this << &vsaInfo);
  OrganizationIdentifier oi;
  if (vsaInfo.oi.IsNull ())
    {
      // refer to 1609.4-2010 chapter 6.4.1.1
      uint8_t oibytes[5] = {0x00, 0x50, 0xC2, 0x4A, 0x40};
      oibytes[4] |= (vsaInfo.managementId & 0x0f);
      oi = OrganizationIdentifier (oibytes, 5);
    }
  else
    {
      oi = vsaInfo.oi;
    }

  // if destination MAC address is the unicast address or  repeat rate is 0,
  // then only single one VSA frame is to be sent.
  if (vsaInfo.peer.IsGroup () && (vsaInfo.repeatRate != 0))
    {
      VsaWork *vsa = new VsaWork ();
      vsa->sentInterval = vsaInfo.sendInterval;
      vsa->channelNumber = vsaInfo.channelNumber;
      vsa->peer = vsaInfo.peer;
      vsa->repeatPeriod = MilliSeconds (VSA_REPEAT_PERIOD * 1000 / vsaInfo.repeatRate);
      vsa->vsc = vsaInfo.vsc;
      vsa->oi = oi;
      vsa->repeat =  Simulator::Schedule (vsa->repeatPeriod, &VsaManager::DoRepeat, this, vsa);
      m_vsas.push_back (vsa);
    }
  DoSendVsa (vsaInfo.sendInterval, vsaInfo.channelNumber, vsaInfo.vsc->Copy (), oi, vsaInfo.peer);
}

void
VsaManager::DoRepeat (VsaWork *vsa)
{
  NS_LOG_FUNCTION (this << vsa);
  vsa->repeat =  Simulator::Schedule (vsa->repeatPeriod, &VsaManager::DoRepeat, this, vsa);
  DoSendVsa (vsa->sentInterval, vsa->channelNumber, vsa->vsc->Copy (), vsa->oi, vsa->peer);
}

void
VsaManager::DoSendVsa (enum VsaTransmitInterval  interval, uint32_t channel,
                       Ptr<Packet> vsc, OrganizationIdentifier oi, Mac48Address peer)
{
  NS_LOG_FUNCTION (this << interval << channel << vsc << oi << peer);
  NS_ASSERT (m_device != 0);
  Ptr<ChannelCoordinator> coordinator = m_device->GetChannelCoordinator ();
  Ptr<ChannelScheduler> scheduler = m_device->GetChannelScheduler ();
  Ptr<ChannelManager> manager = m_device->GetChannelManager ();

  // if the request is for transmitting in SCH Interval (or CCH Interval), but currently
  // is not in SCH Interval (or CCH Interval) and , then the WAVE device  will wait
  // some time to insert this VSA frame into MAC internal queue.
  // if the request is for transmitting in any channel interval, then the WAVE device
  // insert this VSA frame into MAC internal queue immediately.
  if (interval == VSA_TRANSMIT_IN_SCHI)
    {
      Time wait = coordinator->NeedTimeToSchInterval ();
      if (wait != Seconds (0))
        {
          Simulator::Schedule (wait, &VsaManager::DoSendVsa, this,
                               interval, channel, vsc, oi, peer);
          return;
        }
    }
  else if (interval == VSA_TRANSMIT_IN_CCHI)
    {
      Time wait = coordinator->NeedTimeToCchInterval ();
      if (wait != Seconds (0))
        {
          Simulator::Schedule (wait, &VsaManager::DoSendVsa, this,
                               interval, channel, vsc, oi, peer);
          return;
        }
    }
  else
    {
      NS_ASSERT (interval == VSA_TRANSMIT_IN_BOTHI);
      // do nothing here, since VSA_IN_BOTHI allows to sent VSA frames in any interval.
    }

  if (!scheduler->IsChannelAccessAssigned (channel))
    {
      NS_LOG_DEBUG ("there is no channel access assigned for channel " << channel);
      return;
    }

  // refer to 1609.4-2010 chapter 5.4.1
  // Management frames are assigned the highest AC (AC_VO).
  SocketPriorityTag priorityTag;
  priorityTag.SetPriority (7);
  vsc->AddPacketTag (priorityTag);

  WifiTxVector txVector;
  txVector.SetChannelWidth (10);
  txVector.SetTxPowerLevel (manager->GetManagementPowerLevel (channel));
  txVector.SetMode (manager->GetManagementDataRate (channel));
  txVector.SetPreambleType (manager->GetManagementPreamble (channel));
  HigherLayerTxVectorTag tag = HigherLayerTxVectorTag (txVector, manager->GetManagementAdaptable (channel));
  vsc->AddPacketTag (tag);

  Ptr<OcbWifiMac> mac = m_device->GetMac (channel);
  mac->SendVsc (vsc, peer, oi);
}

void
VsaManager::RemoveAll (void)
{
  NS_LOG_FUNCTION (this);
  for (std::vector<VsaWork *>::iterator i = m_vsas.begin ();
       i != m_vsas.end (); ++i)
    {
      if (!(*i)->repeat.IsExpired ())
        {
          (*i)->repeat.Cancel ();
        }
      (*i)->vsc = 0;
      delete (*i);
    }
  m_vsas.clear ();
}

void
VsaManager::RemoveByChannel (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  for (std::vector<VsaWork *>::iterator i = m_vsas.begin ();
       i != m_vsas.end (); )
    {
      if ((*i)->channelNumber == channelNumber)
        {
          if (!(*i)->repeat.IsExpired ())
            {
              (*i)->repeat.Cancel ();
            }
          (*i)->vsc = 0;
          delete (*i);
          i = m_vsas.erase (i);
        }
      else
        {
          ++i;
        }
    }
}


void
VsaManager::RemoveByOrganizationIdentifier (const OrganizationIdentifier &oi)
{
  NS_LOG_FUNCTION (this << oi);
  for (std::vector<VsaWork *>::iterator i = m_vsas.begin ();
       i != m_vsas.end (); )
    {
      if ((*i)->oi == oi)
        {
          if (!(*i)->repeat.IsExpired ())
            {
              (*i)->repeat.Cancel ();
            }
          (*i)->vsc = 0;
          delete (*i);
          i = m_vsas.erase (i);
        }
      else
        {
          ++i;
        }
    }
}

void
VsaManager::SetWaveVsaCallback (Callback<bool, Ptr<const Packet>,const Address &, uint32_t, uint32_t> vsaCallback)
{
  NS_LOG_FUNCTION (this);
  m_vsaReceived = vsaCallback;
}

bool
VsaManager::ReceiveVsc (Ptr<WifiMac> mac, const OrganizationIdentifier &oi, Ptr<const Packet> vsc, const Address &src)
{
  NS_LOG_FUNCTION (this << mac << oi << vsc << src);
  NS_ASSERT (oi == oi_1609);
  if (m_vsaReceived.IsNull ())
    {
      return true;
    }
  uint32_t channelNumber = mac->GetWifiPhy ()->GetChannelNumber ();
  uint32_t managementId = oi.GetManagementId ();
  return m_vsaReceived (vsc, src, managementId, channelNumber);
}
} // namespace ns3
