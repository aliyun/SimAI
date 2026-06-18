/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/log.h"
#include "ns3/abort.h"
#include "multi-user-scheduler.h"
#include "ns3/qos-txop.h"
#include "he-configuration.h"
#include "he-frame-exchange-manager.h"
#include "ns3/wifi-protection.h"
#include "ns3/wifi-acknowledgment.h"
#include "ns3/wifi-mac-trailer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MultiUserScheduler");

NS_OBJECT_ENSURE_REGISTERED (MultiUserScheduler);

TypeId
MultiUserScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MultiUserScheduler")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
  ;
  return tid;
}

MultiUserScheduler::MultiUserScheduler ()
{
}

MultiUserScheduler::~MultiUserScheduler ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
MultiUserScheduler::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_apMac = 0;
  m_heFem = 0;
  m_edca = 0;
  m_dlInfo.psduMap.clear ();
  m_dlInfo.txParams.Clear ();
  m_ulInfo.txParams.Clear ();
  Object::DoDispose ();
}

void
MultiUserScheduler::NotifyNewAggregate ()
{
  NS_LOG_FUNCTION (this);
  if (m_apMac == 0)
    {
      Ptr<ApWifiMac> apMac = this->GetObject<ApWifiMac> ();
      //verify that it's a valid AP mac and that
      //the AP mac was not set before
      if (apMac != 0)
        {
          this->SetWifiMac (apMac);
        }
    }
  Object::NotifyNewAggregate ();
}

void
MultiUserScheduler::DoInitialize (void)
{
  // compute the size in bytes of 8 QoS Null frames. It can be used by subclasses
  // when responding to a BSRP Trigger Frame
  WifiMacHeader header;
  header.SetType (WIFI_MAC_QOSDATA_NULL);
  header.SetDsTo ();
  header.SetDsNotFrom ();
  uint32_t headerSize = header.GetSerializedSize ();

  m_sizeOf8QosNull = 0;
  for (uint8_t i = 0; i < 8; i++)
    {
      m_sizeOf8QosNull = MpduAggregator::GetSizeIfAggregated (headerSize + WIFI_MAC_FCS_LENGTH, m_sizeOf8QosNull);
    }
}

void
MultiUserScheduler::SetWifiMac (Ptr<ApWifiMac> mac)
{
  NS_LOG_FUNCTION (this << mac);
  m_apMac = mac;

  // When VHT DL MU-MIMO will be supported, we will have to lower this requirement
  // and allow a Multi-user scheduler to be installed on a VHT AP.
  NS_ABORT_MSG_IF (m_apMac == 0 || m_apMac->GetHeConfiguration () == 0,
                   "MultiUserScheduler can only be installed on HE APs");

  m_heFem = DynamicCast<HeFrameExchangeManager> (m_apMac->GetFrameExchangeManager ());
  m_heFem->SetMultiUserScheduler (this);
}

Ptr<WifiRemoteStationManager>
MultiUserScheduler::GetWifiRemoteStationManager (void) const
{
  return m_apMac->GetWifiRemoteStationManager ();
}

MultiUserScheduler::TxFormat
MultiUserScheduler::NotifyAccessGranted (Ptr<QosTxop> edca, Time availableTime, bool initialFrame)
{
  NS_LOG_FUNCTION (this << edca << availableTime << initialFrame);

  m_edca = edca;
  m_availableTime = availableTime;
  m_initialFrame = initialFrame;

  TxFormat txFormat = SelectTxFormat ();

  if (txFormat == DL_MU_TX)
    {
      m_dlInfo = ComputeDlMuInfo ();
    }
  else if (txFormat == UL_MU_TX)
    {
      NS_ABORT_MSG_IF (m_heFem == 0, "UL MU PPDUs are only supported by HE APs");
      m_ulInfo = ComputeUlMuInfo ();
      CheckTriggerFrame ();
    }

  if (txFormat != NO_TX)
    {
      m_lastTxFormat = txFormat;
    }
  return txFormat;
}

MultiUserScheduler::TxFormat
MultiUserScheduler::GetLastTxFormat (void) const
{
  return m_lastTxFormat;
}

MultiUserScheduler::DlMuInfo&
MultiUserScheduler::GetDlMuInfo (void)
{
  NS_ABORT_MSG_IF (m_lastTxFormat != DL_MU_TX, "Next transmission is not DL MU");

#ifdef NS3_BUILD_PROFILE_DEBUG
  // check that all the addressed stations support HE
  for (auto& psdu : m_dlInfo.psduMap)
    {
      NS_ABORT_MSG_IF (!GetWifiRemoteStationManager ()->GetHeSupported (psdu.second->GetAddr1 ()),
                       "Station " << psdu.second->GetAddr1 () << " does not support HE");
    }
#endif

  return m_dlInfo;
}

MultiUserScheduler::UlMuInfo&
MultiUserScheduler::GetUlMuInfo (void)
{
  NS_ABORT_MSG_IF (m_lastTxFormat != UL_MU_TX, "Next transmission is not UL MU");

  return m_ulInfo;
}

void
MultiUserScheduler::CheckTriggerFrame (void)
{
  NS_LOG_FUNCTION (this);

  // Set the CS Required subfield to true, unless the UL Length subfield is less
  // than or equal to 76 (see Section 26.5.2.5 of 802.11ax-2021)
  m_ulInfo.trigger.SetCsRequired (m_ulInfo.trigger.GetUlLength () > 76);

  m_heFem->SetTargetRssi (m_ulInfo.trigger);
}

} //namespace ns3
