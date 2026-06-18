/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006, 2009 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "qos-txop.h"
#include "sta-wifi-mac.h"
#include "wifi-phy.h"
#include "mgt-headers.h"
#include "snr-tag.h"
#include "wifi-net-device.h"
#include "ns3/ht-configuration.h"
#include "ns3/he-configuration.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("StaWifiMac");

NS_OBJECT_ENSURE_REGISTERED (StaWifiMac);

TypeId
StaWifiMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::StaWifiMac")
    .SetParent<WifiMac> ()
    .SetGroupName ("Wifi")
    .AddConstructor<StaWifiMac> ()
    .AddAttribute ("ProbeRequestTimeout", "The duration to actively probe the channel.",
                   TimeValue (Seconds (0.05)),
                   MakeTimeAccessor (&StaWifiMac::m_probeRequestTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("WaitBeaconTimeout", "The duration to dwell on a channel while passively scanning for beacon",
                   TimeValue (MilliSeconds (120)),
                   MakeTimeAccessor (&StaWifiMac::m_waitBeaconTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("AssocRequestTimeout", "The interval between two consecutive association request attempts.",
                   TimeValue (Seconds (0.5)),
                   MakeTimeAccessor (&StaWifiMac::m_assocRequestTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("MaxMissedBeacons",
                   "Number of beacons which much be consecutively missed before "
                   "we attempt to restart association.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&StaWifiMac::m_maxMissedBeacons),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("ActiveProbing",
                   "If true, we send probe requests. If false, we don't."
                   "NOTE: if more than one STA in your simulation is using active probing, "
                   "you should enable it at a different simulation time for each STA, "
                   "otherwise all the STAs will start sending probes at the same time resulting in collisions. "
                   "See bug 1060 for more info.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&StaWifiMac::SetActiveProbing, &StaWifiMac::GetActiveProbing),
                   MakeBooleanChecker ())
    .AddTraceSource ("Assoc", "Associated with an access point.",
                     MakeTraceSourceAccessor (&StaWifiMac::m_assocLogger),
                     "ns3::Mac48Address::TracedCallback")
    .AddTraceSource ("DeAssoc", "Association with an access point lost.",
                     MakeTraceSourceAccessor (&StaWifiMac::m_deAssocLogger),
                     "ns3::Mac48Address::TracedCallback")
    .AddTraceSource ("BeaconArrival",
                     "Time of beacons arrival from associated AP",
                     MakeTraceSourceAccessor (&StaWifiMac::m_beaconArrival),
                     "ns3::Time::TracedCallback")
  ;
  return tid;
}

StaWifiMac::StaWifiMac ()
  : m_state (UNASSOCIATED),
    m_aid (0),
    m_waitBeaconEvent (),
    m_probeRequestEvent (),
    m_assocRequestEvent (),
    m_beaconWatchdogEnd (Seconds (0))
{
  NS_LOG_FUNCTION (this);

  //Let the lower layers know that we are acting as a non-AP STA in
  //an infrastructure BSS.
  SetTypeOfStation (STA);
}

void
StaWifiMac::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  StartScanning ();
}

StaWifiMac::~StaWifiMac ()
{
  NS_LOG_FUNCTION (this);
}

uint16_t
StaWifiMac::GetAssociationId (void) const
{
  NS_ASSERT_MSG (IsAssociated (), "This station is not associated to any AP");
  return m_aid;
}

void
StaWifiMac::SetActiveProbing (bool enable)
{
  NS_LOG_FUNCTION (this << enable);
  m_activeProbing = enable;
  if (m_state == WAIT_PROBE_RESP || m_state == WAIT_BEACON)
    {
      NS_LOG_DEBUG ("STA is still scanning, reset scanning process");
      StartScanning ();
    }
}

bool
StaWifiMac::GetActiveProbing (void) const
{
  return m_activeProbing;
}

void
StaWifiMac::SetWifiPhy (const Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  WifiMac::SetWifiPhy (phy);
  GetWifiPhy ()->SetCapabilitiesChangedCallback (MakeCallback (&StaWifiMac::PhyCapabilitiesChanged, this));
}

void
StaWifiMac::SendProbeRequest (void)
{
  NS_LOG_FUNCTION (this);
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_PROBE_REQUEST);
  hdr.SetAddr1 (Mac48Address::GetBroadcast ());
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (Mac48Address::GetBroadcast ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  Ptr<Packet> packet = Create<Packet> ();
  MgtProbeRequestHeader probe;
  probe.SetSsid (GetSsid ());
  probe.SetSupportedRates (GetSupportedRates ());
  if (GetHtSupported ())
    {
      probe.SetExtendedCapabilities (GetExtendedCapabilities ());
      probe.SetHtCapabilities (GetHtCapabilities ());
    }
  if (GetVhtSupported ())
    {
      probe.SetVhtCapabilities (GetVhtCapabilities ());
    }
  if (GetHeSupported ())
    {
      probe.SetHeCapabilities (GetHeCapabilities ());
    }
  packet->AddHeader (probe);

  if (!GetQosSupported ())
    {
      GetTxop ()->Queue (packet, hdr);
    }
  // "A QoS STA that transmits a Management frame determines access category used
  // for medium access in transmission of the Management frame as follows
  // (If dot11QMFActivated is false or not present)
  // — If the Management frame is individually addressed to a non-QoS STA, category
  //   AC_BE should be selected.
  // — If category AC_BE was not selected by the previous step, category AC_VO
  //   shall be selected." (Sec. 10.2.3.2 of 802.11-2020)
  else
    {
      GetVOQueue ()->Queue (packet, hdr);
    }
}

void
StaWifiMac::SendAssociationRequest (bool isReassoc)
{
  NS_LOG_FUNCTION (this << GetBssid () << isReassoc);
  WifiMacHeader hdr;
  hdr.SetType (isReassoc ? WIFI_MAC_MGT_REASSOCIATION_REQUEST : WIFI_MAC_MGT_ASSOCIATION_REQUEST);
  hdr.SetAddr1 (GetBssid ());
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (GetBssid ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  Ptr<Packet> packet = Create<Packet> ();
  if (!isReassoc)
    {
      MgtAssocRequestHeader assoc;
      assoc.SetSsid (GetSsid ());
      assoc.SetSupportedRates (GetSupportedRates ());
      assoc.SetCapabilities (GetCapabilities ());
      assoc.SetListenInterval (0);
      if (GetHtSupported ())
        {
          assoc.SetExtendedCapabilities (GetExtendedCapabilities ());
          assoc.SetHtCapabilities (GetHtCapabilities ());
        }
      if (GetVhtSupported ())
        {
          assoc.SetVhtCapabilities (GetVhtCapabilities ());
        }
      if (GetHeSupported ())
        {
          assoc.SetHeCapabilities (GetHeCapabilities ());
        }
      packet->AddHeader (assoc);
    }
  else
    {
      MgtReassocRequestHeader reassoc;
      reassoc.SetCurrentApAddress (GetBssid ());
      reassoc.SetSsid (GetSsid ());
      reassoc.SetSupportedRates (GetSupportedRates ());
      reassoc.SetCapabilities (GetCapabilities ());
      reassoc.SetListenInterval (0);
      if (GetHtSupported ())
        {
          reassoc.SetExtendedCapabilities (GetExtendedCapabilities ());
          reassoc.SetHtCapabilities (GetHtCapabilities ());
        }
      if (GetVhtSupported ())
        {
          reassoc.SetVhtCapabilities (GetVhtCapabilities ());
        }
      if (GetHeSupported ())
        {
          reassoc.SetHeCapabilities (GetHeCapabilities ());
        }
      packet->AddHeader (reassoc);
    }

  if (!GetQosSupported ())
    {
      GetTxop ()->Queue (packet, hdr);
    }
  // "A QoS STA that transmits a Management frame determines access category used
  // for medium access in transmission of the Management frame as follows
  // (If dot11QMFActivated is false or not present)
  // — If the Management frame is individually addressed to a non-QoS STA, category
  //   AC_BE should be selected.
  // — If category AC_BE was not selected by the previous step, category AC_VO
  //   shall be selected." (Sec. 10.2.3.2 of 802.11-2020)
  else if (!GetWifiRemoteStationManager ()->GetQosSupported (GetBssid ()))
    {
      GetBEQueue ()->Queue (packet, hdr);
    }
  else
    {
      GetVOQueue ()->Queue (packet, hdr);
    }

  if (m_assocRequestEvent.IsRunning ())
    {
      m_assocRequestEvent.Cancel ();
    }
  m_assocRequestEvent = Simulator::Schedule (m_assocRequestTimeout,
                                             &StaWifiMac::AssocRequestTimeout, this);
}

void
StaWifiMac::TryToEnsureAssociated (void)
{
  NS_LOG_FUNCTION (this);
  switch (m_state)
    {
    case ASSOCIATED:
      return;
      break;
    case WAIT_PROBE_RESP:
      /* we have sent a probe request earlier so we
         do not need to re-send a probe request immediately.
         We just need to wait until probe-request-timeout
         or until we get a probe response
       */
      break;
    case WAIT_BEACON:
      /* we have initiated passive scanning, continue to wait
         and gather beacons
       */
      break;
    case UNASSOCIATED:
      /* we were associated but we missed a bunch of beacons
       * so we should assume we are not associated anymore.
       * We try to initiate a scan now.
       */
      m_linkDown ();
      StartScanning ();
      break;
    case WAIT_ASSOC_RESP:
      /* we have sent an association request so we do not need to
         re-send an association request right now. We just need to
         wait until either assoc-request-timeout or until
         we get an association response.
       */
      break;
    case REFUSED:
      /* we have sent an association request and received a negative
         association response. We wait until someone restarts an
         association with a given SSID.
       */
      break;
    }
}

void
StaWifiMac::StartScanning (void)
{
  NS_LOG_FUNCTION (this);
  m_candidateAps.clear ();
  if (m_probeRequestEvent.IsRunning ())
    {
      m_probeRequestEvent.Cancel ();
    }
  if (m_waitBeaconEvent.IsRunning ())
    {
      m_waitBeaconEvent.Cancel ();
    }
  if (GetActiveProbing ())
    {
      SetState (WAIT_PROBE_RESP);
      SendProbeRequest ();
      m_probeRequestEvent = Simulator::Schedule (m_probeRequestTimeout,
                                                 &StaWifiMac::ScanningTimeout,
                                                 this);
    }
  else
    {
      SetState (WAIT_BEACON);
      m_waitBeaconEvent = Simulator::Schedule (m_waitBeaconTimeout,
                                               &StaWifiMac::ScanningTimeout,
                                               this);
    }
}

void
StaWifiMac::ScanningTimeout (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_candidateAps.empty ())
    {
      ApInfo bestAp = m_candidateAps.front();
      m_candidateAps.erase(m_candidateAps.begin ());
      NS_LOG_DEBUG ("Attempting to associate with BSSID " << bestAp.m_bssid);
      Time beaconInterval;
      if (bestAp.m_activeProbing)
        {
          UpdateApInfoFromProbeResp (bestAp.m_probeResp, bestAp.m_apAddr, bestAp.m_bssid);
          beaconInterval = MicroSeconds (bestAp.m_probeResp.GetBeaconIntervalUs ());
        }
      else
        {
          UpdateApInfoFromBeacon (bestAp.m_beacon, bestAp.m_apAddr, bestAp.m_bssid);
          beaconInterval = MicroSeconds (bestAp.m_beacon.GetBeaconIntervalUs ());
        }

      Time delay = beaconInterval * m_maxMissedBeacons;
      RestartBeaconWatchdog (delay);
      SetState (WAIT_ASSOC_RESP);
      SendAssociationRequest (false);
    }
  else
    {
      NS_LOG_DEBUG ("Exhausted list of candidate AP; restart scanning");
      StartScanning ();
    }
}

void
StaWifiMac::AssocRequestTimeout (void)
{
  NS_LOG_FUNCTION (this);
  SetState (WAIT_ASSOC_RESP);
  SendAssociationRequest (false);
}

void
StaWifiMac::MissedBeacons (void)
{
  NS_LOG_FUNCTION (this);
  if (m_beaconWatchdogEnd > Simulator::Now ())
    {
      if (m_beaconWatchdog.IsRunning ())
        {
          m_beaconWatchdog.Cancel ();
        }
      m_beaconWatchdog = Simulator::Schedule (m_beaconWatchdogEnd - Simulator::Now (),
                                              &StaWifiMac::MissedBeacons, this);
      return;
    }
  NS_LOG_DEBUG ("beacon missed");
  // We need to switch to the UNASSOCIATED state. However, if we are receiving
  // a frame, wait until the RX is completed (otherwise, crashes may occur if
  // we are receiving a MU frame because its reception requires the STA-ID)
  Time delay = Seconds (0);
  if (GetWifiPhy ()->IsStateRx ())
    {
      delay = GetWifiPhy ()->GetDelayUntilIdle ();
    }
  Simulator::Schedule (delay, &StaWifiMac::Disassociated, this);
}

void
StaWifiMac::Disassociated (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("Set state to UNASSOCIATED and start scanning");
  SetState (UNASSOCIATED);
  TryToEnsureAssociated ();
}

void
StaWifiMac::RestartBeaconWatchdog (Time delay)
{
  NS_LOG_FUNCTION (this << delay);
  m_beaconWatchdogEnd = std::max (Simulator::Now () + delay, m_beaconWatchdogEnd);
  if (Simulator::GetDelayLeft (m_beaconWatchdog) < delay
      && m_beaconWatchdog.IsExpired ())
    {
      NS_LOG_DEBUG ("really restart watchdog.");
      m_beaconWatchdog = Simulator::Schedule (delay, &StaWifiMac::MissedBeacons, this);
    }
}

bool
StaWifiMac::IsAssociated (void) const
{
  return m_state == ASSOCIATED;
}

bool
StaWifiMac::IsWaitAssocResp (void) const
{
  return m_state == WAIT_ASSOC_RESP;
}

bool
StaWifiMac::CanForwardPacketsTo (Mac48Address to) const
{
  return (IsAssociated ());
}

void
StaWifiMac::Enqueue (Ptr<Packet> packet, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << to);
  if (!CanForwardPacketsTo (to))
    {
      NotifyTxDrop (packet);
      TryToEnsureAssociated ();
      return;
    }
  WifiMacHeader hdr;

  //If we are not a QoS AP then we definitely want to use AC_BE to
  //transmit the packet. A TID of zero will map to AC_BE (through \c
  //QosUtilsMapTidToAc()), so we use that as our default here.
  uint8_t tid = 0;

  //For now, an AP that supports QoS does not support non-QoS
  //associations, and vice versa. In future the AP model should
  //support simultaneously associated QoS and non-QoS STAs, at which
  //point there will need to be per-association QoS state maintained
  //by the association state machine, and consulted here.
  if (GetQosSupported ())
    {
      hdr.SetType (WIFI_MAC_QOSDATA);
      hdr.SetQosAckPolicy (WifiMacHeader::NORMAL_ACK);
      hdr.SetQosNoEosp ();
      hdr.SetQosNoAmsdu ();
      //Transmission of multiple frames in the same TXOP is not
      //supported for now
      hdr.SetQosTxopLimit (0);

      //Fill in the QoS control field in the MAC header
      tid = QosUtilsGetTidForPacket (packet);
      //Any value greater than 7 is invalid and likely indicates that
      //the packet had no QoS tag, so we revert to zero, which'll
      //mean that AC_BE is used.
      if (tid > 7)
        {
          tid = 0;
        }
      hdr.SetQosTid (tid);
    }
  else
    {
      hdr.SetType (WIFI_MAC_DATA);
    }
  if (GetQosSupported ())
    {
      hdr.SetNoOrder (); // explicitly set to 0 for the time being since HT control field is not yet implemented (set it to 1 when implemented)
    }

  hdr.SetAddr1 (GetBssid ());
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (to);
  hdr.SetDsNotFrom ();
  hdr.SetDsTo ();

  if (GetQosSupported ())
    {
      //Sanity check that the TID is valid
      NS_ASSERT (tid < 8);
      GetQosTxop (tid)->Queue (packet, hdr);
    }
  else
    {
      GetTxop ()->Queue (packet, hdr);
    }
}

void
StaWifiMac::Receive (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  const WifiMacHeader* hdr = &mpdu->GetHeader ();
  Ptr<const Packet> packet = mpdu->GetPacket ();
  NS_ASSERT (!hdr->IsCtl ());
  if (hdr->GetAddr3 () == GetAddress ())
    {
      NS_LOG_LOGIC ("packet sent by us.");
      return;
    }
  else if (hdr->GetAddr1 () != GetAddress ()
           && !hdr->GetAddr1 ().IsGroup ())
    {
      NS_LOG_LOGIC ("packet is not for us");
      NotifyRxDrop (packet);
      return;
    }
  if (hdr->IsData ())
    {
      if (!IsAssociated ())
        {
          NS_LOG_LOGIC ("Received data frame while not associated: ignore");
          NotifyRxDrop (packet);
          return;
        }
      if (!(hdr->IsFromDs () && !hdr->IsToDs ()))
        {
          NS_LOG_LOGIC ("Received data frame not from the DS: ignore");
          NotifyRxDrop (packet);
          return;
        }
      if (hdr->GetAddr2 () != GetBssid ())
        {
          NS_LOG_LOGIC ("Received data frame not from the BSS we are associated with: ignore");
          NotifyRxDrop (packet);
          return;
        }
      if (hdr->IsQosData ())
        {
          if (hdr->IsQosAmsdu ())
            {
              NS_ASSERT (hdr->GetAddr3 () == GetBssid ());
              DeaggregateAmsduAndForward (mpdu);
              packet = 0;
            }
          else
            {
              ForwardUp (packet, hdr->GetAddr3 (), hdr->GetAddr1 ());
            }
        }
      else if (hdr->HasData ())
        {
          ForwardUp (packet, hdr->GetAddr3 (), hdr->GetAddr1 ());
        }
      return;
    }
  else if (hdr->IsProbeReq ()
           || hdr->IsAssocReq ()
           || hdr->IsReassocReq ())
    {
      //This is a frame aimed at an AP, so we can safely ignore it.
      NotifyRxDrop (packet);
      return;
    }
  else if (hdr->IsBeacon ())
    {
      NS_LOG_DEBUG ("Beacon received");
      MgtBeaconHeader beacon;
      Ptr<Packet> copy = packet->Copy ();
      copy->RemoveHeader (beacon);
      CapabilityInformation capabilities = beacon.GetCapabilities ();
      NS_ASSERT (capabilities.IsEss ());
      bool goodBeacon = false;
      if (GetSsid ().IsBroadcast ()
          || beacon.GetSsid ().IsEqual (GetSsid ()))
        {
          NS_LOG_LOGIC ("Beacon is for our SSID");
          goodBeacon = true;
        }
      SupportedRates rates = beacon.GetSupportedRates ();
      bool bssMembershipSelectorMatch = false;
      auto selectorList = GetWifiPhy ()->GetBssMembershipSelectorList ();
      for (const auto & selector : selectorList)
        {
          if (rates.IsBssMembershipSelectorRate (selector))
            {
              NS_LOG_LOGIC ("Beacon is matched to our BSS membership selector");
              bssMembershipSelectorMatch = true;
            }
        }
      if (selectorList.size () > 0 && bssMembershipSelectorMatch == false)
        {
          NS_LOG_LOGIC ("No match for BSS membership selector");
          goodBeacon = false;
        }
      if ((IsWaitAssocResp () || IsAssociated ()) && hdr->GetAddr3 () != GetBssid ())
        {
          NS_LOG_LOGIC ("Beacon is not for us");
          goodBeacon = false;
        }
      if (goodBeacon && m_state == ASSOCIATED)
        {
          m_beaconArrival (Simulator::Now ());
          Time delay = MicroSeconds (beacon.GetBeaconIntervalUs () * m_maxMissedBeacons);
          RestartBeaconWatchdog (delay);
          UpdateApInfoFromBeacon (beacon, hdr->GetAddr2 (), hdr->GetAddr3 ());
        }
      if (goodBeacon && m_state == WAIT_BEACON)
        {
          NS_LOG_DEBUG ("Beacon received while scanning from " << hdr->GetAddr2 ());
          SnrTag snrTag;
          bool removed = copy->RemovePacketTag (snrTag);
          NS_ASSERT (removed);
          ApInfo apInfo;
          apInfo.m_apAddr = hdr->GetAddr2 ();
          apInfo.m_bssid = hdr->GetAddr3 ();
          apInfo.m_activeProbing = false;
          apInfo.m_snr = snrTag.Get ();
          apInfo.m_beacon = beacon;
          UpdateCandidateApList (apInfo);
        }
      return;
    }
  else if (hdr->IsProbeResp ())
    {
      if (m_state == WAIT_PROBE_RESP)
        {
          NS_LOG_DEBUG ("Probe response received while scanning from " << hdr->GetAddr2 ());
          MgtProbeResponseHeader probeResp;
          Ptr<Packet> copy = packet->Copy ();
          copy->RemoveHeader (probeResp);
          if (!probeResp.GetSsid ().IsEqual (GetSsid ()))
            {
              NS_LOG_DEBUG ("Probe response is not for our SSID");
              return;
            }
          SnrTag snrTag;
          bool removed = copy->RemovePacketTag (snrTag);
          NS_ASSERT (removed);
          ApInfo apInfo;
          apInfo.m_apAddr = hdr->GetAddr2 ();
          apInfo.m_bssid = hdr->GetAddr3 ();
          apInfo.m_activeProbing = true;
          apInfo.m_snr = snrTag.Get ();
          apInfo.m_probeResp = probeResp;
          UpdateCandidateApList (apInfo);
        }
      return;
    }
  else if (hdr->IsAssocResp () || hdr->IsReassocResp ())
    {
      if (m_state == WAIT_ASSOC_RESP)
        {
          MgtAssocResponseHeader assocResp;
          packet->PeekHeader (assocResp);
          if (m_assocRequestEvent.IsRunning ())
            {
              m_assocRequestEvent.Cancel ();
            }
          if (assocResp.GetStatusCode ().IsSuccess ())
            {
              SetState (ASSOCIATED);
              m_aid = assocResp.GetAssociationId ();
              if (hdr->IsReassocResp ())
                {
                  NS_LOG_DEBUG ("reassociation done");
                }
              else
                {
                  NS_LOG_DEBUG ("association completed");
                }
              UpdateApInfoFromAssocResp (assocResp, hdr->GetAddr2 ());
              if (!m_linkUp.IsNull ())
                {
                  m_linkUp ();
                }
            }
          else
            {
              NS_LOG_DEBUG ("association refused");
              if (m_candidateAps.empty ())
                {
                  SetState (REFUSED);
                }
              else
                {
                  ScanningTimeout ();
                }
            }
        }
      return;
    }

  //Invoke the receive handler of our parent class to deal with any
  //other frames. Specifically, this will handle Block Ack-related
  //Management Action frames.
  WifiMac::Receive (Create<WifiMacQueueItem> (packet, *hdr));
}

void
StaWifiMac::UpdateCandidateApList (ApInfo newApInfo)
{
  NS_LOG_FUNCTION (this << newApInfo.m_bssid << newApInfo.m_apAddr << newApInfo.m_snr << newApInfo.m_activeProbing << newApInfo.m_beacon << newApInfo.m_probeResp);
  // Remove duplicate ApInfo entry
  for (std::vector<ApInfo>::iterator i = m_candidateAps.begin(); i != m_candidateAps.end(); ++i)
    {
      if (newApInfo.m_bssid == (*i).m_bssid)
        {
          m_candidateAps.erase(i);
          break;
        }
    }
  // Insert before the entry with lower SNR
  for (std::vector<ApInfo>::iterator i = m_candidateAps.begin(); i != m_candidateAps.end(); ++i)
    {
      if (newApInfo.m_snr > (*i).m_snr)
        {
          m_candidateAps.insert (i, newApInfo);
          return;
        }
    }
  // If new ApInfo is the lowest, insert at back
  m_candidateAps.push_back(newApInfo);
}

void
StaWifiMac::UpdateApInfoFromBeacon (MgtBeaconHeader beacon, Mac48Address apAddr, Mac48Address bssid)
{
  NS_LOG_FUNCTION (this << beacon << apAddr << bssid);
  SetBssid (bssid);
  CapabilityInformation capabilities = beacon.GetCapabilities ();
  SupportedRates rates = beacon.GetSupportedRates ();
  for (const auto & mode : GetWifiPhy ()->GetModeList ())
    {
      if (rates.IsSupportedRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
        {
          GetWifiRemoteStationManager ()->AddSupportedMode (apAddr, mode);
        }
    }
  bool isShortPreambleEnabled = capabilities.IsShortPreamble ();
  if (GetErpSupported ())
    {
      ErpInformation erpInformation = beacon.GetErpInformation ();
      isShortPreambleEnabled &= !erpInformation.GetBarkerPreambleMode ();
      if (erpInformation.GetUseProtection () != 0)
        {
          GetWifiRemoteStationManager ()->SetUseNonErpProtection (true);
        }
      else
        {
          GetWifiRemoteStationManager ()->SetUseNonErpProtection (false);
        }
      if (capabilities.IsShortSlotTime () == true)
        {
          //enable short slot time
          GetWifiPhy ()->SetSlot (MicroSeconds (9));
        }
      else
        {
          //disable short slot time
          GetWifiPhy ()->SetSlot (MicroSeconds (20));
        }
    }
  if (GetQosSupported ())
    {
      bool qosSupported = false;
      EdcaParameterSet edcaParameters = beacon.GetEdcaParameterSet ();
      if (edcaParameters.IsQosSupported ())
        {
          qosSupported = true;
          //The value of the TXOP Limit field is specified as an unsigned integer, with the least significant octet transmitted first, in units of 32 μs.
          SetEdcaParameters (AC_BE, edcaParameters.GetBeCWmin (), edcaParameters.GetBeCWmax (), edcaParameters.GetBeAifsn (), 32 * MicroSeconds (edcaParameters.GetBeTxopLimit ()));
          SetEdcaParameters (AC_BK, edcaParameters.GetBkCWmin (), edcaParameters.GetBkCWmax (), edcaParameters.GetBkAifsn (), 32 * MicroSeconds (edcaParameters.GetBkTxopLimit ()));
          SetEdcaParameters (AC_VI, edcaParameters.GetViCWmin (), edcaParameters.GetViCWmax (), edcaParameters.GetViAifsn (), 32 * MicroSeconds (edcaParameters.GetViTxopLimit ()));
          SetEdcaParameters (AC_VO, edcaParameters.GetVoCWmin (), edcaParameters.GetVoCWmax (), edcaParameters.GetVoAifsn (), 32 * MicroSeconds (edcaParameters.GetVoTxopLimit ()));
        }
      GetWifiRemoteStationManager ()->SetQosSupport (apAddr, qosSupported);
    }
  if (GetHtSupported ())
    {
      HtCapabilities htCapabilities = beacon.GetHtCapabilities ();
      if (!htCapabilities.IsSupportedMcs (0))
        {
          GetWifiRemoteStationManager ()->RemoveAllSupportedMcs (apAddr);
        }
      else
        {
          GetWifiRemoteStationManager ()->AddStationHtCapabilities (apAddr, htCapabilities);
        }
    }
  if (GetVhtSupported ())
    {
      VhtCapabilities vhtCapabilities = beacon.GetVhtCapabilities ();
      //we will always fill in RxHighestSupportedLgiDataRate field at TX, so this can be used to check whether it supports VHT
      if (vhtCapabilities.GetRxHighestSupportedLgiDataRate () > 0)
        {
          GetWifiRemoteStationManager ()->AddStationVhtCapabilities (apAddr, vhtCapabilities);
          VhtOperation vhtOperation = beacon.GetVhtOperation ();
          for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_VHT))
            {
              if (vhtCapabilities.IsSupportedRxMcs (mcs.GetMcsValue ()))
                {
                  GetWifiRemoteStationManager ()->AddSupportedMcs (apAddr, mcs);
                }
            }
        }
    }
  if (GetHtSupported ())
    {
      ExtendedCapabilities extendedCapabilities = beacon.GetExtendedCapabilities ();
      //TODO: to be completed
    }
  if (GetHeSupported ())
    {
      HeCapabilities heCapabilities = beacon.GetHeCapabilities ();
      if (heCapabilities.GetSupportedMcsAndNss () != 0)
        {
          GetWifiRemoteStationManager ()->AddStationHeCapabilities (apAddr, heCapabilities);
          HeOperation heOperation = beacon.GetHeOperation ();
          for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HE))
            {
              if (heCapabilities.IsSupportedRxMcs (mcs.GetMcsValue ()))
                {
                  GetWifiRemoteStationManager ()->AddSupportedMcs (apAddr, mcs);
                }
            }
        }

      MuEdcaParameterSet muEdcaParameters = beacon.GetMuEdcaParameterSet ();
      if (muEdcaParameters.IsPresent ())
        {
          SetMuEdcaParameters (AC_BE, muEdcaParameters.GetMuCwMin (AC_BE), muEdcaParameters.GetMuCwMax (AC_BE),
                               muEdcaParameters.GetMuAifsn (AC_BE), muEdcaParameters.GetMuEdcaTimer (AC_BE));
          SetMuEdcaParameters (AC_BK, muEdcaParameters.GetMuCwMin (AC_BK), muEdcaParameters.GetMuCwMax (AC_BK),
                               muEdcaParameters.GetMuAifsn (AC_BK), muEdcaParameters.GetMuEdcaTimer (AC_BK));
          SetMuEdcaParameters (AC_VI, muEdcaParameters.GetMuCwMin (AC_VI), muEdcaParameters.GetMuCwMax (AC_VI),
                               muEdcaParameters.GetMuAifsn (AC_VI), muEdcaParameters.GetMuEdcaTimer (AC_VI));
          SetMuEdcaParameters (AC_VO, muEdcaParameters.GetMuCwMin (AC_VO), muEdcaParameters.GetMuCwMax (AC_VO),
                               muEdcaParameters.GetMuAifsn (AC_VO), muEdcaParameters.GetMuEdcaTimer (AC_VO));
        }
    }
  GetWifiRemoteStationManager ()->SetShortPreambleEnabled (isShortPreambleEnabled);
  GetWifiRemoteStationManager ()->SetShortSlotTimeEnabled (capabilities.IsShortSlotTime ());
}

void
StaWifiMac::UpdateApInfoFromProbeResp (MgtProbeResponseHeader probeResp, Mac48Address apAddr, Mac48Address bssid)
{
  NS_LOG_FUNCTION (this << probeResp << apAddr << bssid);
  CapabilityInformation capabilities = probeResp.GetCapabilities ();
  SupportedRates rates = probeResp.GetSupportedRates ();
  for (const auto & selector : GetWifiPhy ()->GetBssMembershipSelectorList ())
    {
      if (!rates.IsBssMembershipSelectorRate (selector))
        {
          NS_LOG_DEBUG ("Supported rates do not fit with the BSS membership selector");
          return;
        }
    }
  for (const auto & mode : GetWifiPhy ()->GetModeList ())
    {
      if (rates.IsSupportedRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
        {
          GetWifiRemoteStationManager ()->AddSupportedMode (apAddr, mode);
          if (rates.IsBasicRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
            {
              GetWifiRemoteStationManager ()->AddBasicMode (mode);
            }
        }
    }

  bool isShortPreambleEnabled = capabilities.IsShortPreamble ();
  if (GetErpSupported ())
    {
      bool isErpAllowed = false;
      for (const auto & mode : GetWifiPhy ()->GetModeList (WIFI_MOD_CLASS_ERP_OFDM))
        {
          if (rates.IsSupportedRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
            {
              isErpAllowed = true;
              break;
            }
        }
      if (!isErpAllowed)
        {
          //disable short slot time and set cwMin to 31
          GetWifiPhy ()->SetSlot (MicroSeconds (20));
          ConfigureContentionWindow (31, 1023);
        }
      else
        {
          ErpInformation erpInformation = probeResp.GetErpInformation ();
          isShortPreambleEnabled &= !erpInformation.GetBarkerPreambleMode ();
          if (GetWifiRemoteStationManager ()->GetShortSlotTimeEnabled ())
            {
              //enable short slot time
              GetWifiPhy ()->SetSlot (MicroSeconds (9));
            }
          else
            {
              //disable short slot time
              GetWifiPhy ()->SetSlot (MicroSeconds (20));
            }
          ConfigureContentionWindow (15, 1023);
        }
    }
  GetWifiRemoteStationManager ()->SetShortPreambleEnabled (isShortPreambleEnabled);
  GetWifiRemoteStationManager ()->SetShortSlotTimeEnabled (capabilities.IsShortSlotTime ());
  SetBssid (bssid);
}

void
StaWifiMac::UpdateApInfoFromAssocResp (MgtAssocResponseHeader assocResp, Mac48Address apAddr)
{
  NS_LOG_FUNCTION (this << assocResp << apAddr);
  CapabilityInformation capabilities = assocResp.GetCapabilities ();
  SupportedRates rates = assocResp.GetSupportedRates ();
  bool isShortPreambleEnabled = capabilities.IsShortPreamble ();
  if (GetErpSupported ())
    {
      bool isErpAllowed = false;
      for (const auto & mode : GetWifiPhy ()->GetModeList (WIFI_MOD_CLASS_ERP_OFDM))
        {
          if (rates.IsSupportedRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
            {
              isErpAllowed = true;
              break;
            }
        }
      if (!isErpAllowed)
        {
          //disable short slot time and set cwMin to 31
          GetWifiPhy ()->SetSlot (MicroSeconds (20));
          ConfigureContentionWindow (31, 1023);
        }
      else
        {
          ErpInformation erpInformation = assocResp.GetErpInformation ();
          isShortPreambleEnabled &= !erpInformation.GetBarkerPreambleMode ();
          if (GetWifiRemoteStationManager ()->GetShortSlotTimeEnabled ())
            {
              //enable short slot time
              GetWifiPhy ()->SetSlot (MicroSeconds (9));
            }
          else
            {
              //disable short slot time
              GetWifiPhy ()->SetSlot (MicroSeconds (20));
            }
          ConfigureContentionWindow (15, 1023);
        }
    }
  GetWifiRemoteStationManager ()->SetShortPreambleEnabled (isShortPreambleEnabled);
  GetWifiRemoteStationManager ()->SetShortSlotTimeEnabled (capabilities.IsShortSlotTime ());
  if (GetQosSupported ())
    {
      bool qosSupported = false;
      EdcaParameterSet edcaParameters = assocResp.GetEdcaParameterSet ();
      if (edcaParameters.IsQosSupported ())
        {
          qosSupported = true;
          //The value of the TXOP Limit field is specified as an unsigned integer, with the least significant octet transmitted first, in units of 32 μs.
          SetEdcaParameters (AC_BE, edcaParameters.GetBeCWmin (), edcaParameters.GetBeCWmax (), edcaParameters.GetBeAifsn (), 32 * MicroSeconds (edcaParameters.GetBeTxopLimit ()));
          SetEdcaParameters (AC_BK, edcaParameters.GetBkCWmin (), edcaParameters.GetBkCWmax (), edcaParameters.GetBkAifsn (), 32 * MicroSeconds (edcaParameters.GetBkTxopLimit ()));
          SetEdcaParameters (AC_VI, edcaParameters.GetViCWmin (), edcaParameters.GetViCWmax (), edcaParameters.GetViAifsn (), 32 * MicroSeconds (edcaParameters.GetViTxopLimit ()));
          SetEdcaParameters (AC_VO, edcaParameters.GetVoCWmin (), edcaParameters.GetVoCWmax (), edcaParameters.GetVoAifsn (), 32 * MicroSeconds (edcaParameters.GetVoTxopLimit ()));
        }
      GetWifiRemoteStationManager ()->SetQosSupport (apAddr, qosSupported);
    }
  if (GetHtSupported ())
    {
      HtCapabilities htCapabilities = assocResp.GetHtCapabilities ();
      if (!htCapabilities.IsSupportedMcs (0))
        {
          GetWifiRemoteStationManager ()->RemoveAllSupportedMcs (apAddr);
        }
      else
        {
          GetWifiRemoteStationManager ()->AddStationHtCapabilities (apAddr, htCapabilities);
        }
    }
  if (GetVhtSupported ())
    {
      VhtCapabilities vhtCapabilities = assocResp.GetVhtCapabilities ();
      //we will always fill in RxHighestSupportedLgiDataRate field at TX, so this can be used to check whether it supports VHT
      if (vhtCapabilities.GetRxHighestSupportedLgiDataRate () > 0)
        {
          GetWifiRemoteStationManager ()->AddStationVhtCapabilities (apAddr, vhtCapabilities);
          VhtOperation vhtOperation = assocResp.GetVhtOperation ();
        }
    }
  if (GetHeSupported ())
    {
      HeCapabilities hecapabilities = assocResp.GetHeCapabilities ();
      if (hecapabilities.GetSupportedMcsAndNss () != 0)
        {
          GetWifiRemoteStationManager ()->AddStationHeCapabilities (apAddr, hecapabilities);
          HeOperation heOperation = assocResp.GetHeOperation ();
          GetHeConfiguration ()->SetAttribute ("BssColor", UintegerValue (heOperation.GetBssColor ()));
        }

      MuEdcaParameterSet muEdcaParameters = assocResp.GetMuEdcaParameterSet ();
      if (muEdcaParameters.IsPresent ())
        {
          SetMuEdcaParameters (AC_BE, muEdcaParameters.GetMuCwMin (AC_BE), muEdcaParameters.GetMuCwMax (AC_BE),
                               muEdcaParameters.GetMuAifsn (AC_BE), muEdcaParameters.GetMuEdcaTimer (AC_BE));
          SetMuEdcaParameters (AC_BK, muEdcaParameters.GetMuCwMin (AC_BK), muEdcaParameters.GetMuCwMax (AC_BK),
                               muEdcaParameters.GetMuAifsn (AC_BK), muEdcaParameters.GetMuEdcaTimer (AC_BK));
          SetMuEdcaParameters (AC_VI, muEdcaParameters.GetMuCwMin (AC_VI), muEdcaParameters.GetMuCwMax (AC_VI),
                               muEdcaParameters.GetMuAifsn (AC_VI), muEdcaParameters.GetMuEdcaTimer (AC_VI));
          SetMuEdcaParameters (AC_VO, muEdcaParameters.GetMuCwMin (AC_VO), muEdcaParameters.GetMuCwMax (AC_VO),
                               muEdcaParameters.GetMuAifsn (AC_VO), muEdcaParameters.GetMuEdcaTimer (AC_VO));
        }
    }
  for (const auto & mode : GetWifiPhy ()->GetModeList ())
    {
      if (rates.IsSupportedRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
        {
          GetWifiRemoteStationManager ()->AddSupportedMode (apAddr, mode);
          if (rates.IsBasicRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
            {
              GetWifiRemoteStationManager ()->AddBasicMode (mode);
            }
        }
    }
  if (GetHtSupported ())
    {
      HtCapabilities htCapabilities = assocResp.GetHtCapabilities ();
      for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HT))
        {
          if (htCapabilities.IsSupportedMcs (mcs.GetMcsValue ()))
            {
              GetWifiRemoteStationManager ()->AddSupportedMcs (apAddr, mcs);
              //here should add a control to add basic MCS when it is implemented
            }
        }
    }
  if (GetVhtSupported ())
    {
      VhtCapabilities vhtcapabilities = assocResp.GetVhtCapabilities ();
      for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_VHT))
        {
          if (vhtcapabilities.IsSupportedRxMcs (mcs.GetMcsValue ()))
            {
              GetWifiRemoteStationManager ()->AddSupportedMcs (apAddr, mcs);
              //here should add a control to add basic MCS when it is implemented
            }
        }
    }
  if (GetHtSupported ())
    {
      ExtendedCapabilities extendedCapabilities = assocResp.GetExtendedCapabilities ();
      //TODO: to be completed
    }
  if (GetHeSupported ())
    {
      HeCapabilities heCapabilities = assocResp.GetHeCapabilities ();
      for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HE))
        {
          if (heCapabilities.IsSupportedRxMcs (mcs.GetMcsValue ()))
            {
              GetWifiRemoteStationManager ()->AddSupportedMcs (apAddr, mcs);
              //here should add a control to add basic MCS when it is implemented
            }
        }
    }
}

SupportedRates
StaWifiMac::GetSupportedRates (void) const
{
  SupportedRates rates;
  for (const auto & mode : GetWifiPhy ()->GetModeList ())
    {
      uint64_t modeDataRate = mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ());
      NS_LOG_DEBUG ("Adding supported rate of " << modeDataRate);
      rates.AddSupportedRate (modeDataRate);
    }
  if (GetHtSupported ())
    {
      for (const auto & selector : GetWifiPhy ()->GetBssMembershipSelectorList ())
        {
          rates.AddBssMembershipSelectorRate (selector);
        }
    }
  return rates;
}

CapabilityInformation
StaWifiMac::GetCapabilities (void) const
{
  CapabilityInformation capabilities;
  capabilities.SetShortPreamble (GetWifiPhy ()->GetShortPhyPreambleSupported () || GetErpSupported ());
  capabilities.SetShortSlotTime (GetShortSlotTimeSupported () && GetErpSupported ());
  return capabilities;
}

void
StaWifiMac::SetState (MacState value)
{
  if (value == ASSOCIATED
      && m_state != ASSOCIATED)
    {
      m_assocLogger (GetBssid ());
    }
  else if (value != ASSOCIATED
           && m_state == ASSOCIATED)
    {
      m_deAssocLogger (GetBssid ());
    }
  m_state = value;
}

void
StaWifiMac::SetEdcaParameters (AcIndex ac, uint32_t cwMin, uint32_t cwMax, uint8_t aifsn, Time txopLimit)
{
  Ptr<QosTxop> edca = GetQosTxop (ac);
  edca->SetMinCw (cwMin);
  edca->SetMaxCw (cwMax);
  edca->SetAifsn (aifsn);
  edca->SetTxopLimit (txopLimit);
}

void
StaWifiMac::SetMuEdcaParameters (AcIndex ac, uint16_t cwMin, uint16_t cwMax, uint8_t aifsn, Time muEdcaTimer)
{
  Ptr<QosTxop> edca = GetQosTxop (ac);
  edca->SetMuCwMin (cwMin);
  edca->SetMuCwMax (cwMax);
  edca->SetMuAifsn (aifsn);
  edca->SetMuEdcaTimer (muEdcaTimer);
}

void
StaWifiMac::PhyCapabilitiesChanged (void)
{
  NS_LOG_FUNCTION (this);
  if (IsAssociated ())
    {
      NS_LOG_DEBUG ("PHY capabilities changed: send reassociation request");
      SetState (WAIT_ASSOC_RESP);
      SendAssociationRequest (true);
    }
}

void
StaWifiMac::NotifyChannelSwitching (void)
{
  NS_LOG_FUNCTION (this);

  WifiMac::NotifyChannelSwitching ();

  if (IsInitialized ())
    {
      Disassociated ();
    }
}

} //namespace ns3
