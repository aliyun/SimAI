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
#include "ns3/pointer.h"
#include "ns3/string.h"
#include "ns3/random-variable-stream.h"
#include "ap-wifi-mac.h"
#include "channel-access-manager.h"
#include "mac-tx-middle.h"
#include "mac-rx-middle.h"
#include "mgt-headers.h"
#include "msdu-aggregator.h"
#include "amsdu-subframe-header.h"
#include "wifi-phy.h"
#include "wifi-net-device.h"
#include "wifi-mac-queue.h"
#include "ns3/ht-configuration.h"
#include "ns3/he-configuration.h"
#include "qos-txop.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ApWifiMac");

NS_OBJECT_ENSURE_REGISTERED (ApWifiMac);

TypeId
ApWifiMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ApWifiMac")
    .SetParent<WifiMac> ()
    .SetGroupName ("Wifi")
    .AddConstructor<ApWifiMac> ()
    .AddAttribute ("BeaconInterval",
                   "Delay between two beacons",
                   TimeValue (MicroSeconds (102400)),
                   MakeTimeAccessor (&ApWifiMac::GetBeaconInterval,
                                     &ApWifiMac::SetBeaconInterval),
                   MakeTimeChecker ())
    .AddAttribute ("BeaconJitter",
                   "A uniform random variable to cause the initial beacon starting time (after simulation time 0) "
                   "to be distributed between 0 and the BeaconInterval.",
                   StringValue ("ns3::UniformRandomVariable"),
                   MakePointerAccessor (&ApWifiMac::m_beaconJitter),
                   MakePointerChecker<UniformRandomVariable> ())
    .AddAttribute ("EnableBeaconJitter",
                   "If beacons are enabled, whether to jitter the initial send event.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&ApWifiMac::m_enableBeaconJitter),
                   MakeBooleanChecker ())
    .AddAttribute ("BeaconGeneration",
                   "Whether or not beacons are generated.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&ApWifiMac::SetBeaconGeneration),
                   MakeBooleanChecker ())
    .AddAttribute ("EnableNonErpProtection", "Whether or not protection mechanism should be used when non-ERP STAs are present within the BSS."
                   "This parameter is only used when ERP is supported by the AP.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&ApWifiMac::m_enableNonErpProtection),
                   MakeBooleanChecker ())
    .AddAttribute ("BsrLifetime",
                   "Lifetime of Buffer Status Reports received from stations.",
                   TimeValue (MilliSeconds (20)),
                   MakeTimeAccessor (&ApWifiMac::m_bsrLifetime),
                   MakeTimeChecker ())
    .AddTraceSource ("AssociatedSta",
                     "A station associated with this access point.",
                     MakeTraceSourceAccessor (&ApWifiMac::m_assocLogger),
                     "ns3::ApWifiMac::AssociationCallback")
    .AddTraceSource ("DeAssociatedSta",
                     "A station lost association with this access point.",
                     MakeTraceSourceAccessor (&ApWifiMac::m_deAssocLogger),
                     "ns3::ApWifiMac::AssociationCallback")
  ;
  return tid;
}

ApWifiMac::ApWifiMac ()
  : m_enableBeaconGeneration (false),
    m_numNonErpStations (0),
    m_numNonHtStations (0),
    m_shortSlotTimeEnabled (false),
    m_shortPreambleEnabled (false)
{
  NS_LOG_FUNCTION (this);
  m_beaconTxop = CreateObject<Txop> (CreateObject<WifiMacQueue> (AC_BEACON));
  m_beaconTxop->SetWifiMac (this);
  m_beaconTxop->SetAifsn (1);
  m_beaconTxop->SetMinCw (0);
  m_beaconTxop->SetMaxCw (0);
  m_beaconTxop->SetChannelAccessManager (m_channelAccessManager);
  m_beaconTxop->SetTxMiddle (m_txMiddle);

  //Let the lower layers know that we are acting as an AP.
  SetTypeOfStation (AP);
}

ApWifiMac::~ApWifiMac ()
{
  NS_LOG_FUNCTION (this);
  m_staList.clear ();
}

void
ApWifiMac::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_beaconTxop->Dispose ();
  m_beaconTxop = 0;
  m_enableBeaconGeneration = false;
  m_beaconEvent.Cancel ();
  WifiMac::DoDispose ();
}

void
ApWifiMac::SetAddress (Mac48Address address)
{
  NS_LOG_FUNCTION (this << address);
  //As an AP, our MAC address is also the BSSID. Hence we are
  //overriding this function and setting both in our parent class.
  WifiMac::SetAddress (address);
  WifiMac::SetBssid (address);
}

Ptr<WifiMacQueue>
ApWifiMac::GetTxopQueue (AcIndex ac) const
{
  if (ac == AC_BEACON)
    {
      return m_beaconTxop->GetWifiMacQueue ();
    }
  return WifiMac::GetTxopQueue (ac);
}

void
ApWifiMac::SetBeaconGeneration (bool enable)
{
  NS_LOG_FUNCTION (this << enable);
  if (!enable)
    {
      m_beaconEvent.Cancel ();
    }
  else if (enable && !m_enableBeaconGeneration)
    {
      m_beaconEvent = Simulator::ScheduleNow (&ApWifiMac::SendOneBeacon, this);
    }
  m_enableBeaconGeneration = enable;
}

Time
ApWifiMac::GetBeaconInterval (void) const
{
  NS_LOG_FUNCTION (this);
  return m_beaconInterval;
}

void
ApWifiMac::SetLinkUpCallback (Callback<void> linkUp)
{
  NS_LOG_FUNCTION (this << &linkUp);
  WifiMac::SetLinkUpCallback (linkUp);

  //The approach taken here is that, from the point of view of an AP,
  //the link is always up, so we immediately invoke the callback if
  //one is set
  linkUp ();
}

void
ApWifiMac::SetBeaconInterval (Time interval)
{
  NS_LOG_FUNCTION (this << interval);
  if ((interval.GetMicroSeconds () % 1024) != 0)
    {
      NS_FATAL_ERROR ("beacon interval should be multiple of 1024us (802.11 time unit), see IEEE Std. 802.11-2012");
    }
  if (interval.GetMicroSeconds () > (1024 * 65535))
    {
      NS_FATAL_ERROR ("beacon interval should be smaller then or equal to 65535 * 1024us (802.11 time unit)");
    }
  m_beaconInterval = interval;
}

int64_t
ApWifiMac::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_beaconJitter->SetStream (stream);
  return 1;
}

void
ApWifiMac::UpdateShortSlotTimeEnabled (void)
{
  NS_LOG_FUNCTION (this);
  if (GetErpSupported () && GetShortSlotTimeSupported () && (m_numNonErpStations == 0))
    {
      for (const auto& sta : m_staList)
        {
          if (!GetWifiRemoteStationManager ()->GetShortSlotTimeSupported (sta.second))
            {
              m_shortSlotTimeEnabled = false;
              return;
            }
        }
      m_shortSlotTimeEnabled = true;
    }
  else
    {
      m_shortSlotTimeEnabled = false;
    }
}

void
ApWifiMac::UpdateShortPreambleEnabled (void)
{
  NS_LOG_FUNCTION (this);
  if (GetErpSupported () && GetWifiPhy ()->GetShortPhyPreambleSupported ())
    {
      for (const auto& sta : m_staList)
        {
          if (!GetWifiRemoteStationManager ()->GetErpOfdmSupported (sta.second) ||
              !GetWifiRemoteStationManager ()->GetShortPreambleSupported (sta.second))
            {
              m_shortPreambleEnabled = false;
              return;
            }
        }
      m_shortPreambleEnabled = true;
    }
  else
    {
      m_shortPreambleEnabled = false;
    }
}

void
ApWifiMac::ForwardDown (Ptr<Packet> packet, Mac48Address from,
                        Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << from << to);
  //If we are not a QoS AP then we definitely want to use AC_BE to
  //transmit the packet. A TID of zero will map to AC_BE (through \c
  //QosUtilsMapTidToAc()), so we use that as our default here.
  uint8_t tid = 0;

  //If we are a QoS AP then we attempt to get a TID for this packet
  if (GetQosSupported ())
    {
      tid = QosUtilsGetTidForPacket (packet);
      //Any value greater than 7 is invalid and likely indicates that
      //the packet had no QoS tag, so we revert to zero, which'll
      //mean that AC_BE is used.
      if (tid > 7)
        {
          tid = 0;
        }
    }

  ForwardDown (packet, from, to, tid);
}

void
ApWifiMac::ForwardDown (Ptr<Packet> packet, Mac48Address from,
                        Mac48Address to, uint8_t tid)
{
  NS_LOG_FUNCTION (this << packet << from << to << +tid);
  WifiMacHeader hdr;

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
      //Transmission of multiple frames in the same Polled TXOP is not supported for now
      hdr.SetQosTxopLimit (0);
      //Fill in the QoS control field in the MAC header
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
  hdr.SetAddr1 (to);
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (from);
  hdr.SetDsFrom ();
  hdr.SetDsNotTo ();

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

bool
ApWifiMac::CanForwardPacketsTo (Mac48Address to) const
{
  return (to.IsGroup () || GetWifiRemoteStationManager ()->IsAssociated (to));
}

void
ApWifiMac::Enqueue (Ptr<Packet> packet, Mac48Address to, Mac48Address from)
{
  NS_LOG_FUNCTION (this << packet << to << from);
  if (CanForwardPacketsTo (to))
    {
      ForwardDown (packet, from, to);
    }
  else
    {
      NotifyTxDrop (packet);
    }
}

void
ApWifiMac::Enqueue (Ptr<Packet> packet, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << to);
  //We're sending this packet with a from address that is our own. We
  //get that address from the lower MAC and make use of the
  //from-spoofing Enqueue() method to avoid duplicated code.
  Enqueue (packet, to, GetAddress ());
}

bool
ApWifiMac::SupportsSendFrom (void) const
{
  NS_LOG_FUNCTION (this);
  return true;
}

SupportedRates
ApWifiMac::GetSupportedRates (void) const
{
  NS_LOG_FUNCTION (this);
  SupportedRates rates;
  //Send the set of supported rates and make sure that we indicate
  //the Basic Rate set in this set of supported rates.
  for (const auto & mode : GetWifiPhy ()->GetModeList ())
    {
      uint64_t modeDataRate = mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ());
      NS_LOG_DEBUG ("Adding supported rate of " << modeDataRate);
      rates.AddSupportedRate (modeDataRate);
      //Add rates that are part of the BSSBasicRateSet (manufacturer dependent!)
      //here we choose to add the mandatory rates to the BSSBasicRateSet,
      //except for 802.11b where we assume that only the non HR-DSSS rates are part of the BSSBasicRateSet
      if (mode.IsMandatory () && (mode.GetModulationClass () != WIFI_MOD_CLASS_HR_DSSS))
        {
          NS_LOG_DEBUG ("Adding basic mode " << mode.GetUniqueName ());
          GetWifiRemoteStationManager ()->AddBasicMode (mode);
        }
    }
  //set the basic rates
  for (uint8_t j = 0; j < GetWifiRemoteStationManager ()->GetNBasicModes (); j++)
    {
      WifiMode mode = GetWifiRemoteStationManager ()->GetBasicMode (j);
      uint64_t modeDataRate = mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ());
      NS_LOG_DEBUG ("Setting basic rate " << mode.GetUniqueName ());
      rates.SetBasicRate (modeDataRate);
    }
  //If it is a HT AP, then add the BSSMembershipSelectorSet
  //The standard says that the BSSMembershipSelectorSet
  //must have its MSB set to 1 (must be treated as a Basic Rate)
  //Also the standard mentioned that at least 1 element should be included in the SupportedRates the rest can be in the ExtendedSupportedRates
  if (GetHtSupported ())
    {
      for (const auto & selector : GetWifiPhy ()->GetBssMembershipSelectorList ())
        {
          rates.AddBssMembershipSelectorRate (selector);
        }
    }
  return rates;
}

DsssParameterSet
ApWifiMac::GetDsssParameterSet (void) const
{
  NS_LOG_FUNCTION (this);
  DsssParameterSet dsssParameters;
  if (GetDsssSupported ())
    {
      dsssParameters.SetDsssSupported (1);
      dsssParameters.SetCurrentChannel (GetWifiPhy ()->GetChannelNumber ());
    }
  return dsssParameters;
}

CapabilityInformation
ApWifiMac::GetCapabilities (void) const
{
  NS_LOG_FUNCTION (this);
  CapabilityInformation capabilities;
  capabilities.SetShortPreamble (m_shortPreambleEnabled);
  capabilities.SetShortSlotTime (m_shortSlotTimeEnabled);
  capabilities.SetEss ();
  return capabilities;
}

ErpInformation
ApWifiMac::GetErpInformation (void) const
{
  NS_LOG_FUNCTION (this);
  ErpInformation information;
  information.SetErpSupported (1);
  if (GetErpSupported ())
    {
      information.SetNonErpPresent (m_numNonErpStations > 0);
      information.SetUseProtection (GetUseNonErpProtection ());
      if (m_shortPreambleEnabled)
        {
          information.SetBarkerPreambleMode (0);
        }
      else
        {
          information.SetBarkerPreambleMode (1);
        }
    }
  return information;
}

EdcaParameterSet
ApWifiMac::GetEdcaParameterSet (void) const
{
  NS_LOG_FUNCTION (this);
  EdcaParameterSet edcaParameters;
  if (GetQosSupported ())
    {
      edcaParameters.SetQosSupported (1);
      Ptr<QosTxop> edca;
      Time txopLimit;

      edca = GetQosTxop (AC_BE);
      txopLimit = edca->GetTxopLimit ();
      edcaParameters.SetBeAci (0);
      edcaParameters.SetBeCWmin (edca->GetMinCw ());
      edcaParameters.SetBeCWmax (edca->GetMaxCw ());
      edcaParameters.SetBeAifsn (edca->GetAifsn ());
      edcaParameters.SetBeTxopLimit (static_cast<uint16_t> (txopLimit.GetMicroSeconds () / 32));

      edca = GetQosTxop (AC_BK);
      txopLimit = edca->GetTxopLimit ();
      edcaParameters.SetBkAci (1);
      edcaParameters.SetBkCWmin (edca->GetMinCw ());
      edcaParameters.SetBkCWmax (edca->GetMaxCw ());
      edcaParameters.SetBkAifsn (edca->GetAifsn ());
      edcaParameters.SetBkTxopLimit (static_cast<uint16_t> (txopLimit.GetMicroSeconds () / 32));

      edca = GetQosTxop (AC_VI);
      txopLimit = edca->GetTxopLimit ();
      edcaParameters.SetViAci (2);
      edcaParameters.SetViCWmin (edca->GetMinCw ());
      edcaParameters.SetViCWmax (edca->GetMaxCw ());
      edcaParameters.SetViAifsn (edca->GetAifsn ());
      edcaParameters.SetViTxopLimit (static_cast<uint16_t> (txopLimit.GetMicroSeconds () / 32));

      edca = GetQosTxop (AC_VO);
      txopLimit = edca->GetTxopLimit ();
      edcaParameters.SetVoAci (3);
      edcaParameters.SetVoCWmin (edca->GetMinCw ());
      edcaParameters.SetVoCWmax (edca->GetMaxCw ());
      edcaParameters.SetVoAifsn (edca->GetAifsn ());
      edcaParameters.SetVoTxopLimit (static_cast<uint16_t> (txopLimit.GetMicroSeconds () / 32));

      edcaParameters.SetQosInfo (0);
    }
  return edcaParameters;
}

MuEdcaParameterSet
ApWifiMac::GetMuEdcaParameterSet (void) const
{
  NS_LOG_FUNCTION (this);
  MuEdcaParameterSet muEdcaParameters;
  if (GetHeSupported ())
    {
      Ptr<HeConfiguration> heConfiguration = GetHeConfiguration ();
      NS_ASSERT (heConfiguration != 0);

      muEdcaParameters.SetQosInfo (0);

      UintegerValue uintegerValue;
      TimeValue timeValue;

      heConfiguration->GetAttribute ("MuBeAifsn", uintegerValue);
      muEdcaParameters.SetMuAifsn (AC_BE, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuBeCwMin", uintegerValue);
      muEdcaParameters.SetMuCwMin (AC_BE, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuBeCwMax", uintegerValue);
      muEdcaParameters.SetMuCwMax (AC_BE, uintegerValue.Get ());
      heConfiguration->GetAttribute ("BeMuEdcaTimer", timeValue);
      muEdcaParameters.SetMuEdcaTimer (AC_BE, timeValue.Get ());

      heConfiguration->GetAttribute ("MuBkAifsn", uintegerValue);
      muEdcaParameters.SetMuAifsn (AC_BK, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuBkCwMin", uintegerValue);
      muEdcaParameters.SetMuCwMin (AC_BK, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuBkCwMax", uintegerValue);
      muEdcaParameters.SetMuCwMax (AC_BK, uintegerValue.Get ());
      heConfiguration->GetAttribute ("BkMuEdcaTimer", timeValue);
      muEdcaParameters.SetMuEdcaTimer (AC_BK, timeValue.Get ());

      heConfiguration->GetAttribute ("MuViAifsn", uintegerValue);
      muEdcaParameters.SetMuAifsn (AC_VI, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuViCwMin", uintegerValue);
      muEdcaParameters.SetMuCwMin (AC_VI, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuViCwMax", uintegerValue);
      muEdcaParameters.SetMuCwMax (AC_VI, uintegerValue.Get ());
      heConfiguration->GetAttribute ("ViMuEdcaTimer", timeValue);
      muEdcaParameters.SetMuEdcaTimer (AC_VI, timeValue.Get ());

      heConfiguration->GetAttribute ("MuVoAifsn", uintegerValue);
      muEdcaParameters.SetMuAifsn (AC_VO, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuVoCwMin", uintegerValue);
      muEdcaParameters.SetMuCwMin (AC_VO, uintegerValue.Get ());
      heConfiguration->GetAttribute ("MuVoCwMax", uintegerValue);
      muEdcaParameters.SetMuCwMax (AC_VO, uintegerValue.Get ());
      heConfiguration->GetAttribute ("VoMuEdcaTimer", timeValue);
      muEdcaParameters.SetMuEdcaTimer (AC_VO, timeValue.Get ());
    }
  return muEdcaParameters;
}

HtOperation
ApWifiMac::GetHtOperation (void) const
{
  NS_LOG_FUNCTION (this);
  HtOperation operation;
  if (GetHtSupported ())
    {
      operation.SetHtSupported (1);
      operation.SetPrimaryChannel (GetWifiPhy ()->GetPrimaryChannelNumber (20));
      operation.SetRifsMode (false);
      operation.SetNonGfHtStasPresent (true);
      if (GetWifiPhy ()->GetChannelWidth () > 20)
        {
          operation.SetSecondaryChannelOffset (1);
          operation.SetStaChannelWidth (1);
        }
      if (m_numNonHtStations == 0)
        {
          operation.SetHtProtection (NO_PROTECTION);
        }
      else
        {
          operation.SetHtProtection (MIXED_MODE_PROTECTION);
        }
      uint64_t maxSupportedRate = 0; //in bit/s
      for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HT))
        {
          uint8_t nss = (mcs.GetMcsValue () / 8) + 1;
          NS_ASSERT (nss > 0 && nss < 5);
          uint64_t dataRate = mcs.GetDataRate (GetWifiPhy ()->GetChannelWidth (), GetHtConfiguration ()->GetShortGuardIntervalSupported () ? 400 : 800, nss);
          if (dataRate > maxSupportedRate)
            {
              maxSupportedRate = dataRate;
              NS_LOG_DEBUG ("Updating maxSupportedRate to " << maxSupportedRate);
            }
        }
      uint8_t maxSpatialStream = GetWifiPhy ()->GetMaxSupportedTxSpatialStreams ();
      auto mcsList = GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HT);
      uint8_t nMcs = mcsList.size ();
      for (const auto& sta : m_staList)
        {
          if (GetWifiRemoteStationManager ()->GetHtSupported (sta.second))
            {
              uint64_t maxSupportedRateByHtSta = 0; //in bit/s
              auto itMcs = mcsList.begin ();
              for (uint8_t j = 0; j < (std::min (nMcs, GetWifiRemoteStationManager ()->GetNMcsSupported (sta.second))); j++)
                {
                  WifiMode mcs = *itMcs++;
                  uint8_t nss = (mcs.GetMcsValue () / 8) + 1;
                  NS_ASSERT (nss > 0 && nss < 5);
                  uint64_t dataRate = mcs.GetDataRate (GetWifiRemoteStationManager ()->GetChannelWidthSupported (sta.second),
                                                       GetWifiRemoteStationManager ()->GetShortGuardIntervalSupported (sta.second) ? 400 : 800, nss);
                  if (dataRate > maxSupportedRateByHtSta)
                    {
                      maxSupportedRateByHtSta = dataRate;
                    }
                }
              if (maxSupportedRateByHtSta < maxSupportedRate)
                {
                  maxSupportedRate = maxSupportedRateByHtSta;
                }
              if (GetWifiRemoteStationManager ()->GetNMcsSupported (sta.second) < nMcs)
                {
                  nMcs = GetWifiRemoteStationManager ()->GetNMcsSupported (sta.second);
                }
              if (GetWifiRemoteStationManager ()->GetNumberOfSupportedStreams (sta.second) < maxSpatialStream)
                {
                  maxSpatialStream = GetWifiRemoteStationManager ()->GetNumberOfSupportedStreams (sta.second);
                }
            }
        }
      operation.SetRxHighestSupportedDataRate (static_cast<uint16_t> (maxSupportedRate / 1e6)); //in Mbit/s
      operation.SetTxMcsSetDefined (nMcs > 0);
      operation.SetTxMaxNSpatialStreams (maxSpatialStream);
      //To be filled in once supported
      operation.SetObssNonHtStasPresent (0);
      operation.SetDualBeacon (0);
      operation.SetDualCtsProtection (0);
      operation.SetStbcBeacon (0);
      operation.SetLSigTxopProtectionFullSupport (0);
      operation.SetPcoActive (0);
      operation.SetPhase (0);
      operation.SetRxMcsBitmask (0);
      operation.SetTxRxMcsSetUnequal (0);
      operation.SetTxUnequalModulation (0);
    }
  return operation;
}

VhtOperation
ApWifiMac::GetVhtOperation (void) const
{
  NS_LOG_FUNCTION (this);
  VhtOperation operation;
  if (GetVhtSupported ())
    {
      operation.SetVhtSupported (1);
      const uint16_t bssBandwidth = GetWifiPhy ()->GetChannelWidth ();
      // Set to 0 for 20 MHz or 40 MHz BSS bandwidth.
      // Set to 1 for 80 MHz, 160 MHz or 80+80 MHz BSS bandwidth.
      operation.SetChannelWidth ((bssBandwidth > 40) ? 1 : 0);
      // For 20, 40, or 80 MHz BSS bandwidth, indicates the channel center frequency
      // index for the 20, 40, or 80 MHz channel on which the VHT BSS operates.
      // For 160 MHz BSS bandwidth and the Channel Width subfield equal to 1,
      // indicates the channel center frequency index of the 80 MHz channel
      // segment that contains the primary channel.
      operation.SetChannelCenterFrequencySegment0 ((bssBandwidth == 160) ?
                                                   GetWifiPhy ()->GetOperatingChannel ().GetPrimaryChannelNumber (80, WIFI_STANDARD_80211ac) :
                                                   GetWifiPhy ()->GetChannelNumber ());
      // For a 20, 40, or 80 MHz BSS bandwidth, this subfield is set to 0.
      // For a 160 MHz BSS bandwidth and the Channel Width subfield equal to 1,
      // indicates the channel center frequency index of the 160 MHz channel on
      // which the VHT BSS operates.
      operation.SetChannelCenterFrequencySegment1 ((bssBandwidth == 160) ? GetWifiPhy ()->GetChannelNumber () : 0);
      uint8_t maxSpatialStream = GetWifiPhy ()->GetMaxSupportedRxSpatialStreams ();
      for (const auto& sta : m_staList)
        {
          if (GetWifiRemoteStationManager ()->GetVhtSupported (sta.second))
            {
              if (GetWifiRemoteStationManager ()->GetNumberOfSupportedStreams (sta.second) < maxSpatialStream)
                {
                  maxSpatialStream = GetWifiRemoteStationManager ()->GetNumberOfSupportedStreams (sta.second);
                }
            }
        }
      for (uint8_t nss = 1; nss <= maxSpatialStream; nss++)
        {
          uint8_t maxMcs = 9; //TBD: hardcode to 9 for now since we assume all MCS values are supported
          operation.SetMaxVhtMcsPerNss (nss, maxMcs);
        }
    }
  return operation;
}

HeOperation
ApWifiMac::GetHeOperation (void) const
{
  NS_LOG_FUNCTION (this);
  HeOperation operation;
  if (GetHeSupported ())
    {
      operation.SetHeSupported (1);
      uint8_t maxSpatialStream = GetWifiPhy ()->GetMaxSupportedRxSpatialStreams ();
      for (const auto& sta : m_staList)
        {
          if (GetWifiRemoteStationManager ()->GetHeSupported (sta.second))
            {
              if (GetWifiRemoteStationManager ()->GetNumberOfSupportedStreams (sta.second) < maxSpatialStream)
                {
                  maxSpatialStream = GetWifiRemoteStationManager ()->GetNumberOfSupportedStreams (sta.second);
                }
            }
        }
      for (uint8_t nss = 1; nss <= maxSpatialStream; nss++)
        {
          operation.SetMaxHeMcsPerNss (nss, 11); //TBD: hardcode to 11 for now since we assume all MCS values are supported
        }
      operation.SetBssColor (GetHeConfiguration ()->GetBssColor ());
    }
  return operation;
}

void
ApWifiMac::SendProbeResp (Mac48Address to)
{
  NS_LOG_FUNCTION (this << to);
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_PROBE_RESPONSE);
  hdr.SetAddr1 (to);
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (GetAddress ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  Ptr<Packet> packet = Create<Packet> ();
  MgtProbeResponseHeader probe;
  probe.SetSsid (GetSsid ());
  probe.SetSupportedRates (GetSupportedRates ());
  probe.SetBeaconIntervalUs (GetBeaconInterval ().GetMicroSeconds ());
  probe.SetCapabilities (GetCapabilities ());
  GetWifiRemoteStationManager ()->SetShortPreambleEnabled (m_shortPreambleEnabled);
  GetWifiRemoteStationManager ()->SetShortSlotTimeEnabled (m_shortSlotTimeEnabled);
  if (GetDsssSupported ())
    {
      probe.SetDsssParameterSet (GetDsssParameterSet ());
    }
  if (GetErpSupported ())
    {
      probe.SetErpInformation (GetErpInformation ());
    }
  if (GetQosSupported ())
    {
      probe.SetEdcaParameterSet (GetEdcaParameterSet ());
    }
  if (GetHtSupported ())
    {
      probe.SetExtendedCapabilities (GetExtendedCapabilities ());
      probe.SetHtCapabilities (GetHtCapabilities ());
      probe.SetHtOperation (GetHtOperation ());
    }
  if (GetVhtSupported ())
    {
      probe.SetVhtCapabilities (GetVhtCapabilities ());
      probe.SetVhtOperation (GetVhtOperation ());
    }
  if (GetHeSupported ())
    {
      probe.SetHeCapabilities (GetHeCapabilities ());
      probe.SetHeOperation (GetHeOperation ());
      probe.SetMuEdcaParameterSet (GetMuEdcaParameterSet ());
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
  else if (!GetWifiRemoteStationManager ()->GetQosSupported (to))
    {
      GetBEQueue ()->Queue (packet, hdr);
    }
  else
    {
      GetVOQueue ()->Queue (packet, hdr);
    }
}

void
ApWifiMac::SendAssocResp (Mac48Address to, bool success, bool isReassoc)
{
  NS_LOG_FUNCTION (this << to << success << isReassoc);
  WifiMacHeader hdr;
  hdr.SetType (isReassoc ? WIFI_MAC_MGT_REASSOCIATION_RESPONSE : WIFI_MAC_MGT_ASSOCIATION_RESPONSE);
  hdr.SetAddr1 (to);
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (GetAddress ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  Ptr<Packet> packet = Create<Packet> ();
  MgtAssocResponseHeader assoc;
  StatusCode code;
  if (success)
    {
      code.SetSuccess ();
      uint16_t aid = 0;
      bool found = false;
      if (isReassoc)
        {
          for (const auto& sta : m_staList)
            {
              if (sta.second == to)
                {
                  aid = sta.first;
                  found = true;
                  break;
                }
            }
        }
      if (!found)
        {
          aid = GetNextAssociationId ();
          m_staList.insert (std::make_pair (aid, to));
          m_assocLogger (aid, to);
          GetWifiRemoteStationManager ()->SetAssociationId (to, aid);
          if (GetWifiRemoteStationManager ()->GetDsssSupported (to) && !GetWifiRemoteStationManager ()->GetErpOfdmSupported (to))
            {
              m_numNonErpStations++;
            }
          if (!GetWifiRemoteStationManager ()->GetHtSupported (to))
            {
              m_numNonHtStations++;
            }
          UpdateShortSlotTimeEnabled ();
          UpdateShortPreambleEnabled ();
        }
      assoc.SetAssociationId (aid);
    }
  else
    {
      code.SetFailure ();
    }
  assoc.SetSupportedRates (GetSupportedRates ());
  assoc.SetStatusCode (code);
  assoc.SetCapabilities (GetCapabilities ());
  if (GetErpSupported ())
    {
      assoc.SetErpInformation (GetErpInformation ());
    }
  if (GetQosSupported ())
    {
      assoc.SetEdcaParameterSet (GetEdcaParameterSet ());
    }
  if (GetHtSupported ())
    {
      assoc.SetExtendedCapabilities (GetExtendedCapabilities ());
      assoc.SetHtCapabilities (GetHtCapabilities ());
      assoc.SetHtOperation (GetHtOperation ());
    }
  if (GetVhtSupported ())
    {
      assoc.SetVhtCapabilities (GetVhtCapabilities ());
      assoc.SetVhtOperation (GetVhtOperation ());
    }
  if (GetHeSupported ())
    {
      assoc.SetHeCapabilities (GetHeCapabilities ());
      assoc.SetHeOperation (GetHeOperation ());
      assoc.SetMuEdcaParameterSet (GetMuEdcaParameterSet ());
    }
  packet->AddHeader (assoc);

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
  else if (!GetWifiRemoteStationManager ()->GetQosSupported (to))
    {
      GetBEQueue ()->Queue (packet, hdr);
    }
  else
    {
      GetVOQueue ()->Queue (packet, hdr);
    }
}

void
ApWifiMac::SendOneBeacon (void)
{
  NS_LOG_FUNCTION (this);
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_BEACON);
  hdr.SetAddr1 (Mac48Address::GetBroadcast ());
  hdr.SetAddr2 (GetAddress ());
  hdr.SetAddr3 (GetAddress ());
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();
  Ptr<Packet> packet = Create<Packet> ();
  MgtBeaconHeader beacon;
  beacon.SetSsid (GetSsid ());
  beacon.SetSupportedRates (GetSupportedRates ());
  beacon.SetBeaconIntervalUs (GetBeaconInterval ().GetMicroSeconds ());
  beacon.SetCapabilities (GetCapabilities ());
  GetWifiRemoteStationManager ()->SetShortPreambleEnabled (m_shortPreambleEnabled);
  GetWifiRemoteStationManager ()->SetShortSlotTimeEnabled (m_shortSlotTimeEnabled);
  if (GetDsssSupported ())
    {
      beacon.SetDsssParameterSet (GetDsssParameterSet ());
    }
  if (GetErpSupported ())
    {
      beacon.SetErpInformation (GetErpInformation ());
    }
  if (GetQosSupported ())
    {
      beacon.SetEdcaParameterSet (GetEdcaParameterSet ());
    }
  if (GetHtSupported ())
    {
      beacon.SetExtendedCapabilities (GetExtendedCapabilities ());
      beacon.SetHtCapabilities (GetHtCapabilities ());
      beacon.SetHtOperation (GetHtOperation ());
    }
  if (GetVhtSupported ())
    {
      beacon.SetVhtCapabilities (GetVhtCapabilities ());
      beacon.SetVhtOperation (GetVhtOperation ());
    }
  if (GetHeSupported ())
    {
      beacon.SetHeCapabilities (GetHeCapabilities ());
      beacon.SetHeOperation (GetHeOperation ());
      beacon.SetMuEdcaParameterSet (GetMuEdcaParameterSet ());
    }
  packet->AddHeader (beacon);

  //The beacon has it's own special queue, so we load it in there
  m_beaconTxop->Queue (packet, hdr);
  m_beaconEvent = Simulator::Schedule (GetBeaconInterval (), &ApWifiMac::SendOneBeacon, this);

  //If a STA that does not support Short Slot Time associates,
  //the AP shall use long slot time beginning at the first Beacon
  //subsequent to the association of the long slot time STA.
  if (GetErpSupported ())
    {
      if (m_shortSlotTimeEnabled)
        {
          //Enable short slot time
          GetWifiPhy ()->SetSlot (MicroSeconds (9));
        }
      else
        {
          //Disable short slot time
          GetWifiPhy ()->SetSlot (MicroSeconds (20));
        }
    }
}

void
ApWifiMac::TxOk (Ptr<const WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  const WifiMacHeader& hdr = mpdu->GetHeader ();
  if ((hdr.IsAssocResp () || hdr.IsReassocResp ())
      && GetWifiRemoteStationManager ()->IsWaitAssocTxOk (hdr.GetAddr1 ()))
    {
      NS_LOG_DEBUG ("associated with sta=" << hdr.GetAddr1 ());
      GetWifiRemoteStationManager ()->RecordGotAssocTxOk (hdr.GetAddr1 ());
    }
}

void
ApWifiMac::TxFailed (WifiMacDropReason timeoutReason, Ptr<const WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << +timeoutReason << *mpdu);
  const WifiMacHeader& hdr = mpdu->GetHeader ();

  if ((hdr.IsAssocResp () || hdr.IsReassocResp ())
      && GetWifiRemoteStationManager ()->IsWaitAssocTxOk (hdr.GetAddr1 ()))
    {
      NS_LOG_DEBUG ("association failed with sta=" << hdr.GetAddr1 ());
      GetWifiRemoteStationManager ()->RecordGotAssocTxFailed (hdr.GetAddr1 ());
    }
}

void
ApWifiMac::Receive (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  const WifiMacHeader* hdr = &mpdu->GetHeader ();
  Ptr<const Packet> packet = mpdu->GetPacket ();
  Mac48Address from = hdr->GetAddr2 ();
  if (hdr->IsData ())
    {
      Mac48Address bssid = hdr->GetAddr1 ();
      if (!hdr->IsFromDs ()
          && hdr->IsToDs ()
          && bssid == GetAddress ()
          && GetWifiRemoteStationManager ()->IsAssociated (from))
        {
          Mac48Address to = hdr->GetAddr3 ();
          if (to == GetAddress ())
            {
              NS_LOG_DEBUG ("frame for me from=" << from);
              if (hdr->IsQosData ())
                {
                  if (hdr->IsQosAmsdu ())
                    {
                      NS_LOG_DEBUG ("Received A-MSDU from=" << from << ", size=" << packet->GetSize ());
                      DeaggregateAmsduAndForward (mpdu);
                      packet = 0;
                    }
                  else
                    {
                      ForwardUp (packet, from, bssid);
                    }
                }
              else if (hdr->HasData ())
                {
                  ForwardUp (packet, from, bssid);
                }
            }
          else if (to.IsGroup ()
                   || GetWifiRemoteStationManager ()->IsAssociated (to))
            {
              NS_LOG_DEBUG ("forwarding frame from=" << from << ", to=" << to);
              Ptr<Packet> copy = packet->Copy ();

              //If the frame we are forwarding is of type QoS Data,
              //then we need to preserve the UP in the QoS control
              //header...
              if (hdr->IsQosData ())
                {
                  ForwardDown (copy, from, to, hdr->GetQosTid ());
                }
              else
                {
                  ForwardDown (copy, from, to);
                }
              ForwardUp (packet, from, to);
            }
          else
            {
              ForwardUp (packet, from, to);
            }
        }
      else if (hdr->IsFromDs ()
               && hdr->IsToDs ())
        {
          //this is an AP-to-AP frame
          //we ignore for now.
          NotifyRxDrop (packet);
        }
      else
        {
          //we can ignore these frames since
          //they are not targeted at the AP
          NotifyRxDrop (packet);
        }
      return;
    }
  else if (hdr->IsMgt ())
    {
      if (hdr->IsProbeReq ()
          && (hdr->GetAddr1 ().IsGroup () || hdr->GetAddr1 () == GetAddress ()))
        {
          // In the case where the Address 1 field contains a group address, the
          // Address 3 field also is validated to verify that the group addressed
          // frame originated from a STA in the BSS of which the receiving STA is
          // a member (Section 9.3.3.1 of 802.11-2020)
          if (hdr->GetAddr1 ().IsGroup ()
              && !hdr->GetAddr3 ().IsBroadcast () && hdr->GetAddr3 () != GetAddress ())
            {
              // not addressed to us
              return;
            }
          MgtProbeRequestHeader probeRequestHeader;
          packet->PeekHeader (probeRequestHeader);
          Ssid ssid = probeRequestHeader.GetSsid ();
          if (ssid == GetSsid () || ssid.IsBroadcast ())
            {
              NS_LOG_DEBUG ("Probe request received from " << from << ": send probe response");
              SendProbeResp (from);
            }
          return;
        }
      else if (hdr->GetAddr1 () == GetAddress ())
        {
          if (hdr->IsAssocReq ())
            {
              NS_LOG_DEBUG ("Association request received from " << from);
              //first, verify that the the station's supported
              //rate set is compatible with our Basic Rate set
              MgtAssocRequestHeader assocReq;
              packet->PeekHeader (assocReq);
              CapabilityInformation capabilities = assocReq.GetCapabilities ();
              GetWifiRemoteStationManager ()->AddSupportedPhyPreamble (from, capabilities.IsShortPreamble ());
              SupportedRates rates = assocReq.GetSupportedRates ();
              bool problem = false;
              if (rates.GetNRates () == 0)
                {
                  problem = true;
                }
              if (GetHtSupported ())
                {
                  //check whether the HT STA supports all MCSs in Basic MCS Set
                  HtCapabilities htcapabilities = assocReq.GetHtCapabilities ();
                  if (htcapabilities.IsSupportedMcs (0))
                    {
                      for (uint8_t i = 0; i < GetWifiRemoteStationManager ()->GetNBasicMcs (); i++)
                        {
                          WifiMode mcs = GetWifiRemoteStationManager ()->GetBasicMcs (i);
                          if (!htcapabilities.IsSupportedMcs (mcs.GetMcsValue ()))
                            {
                              problem = true;
                              break;
                            }
                        }
                    }
                }
              if (GetVhtSupported ())
                {
                  //check whether the VHT STA supports all MCSs in Basic MCS Set
                  VhtCapabilities vhtcapabilities = assocReq.GetVhtCapabilities ();
                  if (vhtcapabilities.GetVhtCapabilitiesInfo () != 0)
                    {
                      for (uint8_t i = 0; i < GetWifiRemoteStationManager ()->GetNBasicMcs (); i++)
                        {
                          WifiMode mcs = GetWifiRemoteStationManager ()->GetBasicMcs (i);
                          if (!vhtcapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                            {
                              problem = true;
                              break;
                            }
                        }
                    }
                }
              if (GetHeSupported ())
                {
                  //check whether the HE STA supports all MCSs in Basic MCS Set
                  HeCapabilities hecapabilities = assocReq.GetHeCapabilities ();
                  if (hecapabilities.GetSupportedMcsAndNss () != 0)
                    {
                      for (uint8_t i = 0; i < GetWifiRemoteStationManager ()->GetNBasicMcs (); i++)
                        {
                          WifiMode mcs = GetWifiRemoteStationManager ()->GetBasicMcs (i);
                          if (!hecapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                            {
                              problem = true;
                              break;
                            }
                        }
                    }
                }
              if (problem)
                {
                  NS_LOG_DEBUG ("One of the Basic Rate set mode is not supported by the station: send association response with an error status");
                  SendAssocResp (hdr->GetAddr2 (), false, false);
                }
              else
                {
                  NS_LOG_DEBUG ("The Basic Rate set modes are supported by the station");
                  //record all its supported modes in its associated WifiRemoteStation
                  for (const auto & mode : GetWifiPhy ()->GetModeList ())
                    {
                      if (rates.IsSupportedRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
                        {
                          GetWifiRemoteStationManager ()->AddSupportedMode (from, mode);
                        }
                    }
                  if (GetErpSupported () && GetWifiRemoteStationManager ()->GetErpOfdmSupported (from) && capabilities.IsShortSlotTime ())
                    {
                      GetWifiRemoteStationManager ()->AddSupportedErpSlotTime (from, true);
                    }
                  if (GetHtSupported ())
                    {
                      HtCapabilities htCapabilities = assocReq.GetHtCapabilities ();
                      if (htCapabilities.IsSupportedMcs (0))
                        {
                          GetWifiRemoteStationManager ()->AddStationHtCapabilities (from, htCapabilities);
                        }
                    }
                  if (GetVhtSupported ())
                    {
                      VhtCapabilities vhtCapabilities = assocReq.GetVhtCapabilities ();
                      //we will always fill in RxHighestSupportedLgiDataRate field at TX, so this can be used to check whether it supports VHT
                      if (vhtCapabilities.GetRxHighestSupportedLgiDataRate () > 0)
                        {
                          GetWifiRemoteStationManager ()->AddStationVhtCapabilities (from, vhtCapabilities);
                          for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_VHT))
                            {
                              if (vhtCapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                                {
                                  GetWifiRemoteStationManager ()->AddSupportedMcs (hdr->GetAddr2 (), mcs);
                                  //here should add a control to add basic MCS when it is implemented
                                }
                            }
                        }
                    }
                  if (GetHtSupported ())
                    {
                      ExtendedCapabilities extendedCapabilities = assocReq.GetExtendedCapabilities ();
                      //TODO: to be completed
                    }
                  if (GetHeSupported ())
                    {
                      HeCapabilities heCapabilities = assocReq.GetHeCapabilities ();
                      if (heCapabilities.GetSupportedMcsAndNss () != 0)
                        {
                          GetWifiRemoteStationManager ()->AddStationHeCapabilities (from, heCapabilities);
                          for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HE))
                            {
                              if (heCapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                                {
                                  GetWifiRemoteStationManager ()->AddSupportedMcs (hdr->GetAddr2 (), mcs);
                                  //here should add a control to add basic MCS when it is implemented
                                }
                            }
                        }
                    }
                  GetWifiRemoteStationManager ()->RecordWaitAssocTxOk (from);
                  NS_LOG_DEBUG ("Send association response with success status");
                  SendAssocResp (hdr->GetAddr2 (), true, false);
                }
              return;
            }
          else if (hdr->IsReassocReq ())
            {
              NS_LOG_DEBUG ("Reassociation request received from " << from);
              //first, verify that the the station's supported
              //rate set is compatible with our Basic Rate set
              MgtReassocRequestHeader reassocReq;
              packet->PeekHeader (reassocReq);
              CapabilityInformation capabilities = reassocReq.GetCapabilities ();
              GetWifiRemoteStationManager ()->AddSupportedPhyPreamble (from, capabilities.IsShortPreamble ());
              SupportedRates rates = reassocReq.GetSupportedRates ();
              bool problem = false;
              if (rates.GetNRates () == 0)
                {
                  problem = true;
                }
              if (GetHtSupported ())
                {
                  //check whether the HT STA supports all MCSs in Basic MCS Set
                  HtCapabilities htcapabilities = reassocReq.GetHtCapabilities ();
                  if (htcapabilities.IsSupportedMcs (0))
                    {
                      for (uint8_t i = 0; i < GetWifiRemoteStationManager ()->GetNBasicMcs (); i++)
                        {
                          WifiMode mcs = GetWifiRemoteStationManager ()->GetBasicMcs (i);
                          if (!htcapabilities.IsSupportedMcs (mcs.GetMcsValue ()))
                            {
                              problem = true;
                              break;
                            }
                        }
                    }
                }
              if (GetVhtSupported ())
                {
                  //check whether the VHT STA supports all MCSs in Basic MCS Set
                  VhtCapabilities vhtcapabilities = reassocReq.GetVhtCapabilities ();
                  if (vhtcapabilities.GetVhtCapabilitiesInfo () != 0)
                    {
                      for (uint8_t i = 0; i < GetWifiRemoteStationManager ()->GetNBasicMcs (); i++)
                        {
                          WifiMode mcs = GetWifiRemoteStationManager ()->GetBasicMcs (i);
                          if (!vhtcapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                            {
                              problem = true;
                              break;
                            }
                        }
                    }
                }
              if (GetHeSupported ())
                {
                  //check whether the HE STA supports all MCSs in Basic MCS Set
                  HeCapabilities hecapabilities = reassocReq.GetHeCapabilities ();
                  if (hecapabilities.GetSupportedMcsAndNss () != 0)
                    {
                      for (uint8_t i = 0; i < GetWifiRemoteStationManager ()->GetNBasicMcs (); i++)
                        {
                          WifiMode mcs = GetWifiRemoteStationManager ()->GetBasicMcs (i);
                          if (!hecapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                            {
                              problem = true;
                              break;
                            }
                        }
                    }
                }
              if (problem)
                {
                  NS_LOG_DEBUG ("One of the Basic Rate set mode is not supported by the station: send reassociation response with an error status");
                  SendAssocResp (hdr->GetAddr2 (), false, true);
                }
              else
                {
                  NS_LOG_DEBUG ("The Basic Rate set modes are supported by the station");
                  //update all its supported modes in its associated WifiRemoteStation
                  for (const auto & mode : GetWifiPhy ()->GetModeList ())
                    {
                      if (rates.IsSupportedRate (mode.GetDataRate (GetWifiPhy ()->GetChannelWidth ())))
                        {
                          GetWifiRemoteStationManager ()->AddSupportedMode (from, mode);
                        }
                    }
                  if (GetErpSupported () && GetWifiRemoteStationManager ()->GetErpOfdmSupported (from) && capabilities.IsShortSlotTime ())
                    {
                      GetWifiRemoteStationManager ()->AddSupportedErpSlotTime (from, true);
                    }
                  if (GetHtSupported ())
                    {
                      HtCapabilities htCapabilities = reassocReq.GetHtCapabilities ();
                      if (htCapabilities.IsSupportedMcs (0))
                        {
                          GetWifiRemoteStationManager ()->AddStationHtCapabilities (from, htCapabilities);
                        }
                    }
                  if (GetVhtSupported ())
                    {
                      VhtCapabilities vhtCapabilities = reassocReq.GetVhtCapabilities ();
                      //we will always fill in RxHighestSupportedLgiDataRate field at TX, so this can be used to check whether it supports VHT
                      if (vhtCapabilities.GetRxHighestSupportedLgiDataRate () > 0)
                        {
                          GetWifiRemoteStationManager ()->AddStationVhtCapabilities (from, vhtCapabilities);
                          for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_VHT))
                            {
                              if (vhtCapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                                {
                                  GetWifiRemoteStationManager ()->AddSupportedMcs (hdr->GetAddr2 (), mcs);
                                  //here should add a control to add basic MCS when it is implemented
                                }
                            }
                        }
                    }
                  if (GetHtSupported ())
                    {
                      ExtendedCapabilities extendedCapabilities = reassocReq.GetExtendedCapabilities ();
                      //TODO: to be completed
                    }
                  if (GetHeSupported ())
                    {
                      HeCapabilities heCapabilities = reassocReq.GetHeCapabilities ();
                      if (heCapabilities.GetSupportedMcsAndNss () != 0)
                        {
                          GetWifiRemoteStationManager ()->AddStationHeCapabilities (from, heCapabilities);
                          for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HE))
                            {
                              if (heCapabilities.IsSupportedTxMcs (mcs.GetMcsValue ()))
                                {
                                  GetWifiRemoteStationManager ()->AddSupportedMcs (hdr->GetAddr2 (), mcs);
                                  //here should add a control to add basic MCS when it is implemented
                                }
                            }
                        }
                    }
                  GetWifiRemoteStationManager ()->RecordWaitAssocTxOk (from);
                  NS_LOG_DEBUG ("Send reassociation response with success status");
                  SendAssocResp (hdr->GetAddr2 (), true, true);
                }
              return;
            }
          else if (hdr->IsDisassociation ())
            {
              NS_LOG_DEBUG ("Disassociation received from " << from);
              GetWifiRemoteStationManager ()->RecordDisassociated (from);
              for (auto it = m_staList.begin (); it != m_staList.end (); ++it)
                {
                  if (it->second == from)
                    {
                      m_staList.erase (it);
                      m_deAssocLogger (it->first, it->second);
                      if (GetWifiRemoteStationManager ()->GetDsssSupported (from) && !GetWifiRemoteStationManager ()->GetErpOfdmSupported (from))
                        {
                          m_numNonErpStations--;
                        }
                      if (!GetWifiRemoteStationManager ()->GetHtSupported (from))
                        {
                          m_numNonHtStations--;
                        }
                      UpdateShortSlotTimeEnabled ();
                      UpdateShortPreambleEnabled ();
                      break;
                    }
                }
              return;
            }
        }
    }

  //Invoke the receive handler of our parent class to deal with any
  //other frames. Specifically, this will handle Block Ack-related
  //Management Action frames.
  WifiMac::Receive (Create<WifiMacQueueItem> (packet, *hdr));
}

void
ApWifiMac::DeaggregateAmsduAndForward (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  for (auto& i : *PeekPointer (mpdu))
    {
      if (i.second.GetDestinationAddr () == GetAddress ())
        {
          ForwardUp (i.first, i.second.GetSourceAddr (),
                     i.second.GetDestinationAddr ());
        }
      else
        {
          Mac48Address from = i.second.GetSourceAddr ();
          Mac48Address to = i.second.GetDestinationAddr ();
          NS_LOG_DEBUG ("forwarding QoS frame from=" << from << ", to=" << to);
          ForwardDown (i.first->Copy (), from, to, mpdu->GetHeader ().GetQosTid ());
        }
    }
}

void
ApWifiMac::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  m_beaconTxop->Initialize ();
  m_beaconEvent.Cancel ();
  if (m_enableBeaconGeneration)
    {
      if (m_enableBeaconJitter)
        {
          Time jitter = MicroSeconds (static_cast<int64_t> (m_beaconJitter->GetValue (0, 1) * (GetBeaconInterval ().GetMicroSeconds ())));
          NS_LOG_DEBUG ("Scheduling initial beacon for access point " << GetAddress () << " at time " << jitter);
          m_beaconEvent = Simulator::Schedule (jitter, &ApWifiMac::SendOneBeacon, this);
        }
      else
        {
          NS_LOG_DEBUG ("Scheduling initial beacon for access point " << GetAddress () << " at time 0");
          m_beaconEvent = Simulator::ScheduleNow (&ApWifiMac::SendOneBeacon, this);
        }
    }
  NS_ABORT_IF (!TraceConnectWithoutContext ("AckedMpdu", MakeCallback (&ApWifiMac::TxOk, this)));
  NS_ABORT_IF (!TraceConnectWithoutContext ("DroppedMpdu", MakeCallback (&ApWifiMac::TxFailed, this)));
  WifiMac::DoInitialize ();
  UpdateShortSlotTimeEnabled ();
  UpdateShortPreambleEnabled ();
}

bool
ApWifiMac::GetUseNonErpProtection (void) const
{
  bool useProtection = (m_numNonErpStations > 0) && m_enableNonErpProtection;
  GetWifiRemoteStationManager ()->SetUseNonErpProtection (useProtection);
  return useProtection;
}

uint16_t
ApWifiMac::GetNextAssociationId (void)
{
  //Return the first free AID value between 1 and 2007
  for (uint16_t nextAid = 1; nextAid <= 2007; nextAid++)
    {
      if (m_staList.find (nextAid) == m_staList.end ())
        {
          return nextAid;
        }
    }
  NS_FATAL_ERROR ("No free association ID available!");
  return 0;
}

const std::map<uint16_t, Mac48Address>&
ApWifiMac::GetStaList (void) const
{
  return m_staList;
}

uint16_t
ApWifiMac::GetAssociationId (Mac48Address addr) const
{
  return GetWifiRemoteStationManager ()->GetAssociationId (addr);
}

uint8_t
ApWifiMac::GetBufferStatus (uint8_t tid, Mac48Address address) const
{
  auto it = m_bufferStatus.find (WifiAddressTidPair (address, tid));
  if (it == m_bufferStatus.end ()
      || it->second.timestamp + m_bsrLifetime < Simulator::Now ())
    {
      return 255;
    }
  return it->second.value;
}

void
ApWifiMac::SetBufferStatus (uint8_t tid, Mac48Address address, uint8_t size)
{
  if (size == 255)
    {
      // no point in storing an unspecified size
      m_bufferStatus.erase (WifiAddressTidPair (address, tid));
    }
  else
    {
      m_bufferStatus[WifiAddressTidPair (address, tid)] = {size, Simulator::Now ()};
    }
}

uint8_t
ApWifiMac::GetMaxBufferStatus (Mac48Address address) const
{
  uint8_t maxSize = 0;
  bool found = false;

  for (uint8_t tid = 0; tid < 8; tid++)
    {
      uint8_t size = GetBufferStatus (tid, address);
      if (size != 255)
        {
          maxSize = std::max (maxSize, size);
          found = true;
        }
    }

  if (found)
    {
      return maxSize;
    }
  return 255;
}

} //namespace ns3
