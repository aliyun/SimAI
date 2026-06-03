/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Orange Labs
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
 * Authors: Rediet <getachew.redieteab@orange.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com> (for logic ported from wifi-phy)
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr> (for logic ported from wifi-phy)
 */

#include <array>
#include "dsss-phy.h"
#include "dsss-ppdu.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-phy.h" //only used for static mode constructor
#include "ns3/wifi-utils.h"
#include "ns3/interference-helper.h"
#include "ns3/simulator.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DsssPhy");

/*******************************************************
 *       HR/DSSS PHY (IEEE 802.11-2016, clause 16)
 *******************************************************/

/* *NS_CHECK_STYLE_OFF* */
const PhyEntity::PpduFormats DsssPhy::m_dsssPpduFormats {
  { WIFI_PREAMBLE_LONG,  { WIFI_PPDU_FIELD_PREAMBLE,      //PHY preamble
                           WIFI_PPDU_FIELD_NON_HT_HEADER, //PHY header
                           WIFI_PPDU_FIELD_DATA } },
  { WIFI_PREAMBLE_SHORT, { WIFI_PPDU_FIELD_PREAMBLE,      //Short PHY preamble
                           WIFI_PPDU_FIELD_NON_HT_HEADER, //Short PHY header
                           WIFI_PPDU_FIELD_DATA } }
};

const PhyEntity::ModulationLookupTable DsssPhy::m_dsssModulationLookupTable {
  // Unique name         Code rate                 Constellation size
  { "DsssRate1Mbps",   { WIFI_CODE_RATE_UNDEFINED, 2 } },
  { "DsssRate2Mbps",   { WIFI_CODE_RATE_UNDEFINED, 4 } },
  { "DsssRate5_5Mbps", { WIFI_CODE_RATE_UNDEFINED, 16 } },
  { "DsssRate11Mbps",  { WIFI_CODE_RATE_UNDEFINED, 256 } }
};
/* *NS_CHECK_STYLE_ON* */

/// DSSS rates in bits per second
static const std::array<uint64_t, 4> s_dsssRatesBpsList = {1000000,  2000000, 5500000, 11000000};

/**
 * Get the array of possible DSSS rates.
 *
 * \return the DSSS rates in bits per second
 */
const std::array<uint64_t, 4>& GetDsssRatesBpsList (void)
{
  return s_dsssRatesBpsList;
};

DsssPhy::DsssPhy ()
{
  NS_LOG_FUNCTION (this);
  for (const auto & rate : GetDsssRatesBpsList ())
    {
      WifiMode mode = GetDsssRate (rate);
      NS_LOG_LOGIC ("Add " << mode << " to list");
      m_modeList.emplace_back (mode);
    }
}

DsssPhy::~DsssPhy ()
{
  NS_LOG_FUNCTION (this);
}

WifiMode
DsssPhy::GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const
{
  switch (field)
    {
      case WIFI_PPDU_FIELD_PREAMBLE: //consider header mode for preamble (useful for InterferenceHelper)
      case WIFI_PPDU_FIELD_NON_HT_HEADER:
        return GetHeaderMode (txVector);
      default:
        return PhyEntity::GetSigMode (field, txVector);
    }
}

WifiMode
DsssPhy::GetHeaderMode (const WifiTxVector& txVector) const
{
  if (txVector.GetPreambleType () == WIFI_PREAMBLE_LONG
      || txVector.GetMode () == GetDsssRate1Mbps ())
    {
      //Section 16.2.3 "PPDU field definitions" and Section 16.2.2.2 "Long PPDU format"; IEEE Std 802.11-2016
      return GetDsssRate1Mbps ();
    }
  else
    {
      //Section 16.2.2.3 "Short PPDU format"; IEEE Std 802.11-2016
      return GetDsssRate2Mbps ();
    }
}

const PhyEntity::PpduFormats &
DsssPhy::GetPpduFormats (void) const
{
  return m_dsssPpduFormats;
}

Time
DsssPhy::GetDuration (WifiPpduField field, const WifiTxVector& txVector) const
{
  if (field == WIFI_PPDU_FIELD_PREAMBLE)
    {
      return GetPreambleDuration (txVector); //SYNC + SFD or shortSYNC + shortSFD
    }
  else if (field == WIFI_PPDU_FIELD_NON_HT_HEADER)
    {
      return GetHeaderDuration (txVector); //PHY header or short PHY header
    }
  else
    {
      return PhyEntity::GetDuration (field, txVector);
    }
}

Time
DsssPhy::GetPreambleDuration (const WifiTxVector& txVector) const
{
  if (txVector.GetPreambleType () == WIFI_PREAMBLE_SHORT
      && (txVector.GetMode ().GetDataRate (22) > 1000000))
    {
      //Section 16.2.2.3 "Short PPDU format" Figure 16-2 "Short PPDU format"; IEEE Std 802.11-2016
      return MicroSeconds (72);
    }
  else
    {
      //Section 16.2.2.2 "Long PPDU format" Figure 16-1 "Long PPDU format"; IEEE Std 802.11-2016
      return MicroSeconds (144);
    }
}

Time
DsssPhy::GetHeaderDuration (const WifiTxVector& txVector) const
{
  if (txVector.GetPreambleType () == WIFI_PREAMBLE_SHORT
      && (txVector.GetMode ().GetDataRate (22) > 1000000))
    {
      //Section 16.2.2.3 "Short PPDU format" and Figure 16-2 "Short PPDU format"; IEEE Std 802.11-2016
      return MicroSeconds (24);
    }
  else
    {
      //Section 16.2.2.2 "Long PPDU format" and Figure 16-1 "Short PPDU format"; IEEE Std 802.11-2016
      return MicroSeconds (48);
    }
}

Time
DsssPhy::GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand /* band */, MpduType /* mpdutype */,
                             bool /* incFlag */, uint32_t & /* totalAmpduSize */, double & /* totalAmpduNumSymbols */,
                             uint16_t /* staId */) const
{
  return MicroSeconds (lrint (ceil ((size * 8.0) / (txVector.GetMode ().GetDataRate (22) / 1.0e6))));
}

Ptr<WifiPpdu>
DsssPhy::BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time ppduDuration)
{
  NS_LOG_FUNCTION (this << psdus << txVector << ppduDuration);
  return Create<DsssPpdu> (psdus.begin ()->second, txVector, ppduDuration,
                           ObtainNextUid (txVector));
}

PhyEntity::PhyFieldRxStatus
DsssPhy::DoEndReceiveField (WifiPpduField field, Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << field << *event);
  if (field == WIFI_PPDU_FIELD_NON_HT_HEADER)
    {
      return EndReceiveHeader (event); //PHY header or short PHY header
    }
  return PhyEntity::DoEndReceiveField (field, event);
}

PhyEntity::PhyFieldRxStatus
DsssPhy::EndReceiveHeader (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  SnrPer snrPer = GetPhyHeaderSnrPer (WIFI_PPDU_FIELD_NON_HT_HEADER, event);
  NS_LOG_DEBUG ("Long/Short PHY header: SNR(dB)=" << RatioToDb (snrPer.snr) << ", PER=" << snrPer.per);
  PhyFieldRxStatus status (GetRandomValue () > snrPer.per);
  if (status.isSuccess)
    {
      NS_LOG_DEBUG ("Received long/short PHY header");
      if (!IsConfigSupported (event->GetPpdu ()))
        {
          status = PhyFieldRxStatus (false, UNSUPPORTED_SETTINGS, DROP);
        }
    }
  else
    {
      NS_LOG_DEBUG ("Abort reception because long/short PHY header reception failed");
      status.reason = L_SIG_FAILURE;
      status.actionIfFailure = ABORT;
    }
  return status;
}

uint16_t
DsssPhy::GetRxChannelWidth (const WifiTxVector& txVector) const
{
  if (m_wifiPhy->GetChannelWidth () > 20)
    {
      /*
       * This is a workaround necessary with HE-capable PHYs,
       * since their DSSS entity will reuse its RxSpectrumModel.
       * Without this hack, SpectrumWifiPhy::GetBand will crash.
       * FIXME: see issue #402 for a better solution.
       */
      return 20;
    }
  return PhyEntity::GetRxChannelWidth (txVector);
}

Ptr<SpectrumValue>
DsssPhy::GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const
{
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  uint16_t centerFrequency = GetCenterFrequencyForChannelWidth (txVector);
  uint16_t channelWidth = txVector.GetChannelWidth ();
  NS_LOG_FUNCTION (this << centerFrequency << channelWidth << txPowerW);
  NS_ABORT_MSG_IF (channelWidth != 22, "Invalid channel width for DSSS");
  Ptr<SpectrumValue> v = WifiSpectrumValueHelper::CreateDsssTxPowerSpectralDensity (centerFrequency, txPowerW, GetGuardBandwidth (channelWidth));
  return v;
}

void
DsssPhy::InitializeModes (void)
{
  for (const auto & rate : GetDsssRatesBpsList ())
    {
      GetDsssRate (rate);
    }
}

WifiMode
DsssPhy::GetDsssRate (uint64_t rate)
{
  switch (rate)
    {
      case 1000000:
        return GetDsssRate1Mbps ();
      case 2000000:
        return GetDsssRate2Mbps ();
      case 5500000:
        return GetDsssRate5_5Mbps ();
      case 11000000:
        return GetDsssRate11Mbps ();
      default:
        NS_ABORT_MSG ("Inexistent rate (" << rate << " bps) requested for HR/DSSS");
        return WifiMode ();
    }
}

#define GET_DSSS_MODE(x, m) \
WifiMode \
DsssPhy::Get ## x (void) \
{ \
  static WifiMode mode = CreateDsssMode (#x, WIFI_MOD_CLASS_ ## m); \
  return mode; \
}; \

// Clause 15 rates (DSSS)
GET_DSSS_MODE (DsssRate1Mbps,   DSSS)
GET_DSSS_MODE (DsssRate2Mbps,   DSSS)
// Clause 16 rates (HR/DSSS)
GET_DSSS_MODE (DsssRate5_5Mbps, HR_DSSS)
GET_DSSS_MODE (DsssRate11Mbps,  HR_DSSS)
#undef GET_DSSS_MODE

WifiMode
DsssPhy::CreateDsssMode (std::string uniqueName,
                         WifiModulationClass modClass)
{
  // Check whether uniqueName is in lookup table
  const auto it = m_dsssModulationLookupTable.find (uniqueName);
  NS_ASSERT_MSG (it != m_dsssModulationLookupTable.end (), "DSSS or HR/DSSS mode cannot be created because it is not in the lookup table!");
  NS_ASSERT_MSG (modClass == WIFI_MOD_CLASS_DSSS || modClass == WIFI_MOD_CLASS_HR_DSSS, "DSSS or HR/DSSS mode must be either WIFI_MOD_CLASS_DSSS or WIFI_MOD_CLASS_HR_DSSS!");

  return WifiModeFactory::CreateWifiMode (uniqueName,
                                          modClass,
                                          true,
                                          MakeBoundCallback (&GetCodeRate, uniqueName),
                                          MakeBoundCallback (&GetConstellationSize, uniqueName),
                                          MakeCallback (&GetDataRateFromTxVector), //PhyRate is equivalent to DataRate
                                          MakeCallback (&GetDataRateFromTxVector),
                                          MakeCallback (&IsAllowed));
}

WifiCodeRate
DsssPhy::GetCodeRate (const std::string& name)
{
  return m_dsssModulationLookupTable.at (name).first;
}

uint16_t
DsssPhy::GetConstellationSize (const std::string& name)
{
  return m_dsssModulationLookupTable.at (name).second;
}

uint64_t
DsssPhy::GetDataRateFromTxVector (const WifiTxVector& txVector, uint16_t /* staId */)
{
  WifiMode mode = txVector.GetMode ();
  return DsssPhy::GetDataRate (mode.GetUniqueName (),
                               mode.GetModulationClass ());
}

uint64_t
DsssPhy::GetDataRate (const std::string& name, WifiModulationClass modClass)
{
  uint16_t constellationSize = GetConstellationSize (name);
  uint16_t divisor = 0;
  if (modClass == WIFI_MOD_CLASS_DSSS)
    {
      divisor = 11;
    }
  else if (modClass == WIFI_MOD_CLASS_HR_DSSS)
    {
      divisor = 8;
    }
  else
    {
      NS_FATAL_ERROR ("Incorrect modulation class, must specify either WIFI_MOD_CLASS_DSSS or WIFI_MOD_CLASS_HR_DSSS!");
    }
  uint16_t numberOfBitsPerSubcarrier = static_cast<uint16_t> (log2 (constellationSize));
  uint64_t dataRate = ((11000000 / divisor) * numberOfBitsPerSubcarrier);
  return dataRate;
}

bool
DsssPhy::IsAllowed (const WifiTxVector& /*txVector*/)
{
  return true;
}

uint32_t
DsssPhy::GetMaxPsduSize (void) const
{
  return 4095;
}

} //namespace ns3

namespace {

/**
 * Constructor class for DSSS modes
 */
static class ConstructorDsss
{
public:
  ConstructorDsss ()
  {
    ns3::DsssPhy::InitializeModes ();
    ns3::Ptr<ns3::DsssPhy> phyEntity = ns3::Create<ns3::DsssPhy> ();
    ns3::WifiPhy::AddStaticPhyEntity (ns3::WIFI_MOD_CLASS_HR_DSSS, phyEntity);
    ns3::WifiPhy::AddStaticPhyEntity (ns3::WIFI_MOD_CLASS_DSSS, phyEntity); //use same entity when plain DSSS modes are used
  }
} g_constructor_dsss; ///< the constructor for DSSS modes

}
