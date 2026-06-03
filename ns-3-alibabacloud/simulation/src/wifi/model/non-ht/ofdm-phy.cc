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

#include "ofdm-phy.h"
#include "ofdm-ppdu.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-utils.h"
#include "ns3/interference-helper.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include <array>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OfdmPhy");

/*******************************************************
 *       OFDM PHY (IEEE 802.11-2016, clause 17)
 *******************************************************/

/* *NS_CHECK_STYLE_OFF* */
const PhyEntity::PpduFormats OfdmPhy::m_ofdmPpduFormats {
  { WIFI_PREAMBLE_LONG, { WIFI_PPDU_FIELD_PREAMBLE,      //STF + LTF
                          WIFI_PPDU_FIELD_NON_HT_HEADER, //SIG
                          WIFI_PPDU_FIELD_DATA } }
};

const PhyEntity::ModulationLookupTable OfdmPhy::m_ofdmModulationLookupTable {
  // Unique name                Code rate           Constellation size
  { "OfdmRate6Mbps",          { WIFI_CODE_RATE_1_2, 2 } },  // 20 MHz
  { "OfdmRate9Mbps",          { WIFI_CODE_RATE_3_4, 2 } },  //  |
  { "OfdmRate12Mbps",         { WIFI_CODE_RATE_1_2, 4 } },  //  V
  { "OfdmRate18Mbps",         { WIFI_CODE_RATE_3_4, 4 } },
  { "OfdmRate24Mbps",         { WIFI_CODE_RATE_1_2, 16 } },
  { "OfdmRate36Mbps",         { WIFI_CODE_RATE_3_4, 16 } },
  { "OfdmRate48Mbps",         { WIFI_CODE_RATE_2_3, 64 } },
  { "OfdmRate54Mbps",         { WIFI_CODE_RATE_3_4, 64 } },
  { "OfdmRate3MbpsBW10MHz",   { WIFI_CODE_RATE_1_2, 2 } },  // 10 MHz
  { "OfdmRate4_5MbpsBW10MHz", { WIFI_CODE_RATE_3_4, 2 } },  //  |
  { "OfdmRate6MbpsBW10MHz",   { WIFI_CODE_RATE_1_2, 4 } },  //  V
  { "OfdmRate9MbpsBW10MHz",   { WIFI_CODE_RATE_3_4, 4 } },
  { "OfdmRate12MbpsBW10MHz",  { WIFI_CODE_RATE_1_2, 16 } },
  { "OfdmRate18MbpsBW10MHz",  { WIFI_CODE_RATE_3_4, 16 } },
  { "OfdmRate24MbpsBW10MHz",  { WIFI_CODE_RATE_2_3, 64 } },
  { "OfdmRate27MbpsBW10MHz",  { WIFI_CODE_RATE_3_4, 64 } },
  { "OfdmRate1_5MbpsBW5MHz",  { WIFI_CODE_RATE_1_2, 2 } },  //  5 MHz
  { "OfdmRate2_25MbpsBW5MHz", { WIFI_CODE_RATE_3_4, 2 } },  //  |
  { "OfdmRate3MbpsBW5MHz",    { WIFI_CODE_RATE_1_2, 4 } },  //  V
  { "OfdmRate4_5MbpsBW5MHz",  { WIFI_CODE_RATE_3_4, 4 } },
  { "OfdmRate6MbpsBW5MHz",    { WIFI_CODE_RATE_1_2, 16 } },
  { "OfdmRate9MbpsBW5MHz",    { WIFI_CODE_RATE_3_4, 16 } },
  { "OfdmRate12MbpsBW5MHz",   { WIFI_CODE_RATE_2_3, 64 } },
  { "OfdmRate13_5MbpsBW5MHz", { WIFI_CODE_RATE_3_4, 64 } }
};

/// OFDM rates in bits per second for each bandwidth (MHz)
const std::map<uint16_t, std::array<uint64_t, 8> > s_ofdmRatesBpsList =
   {{ 20, // MHz
     {  6000000,  9000000, 12000000, 18000000,
       24000000, 36000000, 48000000, 54000000 }},
   { 10, // MHz
     {  3000000,  4500000,  6000000,  9000000,
       12000000, 18000000, 24000000, 27000000 }},
   { 5, // MHz
     {  1500000,  2250000,  3000000,  4500000,
        6000000,  9000000, 12000000, 13500000 }}};

/* *NS_CHECK_STYLE_ON* */

/**
 * Get the array of possible OFDM rates for each bandwidth (MHz).
 *
 * \return the OFDM rates in bits per second
 */
const std::map<uint16_t, std::array<uint64_t, 8> >& GetOfdmRatesBpsList (void)
{
  return s_ofdmRatesBpsList;
};


OfdmPhy::OfdmPhy (OfdmPhyVariant variant /* = OFDM_PHY_DEFAULT */, bool buildModeList /* = true */)
{
  NS_LOG_FUNCTION (this << variant << buildModeList);

  if (buildModeList)
    {
      auto bwRatesMap = GetOfdmRatesBpsList ();

      switch (variant)
        {
          case OFDM_PHY_DEFAULT:
            for (const auto & rate : bwRatesMap.at (20))
              {
                WifiMode mode = GetOfdmRate (rate, 20);
                NS_LOG_LOGIC ("Add " << mode << " to list");
                m_modeList.emplace_back (mode);
              }
            break;
          case OFDM_PHY_10_MHZ:
            for (const auto & rate : bwRatesMap.at (10))
              {
                WifiMode mode = GetOfdmRate (rate, 10);
                NS_LOG_LOGIC ("Add " << mode << " to list");
                m_modeList.emplace_back (mode);
              }
            break;
          case OFDM_PHY_5_MHZ:
            for (const auto & rate : bwRatesMap.at (5))
              {
                WifiMode mode = GetOfdmRate (rate, 5);
                NS_LOG_LOGIC ("Add " << mode << " to list");
                m_modeList.emplace_back (mode);
              }
            break;
          default:
            NS_ABORT_MSG ("Unsupported 11a OFDM variant");
        }
    }
}

OfdmPhy::~OfdmPhy ()
{
  NS_LOG_FUNCTION (this);
}

WifiMode
OfdmPhy::GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const
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
OfdmPhy::GetHeaderMode (const WifiTxVector& txVector) const
{
  switch (txVector.GetChannelWidth ())
    {
      case 5:
        return GetOfdmRate1_5MbpsBW5MHz ();
      case 10:
        return GetOfdmRate3MbpsBW10MHz ();
      case 20:
      default:
        //Section 17.3.2 "PPDU frame format"; IEEE Std 802.11-2016.
        //Actually this is only the first part of the PhyHeader,
        //because the last 16 bits of the PhyHeader are using the
        //same mode of the payload
        return GetOfdmRate6Mbps ();
    }
}

const PhyEntity::PpduFormats &
OfdmPhy::GetPpduFormats (void) const
{
  return m_ofdmPpduFormats;
}

Time
OfdmPhy::GetDuration (WifiPpduField field, const WifiTxVector& txVector) const
{
  switch (field)
    {
      case WIFI_PPDU_FIELD_PREAMBLE:
        return GetPreambleDuration (txVector); //L-STF + L-LTF
      case WIFI_PPDU_FIELD_NON_HT_HEADER:
        return GetHeaderDuration (txVector); //L-SIG
      default:
        return PhyEntity::GetDuration (field, txVector);
    }
}

Time
OfdmPhy::GetPreambleDuration (const WifiTxVector& txVector) const
{
  switch (txVector.GetChannelWidth ())
    {
      case 20:
      default:
        //Section 17.3.3 "PHY preamble (SYNC)" Figure 17-4 "OFDM training structure"
        //also Section 17.3.2.3 "Modulation-dependent parameters" Table 17-4 "Modulation-dependent parameters"; IEEE Std 802.11-2016
        return MicroSeconds (16);
      case 10:
        //Section 17.3.3 "PHY preamble (SYNC)" Figure 17-4 "OFDM training structure"
        //also Section 17.3.2.3 "Modulation-dependent parameters" Table 17-4 "Modulation-dependent parameters"; IEEE Std 802.11-2016
        return MicroSeconds (32);
      case 5:
        //Section 17.3.3 "PHY preamble (SYNC)" Figure 17-4 "OFDM training structure"
        //also Section 17.3.2.3 "Modulation-dependent parameters" Table 17-4 "Modulation-dependent parameters"; IEEE Std 802.11-2016
        return MicroSeconds (64);
    }
}

Time
OfdmPhy::GetHeaderDuration (const WifiTxVector& txVector) const
{
  switch (txVector.GetChannelWidth ())
    {
      case 20:
      default:
        //Section 17.3.3 "PHY preamble (SYNC)" and Figure 17-4 "OFDM training structure"; IEEE Std 802.11-2016
        //also Section 17.3.2.4 "Timing related parameters" Table 17-5 "Timing-related parameters"; IEEE Std 802.11-2016
        //We return the duration of the SIGNAL field only, since the
        //SERVICE field (which strictly speaking belongs to the PHY
        //header, see Section 17.3.2 and Figure 17-1) is sent using the
        //payload mode.
        return MicroSeconds (4);
      case 10:
        //Section 17.3.2.4 "Timing related parameters" Table 17-5 "Timing-related parameters"; IEEE Std 802.11-2016
        return MicroSeconds (8);
      case 5:
        //Section 17.3.2.4 "Timing related parameters" Table 17-5 "Timing-related parameters"; IEEE Std 802.11-2016
        return MicroSeconds (16);
    }
}

Time
OfdmPhy::GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType /* mpdutype */,
                             bool /* incFlag */, uint32_t & /* totalAmpduSize */, double & /* totalAmpduNumSymbols */,
                             uint16_t /* staId */) const
{
  //(Section 17.3.2.4 "Timing related parameters" Table 17-5 "Timing-related parameters"; IEEE Std 802.11-2016
  //corresponds to T_{SYM} in the table)
  Time symbolDuration = MicroSeconds (4);

  double numDataBitsPerSymbol = txVector.GetMode ().GetDataRate (txVector) * symbolDuration.GetNanoSeconds () / 1e9;

  //The number of OFDM symbols in the data field when BCC encoding
  //is used is given in equation 19-32 of the IEEE 802.11-2016 standard.
  double numSymbols = lrint (ceil ((GetNumberServiceBits () + size * 8.0 + 6.0) / (numDataBitsPerSymbol)));

  Time payloadDuration = FemtoSeconds (static_cast<uint64_t> (numSymbols * symbolDuration.GetFemtoSeconds ()));
  payloadDuration += GetSignalExtension (band);
  return payloadDuration;
}

uint8_t
OfdmPhy::GetNumberServiceBits (void) const
{
  return 16;
}

Time
OfdmPhy::GetSignalExtension (WifiPhyBand band) const
{
  return (band == WIFI_PHY_BAND_2_4GHZ) ? MicroSeconds (6) : MicroSeconds (0);
}

Ptr<WifiPpdu>
OfdmPhy::BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time /* ppduDuration */)
{
  NS_LOG_FUNCTION (this << psdus << txVector);
  return Create<OfdmPpdu> (psdus.begin ()->second, txVector, m_wifiPhy->GetPhyBand (),
                           ObtainNextUid (txVector));
}

PhyEntity::PhyFieldRxStatus
OfdmPhy::DoEndReceiveField (WifiPpduField field, Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << field << *event);
  if (field == WIFI_PPDU_FIELD_NON_HT_HEADER)
    {
      return EndReceiveHeader (event); //L-SIG
    }
  return PhyEntity::DoEndReceiveField (field, event);
}

PhyEntity::PhyFieldRxStatus
OfdmPhy::EndReceiveHeader (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  SnrPer snrPer = GetPhyHeaderSnrPer (WIFI_PPDU_FIELD_NON_HT_HEADER, event);
  NS_LOG_DEBUG ("L-SIG: SNR(dB)=" << RatioToDb (snrPer.snr) << ", PER=" << snrPer.per);
  PhyFieldRxStatus status (GetRandomValue () > snrPer.per);
  if (status.isSuccess)
    {
      NS_LOG_DEBUG ("Received non-HT PHY header");
      if (!IsAllConfigSupported (WIFI_PPDU_FIELD_NON_HT_HEADER, event->GetPpdu ()))
        {
          status = PhyFieldRxStatus (false, UNSUPPORTED_SETTINGS, DROP);
        }
    }
  else
    {
      NS_LOG_DEBUG ("Abort reception because non-HT PHY header reception failed");
      status.reason = L_SIG_FAILURE;
      status.actionIfFailure = ABORT;
    }
  return status;
}

bool
OfdmPhy::IsChannelWidthSupported (Ptr<const WifiPpdu> ppdu) const
{
  uint16_t channelWidth = ppdu->GetTxVector ().GetChannelWidth ();
  if ((channelWidth >= 40) && (channelWidth > m_wifiPhy->GetChannelWidth ()))
    {
      NS_LOG_DEBUG ("Packet reception could not be started because not enough channel width (" << channelWidth << " vs " << m_wifiPhy->GetChannelWidth () << ")");
      return false;
    }
  return true;
}

bool
OfdmPhy::IsAllConfigSupported (WifiPpduField /* field */, Ptr<const WifiPpdu> ppdu) const
{
  if (!IsChannelWidthSupported (ppdu))
    {
      return false;
    }
  return IsConfigSupported (ppdu);
}

Ptr<SpectrumValue>
OfdmPhy::GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const
{
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  uint16_t centerFrequency = GetCenterFrequencyForChannelWidth (txVector);
  uint16_t channelWidth = txVector.GetChannelWidth ();
  NS_LOG_FUNCTION (this << centerFrequency << channelWidth << txPowerW);
  const auto & txMaskRejectionParams = GetTxMaskRejectionParams ();
  Ptr<SpectrumValue> v = WifiSpectrumValueHelper::CreateOfdmTxPowerSpectralDensity (centerFrequency, channelWidth, txPowerW, GetGuardBandwidth (channelWidth),
                                                                                    std::get<0> (txMaskRejectionParams), std::get<1> (txMaskRejectionParams), std::get<2> (txMaskRejectionParams));
  return v;
}

void
OfdmPhy::InitializeModes (void)
{
  for (const auto & ratesPerBw : GetOfdmRatesBpsList ())
    {
      for (const auto & rate : ratesPerBw.second)
        {
          GetOfdmRate (rate, ratesPerBw.first);
        }
    }
}

WifiMode
OfdmPhy::GetOfdmRate (uint64_t rate, uint16_t bw)
{
  switch (bw)
    {
      case 20:
        switch (rate)
          {
            case 6000000:
              return GetOfdmRate6Mbps ();
            case 9000000:
              return GetOfdmRate9Mbps ();
            case 12000000:
              return GetOfdmRate12Mbps ();
            case 18000000:
              return GetOfdmRate18Mbps ();
            case 24000000:
              return GetOfdmRate24Mbps ();
            case 36000000:
              return GetOfdmRate36Mbps ();
            case 48000000:
              return GetOfdmRate48Mbps ();
            case 54000000:
              return GetOfdmRate54Mbps ();
            default:
              NS_ABORT_MSG ("Inexistent rate (" << rate << " bps) requested for 11a OFDM (default)");
              return WifiMode ();
          }
        break;
      case 10:
        switch (rate)
          {
            case 3000000:
              return GetOfdmRate3MbpsBW10MHz ();
            case 4500000:
              return GetOfdmRate4_5MbpsBW10MHz ();
            case 6000000:
              return GetOfdmRate6MbpsBW10MHz ();
            case 9000000:
              return GetOfdmRate9MbpsBW10MHz ();
            case 12000000:
              return GetOfdmRate12MbpsBW10MHz ();
            case 18000000:
              return GetOfdmRate18MbpsBW10MHz ();
            case 24000000:
              return GetOfdmRate24MbpsBW10MHz ();
            case 27000000:
              return GetOfdmRate27MbpsBW10MHz ();
            default:
              NS_ABORT_MSG ("Inexistent rate (" << rate << " bps) requested for 11a OFDM (10 MHz)");
              return WifiMode ();
          }
        break;
      case 5:
        switch (rate)
          {
            case 1500000:
              return GetOfdmRate1_5MbpsBW5MHz ();
            case 2250000:
              return GetOfdmRate2_25MbpsBW5MHz ();
            case 3000000:
              return GetOfdmRate3MbpsBW5MHz ();
            case 4500000:
              return GetOfdmRate4_5MbpsBW5MHz ();
            case 6000000:
              return GetOfdmRate6MbpsBW5MHz ();
            case 9000000:
              return GetOfdmRate9MbpsBW5MHz ();
            case 12000000:
              return GetOfdmRate12MbpsBW5MHz ();
            case 13500000:
              return GetOfdmRate13_5MbpsBW5MHz ();
            default:
              NS_ABORT_MSG ("Inexistent rate (" << rate << " bps) requested for 11a OFDM (5 MHz)");
              return WifiMode ();
          }
        break;
      default:
        NS_ABORT_MSG ("Inexistent bandwidth (" << +bw << " MHz) requested for 11a OFDM");
        return WifiMode ();
    }
}

#define GET_OFDM_MODE(x, f) \
WifiMode \
OfdmPhy::Get ## x (void) \
{ \
  static WifiMode mode = CreateOfdmMode (#x, f); \
  return mode; \
}; \

// 20 MHz channel rates (default)
GET_OFDM_MODE (OfdmRate6Mbps,  true)
GET_OFDM_MODE (OfdmRate9Mbps,  false)
GET_OFDM_MODE (OfdmRate12Mbps, true)
GET_OFDM_MODE (OfdmRate18Mbps, false)
GET_OFDM_MODE (OfdmRate24Mbps, true)
GET_OFDM_MODE (OfdmRate36Mbps, false)
GET_OFDM_MODE (OfdmRate48Mbps, false)
GET_OFDM_MODE (OfdmRate54Mbps, false)
// 10 MHz channel rates
GET_OFDM_MODE (OfdmRate3MbpsBW10MHz,   true)
GET_OFDM_MODE (OfdmRate4_5MbpsBW10MHz, false)
GET_OFDM_MODE (OfdmRate6MbpsBW10MHz,   true)
GET_OFDM_MODE (OfdmRate9MbpsBW10MHz,   false)
GET_OFDM_MODE (OfdmRate12MbpsBW10MHz,  true)
GET_OFDM_MODE (OfdmRate18MbpsBW10MHz,  false)
GET_OFDM_MODE (OfdmRate24MbpsBW10MHz,  false)
GET_OFDM_MODE (OfdmRate27MbpsBW10MHz,  false)
// 5 MHz channel rates
GET_OFDM_MODE (OfdmRate1_5MbpsBW5MHz,  true)
GET_OFDM_MODE (OfdmRate2_25MbpsBW5MHz, false)
GET_OFDM_MODE (OfdmRate3MbpsBW5MHz,    true)
GET_OFDM_MODE (OfdmRate4_5MbpsBW5MHz,  false)
GET_OFDM_MODE (OfdmRate6MbpsBW5MHz,    true)
GET_OFDM_MODE (OfdmRate9MbpsBW5MHz,    false)
GET_OFDM_MODE (OfdmRate12MbpsBW5MHz,   false)
GET_OFDM_MODE (OfdmRate13_5MbpsBW5MHz, false)
#undef GET_OFDM_MODE

WifiMode
OfdmPhy::CreateOfdmMode (std::string uniqueName, bool isMandatory)
{
  // Check whether uniqueName is in lookup table
  const auto it = m_ofdmModulationLookupTable.find (uniqueName);
  NS_ASSERT_MSG (it != m_ofdmModulationLookupTable.end (), "OFDM mode cannot be created because it is not in the lookup table!");

  return WifiModeFactory::CreateWifiMode (uniqueName,
                                          WIFI_MOD_CLASS_OFDM,
                                          isMandatory,
                                          MakeBoundCallback (&GetCodeRate, uniqueName),
                                          MakeBoundCallback (&GetConstellationSize, uniqueName),
                                          MakeCallback (&GetPhyRateFromTxVector),
                                          MakeCallback (&GetDataRateFromTxVector),
                                          MakeCallback (&IsAllowed));
}

WifiCodeRate
OfdmPhy::GetCodeRate (const std::string& name)
{
  return m_ofdmModulationLookupTable.at (name).first;
}

uint16_t
OfdmPhy::GetConstellationSize (const std::string& name)
{
  return m_ofdmModulationLookupTable.at (name).second;
}

uint64_t
OfdmPhy::GetPhyRate (const std::string& name, uint16_t channelWidth)
{
  WifiCodeRate codeRate = GetCodeRate (name);
  uint64_t dataRate = GetDataRate (name, channelWidth);
  return CalculatePhyRate (codeRate, dataRate);
}

uint64_t
OfdmPhy::CalculatePhyRate (WifiCodeRate codeRate, uint64_t dataRate)
{
  return (dataRate / GetCodeRatio (codeRate));
}

uint64_t
OfdmPhy::GetPhyRateFromTxVector (const WifiTxVector& txVector, uint16_t /* staId */)
{
  return GetPhyRate (txVector.GetMode ().GetUniqueName (),
                     txVector.GetChannelWidth ());
}

double
OfdmPhy::GetCodeRatio (WifiCodeRate codeRate)
{
  switch (codeRate)
    {
      case WIFI_CODE_RATE_3_4:
        return (3.0 / 4.0);
      case WIFI_CODE_RATE_2_3:
        return (2.0 / 3.0);
      case WIFI_CODE_RATE_1_2:
        return (1.0 / 2.0);
      case WIFI_CODE_RATE_UNDEFINED:
      default:
        NS_FATAL_ERROR ("trying to get code ratio for undefined coding rate");
        return 0;
    }
}

uint64_t
OfdmPhy::GetDataRateFromTxVector (const WifiTxVector& txVector, uint16_t /* staId */)
{
  return GetDataRate (txVector.GetMode ().GetUniqueName (),
                      txVector.GetChannelWidth ());
}

uint64_t
OfdmPhy::GetDataRate (const std::string& name, uint16_t channelWidth)
{
  WifiCodeRate codeRate = GetCodeRate (name);
  uint16_t constellationSize = GetConstellationSize (name);
  return CalculateDataRate (codeRate, constellationSize, channelWidth);
}

uint64_t
OfdmPhy::CalculateDataRate (WifiCodeRate codeRate, uint16_t constellationSize, uint16_t channelWidth)
{
  double symbolDuration = 3.2; //in us
  uint16_t guardInterval = 800; //in ns
  if (channelWidth == 10)
    {
      symbolDuration = 6.4;
      guardInterval = 1600;
    }
  else if (channelWidth == 5)
    {
      symbolDuration = 12.8;
      guardInterval = 3200;
    }
  return CalculateDataRate (symbolDuration, guardInterval,
                            48, static_cast<uint16_t> (log2 (constellationSize)),
                            GetCodeRatio (codeRate));
}

uint64_t
OfdmPhy::CalculateDataRate (double symbolDuration, uint16_t guardInterval,
                            uint16_t usableSubCarriers, uint16_t numberOfBitsPerSubcarrier,
                            double codingRate)
{
  double symbolRate = (1 / (symbolDuration + (static_cast<double> (guardInterval) / 1000))) * 1e6;
  return lrint (ceil (symbolRate * usableSubCarriers * numberOfBitsPerSubcarrier * codingRate));
}

bool
OfdmPhy::IsAllowed (const WifiTxVector& /*txVector*/)
{
  return true;
}

uint32_t
OfdmPhy::GetMaxPsduSize (void) const
{
  return 4095;
}

} //namespace ns3

namespace {

/**
 * Constructor class for OFDM modes
 */
static class ConstructorOfdm
{
public:
  ConstructorOfdm ()
  {
    ns3::OfdmPhy::InitializeModes ();
    ns3::WifiPhy::AddStaticPhyEntity (ns3::WIFI_MOD_CLASS_OFDM, ns3::Create<ns3::OfdmPhy> ()); //default variant will do
  }
} g_constructor_ofdm; ///< the constructor for OFDM modes

}
