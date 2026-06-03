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
 */

#include "ht-phy.h"
#include "ht-ppdu.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-utils.h"
#include "ns3/interference-helper.h"
#include "ns3/log.h"
#include "ns3/assert.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HtPhy");

/*******************************************************
 *       HT PHY (IEEE 802.11-2016, clause 19)
 *******************************************************/

/* *NS_CHECK_STYLE_OFF* */
const PhyEntity::PpduFormats HtPhy::m_htPpduFormats {
  { WIFI_PREAMBLE_HT_MF, { WIFI_PPDU_FIELD_PREAMBLE,      //L-STF + L-LTF
                           WIFI_PPDU_FIELD_NON_HT_HEADER, //L-SIG
                           WIFI_PPDU_FIELD_HT_SIG,        //HT-SIG
                           WIFI_PPDU_FIELD_TRAINING,      //HT-STF + HT-LTFs
                           WIFI_PPDU_FIELD_DATA } }
};
/* *NS_CHECK_STYLE_ON* */

HtPhy::HtPhy (uint8_t maxNss /* = 1 */, bool buildModeList /* = true */)
  : OfdmPhy (OFDM_PHY_DEFAULT, false) //don't add OFDM modes to list
{
  NS_LOG_FUNCTION (this << +maxNss << buildModeList);
  m_maxSupportedNss = maxNss;
  m_bssMembershipSelector = HT_PHY;
  m_maxMcsIndexPerSs = 7;
  m_maxSupportedMcsIndexPerSs = m_maxMcsIndexPerSs;
  if (buildModeList)
    {
      NS_ABORT_MSG_IF (maxNss == 0 || maxNss > 4, "Unsupported max Nss " << +maxNss << " for HT PHY");
      BuildModeList ();
    }
}

HtPhy::~HtPhy ()
{
  NS_LOG_FUNCTION (this);
}

void
HtPhy::BuildModeList (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_modeList.empty ());
  NS_ASSERT (m_bssMembershipSelector == HT_PHY);

  uint8_t index = 0;
  for (uint8_t nss = 1; nss <= m_maxSupportedNss; ++nss)
    {
      for (uint8_t i = 0; i <= m_maxSupportedMcsIndexPerSs; ++i)
        {
          NS_LOG_LOGIC ("Add HtMcs" << +index << " to list");
          m_modeList.emplace_back (CreateHtMcs (index));
          ++index;
        }
      index = 8 * nss;
    }
}

WifiMode
HtPhy::GetMcs (uint8_t index) const
{
  for (const auto & mcs : m_modeList)
    {
      if (mcs.GetMcsValue () == index)
        {
          return mcs;
        }
    }

  // Should have returned if MCS found
  NS_ABORT_MSG ("Unsupported MCS index " << +index << " for this PHY entity");
  return WifiMode ();
}

bool
HtPhy::IsMcsSupported (uint8_t index) const
{
  for (const auto & mcs : m_modeList)
    {
      if (mcs.GetMcsValue () == index)
        {
          return true;
        }
    }
  return false;
}

bool
HtPhy::HandlesMcsModes (void) const
{
  return true;
}

const PhyEntity::PpduFormats &
HtPhy::GetPpduFormats (void) const
{
  return m_htPpduFormats;
}

WifiMode
HtPhy::GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const
{
  switch (field)
    {
      case WIFI_PPDU_FIELD_PREAMBLE: //consider non-HT header mode for preamble (useful for InterferenceHelper)
      case WIFI_PPDU_FIELD_NON_HT_HEADER:
        return GetLSigMode ();
      case WIFI_PPDU_FIELD_TRAINING: //consider HT-SIG mode for training (useful for InterferenceHelper)
      case WIFI_PPDU_FIELD_HT_SIG:
        return GetHtSigMode ();
      default:
        return PhyEntity::GetSigMode (field, txVector);
    }
}

WifiMode
HtPhy::GetLSigMode (void)
{
  return GetOfdmRate6Mbps ();
}

WifiMode
HtPhy::GetHtSigMode (void) const
{
  return GetLSigMode (); //same number of data tones as OFDM (i.e. 48)
}

uint8_t
HtPhy::GetBssMembershipSelector (void) const
{
  return m_bssMembershipSelector;
}

void
HtPhy::SetMaxSupportedMcsIndexPerSs (uint8_t maxIndex)
{
  NS_LOG_FUNCTION (this << +maxIndex);
  NS_ABORT_MSG_IF (maxIndex > m_maxMcsIndexPerSs, "Provided max MCS index " << +maxIndex << " per SS greater than max standard-defined value " << +m_maxMcsIndexPerSs);
  if (maxIndex != m_maxSupportedMcsIndexPerSs)
    {
      NS_LOG_LOGIC ("Rebuild mode list since max MCS index per spatial stream has changed");
      m_maxSupportedMcsIndexPerSs = maxIndex;
      m_modeList.clear ();
      BuildModeList ();
    }
}

uint8_t
HtPhy::GetMaxSupportedMcsIndexPerSs (void) const
{
  return m_maxSupportedMcsIndexPerSs;
}

void
HtPhy::SetMaxSupportedNss (uint8_t maxNss)
{
  NS_LOG_FUNCTION (this << +maxNss);
  NS_ASSERT (m_bssMembershipSelector == HT_PHY);
  NS_ABORT_MSG_IF (maxNss == 0 || maxNss > 4, "Unsupported max Nss " << +maxNss << " for HT PHY");
  if (maxNss != m_maxSupportedNss)
    {
      NS_LOG_LOGIC ("Rebuild mode list since max number of spatial streams has changed");
      m_maxSupportedNss = maxNss;
      m_modeList.clear ();
      BuildModeList ();
    }
}

Time
HtPhy::GetDuration (WifiPpduField field, const WifiTxVector& txVector) const
{
  switch (field)
    {
      case WIFI_PPDU_FIELD_PREAMBLE:
        return MicroSeconds (16); //L-STF + L-LTF or HT-GF-STF + HT-LTF1
      case WIFI_PPDU_FIELD_NON_HT_HEADER:
        return GetLSigDuration (txVector.GetPreambleType ());
      case WIFI_PPDU_FIELD_TRAINING:
        {
          //We suppose here that STBC = 0.
          //If STBC > 0, we need a different mapping between Nss and Nltf
          // (see IEEE 802.11-2016 , section 19.3.9.4.6 "HT-LTF definition").
          uint8_t nDataLtf = 8;
          uint8_t nss = txVector.GetNssMax (); //so as to cover also HE MU case (see section 27.3.10.10 of IEEE P802.11ax/D4.0)
          if (nss < 3)
            {
              nDataLtf = nss;
            }
          else if (nss < 5)
            {
              nDataLtf = 4;
            }
          else if (nss < 7)
            {
              nDataLtf = 6;
            }

          uint8_t nExtensionLtf = (txVector.GetNess () < 3) ? txVector.GetNess () : 4;

          return GetTrainingDuration (txVector, nDataLtf, nExtensionLtf);
        }
      case WIFI_PPDU_FIELD_HT_SIG:
        return GetHtSigDuration ();
      default:
        return PhyEntity::GetDuration (field, txVector);
    }
}

Time
HtPhy::GetLSigDuration (WifiPreamble preamble) const
{
  return MicroSeconds (4);
}

Time
HtPhy::GetTrainingDuration (const WifiTxVector& txVector,
                            uint8_t nDataLtf, uint8_t nExtensionLtf /* = 0 */) const
{
  NS_ABORT_MSG_IF (nDataLtf == 0 || nDataLtf > 4 || nExtensionLtf > 4 || (nDataLtf + nExtensionLtf) > 5,
                   "Unsupported combination of data (" << +nDataLtf << ")  and extension (" << +nExtensionLtf << ")  LTFs numbers for HT"); //see IEEE 802.11-2016, section 19.3.9.4.6 "HT-LTF definition"
  Time duration = MicroSeconds (4) * (nDataLtf + nExtensionLtf);
  return MicroSeconds (4) * (1 /* HT-STF */ + nDataLtf + nExtensionLtf);
}

Time
HtPhy::GetHtSigDuration (void) const
{
  return MicroSeconds (8); //HT-SIG
}

Time
HtPhy::GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype,
                           bool incFlag, uint32_t &totalAmpduSize, double &totalAmpduNumSymbols,
                           uint16_t staId) const
{
  WifiMode payloadMode = txVector.GetMode (staId);
  uint8_t stbc = txVector.IsStbc () ? 2 : 1; //corresponding to m_STBC in Nsym computation (see IEEE 802.11-2016, equations (19-32) and (21-62))
  uint8_t nes = GetNumberBccEncoders (txVector);
  //TODO: Update station managers to consider GI capabilities
  Time symbolDuration = GetSymbolDuration (txVector);

  double numDataBitsPerSymbol = payloadMode.GetDataRate (txVector, staId) * symbolDuration.GetNanoSeconds () / 1e9;
  uint8_t service = GetNumberServiceBits ();

  double numSymbols = 0;
  switch (mpdutype)
    {
      case FIRST_MPDU_IN_AGGREGATE:
        {
          //First packet in an A-MPDU
          numSymbols = (stbc * (service + size * 8.0 + 6 * nes) / (stbc * numDataBitsPerSymbol));
          if (incFlag == 1)
            {
              totalAmpduSize += size;
              totalAmpduNumSymbols += numSymbols;
            }
          break;
        }
      case MIDDLE_MPDU_IN_AGGREGATE:
        {
          //consecutive packets in an A-MPDU
          numSymbols = (stbc * size * 8.0) / (stbc * numDataBitsPerSymbol);
          if (incFlag == 1)
            {
              totalAmpduSize += size;
              totalAmpduNumSymbols += numSymbols;
            }
          break;
        }
      case LAST_MPDU_IN_AGGREGATE:
        {
          //last packet in an A-MPDU
          uint32_t totalSize = totalAmpduSize + size;
          numSymbols = lrint (stbc * ceil ((service + totalSize * 8.0 + 6 * nes) / (stbc * numDataBitsPerSymbol)));
          NS_ASSERT (totalAmpduNumSymbols <= numSymbols);
          numSymbols -= totalAmpduNumSymbols;
          if (incFlag == 1)
            {
              totalAmpduSize = 0;
              totalAmpduNumSymbols = 0;
            }
          break;
        }
      case NORMAL_MPDU:
      case SINGLE_MPDU:
        {
          //Not an A-MPDU or single MPDU (i.e. the current payload contains both service and padding)
          //The number of OFDM symbols in the data field when BCC encoding
          //is used is given in equation 19-32 of the IEEE 802.11-2016 standard.
          numSymbols = lrint (stbc * ceil ((service + size * 8.0 + 6.0 * nes) / (stbc * numDataBitsPerSymbol)));
          break;
        }
      default:
        NS_FATAL_ERROR ("Unknown MPDU type");
    }

  Time payloadDuration = FemtoSeconds (static_cast<uint64_t> (numSymbols * symbolDuration.GetFemtoSeconds ()));
  if (mpdutype == NORMAL_MPDU || mpdutype == SINGLE_MPDU || mpdutype == LAST_MPDU_IN_AGGREGATE)
    {
      payloadDuration += GetSignalExtension (band);
    }
  return payloadDuration;
}

uint8_t
HtPhy::GetNumberBccEncoders (const WifiTxVector& txVector) const
{
  /**
   * Add an encoder when crossing maxRatePerCoder frontier.
   *
   * The value of 320 Mbps and 350 Mbps for normal GI and short GI (resp.)
   * were obtained by observing the rates for which Nes was incremented in tables
   * 19-27 to 19-41 of IEEE 802.11-2016.
   */
  double maxRatePerCoder = (txVector.GetGuardInterval () == 800) ? 320e6 : 350e6;
  return ceil (txVector.GetMode ().GetDataRate (txVector) / maxRatePerCoder);
}

Time
HtPhy::GetSymbolDuration (const WifiTxVector& txVector) const
{
  uint16_t gi = txVector.GetGuardInterval ();
  NS_ASSERT (gi == 400 || gi == 800);
  return NanoSeconds (3200 + gi);
}

Ptr<WifiPpdu>
HtPhy::BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time ppduDuration)
{
  NS_LOG_FUNCTION (this << psdus << txVector << ppduDuration);
  return Create<HtPpdu> (psdus.begin ()->second, txVector, ppduDuration, m_wifiPhy->GetPhyBand (),
                         ObtainNextUid (txVector));
}

PhyEntity::PhyFieldRxStatus
HtPhy::DoEndReceiveField (WifiPpduField field, Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << field << *event);
  switch (field)
    {
      case WIFI_PPDU_FIELD_HT_SIG:
        return EndReceiveHtSig (event);
      case WIFI_PPDU_FIELD_TRAINING:
        return PhyFieldRxStatus (true); //always consider that training has been correctly received
      case WIFI_PPDU_FIELD_NON_HT_HEADER:
      //no break so as to go to OfdmPhy for processing
      default:
        return OfdmPhy::DoEndReceiveField (field, event);
    }
}

PhyEntity::PhyFieldRxStatus
HtPhy::EndReceiveHtSig (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  NS_ASSERT (event->GetTxVector ().GetPreambleType () == WIFI_PREAMBLE_HT_MF);
  SnrPer snrPer = GetPhyHeaderSnrPer (WIFI_PPDU_FIELD_HT_SIG, event);
  NS_LOG_DEBUG ("HT-SIG: SNR(dB)=" << RatioToDb (snrPer.snr) << ", PER=" << snrPer.per);
  PhyFieldRxStatus status (GetRandomValue () > snrPer.per);
  if (status.isSuccess)
    {
      NS_LOG_DEBUG ("Received HT-SIG");
      if (!IsAllConfigSupported (WIFI_PPDU_FIELD_HT_SIG, event->GetPpdu ()))
        {
          status = PhyFieldRxStatus (false, UNSUPPORTED_SETTINGS, DROP);
        }
    }
  else
    {
      NS_LOG_DEBUG ("Drop packet because HT-SIG reception failed");
      status.reason = HT_SIG_FAILURE;
      status.actionIfFailure = DROP;
    }
  return status;
}

bool
HtPhy::IsAllConfigSupported (WifiPpduField field, Ptr<const WifiPpdu> ppdu) const
{
  if (field == WIFI_PPDU_FIELD_NON_HT_HEADER)
    {
      return true; //wait till reception of HT-SIG (or SIG-A) to make decision
    }
  return OfdmPhy::IsAllConfigSupported (field, ppdu);
}

bool
HtPhy::IsConfigSupported (Ptr<const WifiPpdu> ppdu) const
{
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  if (txVector.GetNss () > m_wifiPhy->GetMaxSupportedRxSpatialStreams ())
    {
      NS_LOG_DEBUG ("Packet reception could not be started because not enough RX antennas");
      return false;
    }
  if (!IsModeSupported (txVector.GetMode ()))
    {
      NS_LOG_DEBUG ("Drop packet because it was sent using an unsupported mode (" << txVector.GetMode () << ")");
      return false;
    }
  return true;
}

Ptr<SpectrumValue>
HtPhy::GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const
{
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  uint16_t centerFrequency = GetCenterFrequencyForChannelWidth (txVector);
  uint16_t channelWidth = txVector.GetChannelWidth ();
  NS_LOG_FUNCTION (this << centerFrequency << channelWidth << txPowerW);
  const auto & txMaskRejectionParams = GetTxMaskRejectionParams ();
  Ptr<SpectrumValue> v = WifiSpectrumValueHelper::CreateHtOfdmTxPowerSpectralDensity (centerFrequency, channelWidth, txPowerW, GetGuardBandwidth (channelWidth),
                                                                                      std::get<0> (txMaskRejectionParams), std::get<1> (txMaskRejectionParams), std::get<2> (txMaskRejectionParams));
  return v;
}

void
HtPhy::InitializeModes (void)
{
  for (uint8_t i = 0; i < 32; ++i)
    {
      GetHtMcs (i);
    }
}

WifiMode
HtPhy::GetHtMcs (uint8_t index)
{
#define CASE(x) \
case x: \
  return GetHtMcs ## x (); \

  switch (index)
    {
      CASE ( 0)
      CASE ( 1)
      CASE ( 2)
      CASE ( 3)
      CASE ( 4)
      CASE ( 5)
      CASE ( 6)
      CASE ( 7)
      CASE ( 8)
      CASE ( 9)
      CASE (10)
      CASE (11)
      CASE (12)
      CASE (13)
      CASE (14)
      CASE (15)
      CASE (16)
      CASE (17)
      CASE (18)
      CASE (19)
      CASE (20)
      CASE (21)
      CASE (22)
      CASE (23)
      CASE (24)
      CASE (25)
      CASE (26)
      CASE (27)
      CASE (28)
      CASE (29)
      CASE (30)
      CASE (31)
      default:
        NS_ABORT_MSG ("Inexistent (or not supported) index (" << +index << ") requested for HT");
        return WifiMode ();
    }
#undef CASE
}

#define GET_HT_MCS(x) \
WifiMode \
HtPhy::GetHtMcs ## x (void) \
{ \
  static WifiMode mcs = CreateHtMcs (x); \
  return mcs; \
}; \

GET_HT_MCS (0)
GET_HT_MCS (1)
GET_HT_MCS (2)
GET_HT_MCS (3)
GET_HT_MCS (4)
GET_HT_MCS (5)
GET_HT_MCS (6)
GET_HT_MCS (7)
GET_HT_MCS (8)
GET_HT_MCS (9)
GET_HT_MCS (10)
GET_HT_MCS (11)
GET_HT_MCS (12)
GET_HT_MCS (13)
GET_HT_MCS (14)
GET_HT_MCS (15)
GET_HT_MCS (16)
GET_HT_MCS (17)
GET_HT_MCS (18)
GET_HT_MCS (19)
GET_HT_MCS (20)
GET_HT_MCS (21)
GET_HT_MCS (22)
GET_HT_MCS (23)
GET_HT_MCS (24)
GET_HT_MCS (25)
GET_HT_MCS (26)
GET_HT_MCS (27)
GET_HT_MCS (28)
GET_HT_MCS (29)
GET_HT_MCS (30)
GET_HT_MCS (31)
#undef GET_HT_MCS

WifiMode
HtPhy::CreateHtMcs (uint8_t index)
{
  NS_ASSERT_MSG (index <= 31, "HtMcs index must be <= 31!");
  return WifiModeFactory::CreateWifiMcs ("HtMcs" + std::to_string (index),
                                         index,
                                         WIFI_MOD_CLASS_HT,
                                         false,
                                         MakeBoundCallback (&GetHtCodeRate, index),
                                         MakeBoundCallback (&GetHtConstellationSize, index),
                                         MakeCallback (&GetPhyRateFromTxVector),
                                         MakeCallback (&GetDataRateFromTxVector),
                                         MakeBoundCallback (&GetNonHtReferenceRate, index),
                                         MakeCallback (&IsAllowed));
}

WifiCodeRate
HtPhy::GetHtCodeRate (uint8_t mcsValue)
{
  return GetCodeRate (mcsValue % 8);
}

WifiCodeRate
HtPhy::GetCodeRate (uint8_t mcsValue)
{
  switch (mcsValue)
    {
      case 0:
      case 1:
      case 3:
        return WIFI_CODE_RATE_1_2;
      case 2:
      case 4:
      case 6:
        return WIFI_CODE_RATE_3_4;
      case 5:
        return WIFI_CODE_RATE_2_3;
      case 7:
        return WIFI_CODE_RATE_5_6;
      default:
        return WIFI_CODE_RATE_UNDEFINED;
    }
}

uint16_t
HtPhy::GetHtConstellationSize (uint8_t mcsValue)
{
  return GetConstellationSize (mcsValue % 8);
}

uint16_t
HtPhy::GetConstellationSize (uint8_t mcsValue)
{
  switch (mcsValue)
    {
      case 0:
        return 2;
      case 1:
      case 2:
        return 4;
      case 3:
      case 4:
        return 16;
      case 5:
      case 6:
      case 7:
        return 64;
      default:
        return 0;
    }
}

uint64_t
HtPhy::GetPhyRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss)
{
  WifiCodeRate codeRate = GetHtCodeRate (mcsValue);
  uint64_t dataRate = GetDataRate (mcsValue, channelWidth, guardInterval, nss);
  return CalculatePhyRate (codeRate, dataRate);
}

uint64_t
HtPhy::CalculatePhyRate (WifiCodeRate codeRate, uint64_t dataRate)
{
  return (dataRate / GetCodeRatio (codeRate));
}

uint64_t
HtPhy::GetPhyRateFromTxVector (const WifiTxVector& txVector, uint16_t /* staId */)
{
  return GetPhyRate (txVector.GetMode ().GetMcsValue (),
                     txVector.GetChannelWidth (),
                     txVector.GetGuardInterval (),
                     txVector.GetNss ());
}

double
HtPhy::GetCodeRatio (WifiCodeRate codeRate)
{
  switch (codeRate)
    {
      case WIFI_CODE_RATE_5_6:
        return (5.0 / 6.0);
      default:
        return OfdmPhy::GetCodeRatio (codeRate);
    }
}

uint64_t
HtPhy::GetDataRateFromTxVector (const WifiTxVector& txVector, uint16_t /* staId */)
{
  return GetDataRate (txVector.GetMode ().GetMcsValue (),
                      txVector.GetChannelWidth (),
                      txVector.GetGuardInterval (),
                      txVector.GetNss ());
}

uint64_t
HtPhy::GetDataRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss)
{
  NS_ASSERT (guardInterval == 800 || guardInterval == 400);
  NS_ASSERT (nss <= 4);
  return CalculateDataRate (3.2, guardInterval,
                            GetUsableSubcarriers (channelWidth),
                            static_cast<uint16_t> (log2 (GetHtConstellationSize (mcsValue))),
                            GetCodeRatio (GetHtCodeRate (mcsValue)), nss);
}

uint64_t
HtPhy::CalculateDataRate (double symbolDuration, uint16_t guardInterval,
                          uint16_t usableSubCarriers, uint16_t numberOfBitsPerSubcarrier,
                          double codingRate, uint8_t nss)
{
  return nss * OfdmPhy::CalculateDataRate (symbolDuration, guardInterval,
                                           usableSubCarriers, numberOfBitsPerSubcarrier,
                                           codingRate);
}

uint16_t
HtPhy::GetUsableSubcarriers (uint16_t channelWidth)
{
  return (channelWidth == 40) ? 108 : 52;
}

uint64_t
HtPhy::GetNonHtReferenceRate (uint8_t mcsValue)
{
  WifiCodeRate codeRate = GetHtCodeRate (mcsValue);
  uint16_t constellationSize = GetHtConstellationSize (mcsValue);
  return CalculateNonHtReferenceRate (codeRate, constellationSize);
}

uint64_t
HtPhy::CalculateNonHtReferenceRate (WifiCodeRate codeRate, uint16_t constellationSize)
{
  uint64_t dataRate;
  switch (constellationSize)
    {
      case 2:
        if (codeRate == WIFI_CODE_RATE_1_2)
          {
            dataRate = 6000000;
          }
        else if (codeRate == WIFI_CODE_RATE_3_4)
          {
            dataRate = 9000000;
          }
        else
          {
            NS_FATAL_ERROR ("Trying to get reference rate for a MCS with wrong combination of coding rate and modulation");
          }
        break;
      case 4:
        if (codeRate == WIFI_CODE_RATE_1_2)
          {
            dataRate = 12000000;
          }
        else if (codeRate == WIFI_CODE_RATE_3_4)
          {
            dataRate = 18000000;
          }
        else
          {
            NS_FATAL_ERROR ("Trying to get reference rate for a MCS with wrong combination of coding rate and modulation");
          }
        break;
      case 16:
        if (codeRate == WIFI_CODE_RATE_1_2)
          {
            dataRate = 24000000;
          }
        else if (codeRate == WIFI_CODE_RATE_3_4)
          {
            dataRate = 36000000;
          }
        else
          {
            NS_FATAL_ERROR ("Trying to get reference rate for a MCS with wrong combination of coding rate and modulation");
          }
        break;
      case 64:
        if (codeRate == WIFI_CODE_RATE_1_2 || codeRate == WIFI_CODE_RATE_2_3)
          {
            dataRate = 48000000;
          }
        else if (codeRate == WIFI_CODE_RATE_3_4 || codeRate == WIFI_CODE_RATE_5_6)
          {
            dataRate = 54000000;
          }
        else
          {
            NS_FATAL_ERROR ("Trying to get reference rate for a MCS with wrong combination of coding rate and modulation");
          }
        break;
      default:
        NS_FATAL_ERROR ("Wrong constellation size");
    }
  return dataRate;
}

bool
HtPhy::IsAllowed (const WifiTxVector& /*txVector*/)
{
  return true;
}

uint32_t
HtPhy::GetMaxPsduSize (void) const
{
  return 65535;
}

} //namespace ns3

namespace {

/**
 * Constructor class for HT modes
 */
static class ConstructorHt
{
public:
  ConstructorHt ()
  {
    ns3::HtPhy::InitializeModes ();
    ns3::WifiPhy::AddStaticPhyEntity (ns3::WIFI_MOD_CLASS_HT, ns3::Create<ns3::HtPhy> ()); //dummy Nss
  }
} g_constructor_ht; ///< the constructor for HT modes

}
