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
 *          Sébastien Deronne <sebastien.deronne@gmail.com> (for logic ported from wifi-phy and spectrum-wifi-phy)
 */

#include "he-phy.h"
#include "he-ppdu.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-phy.h"
#include "he-configuration.h"
#include "ns3/wifi-net-device.h"
#include "ns3/sta-wifi-mac.h"
#include "ns3/ap-wifi-mac.h"
#include "ns3/wifi-utils.h"
#include "ns3/interference-helper.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/assert.h"
#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HePhy");

/*******************************************************
 *       HE PHY (P802.11ax/D4.0, clause 27)
 *******************************************************/

/* *NS_CHECK_STYLE_OFF* */
const PhyEntity::PpduFormats HePhy::m_hePpduFormats { //Ignoring PE (Packet Extension)
  { WIFI_PREAMBLE_HE_SU,    { WIFI_PPDU_FIELD_PREAMBLE,      //L-STF + L-LTF
                              WIFI_PPDU_FIELD_NON_HT_HEADER, //L-SIG + RL-SIG
                              WIFI_PPDU_FIELD_SIG_A,         //HE-SIG-A
                              WIFI_PPDU_FIELD_TRAINING,      //HE-STF + HE-LTFs
                              WIFI_PPDU_FIELD_DATA } },
  { WIFI_PREAMBLE_HE_MU,    { WIFI_PPDU_FIELD_PREAMBLE,      //L-STF + L-LTF
                              WIFI_PPDU_FIELD_NON_HT_HEADER, //L-SIG + RL-SIG
                              WIFI_PPDU_FIELD_SIG_A,         //HE-SIG-A
                              WIFI_PPDU_FIELD_SIG_B,         //HE-SIG-B
                              WIFI_PPDU_FIELD_TRAINING,      //HE-STF + HE-LTFs
                              WIFI_PPDU_FIELD_DATA } },
  { WIFI_PREAMBLE_HE_TB,    { WIFI_PPDU_FIELD_PREAMBLE,      //L-STF + L-LTF
                              WIFI_PPDU_FIELD_NON_HT_HEADER, //L-SIG + RL-SIG
                              WIFI_PPDU_FIELD_SIG_A,         //HE-SIG-A
                              WIFI_PPDU_FIELD_TRAINING,      //HE-STF + HE-LTFs
                              WIFI_PPDU_FIELD_DATA } },
  { WIFI_PREAMBLE_HE_ER_SU, { WIFI_PPDU_FIELD_PREAMBLE,      //L-STF + L-LTF
                              WIFI_PPDU_FIELD_NON_HT_HEADER, //L-SIG + RL-SIG
                              WIFI_PPDU_FIELD_SIG_A,         //HE-SIG-A
                              WIFI_PPDU_FIELD_TRAINING,      //HE-STF + HE-LTFs
                              WIFI_PPDU_FIELD_DATA } }
};
/* *NS_CHECK_STYLE_ON* */

HePhy::HePhy (bool buildModeList /* = true */)
  : VhtPhy (false), //don't add VHT modes to list
    m_trigVectorExpirationTime (Seconds (0))
{
  NS_LOG_FUNCTION (this << buildModeList);
  m_bssMembershipSelector = HE_PHY;
  m_maxMcsIndexPerSs = 11;
  m_maxSupportedMcsIndexPerSs = m_maxMcsIndexPerSs;
  m_currentHeTbPpduUid = UINT64_MAX;
  m_previouslyTxPpduUid = UINT64_MAX;
  if (buildModeList)
    {
      BuildModeList ();
    }
}

HePhy::~HePhy ()
{
  NS_LOG_FUNCTION (this);
}

void
HePhy::BuildModeList (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_modeList.empty ());
  NS_ASSERT (m_bssMembershipSelector == HE_PHY);
  for (uint8_t index = 0; index <= m_maxSupportedMcsIndexPerSs; ++index)
    {
      NS_LOG_LOGIC ("Add HeMcs" << +index << " to list");
      m_modeList.emplace_back (CreateHeMcs (index));
    }
}

WifiMode
HePhy::GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const
{
  switch (field)
    {
      case WIFI_PPDU_FIELD_TRAINING: //consider SIG-A (SIG-B) mode for training for the time being for SU/ER-SU/TB (MU) (useful for InterferenceHelper)
        if (txVector.IsDlMu ())
          {
            NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
            //Training comes after SIG-B
            return GetSigBMode (txVector);
          }
        else
          {
            //Training comes after SIG-A
            return GetSigAMode ();
          }
      default:
        return VhtPhy::GetSigMode (field, txVector);
    }
}

WifiMode
HePhy::GetSigAMode (void) const
{
  return GetVhtMcs0 (); //same number of data tones as VHT for 20 MHz (i.e. 52)
}

WifiMode
HePhy::GetSigBMode (const WifiTxVector& txVector) const
{
  NS_ABORT_MSG_IF (!txVector.IsDlMu () || (txVector.GetModulationClass () != WIFI_MOD_CLASS_HE), "HE-SIG-B only available for HE MU");
  /**
   * Get smallest HE MCS index among station's allocations and use the
   * VHT version of the index. This enables to have 800 ns GI, 52 data
   * tones, and 312.5 kHz spacing while ensuring that MCS will be decoded
   * by all stations.
   */
  uint8_t smallestMcs = 5; //maximum MCS for HE-SIG-B
  for (auto & info : txVector.GetHeMuUserInfoMap ())
    {
      smallestMcs = std::min (smallestMcs, info.second.mcs.GetMcsValue ());
    }
  switch (smallestMcs) //GetVhtMcs (mcs) is not static
    {
      case 0:
        return GetVhtMcs0 ();
      case 1:
        return GetVhtMcs1 ();
      case 2:
        return GetVhtMcs2 ();
      case 3:
        return GetVhtMcs3 ();
      case 4:
        return GetVhtMcs4 ();
      case 5:
      default:
        return GetVhtMcs5 ();
    }
}

const PhyEntity::PpduFormats &
HePhy::GetPpduFormats (void) const
{
  return m_hePpduFormats;
}

Time
HePhy::GetLSigDuration (WifiPreamble /* preamble */) const
{
  return MicroSeconds (8); //L-SIG + RL-SIG
}

Time
HePhy::GetTrainingDuration (const WifiTxVector& txVector,
                            uint8_t nDataLtf, uint8_t nExtensionLtf /* = 0 */) const
{
  Time ltfDuration = MicroSeconds (8); //TODO extract from TxVector when available
  Time stfDuration;
  if (txVector.IsUlMu ())
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      stfDuration = MicroSeconds (8);
    }
  else
    {
      stfDuration = MicroSeconds (4);
    }
  NS_ABORT_MSG_IF (nDataLtf > 8, "Unsupported number of LTFs " << +nDataLtf << " for HE");
  NS_ABORT_MSG_IF (nExtensionLtf > 0, "No extension LTFs expected for HE");
  return stfDuration + ltfDuration * nDataLtf; //HE-STF + HE-LTFs
}

Time
HePhy::GetSigADuration (WifiPreamble preamble) const
{
  return (preamble == WIFI_PREAMBLE_HE_ER_SU) ? MicroSeconds (16) : MicroSeconds (8); //HE-SIG-A (first and second symbol)
}

Time
HePhy::GetSigBDuration (const WifiTxVector& txVector) const
{
  if (txVector.IsDlMu ()) //See section 27.3.10.8 of IEEE 802.11ax draft 4.0.
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      /*
       * Compute the number of bits used by common field.
       * Assume that compression bit in HE-SIG-A is not set (i.e. not
       * full band MU-MIMO); the field is present.
       */
      uint16_t bw = txVector.GetChannelWidth ();
      std::size_t commonFieldSize = 4 /* CRC */ + 6 /* tail */;
      if (bw <= 40)
        {
          commonFieldSize += 8; //only one allocation subfield
        }
      else
        {
          commonFieldSize += 8 * (bw / 40) /* one allocation field per 40 MHz */ + 1 /* center RU */;
        }

      /*
       * Compute the number of bits used by user-specific field.
       * MU-MIMO is not supported; only one station per RU.
       * The user-specific field is composed of N user block fields
       * spread over each corresponding HE-SIG-B content channel.
       * Each user block field contains either two or one users' data
       * (the latter being for odd number of stations per content channel).
       * Padding will be handled further down in the code.
       */
      std::pair<std::size_t, std::size_t> numStaPerContentChannel = txVector.GetNumRusPerHeSigBContentChannel ();
      std::size_t maxNumStaPerContentChannel = std::max (numStaPerContentChannel.first, numStaPerContentChannel.second);
      std::size_t maxNumUserBlockFields = maxNumStaPerContentChannel / 2; //handle last user block with single user, if any, further down
      std::size_t userSpecificFieldSize = maxNumUserBlockFields * (2 * 21 /* user fields (2 users) */ + 4 /* tail */ + 6 /* CRC */);
      if (maxNumStaPerContentChannel % 2 != 0)
        {
          userSpecificFieldSize += 21 /* last user field */ + 4 /* CRC */ + 6 /* tail */;
        }

      /*
       * Compute duration of HE-SIG-B considering that padding
       * is added up to the next OFDM symbol.
       * Nss = 1 and GI = 800 ns for HE-SIG-B.
       */
      Time symbolDuration = MicroSeconds (4);
      double numDataBitsPerSymbol = GetSigBMode (txVector).GetDataRate (20, 800, 1) * symbolDuration.GetNanoSeconds () / 1e9;
      double numSymbols = ceil ((commonFieldSize + userSpecificFieldSize) / numDataBitsPerSymbol);

      return FemtoSeconds (static_cast<uint64_t> (numSymbols * symbolDuration.GetFemtoSeconds ()));
    }
  else
    {
      // no SIG-B
      return MicroSeconds (0);
    }
}

Time
HePhy::GetValidPpduDuration (Time ppduDuration, const WifiTxVector& txVector,
                             WifiPhyBand band)
{
  Time tSymbol = NanoSeconds (12800 + txVector.GetGuardInterval ());
  Time preambleDuration = WifiPhy::GetStaticPhyEntity (WIFI_MOD_CLASS_HE)->CalculatePhyPreambleAndHeaderDuration (txVector); //this is quite convoluted but only way of keeping the method static
  uint8_t sigExtension = (band == WIFI_PHY_BAND_2_4GHZ ? 6 : 0);
  uint32_t nSymbols = floor (static_cast<double> ((ppduDuration - preambleDuration).GetNanoSeconds () - (sigExtension * 1000)) / tSymbol.GetNanoSeconds ());
  return preambleDuration + (nSymbols * tSymbol) + MicroSeconds (sigExtension);
}

std::pair<uint16_t, Time>
HePhy::ConvertHeTbPpduDurationToLSigLength (Time ppduDuration, const WifiTxVector& txVector,
                                            WifiPhyBand band)
{
  NS_ABORT_IF (!txVector.IsUlMu () || (txVector.GetModulationClass () != WIFI_MOD_CLASS_HE));
  // update ppduDuration so that it is a valid PPDU duration
  ppduDuration = GetValidPpduDuration (ppduDuration, txVector, band);
  uint8_t sigExtension = (band == WIFI_PHY_BAND_2_4GHZ ? 6 : 0);
  uint8_t m = 2; //HE TB PPDU so m is set to 2
  uint16_t length = ((ceil ((static_cast<double> (ppduDuration.GetNanoSeconds () - (20 * 1000) - (sigExtension * 1000)) / 1000) / 4.0) * 3) - 3 - m);
  return {length, ppduDuration};
}

Time
HePhy::ConvertLSigLengthToHeTbPpduDuration (uint16_t length, const WifiTxVector& txVector, WifiPhyBand band)
{
  NS_ABORT_IF (!txVector.IsUlMu () || (txVector.GetModulationClass () != WIFI_MOD_CLASS_HE));
  uint8_t sigExtension = (band == WIFI_PHY_BAND_2_4GHZ ? 6 : 0);
  uint8_t m = 2; //HE TB PPDU so m is set to 2
  //Equation 27-11 of IEEE P802.11ax/D4.0
  Time calculatedDuration = MicroSeconds (((ceil (static_cast<double> (length + 3 + m) / 3)) * 4) + 20 + sigExtension);
  return GetValidPpduDuration (calculatedDuration, txVector, band);
}

Time
HePhy::CalculateNonOfdmaDurationForHeTb (const WifiTxVector& txVector) const
{
  NS_ABORT_IF (!txVector.IsUlMu () || (txVector.GetModulationClass () != WIFI_MOD_CLASS_HE));
  Time duration = GetDuration (WIFI_PPDU_FIELD_PREAMBLE, txVector)
    + GetDuration (WIFI_PPDU_FIELD_NON_HT_HEADER, txVector)
    + GetDuration (WIFI_PPDU_FIELD_SIG_A, txVector);
  return duration;
}

uint8_t
HePhy::GetNumberBccEncoders (const WifiTxVector& /* txVector */) const
{
  return 1; //only 1 BCC encoder for HE since higher rates are obtained using LDPC
}

Time
HePhy::GetSymbolDuration (const WifiTxVector& txVector) const
{
  uint16_t gi = txVector.GetGuardInterval ();
  NS_ASSERT (gi == 800 || gi == 1600 || gi == 3200);
  return NanoSeconds (12800 + gi);
}

void
HePhy::SetTrigVector (const WifiTxVector& trigVector, Time validity)
{
  m_trigVector = trigVector;
  m_trigVectorExpirationTime = Simulator::Now () + validity;
  NS_LOG_FUNCTION (this << m_trigVector << m_trigVectorExpirationTime.As (Time::US));
}

Ptr<WifiPpdu>
HePhy::BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time ppduDuration)
{
  NS_LOG_FUNCTION (this << psdus << txVector << ppduDuration);
  HePpdu::TxPsdFlag flag;
  if (txVector.IsUlMu ())
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      flag = HePpdu::PSD_HE_TB_NON_OFDMA_PORTION;
    }
  else
    {
      flag = HePpdu::PSD_NON_HE_TB;
    }
  return Create<HePpdu> (psdus, txVector, ppduDuration, m_wifiPhy->GetPhyBand (),
                         ObtainNextUid (txVector), flag,
                         m_wifiPhy->GetOperatingChannel ().GetPrimaryChannelIndex (20));
}

void
HePhy::StartReceivePreamble (Ptr<WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW,
                             Time rxDuration)
{
  NS_LOG_FUNCTION (this << ppdu << rxDuration);
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  auto hePpdu = DynamicCast<HePpdu> (ppdu);
  NS_ASSERT (hePpdu);
  HePpdu::TxPsdFlag psdFlag = hePpdu->GetTxPsdFlag ();
  if (txVector.IsUlMu () && psdFlag == HePpdu::PSD_HE_TB_OFDMA_PORTION)
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      if (m_currentHeTbPpduUid == ppdu->GetUid ()
          && GetCurrentEvent () != 0)
        {
          //AP or STA has already received non-OFDMA part, switch to OFDMA part, and schedule reception of payload (will be canceled for STAs by StartPayload)
          bool ofdmaStarted = !m_beginOfdmaPayloadRxEvents.empty ();
          NS_LOG_INFO ("Switch to OFDMA part (already started? " << (ofdmaStarted ? "Y" : "N") << ") " <<
                       "and schedule OFDMA payload reception in " << GetDuration (WIFI_PPDU_FIELD_TRAINING, txVector).As (Time::NS));
          Ptr<Event> event = CreateInterferenceEvent (ppdu, txVector, rxDuration, rxPowersW, !ofdmaStarted);
          uint16_t staId = GetStaId (ppdu);
          NS_ASSERT (m_beginOfdmaPayloadRxEvents.find (staId) == m_beginOfdmaPayloadRxEvents.end ());
          m_beginOfdmaPayloadRxEvents[staId] = Simulator::Schedule (GetDuration (WIFI_PPDU_FIELD_TRAINING, txVector),
                                                                    &HePhy::StartReceiveOfdmaPayload, this, event);
        }
      else
        {
          //PHY receives the OFDMA payload while having dropped the preamble
          NS_LOG_INFO ("Consider OFDMA part of the HE TB PPDU as interference since device dropped the preamble");
          CreateInterferenceEvent (ppdu, txVector, rxDuration, rxPowersW);
          //the OFDMA part of the HE TB PPDUs will be noise _after_ the completion of the current event
          ErasePreambleEvent (ppdu, rxDuration);
        }
    }
  else
    {
      PhyEntity::StartReceivePreamble (ppdu, rxPowersW, ppdu->GetTxDuration ()); // The actual duration of the PPDU should be used
    }
}

void
HePhy::CancelAllEvents (void)
{
  NS_LOG_FUNCTION (this);
  for (auto & beginOfdmaPayloadRxEvent : m_beginOfdmaPayloadRxEvents)
    {
      beginOfdmaPayloadRxEvent.second.Cancel ();
    }
  m_beginOfdmaPayloadRxEvents.clear ();
  PhyEntity::CancelAllEvents ();
}

void
HePhy::DoAbortCurrentReception (WifiPhyRxfailureReason reason)
{
  NS_LOG_FUNCTION (this << reason);
  if (reason != OBSS_PD_CCA_RESET)
    {
      for (auto & endMpduEvent : m_endOfMpduEvents)
        {
          endMpduEvent.Cancel ();
        }
      m_endOfMpduEvents.clear ();
    }
  else
    {
      PhyEntity::DoAbortCurrentReception (reason);
    }
}

void
HePhy::DoResetReceive (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  if (event->GetPpdu ()->GetType () != WIFI_PPDU_TYPE_UL_MU)
    {
      NS_ASSERT (event->GetEndTime () == Simulator::Now ());
    }
  for (auto & beginOfdmaPayloadRxEvent : m_beginOfdmaPayloadRxEvents)
    {
      beginOfdmaPayloadRxEvent.second.Cancel ();
    }
  m_beginOfdmaPayloadRxEvents.clear ();
}

Ptr<Event>
HePhy::DoGetEvent (Ptr<const WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW)
{
  Ptr<Event> event;
  //We store all incoming preamble events, and a decision is made at the end of the preamble detection window.
  //If a preamble is received after the preamble detection window, it is stored anyway because this is needed for HE TB PPDUs in
  //order to properly update the received power in InterferenceHelper. The map is cleaned anyway at the end of the current reception.
  if (ppdu->GetType () == WIFI_PPDU_TYPE_UL_MU)
    {
      auto uidPreamblePair = std::make_pair (ppdu->GetUid (), ppdu->GetPreamble ());
      const WifiTxVector& txVector = ppdu->GetTxVector ();
      Time rxDuration = CalculateNonOfdmaDurationForHeTb (txVector); //the OFDMA part of the transmission will be added later on
      const auto & currentPreambleEvents = GetCurrentPreambleEvents ();
      auto it = currentPreambleEvents.find (uidPreamblePair);
      if (it != currentPreambleEvents.end ())
        {
          NS_LOG_DEBUG ("Received another HE TB PPDU for UID " << ppdu->GetUid () << " from STA-ID " << ppdu->GetStaId () << " and BSS color " << +txVector.GetBssColor ());
          event = it->second;

          //Update received power of the event associated to that UL MU transmission
          UpdateInterferenceEvent (event, rxPowersW);

          if ((GetCurrentEvent () != 0) && (GetCurrentEvent ()->GetPpdu ()->GetUid () != ppdu->GetUid ()))
            {
              NS_LOG_DEBUG ("Drop packet because already receiving another HE TB PPDU");
              m_wifiPhy->NotifyRxDrop (GetAddressedPsduInPpdu (ppdu), RXING);
            }
          return nullptr;
        }
      else
        {
          NS_LOG_DEBUG ("Received a new HE TB PPDU for UID " << ppdu->GetUid () << " from STA-ID " << ppdu->GetStaId () << " and BSS color " << +txVector.GetBssColor ());
          event = CreateInterferenceEvent (ppdu, txVector, rxDuration, rxPowersW);
          AddPreambleEvent (event);
        }
    }
  else
    {
      event = PhyEntity::DoGetEvent (ppdu, rxPowersW);
    }
  return event;
}

Ptr<const WifiPsdu>
HePhy::GetAddressedPsduInPpdu (Ptr<const WifiPpdu> ppdu) const
{
  if (ppdu->GetType () == WIFI_PPDU_TYPE_DL_MU || ppdu->GetType () == WIFI_PPDU_TYPE_UL_MU)
    {
      auto hePpdu = DynamicCast<const HePpdu> (ppdu);
      NS_ASSERT (hePpdu);
      return hePpdu->GetPsdu (GetBssColor (), GetStaId (ppdu));
    }
  return PhyEntity::GetAddressedPsduInPpdu (ppdu);
}

uint8_t
HePhy::GetBssColor (void) const
{
  uint8_t bssColor = 0;
  if (m_wifiPhy->GetDevice () != nullptr)
    {
      Ptr<HeConfiguration> heConfiguration = m_wifiPhy->GetDevice ()->GetHeConfiguration ();
      if (heConfiguration)
        {
          bssColor = heConfiguration->GetBssColor ();
        }
    }
  return bssColor;
}

uint16_t
HePhy::GetStaId (const Ptr<const WifiPpdu> ppdu) const
{
  if (ppdu->GetType () == WIFI_PPDU_TYPE_UL_MU)
    {
      return ppdu->GetStaId ();
    }
  else if (ppdu->GetType () == WIFI_PPDU_TYPE_DL_MU)
    {
      Ptr<StaWifiMac> mac = DynamicCast<StaWifiMac> (m_wifiPhy->GetDevice ()->GetMac ());
      if (mac && mac->IsAssociated ())
        {
          return mac->GetAssociationId ();
        }
    }
  return PhyEntity::GetStaId (ppdu);
}

PhyEntity::PhyFieldRxStatus
HePhy::ProcessSigA (Ptr<Event> event, PhyFieldRxStatus status)
{
  NS_LOG_FUNCTION (this << *event << status);
  //Notify end of SIG-A (in all cases)
  WifiTxVector txVector = event->GetTxVector ();
  HeSigAParameters params;
  params.rssiW = GetRxPowerWForPpdu (event);
  params.bssColor = txVector.GetBssColor ();
  NotifyEndOfHeSigA (params); //if OBSS_PD CCA_RESET, set power restriction first and wait till field is processed before switching to IDLE

  if (status.isSuccess)
    {
      //Check if PPDU is filtered based on the BSS color
      uint8_t myBssColor = GetBssColor ();
      uint8_t rxBssColor = txVector.GetBssColor ();
      if (myBssColor != 0 && rxBssColor != 0 && myBssColor != rxBssColor)
        {
          NS_LOG_DEBUG ("The BSS color of this PPDU (" << +rxBssColor << ") does not match the device's (" << +myBssColor << "). The PPDU is filtered.");
          return PhyFieldRxStatus (false, FILTERED, DROP);
        }

      // When SIG-A is decoded, we know the type of frame being received. If we stored a
      // valid TRIGVECTOR and we are not receiving a TB PPDU, we drop the frame.
      if (m_trigVectorExpirationTime >= Simulator::Now () && !txVector.IsUlMu ())
        {
          NS_LOG_DEBUG ("Expected an HE TB PPDU, receiving a " << txVector.GetPreambleType ());
          return PhyFieldRxStatus (false, FILTERED, DROP);
        }

      Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
      if (txVector.IsUlMu ())
        {
          NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
          // check that the stored TRIGVECTOR is still valid
          if (m_trigVectorExpirationTime < Simulator::Now ())
            {
              NS_LOG_DEBUG ("No valid TRIGVECTOR, the PHY was not expecting a TB PPDU");
              return PhyFieldRxStatus (false, FILTERED, DROP);
            }
          // We expected a TB PPDU and we are receiving a TB PPDU. However, despite
          // the previous check on BSS Color, we may be receiving a TB PPDU from an
          // OBSS, as BSS Colors are not guaranteed to be different for all APs in
          // range (an example is when BSS Color is 0). We can detect this situation
          // by comparing the TRIGVECTOR with the TXVECTOR of the TB PPDU being received
          if (m_trigVector.GetChannelWidth () != txVector.GetChannelWidth ())
            {
              NS_LOG_DEBUG ("Received channel width different than in TRIGVECTOR");
              return PhyFieldRxStatus (false, FILTERED, DROP);
            }
          if (m_trigVector.GetLength () != txVector.GetLength ())
            {
              NS_LOG_DEBUG ("Received UL Length (" << txVector.GetLength () <<
                            ") different than in TRIGVECTOR (" << m_trigVector.GetLength ()
                            << ")");
              return PhyFieldRxStatus (false, FILTERED, DROP);
            }
          uint16_t staId = ppdu->GetStaId ();
          if (m_trigVector.GetHeMuUserInfoMap ().find (staId) == m_trigVector.GetHeMuUserInfoMap ().end ()
              || m_trigVector.GetHeMuUserInfo (staId) != txVector.GetHeMuUserInfo (staId))
            {
              NS_LOG_DEBUG ("User Info map of TB PPDU being received differs from that of TRIGVECTOR");
              return PhyFieldRxStatus (false, FILTERED, DROP);
            }

          m_currentHeTbPpduUid = ppdu->GetUid (); //to be able to correctly schedule start of OFDMA payload
        }

      if (ppdu->GetType () != WIFI_PPDU_TYPE_DL_MU && !GetAddressedPsduInPpdu (ppdu)) //Final decision on STA-ID correspondence of DL MU is delayed to end of SIG-B
        {
          NS_ASSERT (ppdu->GetType () == WIFI_PPDU_TYPE_UL_MU);
          NS_LOG_DEBUG ("No PSDU addressed to that PHY in the received MU PPDU. The PPDU is filtered.");
          return PhyFieldRxStatus (false, FILTERED, DROP);
        }
    }
  return status;
}

void
HePhy::SetEndOfHeSigACallback (EndOfHeSigACallback callback)
{
  m_endOfHeSigACallback = callback;
}

void
HePhy::NotifyEndOfHeSigA (HeSigAParameters params)
{
  if (!m_endOfHeSigACallback.IsNull ())
    {
      m_endOfHeSigACallback (params);
    }
}

PhyEntity::PhyFieldRxStatus
HePhy::ProcessSigB (Ptr<Event> event, PhyFieldRxStatus status)
{
  NS_LOG_FUNCTION (this << *event << status);
  if (status.isSuccess)
    {
      //Check if PPDU is filtered only if the SIG-B content is supported (not explicitly stated but assumed based on behavior for SIG-A)
      if (!GetAddressedPsduInPpdu (event->GetPpdu ()))
        {
          NS_LOG_DEBUG ("No PSDU addressed to that PHY in the received MU PPDU. The PPDU is filtered.");
          return PhyFieldRxStatus (false, FILTERED, DROP);
        }
    }
  return status;
}

bool
HePhy::IsConfigSupported (Ptr<const WifiPpdu> ppdu) const
{
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  uint16_t staId = GetStaId (ppdu);
  WifiMode txMode = txVector.GetMode (staId);
  uint8_t nss = txVector.GetNssMax ();
  if (txVector.IsDlMu ())
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      for (auto info : txVector.GetHeMuUserInfoMap ())
        {
          if (info.first == staId)
            {
              nss = info.second.nss; //no need to look at other PSDUs
              break;
            }
        }
    }

  if (nss > m_wifiPhy->GetMaxSupportedRxSpatialStreams ())
    {
      NS_LOG_DEBUG ("Packet reception could not be started because not enough RX antennas");
      return false;
    }
  if (!IsModeSupported (txMode))
    {
      NS_LOG_DEBUG ("Drop packet because it was sent using an unsupported mode (" << txVector.GetMode () << ")");
      return false;
    }
  return true;
}

void
HePhy::DoStartReceivePayload (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  const WifiTxVector& txVector = event->GetTxVector ();
  Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
  if (txVector.IsUlMu ())
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      bool isAp = (DynamicCast<ApWifiMac> (m_wifiPhy->GetDevice ()->GetMac ()) != 0);
      if (!isAp)
        {
          NS_LOG_DEBUG ("Ignore HE TB PPDU payload received by STA but keep state in Rx");
          m_endRxPayloadEvents.push_back (Simulator::Schedule (ppdu->GetTxDuration () - CalculatePhyPreambleAndHeaderDuration (txVector),
                                                               &PhyEntity::ResetReceive, this, event));
          //Cancel all scheduled events for OFDMA payload reception
          NS_ASSERT (!m_beginOfdmaPayloadRxEvents.empty () && m_beginOfdmaPayloadRxEvents.begin ()->second.IsRunning ());
          for (auto & beginOfdmaPayloadRxEvent : m_beginOfdmaPayloadRxEvents)
            {
              beginOfdmaPayloadRxEvent.second.Cancel ();
            }
          m_beginOfdmaPayloadRxEvents.clear ();
        }
      else
        {
          NS_LOG_DEBUG ("Receiving PSDU in HE TB PPDU");
          uint16_t staId = GetStaId (ppdu);
          m_signalNoiseMap.insert ({std::make_pair (ppdu->GetUid (), staId), SignalNoiseDbm ()});
          m_statusPerMpduMap.insert ({std::make_pair (ppdu->GetUid (), staId), std::vector<bool> ()});
          //for HE TB PPDUs, ScheduleEndOfMpdus and EndReceive are scheduled by StartReceiveOfdmaPayload
          NS_ASSERT (isAp);
          NS_ASSERT (!m_beginOfdmaPayloadRxEvents.empty ());
          for (auto & beginOfdmaPayloadRxEvent : m_beginOfdmaPayloadRxEvents)
            {
              NS_ASSERT (beginOfdmaPayloadRxEvent.second.IsRunning ());
            }
        }
    }
  else
    {
      PhyEntity::DoStartReceivePayload (event);
    }
}

void
HePhy::DoEndReceivePayload (Ptr<const WifiPpdu> ppdu)
{
  NS_LOG_FUNCTION (this << ppdu);
  if (ppdu->GetType () == WIFI_PPDU_TYPE_UL_MU)
    {
      for (auto it = m_endRxPayloadEvents.begin (); it != m_endRxPayloadEvents.end (); )
        {
          if (it->IsExpired ())
            {
              it = m_endRxPayloadEvents.erase (it);
            }
          else
            {
              it++;
            }
        }
      if (m_endRxPayloadEvents.empty ())
        {
          //We've got the last PPDU of the UL-OFDMA transmission
          NotifyInterferenceRxEndAndClear (true); //reset WifiPhy
        }
    }
  else
    {
      NS_ASSERT (m_wifiPhy->GetLastRxEndTime () == Simulator::Now ());
      PhyEntity::DoEndReceivePayload (ppdu);
    }
}

void
HePhy::StartReceiveOfdmaPayload (Ptr<Event> event)
{
  Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
  const RxPowerWattPerChannelBand& rxPowersW = event->GetRxPowerWPerBand ();
  //The total RX power corresponds to the maximum over all the bands.
  //Only perform this computation if the result needs to be logged.
  auto it = rxPowersW.end ();
  if (g_log.IsEnabled (ns3::LOG_FUNCTION))
    {
      it = std::max_element (rxPowersW.begin (), rxPowersW.end (),
                             [] (const std::pair<WifiSpectrumBand, double> &p1, const std::pair<WifiSpectrumBand, double> &p2) {
                               return p1.second < p2.second;
                             });

    }
  NS_LOG_FUNCTION (this << *event << it->second);
  NS_ASSERT (GetCurrentEvent () != 0);
  auto itEvent = m_beginOfdmaPayloadRxEvents.find (GetStaId (ppdu));
  /**
   * m_beginOfdmaPayloadRxEvents should still be running only for APs, since canceled in StartReceivePayload for STAs.
   * This is because SpectrumWifiPhy does not have access to the device type and thus blindly schedules things, letting
   * the parent WifiPhy class take into account device type.
   */
  NS_ASSERT (itEvent != m_beginOfdmaPayloadRxEvents.end () && itEvent->second.IsExpired ());
  m_beginOfdmaPayloadRxEvents.erase (itEvent);

  Time payloadDuration = ppdu->GetTxDuration () - CalculatePhyPreambleAndHeaderDuration (ppdu->GetTxVector ());
  Ptr<const WifiPsdu> psdu = GetAddressedPsduInPpdu (ppdu);
  ScheduleEndOfMpdus (event);
  m_endRxPayloadEvents.push_back (Simulator::Schedule (payloadDuration, &PhyEntity::EndReceivePayload, this, event));
  m_signalNoiseMap.insert ({std::make_pair (ppdu->GetUid (), ppdu->GetStaId ()), SignalNoiseDbm ()});
  m_statusPerMpduMap.insert ({std::make_pair (ppdu->GetUid (), ppdu->GetStaId ()), std::vector<bool> ()});
  // Notify the MAC about the start of a new HE TB PPDU, so that it can reschedule the timeout
  NotifyPayloadBegin (ppdu->GetTxVector (), payloadDuration);
}

std::pair<uint16_t, WifiSpectrumBand>
HePhy::GetChannelWidthAndBand (const WifiTxVector& txVector, uint16_t staId) const
{
  if (txVector.IsMu ())
    {
      return std::make_pair (HeRu::GetBandwidth (txVector.GetRu (staId).GetRuType ()),
                             GetRuBandForRx (txVector, staId));
    }
  else
    {
      return PhyEntity::GetChannelWidthAndBand (txVector, staId);
    }
}

WifiSpectrumBand
HePhy::GetRuBandForTx (const WifiTxVector& txVector, uint16_t staId) const
{
  NS_ASSERT (txVector.IsMu ());
  WifiSpectrumBand band;
  HeRu::RuSpec ru = txVector.GetRu (staId);
  uint16_t channelWidth = txVector.GetChannelWidth ();
  NS_ASSERT (channelWidth <= m_wifiPhy->GetChannelWidth ());
  HeRu::SubcarrierGroup group = HeRu::GetSubcarrierGroup (channelWidth, ru.GetRuType (), ru.GetPhyIndex ());
  HeRu::SubcarrierRange range = std::make_pair (group.front ().first, group.back ().second);
  // for a TX spectrum, the guard bandwidth is a function of the transmission channel width
  // and the spectrum width equals the transmission channel width (hence bandIndex equals 0)
  band = m_wifiPhy->ConvertHeRuSubcarriers (channelWidth, GetGuardBandwidth (channelWidth),
                                            range, 0);
  return band;
}

WifiSpectrumBand
HePhy::GetRuBandForRx (const WifiTxVector& txVector, uint16_t staId) const
{
  NS_ASSERT (txVector.IsMu ());
  WifiSpectrumBand band;
  HeRu::RuSpec ru = txVector.GetRu (staId);
  uint16_t channelWidth = txVector.GetChannelWidth ();
  NS_ASSERT (channelWidth <= m_wifiPhy->GetChannelWidth ());
  HeRu::SubcarrierGroup group = HeRu::GetSubcarrierGroup (channelWidth, ru.GetRuType (), ru.GetPhyIndex ());
  HeRu::SubcarrierRange range = std::make_pair (group.front ().first, group.back ().second);
  // for an RX spectrum, the guard bandwidth is a function of the operating channel width
  // and the spectrum width equals the operating channel width
  band = m_wifiPhy->ConvertHeRuSubcarriers (channelWidth, GetGuardBandwidth (m_wifiPhy->GetChannelWidth ()),
                                            range, m_wifiPhy->GetOperatingChannel ().GetPrimaryChannelIndex (channelWidth));
  return band;
}

WifiSpectrumBand
HePhy::GetNonOfdmaBand (const WifiTxVector& txVector, uint16_t staId) const
{
  NS_ASSERT (txVector.IsUlMu () && (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE));
  uint16_t channelWidth = txVector.GetChannelWidth ();
  NS_ASSERT (channelWidth <= m_wifiPhy->GetChannelWidth ());

  HeRu::RuSpec ru = txVector.GetRu (staId);
  uint16_t nonOfdmaWidth = GetNonOfdmaWidth (ru);

  // Find the RU that encompasses the non-OFDMA part of the HE TB PPDU for the STA-ID
  HeRu::RuSpec nonOfdmaRu = HeRu::FindOverlappingRu (channelWidth, ru, HeRu::GetRuType (nonOfdmaWidth));
  nonOfdmaRu.SetPhyIndex (channelWidth, m_wifiPhy->GetOperatingChannel ().GetPrimaryChannelIndex (20));

  HeRu::SubcarrierGroup groupPreamble = HeRu::GetSubcarrierGroup (channelWidth, nonOfdmaRu.GetRuType (), nonOfdmaRu.GetPhyIndex ());
  HeRu::SubcarrierRange range = std::make_pair (groupPreamble.front ().first, groupPreamble.back ().second);
  return m_wifiPhy->ConvertHeRuSubcarriers (channelWidth, GetGuardBandwidth (m_wifiPhy->GetChannelWidth ()), range,
                                            m_wifiPhy->GetOperatingChannel ().GetPrimaryChannelIndex (channelWidth));
}

uint16_t
HePhy::GetNonOfdmaWidth (HeRu::RuSpec ru) const
{
  if (ru.GetRuType () == HeRu::RU_26_TONE && ru.GetIndex () == 19)
    {
      // the center 26-tone RU in an 80 MHz channel is not fully covered by
      // any 20 MHz channel, but only by an 80 MHz channel
      return 80;
    }
  return std::max<uint16_t> (HeRu::GetBandwidth (ru.GetRuType ()), 20);
}

uint64_t
HePhy::GetCurrentHeTbPpduUid (void) const
{
  return m_currentHeTbPpduUid;
}

uint16_t
HePhy::GetMeasurementChannelWidth (const Ptr<const WifiPpdu> ppdu) const
{
  uint16_t channelWidth = PhyEntity::GetMeasurementChannelWidth (ppdu);
  /**
   * The PHY shall not issue a PHY-RXSTART.indication primitive in response to a PPDU that does not overlap
   * the primary channel unless the PHY at an AP receives the HE TB PPDU solicited by the AP. For the HE
   * TB PPDU solicited by the AP, the PHY shall issue a PHY-RXSTART.indication primitive for a PPDU
   * received in the primary or at the secondary 20 MHz channel, the secondary 40 MHz channel, or the secondary
   * 80 MHz channel.
   */
  if (channelWidth >= 40 && ppdu->GetUid () != m_previouslyTxPpduUid)
    {
      channelWidth = 20;
    }
  return channelWidth;
}

uint64_t
HePhy::ObtainNextUid (const WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << txVector);
  uint64_t uid;
  if (txVector.IsUlMu ())
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      //Use UID of PPDU containing trigger frame to identify resulting HE TB PPDUs, since the latter should immediately follow the former
      uid = m_wifiPhy->GetPreviouslyRxPpduUid ();
      NS_ASSERT (uid != UINT64_MAX);
    }
  else
    {
      uid = m_globalPpduUid++;
    }
  m_previouslyTxPpduUid = uid; //to be able to identify solicited HE TB PPDUs
  return uid;
}

Ptr<SpectrumValue>
HePhy::GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const
{
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  uint16_t centerFrequency = GetCenterFrequencyForChannelWidth (txVector);
  uint16_t channelWidth = txVector.GetChannelWidth ();
  NS_LOG_FUNCTION (this << centerFrequency << channelWidth << txPowerW);
  auto hePpdu = DynamicCast<const HePpdu> (ppdu);
  NS_ASSERT (hePpdu);
  HePpdu::TxPsdFlag flag = hePpdu->GetTxPsdFlag ();
  Ptr<SpectrumValue> v;
  if (flag == HePpdu::PSD_HE_TB_OFDMA_PORTION)
    {
      WifiSpectrumBand band = GetRuBandForTx (txVector, GetStaId (hePpdu));
      v = WifiSpectrumValueHelper::CreateHeMuOfdmTxPowerSpectralDensity (centerFrequency, channelWidth, txPowerW, GetGuardBandwidth (channelWidth), band);
    }
  else
    {
      if (flag == HePpdu::PSD_HE_TB_NON_OFDMA_PORTION)
        {
          //non-OFDMA portion is sent only on the 20 MHz channels covering the RU
          uint16_t staId = GetStaId (hePpdu);
          centerFrequency = GetCenterFrequencyForNonOfdmaPart (txVector, staId);
          uint16_t ruWidth = HeRu::GetBandwidth (txVector.GetRu (staId).GetRuType ());
          channelWidth = ruWidth < 20 ? 20 : ruWidth;
        }
      const auto & txMaskRejectionParams = GetTxMaskRejectionParams ();
      v = WifiSpectrumValueHelper::CreateHeOfdmTxPowerSpectralDensity (centerFrequency, channelWidth, txPowerW, GetGuardBandwidth (channelWidth),
                                                                       std::get<0> (txMaskRejectionParams), std::get<1> (txMaskRejectionParams), std::get<2> (txMaskRejectionParams));
    }
  return v;
}

uint16_t
HePhy::GetCenterFrequencyForNonOfdmaPart (const WifiTxVector& txVector, uint16_t staId) const
{
  NS_LOG_FUNCTION (this << txVector << staId);
  NS_ASSERT (txVector.IsUlMu () && (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE));
  uint16_t centerFrequency = GetCenterFrequencyForChannelWidth (txVector);
  uint16_t currentWidth = txVector.GetChannelWidth ();

  HeRu::RuSpec ru = txVector.GetRu (staId);
  uint16_t nonOfdmaWidth = GetNonOfdmaWidth (ru);
  if (nonOfdmaWidth != currentWidth)
    {
      //Obtain the index of the non-OFDMA portion
      HeRu::RuSpec nonOfdmaRu = HeRu::FindOverlappingRu (currentWidth, ru, HeRu::GetRuType (nonOfdmaWidth));
      nonOfdmaRu.SetPhyIndex (currentWidth, m_wifiPhy->GetOperatingChannel ().GetPrimaryChannelIndex (20));

      uint16_t startingFrequency = centerFrequency - (currentWidth / 2);
      centerFrequency = startingFrequency + nonOfdmaWidth * (nonOfdmaRu.GetPhyIndex () - 1) + nonOfdmaWidth / 2;
    }
  return centerFrequency;
}

void
HePhy::StartTx (Ptr<WifiPpdu> ppdu)
{
  NS_LOG_FUNCTION (this << ppdu);
  if (ppdu->GetType () == WIFI_PPDU_TYPE_UL_MU)
    {
      //non-OFDMA part
      Time nonOfdmaDuration = CalculateNonOfdmaDurationForHeTb (ppdu->GetTxVector ());
      Transmit (nonOfdmaDuration, ppdu, "non-OFDMA transmission");

      //OFDMA part
      auto hePpdu = DynamicCast<HePpdu> (ppdu->Copy ()); //since flag will be modified
      NS_ASSERT (hePpdu);
      hePpdu->SetTxPsdFlag (HePpdu::PSD_HE_TB_OFDMA_PORTION);
      Time ofdmaDuration = ppdu->GetTxDuration () - nonOfdmaDuration;
      Simulator::Schedule (nonOfdmaDuration, &PhyEntity::Transmit, this, ofdmaDuration, hePpdu, "OFDMA transmission");
    }
  else
    {
      PhyEntity::StartTx (ppdu);
    }
}

Time
HePhy::CalculateTxDuration (WifiConstPsduMap psduMap, const WifiTxVector& txVector, WifiPhyBand band) const
{
  if (txVector.IsUlMu ())
    {
      NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
      return ConvertLSigLengthToHeTbPpduDuration (txVector.GetLength (), txVector, band);
    }

  Time maxDuration = Seconds (0);
  for (auto & staIdPsdu : psduMap)
    {
      if (txVector.IsDlMu ())
        {
          NS_ASSERT (txVector.GetModulationClass () == WIFI_MOD_CLASS_HE);
          WifiTxVector::HeMuUserInfoMap userInfoMap = txVector.GetHeMuUserInfoMap ();
          NS_ABORT_MSG_IF (userInfoMap.find (staIdPsdu.first) == userInfoMap.end (), "STA-ID in psduMap (" << staIdPsdu.first << ") should be referenced in txVector");
        }
      Time current = WifiPhy::CalculateTxDuration (staIdPsdu.second->GetSize (), txVector, band,
                                                   staIdPsdu.first);
      if (current > maxDuration)
        {
          maxDuration = current;
        }
    }
  NS_ASSERT (maxDuration.IsStrictlyPositive ());
  return maxDuration;
}

void
HePhy::InitializeModes (void)
{
  for (uint8_t i = 0; i < 12; ++i)
    {
      GetHeMcs (i);
    }
}

WifiMode
HePhy::GetHeMcs (uint8_t index)
{
#define CASE(x) \
case x: \
  return GetHeMcs ## x (); \

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
      default:
        NS_ABORT_MSG ("Inexistent index (" << +index << ") requested for HE");
        return WifiMode ();
    }
#undef CASE
}

#define GET_HE_MCS(x) \
WifiMode \
HePhy::GetHeMcs ## x (void) \
{ \
  static WifiMode mcs = CreateHeMcs (x); \
  return mcs; \
}; \

GET_HE_MCS (0)
GET_HE_MCS (1)
GET_HE_MCS (2)
GET_HE_MCS (3)
GET_HE_MCS (4)
GET_HE_MCS (5)
GET_HE_MCS (6)
GET_HE_MCS (7)
GET_HE_MCS (8)
GET_HE_MCS (9)
GET_HE_MCS (10)
GET_HE_MCS (11)
#undef GET_HE_MCS

WifiMode
HePhy::CreateHeMcs (uint8_t index)
{
  NS_ASSERT_MSG (index <= 11, "HeMcs index must be <= 11!");
  return WifiModeFactory::CreateWifiMcs ("HeMcs" + std::to_string (index),
                                         index,
                                         WIFI_MOD_CLASS_HE,
                                         false,
                                         MakeBoundCallback (&GetCodeRate, index),
                                         MakeBoundCallback (&GetConstellationSize, index),
                                         MakeCallback (&GetPhyRateFromTxVector),
                                         MakeCallback (&GetDataRateFromTxVector),
                                         MakeBoundCallback (&GetNonHtReferenceRate, index),
                                         MakeCallback (&IsAllowed));
}

WifiCodeRate
HePhy::GetCodeRate (uint8_t mcsValue)
{
  switch (mcsValue)
    {
      case 10:
        return WIFI_CODE_RATE_3_4;
      case 11:
        return WIFI_CODE_RATE_5_6;
      default:
        return VhtPhy::GetCodeRate (mcsValue);
    }
}

uint16_t
HePhy::GetConstellationSize (uint8_t mcsValue)
{
  switch (mcsValue)
    {
      case 10:
      case 11:
        return 1024;
      default:
        return VhtPhy::GetConstellationSize (mcsValue);
    }
}

uint64_t
HePhy::GetPhyRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss)
{
  WifiCodeRate codeRate = GetCodeRate (mcsValue);
  uint64_t dataRate = GetDataRate (mcsValue, channelWidth, guardInterval, nss);
  return HtPhy::CalculatePhyRate (codeRate, dataRate);
}

uint64_t
HePhy::GetPhyRateFromTxVector (const WifiTxVector& txVector, uint16_t staId /* = SU_STA_ID */)
{
  uint16_t bw = txVector.GetChannelWidth ();
  if (txVector.IsMu ())
    {
      bw = HeRu::GetBandwidth (txVector.GetRu (staId).GetRuType ());
    }
  return HePhy::GetPhyRate (txVector.GetMode (staId).GetMcsValue (),
                            bw,
                            txVector.GetGuardInterval (),
                            txVector.GetNss (staId));
}

uint64_t
HePhy::GetDataRateFromTxVector (const WifiTxVector& txVector, uint16_t staId /* = SU_STA_ID */)
{
  uint16_t bw = txVector.GetChannelWidth ();
  if (txVector.IsMu ())
    {
      bw = HeRu::GetBandwidth (txVector.GetRu (staId).GetRuType ());
    }
  return HePhy::GetDataRate (txVector.GetMode (staId).GetMcsValue (),
                             bw,
                             txVector.GetGuardInterval (),
                             txVector.GetNss (staId));
}

uint64_t
HePhy::GetDataRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss)
{
  NS_ASSERT (guardInterval == 800 || guardInterval == 1600 || guardInterval == 3200);
  NS_ASSERT (nss <= 8);
  return HtPhy::CalculateDataRate (12.8, guardInterval,
                                   GetUsableSubcarriers (channelWidth),
                                   static_cast<uint16_t> (log2 (GetConstellationSize (mcsValue))),
                                   HtPhy::GetCodeRatio (GetCodeRate (mcsValue)), nss);
}

uint16_t
HePhy::GetUsableSubcarriers (uint16_t channelWidth)
{
  switch (channelWidth)
    {
      case 2: //26-tone RU
        return 24;
      case 4: //52-tone RU
        return 48;
      case 8: //106-tone RU
        return 102;
      case 20:
      default:
        return 234;
      case 40:
        return 468;
      case 80:
        return 980;
      case 160:
        return 1960;
    }
}

uint64_t
HePhy::GetNonHtReferenceRate (uint8_t mcsValue)
{
  WifiCodeRate codeRate = GetCodeRate (mcsValue);
  uint16_t constellationSize = GetConstellationSize (mcsValue);
  return CalculateNonHtReferenceRate (codeRate, constellationSize);
}

uint64_t
HePhy::CalculateNonHtReferenceRate (WifiCodeRate codeRate, uint16_t constellationSize)
{
  uint64_t dataRate;
  switch (constellationSize)
    {
      case 1024:
        if (codeRate == WIFI_CODE_RATE_3_4 || codeRate == WIFI_CODE_RATE_5_6)
          {
            dataRate = 54000000;
          }
        else
          {
            NS_FATAL_ERROR ("Trying to get reference rate for a MCS with wrong combination of coding rate and modulation");
          }
        break;
      default:
        dataRate = VhtPhy::CalculateNonHtReferenceRate (codeRate, constellationSize);
    }
  return dataRate;
}

bool
HePhy::IsAllowed (const WifiTxVector& /*txVector*/)
{
  return true;
}

WifiConstPsduMap
HePhy::GetWifiConstPsduMap (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector) const
{
  uint16_t staId = SU_STA_ID;

  if (txVector.IsUlMu ())
    {
      NS_ASSERT (txVector.GetHeMuUserInfoMap ().size () == 1);
      staId = txVector.GetHeMuUserInfoMap ().begin ()->first;
    }

  return WifiConstPsduMap ({std::make_pair (staId, psdu)});
}

uint32_t
HePhy::GetMaxPsduSize (void) const
{
  return 6500631;
}

} //namespace ns3

namespace {

/**
 * Constructor class for HE modes
 */
static class ConstructorHe
{
public:
  ConstructorHe ()
  {
    ns3::HePhy::InitializeModes ();
    ns3::WifiPhy::AddStaticPhyEntity (ns3::WIFI_MOD_CLASS_HE, ns3::Create<ns3::HePhy> ());
  }
} g_constructor_he; ///< the constructor for HE modes

}
