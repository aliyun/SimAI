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
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr> (for logic ported from wifi-phy)
 */

#include "phy-entity.h"
#include "spectrum-wifi-phy.h"
#include "wifi-psdu.h"
#include "preamble-detection-model.h"
#include "frame-capture-model.h"
#include "wifi-utils.h"
#include "wifi-spectrum-signal-parameters.h"
#include "interference-helper.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/assert.h"
#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PhyEntity");

std::ostream & operator << (std::ostream &os, const PhyEntity::PhyRxFailureAction &action)
{
  switch (action)
    {
      case PhyEntity::DROP:
        return (os << "DROP");
      case PhyEntity::ABORT:
        return (os << "ABORT");
      case PhyEntity::IGNORE:
        return (os << "IGNORE");
      default:
        NS_FATAL_ERROR ("Unknown action");
        return (os << "unknown");
    }
}

std::ostream & operator << (std::ostream &os, const PhyEntity::PhyFieldRxStatus &status)
{
  if (status.isSuccess)
    {
      return os << "success";
    }
  else
    {
      return os << "failure (" << status.reason << "/" << status.actionIfFailure << ")";
    }
}

/*******************************************************
 *       Abstract base class for PHY entities
 *******************************************************/

uint64_t PhyEntity::m_globalPpduUid = 0;

PhyEntity::~PhyEntity ()
{
  NS_LOG_FUNCTION (this);
  m_modeList.clear ();
  CancelAllEvents ();
}

void
PhyEntity::SetOwner (Ptr<WifiPhy> wifiPhy)
{
  NS_LOG_FUNCTION (this << wifiPhy);
  m_wifiPhy = wifiPhy;
  m_state = m_wifiPhy->m_state;
}

bool
PhyEntity::IsModeSupported (WifiMode mode) const
{
  for (const auto & m : m_modeList)
    {
      if (m == mode)
        {
          return true;
        }
    }
  return false;
}

uint8_t
PhyEntity::GetNumModes (void) const
{
  return m_modeList.size ();
}

WifiMode
PhyEntity::GetMcs (uint8_t /* index */) const
{
  NS_ABORT_MSG ("This method should be used only for HtPhy and child classes. Use GetMode instead.");
  return WifiMode ();
}

bool
PhyEntity::IsMcsSupported (uint8_t /* index */) const
{
  NS_ABORT_MSG ("This method should be used only for HtPhy and child classes. Use IsModeSupported instead.");
  return false;
}

bool
PhyEntity::HandlesMcsModes (void) const
{
  return false;
}

std::list<WifiMode>::const_iterator
PhyEntity::begin (void) const
{
  return m_modeList.begin ();
}

std::list<WifiMode>::const_iterator
PhyEntity::end (void) const
{
  return m_modeList.end ();
}

WifiMode
PhyEntity::GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const
{
  NS_FATAL_ERROR ("PPDU field is not a SIG field (no sense in retrieving the signaled mode) or is unsupported: " << field);
  return WifiMode (); //should be overloaded
}

WifiPpduField
PhyEntity::GetNextField (WifiPpduField currentField, WifiPreamble preamble) const
{
  auto ppduFormats = GetPpduFormats ();
  const auto itPpdu = ppduFormats.find (preamble);
  if (itPpdu != ppduFormats.end ())
    {
      const auto itField = std::find (itPpdu->second.begin (), itPpdu->second.end (), currentField);
      if (itField != itPpdu->second.end ())
        {
          const auto itNextField = std::next (itField, 1);
          if (itNextField != itPpdu->second.end ())
            {
              return *(itNextField);
            }
          NS_FATAL_ERROR ("No field after " << currentField << " for " << preamble << " for the provided PPDU formats");
        }
      else
        {
          NS_FATAL_ERROR ("Unsupported PPDU field " << currentField << " for " << preamble << " for the provided PPDU formats");
        }
    }
  else
    {
      NS_FATAL_ERROR ("Unsupported preamble " << preamble << " for the provided PPDU formats");
    }
  return WifiPpduField::WIFI_PPDU_FIELD_PREAMBLE; // Silence compiler warning
}

Time
PhyEntity::GetDuration (WifiPpduField field, const WifiTxVector& txVector) const
{
  if (field > WIFI_PPDU_FIELD_SIG_B)
    {
      NS_FATAL_ERROR ("Unsupported PPDU field");
    }
  return MicroSeconds (0); //should be overloaded
}

Time
PhyEntity::CalculatePhyPreambleAndHeaderDuration (const WifiTxVector& txVector) const
{
  Time duration = MicroSeconds (0);
  for (uint8_t field = WIFI_PPDU_FIELD_PREAMBLE; field < WIFI_PPDU_FIELD_DATA; ++field)
    {
      duration += GetDuration (static_cast<WifiPpduField> (field), txVector);
    }
  return duration;
}

WifiConstPsduMap
PhyEntity::GetWifiConstPsduMap (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector) const
{
  return WifiConstPsduMap ({std::make_pair (SU_STA_ID, psdu)});
}

Ptr<const WifiPsdu>
PhyEntity::GetAddressedPsduInPpdu (Ptr<const WifiPpdu> ppdu) const
{
  return ppdu->GetPsdu ();
}

PhyEntity::PhyHeaderSections
PhyEntity::GetPhyHeaderSections (const WifiTxVector& txVector, Time ppduStart) const
{
  PhyHeaderSections map;
  WifiPpduField field = WIFI_PPDU_FIELD_PREAMBLE; //preamble always present
  Time start = ppduStart;

  while (field != WIFI_PPDU_FIELD_DATA)
    {
      Time duration = GetDuration (field, txVector);
      map[field] = std::make_pair (std::make_pair (start, start + duration),
                                   GetSigMode (field, txVector));
      //Move to next field
      start += duration;
      field = GetNextField (field, txVector.GetPreambleType ());
    }
  return map;
}

Ptr<WifiPpdu>
PhyEntity::BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time /* ppduDuration */)
{
  NS_LOG_FUNCTION (this << psdus << txVector);
  NS_FATAL_ERROR ("This method is unsupported for the base PhyEntity class. Use the overloaded version in the amendment-specific subclasses instead!");
  return Create<WifiPpdu> (psdus.begin ()->second, txVector); //should be overloaded
}

Time
PhyEntity::GetDurationUpToField (WifiPpduField field, const WifiTxVector& txVector) const
{
  if (field == WIFI_PPDU_FIELD_DATA) //this field is not in the map returned by GetPhyHeaderSections
    {
      return CalculatePhyPreambleAndHeaderDuration (txVector);
    }
  const auto & sections = GetPhyHeaderSections (txVector, NanoSeconds (0));
  auto it = sections.find (field);
  NS_ASSERT (it != sections.end ());
  const auto & startStopTimes = it->second.first;
  return startStopTimes.first; //return the start time of field relatively to the beginning of the PPDU
}

PhyEntity::SnrPer
PhyEntity::GetPhyHeaderSnrPer (WifiPpduField field, Ptr<Event> event) const
{
  uint16_t measurementChannelWidth = GetMeasurementChannelWidth (event->GetPpdu ());
  return m_wifiPhy->m_interference->CalculatePhyHeaderSnrPer (event, measurementChannelWidth, m_wifiPhy->GetPrimaryBand (measurementChannelWidth),
                                                             field);
}

void
PhyEntity::StartReceiveField (WifiPpduField field, Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << field << *event);
  NS_ASSERT (m_wifiPhy); //no sense if no owner WifiPhy instance
  NS_ASSERT (m_wifiPhy->m_endPhyRxEvent.IsExpired ());
  NS_ABORT_MSG_IF (field == WIFI_PPDU_FIELD_PREAMBLE, "Use the StartReceivePreamble method for preamble reception");
  //Handle special cases of data reception
  if (field == WIFI_PPDU_FIELD_DATA)
    {
      StartReceivePayload (event);
      return;
    }

  bool supported = DoStartReceiveField (field, event);
  NS_ABORT_MSG_IF (!supported, "Unknown field " << field << " for this PHY entity"); //TODO see what to do if not supported
  Time duration = GetDuration (field, event->GetTxVector ());
  m_wifiPhy->m_endPhyRxEvent = Simulator::Schedule (duration, &PhyEntity::EndReceiveField, this, field, event);
  m_state->SwitchMaybeToCcaBusy (duration); //keep in CCA busy state up to reception of Data (will then switch to RX)
}

void
PhyEntity::EndReceiveField (WifiPpduField field, Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << field << *event);
  NS_ASSERT (m_wifiPhy); //no sense if no owner WifiPhy instance
  NS_ASSERT (m_wifiPhy->m_endPhyRxEvent.IsExpired ());
  PhyFieldRxStatus status = DoEndReceiveField (field, event);
  WifiTxVector txVector = event->GetTxVector ();
  if (status.isSuccess) //move to next field if reception succeeded
    {
      StartReceiveField (GetNextField (field, txVector.GetPreambleType ()), event);
    }
  else
    {
      Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
      switch (status.actionIfFailure)
        {
          case ABORT:
            //Abort reception, but consider medium as busy
            AbortCurrentReception (status.reason);
            if (event->GetEndTime () > (Simulator::Now () + m_state->GetDelayUntilIdle ()))
              {
                m_wifiPhy->SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (ppdu));
              }
            break;
          case DROP:
            //Notify drop, keep in CCA busy, and perform same processing as IGNORE case
            if (status.reason == FILTERED)
              {
                //PHY-RXSTART is immediately followed by PHY-RXEND (Filtered)
                m_wifiPhy->m_phyRxPayloadBeginTrace (txVector, NanoSeconds (0)); //this callback (equivalent to PHY-RXSTART primitive) is also triggered for filtered PPDUs
              }
            m_wifiPhy->NotifyRxDrop (GetAddressedPsduInPpdu (ppdu), status.reason);
            m_state->SwitchMaybeToCcaBusy (GetRemainingDurationAfterField (ppdu, field)); //keep in CCA busy state till the end
          //no break
          case IGNORE:
            //Keep in Rx state and reset at end
            m_endRxPayloadEvents.push_back (Simulator::Schedule (GetRemainingDurationAfterField (ppdu, field),
                                                                 &PhyEntity::ResetReceive, this, event));
            break;
          default:
            NS_FATAL_ERROR ("Unknown action in case of failure");
        }
    }
}

Time
PhyEntity::GetRemainingDurationAfterField (Ptr<const WifiPpdu> ppdu, WifiPpduField field) const
{
  const WifiTxVector& txVector = ppdu->GetTxVector ();
  return ppdu->GetTxDuration () - (GetDurationUpToField (field, txVector) + GetDuration (field, txVector));
}

bool
PhyEntity::DoStartReceiveField (WifiPpduField field, Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << field << *event);
  NS_ASSERT (field != WIFI_PPDU_FIELD_PREAMBLE && field != WIFI_PPDU_FIELD_DATA); //handled apart for the time being
  auto ppduFormats = GetPpduFormats ();
  auto itFormat = ppduFormats.find (event->GetPpdu ()->GetPreamble ());
  if (itFormat != ppduFormats.end ())
    {
      auto itField = std::find (itFormat->second.begin (), itFormat->second.end (), field);
      if (itField != itFormat->second.end ())
        {
          return true; //supported field so we can start receiving
        }
    }
  return false; //unsupported otherwise
}

PhyEntity::PhyFieldRxStatus
PhyEntity::DoEndReceiveField (WifiPpduField field, Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << field << *event);
  NS_ASSERT (field != WIFI_PPDU_FIELD_DATA); //handled apart for the time being
  if (field == WIFI_PPDU_FIELD_PREAMBLE)
    {
      return DoEndReceivePreamble (event);
    }
  return PhyFieldRxStatus (false); //failed reception by default
}

void
PhyEntity::StartReceivePreamble (Ptr<WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW,
                                 Time rxDuration)
{
  //The total RX power corresponds to the maximum over all the bands
  auto it = std::max_element (rxPowersW.begin (), rxPowersW.end (),
                              [] (const std::pair<WifiSpectrumBand, double> &p1, const std::pair<WifiSpectrumBand, double> &p2) {
                                return p1.second < p2.second;
                              });
  NS_LOG_FUNCTION (this << ppdu << it->second);

  Ptr<Event> event = DoGetEvent (ppdu, rxPowersW);
  if (event == nullptr)
    {
      //PPDU should be simply considered as interference (once it has been accounted for in InterferenceHelper)
      return;
    }

  Time endRx = Simulator::Now () + rxDuration;
  if (m_state->GetState () == WifiPhyState::OFF)
    {
      NS_LOG_DEBUG ("Cannot start RX because device is OFF");
      if (endRx > (Simulator::Now () + m_state->GetDelayUntilIdle ()))
        {
          m_wifiPhy->SwitchMaybeToCcaBusy (m_wifiPhy->GetMeasurementChannelWidth (nullptr));
        }
      DropPreambleEvent (ppdu, WifiPhyRxfailureReason::POWERED_OFF, endRx, m_wifiPhy->GetMeasurementChannelWidth (ppdu));
      return;
    }

  if (ppdu->IsTruncatedTx ())
    {
      NS_LOG_DEBUG ("Packet reception stopped because transmitter has been switched off");
      if (endRx > (Simulator::Now () + m_state->GetDelayUntilIdle ()))
        {
          m_wifiPhy->SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (ppdu));
        }
      DropPreambleEvent (ppdu, WifiPhyRxfailureReason::TRUNCATED_TX, endRx, m_wifiPhy->GetMeasurementChannelWidth (ppdu));
      return;
    }

  switch (m_state->GetState ())
    {
      case WifiPhyState::SWITCHING:
        NS_LOG_DEBUG ("Drop packet because of channel switching");
        /*
         * Packets received on the upcoming channel are added to the event list
         * during the switching state. This way the medium can be correctly sensed
         * when the device listens to the channel for the first time after the
         * switching e.g. after channel switching, the channel may be sensed as
         * busy due to other devices' transmissions started before the end of
         * the switching.
         */
        DropPreambleEvent (ppdu, CHANNEL_SWITCHING, endRx, m_wifiPhy->GetMeasurementChannelWidth (ppdu));
        break;
      case WifiPhyState::RX:
        if (m_wifiPhy->m_frameCaptureModel != 0
            && m_wifiPhy->m_frameCaptureModel->IsInCaptureWindow (m_wifiPhy->m_timeLastPreambleDetected)
            && m_wifiPhy->m_frameCaptureModel->CaptureNewFrame (m_wifiPhy->m_currentEvent, event))
          {
            AbortCurrentReception (FRAME_CAPTURE_PACKET_SWITCH);
            NS_LOG_DEBUG ("Switch to new packet");
            StartPreambleDetectionPeriod (event);
          }
        else
          {
            NS_LOG_DEBUG ("Drop packet because already in Rx");
            DropPreambleEvent (ppdu, RXING, endRx, m_wifiPhy->GetMeasurementChannelWidth (ppdu));
            if (m_wifiPhy->m_currentEvent == 0)
              {
                /*
                 * We are here because the non-legacy PHY header has not been successfully received.
                 * The PHY is kept in RX state for the duration of the PPDU, but EndReceive function is
                 * not called when the reception of the PPDU is finished, which is responsible to clear
                 * m_currentPreambleEvents. As a result, m_currentPreambleEvents should be cleared here.
                 */
                m_wifiPhy->m_currentPreambleEvents.clear ();
              }
          }
        break;
      case WifiPhyState::TX:
        NS_LOG_DEBUG ("Drop packet because already in Tx");
        DropPreambleEvent (ppdu, TXING, endRx, m_wifiPhy->GetMeasurementChannelWidth (ppdu));
        break;
      case WifiPhyState::CCA_BUSY:
        if (m_wifiPhy->m_currentEvent != 0)
          {
            if (m_wifiPhy->m_frameCaptureModel != 0
                && m_wifiPhy->m_frameCaptureModel->IsInCaptureWindow (m_wifiPhy->m_timeLastPreambleDetected)
                && m_wifiPhy->m_frameCaptureModel->CaptureNewFrame (m_wifiPhy->m_currentEvent, event))
              {
                AbortCurrentReception (FRAME_CAPTURE_PACKET_SWITCH);
                NS_LOG_DEBUG ("Switch to new packet");
                StartPreambleDetectionPeriod (event);
              }
            else
              {
                NS_LOG_DEBUG ("Drop packet because already decoding preamble");
                DropPreambleEvent (ppdu, BUSY_DECODING_PREAMBLE, endRx, m_wifiPhy->GetMeasurementChannelWidth (ppdu));
              }
          }
        else
          {
            StartPreambleDetectionPeriod (event);
          }
        break;
      case WifiPhyState::IDLE:
        NS_ASSERT (m_wifiPhy->m_currentEvent == 0);
        StartPreambleDetectionPeriod (event);
        break;
      case WifiPhyState::SLEEP:
        NS_LOG_DEBUG ("Drop packet because in sleep mode");
        DropPreambleEvent (ppdu, SLEEPING, endRx, m_wifiPhy->GetMeasurementChannelWidth (nullptr));
        break;
      default:
        NS_FATAL_ERROR ("Invalid WifiPhy state.");
        break;
    }
}

void
PhyEntity::DropPreambleEvent (Ptr<const WifiPpdu> ppdu, WifiPhyRxfailureReason reason, Time endRx, uint16_t measurementChannelWidth)
{
  NS_LOG_FUNCTION (this << ppdu << reason << endRx << measurementChannelWidth);
  m_wifiPhy->NotifyRxDrop (GetAddressedPsduInPpdu (ppdu), reason);
  auto it = m_wifiPhy->m_currentPreambleEvents.find (std::make_pair (ppdu->GetUid (), ppdu->GetPreamble ()));
  if (it != m_wifiPhy->m_currentPreambleEvents.end ())
    {
      m_wifiPhy->m_currentPreambleEvents.erase (it);
    }
  if (endRx > (Simulator::Now () + m_state->GetDelayUntilIdle ()))
    {
      //that PPDU will be noise _after_ the end of the current event.
      m_wifiPhy->SwitchMaybeToCcaBusy (measurementChannelWidth);
    }
}

void
PhyEntity::ErasePreambleEvent (Ptr<const WifiPpdu> ppdu, Time rxDuration)
{
  NS_LOG_FUNCTION (this << ppdu << rxDuration);
  auto it = m_wifiPhy->m_currentPreambleEvents.find (std::make_pair (ppdu->GetUid (), ppdu->GetPreamble ()));
  if (it != m_wifiPhy->m_currentPreambleEvents.end ())
    {
      m_wifiPhy->m_currentPreambleEvents.erase (it);
    }
  if (m_wifiPhy->m_currentPreambleEvents.empty ())
    {
      m_wifiPhy->Reset ();
    }

  if (rxDuration > m_state->GetDelayUntilIdle ())
    {
      //this PPDU will be noise _after_ the completion of the current event
      m_wifiPhy->SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (ppdu));
    }
}

uint16_t
PhyEntity::GetStaId (const Ptr<const WifiPpdu> /* ppdu */) const
{
  return SU_STA_ID;
}

void
PhyEntity::StartReceivePayload (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  NS_ASSERT (m_wifiPhy->m_endPhyRxEvent.IsExpired ());
  const WifiTxVector& txVector = event->GetTxVector ();
  Time payloadDuration = event->GetPpdu ()->GetTxDuration () - CalculatePhyPreambleAndHeaderDuration (txVector);

  //TODO: Add method in WifiPhy to clear all other PHYs (since this one is starting Rx)
  m_state->SwitchToRx (payloadDuration);
  m_wifiPhy->m_phyRxPayloadBeginTrace (txVector, payloadDuration); //this callback (equivalent to PHY-RXSTART primitive) is triggered only if headers have been correctly decoded and that the mode within is supported

  DoStartReceivePayload (event);
}

void
PhyEntity::DoStartReceivePayload (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
  NS_LOG_DEBUG ("Receiving PSDU");
  uint16_t staId = GetStaId (ppdu);
  m_signalNoiseMap.insert ({std::make_pair (ppdu->GetUid (), staId), SignalNoiseDbm ()});
  m_statusPerMpduMap.insert ({std::make_pair (ppdu->GetUid (), staId), std::vector<bool> ()});
  ScheduleEndOfMpdus (event);
  m_endRxPayloadEvents.push_back (Simulator::Schedule (ppdu->GetTxDuration () - CalculatePhyPreambleAndHeaderDuration (event->GetTxVector ()),
                                                       &PhyEntity::EndReceivePayload, this, event));
}

void
PhyEntity::ScheduleEndOfMpdus (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
  Ptr<const WifiPsdu> psdu = GetAddressedPsduInPpdu (ppdu);
  const WifiTxVector& txVector = event->GetTxVector ();
  uint16_t staId = GetStaId (ppdu);
  Time endOfMpduDuration = NanoSeconds (0);
  Time relativeStart = NanoSeconds (0);
  Time psduDuration = ppdu->GetTxDuration () - CalculatePhyPreambleAndHeaderDuration (txVector);
  Time remainingAmpduDuration = psduDuration;
  size_t nMpdus = psdu->GetNMpdus ();
  MpduType mpduType = (nMpdus > 1) ? FIRST_MPDU_IN_AGGREGATE : (psdu->IsSingle () ? SINGLE_MPDU : NORMAL_MPDU);
  uint32_t totalAmpduSize = 0;
  double totalAmpduNumSymbols = 0.0;
  auto mpdu = psdu->begin ();
  for (size_t i = 0; i < nMpdus && mpdu != psdu->end (); ++mpdu)
    {
      uint32_t size = (mpduType == NORMAL_MPDU) ? psdu->GetSize () : psdu->GetAmpduSubframeSize (i);
      Time mpduDuration = m_wifiPhy->GetPayloadDuration (size, txVector,
                                                         m_wifiPhy->GetPhyBand (), mpduType, true, totalAmpduSize,
                                                         totalAmpduNumSymbols, staId);

      remainingAmpduDuration -= mpduDuration;
      if (i == (nMpdus - 1) && !remainingAmpduDuration.IsZero ()) //no more MPDUs coming
        {
          if (remainingAmpduDuration < NanoSeconds (txVector.GetGuardInterval ())) //enables to ignore padding
            {
              mpduDuration += remainingAmpduDuration; //apply a correction just in case rounding had induced slight shift
            }
        }

      endOfMpduDuration += mpduDuration;
      NS_LOG_INFO ("Schedule end of MPDU #" << i << " in " << endOfMpduDuration.As (Time::NS) <<
                   " (relativeStart=" << relativeStart.As (Time::NS) << ", mpduDuration=" << mpduDuration.As (Time::NS) <<
                   ", remainingAmdpuDuration=" << remainingAmpduDuration.As (Time::NS) << ")");
      m_endOfMpduEvents.push_back (Simulator::Schedule (endOfMpduDuration, &PhyEntity::EndOfMpdu, this, event, Create<WifiPsdu> (*mpdu, false), i, relativeStart, mpduDuration));

      //Prepare next iteration
      ++i;
      relativeStart += mpduDuration;
      mpduType = (i == (nMpdus - 1)) ? LAST_MPDU_IN_AGGREGATE : MIDDLE_MPDU_IN_AGGREGATE;
    }
}

void
PhyEntity::EndOfMpdu (Ptr<Event> event, Ptr<const WifiPsdu> psdu, size_t mpduIndex, Time relativeStart, Time mpduDuration)
{
  NS_LOG_FUNCTION (this << *event << mpduIndex << relativeStart << mpduDuration);
  Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
  WifiTxVector txVector = event->GetTxVector ();
  uint16_t staId = GetStaId (ppdu);

  std::pair<bool, SignalNoiseDbm> rxInfo = GetReceptionStatus (psdu, event, staId, relativeStart, mpduDuration);
  NS_LOG_DEBUG ("Extracted MPDU #" << mpduIndex << ": duration: " << mpduDuration.As (Time::NS) <<
                ", correct reception: " << rxInfo.first << ", Signal/Noise: " << rxInfo.second.signal << "/" << rxInfo.second.noise << "dBm");

  auto signalNoiseIt = m_signalNoiseMap.find (std::make_pair (ppdu->GetUid (), staId));
  NS_ASSERT (signalNoiseIt != m_signalNoiseMap.end ());
  signalNoiseIt->second = rxInfo.second;

  RxSignalInfo rxSignalInfo;
  rxSignalInfo.snr = rxInfo.second.signal / rxInfo.second.noise;
  rxSignalInfo.rssi = rxInfo.second.signal;

  auto statusPerMpduIt = m_statusPerMpduMap.find (std::make_pair (ppdu->GetUid (), staId));
  NS_ASSERT (statusPerMpduIt != m_statusPerMpduMap.end ());
  statusPerMpduIt->second.push_back (rxInfo.first);

  if (rxInfo.first && GetAddressedPsduInPpdu (ppdu)->GetNMpdus () > 1)
    {
      //only done for correct MPDU that is part of an A-MPDU
      m_state->ContinueRxNextMpdu (Copy (psdu), rxSignalInfo, txVector);
    }
}

void
PhyEntity::EndReceivePayload (Ptr<Event> event)
{
  Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
  WifiTxVector txVector = event->GetTxVector ();
  Time psduDuration = ppdu->GetTxDuration () - CalculatePhyPreambleAndHeaderDuration (txVector);
  NS_LOG_FUNCTION (this << *event << psduDuration);
  NS_ASSERT (event->GetEndTime () == Simulator::Now ());
  uint16_t staId = GetStaId (ppdu);
  const auto & channelWidthAndBand = GetChannelWidthAndBand (event->GetTxVector (), staId);
  double snr = m_wifiPhy->m_interference->CalculateSnr (event, channelWidthAndBand.first, txVector.GetNss (staId), channelWidthAndBand.second);

  Ptr<const WifiPsdu> psdu = GetAddressedPsduInPpdu (ppdu);
  m_wifiPhy->NotifyRxEnd (psdu);

  auto signalNoiseIt = m_signalNoiseMap.find (std::make_pair (ppdu->GetUid (), staId));
  NS_ASSERT (signalNoiseIt != m_signalNoiseMap.end ());
  auto statusPerMpduIt = m_statusPerMpduMap.find (std::make_pair (ppdu->GetUid (), staId));
  NS_ASSERT (statusPerMpduIt != m_statusPerMpduMap.end ());

  if (std::count (statusPerMpduIt->second.begin (), statusPerMpduIt->second.end (), true))
    {
      //At least one MPDU has been successfully received
      m_wifiPhy->NotifyMonitorSniffRx (psdu, m_wifiPhy->GetFrequency (), txVector, signalNoiseIt->second, statusPerMpduIt->second, staId);
      RxSignalInfo rxSignalInfo;
      rxSignalInfo.snr = snr;
      rxSignalInfo.rssi = signalNoiseIt->second.signal; //same information for all MPDUs
      m_state->SwitchFromRxEndOk (Copy (psdu), rxSignalInfo, txVector, staId, statusPerMpduIt->second);
      m_wifiPhy->m_previouslyRxPpduUid = ppdu->GetUid (); //store UID only if reception is successful (because otherwise trigger won't be read by MAC layer)
    }
  else
    {
      m_state->SwitchFromRxEndError (Copy (psdu), snr);
    }

  DoEndReceivePayload (ppdu);
  m_wifiPhy->SwitchMaybeToCcaBusy (GetMeasurementChannelWidth (ppdu));
}

void
PhyEntity::DoEndReceivePayload (Ptr<const WifiPpdu> ppdu)
{
  NS_LOG_FUNCTION (this << ppdu);
  NS_ASSERT (m_wifiPhy->GetLastRxEndTime () == Simulator::Now ());
  NotifyInterferenceRxEndAndClear (false); //don't reset WifiPhy

  m_wifiPhy->m_currentEvent = 0;
  m_wifiPhy->m_currentPreambleEvents.clear ();
  m_endRxPayloadEvents.clear ();
}

std::pair<bool, SignalNoiseDbm>
PhyEntity::GetReceptionStatus (Ptr<const WifiPsdu> psdu, Ptr<Event> event, uint16_t staId,
                               Time relativeMpduStart, Time mpduDuration)
{
  NS_LOG_FUNCTION (this << *psdu << *event << staId << relativeMpduStart << mpduDuration);
  const auto & channelWidthAndBand = GetChannelWidthAndBand (event->GetTxVector (), staId);
  SnrPer snrPer = m_wifiPhy->m_interference->CalculatePayloadSnrPer (event, channelWidthAndBand.first, channelWidthAndBand.second, staId,
                                                                    std::make_pair (relativeMpduStart, relativeMpduStart + mpduDuration));

  WifiMode mode = event->GetTxVector ().GetMode (staId);
  NS_LOG_DEBUG ("rate=" << (mode.GetDataRate (event->GetTxVector (), staId)) <<
                ", SNR(dB)=" << RatioToDb (snrPer.snr) << ", PER=" << snrPer.per << ", size=" << psdu->GetSize () <<
                ", relativeStart = " << relativeMpduStart.As (Time::NS) << ", duration = " << mpduDuration.As (Time::NS));

  // There are two error checks: PER and receive error model check.
  // PER check models is typical for Wi-Fi and is based on signal modulation;
  // Receive error model is optional, if we have an error model and
  // it indicates that the packet is corrupt, drop the packet.
  SignalNoiseDbm signalNoise;
  signalNoise.signal = WToDbm (event->GetRxPowerW (channelWidthAndBand.second));
  signalNoise.noise = WToDbm (event->GetRxPowerW (channelWidthAndBand.second) / snrPer.snr);
  if (GetRandomValue () > snrPer.per
      && !(m_wifiPhy->m_postReceptionErrorModel && m_wifiPhy->m_postReceptionErrorModel->IsCorrupt (psdu->GetPacket ()->Copy ())))
    {
      NS_LOG_DEBUG ("Reception succeeded: " << psdu);
      return std::make_pair (true, signalNoise);
    }
  else
    {
      NS_LOG_DEBUG ("Reception failed: " << psdu);
      return std::make_pair (false, signalNoise);
    }
}

std::pair<uint16_t, WifiSpectrumBand>
PhyEntity::GetChannelWidthAndBand (const WifiTxVector& txVector, uint16_t /* staId */) const
{
  uint16_t channelWidth = GetRxChannelWidth (txVector);
  return std::make_pair (channelWidth, m_wifiPhy->GetPrimaryBand (channelWidth));
}

const std::map <std::pair<uint64_t, WifiPreamble>, Ptr<Event> > &
PhyEntity::GetCurrentPreambleEvents (void) const
{
  return m_wifiPhy->m_currentPreambleEvents;
}

void
PhyEntity::AddPreambleEvent (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  Ptr<const WifiPpdu> ppdu = event->GetPpdu ();
  m_wifiPhy->m_currentPreambleEvents.insert ({std::make_pair (ppdu->GetUid (), ppdu->GetPreamble ()), event});
}

Ptr<Event>
PhyEntity::DoGetEvent (Ptr<const WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW)
{
  Ptr<Event> event = CreateInterferenceEvent (ppdu, ppdu->GetTxVector (), ppdu->GetTxDuration (), rxPowersW);

  //We store all incoming preamble events, and a decision is made at the end of the preamble detection window.
  auto uidPreamblePair = std::make_pair (ppdu->GetUid (), ppdu->GetPreamble ());
  NS_ASSERT (m_wifiPhy->m_currentPreambleEvents.find (uidPreamblePair) == m_wifiPhy->m_currentPreambleEvents.end ());
  m_wifiPhy->m_currentPreambleEvents.insert ({uidPreamblePair, event});
  return event;
}

Ptr<Event>
PhyEntity::CreateInterferenceEvent (Ptr<const WifiPpdu> ppdu, const WifiTxVector& txVector, Time duration, RxPowerWattPerChannelBand& rxPower, bool isStartOfdmaRxing /* = false */)
{
  return m_wifiPhy->m_interference->Add (ppdu, txVector, duration, rxPower, isStartOfdmaRxing);
}

void
PhyEntity::UpdateInterferenceEvent (Ptr<Event> event, const RxPowerWattPerChannelBand& rxPower)
{
  m_wifiPhy->m_interference->UpdateEvent (event, rxPower);
}

void
PhyEntity::NotifyInterferenceRxEndAndClear (bool reset)
{
  m_wifiPhy->m_interference->NotifyRxEnd (Simulator::Now ());
  m_signalNoiseMap.clear ();
  m_statusPerMpduMap.clear ();
  for (const auto & endOfMpduEvent : m_endOfMpduEvents)
    {
      NS_ASSERT (endOfMpduEvent.IsExpired ());
    }
  m_endOfMpduEvents.clear ();
  if (reset)
    {
      m_wifiPhy->Reset ();
    }
}

PhyEntity::PhyFieldRxStatus
PhyEntity::DoEndReceivePreamble (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  NS_ASSERT (m_wifiPhy->m_currentPreambleEvents.size () == 1); //Synched on one after detection period
  return PhyFieldRxStatus (true); //always consider that preamble has been correctly received if preamble detection was OK
}

void
PhyEntity::StartPreambleDetectionPeriod (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  NS_LOG_DEBUG ("Sync to signal (power=" << WToDbm (GetRxPowerWForPpdu (event)) << "dBm)");
  m_wifiPhy->m_interference->NotifyRxStart (); //We need to notify it now so that it starts recording events
  m_endPreambleDetectionEvents.push_back (Simulator::Schedule (m_wifiPhy->GetPreambleDetectionDuration (), &PhyEntity::EndPreambleDetectionPeriod, this, event));
}

void
PhyEntity::EndPreambleDetectionPeriod (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  NS_ASSERT (!m_wifiPhy->IsStateRx ());
  NS_ASSERT (m_wifiPhy->m_endPhyRxEvent.IsExpired ()); //since end of preamble reception is scheduled by this method upon success

  //calculate PER on the measurement channel for PHY headers
  uint16_t measurementChannelWidth = GetMeasurementChannelWidth (event->GetPpdu ());
  auto measurementBand = m_wifiPhy->GetPrimaryBand (measurementChannelWidth);
  double maxRxPowerW = -1; //in case current event may not be sent on measurement channel (rxPowerW would be equal to 0)
  Ptr<Event> maxEvent;
  NS_ASSERT (!m_wifiPhy->m_currentPreambleEvents.empty ());
  for (auto preambleEvent : m_wifiPhy->m_currentPreambleEvents)
    {
      double rxPowerW = preambleEvent.second->GetRxPowerW (measurementBand);
      if (rxPowerW > maxRxPowerW)
        {
          maxRxPowerW = rxPowerW;
          maxEvent = preambleEvent.second;
        }
    }

  NS_ASSERT (maxEvent != 0);
  if (maxEvent != event)
    {
      NS_LOG_DEBUG ("Receiver got a stronger packet with UID " << maxEvent->GetPpdu ()->GetUid () << " during preamble detection: drop packet with UID " << event->GetPpdu ()->GetUid ());
      m_wifiPhy->NotifyRxDrop (GetAddressedPsduInPpdu (event->GetPpdu ()), BUSY_DECODING_PREAMBLE);
      auto it = m_wifiPhy->m_currentPreambleEvents.find (std::make_pair (event->GetPpdu ()->GetUid (), event->GetPpdu ()->GetPreamble ()));
      m_wifiPhy->m_currentPreambleEvents.erase (it);
      //This is needed to cleanup the m_firstPowerPerBand so that the first power corresponds to the power at the start of the PPDU
      m_wifiPhy->m_interference->NotifyRxEnd (maxEvent->GetStartTime ());
      //Make sure InterferenceHelper keeps recording events
      m_wifiPhy->m_interference->NotifyRxStart ();
      return;
    }

  m_wifiPhy->m_currentEvent = event;

  double snr = m_wifiPhy->m_interference->CalculateSnr (m_wifiPhy->m_currentEvent, measurementChannelWidth, 1, measurementBand);
  NS_LOG_DEBUG ("SNR(dB)=" << RatioToDb (snr) << " at end of preamble detection period");

  if ((!m_wifiPhy->m_preambleDetectionModel && maxRxPowerW > 0.0)
      || (m_wifiPhy->m_preambleDetectionModel && m_wifiPhy->m_preambleDetectionModel->IsPreambleDetected (m_wifiPhy->m_currentEvent->GetRxPowerW (measurementBand), snr, measurementChannelWidth)))
    {
      //A bit convoluted but it enables to sync all PHYs
      for (auto & it : m_wifiPhy->m_phyEntities)
        {
          it.second->CancelRunningEndPreambleDetectionEvents (true);
        }

      for (auto it = m_wifiPhy->m_currentPreambleEvents.begin (); it != m_wifiPhy->m_currentPreambleEvents.end (); )
        {
          if (it->second != m_wifiPhy->m_currentEvent)
            {
              NS_LOG_DEBUG ("Drop packet with UID " << it->first.first << " and preamble " << it->first.second << " arrived at time " << it->second->GetStartTime ());
              WifiPhyRxfailureReason reason;
              if (m_wifiPhy->m_currentEvent->GetPpdu ()->GetUid () > it->first.first)
                {
                  reason = PREAMBLE_DETECTION_PACKET_SWITCH;
                  //This is needed to cleanup the m_firstPowerPerBand so that the first power corresponds to the power at the start of the PPDU
                  m_wifiPhy->m_interference->NotifyRxEnd (m_wifiPhy->m_currentEvent->GetStartTime ());
                }
              else
                {
                  reason = BUSY_DECODING_PREAMBLE;
                }
              m_wifiPhy->NotifyRxDrop (GetAddressedPsduInPpdu (it->second->GetPpdu ()), reason);
              it = m_wifiPhy->m_currentPreambleEvents.erase (it);
            }
          else
            {
              ++it;
            }
        }

      //Make sure InterferenceHelper keeps recording events
      m_wifiPhy->m_interference->NotifyRxStart ();

      m_wifiPhy->NotifyRxBegin (GetAddressedPsduInPpdu (m_wifiPhy->m_currentEvent->GetPpdu ()), m_wifiPhy->m_currentEvent->GetRxPowerWPerBand ());
      m_wifiPhy->m_timeLastPreambleDetected = Simulator::Now ();

      //Continue receiving preamble
      Time durationTillEnd = GetDuration (WIFI_PPDU_FIELD_PREAMBLE, event->GetTxVector ()) - m_wifiPhy->GetPreambleDetectionDuration ();
      m_state->SwitchMaybeToCcaBusy (durationTillEnd); //will be prolonged by next field
      m_wifiPhy->m_endPhyRxEvent = Simulator::Schedule (durationTillEnd, &PhyEntity::EndReceiveField, this, WIFI_PPDU_FIELD_PREAMBLE, event);
    }
  else
    {
      NS_LOG_DEBUG ("Drop packet because PHY preamble detection failed");
      // Like CCA-SD, CCA-ED is governed by the 4 us CCA window to flag CCA-BUSY
      // for any received signal greater than the CCA-ED threshold.
      DropPreambleEvent (m_wifiPhy->m_currentEvent->GetPpdu (), PREAMBLE_DETECT_FAILURE, m_wifiPhy->m_currentEvent->GetEndTime (), m_wifiPhy->GetMeasurementChannelWidth (m_wifiPhy->m_currentEvent->GetPpdu ()));
      if (m_wifiPhy->m_currentPreambleEvents.empty ())
        {
          //Do not erase events if there are still pending preamble events to be processed
          m_wifiPhy->m_interference->NotifyRxEnd (Simulator::Now ());
        }
      m_wifiPhy->m_currentEvent = 0;
      //Cancel preamble reception
      m_wifiPhy->m_endPhyRxEvent.Cancel ();
    }
}

bool
PhyEntity::IsConfigSupported (Ptr<const WifiPpdu> ppdu) const
{
  WifiMode txMode = ppdu->GetTxVector ().GetMode ();
  if (!IsModeSupported (txMode))
    {
      NS_LOG_DEBUG ("Drop packet because it was sent using an unsupported mode (" << txMode << ")");
      return false;
    }
  return true;
}

void
PhyEntity::CancelAllEvents (void)
{
  NS_LOG_FUNCTION (this);
  for (auto & endPreambleDetectionEvent : m_endPreambleDetectionEvents)
    {
      endPreambleDetectionEvent.Cancel ();
    }
  m_endPreambleDetectionEvents.clear ();
  for (auto & endRxPayloadEvent : m_endRxPayloadEvents)
    {
      endRxPayloadEvent.Cancel ();
    }
  m_endRxPayloadEvents.clear ();
  for (auto & endMpduEvent : m_endOfMpduEvents)
    {
      endMpduEvent.Cancel ();
    }
  m_endOfMpduEvents.clear ();
}

bool
PhyEntity::NoEndPreambleDetectionEvents (void) const
{
  return m_endPreambleDetectionEvents.empty ();
}

void
PhyEntity::CancelRunningEndPreambleDetectionEvents (bool clear /* = false */)
{
  NS_LOG_FUNCTION (this << clear);
  for (auto & endPreambleDetectionEvent : m_endPreambleDetectionEvents)
    {
      if (endPreambleDetectionEvent.IsRunning ())
        {
          endPreambleDetectionEvent.Cancel ();
        }
    }
  if (clear)
    {
      m_endPreambleDetectionEvents.clear ();
    }
}

void
PhyEntity::AbortCurrentReception (WifiPhyRxfailureReason reason)
{
  NS_LOG_FUNCTION (this << reason);
  DoAbortCurrentReception (reason);
  m_wifiPhy->AbortCurrentReception (reason);
}

void
PhyEntity::DoAbortCurrentReception (WifiPhyRxfailureReason reason)
{
  NS_LOG_FUNCTION (this << reason);
  if (m_wifiPhy->m_currentEvent) //Otherwise abort has already been called just before
    {
      for (auto & endMpduEvent : m_endOfMpduEvents)
        {
          endMpduEvent.Cancel ();
        }
      m_endOfMpduEvents.clear ();
    }
}

void
PhyEntity::ResetReceive (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  DoResetReceive (event);
  NS_ASSERT (m_endRxPayloadEvents.size () == 1 && m_endRxPayloadEvents.front ().IsExpired ());
  m_endRxPayloadEvents.clear ();
  m_wifiPhy->ResetReceive (event);
}

void
PhyEntity::DoResetReceive (Ptr<Event> event)
{
  NS_LOG_FUNCTION (this << *event);
  NS_ASSERT (event->GetEndTime () == Simulator::Now ());
}

double
PhyEntity::GetRandomValue (void) const
{
  return m_wifiPhy->m_random->GetValue ();
}

double
PhyEntity::GetRxPowerWForPpdu (Ptr<Event> event) const
{
  return event->GetRxPowerW (m_wifiPhy->GetPrimaryBand (GetMeasurementChannelWidth (event->GetPpdu ())));
}

Ptr<const Event>
PhyEntity::GetCurrentEvent (void) const
{
  return m_wifiPhy->m_currentEvent;
}

uint16_t
PhyEntity::GetMeasurementChannelWidth (const Ptr<const WifiPpdu> ppdu) const
{
  return GetRxChannelWidth (ppdu->GetTxVector ());
}

uint16_t
PhyEntity::GetRxChannelWidth (const WifiTxVector& txVector) const
{
  return std::min (m_wifiPhy->GetChannelWidth (), txVector.GetChannelWidth ());
}

uint64_t
PhyEntity::ObtainNextUid (const WifiTxVector& /* txVector */)
{
  NS_LOG_FUNCTION (this);
  return m_globalPpduUid++;
}

uint16_t
PhyEntity::GetCenterFrequencyForChannelWidth (const WifiTxVector& txVector) const
{
  NS_LOG_FUNCTION (this << txVector);

  return m_wifiPhy->GetOperatingChannel ().GetPrimaryChannelCenterFrequency (txVector.GetChannelWidth ());
}

void
PhyEntity::NotifyPayloadBegin (const WifiTxVector& txVector, const Time& payloadDuration)
{
  m_wifiPhy->m_phyRxPayloadBeginTrace (txVector, payloadDuration);
}

void
PhyEntity::StartTx (Ptr<WifiPpdu> ppdu)
{
  NS_LOG_FUNCTION (this << ppdu);
  Transmit (ppdu->GetTxDuration (), ppdu, "transmission");
}

void
PhyEntity::Transmit (Time txDuration, Ptr<WifiPpdu> ppdu, std::string type)
{
  NS_LOG_FUNCTION (this << txDuration << ppdu << type);
  double txPowerWatts = DbmToW (m_wifiPhy->GetTxPowerForTransmission (ppdu) + m_wifiPhy->GetTxGain ());
  NS_LOG_DEBUG ("Start " << type << ": signal power before antenna gain=" << WToDbm (txPowerWatts) << "dBm");
  Ptr<SpectrumValue> txPowerSpectrum = GetTxPowerSpectralDensity (txPowerWatts, ppdu);
  Ptr<WifiSpectrumSignalParameters> txParams = Create<WifiSpectrumSignalParameters> ();
  txParams->duration = txDuration;
  txParams->psd = txPowerSpectrum;
  txParams->ppdu = ppdu;
  txParams->txCenterFreq = GetCenterFrequencyForChannelWidth (ppdu->GetTxVector ());
  NS_LOG_DEBUG ("Starting " << type << " with power " << WToDbm (txPowerWatts) << " dBm on channel " << +m_wifiPhy->GetChannelNumber () << " for " << txParams->duration.As (Time::MS));
  NS_LOG_DEBUG ("Starting " << type << " with integrated spectrum power " << WToDbm (Integral (*txPowerSpectrum)) << " dBm; spectrum model Uid: " << txPowerSpectrum->GetSpectrumModel ()->GetUid ());
  auto spectrumWifiPhy = DynamicCast<SpectrumWifiPhy> (m_wifiPhy);
  NS_ASSERT (spectrumWifiPhy);
  spectrumWifiPhy->Transmit (txParams);
}

uint16_t
PhyEntity::GetGuardBandwidth (uint16_t currentChannelWidth) const
{
  return m_wifiPhy->GetGuardBandwidth (currentChannelWidth);
}

std::tuple<double, double, double>
PhyEntity::GetTxMaskRejectionParams (void) const
{
  return m_wifiPhy->GetTxMaskRejectionParams ();
}

Time
PhyEntity::CalculateTxDuration (WifiConstPsduMap psduMap, const WifiTxVector& txVector, WifiPhyBand band) const
{
  NS_ASSERT (psduMap.size () == 1);
  const auto & it = psduMap.begin ();
  return WifiPhy::CalculateTxDuration (it->second->GetSize (), txVector, band, it->first);
}

} //namespace ns3
