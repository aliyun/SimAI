/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include <numeric>
#include <algorithm>
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "interference-helper.h"
#include "wifi-phy.h"
#include "error-rate-model.h"
#include "wifi-utils.h"
#include "wifi-psdu.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("InterferenceHelper");

NS_OBJECT_ENSURE_REGISTERED (InterferenceHelper);

/****************************************************************
 *       PHY event class
 ****************************************************************/

Event::Event (Ptr<const WifiPpdu> ppdu, const WifiTxVector& txVector, Time duration, RxPowerWattPerChannelBand&& rxPower)
  : m_ppdu (ppdu),
    m_txVector (txVector),
    m_startTime (Simulator::Now ()),
    m_endTime (m_startTime + duration),
    m_rxPowerW (std::move (rxPower))
{
}

Event::~Event ()
{
  m_ppdu = 0;
  m_rxPowerW.clear ();
}

Ptr<const WifiPpdu>
Event::GetPpdu (void) const
{
  return m_ppdu;
}

Time
Event::GetStartTime (void) const
{
  return m_startTime;
}

Time
Event::GetEndTime (void) const
{
  return m_endTime;
}

Time
Event::GetDuration (void) const
{
  return m_endTime - m_startTime;
}

double
Event::GetRxPowerW (void) const
{
  NS_ASSERT (m_rxPowerW.size () > 0);
  //The total RX power corresponds to the maximum over all the bands
  auto it = std::max_element (m_rxPowerW.begin (), m_rxPowerW.end (),
    [] (const std::pair<WifiSpectrumBand, double>& p1, const std::pair<WifiSpectrumBand, double>& p2) {
      return p1.second < p2.second;
    });
  return it->second;
}

double
Event::GetRxPowerW (WifiSpectrumBand band) const
{
  auto it = m_rxPowerW.find (band);
  NS_ASSERT (it != m_rxPowerW.end ());
  return it->second;
}

const RxPowerWattPerChannelBand&
Event::GetRxPowerWPerBand (void) const
{
  return m_rxPowerW;
}

const WifiTxVector&
Event::GetTxVector (void) const
{
  return m_txVector;
}

void
Event::UpdateRxPowerW (const RxPowerWattPerChannelBand& rxPower)
{
  NS_ASSERT (rxPower.size () == m_rxPowerW.size ());
  //Update power band per band
  for (auto & currentRxPowerW : m_rxPowerW)
    {
      auto band = currentRxPowerW.first;
      auto it = rxPower.find (band);
      if (it != rxPower.end ())
        {
          currentRxPowerW.second += it->second;
        }
    }
}

std::ostream & operator << (std::ostream &os, const Event &event)
{
  os << "start=" << event.GetStartTime () << ", end=" << event.GetEndTime ()
     << ", TXVECTOR=" << event.GetTxVector ()
     << ", power=" << event.GetRxPowerW () << "W"
     << ", PPDU=" << event.GetPpdu ();
  return os;
}

/****************************************************************
 *       Class which records SNIR change events for a
 *       short period of time.
 ****************************************************************/

InterferenceHelper::NiChange::NiChange (double power, Ptr<Event> event)
  : m_power (power),
    m_event (event)
{
}

InterferenceHelper::NiChange::~NiChange ()
{
  m_event = 0;
}

double
InterferenceHelper::NiChange::GetPower (void) const
{
  return m_power;
}

void
InterferenceHelper::NiChange::AddPower (double power)
{
  m_power += power;
}

Ptr<Event>
InterferenceHelper::NiChange::GetEvent (void) const
{
  return m_event;
}


/****************************************************************
 *       The actual InterferenceHelper
 ****************************************************************/

InterferenceHelper::InterferenceHelper ()
  : m_errorRateModel (0),
    m_numRxAntennas (1),
    m_rxing (false)
{
  NS_LOG_FUNCTION (this);
}

InterferenceHelper::~InterferenceHelper ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
InterferenceHelper::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::InterferenceHelper")
    .SetParent<ns3::Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<InterferenceHelper> ()
  ;
  return tid;
}

void
InterferenceHelper::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  RemoveBands ();
  m_errorRateModel = 0;
}

Ptr<Event>
InterferenceHelper::Add (Ptr<const WifiPpdu> ppdu, const WifiTxVector& txVector, Time duration, RxPowerWattPerChannelBand& rxPowerW, bool isStartOfdmaRxing)
{
  Ptr<Event> event = Create<Event> (ppdu, txVector, duration, std::move (rxPowerW));
  AppendEvent (event, isStartOfdmaRxing);
  return event;
}

void
InterferenceHelper::AddForeignSignal (Time duration, RxPowerWattPerChannelBand& rxPowerW)
{
  // Parameters other than duration and rxPowerW are unused for this type
  // of signal, so we provide dummy versions
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_QOSDATA);
  hdr.SetQosTid (0);
  Ptr<WifiPpdu> fakePpdu = Create<WifiPpdu> (Create<WifiPsdu> (Create<Packet> (0), hdr),
                                             WifiTxVector ());
  Add (fakePpdu, WifiTxVector (), duration, rxPowerW);
}

void
InterferenceHelper::RemoveBands(void)
{
  NS_LOG_FUNCTION (this);
  for (auto it : m_niChangesPerBand)
    {
      it.second.clear ();
    }
  m_niChangesPerBand.clear();
  m_firstPowerPerBand.clear();
}

void
InterferenceHelper::AddBand (WifiSpectrumBand band)
{
  NS_LOG_FUNCTION (this << band.first << band.second);
  NS_ASSERT (m_niChangesPerBand.find (band) == m_niChangesPerBand.end ());
  NiChanges niChanges;
  auto result = m_niChangesPerBand.insert ({band, niChanges});
  NS_ASSERT (result.second);
  // Always have a zero power noise event in the list
  AddNiChangeEvent (Time (0), NiChange (0.0, 0), result.first);
  m_firstPowerPerBand.insert ({band, 0.0});
}

void
InterferenceHelper::SetNoiseFigure (double value)
{
  m_noiseFigure = value;
}

void
InterferenceHelper::SetErrorRateModel (const Ptr<ErrorRateModel> rate)
{
  m_errorRateModel = rate;
}

Ptr<ErrorRateModel>
InterferenceHelper::GetErrorRateModel (void) const
{
  return m_errorRateModel;
}

void
InterferenceHelper::SetNumberOfReceiveAntennas (uint8_t rx)
{
  m_numRxAntennas = rx;
}

Time
InterferenceHelper::GetEnergyDuration (double energyW, WifiSpectrumBand band)
{
  Time now = Simulator::Now ();
  auto niIt = m_niChangesPerBand.find (band);
  NS_ASSERT (niIt != m_niChangesPerBand.end ());
  auto i = GetPreviousPosition (now, niIt);
  Time end = i->first;
  for (; i != niIt->second.end (); ++i)
    {
      double noiseInterferenceW = i->second.GetPower ();
      end = i->first;
      if (noiseInterferenceW < energyW)
        {
          break;
        }
    }
  return end > now ? end - now : MicroSeconds (0);
}

void
InterferenceHelper::AppendEvent (Ptr<Event> event, bool isStartOfdmaRxing)
{
  NS_LOG_FUNCTION (this << event << isStartOfdmaRxing);
  for (auto const& it : event->GetRxPowerWPerBand ())
    {
      WifiSpectrumBand band = it.first;
      auto niIt = m_niChangesPerBand.find (band);
      NS_ASSERT (niIt != m_niChangesPerBand.end ());
      double previousPowerStart = 0;
      double previousPowerEnd = 0;
      auto previousPowerPosition = GetPreviousPosition (event->GetStartTime (), niIt);
      previousPowerStart = previousPowerPosition->second.GetPower ();
      previousPowerEnd = GetPreviousPosition (event->GetEndTime (), niIt)->second.GetPower ();
      if (!m_rxing)
        {
          m_firstPowerPerBand.find (band)->second = previousPowerStart;
          // Always leave the first zero power noise event in the list
          niIt->second.erase (++(niIt->second.begin ()), ++previousPowerPosition);
        }
      else if (isStartOfdmaRxing)
        {
          //When the first UL-OFDMA payload is received, we need to set m_firstPowerPerBand
          //so that it takes into account interferences that arrived between the start of the
          //UL MU transmission and the start of UL-OFDMA payload.
          m_firstPowerPerBand.find (band)->second = previousPowerStart;
        }
      auto first = AddNiChangeEvent (event->GetStartTime (), NiChange (previousPowerStart, event), niIt);
      auto last = AddNiChangeEvent (event->GetEndTime (), NiChange (previousPowerEnd, event), niIt);
      for (auto i = first; i != last; ++i)
        {
          i->second.AddPower (it.second);
        }
    }
}

void
InterferenceHelper::UpdateEvent (Ptr<Event> event, const RxPowerWattPerChannelBand& rxPower)
{
  NS_LOG_FUNCTION (this << event);
  //This is called for UL MU events, in order to scale power as long as UL MU PPDUs arrive
  for (auto const& it : rxPower)
    {
      WifiSpectrumBand band = it.first;
      auto niIt = m_niChangesPerBand.find (band);
      NS_ASSERT (niIt != m_niChangesPerBand.end ());
      auto first = GetPreviousPosition (event->GetStartTime (), niIt);
      auto last = GetPreviousPosition (event->GetEndTime (), niIt);
      for (auto i = first; i != last; ++i)
        {
          i->second.AddPower (it.second);
        }
    }
    event->UpdateRxPowerW (rxPower);
}

double
InterferenceHelper::CalculateSnr (double signal, double noiseInterference, uint16_t channelWidth, uint8_t nss) const
{
  NS_LOG_FUNCTION (this << signal << noiseInterference << channelWidth << +nss);
  //thermal noise at 290K in J/s = W
  static const double BOLTZMANN = 1.3803e-23;
  //Nt is the power of thermal noise in W
  double Nt = BOLTZMANN * 290 * channelWidth * 1e6;
  //receiver noise Floor (W) which accounts for thermal noise and non-idealities of the receiver
  double noiseFloor = m_noiseFigure * Nt;
  double noise = noiseFloor + noiseInterference;
  double snr = signal / noise; //linear scale
  NS_LOG_DEBUG ("bandwidth(MHz)=" << channelWidth << ", signal(W)= " << signal << ", noise(W)=" << noiseFloor << ", interference(W)=" << noiseInterference << ", snr=" << RatioToDb(snr) << "dB");
  if (m_errorRateModel->IsAwgn ())
    {
      double gain = 1;
      if (m_numRxAntennas > nss)
        {
          gain = static_cast<double> (m_numRxAntennas) / nss; //compute gain offered by diversity for AWGN
        }
      NS_LOG_DEBUG ("SNR improvement thanks to diversity: " << 10 * std::log10 (gain) << "dB");
      snr *= gain;
    }
  return snr;
}

double
InterferenceHelper::CalculateNoiseInterferenceW (Ptr<Event> event, NiChangesPerBand *nis, WifiSpectrumBand band) const
{
  NS_LOG_FUNCTION (this << band.first << band.second);
  auto firstPower_it = m_firstPowerPerBand.find (band);
  NS_ASSERT (firstPower_it != m_firstPowerPerBand.end ());
  double noiseInterferenceW = firstPower_it->second;
  auto niIt = m_niChangesPerBand.find (band);
  NS_ASSERT (niIt != m_niChangesPerBand.end ());
  auto it = niIt->second.find (event->GetStartTime ());
  for (; it != niIt->second.end () && it->first < Simulator::Now (); ++it)
    {
      noiseInterferenceW = it->second.GetPower () - event->GetRxPowerW (band);
    }
  it = niIt->second.find (event->GetStartTime ());
  NS_ASSERT (it != niIt->second.end ());
  for (; it != niIt->second.end () && it->second.GetEvent () != event; ++it);
  NiChanges ni;
  ni.emplace (event->GetStartTime (), NiChange (0, event));
  while (++it != niIt->second.end () && it->second.GetEvent () != event)
    {
      ni.insert (*it);
    }
  ni.emplace (event->GetEndTime (), NiChange (0, event));
  nis->insert ({band, ni});
  NS_ASSERT_MSG (noiseInterferenceW >= 0, "CalculateNoiseInterferenceW returns negative value " << noiseInterferenceW);
  return noiseInterferenceW;
}

double
InterferenceHelper::CalculateChunkSuccessRate (double snir, Time duration, WifiMode mode, const WifiTxVector& txVector, WifiPpduField field) const
{
  if (duration.IsZero ())
    {
      return 1.0;
    }
  uint64_t rate = mode.GetDataRate (txVector.GetChannelWidth ());
  uint64_t nbits = static_cast<uint64_t> (rate * duration.GetSeconds ());
  double csr = m_errorRateModel->GetChunkSuccessRate (mode, txVector, snir, nbits, m_numRxAntennas, field);
  return csr;
}

double
InterferenceHelper::CalculatePayloadChunkSuccessRate (double snir, Time duration, const WifiTxVector& txVector, uint16_t staId) const
{
  if (duration.IsZero ())
    {
      return 1.0;
    }
  WifiMode mode = txVector.GetMode (staId);
  uint64_t rate = mode.GetDataRate (txVector, staId);
  uint64_t nbits = static_cast<uint64_t> (rate * duration.GetSeconds ());
  nbits /= txVector.GetNss (staId); //divide effective number of bits by NSS to achieve same chunk error rate as SISO for AWGN
  double csr = m_errorRateModel->GetChunkSuccessRate (mode, txVector, snir, nbits, m_numRxAntennas, WIFI_PPDU_FIELD_DATA, staId);
  return csr;
}

double
InterferenceHelper::CalculatePayloadPer (Ptr<const Event> event, uint16_t channelWidth,
                                         NiChangesPerBand *nis, WifiSpectrumBand band,
                                         uint16_t staId, std::pair<Time, Time> window) const
{
  NS_LOG_FUNCTION (this << channelWidth << band.first << band.second << staId << window.first << window.second);
  double psr = 1.0; /* Packet Success Rate */
  const auto& niIt = nis->find (band)->second;
  auto j = niIt.cbegin ();
  Time previous = j->first;
  WifiMode payloadMode = event->GetTxVector ().GetMode (staId);
  Time phyPayloadStart = j->first;
  if (event->GetPpdu ()->GetType () != WIFI_PPDU_TYPE_UL_MU) //j->first corresponds to the start of the UL-OFDMA payload
    {
      phyPayloadStart = j->first + WifiPhy::CalculatePhyPreambleAndHeaderDuration (event->GetTxVector ());
    }
  Time windowStart = phyPayloadStart + window.first;
  Time windowEnd = phyPayloadStart + window.second;
  double noiseInterferenceW = m_firstPowerPerBand.find (band)->second;
  double powerW = event->GetRxPowerW (band);
  while (++j != niIt.cend ())
    {
      Time current = j->first;
      NS_LOG_DEBUG ("previous= " << previous << ", current=" << current);
      NS_ASSERT (current >= previous);
      double snr = CalculateSnr (powerW, noiseInterferenceW, channelWidth, event->GetTxVector ().GetNss (staId));
      //Case 1: Both previous and current point to the windowed payload
      if (previous >= windowStart)
        {
          psr *= CalculatePayloadChunkSuccessRate (snr, Min (windowEnd, current) - previous, event->GetTxVector (), staId);
          NS_LOG_DEBUG ("Both previous and current point to the windowed payload: mode=" << payloadMode << ", psr=" << psr);
        }
      //Case 2: previous is before windowed payload and current is in the windowed payload
      else if (current >= windowStart)
        {
          psr *= CalculatePayloadChunkSuccessRate (snr, Min (windowEnd, current) - windowStart, event->GetTxVector (), staId);
          NS_LOG_DEBUG ("previous is before windowed payload and current is in the windowed payload: mode=" << payloadMode << ", psr=" << psr);
        }
      noiseInterferenceW = j->second.GetPower () - powerW;
      previous = j->first;
      if (previous > windowEnd)
        {
          NS_LOG_DEBUG ("Stop: new previous=" << previous << " after time window end=" << windowEnd);
          break;
        }
    }
  double per = 1 - psr;
  return per;
}

double
InterferenceHelper::CalculatePhyHeaderSectionPsr (Ptr<const Event> event, NiChangesPerBand *nis,
                                                  uint16_t channelWidth, WifiSpectrumBand band,
                                                  PhyEntity::PhyHeaderSections phyHeaderSections) const
{
  NS_LOG_FUNCTION (this << band.first << band.second);
  double psr = 1.0; /* Packet Success Rate */
  auto niIt = nis->find (band)->second;
  auto j = niIt.begin ();

  NS_ASSERT (!phyHeaderSections.empty ());
  Time stopLastSection = Seconds (0);
  for (const auto & section : phyHeaderSections)
    {
      stopLastSection = Max (stopLastSection, section.second.first.second);
    }

  Time previous = j->first;
  double noiseInterferenceW = m_firstPowerPerBand.find (band)->second;
  double powerW = event->GetRxPowerW (band);
  while (++j != niIt.end ())
    {
      Time current = j->first;
      NS_LOG_DEBUG ("previous= " << previous << ", current=" << current);
      NS_ASSERT (current >= previous);
      double snr = CalculateSnr (powerW, noiseInterferenceW, channelWidth, 1);
      for (const auto & section : phyHeaderSections)
        {
          Time start = section.second.first.first;
          Time stop = section.second.first.second;

          if (previous <= stop || current >= start)
            {
              Time duration = Min (stop, current) - Max (start, previous);
              if (duration.IsStrictlyPositive ())
                {
                  psr *= CalculateChunkSuccessRate (snr, duration, section.second.second, event->GetTxVector (), section.first);
                  NS_LOG_DEBUG ("Current NI change in " << section.first << " [" << start << ", " << stop << "] for "
                                << duration.As (Time::NS) << ": mode=" << section.second.second << ", psr=" << psr);
                }
            }
        }
      noiseInterferenceW = j->second.GetPower () - powerW;
      previous = j->first;
      if (previous > stopLastSection)
        {
          NS_LOG_DEBUG ("Stop: new previous=" << previous << " after stop of last section=" << stopLastSection);
          break;
        }
    }
  return psr;
}

double
InterferenceHelper::CalculatePhyHeaderPer (Ptr<const Event> event, NiChangesPerBand *nis,
                                           uint16_t channelWidth, WifiSpectrumBand band,
                                           WifiPpduField header) const
{
  NS_LOG_FUNCTION (this << band.first << band.second << header);
  auto niIt = nis->find (band)->second;
  auto phyEntity = WifiPhy::GetStaticPhyEntity (event->GetTxVector ().GetModulationClass ());

  PhyEntity::PhyHeaderSections sections;
  for (const auto & section : phyEntity->GetPhyHeaderSections (event->GetTxVector (), niIt.begin ()->first))
    {
      if (section.first == header)
        {
          sections[header] = section.second;
        }
    }

  double psr = 1.0;
  if (!sections.empty () > 0)
    {
      psr = CalculatePhyHeaderSectionPsr (event, nis, channelWidth, band, sections);
    }
  return 1 - psr;
}

struct PhyEntity::SnrPer
InterferenceHelper::CalculatePayloadSnrPer (Ptr<Event> event, uint16_t channelWidth, WifiSpectrumBand band,
                                            uint16_t staId, std::pair<Time, Time> relativeMpduStartStop) const
{
  NS_LOG_FUNCTION (this << channelWidth << band.first << band.second << staId << relativeMpduStartStop.first << relativeMpduStartStop.second);
  NiChangesPerBand ni;
  double noiseInterferenceW = CalculateNoiseInterferenceW (event, &ni, band);
  double snr = CalculateSnr (event->GetRxPowerW (band),
                             noiseInterferenceW,
                             channelWidth,
                             event->GetTxVector ().GetNss (staId));

  /* calculate the SNIR at the start of the MPDU (located through windowing) and accumulate
   * all SNIR changes in the SNIR vector.
   */
  double per = CalculatePayloadPer (event, channelWidth, &ni, band, staId, relativeMpduStartStop);

  return PhyEntity::SnrPer (snr, per);
}

double
InterferenceHelper::CalculateSnr (Ptr<Event> event, uint16_t channelWidth, uint8_t nss, WifiSpectrumBand band) const
{
  NiChangesPerBand ni;
  double noiseInterferenceW = CalculateNoiseInterferenceW (event, &ni, band);
  double snr = CalculateSnr (event->GetRxPowerW (band),
                             noiseInterferenceW,
                             channelWidth,
                             nss);
  return snr;
}

struct PhyEntity::SnrPer
InterferenceHelper::CalculatePhyHeaderSnrPer (Ptr<Event> event, uint16_t channelWidth, WifiSpectrumBand band,
                                              WifiPpduField header) const
{
  NS_LOG_FUNCTION (this << band.first << band.second << header);
  NiChangesPerBand ni;
  double noiseInterferenceW = CalculateNoiseInterferenceW (event, &ni, band);
  double snr = CalculateSnr (event->GetRxPowerW (band),
                             noiseInterferenceW,
                             channelWidth,
                             1);
  
  /* calculate the SNIR at the start of the PHY header and accumulate
   * all SNIR changes in the SNIR vector.
   */
  double per = CalculatePhyHeaderPer (event, &ni, channelWidth, band, header);
  
  return PhyEntity::SnrPer (snr, per);
}

void
InterferenceHelper::EraseEvents (void)
{
  for (auto niIt = m_niChangesPerBand.begin(); niIt != m_niChangesPerBand.end(); ++niIt)
    {
      niIt->second.clear ();
      // Always have a zero power noise event in the list
      AddNiChangeEvent (Time (0), NiChange (0.0, 0), niIt);
      m_firstPowerPerBand.at (niIt->first) = 0.0;
    }
  m_rxing = false;
}

InterferenceHelper::NiChanges::iterator
InterferenceHelper::GetNextPosition (Time moment, NiChangesPerBand::iterator niIt)
{
  return niIt->second.upper_bound (moment);
}

InterferenceHelper::NiChanges::iterator
InterferenceHelper::GetPreviousPosition (Time moment, NiChangesPerBand::iterator niIt)
{
  auto it = GetNextPosition (moment, niIt);
  // This is safe since there is always an NiChange at time 0,
  // before moment.
  --it;
  return it;
}

InterferenceHelper::NiChanges::iterator
InterferenceHelper::AddNiChangeEvent (Time moment, NiChange change, NiChangesPerBand::iterator niIt)
{
  return niIt->second.insert (GetNextPosition (moment, niIt), std::make_pair (moment, change));
}

void
InterferenceHelper::NotifyRxStart ()
{
  NS_LOG_FUNCTION (this);
  m_rxing = true;
}

void
InterferenceHelper::NotifyRxEnd (Time endTime)
{
  NS_LOG_FUNCTION (this << endTime);
  m_rxing = false;
  //Update m_firstPowerPerBand for frame capture
  for (auto niIt = m_niChangesPerBand.begin(); niIt != m_niChangesPerBand.end(); ++niIt)
    {
      NS_ASSERT (niIt->second.size () > 1);
      auto it = GetPreviousPosition (endTime, niIt);
      it--;
      m_firstPowerPerBand.find (niIt->first)->second = it->second.GetPower ();
    }
}

} //namespace ns3
