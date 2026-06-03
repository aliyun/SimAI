/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Author:
 *  kwong yin <kwong-sang.yin@boeing.com>
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 *  Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */

#include "lr-wpan-csmaca.h"
#include <ns3/random-variable-stream.h>
#include <ns3/simulator.h>
#include <ns3/log.h>
#include <algorithm>

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT                                   \
  std::clog << "[address " << m_mac->GetShortAddress () << "] ";

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LrWpanCsmaCa");

NS_OBJECT_ENSURE_REGISTERED (LrWpanCsmaCa);

TypeId
LrWpanCsmaCa::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LrWpanCsmaCa")
    .SetParent<Object> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<LrWpanCsmaCa> ()
  ;
  return tid;
}

LrWpanCsmaCa::LrWpanCsmaCa ()
{
  // TODO-- make these into ns-3 attributes

  m_isSlotted = false;
  m_NB = 0;
  m_CW = 2;
  m_macBattLifeExt = false;
  m_macMinBE = 3;
  m_macMaxBE = 5;
  m_macMaxCSMABackoffs = 4;
  m_aUnitBackoffPeriod = 20; // symbols
  m_random = CreateObject<UniformRandomVariable> ();
  m_BE = m_macMinBE;
  m_ccaRequestRunning = false;
  m_randomBackoffPeriodsLeft = 0;
  m_coorDest = false;
}

LrWpanCsmaCa::~LrWpanCsmaCa ()
{
  m_mac = 0;
}

void
LrWpanCsmaCa::DoDispose ()
{
  m_lrWpanMacStateCallback = MakeNullCallback <void, LrWpanMacState> ();
  m_lrWpanMacTransCostCallback = MakeNullCallback <void, uint32_t> ();

  Cancel ();
  m_mac = 0;
}

void
LrWpanCsmaCa::SetMac (Ptr<LrWpanMac> mac)
{
  m_mac = mac;
}

Ptr<LrWpanMac>
LrWpanCsmaCa::GetMac (void) const
{
  return m_mac;
}

void
LrWpanCsmaCa::SetSlottedCsmaCa (void)
{
  NS_LOG_FUNCTION (this);
  m_isSlotted = true;
}

void
LrWpanCsmaCa::SetUnSlottedCsmaCa (void)
{
  NS_LOG_FUNCTION (this);
  m_isSlotted = false;
}

bool
LrWpanCsmaCa::IsSlottedCsmaCa (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_isSlotted);
}

bool
LrWpanCsmaCa::IsUnSlottedCsmaCa (void) const
{
  NS_LOG_FUNCTION (this);
  return (!m_isSlotted);
}

void
LrWpanCsmaCa::SetMacMinBE (uint8_t macMinBE)
{
  NS_LOG_FUNCTION (this << macMinBE);
  m_macMinBE = macMinBE;
}

uint8_t
LrWpanCsmaCa::GetMacMinBE (void) const
{
  NS_LOG_FUNCTION (this);
  return m_macMinBE;
}

void
LrWpanCsmaCa::SetMacMaxBE (uint8_t macMaxBE)
{
  NS_LOG_FUNCTION (this << macMaxBE);
  m_macMinBE = macMaxBE;
}

uint8_t
LrWpanCsmaCa::GetMacMaxBE (void) const
{
  NS_LOG_FUNCTION (this);
  return m_macMaxBE;
}

void
LrWpanCsmaCa::SetMacMaxCSMABackoffs (uint8_t macMaxCSMABackoffs)
{
  NS_LOG_FUNCTION (this << macMaxCSMABackoffs);
  m_macMaxCSMABackoffs = macMaxCSMABackoffs;
}

uint8_t
LrWpanCsmaCa::GetMacMaxCSMABackoffs (void) const
{
  NS_LOG_FUNCTION (this);
  return m_macMaxCSMABackoffs;
}

void
LrWpanCsmaCa::SetUnitBackoffPeriod (uint64_t unitBackoffPeriod)
{
  NS_LOG_FUNCTION (this << unitBackoffPeriod);
  m_aUnitBackoffPeriod = unitBackoffPeriod;
}

uint64_t
LrWpanCsmaCa::GetUnitBackoffPeriod (void) const
{
  NS_LOG_FUNCTION (this);
  return m_aUnitBackoffPeriod;
}


Time
LrWpanCsmaCa::GetTimeToNextSlot (void) const
{
  NS_LOG_FUNCTION (this);

  // The reference for the beginning of the SUPERFRAME (the active period) changes depending
  // on the data packet being sent from the Coordinator/outgoing frame (Tx beacon time reference)
  // or other device/incoming frame (Rx beacon time reference ).

  Time elapsedSuperframe;  // (i.e  The beacon + the elapsed CAP)
  Time currentTime;
  double symbolsToBoundary;
  Time nextBoundary;
  uint64_t elapsedSuperframeSymbols;
  uint64_t symbolRate;
  Time timeAtBoundary;
  [[maybe_unused]] Time elapsedCap;
  [[maybe_unused]] Time beaconTime;


  currentTime = Simulator::Now ();
  symbolRate = (uint64_t) m_mac->GetPhy ()->GetDataOrSymbolRate (false); //symbols per second

  if (m_coorDest)
    {
      // Take the Incoming Frame Reference
      elapsedSuperframe = currentTime - m_mac->m_macBeaconRxTime;

      beaconTime = Seconds ((double) m_mac->m_rxBeaconSymbols / symbolRate);
      elapsedCap = elapsedSuperframe - beaconTime;
      NS_LOG_DEBUG ("Elapsed incoming CAP symbols: " << (elapsedCap.GetSeconds () * symbolRate)  << " (" << elapsedCap.As (Time::S) << ")");
    }
  else
    {
      // Take the Outgoing Frame Reference
      elapsedSuperframe = currentTime - m_mac->m_macBeaconTxTime;
    }

  // get a close value to the the boundary in symbols
  elapsedSuperframeSymbols = elapsedSuperframe.GetSeconds () * symbolRate;
  symbolsToBoundary = m_aUnitBackoffPeriod - std::fmod ((double) elapsedSuperframeSymbols,m_aUnitBackoffPeriod);

  timeAtBoundary = Seconds ((double)(elapsedSuperframeSymbols + symbolsToBoundary) / symbolRate);

  // get the exact time boundary
  nextBoundary = timeAtBoundary - elapsedSuperframe;

  NS_LOG_DEBUG ("Elapsed Superframe symbols: " << elapsedSuperframeSymbols << " ("
                                               << elapsedSuperframe.As (Time::S) << ")");

  NS_LOG_DEBUG ("Next backoff period boundary in approx. " << nextBoundary.GetSeconds () * symbolRate << " symbols ("
                                                           << nextBoundary.As (Time::S) << ")");

  return nextBoundary;

}


void
LrWpanCsmaCa::Start ()
{
  NS_LOG_FUNCTION (this);
  m_NB = 0;
  if (IsSlottedCsmaCa ())
    {
      // TODO: Check if the current PHY is using the Japanese band 950 Mhz:
      //       (IEEE_802_15_4_950MHZ_BPSK and IEEE_802_15_4_950MHZ_2GFSK)
      //       if in use, m_CW = 1.
      //       Currently 950 Mhz band PHYs are not supported in ns-3.
      //       To know the current used PHY, making the method for GetPhy()->GetMyPhyOption()
      //       public is necessary. Alternatively, the current PHY used
      //       can be known using phyCurrentPage variable.

      m_CW = 2;

      if (m_macBattLifeExt)
        {
          m_BE = std::min (static_cast<uint8_t> (2), m_macMinBE);
        }
      else
        {
          m_BE = m_macMinBE;
        }

      // m_coorDest to decide between incoming and outgoing superframes times
      m_coorDest = m_mac->isCoordDest ();

      // Locate backoff period boundary. (i.e. a time delay to align with the next backoff period boundary)
      Time backoffBoundary = GetTimeToNextSlot ();
      m_randomBackoffEvent = Simulator::Schedule (backoffBoundary, &LrWpanCsmaCa::RandomBackoffDelay, this);

    }
  else
    {
      m_BE = m_macMinBE;
      m_randomBackoffEvent = Simulator::ScheduleNow (&LrWpanCsmaCa::RandomBackoffDelay, this);
    }
}

void
LrWpanCsmaCa::Cancel ()
{
  m_randomBackoffEvent.Cancel ();
  m_requestCcaEvent.Cancel ();
  m_canProceedEvent.Cancel ();
}



void
LrWpanCsmaCa::RandomBackoffDelay ()
{
  NS_LOG_FUNCTION (this);

  uint64_t upperBound = (uint64_t) pow (2, m_BE) - 1;
  Time randomBackoff;
  uint64_t symbolRate;
  Time timeLeftInCap;

  symbolRate = (uint64_t) m_mac->GetPhy ()->GetDataOrSymbolRate (false); //symbols per second

  // We should not recalculate the random backoffPeriods if we are in a slotted CSMA-CA and the
  // transmission was previously deferred (m_randomBackoffPeriods != 0)
  if (m_randomBackoffPeriodsLeft == 0 || IsUnSlottedCsmaCa ())
    {
      m_randomBackoffPeriodsLeft = (uint64_t)m_random->GetValue (0, upperBound + 1);
    }

  randomBackoff = Seconds ((double) (m_randomBackoffPeriodsLeft * GetUnitBackoffPeriod ()) / symbolRate);

  if (IsUnSlottedCsmaCa ())
    {
      NS_LOG_DEBUG ("Unslotted CSMA-CA: requesting CCA after backoff of " << m_randomBackoffPeriodsLeft <<
                    " periods (" << randomBackoff.As (Time::S) << ")");
      m_requestCcaEvent = Simulator::Schedule (randomBackoff, &LrWpanCsmaCa::RequestCCA, this);
    }
  else
    {
      // We must make sure there is enough time left in the CAP, otherwise we continue in
      // the CAP of the next superframe after the transmission/reception of the beacon (and the IFS)
      timeLeftInCap = GetTimeLeftInCap ();

      NS_LOG_DEBUG ("Slotted CSMA-CA: proceeding after random backoff of " << m_randomBackoffPeriodsLeft <<
                    " periods ("  << (randomBackoff.GetSeconds () * symbolRate) << " symbols or " << randomBackoff.As (Time::S) << ")");




      NS_LOG_DEBUG ("Backoff periods left in CAP: " << ((timeLeftInCap.GetSeconds () *  symbolRate) / m_aUnitBackoffPeriod) << " ("
                                                    << (timeLeftInCap.GetSeconds () *  symbolRate) << " symbols or "
                                                    << timeLeftInCap.As (Time::S) << ")");


      if (randomBackoff > timeLeftInCap)
        {
          uint64_t usedBackoffs = (double)(timeLeftInCap.GetSeconds () *  symbolRate) / m_aUnitBackoffPeriod;
          m_randomBackoffPeriodsLeft -= usedBackoffs;
          NS_LOG_DEBUG ("No time in CAP to complete backoff delay, deferring to the next CAP");
          m_endCapEvent = Simulator::Schedule (timeLeftInCap, &LrWpanCsmaCa::DeferCsmaTimeout, this);
        }
      else
        {
          m_canProceedEvent = Simulator::Schedule (randomBackoff, &LrWpanCsmaCa::CanProceed, this);
        }

    }
}


Time
LrWpanCsmaCa::GetTimeLeftInCap ()
{
  Time currentTime;
  uint64_t capSymbols;
  Time endCapTime;
  uint64_t activeSlot;
  uint64_t symbolRate;
  Time rxBeaconTime;


  // At this point, the currentTime should be aligned on a backoff period boundary
  currentTime  = Simulator::Now ();
  symbolRate = (uint64_t) m_mac->GetPhy ()->GetDataOrSymbolRate (false); //symbols per second


  if (m_coorDest)
    { // Take Incoming frame reference
      activeSlot = m_mac->m_incomingSuperframeDuration / 16;
      capSymbols = activeSlot * (m_mac->m_incomingFnlCapSlot + 1);
      endCapTime = m_mac->m_macBeaconRxTime +
        Seconds ((double) capSymbols / symbolRate);
    }
  else
    { // Take Outgoing frame reference
      activeSlot = m_mac->m_superframeDuration / 16;
      capSymbols = activeSlot * (m_mac->m_fnlCapSlot + 1);
      endCapTime = m_mac->m_macBeaconTxTime +
        Seconds ((double) capSymbols / symbolRate);
    }

  return (endCapTime - currentTime);
}


void
LrWpanCsmaCa::CanProceed ()
{
  NS_LOG_FUNCTION (this);

  Time timeLeftInCap;
  uint16_t ccaSymbols;
  uint32_t transactionSymbols;
  Time transactionTime;
  uint64_t symbolRate;

  ccaSymbols = 0;
  m_randomBackoffPeriodsLeft = 0;
  symbolRate = (uint64_t) m_mac->GetPhy ()->GetDataOrSymbolRate (false);
  timeLeftInCap = GetTimeLeftInCap ();


  // TODO: On the 950 Mhz Band (Japanese Band)
  //       only a single CCA check is performed;
  //       the CCA check duration time is:
  //
  //       CCA symbols = phyCCADuration * m_CW (1)
  //       other PHYs:
  //       CCA symbols = 8 * m_CW(2)
  //
  //       note: phyCCADuration & 950Mhz band PHYs are
  //             not currently implemented in ns-3.
  ccaSymbols += 8 * m_CW;

  // The MAC sublayer shall proceed if the remaining CSMA-CA algorithm steps
  // can be completed before the end of the CAP.
  // See IEEE 802.15.4-2011 (Sections 5.1.1.1 and 5.1.1.4)
  // Transaction = 2 CCA + frame transmission (SHR+PHR+PPDU) + turnaroudtime*2 (Rx->Tx & Tx->Rx) + IFS (LIFS or SIFS) and Ack time (if ack flag true)

  transactionSymbols = ccaSymbols + m_mac->GetTxPacketSymbols ();

  if (m_mac->isTxAckReq ())
    {
      NS_LOG_DEBUG ("ACK duration symbols: " <<  m_mac->GetMacAckWaitDuration ());
      transactionSymbols +=  m_mac->GetMacAckWaitDuration ();
    }
  else
    {
      //time the PHY takes to switch from Rx to Tx and Tx to Rx
      transactionSymbols += (m_mac->GetPhy ()->aTurnaroundTime *2);
    }
  transactionSymbols +=  m_mac->GetIfsSize ();

  // Report the transaction cost
  if (!m_lrWpanMacTransCostCallback.IsNull ())
    {
      m_lrWpanMacTransCostCallback (transactionSymbols);
    }

  transactionTime = Seconds ((double) transactionSymbols / symbolRate);
  NS_LOG_DEBUG ("Total required transaction: " << transactionSymbols << " symbols (" << transactionTime.As (Time::S) << ")");

  if (transactionTime > timeLeftInCap)
    {
      NS_LOG_DEBUG ("Transaction of " << transactionSymbols << " symbols " <<
                    "cannot be completed in CAP, deferring transmission to the next CAP");



      NS_LOG_DEBUG ("Symbols left in CAP: " << (timeLeftInCap.GetSeconds () * symbolRate) <<
                    " (" << timeLeftInCap.As (Time::S) << ")");

      m_endCapEvent = Simulator::Schedule (timeLeftInCap, &LrWpanCsmaCa::DeferCsmaTimeout, this);
    }
  else
    {
      m_requestCcaEvent = Simulator::ScheduleNow (&LrWpanCsmaCa::RequestCCA,this);
    }

}

void
LrWpanCsmaCa::RequestCCA ()
{
  NS_LOG_FUNCTION (this);
  m_ccaRequestRunning = true;
  m_mac->GetPhy ()->PlmeCcaRequest ();
}

void
LrWpanCsmaCa::DeferCsmaTimeout ()
{
  NS_LOG_FUNCTION (this);
  m_lrWpanMacStateCallback (MAC_CSMA_DEFERRED);
}

void
LrWpanCsmaCa::PlmeCcaConfirm (LrWpanPhyEnumeration status)
{
  NS_LOG_FUNCTION (this << status);

  // Only react on this event, if we are actually waiting for a CCA.
  // If the CSMA algorithm was canceled, we could still receive this event from
  // the PHY. In this case we ignore the event.
  if (m_ccaRequestRunning)
    {
      m_ccaRequestRunning = false;
      if (status == IEEE_802_15_4_PHY_IDLE)
        {
          if (IsSlottedCsmaCa ())
            {
              m_CW--;
              if (m_CW == 0)
                {
                  // inform MAC channel is idle
                  if (!m_lrWpanMacStateCallback.IsNull ())
                    {
                      NS_LOG_LOGIC ("Notifying MAC of idle channel");
                      m_lrWpanMacStateCallback (CHANNEL_IDLE);
                    }
                }
              else
                {
                  NS_LOG_LOGIC ("Perform CCA again, m_CW = " << m_CW);
                  m_requestCcaEvent = Simulator::ScheduleNow (&LrWpanCsmaCa::RequestCCA, this); // Perform CCA again
                }
            }
          else
            {
              // inform MAC, channel is idle
              if (!m_lrWpanMacStateCallback.IsNull ())
                {
                  NS_LOG_LOGIC ("Notifying MAC of idle channel");
                  m_lrWpanMacStateCallback (CHANNEL_IDLE);
                }
            }
        }
      else
        {
          if (IsSlottedCsmaCa ())
            {
              m_CW = 2;
            }
          m_BE = std::min (static_cast<uint16_t> (m_BE + 1), static_cast<uint16_t> (m_macMaxBE));
          m_NB++;
          if (m_NB > m_macMaxCSMABackoffs)
            {
              // no channel found so cannot send pkt
              NS_LOG_DEBUG ("Channel access failure");
              if (!m_lrWpanMacStateCallback.IsNull ())
                {
                  NS_LOG_LOGIC ("Notifying MAC of Channel access failure");
                  m_lrWpanMacStateCallback (CHANNEL_ACCESS_FAILURE);
                }
              return;
            }
          else
            {
              NS_LOG_DEBUG ("Perform another backoff; m_NB = " << static_cast<uint16_t> (m_NB));
              m_randomBackoffEvent = Simulator::ScheduleNow (&LrWpanCsmaCa::RandomBackoffDelay, this); //Perform another backoff (step 2)
            }
        }
    }
}


void
LrWpanCsmaCa::SetLrWpanMacTransCostCallback (LrWpanMacTransCostCallback c)
{
  NS_LOG_FUNCTION (this);
  m_lrWpanMacTransCostCallback = c;
}


void
LrWpanCsmaCa::SetLrWpanMacStateCallback (LrWpanMacStateCallback c)
{
  NS_LOG_FUNCTION (this);
  m_lrWpanMacStateCallback = c;
}

void
LrWpanCsmaCa::SetBatteryLifeExtension (bool batteryLifeExtension)
{
  m_macBattLifeExt = batteryLifeExtension;
}


int64_t
LrWpanCsmaCa::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this);
  m_random->SetStream (stream);
  return 1;
}

uint8_t
LrWpanCsmaCa::GetNB (void)
{
  return m_NB;
}

bool
LrWpanCsmaCa::GetBatteryLifeExtension (void)
{
  return m_macBattLifeExt;
}

} //namespace ns3
