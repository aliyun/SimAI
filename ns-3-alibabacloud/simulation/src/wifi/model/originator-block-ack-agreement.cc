/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009, 2010 MIRKO BANCHI
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
 * Authors: Mirko Banchi <mk.banchi@gmail.com>
 *          Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/log.h"
#include "originator-block-ack-agreement.h"
#include "wifi-mac-queue-item.h"
#include "wifi-utils.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OriginatorBlockAckAgreement");

OriginatorBlockAckAgreement::OriginatorBlockAckAgreement (Mac48Address recipient, uint8_t tid)
  : BlockAckAgreement (recipient, tid),
    m_state (PENDING)
{
}

OriginatorBlockAckAgreement::~OriginatorBlockAckAgreement ()
{
}

void
OriginatorBlockAckAgreement::SetState (State state)
{
  m_state = state;
}

bool
OriginatorBlockAckAgreement::IsPending (void) const
{
  return m_state == PENDING;
}

bool
OriginatorBlockAckAgreement::IsEstablished (void) const
{
  return m_state == ESTABLISHED;
}

bool
OriginatorBlockAckAgreement::IsRejected (void) const
{
  return m_state == REJECTED;
}

bool
OriginatorBlockAckAgreement::IsNoReply (void) const
{
  return m_state == NO_REPLY;
}

bool
OriginatorBlockAckAgreement::IsReset (void) const
{
  return m_state == RESET;
}

uint16_t
OriginatorBlockAckAgreement::GetStartingSequence (void) const
{
  if (m_txWindow.GetWinSize () == 0)
    {
      // the TX window has not been initialized yet
      return m_startingSeq;
    }
  return m_txWindow.GetWinStart ();
}

std::size_t
OriginatorBlockAckAgreement::GetDistance (uint16_t seqNumber) const
{
  return BlockAckAgreement::GetDistance (seqNumber, m_txWindow.GetWinStart ());
}

void
OriginatorBlockAckAgreement::InitTxWindow (void)
{
  m_txWindow.Init (m_startingSeq, m_bufferSize);
}

void
OriginatorBlockAckAgreement::AdvanceTxWindow (void)
{
  while (m_txWindow.At (0))
    {
      m_txWindow.Advance (1); // reset the current head -- ensures loop termination
    }
}

void
OriginatorBlockAckAgreement::NotifyTransmittedMpdu (Ptr<const WifiMacQueueItem> mpdu)
{
  uint16_t mpduSeqNumber = mpdu->GetHeader ().GetSequenceNumber ();
  uint16_t distance = GetDistance (mpduSeqNumber);

  if (distance >= SEQNO_SPACE_HALF_SIZE)
    {
      NS_LOG_DEBUG ("Transmitted an old MPDU, do nothing.");
      return;
    }

  // advance the transmit window if an MPDU beyond the current transmit window
  // is transmitted (see Section 10.24.7.7 of 802.11-2016)
  if (distance >= m_txWindow.GetWinSize ())
    {
      std::size_t count = distance - m_txWindow.GetWinSize () + 1;
      m_txWindow.Advance (count);
      // transmit window may advance further
      AdvanceTxWindow ();
      NS_LOG_DEBUG ("Transmitted MPDU beyond current transmit window. New starting sequence number: "
                    << m_txWindow.GetWinStart ());
    }
}

void
OriginatorBlockAckAgreement::NotifyAckedMpdu (Ptr<const WifiMacQueueItem> mpdu)
{
  uint16_t mpduSeqNumber = mpdu->GetHeader ().GetSequenceNumber ();
  uint16_t distance = GetDistance (mpduSeqNumber);

  if (distance >= SEQNO_SPACE_HALF_SIZE)
    {
      NS_LOG_DEBUG ("Acked an old MPDU, do nothing.");
      return;
    }

  // when an MPDU is transmitted, the transmit window is updated such that the
  // transmitted MPDU is in the window, hence we cannot be notified of the
  // acknowledgment of an MPDU which is beyond the transmit window
  m_txWindow.At (distance) = true;

  // the starting sequence number can be advanced to the sequence number of
  // the nearest unacknowledged MPDU
  AdvanceTxWindow ();
  NS_LOG_DEBUG ("Starting sequence number: " << m_txWindow.GetWinStart ());
}

void
OriginatorBlockAckAgreement::NotifyDiscardedMpdu (Ptr<const WifiMacQueueItem> mpdu)
{
  uint16_t mpduSeqNumber = mpdu->GetHeader ().GetSequenceNumber ();
  uint16_t distance = GetDistance (mpduSeqNumber);

  if (distance >= SEQNO_SPACE_HALF_SIZE)
    {
      NS_LOG_DEBUG ("Discarded an old MPDU, do nothing.");
      return;
    }

  m_txWindow.Advance (distance + 1);
  // transmit window may advance further
  AdvanceTxWindow ();
  NS_LOG_DEBUG ("Discarded MPDU within current transmit window. New starting sequence number: "
                << m_txWindow.GetWinStart ());
}

} //namespace ns3
