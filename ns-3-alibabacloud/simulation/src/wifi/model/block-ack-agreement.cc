/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Mirko Banchi <mk.banchi@gmail.com>
 */

#include "ns3/log.h"
#include "block-ack-agreement.h"
#include "wifi-utils.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BlockAckAgreement");

BlockAckAgreement::BlockAckAgreement (Mac48Address peer, uint8_t tid)
  : m_peer (peer),
    m_amsduSupported (0),
    m_blockAckPolicy (1),
    m_tid (tid),
    m_htSupported (0),
    m_inactivityEvent ()
{
  NS_LOG_FUNCTION (this << peer << +tid);
}

BlockAckAgreement::~BlockAckAgreement ()
{
  NS_LOG_FUNCTION (this);
  m_inactivityEvent.Cancel ();
}

void
BlockAckAgreement::SetBufferSize (uint16_t bufferSize)
{
  NS_LOG_FUNCTION (this << bufferSize);
  NS_ASSERT (bufferSize <= 256);
  NS_ASSERT (bufferSize % 16 == 0);
  m_bufferSize = bufferSize;
}

void
BlockAckAgreement::SetTimeout (uint16_t timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  m_timeout = timeout;
}

void
BlockAckAgreement::SetStartingSequence (uint16_t seq)
{
  NS_LOG_FUNCTION (this << seq);
  NS_ASSERT (seq < 4096);
  m_startingSeq = seq;
}

void
BlockAckAgreement::SetStartingSequenceControl (uint16_t seq)
{
  NS_LOG_FUNCTION (this << seq);
  NS_ASSERT (((seq >> 4) & 0x0fff) < 4096);
  m_startingSeq = (seq >> 4) & 0x0fff;
}

void
BlockAckAgreement::SetImmediateBlockAck (void)
{
  NS_LOG_FUNCTION (this);
  m_blockAckPolicy = 1;
}

void
BlockAckAgreement::SetDelayedBlockAck (void)
{
  NS_LOG_FUNCTION (this);
  m_blockAckPolicy = 0;
}

void
BlockAckAgreement::SetAmsduSupport (bool supported)
{
  NS_LOG_FUNCTION (this << supported);
  m_amsduSupported = supported;
}

uint8_t
BlockAckAgreement::GetTid (void) const
{
  return m_tid;
}

Mac48Address
BlockAckAgreement::GetPeer (void) const
{
  NS_LOG_FUNCTION (this);
  return m_peer;
}

uint16_t
BlockAckAgreement::GetBufferSize (void) const
{
  return m_bufferSize;
}

uint16_t
BlockAckAgreement::GetTimeout (void) const
{
  return m_timeout;
}

uint16_t
BlockAckAgreement::GetStartingSequence (void) const
{
  return m_startingSeq;
}

uint16_t
BlockAckAgreement::GetStartingSequenceControl (void) const
{
  uint16_t seqControl = (m_startingSeq << 4) & 0xfff0;
  return seqControl;
}

bool
BlockAckAgreement::IsImmediateBlockAck (void) const
{
  return m_blockAckPolicy == 1;
}

bool
BlockAckAgreement::IsAmsduSupported (void) const
{
  return m_amsduSupported == 1;
}

uint16_t
BlockAckAgreement::GetWinEnd (void) const
{
  return (GetStartingSequence () + GetBufferSize () - 1) % SEQNO_SPACE_SIZE;
}

void
BlockAckAgreement::SetHtSupported (bool htSupported)
{
  NS_LOG_FUNCTION (this << htSupported);
  m_htSupported = htSupported;
}

bool
BlockAckAgreement::IsHtSupported (void) const
{
  return m_htSupported == 1;
}

BlockAckType
BlockAckAgreement::GetBlockAckType (void) const
{
  if (!m_htSupported)
    {
      return BlockAckType::BASIC;
    }
  // Multi-TID Block Ack is not currently supported
  if (m_bufferSize > 64)
    {
      return {BlockAckType::COMPRESSED, {32}};
    }
  return {BlockAckType::COMPRESSED, {8}};
}

BlockAckReqType
BlockAckAgreement::GetBlockAckReqType (void) const
{
  if (!m_htSupported)
    {
      return BlockAckReqType::BASIC;
    }
  // Multi-TID Block Ack Request is not currently supported
  return BlockAckReqType::COMPRESSED;
}

std::size_t
BlockAckAgreement::GetDistance (uint16_t seqNumber, uint16_t startingSeqNumber)
{
  NS_ASSERT (seqNumber < SEQNO_SPACE_SIZE && startingSeqNumber < SEQNO_SPACE_SIZE);
  return (seqNumber - startingSeqNumber + SEQNO_SPACE_SIZE) % SEQNO_SPACE_SIZE;
}

} //namespace ns3
