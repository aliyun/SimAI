/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "wifi-acknowledgment.h"
#include "wifi-utils.h"
#include "ns3/mac48-address.h"

namespace ns3 {

/*
 * WifiAcknowledgment
 */

WifiAcknowledgment::WifiAcknowledgment (Method m)
  : method (m),
    acknowledgmentTime (Time::Min ())   // uninitialized
{
}

WifiAcknowledgment::~WifiAcknowledgment ()
{
}

WifiMacHeader::QosAckPolicy
WifiAcknowledgment::GetQosAckPolicy (Mac48Address receiver, uint8_t tid) const
{
  auto it = m_ackPolicy.find ({receiver, tid});
  NS_ASSERT (it != m_ackPolicy.end ());
  return it->second;
}

void
WifiAcknowledgment::SetQosAckPolicy (Mac48Address receiver, uint8_t tid,
                                     WifiMacHeader::QosAckPolicy ackPolicy)
{
  NS_ABORT_MSG_IF (!CheckQosAckPolicy (receiver, tid, ackPolicy), "QoS Ack policy not admitted");
  m_ackPolicy[{receiver, tid}] = ackPolicy;
}

/*
 * WifiNoAck
 */

WifiNoAck::WifiNoAck ()
  : WifiAcknowledgment (NONE)
{
  acknowledgmentTime = Seconds (0);
}

std::unique_ptr<WifiAcknowledgment>
WifiNoAck::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiNoAck (*this));
}

bool
WifiNoAck::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const
{
  if (ackPolicy == WifiMacHeader::NO_ACK || ackPolicy == WifiMacHeader::BLOCK_ACK)
    {
      return true;
    }
  return false;
}

void
WifiNoAck::Print (std::ostream &os) const
{
  os << "NONE";
}


/*
 * WifiNormalAck
 */

WifiNormalAck::WifiNormalAck ()
  : WifiAcknowledgment (NORMAL_ACK)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiNormalAck::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiNormalAck (*this));
}

bool
WifiNormalAck::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const
{
  if (ackPolicy == WifiMacHeader::NORMAL_ACK)
    {
      return true;
    }
  return false;
}

void
WifiNormalAck::Print (std::ostream &os) const
{
  os << "NORMAL_ACK";
}

/*
 * WifiBlockAck
 */

WifiBlockAck::WifiBlockAck ()
  : WifiAcknowledgment (BLOCK_ACK)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiBlockAck::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiBlockAck (*this));
}

bool
WifiBlockAck::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const
{
  if (ackPolicy == WifiMacHeader::NORMAL_ACK)
    {
      return true;
    }
  return false;
}

void
WifiBlockAck::Print (std::ostream &os) const
{
  os << "BLOCK_ACK";
}


/*
 * WifiBarBlockAck
 */

WifiBarBlockAck::WifiBarBlockAck ()
  : WifiAcknowledgment (BAR_BLOCK_ACK)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiBarBlockAck::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiBarBlockAck (*this));
}

bool
WifiBarBlockAck::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid, WifiMacHeader::QosAckPolicy ackPolicy) const
{
  if (ackPolicy == WifiMacHeader::BLOCK_ACK)
    {
      return true;
    }
  return false;
}

void
WifiBarBlockAck::Print (std::ostream &os) const
{
  os << "BAR_BLOCK_ACK";
}


/*
 * WifiDlMuBarBaSequence
 */

WifiDlMuBarBaSequence::WifiDlMuBarBaSequence ()
  : WifiAcknowledgment (DL_MU_BAR_BA_SEQUENCE)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiDlMuBarBaSequence::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiDlMuBarBaSequence (*this));
}

bool
WifiDlMuBarBaSequence::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid,
                                          WifiMacHeader::QosAckPolicy ackPolicy) const
{
  if (ackPolicy == WifiMacHeader::NORMAL_ACK)
    {
      // The given receiver must be the only one to send an immediate reply
      if (stationsReplyingWithNormalAck.size () == 1
          && stationsReplyingWithNormalAck.begin ()->first == receiver)
        {
          return true;
        }

      if (stationsReplyingWithBlockAck.size () == 1
          && stationsReplyingWithBlockAck.begin ()->first == receiver)
        {
          return true;
        }

      return false;
    }

  if (ackPolicy == WifiMacHeader::BLOCK_ACK)
    {
      return true;
    }

  return false;
}

void
WifiDlMuBarBaSequence::Print (std::ostream &os) const
{
  os << "DL_MU_BAR_BA_SEQUENCE [";
  for (const auto& sta : stationsReplyingWithNormalAck)
    {
      os << " (ACK) " << sta.first;
    }
  for (const auto& sta : stationsReplyingWithBlockAck)
    {
      os << " (BA) " << sta.first;
    }
  for (const auto& sta : stationsSendBlockAckReqTo)
    {
      os << " (BAR+BA) " << sta.first;
    }
  os << "]";
}


/*
 * WifiDlMuTfMuBar
 */

WifiDlMuTfMuBar::WifiDlMuTfMuBar ()
  : WifiAcknowledgment (DL_MU_TF_MU_BAR),
    ulLength (0)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiDlMuTfMuBar::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiDlMuTfMuBar (*this));
}

bool
WifiDlMuTfMuBar::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid,
                                    WifiMacHeader::QosAckPolicy ackPolicy) const
{
  // the only admitted ack policy is Block Ack because stations need to wait for a MU-BAR
  if (ackPolicy == WifiMacHeader::BLOCK_ACK)
    {
      return true;
    }

  return false;
}

void
WifiDlMuTfMuBar::Print (std::ostream &os) const
{
  os << "DL_MU_TF_MU_BAR [";
  for (const auto& sta : stationsReplyingWithBlockAck)
    {
      os << " (BA) " << sta.first;
    }
  os << "]";
}


/*
 * WifiDlMuAggregateTf
 */

WifiDlMuAggregateTf::WifiDlMuAggregateTf ()
  : WifiAcknowledgment (DL_MU_AGGREGATE_TF),
    ulLength (0)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiDlMuAggregateTf::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiDlMuAggregateTf (*this));
}

bool
WifiDlMuAggregateTf::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid,
                                        WifiMacHeader::QosAckPolicy ackPolicy) const
{
  // the only admitted ack policy is No explicit acknowledgment or TB PPDU Ack policy
  if (ackPolicy == WifiMacHeader::NO_EXPLICIT_ACK)
    {
      return true;
    }

  return false;
}

void
WifiDlMuAggregateTf::Print (std::ostream &os) const
{
  os << "DL_MU_AGGREGATE_TF [";
  for (const auto& sta : stationsReplyingWithBlockAck)
    {
      os << " (BA) " << sta.first;
    }
  os << "]";
}


/*
 * WifiUlMuMultiStaBa
 */

WifiUlMuMultiStaBa::WifiUlMuMultiStaBa ()
  : WifiAcknowledgment (UL_MU_MULTI_STA_BA),
    baType (BlockAckType::MULTI_STA)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiUlMuMultiStaBa::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiUlMuMultiStaBa (*this));
}

bool
WifiUlMuMultiStaBa::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid,
                                       WifiMacHeader::QosAckPolicy ackPolicy) const
{
  // a Basic Trigger Frame has no QoS ack policy
  return true;
}

void
WifiUlMuMultiStaBa::Print (std::ostream &os) const
{
  os << "UL_MU_MULTI_STA_BA [";
  for (const auto& sta : stationsReceivingMultiStaBa)
    {
      os << "(" << sta.first.first << "," << +sta.first.second << ") ";
    }
  os << "]";
}


/*
 * WifiAckAfterTbPpdu
 */

WifiAckAfterTbPpdu::WifiAckAfterTbPpdu ()
  : WifiAcknowledgment (ACK_AFTER_TB_PPDU)
{
}

std::unique_ptr<WifiAcknowledgment>
WifiAckAfterTbPpdu::Copy (void) const
{
  return std::unique_ptr<WifiAcknowledgment> (new WifiAckAfterTbPpdu (*this));
}

bool
WifiAckAfterTbPpdu::CheckQosAckPolicy (Mac48Address receiver, uint8_t tid,
                                        WifiMacHeader::QosAckPolicy ackPolicy) const
{
  if (ackPolicy == WifiMacHeader::NORMAL_ACK)
    {
      return true;
    }

  return false;
}

void
WifiAckAfterTbPpdu::Print (std::ostream &os) const
{
  os << "ACK_AFTER_TB_PPDU";
}


std::ostream & operator << (std::ostream &os, const WifiAcknowledgment* acknowledgment)
{
  acknowledgment->Print (os);
  return os;
}

} //namespace ns3
