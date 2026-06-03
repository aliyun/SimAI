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
 * Authors: Mirko Banchi <mk.banchi@gmail.com>
 *          Cecchi Niccolò <insa@igeek.it>
 */

#include "ns3/socket.h"
#include "ns3/queue-item.h"
#include "qos-utils.h"
#include "wifi-mac-header.h"
#include "mgt-headers.h"
#include "ctrl-headers.h"

namespace ns3 {

std::size_t
WifiAddressTidHash::operator()(const WifiAddressTidPair& addressTidPair) const
{
  uint8_t buffer[7];
  addressTidPair.first.CopyTo (buffer);
  buffer[6] = addressTidPair.second;

  std::string s (buffer, buffer + 7);
  return std::hash<std::string>{} (s);
}

std::size_t
WifiAddressHash::operator()(const Mac48Address& address) const
{
  uint8_t buffer[6];
  address.CopyTo (buffer);

  std::string s (buffer, buffer + 6);
  return std::hash<std::string>{} (s);
}

WifiAc::WifiAc (uint8_t lowTid, uint8_t highTid)
  : m_lowTid (lowTid),
    m_highTid (highTid)
{
}

uint8_t
WifiAc::GetLowTid (void) const
{
  return m_lowTid;
}

uint8_t
WifiAc::GetHighTid (void) const
{
  return m_highTid;
}

uint8_t
WifiAc::GetOtherTid (uint8_t tid) const
{
  if (tid == m_lowTid)
    {
      return m_highTid;
    }
  if (tid == m_highTid)
    {
      return m_lowTid;
    }
  NS_ABORT_MSG ("TID " << tid << " does not belong to this AC");
}

bool operator> (enum AcIndex left, enum AcIndex right)
{
  NS_ABORT_MSG_IF (left > 3 || right > 3, "Cannot compare non-QoS ACs");

  if (left == right)
    {
      return false;
    }
  if (left == AC_BK)
    {
      return false;
    }
  if (right == AC_BK)
    {
      return true;
    }
  return static_cast<uint8_t> (left) > static_cast<uint8_t> (right);
}

bool operator>= (enum AcIndex left, enum AcIndex right)
{
  NS_ABORT_MSG_IF (left > 3 || right > 3, "Cannot compare non-QoS ACs");

  return (left == right || left > right);
}

bool operator< (enum AcIndex left, enum AcIndex right)
{
  return !(left >= right);
}

bool operator<= (enum AcIndex left, enum AcIndex right)
{
  return !(left > right);
}

const std::map<AcIndex, WifiAc> wifiAcList = { {AC_BE, {0, 3}},
                                               {AC_BK, {1, 2}},
                                               {AC_VI, {4, 5}},
                                               {AC_VO, {6, 7}} };

AcIndex
QosUtilsMapTidToAc (uint8_t tid)
{
  NS_ASSERT_MSG (tid < 8, "Tid " << +tid << " out of range");
  switch (tid)
    {
    case 0:
    case 3:
      return AC_BE;
      break;
    case 1:
    case 2:
      return AC_BK;
      break;
    case 4:
    case 5:
      return AC_VI;
      break;
    case 6:
    case 7:
      return AC_VO;
      break;
    }
  return AC_UNDEF;
}

uint8_t
QosUtilsGetTidForPacket (Ptr<const Packet> packet)
{
  SocketPriorityTag qos;
  uint8_t tid = 8;
  if (packet->PeekPacketTag (qos))
    {
      if (qos.GetPriority () < 8)
        {
          tid = qos.GetPriority ();
        }
    }
  return tid;
}

uint32_t
QosUtilsMapSeqControlToUniqueInteger (uint16_t seqControl, uint16_t endSequence)
{
  uint32_t integer = 0;
  uint16_t numberSeq = (seqControl >> 4) & 0x0fff;
  integer = (4096 - (endSequence + 1) + numberSeq) % 4096;
  integer *= 16;
  integer += (seqControl & 0x000f);
  return integer;
}

bool
QosUtilsIsOldPacket (uint16_t startingSeq, uint16_t seqNumber)
{
  NS_ASSERT (startingSeq < 4096);
  NS_ASSERT (seqNumber < 4096);
  uint16_t distance = ((seqNumber - startingSeq) + 4096) % 4096;
  return (distance >= 2048);
}

uint8_t
GetTid (Ptr<const Packet> packet, const WifiMacHeader hdr)
{
  NS_ASSERT (hdr.IsQosData () || packet != 0);
  if (hdr.IsQosData ())
    {
      return hdr.GetQosTid ();
    }
  else if (hdr.IsBlockAckReq ())
    {
      CtrlBAckRequestHeader baReqHdr;
      packet->PeekHeader (baReqHdr);
      return baReqHdr.GetTidInfo ();
    }
  else if (hdr.IsBlockAck ())
    {
      CtrlBAckResponseHeader baRespHdr;
      packet->PeekHeader (baRespHdr);
      return baRespHdr.GetTidInfo ();
    }
  else if (hdr.IsMgt () && hdr.IsAction ())
    {
      Ptr<Packet> pkt = packet->Copy ();
      WifiActionHeader actionHdr;
      pkt->RemoveHeader (actionHdr);

      if (actionHdr.GetCategory () == WifiActionHeader::BLOCK_ACK)
        {
          switch (actionHdr.GetAction ().blockAck)
            {
            case WifiActionHeader::BLOCK_ACK_ADDBA_REQUEST:
              {
                MgtAddBaRequestHeader reqHdr;
                pkt->RemoveHeader (reqHdr);
                return reqHdr.GetTid ();
              }
            case WifiActionHeader::BLOCK_ACK_ADDBA_RESPONSE:
              {
                MgtAddBaResponseHeader respHdr;
                pkt->RemoveHeader (respHdr);
                return respHdr.GetTid ();
              }
            case WifiActionHeader::BLOCK_ACK_DELBA:
              {
                MgtDelBaHeader delHdr;
                pkt->RemoveHeader (delHdr);
                return delHdr.GetTid ();
              }
            default:
              {
                NS_FATAL_ERROR ("Cannot extract Traffic ID from this BA action frame");
              }
            }
        }
      else
        {
          NS_FATAL_ERROR ("Cannot extract Traffic ID from this action frame");
        }
    }
  else
    {
      NS_FATAL_ERROR ("Packet has no Traffic ID");
    }
  return 0; // Silence compiler warning about lack of return value
}

uint8_t
SelectQueueByDSField (Ptr<QueueItem> item)
{
  uint8_t dscp, priority = 0;
  if (item->GetUint8Value (QueueItem::IP_DSFIELD, dscp))
    {
      // if the QoS map element is implemented, it should be used here
      // to set the priority.
      // User priority is set to the three most significant bits of the DS field
      priority = dscp >> 5;
    }

  // replace the priority tag
  SocketPriorityTag priorityTag;
  priorityTag.SetPriority (priority);
  item->GetPacket ()->ReplacePacketTag (priorityTag);

  // if the admission control were implemented, here we should check whether
  // the access category assigned to the packet should be downgraded

  return static_cast<uint8_t> (QosUtilsMapTidToAc (priority));
}


} //namespace ns3
