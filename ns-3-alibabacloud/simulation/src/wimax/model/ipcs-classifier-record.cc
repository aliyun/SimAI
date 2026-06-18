/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */
#include "ipcs-classifier-record.h"
#include <stdint.h>
#include "ns3/ipv4-address.h"
#include "wimax-tlv.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("IpcsClassifierRecord");

IpcsClassifierRecord::IpcsClassifierRecord (void)
{
  m_priority = 255;
  m_priority = 0;
  m_index = 0;
  m_tosLow = 0;
  m_tosHigh = 0;
  m_tosMask = 0;
  m_cid = 0;
  m_protocol.push_back (6); // tcp
  m_protocol.push_back (17); // udp
  AddSrcAddr (Ipv4Address ("0.0.0.0"), Ipv4Mask ("0.0.0.0"));
  AddDstAddr (Ipv4Address ("0.0.0.0"), Ipv4Mask ("0.0.0.0"));
  AddSrcPortRange (0, 65535);
  AddDstPortRange (0, 65535);
}

IpcsClassifierRecord::~IpcsClassifierRecord (void)
{
}

IpcsClassifierRecord::IpcsClassifierRecord (Tlv tlv)
{
  NS_ASSERT_MSG (tlv.GetType () == CsParamVectorTlvValue::Packet_Classification_Rule, "Invalid TLV");
  ClassificationRuleVectorTlvValue* rules = ((ClassificationRuleVectorTlvValue*)(tlv.PeekValue ()));
  m_priority = 0;
  m_index = 0;
  m_tosLow = 0;
  m_tosHigh = 0;
  m_tosMask = 0;
  m_cid = 0;
  for (std::vector<Tlv*>::const_iterator iter = rules->Begin (); iter != rules->End (); ++iter)
    {
      switch ((*iter)->GetType ())
        {
        case ClassificationRuleVectorTlvValue::Priority:
          {
            m_priority = ((U8TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        case ClassificationRuleVectorTlvValue::ToS:
          {
            NS_FATAL_ERROR ("ToS Not implemented-- please implement and contribute a patch");
            break;
          }
        case ClassificationRuleVectorTlvValue::Protocol:
          {
            ProtocolTlvValue * list = (ProtocolTlvValue *)(*iter)->PeekValue ();
            for (std::vector<uint8_t>::const_iterator iter2 = list->Begin (); iter2 != list->End (); ++iter2)
              {
                AddProtocol (*iter2);
              }
            break;
          }
        case ClassificationRuleVectorTlvValue::IP_src:
          {
            Ipv4AddressTlvValue * list = (Ipv4AddressTlvValue *)(*iter)->PeekValue ();
            for (std::vector<Ipv4AddressTlvValue::ipv4Addr>::const_iterator iter2 = list->Begin (); iter2 != list->End (); ++iter2)
              {
                AddSrcAddr ((*iter2).Address, (*iter2).Mask);
              }
            break;
          }
        case ClassificationRuleVectorTlvValue::IP_dst:
          {
            Ipv4AddressTlvValue * list = (Ipv4AddressTlvValue *)(*iter)->PeekValue ();
            for (std::vector<Ipv4AddressTlvValue::ipv4Addr>::const_iterator iter2 = list->Begin (); iter2 != list->End (); ++iter2)
              {
                AddDstAddr ((*iter2).Address, (*iter2).Mask);
              }
            break;
          }
        case ClassificationRuleVectorTlvValue::Port_src:
          {
            PortRangeTlvValue * list = (PortRangeTlvValue *)(*iter)->PeekValue ();
            for (std::vector<PortRangeTlvValue::PortRange>::const_iterator iter2 = list->Begin (); iter2 != list->End (); ++iter2)
              {
                AddSrcPortRange ((*iter2).PortLow, (*iter2).PortHigh);
              }
            break;
          }
        case ClassificationRuleVectorTlvValue::Port_dst:
          {
            PortRangeTlvValue * list = (PortRangeTlvValue *)(*iter)->PeekValue ();
            for (std::vector<PortRangeTlvValue::PortRange>::const_iterator iter2 = list->Begin (); iter2 != list->End (); ++iter2)
              {
                AddDstPortRange ((*iter2).PortLow, (*iter2).PortHigh);
              }
            break;
          }
        case ClassificationRuleVectorTlvValue::Index:
          {
            m_index = ((U16TlvValue*)((*iter)->PeekValue ()))->GetValue ();
            break;
          }
        }
    }
}

IpcsClassifierRecord::IpcsClassifierRecord (Ipv4Address SrcAddress,
                                            Ipv4Mask SrcMask,
                                            Ipv4Address DstAddress,
                                            Ipv4Mask DstMask,
                                            uint16_t SrcPortLow,
                                            uint16_t SrcPortHigh,
                                            uint16_t DstPortLow,
                                            uint16_t DstPortHigh,
                                            uint8_t protocol,
                                            uint8_t priority)
{
  m_priority = priority;
  m_protocol.push_back (protocol);
  AddSrcAddr (SrcAddress, SrcMask);
  AddDstAddr (DstAddress, DstMask);
  AddSrcPortRange (SrcPortLow, SrcPortHigh);
  AddDstPortRange (DstPortLow, DstPortHigh);
  m_index = 0;
  m_tosLow = 0;
  m_tosHigh = 0;
  m_tosMask = 0;
  m_cid = 0;
}

void
IpcsClassifierRecord::AddSrcAddr (Ipv4Address srcAddress, Ipv4Mask srcMask)
{
  struct ipv4Addr tmp;
  tmp.Address = srcAddress;
  tmp.Mask = srcMask;
  m_srcAddr.push_back (tmp);
}
void
IpcsClassifierRecord::AddDstAddr (Ipv4Address dstAddress, Ipv4Mask dstMask)
{
  struct ipv4Addr tmp;
  tmp.Address = dstAddress;
  tmp.Mask = dstMask;
  m_dstAddr.push_back (tmp);
}
void
IpcsClassifierRecord::AddSrcPortRange (uint16_t srcPortLow, uint16_t srcPortHigh)
{
  struct PortRange tmp;
  tmp.PortLow = srcPortLow;
  tmp.PortHigh = srcPortHigh;
  m_srcPortRange.push_back (tmp);

}
void
IpcsClassifierRecord::AddDstPortRange (uint16_t dstPortLow, uint16_t dstPortHigh)
{
  struct PortRange tmp;
  tmp.PortLow = dstPortLow;
  tmp.PortHigh = dstPortHigh;
  m_dstPortRange.push_back (tmp);
}
void
IpcsClassifierRecord::AddProtocol (uint8_t proto)
{
  m_protocol.push_back (proto);
}
void
IpcsClassifierRecord::SetPriority (uint8_t prio)
{
  m_priority = prio;
}
void
IpcsClassifierRecord::SetCid (uint16_t cid)
{
  m_cid = cid;
}
void
IpcsClassifierRecord::SetIndex (uint16_t index)
{
  m_index = index;
}

uint16_t
IpcsClassifierRecord::GetIndex (void) const
{
  return m_index;
}
uint16_t
IpcsClassifierRecord::GetCid (void) const
{
  return m_cid;
}
uint8_t
IpcsClassifierRecord::GetPriority (void) const
{
  return m_priority;
}

bool
IpcsClassifierRecord::CheckMatchSrcAddr (Ipv4Address srcAddress) const
{
  for (std::vector<struct ipv4Addr>::const_iterator iter = m_srcAddr.begin (); iter != m_srcAddr.end (); ++iter)
    {
      NS_LOG_INFO ("src addr check match: pkt=" << srcAddress << " cls=" << (*iter).Address << "/" << (*iter).Mask);
      if (srcAddress.CombineMask ((*iter).Mask) == (*iter).Address)
        {
          return true;
        }
    }
  NS_LOG_INFO ("NOT OK!");
  return false;
}
bool
IpcsClassifierRecord::CheckMatchDstAddr (Ipv4Address dstAddress) const
{

  for (std::vector<struct ipv4Addr>::const_iterator iter = m_dstAddr.begin (); iter != m_dstAddr.end (); ++iter)
    {
      NS_LOG_INFO ("dst addr check match: pkt=" << dstAddress << " cls=" << (*iter).Address << "/" << (*iter).Mask);
      if (dstAddress.CombineMask ((*iter).Mask) == (*iter).Address)
        {
          return true;
        }
    }
  NS_LOG_INFO ("NOT OK!");
  return false;
}
bool
IpcsClassifierRecord::CheckMatchSrcPort (uint16_t port) const
{
  for (std::vector<struct PortRange>::const_iterator iter = m_srcPortRange.begin (); iter != m_srcPortRange.end (); ++iter)
    {
      NS_LOG_INFO ("src port check match: pkt=" << port << " cls= [" << (*iter).PortLow << " TO " << (*iter).PortHigh
                                                << "]");
      if (port >= (*iter).PortLow && port <= (*iter).PortHigh)
        {
          return true;
        }
    }
  NS_LOG_INFO ("NOT OK!");
  return false;
}
bool
IpcsClassifierRecord::CheckMatchDstPort (uint16_t port) const
{
  for (std::vector<struct PortRange>::const_iterator iter = m_dstPortRange.begin (); iter != m_dstPortRange.end (); ++iter)
    {
      NS_LOG_INFO ("dst port check match: pkt=" << port << " cls= [" << (*iter).PortLow << " TO " << (*iter).PortHigh
                                                << "]");
      if (port >= (*iter).PortLow && port <= (*iter).PortHigh)
        {
          return true;
        }
    }
  NS_LOG_INFO ("NOT OK!");
  return false;
}
bool
IpcsClassifierRecord::CheckMatchProtocol (uint8_t proto) const
{
  for (std::vector<uint8_t>::const_iterator iter = m_protocol.begin (); iter != m_protocol.end (); ++iter)
    {
      NS_LOG_INFO ("proto check match: pkt=" << (uint16_t) proto << " cls=" << (uint16_t) proto);
      if (proto == (*iter))
        {
          return true;
        }
    }
  NS_LOG_INFO ("NOT OK!");
  return false;
}
bool
IpcsClassifierRecord::CheckMatch (Ipv4Address srcAddress,
                                  Ipv4Address dstAddress,
                                  uint16_t srcPort,
                                  uint16_t dstPort,
                                  uint8_t proto) const
{
  return (CheckMatchProtocol (proto) && CheckMatchDstPort (dstPort) && CheckMatchSrcPort (srcPort)
          && CheckMatchDstAddr (dstAddress) && CheckMatchSrcAddr (srcAddress));
}

Tlv
IpcsClassifierRecord::ToTlv (void) const
{
  Ipv4AddressTlvValue ipv4AddrValSrc;
  for (std::vector<struct ipv4Addr>::const_iterator iter = m_srcAddr.begin (); iter != m_srcAddr.end (); ++iter)
    {
      ipv4AddrValSrc.Add ((*iter).Address, (*iter).Mask);
    }

  Ipv4AddressTlvValue ipv4AddrValDst;
  for (std::vector<struct ipv4Addr>::const_iterator iter = m_dstAddr.begin (); iter != m_dstAddr.end (); ++iter)
    {
      ipv4AddrValDst.Add ((*iter).Address, (*iter).Mask);
    }

  ProtocolTlvValue protoVal;
  for (std::vector<uint8_t>::const_iterator iter = m_protocol.begin (); iter != m_protocol.end (); ++iter)
    {
      protoVal.Add ((*iter));
    }

  PortRangeTlvValue portValueSrc;
  for (std::vector<struct PortRange>::const_iterator iter = m_srcPortRange.begin (); iter != m_srcPortRange.end (); ++iter)
    {
      portValueSrc.Add ((*iter).PortLow, (*iter).PortHigh);
    }

  PortRangeTlvValue portValueDst;
  for (std::vector<struct PortRange>::const_iterator iter = m_dstPortRange.begin (); iter != m_dstPortRange.end (); ++iter)
    {
      portValueDst.Add ((*iter).PortLow, (*iter).PortHigh);
    }

  ClassificationRuleVectorTlvValue ClassVectVal;
  ClassVectVal.Add (Tlv (ClassificationRuleVectorTlvValue::Priority, 1, U8TlvValue (m_priority)));
  ClassVectVal.Add (Tlv (ClassificationRuleVectorTlvValue::Protocol, protoVal.GetSerializedSize (), protoVal));
  ClassVectVal.Add (Tlv (ClassificationRuleVectorTlvValue::IP_src, ipv4AddrValSrc.GetSerializedSize (), ipv4AddrValSrc));
  ClassVectVal.Add (Tlv (ClassificationRuleVectorTlvValue::IP_dst, ipv4AddrValDst.GetSerializedSize (), ipv4AddrValDst));
  ClassVectVal.Add (Tlv (ClassificationRuleVectorTlvValue::Port_src, portValueSrc.GetSerializedSize (), portValueSrc));
  ClassVectVal.Add (Tlv (ClassificationRuleVectorTlvValue::Port_dst, portValueDst.GetSerializedSize (), portValueDst));
  ClassVectVal.Add (Tlv (ClassificationRuleVectorTlvValue::Index, 2, U16TlvValue (1)));

  Tlv tmp_tlv (CsParamVectorTlvValue::Packet_Classification_Rule, ClassVectVal.GetSerializedSize (), ClassVectVal);

  return tmp_tlv;
}

} // namespace ns3
