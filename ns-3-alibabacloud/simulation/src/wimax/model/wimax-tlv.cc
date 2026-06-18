/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2009 INRIA, UDcast
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
 *
 */

#include "wimax-tlv.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Tlv");
  
// NS_OBJECT_ENSURE_REGISTERED ("Tlv");

/* static */
TypeId
Tlv::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Tlv")
    .SetParent<Header> ()
    .SetGroupName ("Wimax")
    .AddConstructor<Tlv> ()
    ;
  return tid;
}
  
TypeId Tlv::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void Tlv::Print (std::ostream &os) const
{
  os << "TLV type = " << (uint32_t) m_type << " TLV Length = " << (uint64_t) m_length;
}

Tlv::Tlv (uint8_t type, uint64_t length, const TlvValue & value)
{
  m_type = type;
  m_length = length;
  m_value = value.Copy ();
}

Tlv::Tlv ()
{
  m_type = 0;
  m_length = 0;
  m_value = 0;
}

Tlv::~Tlv ()
{
  if (m_value != 0)
    {
      delete m_value;
      m_value = 0;
    }
}

TlvValue *
Tlv::CopyValue (void) const
{
  return m_value->Copy ();
}

Tlv::Tlv (const Tlv & tlv)
{
  m_type = tlv.GetType ();
  m_length = tlv.GetLength ();
  m_value = tlv.CopyValue ();
}

Tlv &
Tlv::operator = (Tlv const& o)
{
  if (m_value != 0)
    {
      delete m_value;
    }
  m_type = o.GetType ();
  m_length = o.GetLength ();
  m_value = o.CopyValue ();

  return *this;
}

uint32_t
Tlv::GetSerializedSize (void) const
{
  return 1 + GetSizeOfLen () + m_value->GetSerializedSize ();
}

uint8_t
Tlv::GetSizeOfLen (void) const
{
  uint8_t sizeOfLen = 1;

  if (m_length > 127)
    {
      sizeOfLen = 2;
      uint64_t testValue = 0xFF;
      while (m_length > testValue)
        {
          sizeOfLen++;
          testValue *= 0xFF;
        }
    }
  return sizeOfLen;
}

void
Tlv::Serialize (Buffer::Iterator i) const
{
  i.WriteU8 (m_type);
  uint8_t lenSize = GetSizeOfLen ();
  if (lenSize == 1)
    {
      i.WriteU8 (m_length);
    }
  else
    {
      i.WriteU8 ((lenSize-1) | WIMAX_TLV_EXTENDED_LENGTH_MASK);
      for (int j = 0; j < lenSize - 1; j++)
        {
          i.WriteU8 ((uint8_t)(m_length >> ((lenSize - 1 - 1 - j) * 8)));
        }
    }
  m_value->Serialize (i);
}

uint32_t
Tlv::Deserialize (Buffer::Iterator i)
{
  // read the type of tlv
  m_type = i.ReadU8 ();

  // read the length
  uint8_t lenSize = i.ReadU8 ();
  uint32_t serializedSize = 2;
  if (lenSize < 127)
    {
      m_length = lenSize;
    }
  else
    {
      lenSize &= ~WIMAX_TLV_EXTENDED_LENGTH_MASK;
      for (int j = 0; j < lenSize; j++)
        {
          m_length <<= 8;
          m_length |= i.ReadU8 ();
          serializedSize++;
        }
    }
  switch (m_type)
    {
    case HMAC_TUPLE:
      /// \todo implement Deserialize HMAC_TUPLE
      NS_FATAL_ERROR ("Not implemented-- please implement and contribute a patch");
      break;
    case MAC_VERSION_ENCODING:
      /// \todo implement Deserialize MAC_VERSION_ENCODING
      NS_FATAL_ERROR ("Not implemented-- please implement and contribute a patch");
      break;
    case CURRENT_TRANSMIT_POWER:
      /// \todo implement Deserialize CURRENT_TRANSMIT_POWER
      NS_FATAL_ERROR ("Not implemented-- please implement and contribute a patch");
      break;
    case DOWNLINK_SERVICE_FLOW:
      {
        SfVectorTlvValue val;
        serializedSize += val.Deserialize (i, m_length);
        m_value = val.Copy ();
        break;
      }
    case UPLINK_SERVICE_FLOW:
      {
        SfVectorTlvValue val;
        serializedSize += val.Deserialize (i, m_length);
        m_value = val.Copy ();
        break;
      }
    case VENDOR_ID_EMCODING:
      /// \todo implement Deserialize VENDOR_ID_EMCODING
      NS_FATAL_ERROR ("Not implemented-- please implement and contribute a patch");
      break;
    case VENDOR_SPECIFIC_INFORMATION:
      /// \todo implement Deserialize  VENDOR_SPECIFIC_INFORMATION
      NS_FATAL_ERROR ("Not implemented-- please implement and contribute a patch");
      break;
    default:
      NS_ASSERT_MSG (false, "Unknown tlv type.");
      break;
    }

  return serializedSize;
}

uint8_t
Tlv::GetType (void) const
{
  return m_type;
}
uint64_t
Tlv::GetLength (void) const
{
  return m_length;
}
TlvValue*
Tlv::PeekValue (void)
{
  return m_value;
}

Tlv *
Tlv::Copy (void) const
{
  return new Tlv (m_type, m_length, *m_value);
}
// ==============================================================================
VectorTlvValue::VectorTlvValue ()
{
  m_tlvList = new std::vector<Tlv*>;
}

VectorTlvValue::~VectorTlvValue ()
{
  for (std::vector<Tlv*>::const_iterator iter = m_tlvList->begin (); iter != m_tlvList->end (); ++iter)
    {
      delete (*iter);
    }
  m_tlvList->clear ();
  delete m_tlvList;
}

uint32_t
VectorTlvValue::GetSerializedSize (void) const
{
  uint32_t size = 0;
  for (std::vector<Tlv*>::const_iterator iter = m_tlvList->begin (); iter != m_tlvList->end (); ++iter)
    {
      size += (*iter)->GetSerializedSize ();
    }
  return size;
}

void
VectorTlvValue::Serialize (Buffer::Iterator i) const
{
  for (std::vector<Tlv*>::const_iterator iter = m_tlvList->begin (); iter != m_tlvList->end (); ++iter)
    {
      (*iter)->Serialize (i);
      i.Next ((*iter)->GetSerializedSize ());
    }
}

VectorTlvValue::Iterator
VectorTlvValue::Begin () const
{
  return m_tlvList->begin ();
}

VectorTlvValue::Iterator
VectorTlvValue::End () const
{
  return m_tlvList->end ();
}

void
VectorTlvValue::Add (const Tlv & val)
{
  m_tlvList->push_back (val.Copy ());
}

// ==============================================================================
SfVectorTlvValue::SfVectorTlvValue ()
{

}

SfVectorTlvValue *
SfVectorTlvValue::Copy (void) const
{
  SfVectorTlvValue * tmp = new SfVectorTlvValue ();
  for (std::vector<Tlv*>::const_iterator iter = Begin (); iter != End (); ++iter)
    {
      tmp->Add (Tlv ((*iter)->GetType (), (*iter)->GetLength (), *(*iter)->PeekValue ()));
    }
  return tmp;
}

uint32_t
SfVectorTlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLen)
{
  uint64_t serializedSize = 0;
  while (serializedSize < valueLen)
    {
      uint8_t type = i.ReadU8 ();
      // read the length
      uint8_t lenSize = i.ReadU8 ();
      serializedSize += 2;
      uint64_t length = 0;
      if (lenSize < 127)
        {
          length = lenSize;
        }
      else
        {
          lenSize &= ~WIMAX_TLV_EXTENDED_LENGTH_MASK;
          for (int j = 0; j < lenSize; j++)
            {
              length <<= 8;
              length |= i.ReadU8 ();
              serializedSize++;
            }
        }
      switch (type)
        {
        case SFID:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (SFID, 4, val));
            break;
          }
        case CID:
          {
            U16TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (CID, 2, val));
            break;
          }
        case Service_Class_Name:
          NS_FATAL_ERROR ("Not implemented-- please implement and contribute a patch");
          break;
        case reserved1:
          // NOTHING
          break;
        case QoS_Parameter_Set_Type:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (QoS_Parameter_Set_Type, 1, val));
            break;
          }
        case Traffic_Priority:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Traffic_Priority, 1, val));
            break;
          }
        case Maximum_Sustained_Traffic_Rate:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Maximum_Sustained_Traffic_Rate, 4, val));
            break;
          }
        case Maximum_Traffic_Burst:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Maximum_Traffic_Burst, 4, val));
            break;
          }
        case Minimum_Reserved_Traffic_Rate:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Minimum_Reserved_Traffic_Rate, 4, val));
            break;
          }
        case Minimum_Tolerable_Traffic_Rate:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Minimum_Tolerable_Traffic_Rate, 4, val));
            break;
          }
        case Service_Flow_Scheduling_Type:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Service_Flow_Scheduling_Type, 1, val));
            break;
          }
        case Request_Transmission_Policy:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Request_Transmission_Policy, 4, val));
            break;
          }
        case Tolerated_Jitter:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Tolerated_Jitter, 4, val));
            break;
          }
        case Maximum_Latency:
          {
            U32TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Maximum_Latency, 4, val));
            break;
          }
        case Fixed_length_versus_Variable_length_SDU_Indicator:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Fixed_length_versus_Variable_length_SDU_Indicator, 1, val));
            break;
          }
        case SDU_Size:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (SDU_Size, 1, val));
            break;
          }
        case Target_SAID:
          {
            U16TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Target_SAID, 2, val));
            break;
          }
        case ARQ_Enable:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (ARQ_Enable, 1, val));
            break;
          }
        case ARQ_WINDOW_SIZE:
          {
            U16TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (ARQ_WINDOW_SIZE, 2, val));
            break;
          }
        case ARQ_RETRY_TIMEOUT_Transmitter_Delay:
          break;
        case ARQ_RETRY_TIMEOUT_Receiver_Delay:
          break;
        case ARQ_BLOCK_LIFETIME:
          break;
        case ARQ_SYNC_LOSS:
          break;
        case ARQ_DELIVER_IN_ORDER:
          break;
        case ARQ_PURGE_TIMEOUT:
          break;
        case ARQ_BLOCK_SIZE:
          break;
        case reserved2:
          break;
        case CS_Specification:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (CS_Specification, 1, val));
            break;
          }
        case IPV4_CS_Parameters:
          {
            CsParamVectorTlvValue val;
            uint32_t size = val.Deserialize (i, length);
            serializedSize += size;
            Add (Tlv (IPV4_CS_Parameters, size, val));
            break;
          }
        default:
          NS_ASSERT_MSG (false, "Unknown tlv type.");
          break;
        }
      i.Next (length);
    }
  return serializedSize;
}

// ==============================================================================

U8TlvValue::U8TlvValue (uint8_t value)
{
  m_value = value;
}

U8TlvValue::U8TlvValue ()
{
  m_value = 0;
}

U8TlvValue::~U8TlvValue ()
{
}
uint32_t
U8TlvValue::GetSerializedSize (void) const
{
  return 1;
}
void
U8TlvValue::Serialize (Buffer::Iterator i) const
{
  i.WriteU8 (m_value);
}
uint32_t
U8TlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLen)
{
  return Deserialize (i);
}

uint32_t
U8TlvValue::Deserialize (Buffer::Iterator i)
{
  m_value = i.ReadU8 ();
  return 1;
}

uint8_t
U8TlvValue::GetValue (void) const
{
  return m_value;
}

U8TlvValue *
U8TlvValue::Copy (void) const
{
  U8TlvValue * tmp = new U8TlvValue (m_value);
  return tmp;
}
// ==============================================================================
U16TlvValue::U16TlvValue (uint16_t value)
{
  m_value = value;
}

U16TlvValue::U16TlvValue ()
{
  m_value = 0;
}

U16TlvValue::~U16TlvValue (void)
{
}

uint32_t
U16TlvValue::GetSerializedSize (void) const
{
  return 2;
}
void
U16TlvValue::Serialize (Buffer::Iterator i) const
{
  i.WriteHtonU16 (m_value);
}
uint32_t
U16TlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLen)
{
  return Deserialize (i);
}

uint32_t
U16TlvValue::Deserialize (Buffer::Iterator i)
{
  m_value = i.ReadNtohU16 ();
  return 2;
}

uint16_t
U16TlvValue::GetValue (void) const
{
  return m_value;
}

U16TlvValue *
U16TlvValue::Copy (void) const
{
  U16TlvValue * tmp = new U16TlvValue (m_value);
  return tmp;
}
// ==============================================================================
U32TlvValue::U32TlvValue (uint32_t value)
{
  m_value = value;
}

U32TlvValue::U32TlvValue ()
{
  m_value = 0;
}

U32TlvValue::~U32TlvValue (void)
{
}

uint32_t U32TlvValue::GetSerializedSize (void) const
{
  return 4;
}
void
U32TlvValue::Serialize (Buffer::Iterator i) const
{
  i.WriteHtonU32 (m_value);
}
uint32_t
U32TlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLen)
{
  return Deserialize (i);
}

uint32_t
U32TlvValue::Deserialize (Buffer::Iterator i)
{
  m_value = i.ReadNtohU32 ();
  return 4;
}
uint32_t
U32TlvValue::GetValue (void) const
{
  return m_value;
}

U32TlvValue *
U32TlvValue::Copy (void) const
{
  U32TlvValue * tmp = new U32TlvValue (m_value);
  return tmp;
}
// ==============================================================================
uint32_t
CsParamVectorTlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLength)
{
  uint64_t serializedSize = 0;
  uint8_t lenSize = 0;
  uint8_t type = 0;
  while (serializedSize < valueLength)
    {
      type = i.ReadU8 ();
      // read the length
      lenSize = i.ReadU8 ();
      serializedSize += 2;
      uint64_t length = 0;
      if (lenSize < 127)
        {
          length = lenSize;
        }
      else
        {
          lenSize &= ~WIMAX_TLV_EXTENDED_LENGTH_MASK;
          for (int j = 0; j < lenSize; j++)
            {
              length <<= 8;
              length |= i.ReadU8 ();
              serializedSize++;
            }
        }
      switch (type)
        {
        case Classifier_DSC_Action:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Classifier_DSC_Action, 1, val));
            break;
          }
        case Packet_Classification_Rule:
          {
            ClassificationRuleVectorTlvValue val;
            serializedSize += val.Deserialize (i, length);
            Add (Tlv (Packet_Classification_Rule, val.GetSerializedSize (), val));
            break;
          }
        }
      i.Next (length);
    }
  return serializedSize;
}

CsParamVectorTlvValue::CsParamVectorTlvValue ()
{

}

CsParamVectorTlvValue *
CsParamVectorTlvValue::Copy (void) const
{
  CsParamVectorTlvValue * tmp = new CsParamVectorTlvValue ();
  for (std::vector<Tlv*>::const_iterator iter = Begin (); iter != End (); ++iter)
    {
      tmp->Add (Tlv ((*iter)->GetType (), (*iter)->GetLength (), *(*iter)->PeekValue ()));
    }
  return tmp;
}
// ==============================================================================

ClassificationRuleVectorTlvValue::ClassificationRuleVectorTlvValue ()
{

}

ClassificationRuleVectorTlvValue *
ClassificationRuleVectorTlvValue::Copy (void) const
{
  ClassificationRuleVectorTlvValue * tmp = new ClassificationRuleVectorTlvValue ();
  for (std::vector<Tlv*>::const_iterator iter = Begin (); iter != End (); ++iter)
    {
      tmp->Add (Tlv ((*iter)->GetType (), (*iter)->GetLength (), *(*iter)->PeekValue ()));
    }
  return tmp;
}

uint32_t
ClassificationRuleVectorTlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLength)
{
  uint64_t serializedSize = 0;
  uint8_t lenSize = 0;
  uint8_t type = 0;
  while (serializedSize < valueLength)
    {
      type = i.ReadU8 ();
      // read the length
      lenSize = i.ReadU8 ();
      serializedSize += 2;
      uint64_t length = 0;
      if (lenSize < 127)
        {
          length = lenSize;
        }
      else
        {
          lenSize &= ~WIMAX_TLV_EXTENDED_LENGTH_MASK;
          for (int j = 0; j < lenSize; j++)
            {
              length <<= 8;
              length |= i.ReadU8 ();
              serializedSize++;
            }
        }
      switch (type)
        {
        case Priority:
          {
            U8TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Priority, 1, val));
            break;
          }
        case ToS:
          {
            TosTlvValue val;
            serializedSize += val.Deserialize (i, length);
            Add (Tlv (ToS, val.GetSerializedSize (), val));
            break;
          }
        case Protocol:
          {
            ProtocolTlvValue val;
            serializedSize += val.Deserialize (i, length);
            Add (Tlv (Protocol, val.GetSerializedSize (), val));
            break;
          }
        case IP_src:
          {
            Ipv4AddressTlvValue val;
            serializedSize += val.Deserialize (i, length);
            Add (Tlv (IP_src, val.GetSerializedSize (), val));
            break;
          }
        case IP_dst:
          {
            Ipv4AddressTlvValue val;
            serializedSize += val.Deserialize (i, length);
            Add (Tlv (IP_dst, val.GetSerializedSize (), val));
            break;
          }
        case Port_src:
          {
            PortRangeTlvValue val;
            serializedSize += val.Deserialize (i, length);
            Add (Tlv (Port_src, val.GetSerializedSize (), val));
            break;
          }
        case Port_dst:
          {
            PortRangeTlvValue val;
            serializedSize += val.Deserialize (i, length);
            Add (Tlv (Port_dst, val.GetSerializedSize (), val));
            break;
          }
        case Index:
          {
            U16TlvValue val;
            serializedSize += val.Deserialize (i);
            Add (Tlv (Index, 2, val));
            break;
          }
        }
      i.Next (length);
    }
  return serializedSize;
}

// ==============================================================================
TosTlvValue::TosTlvValue ()
{
  m_low = 0;
  m_high = 0;
  m_mask = 0;
}
TosTlvValue::TosTlvValue (uint8_t low, uint8_t high, uint8_t mask)
{
  m_low = low;
  m_high = high;
  m_mask = mask;
}
TosTlvValue::~TosTlvValue ()
{
}

uint32_t
TosTlvValue::GetSerializedSize (void) const
{
  return 3;
}
void
TosTlvValue::Serialize (Buffer::Iterator i) const
{
  i.WriteU8 (m_low);
  i.WriteU8 (m_high);
  i.WriteU8 (m_mask);
}
uint32_t
TosTlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLength)
{
  m_low = i.ReadU8 ();
  m_high = i.ReadU8 ();
  m_mask = i.ReadU8 ();
  return 3;
}
uint8_t
TosTlvValue::GetLow (void) const
{
  return m_low;
}
uint8_t
TosTlvValue::GetHigh (void) const
{
  return m_high;
}
uint8_t
TosTlvValue::GetMask (void) const
{
  return m_mask;
}

TosTlvValue *
TosTlvValue::Copy (void) const
{
  return new TosTlvValue (m_low, m_high, m_mask);
}

// ==============================================================================
PortRangeTlvValue::PortRangeTlvValue ()
{
  m_portRange = new std::vector<struct PortRange>;
}
PortRangeTlvValue::~PortRangeTlvValue ()
{
  m_portRange->clear ();
  delete m_portRange;
}

uint32_t
PortRangeTlvValue::GetSerializedSize (void) const
{
  return m_portRange->size () * 4; // a port range is defined by 2 ports, each using 2 bytes
}
void
PortRangeTlvValue::Serialize (Buffer::Iterator i) const
{
  for (std::vector<struct PortRange>::const_iterator iter = m_portRange->begin (); iter != m_portRange->end (); ++iter)
    {
      i.WriteHtonU16 ((*iter).PortLow);
      i.WriteHtonU16 ((*iter).PortHigh);
    }
}
uint32_t
PortRangeTlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLength)
{
  uint64_t len = 0;
  while (len < valueLength)
    {
      uint16_t low = i.ReadNtohU16 ();
      uint16_t high = i.ReadNtohU16 ();
      Add (low, high);
      len += 4;
    }
  return len;
}
void
PortRangeTlvValue::Add (uint16_t portLow, uint16_t portHigh)
{
  struct PortRange tmp;
  tmp.PortLow = portLow;
  tmp.PortHigh = portHigh;
  m_portRange->push_back (tmp);
}
PortRangeTlvValue::Iterator
PortRangeTlvValue::Begin (void) const
{
  return m_portRange->begin ();
}

PortRangeTlvValue::Iterator
PortRangeTlvValue::End (void) const
{
  return m_portRange->end ();
}

PortRangeTlvValue *
PortRangeTlvValue::Copy (void) const
{
  PortRangeTlvValue * tmp = new PortRangeTlvValue ();
  for (std::vector<struct PortRange>::const_iterator iter = m_portRange->begin (); iter != m_portRange->end (); ++iter)
    {
      tmp->Add ((*iter).PortLow, (*iter).PortHigh);
    }
  return tmp;
}

// ==============================================================================

ProtocolTlvValue::ProtocolTlvValue ()
{
  m_protocol = new std::vector<uint8_t>;
}
ProtocolTlvValue::~ProtocolTlvValue ()
{
  if (m_protocol != 0)
    {
      m_protocol->clear ();
      delete m_protocol;
      m_protocol = 0;
    }
}

uint32_t
ProtocolTlvValue::GetSerializedSize (void) const
{
  return m_protocol->size ();
}

void
ProtocolTlvValue::Serialize (Buffer::Iterator i) const
{
  for (std::vector<uint8_t>::const_iterator iter = m_protocol->begin (); iter != m_protocol->end (); ++iter)
    {
      i.WriteU8 ((*iter));
    }
}

uint32_t
ProtocolTlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLength)
{
  uint64_t len = 0;
  while (len < valueLength)
    {
      Add (i.ReadU8 ());
      len++;
    }
  return len;
}

void
ProtocolTlvValue::Add (uint8_t protocol)
{
  m_protocol->push_back (protocol);
}

ProtocolTlvValue::Iterator
ProtocolTlvValue::Begin (void) const
{
  return m_protocol->begin ();
}

ProtocolTlvValue::Iterator
ProtocolTlvValue::End (void) const
{
  return m_protocol->end ();
}

ProtocolTlvValue*
ProtocolTlvValue::Copy (void) const
{
  ProtocolTlvValue* tmp = new ProtocolTlvValue ();
  for (std::vector<uint8_t>::const_iterator iter = m_protocol->begin (); iter != m_protocol->end (); ++iter)
    {
      tmp->Add ((*iter));
    }
  return tmp;
}

// ==============================================================================

Ipv4AddressTlvValue::Ipv4AddressTlvValue ()
{
  m_ipv4Addr = new std::vector<struct ipv4Addr>;
}

Ipv4AddressTlvValue::~Ipv4AddressTlvValue ()
{
  if (m_ipv4Addr != 0)
    {
      m_ipv4Addr->clear ();
      delete m_ipv4Addr;
      m_ipv4Addr = 0;
    }
}

uint32_t
Ipv4AddressTlvValue::GetSerializedSize (void) const
{
  return m_ipv4Addr->size () * 8; // IPv4 address and mask are 4 bytes each
}

void
Ipv4AddressTlvValue::Serialize (Buffer::Iterator i) const
{
  for (std::vector<struct ipv4Addr>::const_iterator iter = m_ipv4Addr->begin (); iter != m_ipv4Addr->end (); ++iter)
    {
      i.WriteHtonU32 ((*iter).Address.Get ());
      i.WriteHtonU32 ((*iter).Mask.Get ());
    }
}

uint32_t
Ipv4AddressTlvValue::Deserialize (Buffer::Iterator i, uint64_t valueLength)
{
  uint64_t len = 0;
  while (len < valueLength)
    {
      uint32_t addr = i.ReadNtohU32 ();
      uint32_t mask = i.ReadNtohU32 ();
      Add (Ipv4Address (addr), Ipv4Mask (mask));
      len += 8;
    }
  return len;
}

void
Ipv4AddressTlvValue::Add (Ipv4Address address, Ipv4Mask Mask)
{
  struct ipv4Addr tmp;
  tmp.Address = address;
  tmp.Mask = Mask;
  m_ipv4Addr->push_back (tmp);
}

Ipv4AddressTlvValue::Iterator
Ipv4AddressTlvValue::Begin () const
{
  return m_ipv4Addr->begin ();
}

Ipv4AddressTlvValue::Iterator
Ipv4AddressTlvValue::End () const
{
  return m_ipv4Addr->end ();
}

Ipv4AddressTlvValue *
Ipv4AddressTlvValue::Copy (void) const
{
  Ipv4AddressTlvValue * tmp = new Ipv4AddressTlvValue ();
  for (std::vector<struct ipv4Addr>::const_iterator iter = m_ipv4Addr->begin (); iter != m_ipv4Addr->end (); ++iter)
    {
      tmp->Add ((*iter).Address, (*iter).Mask);
    }
  return tmp;
}

}
