/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "ie-dot11s-preq.h"
#include "ns3/address-utils.h"
#include "ns3/assert.h"
#include "ns3/packet.h"

namespace ns3 {
namespace dot11s {
/*************************
 * DestinationAddressUnit
 ************************/
DestinationAddressUnit::DestinationAddressUnit () :
  m_do (false), m_rf (false), m_usn (false), m_destinationAddress (Mac48Address ()), m_destSeqNumber (0)
{
}
void
DestinationAddressUnit::SetFlags (bool doFlag, bool rfFlag, bool usnFlag)
{
  m_do = doFlag;
  m_rf = rfFlag;
  m_usn = usnFlag;
}

void
DestinationAddressUnit::SetDestSeqNumber (uint32_t dest_seq_number)
{
  m_destSeqNumber = dest_seq_number;
  if (m_destSeqNumber != 0)
    {
      m_usn = true;
    }
}
void
DestinationAddressUnit::SetDestinationAddress (Mac48Address dest_address)
{
  m_destinationAddress = dest_address;
}
bool
DestinationAddressUnit::IsDo ()
{
  return m_do;
}

bool
DestinationAddressUnit::IsRf ()
{
  return m_rf;
}
bool
DestinationAddressUnit::IsUsn ()
{
  return m_usn;
}
uint32_t
DestinationAddressUnit::GetDestSeqNumber () const
{
  return m_destSeqNumber;
}
Mac48Address
DestinationAddressUnit::GetDestinationAddress () const
{
  return m_destinationAddress;
}
/********************************
 * IePreq
 *******************************/
IePreq::~IePreq ()
{
}
IePreq::IePreq () :
  m_maxSize (32), m_flags (0), m_hopCount (0), m_ttl (0), m_preqId (0), m_originatorAddress (
    Mac48Address::GetBroadcast ()), m_originatorSeqNumber (0), m_lifetime (0), m_metric (0),
  m_destCount (0)
{
}
WifiInformationElementId
IePreq::ElementId () const
{
  return IE_PREQ;
}
void
IePreq::SetUnicastPreq ()
{
  m_flags |= 1 << 1;
}

void
IePreq::SetNeedNotPrep ()
{
  m_flags |= 1 << 2;
}
void
IePreq::SetHopcount (uint8_t hopcount)
{
  m_hopCount = hopcount;
}
void
IePreq::SetTTL (uint8_t ttl)
{
  m_ttl = ttl;
}
void
IePreq::SetPreqID (uint32_t preq_id)
{
  m_preqId = preq_id;
}
void
IePreq::SetMetric (uint32_t metric)
{
  m_metric = metric;
}
void
IePreq::SetOriginatorAddress (Mac48Address originator_address)
{
  m_originatorAddress = originator_address;
}
void
IePreq::SetOriginatorSeqNumber (uint32_t originator_seq_number)
{
  m_originatorSeqNumber = originator_seq_number;
}
void
IePreq::SetLifetime (uint32_t lifetime)
{
  m_lifetime = lifetime;
}
void
IePreq::SetDestCount (uint8_t dest_count)
{
  m_destCount = dest_count;
}
bool
IePreq::IsUnicastPreq () const
{
  return (m_flags & (1 << 1));
}
bool
IePreq::IsNeedNotPrep () const
{
  return (m_flags & (1 << 2));
}
uint8_t
IePreq::GetHopCount () const
{
  return m_hopCount;
}
uint8_t
IePreq::GetTtl () const
{
  return m_ttl;
}
uint32_t
IePreq::GetPreqID () const
{
  return m_preqId;
}
uint32_t
IePreq::GetMetric () const
{
  return m_metric;
}
Mac48Address
IePreq::GetOriginatorAddress () const
{
  return m_originatorAddress;
}
uint32_t
IePreq::GetOriginatorSeqNumber () const
{
  return m_originatorSeqNumber;
}
uint32_t
IePreq::GetLifetime () const
{
  return m_lifetime;
}

uint8_t
IePreq::GetDestCount () const
{
  return m_destCount;
}
void
IePreq::DecrementTtl ()
{
  m_ttl--;
  m_hopCount++;
}
void
IePreq::IncrementMetric (uint32_t metric)
{
  m_metric += metric;
}
void
IePreq::SerializeInformationField (Buffer::Iterator i) const
{
  i.WriteU8 (m_flags);
  i.WriteU8 (m_hopCount);
  i.WriteU8 (m_ttl);
  i.WriteHtolsbU32 (m_preqId);
  WriteTo (i, m_originatorAddress);
  i.WriteHtolsbU32 (m_originatorSeqNumber);
  i.WriteHtolsbU32 (m_lifetime);
  i.WriteHtolsbU32 (m_metric);
  i.WriteU8 (m_destCount);
  int written = 0;
  for (std::vector<Ptr<DestinationAddressUnit> >::const_iterator j = m_destinations.begin (); j
       != m_destinations.end (); j++)
    {
      uint8_t flags = 0;
      if ((*j)->IsDo ())
        {
          flags |= 1 << 0;
        }
      if ((*j)->IsRf ())
        {
          flags |= 1 << 1;
        }
      if ((*j)->IsUsn ())
        {
          flags |= 1 << 2;
        }
      i.WriteU8 (flags);
      WriteTo (i, (*j)->GetDestinationAddress ());
      i.WriteHtolsbU32 ((*j)->GetDestSeqNumber ());
      written++;
      if (written > m_maxSize)
        {
          break;
        }
    }
}
uint8_t
IePreq::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  m_flags = i.ReadU8 ();
  m_hopCount = i.ReadU8 ();
  m_ttl = i.ReadU8 ();
  m_preqId = i.ReadLsbtohU32 ();
  ReadFrom (i, m_originatorAddress);
  m_originatorSeqNumber = i.ReadLsbtohU32 ();
  m_lifetime = i.ReadLsbtohU32 ();
  m_metric = i.ReadLsbtohU32 ();
  m_destCount = i.ReadU8 ();
  for (int j = 0; j < m_destCount; j++)
    {
      Ptr<DestinationAddressUnit> new_element = Create<DestinationAddressUnit> ();
      bool doFlag = false;
      bool rfFlag = false;
      bool usnFlag = false;
      uint8_t flags = i.ReadU8 ();
      if (flags & (1 << 0))
        {
          doFlag = true;
        }
      if (flags & (1 << 1))
        {
          rfFlag = true;
        }
      if (flags & (1 << 2))
        {
          usnFlag = true;
        }
      new_element->SetFlags (doFlag, rfFlag, usnFlag);
      Mac48Address addr;
      ReadFrom (i, addr);
      new_element->SetDestinationAddress (addr);
      new_element->SetDestSeqNumber (i.ReadLsbtohU32 ());
      m_destinations.push_back (new_element);
      NS_ASSERT (28 + j * 11 < length);
    }
  return i.GetDistanceFrom (start);
}
uint8_t
IePreq::GetInformationFieldSize () const
{
  uint8_t retval = 1 //Flags
    + 1   //Hopcount
    + 1   //TTL
    + 4   //PREQ ID
    + 6   //Source address (originator)
    + 4   //Originator seqno
    + 4   //Lifetime
    + 4   //metric
    + 1;   //destination count
  if (m_destCount > m_maxSize)
    {
      retval += (m_maxSize * 11);
    }
  else
    {
      retval += (m_destCount * 11);
    }
  return retval;
}
void
IePreq::Print (std::ostream &os) const
{
  os << "PREQ=(originator address=" << m_originatorAddress
     << ", TTL=" << (uint16_t) m_ttl
     << ", hop count=" << (uint16_t) m_hopCount
     << ", metric=" << m_metric
     << ", seqno=" << m_originatorSeqNumber
     << ", lifetime=" << m_lifetime
     << ", preq ID=" << m_preqId
     << ", Destinations=(";
  for (int j = 0; j < m_destCount; j++)
    {
      os << m_destinations[j]->GetDestinationAddress ();
    }
  os << ")";
}
std::vector<Ptr<DestinationAddressUnit> >
IePreq::GetDestinationList ()
{
  return m_destinations;
}
void
IePreq::AddDestinationAddressElement (bool doFlag, bool rfFlag, Mac48Address dest_address,
                                      uint32_t dest_seq_number)
{
  for (std::vector<Ptr<DestinationAddressUnit> >::const_iterator i = m_destinations.begin (); i
       != m_destinations.end (); i++)
    {
      if ((*i)->GetDestinationAddress () == dest_address)
        {
          return;
        }
    }
  /// \todo check overflow
  Ptr<DestinationAddressUnit> new_element = Create<DestinationAddressUnit> ();
  new_element->SetFlags (doFlag, rfFlag, (dest_seq_number == 0));
  new_element->SetDestinationAddress (dest_address);
  new_element->SetDestSeqNumber (dest_seq_number);
  m_destinations.push_back (new_element);
  m_destCount++;
}
void
IePreq::DelDestinationAddressElement (Mac48Address dest_address)
{
  for (std::vector<Ptr<DestinationAddressUnit> >::iterator i = m_destinations.begin (); i
       != m_destinations.end (); i++)
    {
      if ((*i)->GetDestinationAddress () == dest_address)
        {
          m_destinations.erase (i);
          m_destCount--;
          break;
        }
    }
}
void
IePreq::ClearDestinationAddressElements ()
{
  for (std::vector<Ptr<DestinationAddressUnit> >::iterator j = m_destinations.begin (); j
       != m_destinations.end (); j++)
    {
      (*j) = 0;
    }
  m_destinations.clear ();
  m_destCount = 0;
}
bool
operator== (const DestinationAddressUnit & a, const DestinationAddressUnit & b)
{
  return (a.m_do == b.m_do && a.m_rf == b.m_rf && a.m_usn == b.m_usn && a.m_destinationAddress
          == b.m_destinationAddress && a.m_destSeqNumber == b.m_destSeqNumber);
}
bool
operator== (const IePreq & a, const IePreq & b)
{
  bool ok = (a.m_flags == b.m_flags && a.m_hopCount == b.m_hopCount && a.m_ttl == b.m_ttl && a.m_preqId
             == b.m_preqId && a.m_originatorAddress == b.m_originatorAddress && a.m_originatorSeqNumber
             == b.m_originatorSeqNumber && a.m_lifetime == b.m_lifetime && a.m_metric == b.m_metric && a.m_destCount
             == b.m_destCount);

  if (!ok)
    {
      return false;
    }
  if (a.m_destinations.size () != b.m_destinations.size ())
    {
      return false;
    }
  for (size_t i = 0; i < a.m_destinations.size (); ++i)
    {
      if (!(*(PeekPointer (a.m_destinations[i])) == *(PeekPointer (b.m_destinations[i]))))
        {
          return false;
        }
    }
  return true;
}
bool
IePreq::MayAddAddress (Mac48Address originator)
{
  if (m_originatorAddress != originator)
    {
      return false;
    }
  if (m_destinations[0]->GetDestinationAddress () == Mac48Address::GetBroadcast ())
    {
      return false;
    }
  // -fstrict-overflow sensitive, see bug 1868
  if ( GetInformationFieldSize () > 255 - 11 )
    {
      return false;
    }
  return true;
}
bool
IePreq::IsFull () const
{
  // -fstrict-overflow sensitive, see bug 1868
  return ( GetInformationFieldSize () > 255 - 11 );
}
std::ostream &
operator << (std::ostream &os, const IePreq &a)
{
  a.Print (os);
  return os;
}
} // namespace dot11s
} // namespace ns3

