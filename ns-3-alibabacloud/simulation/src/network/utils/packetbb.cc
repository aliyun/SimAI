/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* vim: set ts=2 sw=2 sta expandtab ai si cin: */
/* 
 * Copyright (c) 2009 Drexel University
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
 * Author: Tom Wambold <tom5760@gmail.com>
 */
/* These classes implement RFC 5444 - The Generalized Mobile Ad Hoc Network
 * (MANET) Packet/PbbMessage Format
 * See: http://tools.ietf.org/html/rfc5444 for details */

#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "packetbb.h"

static const uint8_t VERSION = 0;
/* Packet flags */
static const uint8_t PHAS_SEQ_NUM = 0x8;
static const uint8_t PHAS_TLV = 0x4;

/* PbbMessage flags */
static const uint8_t MHAS_ORIG = 0x80;
static const uint8_t MHAS_HOP_LIMIT = 0x40;
static const uint8_t MHAS_HOP_COUNT = 0x20;
static const uint8_t MHAS_SEQ_NUM = 0x10;

/* Address block flags */
static const uint8_t AHAS_HEAD = 0x80;
static const uint8_t AHAS_FULL_TAIL = 0x40;
static const uint8_t AHAS_ZERO_TAIL = 0x20;
static const uint8_t AHAS_SINGLE_PRE_LEN = 0x10;
static const uint8_t AHAS_MULTI_PRE_LEN = 0x08;

/* TLV Flags */
static const uint8_t THAS_TYPE_EXT = 0x80;
static const uint8_t THAS_SINGLE_INDEX = 0x40;
static const uint8_t THAS_MULTI_INDEX = 0x20;
static const uint8_t THAS_VALUE = 0x10;
static const uint8_t THAS_EXT_LEN = 0x08;
static const uint8_t TIS_MULTIVALUE = 0x04;

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketBB");

NS_OBJECT_ENSURE_REGISTERED (PbbPacket);

PbbTlvBlock::PbbTlvBlock (void)
{
  NS_LOG_FUNCTION (this);
  return;
}

PbbTlvBlock::~PbbTlvBlock (void)
{
  NS_LOG_FUNCTION (this);
  Clear ();
}

PbbTlvBlock::Iterator
PbbTlvBlock::Begin (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.begin ();
}

PbbTlvBlock::ConstIterator
PbbTlvBlock::Begin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.begin ();
}

PbbTlvBlock::Iterator
PbbTlvBlock::End (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.end ();
}

PbbTlvBlock::ConstIterator
PbbTlvBlock::End (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.end ();
}

int
PbbTlvBlock::Size (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.size ();
}

bool
PbbTlvBlock::Empty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.empty ();
}

Ptr<PbbTlv>
PbbTlvBlock::Front (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.front ();
}

Ptr<PbbTlv>
PbbTlvBlock::Back (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.back ();
}

void
PbbTlvBlock::PushFront (Ptr<PbbTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.push_front (tlv);
}

void
PbbTlvBlock::PopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.pop_front ();
}

void
PbbTlvBlock::PushBack (Ptr<PbbTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.push_back (tlv);
}

void
PbbTlvBlock::PopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.pop_back ();
}

PbbTlvBlock::Iterator
PbbTlvBlock::Insert (PbbTlvBlock::Iterator position, const Ptr<PbbTlv> tlv)
{
  NS_LOG_FUNCTION (this << &position << tlv);
  return m_tlvList.insert (position, tlv);
}

PbbTlvBlock::Iterator
PbbTlvBlock::Erase (PbbTlvBlock::Iterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_tlvList.erase (position);
}

PbbTlvBlock::Iterator
PbbTlvBlock::Erase (PbbTlvBlock::Iterator first, PbbTlvBlock::Iterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_tlvList.erase (first, last);
}

void
PbbTlvBlock::Clear (void)
{
  NS_LOG_FUNCTION (this);
  for (Iterator iter = Begin (); iter != End (); iter++)
    {
      *iter = 0;
    }
  m_tlvList.clear ();
}

uint32_t
PbbTlvBlock::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  /* tlv size */
  uint32_t size = 2;
  for (ConstIterator iter = Begin (); iter != End (); iter++)
    {
      size += (*iter)->GetSerializedSize ();
    }
  return size;
}

void
PbbTlvBlock::Serialize (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  if (Empty ())
    {
      start.WriteHtonU16 (0);
      return;
    }

  /* We need to write the size of the TLV block in front, so save its
   * position. */
  Buffer::Iterator tlvsize = start;
  start.Next (2);
  for (ConstIterator iter = Begin (); iter != End (); iter++)
    {
      (*iter)->Serialize (start);
    }
  /* - 2 to not include the size field */
  uint16_t size = start.GetDistanceFrom (tlvsize) - 2;
  tlvsize.WriteHtonU16 (size);
}

void
PbbTlvBlock::Deserialize (Buffer::Iterator &start)
{
  NS_LOG_FUNCTION (this << &start);
  uint16_t size = start.ReadNtohU16 ();

  Buffer::Iterator tlvstart = start;
  if (size > 0)
    {
      while (start.GetDistanceFrom (tlvstart) < size)
        {
          Ptr<PbbTlv> newtlv = Create<PbbTlv> ();
          newtlv->Deserialize (start);
          PushBack (newtlv);
        }
    }
}

void
PbbTlvBlock::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  Print (os, 0);
}

void
PbbTlvBlock::Print (std::ostream &os, int level) const
{
  NS_LOG_FUNCTION (this << &os << level);
  std::string prefix = "";
  for (int i = 0; i < level; i++)
    {
      prefix.append ("\t");
    }

  os << prefix << "TLV Block {" << std::endl;
  os << prefix << "\tsize = " << Size () << std::endl;
  os << prefix << "\tmembers [" << std::endl;

  for (ConstIterator iter = Begin (); iter != End (); iter++)
    {
      (*iter)->Print (os, level+2);
    }

  os << prefix << "\t]" << std::endl;
  os << prefix << "}" << std::endl;
}

bool
PbbTlvBlock::operator== (const PbbTlvBlock &other) const
{
  if (Size () != other.Size ())
    {
      return false;
    }

  ConstIterator ti, oi;
  for (ti = Begin (), oi = other.Begin ();
       ti != End () && oi != other.End ();
       ti++, oi++)
    {
      if (**ti != **oi)
        {
          return false;
        }
    }
  return true;
}

bool
PbbTlvBlock::operator!= (const PbbTlvBlock &other) const
{
  return !(*this == other);
}

/* End PbbTlvBlock class */

PbbAddressTlvBlock::PbbAddressTlvBlock (void)
{
  NS_LOG_FUNCTION (this);
  return;
}

PbbAddressTlvBlock::~PbbAddressTlvBlock (void)
{
  NS_LOG_FUNCTION (this);
  Clear ();
}

PbbAddressTlvBlock::Iterator
PbbAddressTlvBlock::Begin (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.begin ();
}

PbbAddressTlvBlock::ConstIterator
PbbAddressTlvBlock::Begin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.begin ();
}

PbbAddressTlvBlock::Iterator
PbbAddressTlvBlock::End (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.end ();
}

PbbAddressTlvBlock::ConstIterator
PbbAddressTlvBlock::End (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.end ();
}

int
PbbAddressTlvBlock::Size (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.size ();
}

bool
PbbAddressTlvBlock::Empty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.empty ();
}

Ptr<PbbAddressTlv>
PbbAddressTlvBlock::Front (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.front ();
}

Ptr<PbbAddressTlv>
PbbAddressTlvBlock::Back (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.back ();
}

void
PbbAddressTlvBlock::PushFront (Ptr<PbbAddressTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.push_front (tlv);
}

void
PbbAddressTlvBlock::PopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.pop_front ();
}

void
PbbAddressTlvBlock::PushBack (Ptr<PbbAddressTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.push_back (tlv);
}

void
PbbAddressTlvBlock::PopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.pop_back ();
}

PbbAddressTlvBlock::Iterator
PbbAddressTlvBlock::Insert (PbbAddressTlvBlock::Iterator position, const Ptr<PbbAddressTlv> tlv)
{
  NS_LOG_FUNCTION (this << &position << tlv);
  return m_tlvList.insert (position, tlv);
}

PbbAddressTlvBlock::Iterator
PbbAddressTlvBlock::Erase (PbbAddressTlvBlock::Iterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_tlvList.erase (position);
}

PbbAddressTlvBlock::Iterator
PbbAddressTlvBlock::Erase (PbbAddressTlvBlock::Iterator first, PbbAddressTlvBlock::Iterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_tlvList.erase (first, last);
}

void
PbbAddressTlvBlock::Clear (void)
{
  NS_LOG_FUNCTION (this);
  for (Iterator iter = Begin (); iter != End (); iter++)
    {
      *iter = 0;
    }
  m_tlvList.clear ();
}

uint32_t
PbbAddressTlvBlock::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  /* tlv size */
  uint32_t size = 2;
  for (ConstIterator iter = Begin (); iter != End (); iter++)
    {
      size += (*iter)->GetSerializedSize ();
    }
  return size;
}

void
PbbAddressTlvBlock::Serialize (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  if (Empty ())
    {
      start.WriteHtonU16 (0);
      return;
    }

  /* We need to write the size of the TLV block in front, so save its
   * position. */
  Buffer::Iterator tlvsize = start;
  start.Next (2);
  for (ConstIterator iter = Begin (); iter != End (); iter++)
    {
      (*iter)->Serialize (start);
    }
  /* - 2 to not include the size field */
  uint16_t size = start.GetDistanceFrom (tlvsize) - 2;
  tlvsize.WriteHtonU16 (size);
}

void
PbbAddressTlvBlock::Deserialize (Buffer::Iterator &start)
{
  NS_LOG_FUNCTION (this << &start);
  uint16_t size = start.ReadNtohU16 ();

  Buffer::Iterator tlvstart = start;
  if (size > 0)
    {
      while (start.GetDistanceFrom (tlvstart) < size)
        {
          Ptr<PbbAddressTlv> newtlv = Create<PbbAddressTlv> ();
          newtlv->Deserialize (start);
          PushBack (newtlv);
        }
    }
}

void
PbbAddressTlvBlock::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  Print (os, 0);
}

void
PbbAddressTlvBlock::Print (std::ostream &os, int level) const
{
  NS_LOG_FUNCTION (this << &os << level);
  std::string prefix = "";
  for (int i = 0; i < level; i++)
    {
      prefix.append ("\t");
    }

  os << prefix << "TLV Block {" << std::endl;
  os << prefix << "\tsize = " << Size () << std::endl;
  os << prefix << "\tmembers [" << std::endl;

  for (ConstIterator iter = Begin (); iter != End (); iter++)
    {
      (*iter)->Print (os, level+2);
    }

  os << prefix << "\t]" << std::endl;
  os << prefix << "}" << std::endl;
}

bool
PbbAddressTlvBlock::operator== (const PbbAddressTlvBlock &other) const
{
  if (Size () != other.Size ())
    {
      return false;
    }

  ConstIterator it, ot;
  for (it = Begin (), ot = other.Begin ();
       it != End () && ot != other.End ();
       it++, ot++)
    {
      if (**it != **ot)
        {
          return false;
        }
    }
  return true;
}

bool
PbbAddressTlvBlock::operator!= (const PbbAddressTlvBlock &other) const
{
  return !(*this == other);
}


/* End PbbAddressTlvBlock Class */

PbbPacket::PbbPacket (void)
{
  NS_LOG_FUNCTION (this);
  m_version = VERSION;
  m_hasseqnum = false;
}

PbbPacket::~PbbPacket (void)
{
  NS_LOG_FUNCTION (this);
  MessageClear ();
}

uint8_t
PbbPacket::GetVersion (void) const
{
  NS_LOG_FUNCTION (this);
  return m_version;
}

void
PbbPacket::SetSequenceNumber (uint16_t number)
{
  NS_LOG_FUNCTION (this << number);
  m_seqnum = number;
  m_hasseqnum = true;
}

uint16_t
PbbPacket::GetSequenceNumber (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasSequenceNumber ());
  return m_seqnum;
}

bool
PbbPacket::HasSequenceNumber (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasseqnum;
}

/* Manipulating Packet TLVs */

PbbPacket::TlvIterator
PbbPacket::TlvBegin (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Begin ();
}

PbbPacket::ConstTlvIterator
PbbPacket::TlvBegin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Begin ();
}

PbbPacket::TlvIterator
PbbPacket::TlvEnd (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.End ();
}

PbbPacket::ConstTlvIterator
PbbPacket::TlvEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.End ();
}

int
PbbPacket::TlvSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Size ();
}

bool
PbbPacket::TlvEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Empty ();
}

Ptr<PbbTlv>
PbbPacket::TlvFront (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Front ();
}

const Ptr<PbbTlv>
PbbPacket::TlvFront (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Front ();
}

Ptr<PbbTlv>
PbbPacket::TlvBack (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Back ();
}

const Ptr<PbbTlv>
PbbPacket::TlvBack (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Back ();
}

void
PbbPacket::TlvPushFront (Ptr<PbbTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.PushFront (tlv);
}

void
PbbPacket::TlvPopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.PopFront ();
}

void
PbbPacket::TlvPushBack (Ptr<PbbTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.PushBack (tlv);
}

void
PbbPacket::TlvPopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.PopBack ();
}

PbbPacket::TlvIterator
PbbPacket::Erase (PbbPacket::TlvIterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_tlvList.Erase (position);
}

PbbPacket::TlvIterator
PbbPacket::Erase (PbbPacket::TlvIterator first, PbbPacket::TlvIterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_tlvList.Erase (first, last);
}

void
PbbPacket::TlvClear (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.Clear ();
}

/* Manipulating Packet Messages */

PbbPacket::MessageIterator
PbbPacket::MessageBegin (void)
{
  NS_LOG_FUNCTION (this);
  return m_messageList.begin ();
}

PbbPacket::ConstMessageIterator
PbbPacket::MessageBegin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_messageList.begin ();
}

PbbPacket::MessageIterator
PbbPacket::MessageEnd (void)
{
  NS_LOG_FUNCTION (this);
  return m_messageList.end ();
}

PbbPacket::ConstMessageIterator
PbbPacket::MessageEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_messageList.end ();
}

int
PbbPacket::MessageSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_messageList.size ();
}

bool
PbbPacket::MessageEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_messageList.empty ();
}

Ptr<PbbMessage>
PbbPacket::MessageFront (void)
{
  NS_LOG_FUNCTION (this);
  return m_messageList.front ();
}

const Ptr<PbbMessage>
PbbPacket::MessageFront (void) const
{
  NS_LOG_FUNCTION (this);
  return m_messageList.front ();
}

Ptr<PbbMessage>
PbbPacket::MessageBack (void)
{
  NS_LOG_FUNCTION (this);
  return m_messageList.back ();
}

const Ptr<PbbMessage>
PbbPacket::MessageBack (void) const
{
  NS_LOG_FUNCTION (this);
  return m_messageList.back ();
}

void
PbbPacket::MessagePushFront (Ptr<PbbMessage> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_messageList.push_front (tlv);
}

void
PbbPacket::MessagePopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_messageList.pop_front ();
}

void
PbbPacket::MessagePushBack (Ptr<PbbMessage> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_messageList.push_back (tlv);
}

void
PbbPacket::MessagePopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_messageList.pop_back ();
}

PbbPacket::MessageIterator
PbbPacket::Erase (PbbPacket::MessageIterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_messageList.erase (position);
}

PbbPacket::MessageIterator
PbbPacket::Erase (PbbPacket::MessageIterator first,
                  PbbPacket::MessageIterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_messageList.erase (first, last);
}

void
PbbPacket::MessageClear (void)
{
  NS_LOG_FUNCTION (this);
  for (MessageIterator iter = MessageBegin (); iter != MessageEnd (); iter++)
    {
      *iter = 0;
    }
  m_messageList.clear ();
}


TypeId
PbbPacket::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PbbPacket")
    .SetParent<Header> ()
    .SetGroupName("Network")
    .AddConstructor<PbbPacket> ()
  ;
  return tid;
}

TypeId
PbbPacket::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
PbbPacket::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  /* Version number + flags */
  uint32_t size = 1;

  if (HasSequenceNumber ())
    {
      size += 2;
    }

  if (!TlvEmpty ())
    {
      size += m_tlvList.GetSerializedSize ();
    }

  for (ConstMessageIterator iter = MessageBegin ();
       iter != MessageEnd ();
       iter++)
    {
      size += (*iter)->GetSerializedSize ();
    }

  return size;
}

void
PbbPacket::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  /* We remember the start, so we can write the flags after we check for a
   * sequence number and TLV. */
  Buffer::Iterator bufref = start;
  start.Next ();

  uint8_t flags = VERSION;
  /* Make room for 4 bit flags */
  flags <<= 4;

  if (HasSequenceNumber ())
    {
      flags |= PHAS_SEQ_NUM;
      start.WriteHtonU16 (GetSequenceNumber ());
    }

  if (!TlvEmpty ())
    {
      flags |= PHAS_TLV;
      m_tlvList.Serialize (start);
    }

  bufref.WriteU8 (flags);

  for (ConstMessageIterator iter = MessageBegin ();
       iter != MessageEnd ();
       iter++)
    {
      (*iter)->Serialize (start);
    }
}

uint32_t
PbbPacket::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator begin = start;

  uint8_t flags = start.ReadU8 ();

  if (flags & PHAS_SEQ_NUM)
    {
      SetSequenceNumber (start.ReadNtohU16 ());
    }

  if (flags & PHAS_TLV)
    {
      m_tlvList.Deserialize (start);
    }

  while (!start.IsEnd ())
    {
      Ptr<PbbMessage> newmsg = PbbMessage::DeserializeMessage (start);
      if (newmsg == 0)
        {
          return start.GetDistanceFrom (begin);
        }
      MessagePushBack (newmsg);
    }

  flags >>= 4;
  m_version = flags;

  return start.GetDistanceFrom (begin);
}

void
PbbPacket::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "PbbPacket {" << std::endl;

  if (HasSequenceNumber ())
    {
      os << "\tsequence number = " << GetSequenceNumber ();
    }

  os << std::endl;

  m_tlvList.Print (os, 1);

  for (ConstMessageIterator iter = MessageBegin ();
       iter != MessageEnd ();
       iter++)
    {
      (*iter)->Print (os, 1);
    }

  os << "}" << std::endl;
}

bool
PbbPacket::operator== (const PbbPacket &other) const
{
  if (GetVersion () != other.GetVersion ())
    {
      return false;
    }

  if (HasSequenceNumber () != other.HasSequenceNumber ())
    {
      return false;
    }

  if (HasSequenceNumber ())
    {
      if (GetSequenceNumber () != other.GetSequenceNumber ())
        return false;
    }

  if (m_tlvList != other.m_tlvList)
    {
      return false;
    }

  if (MessageSize () != other.MessageSize ())
    {
      return false;
    }

  ConstMessageIterator tmi, omi;
  for (tmi = MessageBegin (), omi = other.MessageBegin ();
       tmi != MessageEnd () && omi != other.MessageEnd ();
       tmi++, omi++)
    {
      if (**tmi != **omi)
        {
          return false;
        }
    }
  return true;
}

bool
PbbPacket::operator!= (const PbbPacket &other) const
{
  return !(*this == other);
}

/* End PbbPacket class */

PbbMessage::PbbMessage ()
{
  NS_LOG_FUNCTION (this);
  /* Default to IPv4 */
  m_addrSize = IPV4;
  m_hasOriginatorAddress = false;
  m_hasHopLimit = false;
  m_hasHopCount = false;
  m_hasSequenceNumber = false;
}

PbbMessage::~PbbMessage ()
{
  NS_LOG_FUNCTION (this);
  AddressBlockClear ();
}

void
PbbMessage::SetType (uint8_t type)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (type));
  m_type = type;
}

uint8_t
PbbMessage::GetType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_type;
}

PbbAddressLength
PbbMessage::GetAddressLength (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addrSize;
}

void
PbbMessage::SetOriginatorAddress (Address address)
{
  NS_LOG_FUNCTION (this << address);
  m_originatorAddress = address;
  m_hasOriginatorAddress = true;
}

Address
PbbMessage::GetOriginatorAddress (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasOriginatorAddress ());
  return m_originatorAddress;
}

bool
PbbMessage::HasOriginatorAddress (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasOriginatorAddress;
}

void
PbbMessage::SetHopLimit (uint8_t hopLimit)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (hopLimit));
  m_hopLimit = hopLimit;
  m_hasHopLimit = true;
}

uint8_t
PbbMessage::GetHopLimit (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasHopLimit ());
  return m_hopLimit;
}

bool
PbbMessage::HasHopLimit (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasHopLimit;
}

void
PbbMessage::SetHopCount (uint8_t hopCount)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (hopCount));
  m_hopCount = hopCount;
  m_hasHopCount = true;
}

uint8_t
PbbMessage::GetHopCount (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasHopCount ());
  return m_hopCount;
}

bool
PbbMessage::HasHopCount (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasHopCount;
}

void
PbbMessage::SetSequenceNumber (uint16_t sequenceNumber)
{
  NS_LOG_FUNCTION (this << sequenceNumber);
  m_sequenceNumber = sequenceNumber;
  m_hasSequenceNumber = true;
}

uint16_t
PbbMessage::GetSequenceNumber (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasSequenceNumber ());
  return m_sequenceNumber;
}

bool
PbbMessage::HasSequenceNumber (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasSequenceNumber;
}

/* Manipulating PbbMessage TLVs */

PbbMessage::TlvIterator
PbbMessage::TlvBegin (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Begin ();
}

PbbMessage::ConstTlvIterator
PbbMessage::TlvBegin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Begin ();
}

PbbMessage::TlvIterator
PbbMessage::TlvEnd (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.End ();
}

PbbMessage::ConstTlvIterator
PbbMessage::TlvEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.End ();
}

int
PbbMessage::TlvSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Size ();
}

bool
PbbMessage::TlvEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Empty ();
}

Ptr<PbbTlv>
PbbMessage::TlvFront (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Front ();
}

const Ptr<PbbTlv>
PbbMessage::TlvFront (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Front ();
}

Ptr<PbbTlv>
PbbMessage::TlvBack (void)
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Back ();
}

const Ptr<PbbTlv>
PbbMessage::TlvBack (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tlvList.Back ();
}

void
PbbMessage::TlvPushFront (Ptr<PbbTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.PushFront (tlv);
}

void
PbbMessage::TlvPopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.PopFront ();
}

void
PbbMessage::TlvPushBack (Ptr<PbbTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_tlvList.PushBack (tlv);
}

void
PbbMessage::TlvPopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.PopBack ();
}

PbbMessage::TlvIterator
PbbMessage::TlvErase (PbbMessage::TlvIterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_tlvList.Erase (position);
}

PbbMessage::TlvIterator
PbbMessage::TlvErase (PbbMessage::TlvIterator first, PbbMessage::TlvIterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_tlvList.Erase (first, last);
}

void
PbbMessage::TlvClear (void)
{
  NS_LOG_FUNCTION (this);
  m_tlvList.Clear ();
}

/* Manipulating Address Block and Address TLV pairs */

PbbMessage::AddressBlockIterator
PbbMessage::AddressBlockBegin (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.begin ();
}

PbbMessage::ConstAddressBlockIterator
PbbMessage::AddressBlockBegin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.begin ();
}

PbbMessage::AddressBlockIterator
PbbMessage::AddressBlockEnd (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.end ();
}

PbbMessage::ConstAddressBlockIterator
PbbMessage::AddressBlockEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.end ();
}

int
PbbMessage::AddressBlockSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.size ();
}

bool
PbbMessage::AddressBlockEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.empty ();
}

Ptr<PbbAddressBlock>
PbbMessage::AddressBlockFront (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.front ();
}

const Ptr<PbbAddressBlock>
PbbMessage::AddressBlockFront (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.front ();
}

Ptr<PbbAddressBlock>
PbbMessage::AddressBlockBack (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.back ();
}

const Ptr<PbbAddressBlock>
PbbMessage::AddressBlockBack (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressBlockList.back ();
}

void
PbbMessage::AddressBlockPushFront (Ptr<PbbAddressBlock> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_addressBlockList.push_front (tlv);
}

void
PbbMessage::AddressBlockPopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_addressBlockList.pop_front ();
}

void
PbbMessage::AddressBlockPushBack (Ptr<PbbAddressBlock> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_addressBlockList.push_back (tlv);
}

void
PbbMessage::AddressBlockPopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_addressBlockList.pop_back ();
}

PbbMessage::AddressBlockIterator
PbbMessage::AddressBlockErase (PbbMessage::AddressBlockIterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_addressBlockList.erase (position);
}

PbbMessage::AddressBlockIterator
PbbMessage::AddressBlockErase (PbbMessage::AddressBlockIterator first,
                               PbbMessage::AddressBlockIterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_addressBlockList.erase (first, last);
}

void
PbbMessage::AddressBlockClear (void)
{
  NS_LOG_FUNCTION (this);
  for (AddressBlockIterator iter = AddressBlockBegin ();
       iter != AddressBlockEnd ();
       iter++)
    {
      *iter = 0;
    }
  return m_addressBlockList.clear ();
}

uint32_t
PbbMessage::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  /* msg-type + (msg-flags + msg-addr-length) + 2msg-size */
  uint32_t size = 4;

  if (HasOriginatorAddress ())
    {
      size += GetAddressLength () + 1;
    }

  if (HasHopLimit ())
    {
      size++;
    }

  if (HasHopCount ())
    {
      size++;
    }

  if (HasSequenceNumber ())
    {
      size += 2;
    }

  size += m_tlvList.GetSerializedSize ();

  for (ConstAddressBlockIterator iter = AddressBlockBegin ();
       iter != AddressBlockEnd ();
       iter++)
    {
      size += (*iter)->GetSerializedSize ();
    }

  return size;
}

void
PbbMessage::Serialize (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator front = start;

  start.WriteU8 (GetType ());

  /* Save a reference to the spot where we will later write the flags */
  Buffer::Iterator bufref = start;
  start.Next (1);

  uint8_t flags = 0;

  flags = GetAddressLength ();

  Buffer::Iterator sizeref = start;
  start.Next (2);

  if (HasOriginatorAddress ())
    {
      flags |= MHAS_ORIG;
      SerializeOriginatorAddress (start);
    }

  if (HasHopLimit ())
    {
      flags |= MHAS_HOP_LIMIT;
      start.WriteU8 (GetHopLimit ());
    }

  if (HasHopCount ())
    {
      flags |= MHAS_HOP_COUNT;
      start.WriteU8 (GetHopCount ());
    }

  if (HasSequenceNumber ())
    {
      flags |= MHAS_SEQ_NUM;
      start.WriteHtonU16 (GetSequenceNumber ());
    }

  bufref.WriteU8 (flags);

  m_tlvList.Serialize (start);

  for (ConstAddressBlockIterator iter = AddressBlockBegin ();
       iter != AddressBlockEnd ();
       iter++)
    {
      (*iter)->Serialize (start);
    }

  sizeref.WriteHtonU16 (front.GetDistanceFrom (start));
}

Ptr<PbbMessage>
PbbMessage::DeserializeMessage (Buffer::Iterator &start)
{
  NS_LOG_FUNCTION (&start);
  /* We need to read the msg-addr-len field to determine what kind of object to
   * construct. */
  start.Next ();
  uint8_t addrlen = start.ReadU8 ();
  start.Prev (2); /* Go back to the start */

  /* The first four bytes of the flag is the address length.  Set the last four
   * bytes to 0 to read it. */
  addrlen = (addrlen & 0xf);

  Ptr<PbbMessage> newmsg;

  switch (addrlen)
    {
    case 0:
    case IPV4:
      newmsg = Create<PbbMessageIpv4> ();
      break;
    case IPV6:
      newmsg = Create<PbbMessageIpv6> ();
      break;
    default:
      return 0;
      break;
    }
  newmsg->Deserialize (start);
  return newmsg;
}

void
PbbMessage::Deserialize (Buffer::Iterator &start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator front = start;
  SetType (start.ReadU8 ());
  uint8_t flags = start.ReadU8 ();

  uint16_t size = start.ReadNtohU16 ();

  if (flags & MHAS_ORIG)
    {
      SetOriginatorAddress (DeserializeOriginatorAddress (start));
    }

  if (flags & MHAS_HOP_LIMIT)
    {
      SetHopLimit (start.ReadU8 ());
    }

  if (flags & MHAS_HOP_COUNT)
    {
      SetHopCount (start.ReadU8 ());
    }

  if (flags & MHAS_SEQ_NUM)
    {
      SetSequenceNumber (start.ReadNtohU16 ());
    }

  m_tlvList.Deserialize (start);

  if (size > 0)
    {
      while (start.GetDistanceFrom (front) < size)
        {
          Ptr<PbbAddressBlock> newab = AddressBlockDeserialize (start);
          AddressBlockPushBack (newab);
        }
    }
}

void
PbbMessage::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  Print (os, 0);
}

void
PbbMessage::Print (std::ostream &os, int level) const
{
  NS_LOG_FUNCTION (this << &os << level);
  std::string prefix = "";
  for (int i = 0; i < level; i++)
    {
      prefix.append ("\t");
    }

  os << prefix << "PbbMessage {" << std::endl;

  os << prefix << "\tmessage type = " << (int)GetType () << std::endl;
  os << prefix << "\taddress size = " << GetAddressLength () << std::endl;

  if (HasOriginatorAddress ())
    {
      os << prefix << "\toriginator address = ";
      PrintOriginatorAddress (os);
      os << std::endl;
    }

  if (HasHopLimit ())
    {
      os << prefix << "\thop limit = " << (int)GetHopLimit () << std::endl;
    }

  if (HasHopCount ())
    {
      os << prefix << "\thop count = " << (int)GetHopCount () << std::endl;
    }

  if (HasSequenceNumber ())
    {
      os << prefix << "\tseqnum = " << GetSequenceNumber () << std::endl;
    }

  m_tlvList.Print (os, level+1);

  for (ConstAddressBlockIterator iter = AddressBlockBegin ();
       iter != AddressBlockEnd ();
       iter++)
    {
      (*iter)->Print (os, level+1);
    }
  os << prefix << "}" << std::endl;
}

bool
PbbMessage::operator== (const PbbMessage &other) const
{
  if (GetAddressLength () != other.GetAddressLength ())
    {
      return false;
    }

  if (GetType () != other.GetType ())
    {
      return false;
    }

  if (HasOriginatorAddress () != other.HasOriginatorAddress ())
    {
      return false;
    }

  if (HasOriginatorAddress ())
    {
      if (GetOriginatorAddress () != other.GetOriginatorAddress ())
        {
          return false;
        }
    }

  if (HasHopLimit () != other.HasHopLimit ())
    {
      return false;
    }

  if (HasHopLimit ())
    {
      if (GetHopLimit () != other.GetHopLimit ())
        {
          return false;
        }
    }

  if (HasHopCount () != other.HasHopCount ())
    {
      return false;
    }

  if (HasHopCount ())
    {
      if (GetHopCount () != other.GetHopCount ())
        {
          return false;
        }
    }

  if (HasSequenceNumber () != other.HasSequenceNumber ())
    {
      return false;
    }

  if (HasSequenceNumber ())
    {
      if (GetSequenceNumber () != other.GetSequenceNumber ())
        {
          return false;
        }
    }

  if (m_tlvList != other.m_tlvList)
    {
      return false;
    }

  if (AddressBlockSize () != other.AddressBlockSize ())
    {
      return false;
    }

  ConstAddressBlockIterator tai, oai;
  for (tai = AddressBlockBegin (), oai = other.AddressBlockBegin ();
       tai != AddressBlockEnd () && oai != other.AddressBlockEnd ();
       tai++, oai++)
    {
      if (**tai != **oai)
        {
          return false;
        }
    }
  return true;
}

bool
PbbMessage::operator!= (const PbbMessage &other) const
{
  return !(*this == other);
}

/* End PbbMessage Class */

PbbMessageIpv4::PbbMessageIpv4 ()
{
  NS_LOG_FUNCTION (this);
}

PbbMessageIpv4::~PbbMessageIpv4 ()
{
  NS_LOG_FUNCTION (this);
}

PbbAddressLength
PbbMessageIpv4::GetAddressLength (void) const
{
  NS_LOG_FUNCTION (this);
  return IPV4;
}

void
PbbMessageIpv4::SerializeOriginatorAddress (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  uint8_t* buffer = new uint8_t[GetAddressLength () + 1];
  Ipv4Address::ConvertFrom (GetOriginatorAddress ()).Serialize (buffer);
  start.Write (buffer, GetAddressLength () + 1);
  delete[] buffer;
}

Address
PbbMessageIpv4::DeserializeOriginatorAddress (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  uint8_t* buffer = new uint8_t[GetAddressLength () + 1];
  start.Read (buffer, GetAddressLength () + 1);
  Address result = Ipv4Address::Deserialize (buffer);
  delete[] buffer;
  return result;
}

void
PbbMessageIpv4::PrintOriginatorAddress (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  Ipv4Address::ConvertFrom (GetOriginatorAddress ()).Print (os);
}

Ptr<PbbAddressBlock>
PbbMessageIpv4::AddressBlockDeserialize (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  Ptr<PbbAddressBlock> newab = Create<PbbAddressBlockIpv4> ();
  newab->Deserialize (start);
  return newab;
}

/* End PbbMessageIpv4 Class */

PbbMessageIpv6::PbbMessageIpv6 ()
{
  NS_LOG_FUNCTION (this);
}

PbbMessageIpv6::~PbbMessageIpv6 ()
{
  NS_LOG_FUNCTION (this);
}

PbbAddressLength
PbbMessageIpv6::GetAddressLength (void) const
{
  NS_LOG_FUNCTION (this);
  return IPV6;
}

void
PbbMessageIpv6::SerializeOriginatorAddress (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  uint8_t* buffer = new uint8_t[GetAddressLength () + 1];
  Ipv6Address::ConvertFrom (GetOriginatorAddress ()).Serialize (buffer);
  start.Write (buffer, GetAddressLength () + 1);
  delete[] buffer;
}

Address
PbbMessageIpv6::DeserializeOriginatorAddress (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  uint8_t* buffer = new uint8_t[GetAddressLength () + 1];
  start.Read (buffer, GetAddressLength () + 1);
  Address res = Ipv6Address::Deserialize (buffer);
  delete[] buffer;
  return res;
}

void
PbbMessageIpv6::PrintOriginatorAddress (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  Ipv6Address::ConvertFrom (GetOriginatorAddress ()).Print (os);
}

Ptr<PbbAddressBlock>
PbbMessageIpv6::AddressBlockDeserialize (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  Ptr<PbbAddressBlock> newab = Create<PbbAddressBlockIpv6> ();
  newab->Deserialize (start);
  return newab;
}

/* End PbbMessageIpv6 Class */

PbbAddressBlock::PbbAddressBlock ()
{
  NS_LOG_FUNCTION (this);
}

PbbAddressBlock::~PbbAddressBlock ()
{
  NS_LOG_FUNCTION (this);
}

/* Manipulating the address block */

PbbAddressBlock::AddressIterator
PbbAddressBlock::AddressBegin (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressList.begin ();
}

PbbAddressBlock::ConstAddressIterator
PbbAddressBlock::AddressBegin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressList.begin ();
}

PbbAddressBlock::AddressIterator
PbbAddressBlock::AddressEnd (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressList.end ();
}

PbbAddressBlock::ConstAddressIterator
PbbAddressBlock::AddressEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressList.end ();
}

int
PbbAddressBlock::AddressSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressList.size ();
}

bool
PbbAddressBlock::AddressEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressList.empty ();
}

Address
PbbAddressBlock::AddressFront (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressList.front ();
}

Address
PbbAddressBlock::AddressBack (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressList.back ();
}

void
PbbAddressBlock::AddressPushFront (Address tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_addressList.push_front (tlv);
}

void
PbbAddressBlock::AddressPopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_addressList.pop_front ();
}

void
PbbAddressBlock::AddressPushBack (Address tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_addressList.push_back (tlv);
}

void
PbbAddressBlock::AddressPopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_addressList.pop_back ();
}

PbbAddressBlock::AddressIterator
PbbAddressBlock::AddressErase (PbbAddressBlock::AddressIterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_addressList.erase (position);
}

PbbAddressBlock::AddressIterator
PbbAddressBlock::AddressErase (PbbAddressBlock::AddressIterator first,
                               PbbAddressBlock::AddressIterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_addressList.erase (first, last);
}

void
PbbAddressBlock::AddressClear (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressList.clear ();
}

/* Manipulating the prefix list */

PbbAddressBlock::PrefixIterator
PbbAddressBlock::PrefixBegin (void)
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.begin ();
}

PbbAddressBlock::ConstPrefixIterator
PbbAddressBlock::PrefixBegin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.begin ();
}

PbbAddressBlock::PrefixIterator
PbbAddressBlock::PrefixEnd (void)
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.end ();
}

PbbAddressBlock::ConstPrefixIterator
PbbAddressBlock::PrefixEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.end ();
}

int
PbbAddressBlock::PrefixSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.size ();
}

bool
PbbAddressBlock::PrefixEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.empty ();
}

uint8_t
PbbAddressBlock::PrefixFront (void) const
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.front ();
}

uint8_t
PbbAddressBlock::PrefixBack (void) const
{
  NS_LOG_FUNCTION (this);
  return m_prefixList.back ();
}

void
PbbAddressBlock::PrefixPushFront (uint8_t prefix)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (prefix));
  m_prefixList.push_front (prefix);
}

void
PbbAddressBlock::PrefixPopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_prefixList.pop_front ();
}

void
PbbAddressBlock::PrefixPushBack (uint8_t prefix)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (prefix));
  m_prefixList.push_back (prefix);
}

void
PbbAddressBlock::PrefixPopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_prefixList.pop_back ();
}

PbbAddressBlock::PrefixIterator
PbbAddressBlock::PrefixInsert (PbbAddressBlock::PrefixIterator position, const uint8_t value)
{
  NS_LOG_FUNCTION (this << &position << static_cast<uint32_t> (value));
  return m_prefixList.insert (position, value);
}

PbbAddressBlock::PrefixIterator
PbbAddressBlock::PrefixErase (PbbAddressBlock::PrefixIterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_prefixList.erase (position);
}

PbbAddressBlock::PrefixIterator
PbbAddressBlock::PrefixErase (PbbAddressBlock::PrefixIterator first, PbbAddressBlock::PrefixIterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_prefixList.erase (first, last);
}

void
PbbAddressBlock::PrefixClear (void)
{
  NS_LOG_FUNCTION (this);
  m_prefixList.clear ();
}

/* Manipulating the TLV block */

PbbAddressBlock::TlvIterator
PbbAddressBlock::TlvBegin (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Begin ();
}

PbbAddressBlock::ConstTlvIterator
PbbAddressBlock::TlvBegin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Begin ();
}

PbbAddressBlock::TlvIterator
PbbAddressBlock::TlvEnd (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.End ();
}

PbbAddressBlock::ConstTlvIterator
PbbAddressBlock::TlvEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.End ();
}

int
PbbAddressBlock::TlvSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Size ();
}

bool
PbbAddressBlock::TlvEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Empty ();
}

Ptr<PbbAddressTlv>
PbbAddressBlock::TlvFront (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Front ();
}

const Ptr<PbbAddressTlv>
PbbAddressBlock::TlvFront (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Front ();
}

Ptr<PbbAddressTlv>
PbbAddressBlock::TlvBack (void)
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Back ();
}

const Ptr<PbbAddressTlv>
PbbAddressBlock::TlvBack (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addressTlvList.Back ();
}

void
PbbAddressBlock::TlvPushFront (Ptr<PbbAddressTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_addressTlvList.PushFront (tlv);
}

void
PbbAddressBlock::TlvPopFront (void)
{
  NS_LOG_FUNCTION (this);
  m_addressTlvList.PopFront ();
}

void
PbbAddressBlock::TlvPushBack (Ptr<PbbAddressTlv> tlv)
{
  NS_LOG_FUNCTION (this << tlv);
  m_addressTlvList.PushBack (tlv);
}

void
PbbAddressBlock::TlvPopBack (void)
{
  NS_LOG_FUNCTION (this);
  m_addressTlvList.PopBack ();
}

PbbAddressBlock::TlvIterator
PbbAddressBlock::TlvErase (PbbAddressBlock::TlvIterator position)
{
  NS_LOG_FUNCTION (this << &position);
  return m_addressTlvList.Erase (position);
}

PbbAddressBlock::TlvIterator
PbbAddressBlock::TlvErase (PbbAddressBlock::TlvIterator first,
                           PbbAddressBlock::TlvIterator last)
{
  NS_LOG_FUNCTION (this << &first << &last);
  return m_addressTlvList.Erase (first, last);
}

void
PbbAddressBlock::TlvClear (void)
{
  NS_LOG_FUNCTION (this);
  m_addressTlvList.Clear ();
}
uint32_t
PbbAddressBlock::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  /* num-addr + flags */
  uint32_t size = 2;

  if (AddressSize () == 1)
    {
      size += GetAddressLength () + PrefixSize ();
    }
  else if (AddressSize () > 0)
    {
      uint8_t* head = new uint8_t[GetAddressLength ()];
      uint8_t headlen = 0;
      uint8_t* tail = new uint8_t[GetAddressLength ()];
      uint8_t taillen = 0;

      GetHeadTail (head, headlen, tail, taillen);

      if (headlen > 0)
        {
          size += 1 + headlen;
        }

      if (taillen > 0)
        {
          size++;
          if (!HasZeroTail (tail, taillen))
            {
              size += taillen;
            }
        }

      /* mid size */
      size += (GetAddressLength () - headlen - taillen) * AddressSize ();

      size += PrefixSize ();

      delete[] head;
      delete[] tail;
    }

  size += m_addressTlvList.GetSerializedSize ();

  return size;
}

void
PbbAddressBlock::Serialize (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  start.WriteU8 (AddressSize ());
  Buffer::Iterator bufref = start;
  uint8_t flags = 0;
  start.Next ();

  if (AddressSize () == 1)
    {
      uint8_t* buf = new uint8_t[GetAddressLength ()];
      SerializeAddress (buf, AddressBegin ());
      start.Write (buf, GetAddressLength ());

      if (PrefixSize () == 1)
        {
          start.WriteU8 (PrefixFront ());
          flags |= AHAS_SINGLE_PRE_LEN;
        }
      bufref.WriteU8 (flags);
      delete[] buf;
    }
  else if (AddressSize () > 0)
    {
      uint8_t* head = new uint8_t[GetAddressLength ()];
      uint8_t* tail = new uint8_t[GetAddressLength ()];
      uint8_t headlen = 0;
      uint8_t taillen = 0;

      GetHeadTail (head, headlen, tail, taillen);

      if (headlen > 0)
        {
          flags |= AHAS_HEAD;
          start.WriteU8 (headlen);
          start.Write (head, headlen);
        }

      if (taillen > 0)
        {
          start.WriteU8 (taillen);

          if (HasZeroTail (tail, taillen))
            {
              flags |= AHAS_ZERO_TAIL;
            }
          else
            {
              flags |= AHAS_FULL_TAIL;
              start.Write (tail, taillen);
            }
        }

      if (headlen + taillen < GetAddressLength ())
        {
          uint8_t* mid = new uint8_t[GetAddressLength ()];
          for (PbbAddressBlock::ConstAddressIterator iter = AddressBegin ();
               iter != AddressEnd ();
               iter++)
            {
              SerializeAddress (mid, iter);
              start.Write (mid + headlen, GetAddressLength () - headlen - taillen);
            }
          delete[] mid;
        }

      flags |= GetPrefixFlags ();
      bufref.WriteU8 (flags);

      for (ConstPrefixIterator iter = PrefixBegin ();
           iter != PrefixEnd ();
           iter++)
        {
          start.WriteU8 (*iter);
        }

      delete[] head;
      delete[] tail;
    }

  m_addressTlvList.Serialize (start);
}

void
PbbAddressBlock::Deserialize (Buffer::Iterator &start)
{
  NS_LOG_FUNCTION (this << &start);
  uint8_t numaddr = start.ReadU8 ();
  uint8_t flags = start.ReadU8 ();

  if (numaddr > 0)
    {
      uint8_t headlen = 0;
      uint8_t taillen = 0;
      uint8_t* addrtmp = new uint8_t[GetAddressLength ()];
      memset (addrtmp, 0, GetAddressLength ());

      if (flags & AHAS_HEAD)
        {
          headlen = start.ReadU8 ();
          start.Read (addrtmp, headlen);
        }

      if ((flags & AHAS_FULL_TAIL) ^ (flags & AHAS_ZERO_TAIL))
        {
          taillen = start.ReadU8 ();

          if (flags & AHAS_FULL_TAIL)
            {
              start.Read (addrtmp + GetAddressLength () - taillen, taillen);
            }
        }

      for (int i = 0; i < numaddr; i++)
        {
          start.Read (addrtmp + headlen, GetAddressLength () - headlen - taillen);
          AddressPushBack (DeserializeAddress (addrtmp));
        }

      if (flags & AHAS_SINGLE_PRE_LEN)
        {
          PrefixPushBack (start.ReadU8 ());
        }
      else if (flags & AHAS_MULTI_PRE_LEN)
        {
          for (int i = 0; i < numaddr; i++)
            {
              PrefixPushBack (start.ReadU8 ());
            }
        }

      delete[] addrtmp;
    }

  m_addressTlvList.Deserialize (start);
}

void
PbbAddressBlock::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  Print (os, 0);
}

void
PbbAddressBlock::Print (std::ostream &os, int level) const
{
  NS_LOG_FUNCTION (this << &os << level);
  std::string prefix = "";
  for (int i = 0; i < level; i++)
    {
      prefix.append ("\t");
    }

  os << prefix << "PbbAddressBlock {" << std::endl;
  os << prefix << "\taddresses = " << std::endl;
  for (ConstAddressIterator iter = AddressBegin ();
       iter != AddressEnd ();
       iter++)
    {
      os << prefix << "\t\t";
      PrintAddress (os, iter);
      os << std::endl;
    }

  os << prefix << "\tprefixes = " << std::endl;
  for (ConstPrefixIterator iter = PrefixBegin ();
       iter != PrefixEnd ();
       iter++)
    {
      os << prefix << "\t\t" << (int)(*iter) << std::endl;
    }

  m_addressTlvList.Print (os, level+1);
}

bool
PbbAddressBlock::operator== (const PbbAddressBlock &other) const
{
  if (AddressSize () != other.AddressSize ())
    {
      return false;
    }

  ConstAddressIterator tai, oai;
  for (tai = AddressBegin (), oai = other.AddressBegin ();
       tai != AddressEnd () && oai != other.AddressEnd ();
       tai++, oai++)
    {
      if (*tai != *oai)
        {
          return false;
        }
    }

  if (PrefixSize () != other.PrefixSize ())
    {
      return false;
    }

  ConstPrefixIterator tpi, opi;
  for (tpi = PrefixBegin (), opi = other.PrefixBegin ();
       tpi != PrefixEnd () && opi != other.PrefixEnd ();
       tpi++, opi++)
    {
      if (*tpi != *opi)
        {
          return false;
        }
    }

  if (m_addressTlvList != other.m_addressTlvList)
    {
      return false;
    }

  return true;
}

bool
PbbAddressBlock::operator!= (const PbbAddressBlock &other) const
{
  return !(*this == other);
}

uint8_t
PbbAddressBlock::GetPrefixFlags (void) const
{
  NS_LOG_FUNCTION (this);
  switch (PrefixSize ())
    {
    case 0:
      return 0;
      break;
    case 1:
      return AHAS_SINGLE_PRE_LEN;
      break;
    default:
      return AHAS_MULTI_PRE_LEN;
      break;
    }

  /* Quiet compiler */
  return 0;
}

void
PbbAddressBlock::GetHeadTail (uint8_t *head, uint8_t &headlen,
                              uint8_t *tail, uint8_t &taillen) const
{
  NS_LOG_FUNCTION (this << &head << static_cast<uint32_t> (headlen)
                   << &tail << static_cast<uint32_t> (taillen));
  headlen = GetAddressLength ();
  taillen = headlen;

  /* Temporary automatic buffers to store serialized addresses */
  uint8_t * buflast = new uint8_t[GetAddressLength ()];
  uint8_t * bufcur = new uint8_t[GetAddressLength ()];
  uint8_t * tmp;

  SerializeAddress (buflast, AddressBegin ());

  /* Skip the first item */
  for (PbbAddressBlock::ConstAddressIterator iter = AddressBegin ()++;
       iter != AddressEnd ();
       iter++)
    {
      SerializeAddress (bufcur, iter);

      int i;
      for (i = 0; i < headlen; i++)
        {
          if (buflast[i] != bufcur[i])
            {
              headlen = i;
              break;
            }
        }

      /* If headlen == fulllen - 1, then tail is 0 */
      if (GetAddressLength () - headlen > 0)
        {
          for (i = GetAddressLength () - 1;
               GetAddressLength () - 1 - i <= taillen && i > headlen;
               i--)
            {
              if (buflast[i] != bufcur[i])
                {
                  break;
                }
            }
          taillen = GetAddressLength () - 1 - i;
        }
      else if (headlen == 0)
        {
          taillen = 0;
          break;
        }

      tmp = buflast;
      buflast = bufcur;
      bufcur = tmp;
    }

  memcpy (head, bufcur, headlen);
  memcpy (tail, bufcur + (GetAddressLength () - taillen), taillen);

  delete[] buflast;
  delete[] bufcur;
}

bool
PbbAddressBlock::HasZeroTail (const uint8_t *tail, uint8_t taillen) const
{
  NS_LOG_FUNCTION (this << &tail << static_cast<uint32_t> (taillen));
  int i;
  for (i = 0; i < taillen; i++)
    {
      if (tail[i] != 0)
        {
          break;
        }
    }
  return i == taillen;
}

/* End PbbAddressBlock Class */

PbbAddressBlockIpv4::PbbAddressBlockIpv4 ()
{
  NS_LOG_FUNCTION (this);
}

PbbAddressBlockIpv4::~PbbAddressBlockIpv4 ()
{
  NS_LOG_FUNCTION (this);
}

uint8_t
PbbAddressBlockIpv4::GetAddressLength (void) const
{
  NS_LOG_FUNCTION (this);
  return 4;
}

void
PbbAddressBlockIpv4::SerializeAddress (uint8_t *buffer, ConstAddressIterator iter) const
{
  NS_LOG_FUNCTION (this << &buffer << &iter);
  Ipv4Address::ConvertFrom (*iter).Serialize (buffer);
}

Address
PbbAddressBlockIpv4::DeserializeAddress (uint8_t *buffer) const
{
  NS_LOG_FUNCTION (this << &buffer);
  return Ipv4Address::Deserialize (buffer);
}

void
PbbAddressBlockIpv4::PrintAddress (std::ostream &os, ConstAddressIterator iter) const
{
  NS_LOG_FUNCTION (this << &os << &iter);
  Ipv4Address::ConvertFrom (*iter).Print (os);
}

/* End PbbAddressBlockIpv4 Class */

PbbAddressBlockIpv6::PbbAddressBlockIpv6 ()
{
  NS_LOG_FUNCTION (this);
}

PbbAddressBlockIpv6::~PbbAddressBlockIpv6 ()
{
  NS_LOG_FUNCTION (this);
}

uint8_t
PbbAddressBlockIpv6::GetAddressLength (void) const
{
  NS_LOG_FUNCTION (this);
  return 16;
}

void
PbbAddressBlockIpv6::SerializeAddress (uint8_t *buffer, ConstAddressIterator iter) const
{
  NS_LOG_FUNCTION (this << &buffer << &iter);
  Ipv6Address::ConvertFrom (*iter).Serialize (buffer);
}

Address
PbbAddressBlockIpv6::DeserializeAddress (uint8_t *buffer) const
{
  NS_LOG_FUNCTION (this << &buffer);
  return Ipv6Address::Deserialize (buffer);
}

void
PbbAddressBlockIpv6::PrintAddress (std::ostream &os, ConstAddressIterator iter) const
{
  NS_LOG_FUNCTION (this << &os << &iter);
  Ipv6Address::ConvertFrom (*iter).Print (os);
}

/* End PbbAddressBlockIpv6 Class */

PbbTlv::PbbTlv (void)
{
  NS_LOG_FUNCTION (this);
  m_hasTypeExt = false;
  m_hasIndexStart = false;
  m_hasIndexStop = false;
  m_isMultivalue = false;
  m_hasValue = false;
}

PbbTlv::~PbbTlv (void)
{
  NS_LOG_FUNCTION (this);
  m_value.RemoveAtEnd (m_value.GetSize ());
}

void
PbbTlv::SetType (uint8_t type)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (type));
  m_type = type;
}

uint8_t
PbbTlv::GetType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_type;
}

void
PbbTlv::SetTypeExt (uint8_t typeExt)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (typeExt));
  m_typeExt = typeExt;
  m_hasTypeExt = true;
}

uint8_t
PbbTlv::GetTypeExt (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasTypeExt ());
  return m_typeExt;
}

bool
PbbTlv::HasTypeExt (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasTypeExt;
}

void
PbbTlv::SetIndexStart (uint8_t index)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (index));
  m_indexStart = index;
  m_hasIndexStart = true;
}

uint8_t
PbbTlv::GetIndexStart (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasIndexStart ());
  return m_indexStart;
}

bool
PbbTlv::HasIndexStart (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasIndexStart;
}

void
PbbTlv::SetIndexStop (uint8_t index)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (index));
  m_indexStop = index;
  m_hasIndexStop = true;
}

uint8_t
PbbTlv::GetIndexStop (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasIndexStop ());
  return m_indexStop;
}

bool
PbbTlv::HasIndexStop (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasIndexStop;
}

void
PbbTlv::SetMultivalue (bool isMultivalue)
{
  NS_LOG_FUNCTION (this << isMultivalue);
  m_isMultivalue = isMultivalue;
}

bool
PbbTlv::IsMultivalue (void) const
{
  NS_LOG_FUNCTION (this);
  return m_isMultivalue;
}

void
PbbTlv::SetValue (Buffer start)
{
  NS_LOG_FUNCTION (this << &start);
  m_hasValue = true;
  m_value = start;
}

void
PbbTlv::SetValue (const uint8_t * buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << &buffer << size);
  m_hasValue = true;
  m_value.AddAtStart (size);
  m_value.Begin ().Write (buffer, size);
}

Buffer
PbbTlv::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (HasValue ());
  return m_value;
}

bool
PbbTlv::HasValue (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hasValue;
}

uint32_t
PbbTlv::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  /* type + flags */
  uint32_t size = 2;

  if (HasTypeExt ())
    {
      size++;
    }

  if (HasIndexStart ())
    {
      size++;
    }

  if (HasIndexStop ())
    {
      size++;
    }

  if (HasValue ())
    {
      if (GetValue ().GetSize () > 255)
        {
          size += 2;
        } 
      else 
        {
          size++;
        }
      size += GetValue ().GetSize ();
    }

  return size;
}

void
PbbTlv::Serialize (Buffer::Iterator &start) const
{
  NS_LOG_FUNCTION (this << &start);
  start.WriteU8 (GetType ());

  Buffer::Iterator bufref = start;
  uint8_t flags = 0;
  start.Next ();

  if (HasTypeExt ())
    {
      flags |= THAS_TYPE_EXT;
      start.WriteU8 (GetTypeExt ());
    }

  if (HasIndexStart ())
    {
      start.WriteU8 (GetIndexStart ());

      if (HasIndexStop ())
        {
          flags |= THAS_MULTI_INDEX;
          start.WriteU8 (GetIndexStop ());
        } 
      else
        {
          flags |= THAS_SINGLE_INDEX;
        }
    }

  if (HasValue ()) 
    {
      flags |= THAS_VALUE;

      uint32_t size = GetValue ().GetSize ();
      if (size > 255)
        {
          flags |= THAS_EXT_LEN;
          start.WriteHtonU16 (size);
        }
      else
        {
          start.WriteU8 (size);
        }

      if (IsMultivalue ())
        {
          flags |= TIS_MULTIVALUE;
        }

      start.Write (GetValue ().Begin (), GetValue ().End ());
    }

  bufref.WriteU8 (flags);
}

void
PbbTlv::Deserialize (Buffer::Iterator &start)
{
  NS_LOG_FUNCTION (this << &start);
  SetType (start.ReadU8 ());

  uint8_t flags = start.ReadU8 ();

  if (flags & THAS_TYPE_EXT)
    {
      SetTypeExt (start.ReadU8 ());
    }

  if (flags & THAS_MULTI_INDEX)
    {
      SetIndexStart (start.ReadU8 ());
      SetIndexStop (start.ReadU8 ());
    }
  else if (flags & THAS_SINGLE_INDEX)
    {
      SetIndexStart (start.ReadU8 ());
    }

  if (flags & THAS_VALUE)
    {
      uint16_t len = 0;

      if (flags & THAS_EXT_LEN)
        {
          len = start.ReadNtohU16 ();
        }
      else
        {
          len = start.ReadU8 ();
        }

      m_value.AddAtStart (len);

      Buffer::Iterator valueStart = start;
      start.Next (len);
      m_value.Begin ().Write (valueStart, start);
      m_hasValue = true;
    }
}

void
PbbTlv::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  Print (os, 0);
}

void
PbbTlv::Print (std::ostream &os, int level) const
{
  NS_LOG_FUNCTION (this << &os << level);
  std::string prefix = "";
  for (int i = 0; i < level; i++)
    {
      prefix.append ("\t");
    }

  os << prefix << "PbbTlv {" << std::endl;
  os << prefix << "\ttype = " << (int)GetType () << std::endl;

  if (HasTypeExt ())
    {
      os << prefix << "\ttypeext = " << (int)GetTypeExt () << std::endl;
    }

  if (HasIndexStart ())
    {
      os << prefix << "\tindexStart = " << (int)GetIndexStart () << std::endl;
    }

  if (HasIndexStop ())
    {
      os << prefix << "\tindexStop = " << (int)GetIndexStop () << std::endl;
    }

  os << prefix << "\tisMultivalue = " << IsMultivalue () << std::endl;

  if (HasValue ())
    {
      os << prefix << "\thas value; size = " << GetValue ().GetSize () << std::endl;
    }

  os << prefix << "}" << std::endl;
}

bool
PbbTlv::operator== (const PbbTlv &other) const
{
  if (GetType () != other.GetType ())
    {
      return false;
    }

  if (HasTypeExt () != other.HasTypeExt ())
    {
      return false;
    }

  if (HasTypeExt ())
    {
      if (GetTypeExt () != other.GetTypeExt ())
        {
          return false;
        }
    }

  if (HasValue () != other.HasValue ())
    {
      return false;
    }

  if (HasValue ())
    {
      Buffer tv = GetValue ();
      Buffer ov = other.GetValue ();
      if (tv.GetSize () != ov.GetSize ())
        {
          return false;
        }

      /* The docs say I probably shouldn't use Buffer::PeekData, but I think it
       * is justified in this case. */
      if (memcmp (tv.PeekData (), ov.PeekData (), tv.GetSize ()) != 0)
        {
          return false;
        }
    }
  return true;
}

bool
PbbTlv::operator!= (const PbbTlv &other) const
{
  return !(*this == other);
}

/* End PbbTlv Class */

void 
PbbAddressTlv::SetIndexStart (uint8_t index)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (index));
  PbbTlv::SetIndexStart (index);
}

uint8_t
PbbAddressTlv::GetIndexStart (void) const
{
  NS_LOG_FUNCTION (this);
  return PbbTlv::GetIndexStart ();
}

bool
PbbAddressTlv::HasIndexStart (void) const
{
  NS_LOG_FUNCTION (this);
  return PbbTlv::HasIndexStart ();
}

void 
PbbAddressTlv::SetIndexStop (uint8_t index)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (index));
  PbbTlv::SetIndexStop (index);
}

uint8_t
PbbAddressTlv::GetIndexStop (void) const
{
  NS_LOG_FUNCTION (this);
  return PbbTlv::GetIndexStop ();
}

bool
PbbAddressTlv::HasIndexStop (void) const
{
  NS_LOG_FUNCTION (this);
  return PbbTlv::HasIndexStop ();
}

void
PbbAddressTlv::SetMultivalue (bool isMultivalue)
{
  NS_LOG_FUNCTION (this << isMultivalue);
  PbbTlv::SetMultivalue (isMultivalue);
}

bool
PbbAddressTlv::IsMultivalue (void) const
{
  NS_LOG_FUNCTION (this);
  return PbbTlv::IsMultivalue ();
}

} /* namespace ns3 */
