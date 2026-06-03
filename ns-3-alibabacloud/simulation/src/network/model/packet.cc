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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "packet.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include <string>
#include <cstdarg>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Packet");

uint32_t Packet::m_globalUid = 0;

TypeId 
ByteTagIterator::Item::GetTypeId (void) const
{
  return m_tid;
}
uint32_t 
ByteTagIterator::Item::GetStart (void) const
{
  return m_start;
}
uint32_t 
ByteTagIterator::Item::GetEnd (void) const
{
  return m_end;
}
void 
ByteTagIterator::Item::GetTag (Tag &tag) const
{
  if (tag.GetInstanceTypeId () != GetTypeId ())
    {
      NS_FATAL_ERROR ("The tag you provided is not of the right type.");
    }
  tag.Deserialize (m_buffer);
}
ByteTagIterator::Item::Item (TypeId tid, uint32_t start, uint32_t end, TagBuffer buffer)
  : m_tid (tid),
    m_start (start),
    m_end (end),
    m_buffer (buffer)
{
}
bool
ByteTagIterator::HasNext (void) const
{
  return m_current.HasNext ();
}
ByteTagIterator::Item
ByteTagIterator::Next (void)
{
  ByteTagList::Iterator::Item i = m_current.Next ();
  return ByteTagIterator::Item (i.tid,
                                i.start-m_current.GetOffsetStart (),
                                i.end-m_current.GetOffsetStart (),
                                i.buf);
}
ByteTagIterator::ByteTagIterator (ByteTagList::Iterator i)
  : m_current (i)
{
}


PacketTagIterator::PacketTagIterator (const struct PacketTagList::TagData *head)
  : m_current (head)
{
}
bool
PacketTagIterator::HasNext (void) const
{
  return m_current != 0;
}
PacketTagIterator::Item
PacketTagIterator::Next (void)
{
  NS_ASSERT (HasNext ());
  const struct PacketTagList::TagData *prev = m_current;
  m_current = m_current->next;
  return PacketTagIterator::Item (prev);
}

PacketTagIterator::Item::Item (const struct PacketTagList::TagData *data)
  : m_data (data)
{
}
TypeId
PacketTagIterator::Item::GetTypeId (void) const
{
  return m_data->tid;
}
void
PacketTagIterator::Item::GetTag (Tag &tag) const
{
  NS_ASSERT (tag.GetInstanceTypeId () == m_data->tid);
  tag.Deserialize (TagBuffer ((uint8_t*)m_data->data,
                              (uint8_t*)m_data->data + m_data->size));
}


Ptr<Packet> 
Packet::Copy (void) const
{
  // we need to invoke the copy constructor directly
  // rather than calling Create because the copy constructor
  // is private.
  return Ptr<Packet> (new Packet (*this), false);
}

Packet::Packet ()
  : m_buffer (),
    m_byteTagList (),
    m_packetTagList (),
    /* The upper 32 bits of the packet id in 
     * metadata is for the system id. For non-
     * distributed simulations, this is simply 
     * zero.  The lower 32 bits are for the 
     * global UID
     */
    m_metadata (static_cast<uint64_t> (Simulator::GetSystemId ()) << 32 | m_globalUid, 0),
    m_nixVector (0)
{
  m_globalUid++;
}

Packet::Packet (const Packet &o)
  : m_buffer (o.m_buffer),
    m_byteTagList (o.m_byteTagList),
    m_packetTagList (o.m_packetTagList),
    m_metadata (o.m_metadata)
{
  o.m_nixVector ? m_nixVector = o.m_nixVector->Copy ()
    : m_nixVector = 0;
}

Packet &
Packet::operator = (const Packet &o)
{
  if (this == &o)
    {
      return *this;
    }
  m_buffer = o.m_buffer;
  m_byteTagList = o.m_byteTagList;
  m_packetTagList = o.m_packetTagList;
  m_metadata = o.m_metadata;
  o.m_nixVector ? m_nixVector = o.m_nixVector->Copy () 
    : m_nixVector = 0;
  return *this;
}

Packet::Packet (uint32_t size)
  : m_buffer (size),
    m_byteTagList (),
    m_packetTagList (),
    /* The upper 32 bits of the packet id in 
     * metadata is for the system id. For non-
     * distributed simulations, this is simply 
     * zero.  The lower 32 bits are for the 
     * global UID
     */
    m_metadata (static_cast<uint64_t> (Simulator::GetSystemId ()) << 32 | m_globalUid, size),
    m_nixVector (0)
{
  m_globalUid++;
}
Packet::Packet (uint8_t const *buffer, uint32_t size, bool magic)
  : m_buffer (0, false),
    m_byteTagList (),
    m_packetTagList (),
    m_metadata (0,0),
    m_nixVector (0)
{
  NS_ASSERT (magic);
  Deserialize (buffer, size);
}

Packet::Packet (uint8_t const*buffer, uint32_t size)
  : m_buffer (),
    m_byteTagList (),
    m_packetTagList (),
    /* The upper 32 bits of the packet id in 
     * metadata is for the system id. For non-
     * distributed simulations, this is simply 
     * zero.  The lower 32 bits are for the 
     * global UID
     */
    m_metadata (static_cast<uint64_t> (Simulator::GetSystemId ()) << 32 | m_globalUid, size),
    m_nixVector (0)
{
  m_globalUid++;
  m_buffer.AddAtStart (size);
  Buffer::Iterator i = m_buffer.Begin ();
  i.Write (buffer, size);
}

Packet::Packet (const Buffer &buffer,  const ByteTagList &byteTagList, 
                const PacketTagList &packetTagList, const PacketMetadata &metadata)
  : m_buffer (buffer),
    m_byteTagList (byteTagList),
    m_packetTagList (packetTagList),
    m_metadata (metadata),
    m_nixVector (0)
{
}

Ptr<Packet>
Packet::CreateFragment (uint32_t start, uint32_t length) const
{
  NS_LOG_FUNCTION (this << start << length);
  Buffer buffer = m_buffer.CreateFragment (start, length);
  ByteTagList byteTagList = m_byteTagList;
  byteTagList.Adjust (-start);
  NS_ASSERT (m_buffer.GetSize () >= start + length);
  uint32_t end = m_buffer.GetSize () - (start + length);
  PacketMetadata metadata = m_metadata.CreateFragment (start, end);
  // again, call the constructor directly rather than
  // through Create because it is private.
  Ptr<Packet> ret = Ptr<Packet> (new Packet (buffer, byteTagList, m_packetTagList, metadata), false);
  ret->SetNixVector (GetNixVector ());
  return ret;
}

void
Packet::SetNixVector (Ptr<NixVector> nixVector) const
{
  m_nixVector = nixVector;
}

Ptr<NixVector>
Packet::GetNixVector (void) const
{
  return m_nixVector;
} 

void
Packet::AddHeader (const Header &header)
{
  uint32_t size = header.GetSerializedSize ();
  NS_LOG_FUNCTION (this << header.GetInstanceTypeId ().GetName () << size);
  m_buffer.AddAtStart (size);
  m_byteTagList.Adjust (size);
  m_byteTagList.AddAtStart (size);
  header.Serialize (m_buffer.Begin ());
  m_metadata.AddHeader (header, size);
}
uint32_t
Packet::RemoveHeader (Header &header, uint32_t size)
{
  Buffer::Iterator end;
  end = m_buffer.Begin ();
  end.Next (size);
  uint32_t deserialized = header.Deserialize (m_buffer.Begin (), end);
  NS_LOG_FUNCTION (this << header.GetInstanceTypeId ().GetName () << deserialized);
  m_buffer.RemoveAtStart (deserialized);
  m_byteTagList.Adjust (-deserialized);
  m_metadata.RemoveHeader (header, deserialized);
  return deserialized;
}
uint32_t
Packet::RemoveHeader (Header &header)
{
  uint32_t deserialized = header.Deserialize (m_buffer.Begin ());
  NS_LOG_FUNCTION (this << header.GetInstanceTypeId ().GetName () << deserialized);
  m_buffer.RemoveAtStart (deserialized);
  m_byteTagList.Adjust (-deserialized);
  m_metadata.RemoveHeader (header, deserialized);
  return deserialized;
}
uint32_t
Packet::PeekHeader (Header &header) const
{
  uint32_t deserialized = header.Deserialize (m_buffer.Begin ());
  NS_LOG_FUNCTION (this << header.GetInstanceTypeId ().GetName () << deserialized);
  return deserialized;
}
uint32_t
Packet::PeekHeader (Header &header, uint32_t size) const
{
  Buffer::Iterator end;
  end = m_buffer.Begin ();
  end.Next (size);
  uint32_t deserialized = header.Deserialize (m_buffer.Begin (), end);
  NS_LOG_FUNCTION (this << header.GetInstanceTypeId ().GetName () << deserialized);
  return deserialized;
}
void
Packet::AddTrailer (const Trailer &trailer)
{
  uint32_t size = trailer.GetSerializedSize ();
  NS_LOG_FUNCTION (this << trailer.GetInstanceTypeId ().GetName () << size);
  m_byteTagList.AddAtEnd (GetSize ());
  m_buffer.AddAtEnd (size);
  Buffer::Iterator end = m_buffer.End ();
  trailer.Serialize (end);
  m_metadata.AddTrailer (trailer, size);
}
uint32_t
Packet::RemoveTrailer (Trailer &trailer)
{
  uint32_t deserialized = trailer.Deserialize (m_buffer.End ());
  NS_LOG_FUNCTION (this << trailer.GetInstanceTypeId ().GetName () << deserialized);
  m_buffer.RemoveAtEnd (deserialized);
  m_metadata.RemoveTrailer (trailer, deserialized);
  return deserialized;
}
uint32_t
Packet::PeekTrailer (Trailer &trailer)
{
  uint32_t deserialized = trailer.Deserialize (m_buffer.End ());
  NS_LOG_FUNCTION (this << trailer.GetInstanceTypeId ().GetName () << deserialized);
  return deserialized;
}

void 
Packet::AddAtEnd (Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (this << packet << packet->GetSize ());
  m_byteTagList.AddAtEnd (GetSize ());
  ByteTagList copy = packet->m_byteTagList;
  copy.AddAtStart (0);
  copy.Adjust (GetSize ());
  m_byteTagList.Add (copy);
  m_buffer.AddAtEnd (packet->m_buffer);
  m_metadata.AddAtEnd (packet->m_metadata);
}
void
Packet::AddPaddingAtEnd (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  m_byteTagList.AddAtEnd (GetSize ());
  m_buffer.AddAtEnd (size);
  m_metadata.AddPaddingAtEnd (size);
}
void 
Packet::RemoveAtEnd (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  m_buffer.RemoveAtEnd (size);
  m_metadata.RemoveAtEnd (size);
}
void 
Packet::RemoveAtStart (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  m_buffer.RemoveAtStart (size);
  m_byteTagList.Adjust (-size);
  m_metadata.RemoveAtStart (size);
}

void 
Packet::RemoveAllByteTags (void)
{
  NS_LOG_FUNCTION (this);
  m_byteTagList.RemoveAll ();
}

uint32_t 
Packet::CopyData (uint8_t *buffer, uint32_t size) const
{
  return m_buffer.CopyData (buffer, size);
}

void
Packet::CopyData (std::ostream *os, uint32_t size) const
{
  return m_buffer.CopyData (os, size);
}

uint64_t 
Packet::GetUid (void) const
{
  return m_metadata.GetUid ();
}

void 
Packet::PrintByteTags (std::ostream &os) const
{
  ByteTagIterator i = GetByteTagIterator ();
  while (i.HasNext ())
    {
      ByteTagIterator::Item item = i.Next ();
      os << item.GetTypeId ().GetName () << " [" << item.GetStart () << "-" << item.GetEnd () << "]";
      Callback<ObjectBase *> constructor = item.GetTypeId ().GetConstructor ();
      if (constructor.IsNull ())
        {
          if (i.HasNext ())
            {
              os << " ";
            }
          continue;
        }
      Tag *tag = dynamic_cast<Tag *> (constructor ());
      NS_ASSERT (tag != 0);
      os << " ";
      item.GetTag (*tag);
      tag->Print (os);
      if (i.HasNext ())
        {
          os << " ";
        }
      delete tag;
    }
}

std::string
Packet::ToString() const
{
  std::ostringstream oss;
  Print (oss);
  return oss.str();
}

void 
Packet::Print (std::ostream &os) const
{
  PacketMetadata::ItemIterator i = m_metadata.BeginItem (m_buffer);
  while (i.HasNext ())
    {
      PacketMetadata::Item item = i.Next ();
      if (item.isFragment)
        {
          switch (item.type) {
            case PacketMetadata::Item::PAYLOAD:
              os << "Payload";
              break;
            case PacketMetadata::Item::HEADER:
            case PacketMetadata::Item::TRAILER:
              os << item.tid.GetName ();
              break;
            }
          os << " Fragment [" << item.currentTrimedFromStart<<":"
             << (item.currentTrimedFromStart + item.currentSize) << "]";
        }
      else
        {
          switch (item.type) {
            case PacketMetadata::Item::PAYLOAD:
              os << "Payload (size=" << item.currentSize << ")";
              break;
            case PacketMetadata::Item::HEADER:
            case PacketMetadata::Item::TRAILER:
              os << item.tid.GetName () << " (";
              {
                NS_ASSERT (item.tid.HasConstructor ());
                Callback<ObjectBase *> constructor = item.tid.GetConstructor ();
                NS_ASSERT (!constructor.IsNull ());
                ObjectBase *instance = constructor ();
                NS_ASSERT (instance != 0);
                Chunk *chunk = dynamic_cast<Chunk *> (instance);
                NS_ASSERT (chunk != 0);
                if (item.type == PacketMetadata::Item::HEADER)
                  {
                    Buffer::Iterator end = item.current;
                    end.Next (item.currentSize); // move from start 
                    chunk->Deserialize (item.current, end);
                  }
                else if (item.type == PacketMetadata::Item::TRAILER)
                  {
                    Buffer::Iterator start = item.current;
                    start.Prev (item.currentSize); // move from end
                    chunk->Deserialize (start, item.current);
                  }
                else
                  {
                    chunk->Deserialize (item.current);
                  }    
                chunk->Print (os);
                delete chunk;
              }
              os << ")";
              break;
            }
        }
      if (i.HasNext ())
        {
          os << " ";
        }
    }
#if 0
  // The code below will work only if headers and trailers
  // define the right attributes which is not the case for
  // now. So, as a temporary measure, we use the 
  // headers' and trailers' Print method as shown above.
  PacketMetadata::ItemIterator i = m_metadata.BeginItem (m_buffer);
  while (i.HasNext ())
    {
      PacketMetadata::Item item = i.Next ();
      if (item.isFragment)
        {
          switch (item.type) {
            case PacketMetadata::Item::PAYLOAD:
              os << "Payload";
              break;
            case PacketMetadata::Item::HEADER:
            case PacketMetadata::Item::TRAILER:
              os << item.tid.GetName ();
              break;
            }
          os << " Fragment [" << item.currentTrimedFromStart<<":"
             << (item.currentTrimedFromStart + item.currentSize) << "]";
        }
      else
        {
          switch (item.type) {
            case PacketMetadata::Item::PAYLOAD:
              os << "Payload (size=" << item.currentSize << ")";
              break;
            case PacketMetadata::Item::HEADER:
            case PacketMetadata::Item::TRAILER:
              os << item.tid.GetName () << "(";
              {
                NS_ASSERT (item.tid.HasConstructor ());
                Callback<ObjectBase *> constructor = item.tid.GetConstructor ();
                NS_ASSERT (constructor.IsNull ());
                ObjectBase *instance = constructor ();
                NS_ASSERT (instance != 0);
                Chunk *chunk = dynamic_cast<Chunk *> (instance);
                NS_ASSERT (chunk != 0);
                chunk->Deserialize (item.current);
                for (uint32_t j = 0; j < item.tid.GetAttributeN (); j++)
                  {
                    std::string attrName = item.tid.GetAttributeName (j);
                    std::string value;
                    bool ok = chunk->GetAttribute (attrName, value);
                    NS_ASSERT (ok);
                    os << attrName << "=" << value;
                    if ((j + 1) < item.tid.GetAttributeN ())
                      {
                        os << ",";
                      }
                  }
              }
              os << ")";
              break;
            }
        }
      if (i.HasNext ())
        {
          os << " ";
        }
    }
#endif
}

PacketMetadata::ItemIterator 
Packet::BeginItem (void) const
{
  return m_metadata.BeginItem (m_buffer);
}

void
Packet::EnablePrinting (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  PacketMetadata::Enable ();
}

void
Packet::EnableChecking (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  PacketMetadata::EnableChecking ();
}

uint32_t Packet::GetSerializedSize (void) const
{
  uint32_t size = 0;

  if (m_nixVector)
    {
      // increment total size by the size of the nix-vector
      // ensuring 4-byte boundary
      size += ((m_nixVector->GetSerializedSize () + 3) & (~3));

      // add 4-bytes for entry of total length of nix-vector
      size += 4;
    }
  else
    {
      // if no nix-vector, still have to add 4-bytes
      // to account for the entry of total size for 
      // nix-vector in the buffer
      size += 4;
    }

  // increment total size by size of packet tag list
  // ensuring 4-byte boundary
  size += ((m_packetTagList.GetSerializedSize () + 3) & (~3));

  // add 4-bytes for entry of total length of packet tag list
  size += 4;

  // increment total size by size of byte tag list
  // ensuring 4-byte boundary
  size += ((m_byteTagList.GetSerializedSize () + 3) & (~3));

  // add 4-bytes for entry of total length of byte tag list
  size += 4;

  // increment total size by size of meta-data 
  // ensuring 4-byte boundary
  size += ((m_metadata.GetSerializedSize () + 3) & (~3));

  // add 4-bytes for entry of total length of meta-data
  size += 4;

  // increment total size by size of buffer 
  // ensuring 4-byte boundary
  size += ((m_buffer.GetSerializedSize () + 3) & (~3));

  // add 4-bytes for entry of total length of buffer 
  size += 4;

  return size;
}

uint32_t 
Packet::Serialize (uint8_t* buffer, uint32_t maxSize) const
{
  uint32_t* p = reinterpret_cast<uint32_t *> (buffer);
  uint32_t size = 0;

  // if nix-vector exists, serialize it
  if (m_nixVector)
    {
      uint32_t nixSize = m_nixVector->GetSerializedSize ();
      if (size + nixSize <= maxSize)
        {
          // put the total length of nix-vector in the
          // buffer. this includes 4-bytes for total 
          // length itself
          *p++ = nixSize + 4;
          size += nixSize;

          // serialize the nix-vector
          uint32_t serialized = 
            m_nixVector->Serialize (p, nixSize);
          if (serialized)
            {
              // increment p by nixSize bytes
              // ensuring 4-byte boundary
              p += ((nixSize+3) & (~3)) / 4;
            }
          else
            {
              return 0;
            }
        }
      else 
        {
          return 0;
        }
    }
  else
    { 
      // no nix vector, set zero length, 
      // ie 4-bytes, since it must include 
      // length for itself
      if (size + 4 <= maxSize)
        {
          size += 4;
          *p++ = 4;
        }
      else
        {
          return 0;
        }
    }

  // Serialize byte tag list
  uint32_t byteTagSize = m_byteTagList.GetSerializedSize ();
  if (size + byteTagSize <= maxSize)
    {
      // put the total length of byte tag list in the
      // buffer. this includes 4-bytes for total
      // length itself
      *p++ = byteTagSize + 4;
      size += byteTagSize;

      // serialize the byte tag list
      uint32_t serialized = m_byteTagList.Serialize (p, byteTagSize);
      if (serialized)
        {
          // increment p by byteTagSize bytes
          // ensuring 4-byte boundary
          p += ((byteTagSize+3) & (~3)) / 4;
        }
      else
        {
          return 0;
        }
    }
  else
    {
      return 0;
    }

  // Serialize packet tag list
  uint32_t packetTagSize = m_packetTagList.GetSerializedSize ();
  if (size + packetTagSize <= maxSize)
    {
      // put the total length of packet tag list in the
      // buffer. this includes 4-bytes for total
      // length itself
      *p++ = packetTagSize + 4;
      size += packetTagSize;

      // serialize the packet tag list
      uint32_t serialized = m_packetTagList.Serialize (p, packetTagSize);
      if (serialized)
        {
          // increment p by packetTagSize bytes
          // ensuring 4-byte boundary
          p += ((packetTagSize+3) & (~3)) / 4;
        }
      else
        {
          return 0;
        }
    }
  else
    {
      return 0;
    }

  // Serialize Metadata
  uint32_t metaSize = m_metadata.GetSerializedSize ();
  if (size + metaSize <= maxSize)
    {
      // put the total length of metadata in the
      // buffer. this includes 4-bytes for total 
      // length itself
      *p++ = metaSize + 4;
      size += metaSize;

      // serialize the metadata
      uint32_t serialized = m_metadata.Serialize (reinterpret_cast<uint8_t *> (p), metaSize);
      if (serialized)
        {
          // increment p by metaSize bytes
          // ensuring 4-byte boundary
          p += ((metaSize+3) & (~3)) / 4;
        }
      else
        {
          return 0;
        }
    }
  else
    {
      return 0;
    }

  // Serialize the packet contents
  uint32_t bufSize = m_buffer.GetSerializedSize ();
  if (size + bufSize <= maxSize)
    {
      // put the total length of the buffer in the
      // buffer. this includes 4-bytes for total
      // length itself
      *p++ = bufSize + 4;

      // serialize the buffer
      uint32_t serialized = m_buffer.Serialize (reinterpret_cast<uint8_t *> (p), bufSize);
      if (!serialized)
        {
          return 0;
        }
    }
  else
    {
      return 0;
    }

  // Serialized successfully
  return 1;
}

uint32_t 
Packet::Deserialize (const uint8_t* buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this);

  const uint32_t* p = reinterpret_cast<const uint32_t *> (buffer);

  // read nix-vector
  NS_ASSERT (!m_nixVector);
  uint32_t nixSize = *p++;

  // if size less than nixSize, the buffer 
  // will be overrun, assert
  NS_ASSERT (size >= nixSize);

  if (nixSize > 4)
    {
      Ptr<NixVector> nix = Create<NixVector> ();
      uint32_t nixDeserialized = nix->Deserialize (p, nixSize);
      if (!nixDeserialized)
        {
          // nix-vector not deserialized
          // completely
          return 0;
        }
      m_nixVector = nix;
      // increment p by nixSize ensuring
      // 4-byte boundary
      p += ((((nixSize - 4) + 3) & (~3)) / 4);
    }
  size -= nixSize;

  // read byte tags
  uint32_t byteTagSize = *p++;

  // if size less than byteTagSize, the buffer
  // will be overrun, assert
  NS_ASSERT (size >= byteTagSize);
  
  uint32_t byteTagDeserialized =
    m_byteTagList.Deserialize (p, byteTagSize);
  if (!byteTagDeserialized)
    {
      // byte tags not deserialized completely
      return 0;
    }
  // increment p by byteTagSize ensuring
  // 4-byte boundary
  p += ((((byteTagSize - 4) + 3) & (~3)) / 4);
  size -= byteTagSize;

  // read packet tags
  uint32_t packetTagSize = *p++;

  // if size less than packetTagSize, the buffer
  // will be overrun, assert
  NS_ASSERT (size >= packetTagSize);

  uint32_t packetTagDeserialized =
    m_packetTagList.Deserialize (p, packetTagSize);
  if (!packetTagDeserialized)
    {
      // packet tags not deserialized completely
      return 0;
    }
  // increment p by packetTagSize ensuring
  // 4-byte boundary
  p += ((((packetTagSize - 4) + 3) & (~3)) / 4);
  size -= packetTagSize;

  // read metadata
  uint32_t metaSize = *p++;

  // if size less than metaSize, the buffer
  // will be overrun, assert
  NS_ASSERT (size >= metaSize);

  uint32_t metadataDeserialized =
    m_metadata.Deserialize (reinterpret_cast<const uint8_t *> (p), metaSize);
  if (!metadataDeserialized)
    {
      // meta-data not deserialized
      // completely
      return 0;
    }
  // increment p by metaSize ensuring
  // 4-byte boundary
  p += ((((metaSize - 4) + 3) & (~3)) / 4);
  size -= metaSize;

  // read buffer contents
  uint32_t bufSize = *p++;

  // if size less than bufSize, the buffer
  // will be overrun, assert
  NS_ASSERT (size >= bufSize);

  uint32_t bufferDeserialized =
    m_buffer.Deserialize (reinterpret_cast<const uint8_t *> (p), bufSize);
  if (!bufferDeserialized)
    {
      // buffer not deserialized
      // completely
      return 0;
    }
  size -= bufSize;

  // return zero if did not deserialize the
  // number of expected bytes
  return (size == 0);
}

void 
Packet::AddByteTag (const Tag &tag) const
{
  NS_LOG_FUNCTION (this << tag.GetInstanceTypeId ().GetName () << tag.GetSerializedSize ());
  ByteTagList *list = const_cast<ByteTagList *> (&m_byteTagList);
  TagBuffer buffer = list->Add (tag.GetInstanceTypeId (), tag.GetSerializedSize (),
                                0,
                                GetSize ());
  tag.Serialize (buffer);
}
void
Packet::AddByteTag (const Tag &tag, uint32_t start, uint32_t end) const
{
  NS_LOG_FUNCTION (this << tag.GetInstanceTypeId ().GetName () << tag.GetSerializedSize ());
  NS_ABORT_MSG_IF (end < start, "Invalid byte range");
  ByteTagList *list = const_cast<ByteTagList *> (&m_byteTagList);
  TagBuffer buffer = list->Add (tag.GetInstanceTypeId (), tag.GetSerializedSize (),
                                static_cast<int32_t> (start),
                                static_cast<int32_t> (end));
  tag.Serialize (buffer);
}
ByteTagIterator 
Packet::GetByteTagIterator (void) const
{
  return ByteTagIterator (m_byteTagList.Begin (0, GetSize ()));
}

bool 
Packet::FindFirstMatchingByteTag (Tag &tag) const
{
  TypeId tid = tag.GetInstanceTypeId ();
  ByteTagIterator i = GetByteTagIterator ();
  while (i.HasNext ())
    {
      ByteTagIterator::Item item = i.Next ();
      if (tid == item.GetTypeId ())
        {
          item.GetTag (tag);
          return true;
        }
    }
  return false;
}

void 
Packet::AddPacketTag (const Tag &tag) const
{
  NS_LOG_FUNCTION (this << tag.GetInstanceTypeId ().GetName () << tag.GetSerializedSize ());
  m_packetTagList.Add (tag);
}

bool 
Packet::RemovePacketTag (Tag &tag)
{
  NS_LOG_FUNCTION (this << tag.GetInstanceTypeId ().GetName () << tag.GetSerializedSize ());
  bool found = m_packetTagList.Remove (tag);
  return found;
}
bool
Packet::ReplacePacketTag (Tag &tag)
{
  NS_LOG_FUNCTION (this << tag.GetInstanceTypeId ().GetName () << tag.GetSerializedSize ());
  bool found = m_packetTagList.Replace (tag);
  return found;
}

bool 
Packet::PeekPacketTag (Tag &tag) const
{
  bool found = m_packetTagList.Peek (tag);
  return found;
}
void 
Packet::RemoveAllPacketTags (void)
{
  NS_LOG_FUNCTION (this);
  m_packetTagList.RemoveAll ();
}

void 
Packet::PrintPacketTags (std::ostream &os) const
{
  PacketTagIterator i = GetPacketTagIterator ();
  while (i.HasNext ())
    {
      PacketTagIterator::Item item = i.Next ();
      NS_ASSERT (item.GetTypeId ().HasConstructor ());
      Callback<ObjectBase *> constructor = item.GetTypeId ().GetConstructor ();
      NS_ASSERT (!constructor.IsNull ());
      ObjectBase *instance = constructor ();
      Tag *tag = dynamic_cast<Tag *> (instance);
      NS_ASSERT (tag != 0);
      item.GetTag (*tag);
      tag->Print (os);
      delete tag;
      if (i.HasNext ())
        {
          os << " ";
        }
    }
}

PacketTagIterator 
Packet::GetPacketTagIterator (void) const
{
  return PacketTagIterator (m_packetTagList.Head ());
}

std::ostream& operator<< (std::ostream& os, const Packet &packet)
{
  packet.Print (os);
  return os;
}

uint8_t* Packet::GetBuffer() const{
	return m_buffer.GetBuffer();
}

} // namespace ns3
