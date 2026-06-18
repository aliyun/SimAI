/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
#include "byte-tag-list.h"
#include "ns3/log.h"
#include "ns3/atomic-counter.h"
#include <vector>
#include <cstring>
#include <limits>

#ifndef NS3_MTP
#define USE_FREE_LIST 1
#endif

#define FREE_LIST_SIZE 1000
#define OFFSET_MAX (std::numeric_limits<int32_t>::max ())

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ByteTagList");

/**
 * \ingroup packet
 *
 * \brief Internal representation of the byte tags stored in a packet.
 *
 * This structure is only used by ByteTagList and should not be accessed directly.
 */
struct ByteTagListData {
  uint32_t size;   //!< size of the data
#ifdef NS3_MTP
  AtomicCounter count;
#else
  uint32_t count;  //!< use counter (for smart deallocation)
#endif
  uint32_t dirty;  //!< number of bytes actually in use
  uint8_t data[4]; //!< data
};

#ifdef USE_FREE_LIST
/**
 * \ingroup packet
 *
 * \brief Container class for struct ByteTagListData
 *
 * Internal use only.
 */
static class ByteTagListDataFreeList : public std::vector<struct ByteTagListData *>
{
public:
  ~ByteTagListDataFreeList ();
} g_freeList; //!< Container for struct ByteTagListData
static uint32_t g_maxSize = 0; //!< maximum data size (used for allocation)

ByteTagListDataFreeList::~ByteTagListDataFreeList ()
{
  NS_LOG_FUNCTION (this);
  for (ByteTagListDataFreeList::iterator i = begin ();
       i != end (); i++)
    {
      uint8_t *buffer = (uint8_t *)(*i);
      delete [] buffer;
    }
}
#endif /* USE_FREE_LIST */

ByteTagList::Iterator::Item::Item (TagBuffer buf_)
  : buf (buf_)
{
  NS_LOG_FUNCTION (this << &buf_);
}

bool
ByteTagList::Iterator::HasNext (void) const
{
  NS_LOG_FUNCTION (this);
  return m_current < m_end;
}
struct ByteTagList::Iterator::Item
ByteTagList::Iterator::Next (void)
{
  NS_ASSERT (HasNext ());
  struct Item item = Item (TagBuffer (m_current+16, m_end));
  item.tid.SetUid (m_nextTid);
  item.size = m_nextSize;
  item.start = std::max (m_nextStart, m_offsetStart);
  item.end = std::min (m_nextEnd, m_offsetEnd);
  m_current += 4 + 4 + 4 + 4 + item.size;
  item.buf.TrimAtEnd (m_end - m_current);
  PrepareForNext ();
  return item;
}
void
ByteTagList::Iterator::PrepareForNext (void)
{
  NS_LOG_FUNCTION (this);
  while (m_current < m_end)
    {
      TagBuffer buf = TagBuffer (m_current, m_end);
      m_nextTid = buf.ReadU32 ();
      m_nextSize = buf.ReadU32 ();
      m_nextStart = buf.ReadU32 () + m_adjustment;
      m_nextEnd = buf.ReadU32 () + m_adjustment;
      if (m_nextStart >= m_offsetEnd || m_nextEnd <= m_offsetStart)
        {
          m_current += 4 + 4 + 4 + 4 + m_nextSize;
        }
      else
        {
          break;
        }
    }
}
ByteTagList::Iterator::Iterator (uint8_t *start, uint8_t *end, int32_t offsetStart, int32_t offsetEnd, int32_t adjustment)
  : m_current (start),
    m_end (end),
    m_offsetStart (offsetStart),
    m_offsetEnd (offsetEnd),
    m_adjustment (adjustment)
{
  NS_LOG_FUNCTION (this << &start << &end << offsetStart << offsetEnd << adjustment);
  PrepareForNext ();
}

uint32_t 
ByteTagList::Iterator::GetOffsetStart (void) const
{
  NS_LOG_FUNCTION (this);
  return m_offsetStart;
}


ByteTagList::ByteTagList ()
  : m_minStart (INT32_MAX),
    m_maxEnd (INT32_MIN),
    m_adjustment (0),
    m_used (0),
    m_data (0)
{
  NS_LOG_FUNCTION (this);
}
ByteTagList::ByteTagList (const ByteTagList &o)
  : m_minStart (o.m_minStart),
    m_maxEnd (o.m_maxEnd),
    m_adjustment (o.m_adjustment),
    m_used (o.m_used),
    m_data (o.m_data)
{
  NS_LOG_FUNCTION (this << &o);
  if (m_data != 0)
    {
      m_data->count++;
    }
}
ByteTagList &
ByteTagList::operator = (const ByteTagList &o)
{
  if (this == &o)
    {
      return *this;
    }

  Deallocate (m_data);
  m_minStart = o.m_minStart;
  m_maxEnd = o.m_maxEnd;
  m_adjustment = o.m_adjustment;
  m_data = o.m_data;
  m_used = o.m_used;
  if (m_data != 0)
    {
      m_data->count++;
    }
  return *this;
}
ByteTagList::~ByteTagList ()
{
  NS_LOG_FUNCTION (this);
  Deallocate (m_data);
  m_data = 0;
  m_used = 0;
}

TagBuffer
ByteTagList::Add (TypeId tid, uint32_t bufferSize, int32_t start, int32_t end)
{
  NS_LOG_FUNCTION (this << tid << bufferSize << start << end);
  uint32_t spaceNeeded = m_used + bufferSize + 4 + 4 + 4 + 4;
  NS_ASSERT (m_used <= spaceNeeded);
  if (m_data == 0)
    {
      m_data = Allocate (spaceNeeded);
      m_used = 0;
    } 
  else if (m_data->size < spaceNeeded ||
           (m_data->count != 1 && m_data->dirty != m_used))
    {
      struct ByteTagListData *newData = Allocate (spaceNeeded);
      std::memcpy (&newData->data, &m_data->data, m_used);
      Deallocate (m_data);
      m_data = newData;
    }
  TagBuffer tag = TagBuffer (&m_data->data[m_used], 
                             &m_data->data[spaceNeeded]);
  tag.WriteU32 (tid.GetUid ());
  tag.WriteU32 (bufferSize);
  tag.WriteU32 (start - m_adjustment);
  tag.WriteU32 (end - m_adjustment);
  if (start - m_adjustment < m_minStart)
    {
      m_minStart = start - m_adjustment;
    }
  if (end - m_adjustment > m_maxEnd)
    {
      m_maxEnd = end - m_adjustment;
    }
  m_used = spaceNeeded;
  m_data->dirty = m_used;
  return tag;
}

void 
ByteTagList::Add (const ByteTagList &o)
{
  NS_LOG_FUNCTION (this << &o);
  ByteTagList::Iterator i = o.BeginAll ();
  while (i.HasNext ())
    {
      ByteTagList::Iterator::Item item = i.Next ();
      TagBuffer buf = Add (item.tid, item.size, item.start, item.end);
      buf.CopyFrom (item.buf);
    }
}

void 
ByteTagList::RemoveAll (void)
{
  NS_LOG_FUNCTION (this);
  Deallocate (m_data);
  m_minStart = INT32_MAX;
  m_maxEnd = INT32_MIN;
  m_adjustment = 0;
  m_data = 0;
  m_used = 0;
}

ByteTagList::Iterator 
ByteTagList::BeginAll (void) const
{
  NS_LOG_FUNCTION (this);
  // I am not totally sure but I might need to use 
  // INT32_MIN instead of zero below.
  return Begin (0, OFFSET_MAX);
}

ByteTagList::Iterator 
ByteTagList::Begin (int32_t offsetStart, int32_t offsetEnd) const
{
  NS_LOG_FUNCTION (this << offsetStart << offsetEnd);
  if (m_data == 0)
    {
      return Iterator (0, 0, offsetStart, offsetEnd, 0);
    }
  else
    {
      return Iterator (m_data->data, &m_data->data[m_used], offsetStart, offsetEnd, m_adjustment);
    }
}

void 
ByteTagList::AddAtEnd (int32_t appendOffset)
{
  NS_LOG_FUNCTION (this << appendOffset);
  if (m_maxEnd <= appendOffset - m_adjustment)
    {
      return;
    }
  ByteTagList list;
  ByteTagList::Iterator i = BeginAll ();
  while (i.HasNext ())
    {
      ByteTagList::Iterator::Item item = i.Next ();

      if (item.start >= appendOffset)
        {
          continue;
        }
      if (item.end > appendOffset)
        {
          item.end = appendOffset;
        }
      TagBuffer buf = list.Add (item.tid, item.size, item.start, item.end);
      buf.CopyFrom (item.buf);
      if (item.end > m_maxEnd)
        {
          m_maxEnd = item.end;
        }
    }
  *this = list;
}

void 
ByteTagList::AddAtStart (int32_t prependOffset)
{
  NS_LOG_FUNCTION (this << prependOffset);
  if (m_minStart >= prependOffset - m_adjustment)
    {
      return;
    }
  m_minStart = INT32_MAX;
  ByteTagList list;
  ByteTagList::Iterator i = BeginAll ();
  while (i.HasNext ())
    {
      ByteTagList::Iterator::Item item = i.Next ();

      if (item.end <= prependOffset)
        {
          continue;
        }
      if (item.start < prependOffset)
        {
          item.start = prependOffset;
        }
      TagBuffer buf = list.Add (item.tid, item.size, item.start, item.end);
      buf.CopyFrom (item.buf);
      if (item.start < m_minStart)
        {
          m_minStart = item.start;
        }
    }
  *this = list;
}

#ifdef USE_FREE_LIST

struct ByteTagListData *
ByteTagList::Allocate (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  while (!g_freeList.empty ())
    {
      struct ByteTagListData *data = g_freeList.back ();
      g_freeList.pop_back ();
      NS_ASSERT (data != 0);
      if (data->size >= size)
        {
          data->count = 1;
          data->dirty = 0;
          return data;
        }
      uint8_t *buffer = (uint8_t *)data;
      delete [] buffer;
    }
  uint8_t *buffer = new uint8_t [std::max (size, g_maxSize) + sizeof (struct ByteTagListData) - 4];
  struct ByteTagListData *data = (struct ByteTagListData *)buffer;
  data->count = 1;
  data->size = size;
  data->dirty = 0;
  return data;
}

void 
ByteTagList::Deallocate (struct ByteTagListData *data)
{
  NS_LOG_FUNCTION (this << data);
  if (data == 0)
    {
      return;
    }
  g_maxSize = std::max (g_maxSize, data->size);
  data->count--;
  if (data->count == 0)
    {
      if (g_freeList.size () > FREE_LIST_SIZE ||
          data->size < g_maxSize)
        {
          uint8_t *buffer = (uint8_t *)data;
          delete [] buffer;
        }
      else
        {
          g_freeList.push_back (data);
        }
    }
}

#else /* USE_FREE_LIST */

struct ByteTagListData *
ByteTagList::Allocate (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  uint8_t *buffer = new uint8_t [size + sizeof (struct ByteTagListData) - 4];
  struct ByteTagListData *data = (struct ByteTagListData *)buffer;
  data->count = 1;
  data->size = size;
  data->dirty = 0;
  return data;
}

void 
ByteTagList::Deallocate (struct ByteTagListData *data)
{
  NS_LOG_FUNCTION (this << data);
  if (data == 0)
    {
      return;
    }
  if (data->count-- == 0)
    {
#ifdef NS3_MTP
      std::atomic_thread_fence (std::memory_order_acquire);
#endif
      uint8_t *buffer = (uint8_t *)data;
      delete [] buffer;
    }
}

#endif /* USE_FREE_LIST */

uint32_t
ByteTagList::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION_NOARGS ();

  uint32_t size = 0;

  // Number of tags in list
  size += 4; // numberOfTags

  ByteTagList::Iterator i = BeginAll ();
  while (i.HasNext ())
    {
      ByteTagList::Iterator::Item item = i.Next ();

      // TypeId hash; ensure size is multiple of 4 bytes
      uint32_t hashSize = (sizeof (TypeId::hash_t)+3) & (~3);
      size += hashSize;

      size += 3 * 4; // size, start, end

      // tag data; ensure size is multiple of 4 bytes
      uint32_t tagWordSize = (item.size+3) & (~3);
      size += tagWordSize;
    }

  return size;
}

uint32_t
ByteTagList::Serialize (uint32_t* buffer, uint32_t maxSize) const
{
  NS_LOG_FUNCTION (this << buffer << maxSize);

  uint32_t* p = buffer;
  uint32_t size = 0;

  uint32_t* numberOfTags = 0;

  if (size + 4 <= maxSize)
    {
      numberOfTags = p;
      *p++ = 0;
      size += 4;
    }
  else
    {
      return 0;
    }

  ByteTagList::Iterator i = BeginAll ();
  while (i.HasNext ())
    {
      ByteTagList::Iterator::Item item = i.Next ();

      NS_LOG_INFO ("Serializing " << item.tid);

      // ensure size is multiple of 4 bytes for 4 byte boundaries
      uint32_t hashSize = (sizeof (TypeId::hash_t)+3) & (~3);
      if (size + hashSize <= maxSize)
        {
          TypeId::hash_t tid = item.tid.GetHash ();
          memcpy (p, &tid, sizeof (TypeId::hash_t));
          p += hashSize / 4;
          size += hashSize;
        }
      else
        {
          return 0;
        }

      if (size + 4 <= maxSize)
        {
          *p++ = item.size;
          size += 4;
        }
      else
        {
          return 0;
        }

      if (size + 4 <= maxSize)
        {
          *p++ = item.start;
          size += 4;
        }
      else
        {
          return 0;
        }

      if (size + 4 <= maxSize)
        {
          *p++ = item.end;
          size += 4;
        }
      else
        {
          return 0;
        }

      // ensure size is multiple of 4 bytes for 4 byte boundaries
      uint32_t tagWordSize = (item.size+3) & (~3);

      if (size + tagWordSize <= maxSize)
        {
          item.buf.Read (reinterpret_cast<uint8_t *> (p), item.size);
          size += tagWordSize;
          p += tagWordSize / 4;
        }
      else
        {
          return 0;
        }

      (*numberOfTags)++;
    }

  // Serialized successfully
  return 1;
}

uint32_t
ByteTagList::Deserialize (const uint32_t* buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << buffer << size);
  const uint32_t* p = buffer;
  uint32_t sizeCheck = size - 4;

  NS_ASSERT (sizeCheck >= 4);
  uint32_t numberTagData = *p++;
  sizeCheck -= 4;

  NS_LOG_INFO ("Deserializing number of tags " << numberTagData);

  for(uint32_t i = 0; i < numberTagData; ++i)
    {
      uint32_t hashSize = (sizeof (TypeId::hash_t)+3) & (~3);
      NS_ASSERT (sizeCheck >= hashSize);
      TypeId::hash_t hash;
      memcpy (&hash, p, sizeof (TypeId::hash_t));
      p += hashSize / 4;
      sizeCheck -= hashSize;

      TypeId tid = TypeId::LookupByHash (hash);

      NS_ASSERT (sizeCheck >= 4);
      uint32_t bufferSize = *p++;
      sizeCheck -= 4;

      NS_ASSERT (sizeCheck >= 4);
      uint32_t start = *p++;
      sizeCheck -= 4;

      NS_ASSERT (sizeCheck >= 4);
      uint32_t end = *p++;
      sizeCheck -= 4;

      NS_ASSERT (sizeCheck >= bufferSize);
      TagBuffer buf = Add (tid, bufferSize, start, end);
      buf.Write ( reinterpret_cast<const uint8_t *> (p), bufferSize);

      // ensure 4 byte boundary
      uint32_t tagSizeBytes = (bufferSize+3) & (~3);
      sizeCheck -= tagSizeBytes;
      p += tagSizeBytes / 4;
    }

  NS_ASSERT (sizeCheck == 0);

  // return zero if buffer did not
  // contain a complete message
  return (sizeCheck != 0) ? 0 : 1;
}

} // namespace ns3
