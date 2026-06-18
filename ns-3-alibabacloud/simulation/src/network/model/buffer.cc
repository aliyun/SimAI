/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006,2007 INRIA
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
#include "buffer.h"
#include "ns3/assert.h"
#include "ns3/log.h"

#define LOG_INTERNAL_STATE(y)                                                                    \
  NS_LOG_LOGIC (y << "start="<<m_start<<", end="<<m_end<<", zero start="<<m_zeroAreaStart<<              \
                ", zero end="<<m_zeroAreaEnd<<", count="<<m_data->m_count<<", size="<<m_data->m_size<<   \
                ", dirty start="<<m_data->m_dirtyStart<<", dirty end="<<m_data->m_dirtyEnd)

namespace {

/**
 * \ingroup packet
 * \brief Zero-filled buffer.
 */
static struct Zeroes
{
  Zeroes ()
    : size (1000)
  {
    memset (buffer, 0, size);
  }
  char buffer[1000];    //!< buffer containing zero values
  const uint32_t size;  //!< buffer size
} g_zeroes; //!< Zero-filled buffer

}

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Buffer");


uint32_t Buffer::g_recommendedStart = 0;
#ifdef BUFFER_FREE_LIST
/* The following macros are pretty evil but they are needed to allow us to
 * keep track of 3 possible states for the g_freeList variable:
 *  - uninitialized means that no one has created a buffer yet
 *    so no one has created the associated free list (it is created
 *    on-demand when the first buffer is created)
 *  - initialized means that the free list exists and is valid
 *  - destroyed means that the static destructors of this compilation unit
 *    have run so, the free list has been cleared from its content
 * The key is that in destroyed state, we are careful not re-create it
 * which is a typical weakness of lazy evaluation schemes which use 
 * '0' as a special value to indicate both un-initialized and destroyed.
 * Note that it is important to use '0' as the marker for un-initialized state
 * because the variable holding this state information is initialized to zero
 * which the compiler assigns to zero-memory which is initialized to _zero_
 * before the constructors run so this ensures perfect handling of crazy 
 * constructor orderings.
 */
#define MAGIC_DESTROYED (~(long) 0)
#define IS_UNINITIALIZED(x) (x == (Buffer::FreeList*)0)
#define IS_DESTROYED(x) (x == (Buffer::FreeList*)MAGIC_DESTROYED)
#define IS_INITIALIZED(x) (!IS_UNINITIALIZED (x) && !IS_DESTROYED (x))
#define DESTROYED ((Buffer::FreeList*)MAGIC_DESTROYED)
#define UNINITIALIZED ((Buffer::FreeList*)0)
uint32_t Buffer::g_maxSize = 0;
Buffer::FreeList *Buffer::g_freeList = 0;
struct Buffer::LocalStaticDestructor Buffer::g_localStaticDestructor;

Buffer::LocalStaticDestructor::~LocalStaticDestructor(void)
{
  NS_LOG_FUNCTION (this);
  if (IS_INITIALIZED (g_freeList))
    {
      for (Buffer::FreeList::iterator i = g_freeList->begin ();
           i != g_freeList->end (); i++)
        {
          Buffer::Deallocate (*i);
        }
      delete g_freeList;
      g_freeList = DESTROYED;
    }
}

void
Buffer::Recycle (struct Buffer::Data *data)
{
  NS_LOG_FUNCTION (data);
  NS_ASSERT (data->m_count == 0);
  NS_ASSERT (!IS_UNINITIALIZED (g_freeList));
  g_maxSize = std::max (g_maxSize, data->m_size);
  /* feed into free list */
  if (data->m_size < g_maxSize ||
      IS_DESTROYED (g_freeList) ||
      g_freeList->size () > 1000)
    {
      Buffer::Deallocate (data);
    }
  else
    {
      NS_ASSERT (IS_INITIALIZED (g_freeList));
      g_freeList->push_back (data);
    }
}

Buffer::Data *
Buffer::Create (uint32_t dataSize)
{
  NS_LOG_FUNCTION (dataSize);
  /* try to find a buffer correctly sized. */
  if (IS_UNINITIALIZED (g_freeList))
    {
      g_freeList = new Buffer::FreeList ();
    }
  else if (IS_INITIALIZED (g_freeList))
    {
      while (!g_freeList->empty ()) 
        {
          struct Buffer::Data *data = g_freeList->back ();
          g_freeList->pop_back ();
          if (data->m_size >= dataSize) 
            {
              data->m_count = 1;
              return data;
            }
          Buffer::Deallocate (data);
        }
    }
  struct Buffer::Data *data = Buffer::Allocate (dataSize);
  NS_ASSERT (data->m_count == 1);
  return data;
}
#else /* BUFFER_FREE_LIST */
void
Buffer::Recycle (struct Buffer::Data *data)
{
  NS_LOG_FUNCTION (data);
  NS_ASSERT (data->m_count == 0);
#ifdef NS3_MTP
  std::atomic_thread_fence (std::memory_order_acquire);
#endif
  Deallocate (data);
}

Buffer::Data *
Buffer::Create (uint32_t size)
{
  NS_LOG_FUNCTION (size);
  return Allocate (size);
}
#endif /* BUFFER_FREE_LIST */

struct Buffer::Data *
Buffer::Allocate (uint32_t reqSize)
{
  NS_LOG_FUNCTION (reqSize);
  if (reqSize == 0) 
    {
      reqSize = 1;
    }
  NS_ASSERT (reqSize >= 1);
  uint32_t size = reqSize - 1 + sizeof (struct Buffer::Data);
  uint8_t *b = new uint8_t [size];
  struct Buffer::Data *data = reinterpret_cast<struct Buffer::Data*>(b);
  data->m_size = reqSize;
  data->m_count = 1;
  return data;
}

void
Buffer::Deallocate (struct Buffer::Data *data)
{
  NS_LOG_FUNCTION (data);
  NS_ASSERT (data->m_count == 0);
  uint8_t *buf = reinterpret_cast<uint8_t *> (data);
  delete [] buf;
}

Buffer::Buffer ()
{
  NS_LOG_FUNCTION (this);
  Initialize (0);
}

Buffer::Buffer (uint32_t dataSize)
{
  NS_LOG_FUNCTION (this << dataSize);
  Initialize (dataSize);
}

Buffer::Buffer (uint32_t dataSize, bool initialize)
{
  NS_LOG_FUNCTION (this << dataSize << initialize);
  if (initialize == true)
    {
      Initialize (dataSize);
    }
}

bool
Buffer::CheckInternalState (void) const
{
  NS_LOG_FUNCTION (this);
#if 0
  // If you want to modify any code in this file, enable this checking code.
  // Otherwise, there is not much point is enabling it because the
  // current implementation has been fairly seriously tested and the cost
  // of this constant checking is pretty high, even for a debug build.
  bool offsetsOk = 
    m_start <= m_zeroAreaStart &&
    m_zeroAreaStart <= m_zeroAreaEnd &&
    m_zeroAreaEnd <= m_end;
  bool dirtyOk =
    m_start >= m_data->m_dirtyStart &&
    m_end <= m_data->m_dirtyEnd;
  bool internalSizeOk = m_end - (m_zeroAreaEnd - m_zeroAreaStart) <= m_data->m_size &&
    m_start <= m_data->m_size &&
    m_zeroAreaStart <= m_data->m_size;

  bool ok = m_data->m_count > 0 && offsetsOk && dirtyOk && internalSizeOk;
  if (!ok)
    {
      LOG_INTERNAL_STATE ("check " << this << 
                          ", " << (offsetsOk ? "true" : "false") <<
                          ", " << (dirtyOk ? "true" : "false") <<
                          ", " << (internalSizeOk ? "true" : "false") << " ");
    }
  return ok;
#else
  return true;
#endif
}

void
Buffer::Initialize (uint32_t zeroSize)
{
  NS_LOG_FUNCTION (this << zeroSize);
  m_data = Buffer::Create (0);
  m_start = std::min (m_data->m_size, g_recommendedStart);
  m_maxZeroAreaStart = m_start;
  m_zeroAreaStart = m_start;
  m_zeroAreaEnd = m_zeroAreaStart + zeroSize;
  m_end = m_zeroAreaEnd;
  m_data->m_dirtyStart = m_start;
  m_data->m_dirtyEnd = m_end;
  NS_ASSERT (CheckInternalState ());
}

Buffer &
Buffer::operator = (Buffer const&o)
{
  NS_ASSERT (CheckInternalState ());
  if (m_data != o.m_data) 
    {
      // not assignment to self.
      if (m_data->m_count-- == 1) 
        {
          Recycle (m_data);
        }
      m_data = o.m_data;
      m_data->m_count++;
    }
  g_recommendedStart = std::max (g_recommendedStart, m_maxZeroAreaStart);
  m_maxZeroAreaStart = o.m_maxZeroAreaStart;
  m_zeroAreaStart = o.m_zeroAreaStart;
  m_zeroAreaEnd = o.m_zeroAreaEnd;
  m_start = o.m_start;
  m_end = o.m_end;
  NS_ASSERT (CheckInternalState ());
  return *this;
}

Buffer::~Buffer ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (CheckInternalState ());
  g_recommendedStart = std::max (g_recommendedStart, m_maxZeroAreaStart);
  if (m_data->m_count-- == 1) 
    {
      Recycle (m_data);
    }
}

uint32_t
Buffer::GetInternalSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_zeroAreaStart - m_start + m_end - m_zeroAreaEnd;
}
uint32_t
Buffer::GetInternalEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_end - (m_zeroAreaEnd - m_zeroAreaStart);
}

void
Buffer::AddAtStart (uint32_t start)
{
  NS_LOG_FUNCTION (this << start);
  NS_ASSERT (CheckInternalState ());
  bool isDirty = m_data->m_count > 1 && m_start > m_data->m_dirtyStart;
  if (m_start >= start && !isDirty)
    {
      /* enough space in the buffer and not dirty. 
       * To add: |..|
       * Before: |*****---------***|
       * After:  |***..---------***|
       */
      NS_ASSERT (m_data->m_count == 1 || m_start == m_data->m_dirtyStart);
      m_start -= start;
      // update dirty area
      m_data->m_dirtyStart = m_start;
    } 
  else
    {
      uint32_t newSize = GetInternalSize () + start;
      struct Buffer::Data *newData = Buffer::Create (newSize);
      memcpy (newData->m_data + start, m_data->m_data + m_start, GetInternalSize ());
      if (m_data->m_count-- == 1) 
        {
          Buffer::Recycle (m_data);
        }
      m_data = newData;

      int32_t delta = start - m_start;
      m_start += delta;
      m_zeroAreaStart += delta;
      m_zeroAreaEnd += delta;
      m_end += delta;
      m_start -= start;

      // update dirty area
      m_data->m_dirtyStart = m_start;
      m_data->m_dirtyEnd = m_end;
    }
  m_maxZeroAreaStart = std::max (m_maxZeroAreaStart, m_zeroAreaStart);
  LOG_INTERNAL_STATE ("add start=" << start << ", ");
  NS_ASSERT (CheckInternalState ());
}
void
Buffer::AddAtEnd (uint32_t end)
{
  NS_LOG_FUNCTION (this << end);
  NS_ASSERT (CheckInternalState ());
  bool isDirty = m_data->m_count > 1 && m_end < m_data->m_dirtyEnd;
  if (GetInternalEnd () + end <= m_data->m_size && !isDirty)
    {
      /* enough space in buffer and not dirty
       * Add:    |...|
       * Before: |**----*****|
       * After:  |**----...**|
       */
      NS_ASSERT (m_data->m_count == 1 || m_end == m_data->m_dirtyEnd);
      m_end += end;
      // update dirty area.
      m_data->m_dirtyEnd = m_end;
    } 
  else
    {
      uint32_t newSize = GetInternalSize () + end;
      struct Buffer::Data *newData = Buffer::Create (newSize);
      memcpy (newData->m_data, m_data->m_data + m_start, GetInternalSize ());
      if (m_data->m_count-- == 1) 
        {
          Buffer::Recycle (m_data);
        }
      m_data = newData;

      int32_t delta = -m_start;
      m_zeroAreaStart += delta;
      m_zeroAreaEnd += delta;
      m_end += delta;
      m_start += delta;
      m_end += end;

      // update dirty area
      m_data->m_dirtyStart = m_start;
      m_data->m_dirtyEnd = m_end;
    } 
  m_maxZeroAreaStart = std::max (m_maxZeroAreaStart, m_zeroAreaStart);
  LOG_INTERNAL_STATE ("add end=" << end << ", ");
  NS_ASSERT (CheckInternalState ());
}

void
Buffer::AddAtEnd (const Buffer &o)
{
  NS_LOG_FUNCTION (this << &o);

  if (m_data->m_count == 1 &&
      (m_end == m_zeroAreaEnd || m_zeroAreaStart == m_zeroAreaEnd) &&
      m_end == m_data->m_dirtyEnd &&
      o.m_start == o.m_zeroAreaStart &&
      o.m_zeroAreaEnd - o.m_zeroAreaStart > 0)
    {
      /**
       * This is an optimization which kicks in when
       * we attempt to aggregate two buffers which contain
       * adjacent zero areas.
       */
      if (m_zeroAreaStart == m_zeroAreaEnd)
        {
          m_zeroAreaStart = m_end;
        }
      uint32_t zeroSize = o.m_zeroAreaEnd - o.m_zeroAreaStart;
      m_zeroAreaEnd = m_end + zeroSize;
      m_end = m_zeroAreaEnd;
      m_data->m_dirtyEnd = m_zeroAreaEnd;
      uint32_t endData = o.m_end - o.m_zeroAreaEnd;
      AddAtEnd (endData);
      Buffer::Iterator dst = End ();
      dst.Prev (endData);
      Buffer::Iterator src = o.End ();
      src.Prev (endData);
      dst.Write (src, o.End ());
      NS_ASSERT (CheckInternalState ());
      return;
    }

  *this = CreateFullCopy ();
  AddAtEnd (o.GetSize ());
  Buffer::Iterator destStart = End ();
  destStart.Prev (o.GetSize ());
  destStart.Write (o.Begin (), o.End ());
  NS_ASSERT (CheckInternalState ());
}

void 
Buffer::RemoveAtStart (uint32_t start)
{
  NS_LOG_FUNCTION (this << start);
  NS_ASSERT (CheckInternalState ());
  uint32_t newStart = m_start + start;
  if (newStart <= m_zeroAreaStart)
    {
      /* only remove start of buffer 
       */
      m_start = newStart;
    }
  else if (newStart <= m_zeroAreaEnd)
    {
      /* remove start of buffer _and_ start of zero area
       */
      uint32_t delta = newStart - m_zeroAreaStart;
      m_start = m_zeroAreaStart;
      m_zeroAreaEnd -= delta;
      m_end -= delta;
    } 
  else if (newStart <= m_end)
    {
      /* remove start of buffer, complete zero area, and part
       * of end of buffer 
       */
      NS_ASSERT (m_end >= start);
      uint32_t zeroSize = m_zeroAreaEnd - m_zeroAreaStart;
      m_start = newStart - zeroSize;
      m_end -= zeroSize;
      m_zeroAreaStart = m_start;
      m_zeroAreaEnd = m_start;
    }
  else 
    {
      /* remove all buffer */
      m_end -= m_zeroAreaEnd - m_zeroAreaStart;
      m_start = m_end;
      m_zeroAreaEnd = m_end;
      m_zeroAreaStart = m_end;
    }
  m_maxZeroAreaStart = std::max (m_maxZeroAreaStart, m_zeroAreaStart);
  LOG_INTERNAL_STATE ("rem start=" << start << ", ");
  NS_ASSERT (CheckInternalState ());
}
void 
Buffer::RemoveAtEnd (uint32_t end)
{
  NS_LOG_FUNCTION (this << end);
  NS_ASSERT (CheckInternalState ());
  uint32_t newEnd = m_end - std::min (end, m_end - m_start);
  if (newEnd > m_zeroAreaEnd)
    {
      /* remove part of end of buffer */
      m_end = newEnd;
    }
  else if (newEnd > m_zeroAreaStart)
    {
      /* remove end of buffer, part of zero area */
      m_end = newEnd;
      m_zeroAreaEnd = newEnd;
    }
  else if (newEnd > m_start)
    {
      /* remove end of buffer, zero area, part of start of buffer */
      m_end = newEnd;
      m_zeroAreaEnd = newEnd;
      m_zeroAreaStart = newEnd;
    }
  else
    {
      /* remove all buffer */
      m_end = m_start;
      m_zeroAreaEnd = m_start;
      m_zeroAreaStart = m_start;
    }
  m_maxZeroAreaStart = std::max (m_maxZeroAreaStart, m_zeroAreaStart);
  LOG_INTERNAL_STATE ("rem end=" << end << ", ");
  NS_ASSERT (CheckInternalState ());
}

Buffer 
Buffer::CreateFragment (uint32_t start, uint32_t length) const
{
  NS_LOG_FUNCTION (this << start << length);
  NS_ASSERT (CheckInternalState ());
  Buffer tmp = *this;
  tmp.RemoveAtStart (start);
  tmp.RemoveAtEnd (GetSize () - (start + length));
  NS_ASSERT (CheckInternalState ());
  return tmp;
}

Buffer 
Buffer::CreateFullCopy (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (CheckInternalState ());
  if (m_zeroAreaEnd - m_zeroAreaStart != 0) 
    {
      Buffer tmp;
      tmp.AddAtStart (m_zeroAreaEnd - m_zeroAreaStart);
      tmp.Begin ().WriteU8 (0, m_zeroAreaEnd - m_zeroAreaStart);
      uint32_t dataStart = m_zeroAreaStart - m_start;
      tmp.AddAtStart (dataStart);
      tmp.Begin ().Write (m_data->m_data+m_start, dataStart);
      uint32_t dataEnd = m_end - m_zeroAreaEnd;
      tmp.AddAtEnd (dataEnd);
      Buffer::Iterator i = tmp.End ();
      i.Prev (dataEnd);
      i.Write (m_data->m_data+m_zeroAreaStart,dataEnd);
      NS_ASSERT (tmp.CheckInternalState ());
      return tmp;
    }
  NS_ASSERT (CheckInternalState ());
  return *this;
}

uint32_t 
Buffer::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  uint32_t dataStart = (m_zeroAreaStart - m_start + 3) & (~0x3);
  uint32_t dataEnd = (m_end - m_zeroAreaEnd + 3) & (~0x3);

  // total size 4-bytes for dataStart length 
  // + X number of bytes for dataStart 
  // + 4-bytes for dataEnd length 
  // + X number of bytes for dataEnd
  uint32_t sz = sizeof (uint32_t)
    + sizeof (uint32_t)
    + dataStart
    + sizeof (uint32_t)
    + dataEnd;

  return sz;
}

uint32_t
Buffer::Serialize (uint8_t* buffer, uint32_t maxSize) const
{
  NS_LOG_FUNCTION (this << &buffer << maxSize);
  uint32_t* p = reinterpret_cast<uint32_t *> (buffer);
  uint32_t size = 0;

  // Add the zero data length
  if (size + 4 <= maxSize)
    {
      size += 4;
      *p++ = m_zeroAreaEnd - m_zeroAreaStart;
    }
  else
    {
      return 0;
    }

  // Add the length of actual start data
  uint32_t dataStartLength = m_zeroAreaStart - m_start;
  if (size + 4 <= maxSize)
    {
      size += 4;
      *p++ = dataStartLength;
    }
  else
    {
      return 0;
    }

  // Add the actual data
  if (size + ((dataStartLength + 3) & (~3))  <= maxSize)
    {
      size += (dataStartLength + 3) & (~3);
      memcpy (p, m_data->m_data + m_start, dataStartLength);
      p += (((dataStartLength + 3) & (~3))/4); // Advance p, insuring 4 byte boundary
    }
  else
    {
      return 0;
    }

  // Add the length of the actual end data
  uint32_t dataEndLength = m_end - m_zeroAreaEnd;
  if (size + 4 <= maxSize)
    {
      size += 4;
      *p++ = dataEndLength;
    }
  else
    {
      return 0;
    }

  // Add the actual data
  if (size + ((dataEndLength + 3) & (~3)) <= maxSize)
    {
      // The following line is unnecessary.
      // size += (dataEndLength + 3) & (~3);
      memcpy (p, m_data->m_data+m_zeroAreaStart, dataEndLength);
      // The following line is unnecessary.
      // p += (((dataEndLength + 3) & (~3))/4); // Advance p, insuring 4 byte boundary
    }
  else
    {
      return 0;
    }

  // Serialized everything successfully
  return 1;
}

uint32_t 
Buffer::Deserialize (const uint8_t *buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << &buffer << size);
  const uint32_t* p = reinterpret_cast<const uint32_t *> (buffer);
  uint32_t sizeCheck = size-4;

  NS_ASSERT (sizeCheck >= 4);
  uint32_t zeroDataLength = *p++;
  sizeCheck -= 4;

  // Create zero bytes
  Initialize (zeroDataLength);

  // Add start data
  NS_ASSERT (sizeCheck >= 4);
  uint32_t dataStartLength = *p++;
  sizeCheck -= 4;
  AddAtStart (dataStartLength);

  NS_ASSERT (sizeCheck >= dataStartLength);
  Begin ().Write (reinterpret_cast<uint8_t *> (const_cast<uint32_t *> (p)), dataStartLength);
  p += (((dataStartLength+3)&(~3))/4); // Advance p, insuring 4 byte boundary
  sizeCheck -= ((dataStartLength+3)&(~3));

  // Add end data
  NS_ASSERT (sizeCheck >= 4);
  uint32_t dataEndLength = *p++;
  sizeCheck -= 4;
  AddAtEnd (dataEndLength);

  NS_ASSERT (sizeCheck >= dataEndLength);
  Buffer::Iterator tmp = End ();
  tmp.Prev (dataEndLength);
  tmp.Write (reinterpret_cast<uint8_t *> (const_cast<uint32_t *> (p)), dataEndLength);
  // The following line is unnecessary.
  // p += (((dataEndLength+3)&(~3))/4); // Advance p, insuring 4 byte boundary
  sizeCheck -= ((dataEndLength+3)&(~3));

  NS_ASSERT (sizeCheck == 0);
  // return zero if buffer did not 
  // contain a complete message
  return (sizeCheck != 0) ? 0 : 1;
}


void
Buffer::TransformIntoRealBuffer (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (CheckInternalState ());
  Buffer tmp = CreateFullCopy ();
  *const_cast<Buffer *> (this) = tmp;
  NS_ASSERT (CheckInternalState ());
}


uint8_t const*
Buffer::PeekData (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (CheckInternalState ());
  TransformIntoRealBuffer ();
  NS_ASSERT (CheckInternalState ());
  return m_data->m_data + m_start;
}

void
Buffer::CopyData (std::ostream *os, uint32_t size) const
{
  NS_LOG_FUNCTION (this << &os << size);
  if (size > 0)
    {
      uint32_t tmpsize = std::min (m_zeroAreaStart-m_start, size);
      os->write ((const char*)(m_data->m_data + m_start), tmpsize);
      if (size > tmpsize) 
        { 
          size -= m_zeroAreaStart-m_start;
          tmpsize = std::min (m_zeroAreaEnd - m_zeroAreaStart, size);
          uint32_t left = tmpsize;
          while (left > 0)
            {
              uint32_t toWrite = std::min (left, g_zeroes.size);
              os->write (g_zeroes.buffer, toWrite);
              left -= toWrite;
            }
          if (size > tmpsize)
            {
              size -= tmpsize;
              tmpsize = std::min (m_end - m_zeroAreaEnd, size);
              os->write ((const char*)(m_data->m_data + m_zeroAreaStart), tmpsize); 
            }
        }
    }
}

uint32_t 
Buffer::CopyData (uint8_t *buffer, uint32_t size) const
{
  NS_LOG_FUNCTION (this << &buffer << size);
  uint32_t originalSize = size;
  if (size > 0)
    {
      uint32_t tmpsize = std::min (m_zeroAreaStart-m_start, size);
      memcpy (buffer, (const char*)(m_data->m_data + m_start), tmpsize);
      buffer += tmpsize;
      size -= tmpsize;
      if (size > 0) 
        { 
          tmpsize = std::min (m_zeroAreaEnd - m_zeroAreaStart, size);
          uint32_t left = tmpsize;
          while (left > 0)
            {
              uint32_t toWrite = std::min (left, g_zeroes.size);
              memcpy (buffer, g_zeroes.buffer, toWrite);
              left -= toWrite;
              buffer += toWrite;
            }
          size -= tmpsize;
          if (size > 0)
            {
              tmpsize = std::min (m_end - m_zeroAreaEnd, size);
              memcpy (buffer, (const char*)(m_data->m_data + m_zeroAreaStart), tmpsize);
              size -= tmpsize;
            }
        }
    }
  return originalSize - size;
}

/******************************************************
 *            The buffer iterator below.
 ******************************************************/


uint32_t
Buffer::Iterator::GetDistanceFrom (Iterator const &o) const
{
  NS_LOG_FUNCTION (this << &o);
  NS_ASSERT (m_data == o.m_data);
  int32_t diff = m_current - o.m_current;
  if (diff < 0)
    {
      return -diff;
    }
  else
    {
      return diff;
    }
}

bool 
Buffer::Iterator::IsEnd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_current == m_dataEnd;
}
bool 
Buffer::Iterator::IsStart (void) const
{
  NS_LOG_FUNCTION (this);
  return m_current == m_dataStart;
}

bool 
Buffer::Iterator::CheckNoZero (uint32_t start, uint32_t end) const
{
  NS_LOG_FUNCTION (this << &start << &end);
  return !(start < m_dataStart ||
           end > m_dataEnd ||
           (end > m_zeroStart && start < m_zeroEnd && m_zeroEnd != m_zeroStart && start != end)
           );
}
bool 
Buffer::Iterator::Check (uint32_t i) const
{
  NS_LOG_FUNCTION (this << &i);
  return i >= m_dataStart && 
         !(i >= m_zeroStart && i < m_zeroEnd) &&
         i <= m_dataEnd;
}


void 
Buffer::Iterator::Write (Iterator start, Iterator end)
{
  NS_LOG_FUNCTION (this << &start << &end);
  NS_ASSERT (start.m_data == end.m_data);
  NS_ASSERT (start.m_current <= end.m_current);
  NS_ASSERT (start.m_zeroStart == end.m_zeroStart);
  NS_ASSERT (start.m_zeroEnd == end.m_zeroEnd);
  NS_ASSERT (m_data != start.m_data);
  uint32_t size = end.m_current - start.m_current;
  NS_ASSERT_MSG (CheckNoZero (m_current, m_current + size),
                 GetWriteErrorMessage ());
  if (start.m_current <= start.m_zeroStart)
    {
      uint32_t toCopy = std::min (size, start.m_zeroStart - start.m_current);
      memcpy (&m_data[m_current], &start.m_data[start.m_current], toCopy);
      start.m_current += toCopy;
      m_current += toCopy;
      size -= toCopy;
    }
  if (start.m_current <= start.m_zeroEnd)
    {
      uint32_t toCopy = std::min (size, start.m_zeroEnd - start.m_current);
      memset (&m_data[m_current], 0, toCopy);
      start.m_current += toCopy;
      m_current += toCopy;
      size -= toCopy;
    }
  uint32_t toCopy = std::min (size, start.m_dataEnd - start.m_current);
  uint8_t *from = &start.m_data[start.m_current - (start.m_zeroEnd-start.m_zeroStart)];
  uint8_t *to = &m_data[m_current];
  memcpy (to, from, toCopy);
  m_current += toCopy;
}

void 
Buffer::Iterator::WriteU16 (uint16_t data)
{
  NS_LOG_FUNCTION (this << data);
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
}
void 
Buffer::Iterator::WriteU32 (uint32_t data)
{
  NS_LOG_FUNCTION (this << data);
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
}
void 
Buffer::Iterator::WriteU64 (uint64_t data)
{
  NS_LOG_FUNCTION (this << data);
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
  data >>= 8;
  WriteU8 (data & 0xff);
}
void 
Buffer::Iterator::WriteHtolsbU16 (uint16_t data)
{
  NS_LOG_FUNCTION (this << data);
  WriteU8 ((data >> 0) & 0xff);
  WriteU8 ((data >> 8) & 0xff);
}
void 
Buffer::Iterator::WriteHtolsbU32 (uint32_t data)
{
  NS_LOG_FUNCTION (this << data);
  WriteU8 ((data >> 0) & 0xff);
  WriteU8 ((data >> 8) & 0xff);
  WriteU8 ((data >> 16) & 0xff);
  WriteU8 ((data >> 24) & 0xff);
}
void 
Buffer::Iterator::WriteHtolsbU64 (uint64_t data)
{
  NS_LOG_FUNCTION (this << data);
  WriteU8 ((data >> 0) & 0xff);
  WriteU8 ((data >> 8) & 0xff);
  WriteU8 ((data >> 16) & 0xff);
  WriteU8 ((data >> 24) & 0xff);
  WriteU8 ((data >> 32) & 0xff);
  WriteU8 ((data >> 40) & 0xff);
  WriteU8 ((data >> 48) & 0xff);
  WriteU8 ((data >> 56) & 0xff);
}

void 
Buffer::Iterator::WriteHtonU64 (uint64_t data)
{
  NS_LOG_FUNCTION (this << data);
  WriteU8 ((data >> 56) & 0xff);
  WriteU8 ((data >> 48) & 0xff);
  WriteU8 ((data >> 40) & 0xff);
  WriteU8 ((data >> 32) & 0xff);
  WriteU8 ((data >> 24) & 0xff);
  WriteU8 ((data >> 16) & 0xff);
  WriteU8 ((data >> 8) & 0xff);
  WriteU8 ((data >> 0) & 0xff);
}
void 
Buffer::Iterator::Write (uint8_t const*buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << &buffer << size);
  NS_ASSERT_MSG (CheckNoZero (m_current, size),
                 GetWriteErrorMessage ());
  uint8_t *to;
  if (m_current <= m_zeroStart)
    {
      to = &m_data[m_current];
    }
  else
    {
      to = &m_data[m_current - (m_zeroEnd - m_zeroStart)];
    }
  memcpy (to, buffer, size);
  m_current += size;
}

uint32_t 
Buffer::Iterator::ReadU32 (void)
{
  NS_LOG_FUNCTION (this);
  uint8_t byte0 = ReadU8 ();
  uint8_t byte1 = ReadU8 ();
  uint8_t byte2 = ReadU8 ();
  uint8_t byte3 = ReadU8 ();
  uint32_t data = byte3;
  data <<= 8;
  data |= byte2;
  data <<= 8;
  data |= byte1;
  data <<= 8;
  data |= byte0;
  return data;
}
uint64_t 
Buffer::Iterator::ReadU64 (void)
{
  NS_LOG_FUNCTION (this);
  uint8_t byte0 = ReadU8 ();
  uint8_t byte1 = ReadU8 ();
  uint8_t byte2 = ReadU8 ();
  uint8_t byte3 = ReadU8 ();
  uint8_t byte4 = ReadU8 ();
  uint8_t byte5 = ReadU8 ();
  uint8_t byte6 = ReadU8 ();
  uint8_t byte7 = ReadU8 ();
  uint64_t data = byte7;
  data <<= 8;
  data |= byte6;
  data <<= 8;
  data |= byte5;
  data <<= 8;
  data |= byte4;
  data <<= 8;
  data |= byte3;
  data <<= 8;
  data |= byte2;
  data <<= 8;
  data |= byte1;
  data <<= 8;
  data |= byte0;

  return data;
}
uint16_t 
Buffer::Iterator::SlowReadNtohU16 (void)
{
  NS_LOG_FUNCTION (this);
  uint16_t retval = 0;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  return retval;
}
uint32_t 
Buffer::Iterator::SlowReadNtohU32 (void)
{
  NS_LOG_FUNCTION (this);
  uint32_t retval = 0;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  return retval;
}
uint64_t 
Buffer::Iterator::ReadNtohU64 (void)
{
  NS_LOG_FUNCTION (this);
  uint64_t retval = 0;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  retval <<= 8;
  retval |= ReadU8 ();
  return retval;
}
uint16_t 
Buffer::Iterator::ReadLsbtohU16 (void)
{
  NS_LOG_FUNCTION (this);
  uint8_t byte0 = ReadU8 ();
  uint8_t byte1 = ReadU8 ();
  uint16_t data = byte1;
  data <<= 8;
  data |= byte0;
  return data;
}
uint32_t 
Buffer::Iterator::ReadLsbtohU32 (void)
{
  NS_LOG_FUNCTION (this);
  uint8_t byte0 = ReadU8 ();
  uint8_t byte1 = ReadU8 ();
  uint8_t byte2 = ReadU8 ();
  uint8_t byte3 = ReadU8 ();
  uint32_t data = byte3;
  data <<= 8;
  data |= byte2;
  data <<= 8;
  data |= byte1;
  data <<= 8;
  data |= byte0;
  return data;
}
uint64_t 
Buffer::Iterator::ReadLsbtohU64 (void)
{
  NS_LOG_FUNCTION (this);
  uint8_t byte0 = ReadU8 ();
  uint8_t byte1 = ReadU8 ();
  uint8_t byte2 = ReadU8 ();
  uint8_t byte3 = ReadU8 ();
  uint8_t byte4 = ReadU8 ();
  uint8_t byte5 = ReadU8 ();
  uint8_t byte6 = ReadU8 ();
  uint8_t byte7 = ReadU8 ();
  uint64_t data = byte7;
  data <<= 8;
  data |= byte6;
  data <<= 8;
  data |= byte5;
  data <<= 8;
  data |= byte4;
  data <<= 8;
  data |= byte3;
  data <<= 8;
  data |= byte2;
  data <<= 8;
  data |= byte1;
  data <<= 8;
  data |= byte0;

  return data;
}
void 
Buffer::Iterator::Read (uint8_t *buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << &buffer << size);
  for (uint32_t i = 0; i < size; i++)
    {
      buffer[i] = ReadU8 ();
    }
}

uint16_t
Buffer::Iterator::CalculateIpChecksum (uint16_t size)
{
  NS_LOG_FUNCTION (this << size);
  return CalculateIpChecksum (size, 0);
}

uint16_t
Buffer::Iterator::CalculateIpChecksum (uint16_t size, uint32_t initialChecksum)
{
  NS_LOG_FUNCTION (this << size << initialChecksum);
  /* see RFC 1071 to understand this code. */
  uint32_t sum = initialChecksum;

  for (int j = 0; j < size/2; j++)
    sum += ReadU16 ();

  if (size & 1)
    sum += ReadU8 ();

  while (sum >> 16)
    sum = (sum & 0xffff) + (sum >> 16);
  return ~sum;
}

uint32_t 
Buffer::Iterator::GetSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dataEnd - m_dataStart;
}

uint32_t
Buffer::Iterator::GetRemainingSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dataEnd - m_current;
}


std::string 
Buffer::Iterator::GetReadErrorMessage (void) const
{
  NS_LOG_FUNCTION (this);
  std::string str = "You have attempted to read beyond the bounds of the "
    "available buffer space. This usually indicates that a "
    "Header::Deserialize or Trailer::Deserialize method "
    "is trying to read data which was not written by "
    "a Header::Serialize or Trailer::Serialize method. "
    "In short: check the code of your Serialize and Deserialize "
    "methods.";
  return str;
}
std::string 
Buffer::Iterator::GetWriteErrorMessage (void) const
{
  NS_LOG_FUNCTION (this);
  std::string str;
  if (m_current < m_dataStart)
    {
      str = "You have attempted to write before the start of the available "
        "buffer space. This usually indicates that Trailer::GetSerializedSize "
        "returned a size which is too small compared to what Trailer::Serialize "
        "is actually using.";
    }
  else if (m_current >= m_dataEnd)
    {
      str = "You have attempted to write after the end of the available "
        "buffer space. This usually indicates that Header::GetSerializedSize "
        "returned a size which is too small compared to what Header::Serialize "
        "is actually using.";
    }
  else
    {
      NS_ASSERT (m_current >= m_zeroStart && m_current < m_zeroEnd);
      str = "You have attempted to write inside the payload area of the "
        "buffer. This usually indicates that your Serialize method uses more "
        "buffer space than what your GetSerialized method returned.";
    }
  return str;
}

uint8_t* Buffer::GetBuffer() const{
	return m_data->m_data + m_start;
}

} // namespace ns3


