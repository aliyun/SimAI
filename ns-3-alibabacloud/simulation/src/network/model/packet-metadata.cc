/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006,2007 INRIA
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
#include <utility>
#include <list>
#include "ns3/assert.h"
#include "ns3/fatal-error.h"
#include "ns3/log.h"
#include "packet-metadata.h"
#include "buffer.h"
#include "header.h"
#include "trailer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketMetadata");

bool PacketMetadata::m_enable = false;
bool PacketMetadata::m_enableChecking = false;
bool PacketMetadata::m_metadataSkipped = false;
uint32_t PacketMetadata::m_maxSize = 0;
uint16_t PacketMetadata::m_chunkUid = 0;
PacketMetadata::DataFreeList PacketMetadata::m_freeList;

PacketMetadata::DataFreeList::~DataFreeList ()
{
  NS_LOG_FUNCTION (this);
  for (iterator i = begin (); i != end (); i++)
    {
      PacketMetadata::Deallocate (*i);
    }
  PacketMetadata::m_enable = false;
}

void 
PacketMetadata::Enable (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  NS_ASSERT_MSG (!m_metadataSkipped,
                 "Error: attempting to enable the packet metadata "
                 "subsystem too late in the simulation, which is not allowed.\n"
                 "A common cause for this problem is to enable ASCII tracing "
                 "after sending any packets.  One way to fix this problem is "
                 "to call ns3::PacketMetadata::Enable () near the beginning of"
                 " the program, before any packets are sent.");
#ifndef NS3_MTP
  m_enable = true;
#endif
}

void 
PacketMetadata::EnableChecking (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  Enable ();
#ifndef NS3_MTP
  m_enableChecking = true;
#endif
}

void
PacketMetadata::ReserveCopy (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  struct PacketMetadata::Data *newData = PacketMetadata::Create (m_used + size);
  memcpy (newData->m_data, m_data->m_data, m_used);
  newData->m_dirtyEnd = m_used;
  if (m_data->m_count-- == 1) 
    {
      PacketMetadata::Recycle (m_data);
    }
  m_data = newData;
  if (m_head != 0xffff)
    {
      uint8_t *start;
      NS_ASSERT (m_tail != 0xffff);
      // clear the next field of the tail
      start = &m_data->m_data[m_tail];
      Append16 (0xffff, start);
      // clear the prev field of the head
      start = &m_data->m_data[m_head] + 2;
      Append16 (0xffff, start);
    }
}
void
PacketMetadata::Reserve (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  NS_ASSERT (m_data != 0);
  if (m_data->m_size >= m_used + size &&
      (m_head == 0xffff ||
       m_data->m_count == 1 ||
       m_data->m_dirtyEnd == m_used))
    {
      /* enough room, not dirty. */
    }
  else 
    {
      /* (enough room and dirty) or (not enough room) */
      ReserveCopy (size);
    }
}

bool
PacketMetadata::IsSharedPointerOk (uint16_t pointer) const
{
  NS_LOG_FUNCTION (this << pointer);
  bool ok = pointer == 0xffff || pointer <= m_data->m_size;
  return ok;
}
bool
PacketMetadata::IsPointerOk (uint16_t pointer) const
{
  NS_LOG_FUNCTION (this << pointer);
  bool ok = pointer == 0xffff || pointer <= m_used;
  return ok;
}

bool
PacketMetadata::IsStateOk (void) const
{
  NS_LOG_FUNCTION (this);
  bool ok = m_used <= m_data->m_size;
  ok &= IsPointerOk (m_head);
  ok &= IsPointerOk (m_tail);
  uint16_t current = m_head;
  while (ok && current != 0xffff)
    {
      struct PacketMetadata::SmallItem item;
      PacketMetadata::ExtraItem extraItem;
      ReadItems (current, &item, &extraItem);
      ok &= IsSharedPointerOk (item.next);
      ok &= IsSharedPointerOk (item.prev);
      if (current != m_head)
        {
          ok &= IsPointerOk (item.prev);
        }
      if (current != m_tail)
        {
          ok &= IsPointerOk (item.next);
        }
      if (current == m_tail)
        {
          break;
        }
      current = item.next;
    }
  return ok;
}

uint32_t 
PacketMetadata::GetUleb128Size (uint32_t value) const
{
  NS_LOG_FUNCTION (this << value);
  if (value < 0x80)
    {
      return 1;
    }
  if (value < 0x4000)
    {
      return 2;
    }
  if (value < 0x200000)
    {
      return 3;
    }
  if (value < 0x10000000)
    {
      return 4;
    }
  return 5;
}
uint32_t
PacketMetadata::ReadUleb128 (const uint8_t **pBuffer) const
{
  NS_LOG_FUNCTION (this << &pBuffer);
  const uint8_t *buffer = *pBuffer;
  uint32_t result;
  uint8_t byte;

  byte = buffer[0];
  result = (byte & (~0x80));
  if (!(byte & 0x80))
    {
      *pBuffer = buffer + 1;
      return result;
    }
  byte = buffer[1];
  result |= (byte & (~0x80)) << 7;
  if (!(byte & 0x80))
    {
      *pBuffer = buffer + 2;
      return result;
    }
  byte = buffer[2];
  result |= (byte & (~0x80)) << 14;
  if (!(byte & 0x80))
    {
      *pBuffer = buffer + 3;
      return result;
    }
  byte = buffer[3];
  result |= (byte & (~0x80)) << 21;
  if (!(byte & 0x80))
    {
      *pBuffer = buffer + 4;
      return result;
    }
  byte = buffer[4];
  result |= (byte & (~0x80)) << 28;
  if (!(byte & 0x80))
    {
      *pBuffer = buffer + 5;
      return result;
    }
  /* This means that the LEB128 number was not valid.
   * ie: the last (5th) byte did not have the high-order bit zeroed.
   */
  NS_ASSERT (false);
  return 0;
}

void
PacketMetadata::Append16 (uint16_t value, uint8_t *buffer)
{
  NS_LOG_FUNCTION (this << value << &buffer);
  buffer[0] = value & 0xff;
  value >>= 8;
  buffer[1] = value;
}
void
PacketMetadata::Append32 (uint32_t value,  uint8_t *buffer)
{
  NS_LOG_FUNCTION (this << value << &buffer);
  buffer[0] = value & 0xff;
  buffer[1] = (value >> 8) & 0xff;
  buffer[2] = (value >> 16) & 0xff;
  buffer[3] = (value >> 24) & 0xff;
}

void
PacketMetadata::AppendValueExtra (uint32_t value, uint8_t *buffer)
{
  NS_LOG_FUNCTION (this << value << &buffer);
  if (value < 0x200000)
    {
      uint8_t byte = value & (~0x80);
      buffer[0] = 0x80 | byte;
      value >>= 7;
      byte = value & (~0x80);
      buffer[1] = 0x80 | byte;
      value >>= 7;
      buffer[2] = value;
      return;
    }
  if (value < 0x10000000)
    {
      uint8_t byte = value & (~0x80);
      buffer[0] = 0x80 | byte;
      value >>= 7;
      byte = value & (~0x80);
      buffer[1] = 0x80 | byte;
      value >>= 7;
      byte = value & (~0x80);
      buffer[2] = 0x80 | byte;
      value >>= 7;
      buffer[3] = value;
      return;
    }
  {
    uint8_t byte = value & (~0x80);
    buffer[0] = 0x80 | byte;
    value >>= 7;
    byte = value & (~0x80);
    buffer[1] = 0x80 | byte;
    value >>= 7;
    byte = value & (~0x80);
    buffer[2] = 0x80 | byte;
    value >>= 7;
    byte = value & (~0x80);
    buffer[3] = 0x80 | byte;
    value >>= 7;
    buffer[4] = value;
  }
}

void
PacketMetadata::AppendValue (uint32_t value, uint8_t *buffer)
{
  NS_LOG_FUNCTION (this << value << &buffer);
  if (value < 0x80)
    {
      buffer[0] = value;
      return;
    }
  if (value < 0x4000)
    {
      uint8_t byte = value & (~0x80);
      buffer[0] = 0x80 | byte;
      value >>= 7;
      buffer[1] = value;
      return;
    }
  AppendValueExtra (value, buffer);
}

void
PacketMetadata::UpdateTail (uint16_t written)
{
  NS_LOG_FUNCTION (this << written);
  if (m_head == 0xffff)
    {
      NS_ASSERT (m_tail == 0xffff);
      m_head = m_used;
      m_tail = m_used;
    } 
  else
    {
      NS_ASSERT (m_tail != 0xffff);
      // overwrite the next field of the previous tail of the list.
      uint8_t *previousTail = &m_data->m_data[m_tail];
      Append16 (m_used, previousTail);
      // update the tail of the list to the new node.
      m_tail = m_used;
    }
  NS_ASSERT (m_tail != 0xffff);
  NS_ASSERT (m_head != 0xffff);
  NS_ASSERT (written >= 8);
  m_used += written;
  m_data->m_dirtyEnd = m_used;
}


void
PacketMetadata::UpdateHead (uint16_t written)
{
  NS_LOG_FUNCTION (this << written);
  if (m_head == 0xffff)
    {
      NS_ASSERT (m_tail == 0xffff);
      m_head = m_used;
      m_tail = m_used;
    } 
  else
    {
      NS_ASSERT (m_head != 0xffff);
      // overwrite the prev field of the previous head of the list.
      uint8_t *previousHead = &m_data->m_data[m_head + 2];
      Append16 (m_used, previousHead);
      // update the head of list to the new node.
      m_head = m_used;
    }
  NS_ASSERT (m_tail != 0xffff);
  NS_ASSERT (m_head != 0xffff);
  NS_ASSERT (written >= 8);
  m_used += written;
  m_data->m_dirtyEnd = m_used;
}

uint16_t
PacketMetadata::AddSmall (const struct PacketMetadata::SmallItem *item)
{
  NS_LOG_FUNCTION (this << item->next << item->prev << item->typeUid << item->size << item->chunkUid);
  NS_ASSERT (m_data != 0);
  NS_ASSERT (m_used != item->prev && m_used != item->next);
  uint32_t typeUidSize = GetUleb128Size (item->typeUid);
  uint32_t sizeSize = GetUleb128Size (item->size);
  uint32_t n =  2 + 2 + typeUidSize + sizeSize + 2;
  if (m_used + n > m_data->m_size ||
      (m_head != 0xffff &&
       m_data->m_count != 1 &&
       m_used != m_data->m_dirtyEnd))
    {
      ReserveCopy (n);
    }
  uint8_t *buffer = &m_data->m_data[m_used];
  Append16 (item->next, buffer);
  buffer += 2;
  Append16 (item->prev, buffer);
  buffer += 2;
  AppendValue (item->typeUid, buffer);
  buffer += typeUidSize;
  AppendValue (item->size, buffer);
  buffer += sizeSize;
  Append16 (item->chunkUid, buffer);
  return n;
}

uint16_t
PacketMetadata::AddBig (uint32_t next, uint32_t prev, 
                        const PacketMetadata::SmallItem *item,
                        const PacketMetadata::ExtraItem *extraItem)
{
  NS_LOG_FUNCTION (this << next << prev <<
                   item->next << item->prev << item->typeUid << item->size << item->chunkUid <<
                   extraItem->fragmentStart << extraItem->fragmentEnd << extraItem->packetUid);
  NS_ASSERT (m_data != 0);
  uint32_t typeUid = ((item->typeUid & 0x1) == 0x1) ? item->typeUid : item->typeUid+1;
  NS_ASSERT (m_used != prev && m_used != next);

  uint32_t typeUidSize = GetUleb128Size (typeUid);
  uint32_t sizeSize = GetUleb128Size (item->size);
  uint32_t fragStartSize = GetUleb128Size (extraItem->fragmentStart);
  uint32_t fragEndSize = GetUleb128Size (extraItem->fragmentEnd);
  uint32_t n = 2 + 2 + typeUidSize + sizeSize + 2 + fragStartSize + fragEndSize + 4;

  if (m_used + n > m_data->m_size ||
      (m_head != 0xffff &&
       m_data->m_count != 1 &&
       m_used != m_data->m_dirtyEnd))
    {
      ReserveCopy (n);
    }

  uint8_t *buffer = &m_data->m_data[m_used];

  Append16 (next, buffer);
  buffer += 2;
  Append16 (prev, buffer);
  buffer += 2;
  AppendValue (typeUid, buffer);
  buffer += typeUidSize;
  AppendValue (item->size, buffer);
  buffer += sizeSize;
  Append16 (item->chunkUid, buffer);
  buffer += 2;
  AppendValue (extraItem->fragmentStart, buffer);
  buffer += fragStartSize;
  AppendValue (extraItem->fragmentEnd, buffer);
  buffer += fragEndSize;
  Append32 (extraItem->packetUid, buffer);

  return n;
}

void
PacketMetadata::ReplaceTail (PacketMetadata::SmallItem *item, 
                             PacketMetadata::ExtraItem *extraItem,
                             uint32_t available)
{
  NS_LOG_FUNCTION (this <<
                   item->next << item->prev << item->typeUid << item->size << item->chunkUid <<
                   extraItem->fragmentStart << extraItem->fragmentEnd << extraItem->packetUid <<
                   available);

  NS_ASSERT (m_data != 0);
  /* If the tail we want to replace is located at the end of the data array,
   * and if there is extra room at the end of this array, then, 
   * we can try to use that extra space to avoid falling in the slow
   * path below.
   */
  if (m_tail + available == m_used &&
      m_used == m_data->m_dirtyEnd)
    {
      available = m_data->m_size - m_tail;
    }

  uint32_t typeUid = ((item->typeUid & 0x1) == 0x1) ? item->typeUid : item->typeUid+1;
  uint32_t typeUidSize = GetUleb128Size (typeUid);
  uint32_t sizeSize = GetUleb128Size (item->size);
  uint32_t fragStartSize = GetUleb128Size (extraItem->fragmentStart);
  uint32_t fragEndSize = GetUleb128Size (extraItem->fragmentEnd);
  uint32_t n = 2 + 2 + typeUidSize + sizeSize + 2 + fragStartSize + fragEndSize + 4;

  if (available >= n &&
      m_data->m_count == 1)
    {
      uint8_t *buffer = &m_data->m_data[m_tail];
      Append16 (item->next, buffer);
      buffer += 2;
      Append16 (item->prev, buffer);
      buffer += 2;
      AppendValue (typeUid, buffer);
      buffer += typeUidSize;
      AppendValue (item->size, buffer);
      buffer += sizeSize;
      Append16 (item->chunkUid, buffer);
      buffer += 2;
      AppendValue (extraItem->fragmentStart, buffer);
      buffer += fragStartSize;
      AppendValue (extraItem->fragmentEnd, buffer);
      buffer += fragEndSize;
      Append32 (extraItem->packetUid, buffer);
      m_used = std::max (m_used, (uint16_t)(buffer - &m_data->m_data[0]));
      m_data->m_dirtyEnd = m_used;
      return;
    }

  /* Below is the slow path which is hit if the new tail we want 
   * to append is bigger than the previous tail.
   */

  // create a copy of the packet without its tail.
  PacketMetadata h (m_packetUid, 0);
  uint16_t current = m_head;
  while (current != 0xffff && current != m_tail)
    {
      struct PacketMetadata::SmallItem tmpItem;
      PacketMetadata::ExtraItem tmpExtraItem;
      ReadItems (current, &tmpItem, &tmpExtraItem);
      uint16_t written = h.AddBig (0xffff, h.m_tail, 
                                   &tmpItem, &tmpExtraItem);
      h.UpdateTail (written);
      current = tmpItem.next;
    }
  // append new tail.
  uint16_t written = h.AddBig (0xffff, h.m_tail, item, extraItem);
  h.UpdateTail (written);

  *this = h;
}


uint32_t
PacketMetadata::ReadItems (uint16_t current, 
                           struct PacketMetadata::SmallItem *item,
                           struct PacketMetadata::ExtraItem *extraItem) const
{
  NS_LOG_FUNCTION (this << current << item->chunkUid << item->prev << item->next << item->size <<
                        item->typeUid << extraItem->fragmentEnd << extraItem->fragmentStart <<
                        extraItem->packetUid);
  NS_ASSERT (current <= m_data->m_size);
  const uint8_t *buffer = &m_data->m_data[current];
  item->next = buffer[0];
  item->next |= (buffer[1]) << 8;
  item->prev = buffer[2];
  item->prev |= (buffer[3]) << 8;
  buffer += 4;
  item->typeUid = ReadUleb128 (&buffer);
  item->size = ReadUleb128 (&buffer);
  item->chunkUid = buffer[0];
  item->chunkUid |= (buffer[1]) << 8;
  buffer += 2;

  bool isExtra = (item->typeUid & 0x1) == 0x1;
  if (isExtra)
    {
      extraItem->fragmentStart = ReadUleb128 (&buffer);
      extraItem->fragmentEnd = ReadUleb128 (&buffer);
      extraItem->packetUid = buffer[0];
      extraItem->packetUid |= buffer[1] << 8;
      extraItem->packetUid |= buffer[2] << 16;
      extraItem->packetUid |= buffer[3] << 24;
      buffer += 4;
    }
  else
    {
      extraItem->fragmentStart = 0;
      extraItem->fragmentEnd = item->size;
      extraItem->packetUid = m_packetUid;
    }
  NS_ASSERT (buffer <= &m_data->m_data[m_data->m_size]);
  return buffer - &m_data->m_data[current];
}

struct PacketMetadata::Data *
PacketMetadata::Create (uint32_t size)
{
  NS_LOG_FUNCTION (size);
  NS_LOG_LOGIC ("create size="<<size<<", max="<<m_maxSize);
  if (size > m_maxSize)
    {
      m_maxSize = size;
    }
  while (!m_freeList.empty ()) 
    {
      struct PacketMetadata::Data *data = m_freeList.back ();
      m_freeList.pop_back ();
      if (data->m_size >= size) 
        {
          NS_LOG_LOGIC ("create found size="<<data->m_size);
          data->m_count = 1;
          return data;
        }
      NS_LOG_LOGIC ("create dealloc size="<<data->m_size);
      PacketMetadata::Deallocate (data);
    }
  NS_LOG_LOGIC ("create alloc size="<<m_maxSize);
  return PacketMetadata::Allocate (m_maxSize);
}

void
PacketMetadata::Recycle (struct PacketMetadata::Data *data)
{
  NS_LOG_FUNCTION (data);
#ifdef NS3_MTP
  std::atomic_thread_fence (std::memory_order_acquire);
#endif
  if (!m_enable)
    {
      PacketMetadata::Deallocate (data);
      return;
    } 
  NS_LOG_LOGIC ("recycle size="<<data->m_size<<", list="<<m_freeList.size ());
  NS_ASSERT (data->m_count == 0);
  if (m_freeList.size () > 1000 ||
      data->m_size < m_maxSize) 
    {
      PacketMetadata::Deallocate (data);
    } 
  else 
    {
      m_freeList.push_back (data);
    }
}

struct PacketMetadata::Data *
PacketMetadata::Allocate (uint32_t n)
{
  NS_LOG_FUNCTION (n);
  uint32_t size = sizeof (struct Data);
  if (n <= PACKET_METADATA_DATA_M_DATA_SIZE)
    {
      n = PACKET_METADATA_DATA_M_DATA_SIZE;
    }
  size += n - PACKET_METADATA_DATA_M_DATA_SIZE;
  uint8_t *buf = new uint8_t [size];
  struct PacketMetadata::Data *data = (struct PacketMetadata::Data *)buf;
  data->m_size = n;
  data->m_count = 1;
  data->m_dirtyEnd = 0;
  return data;
}
void 
PacketMetadata::Deallocate (struct PacketMetadata::Data *data)
{
  NS_LOG_FUNCTION (data);
  uint8_t *buf = (uint8_t *)data;
  delete [] buf;
}


PacketMetadata 
PacketMetadata::CreateFragment (uint32_t start, uint32_t end) const
{
  NS_LOG_FUNCTION (this << start << end);
  PacketMetadata fragment = *this;
  fragment.RemoveAtStart (start);
  fragment.RemoveAtEnd (end);
  return fragment;
}

void 
PacketMetadata::AddHeader (const Header &header, uint32_t size)
{
  NS_LOG_FUNCTION (this << &header << size);
  NS_ASSERT (IsStateOk ());
  uint32_t uid = header.GetInstanceTypeId ().GetUid () << 1;
  DoAddHeader (uid, size);
  NS_ASSERT (IsStateOk ());
}
void
PacketMetadata::DoAddHeader (uint32_t uid, uint32_t size)
{
  NS_LOG_FUNCTION (this << uid << size);
  if (!m_enable)
    {
      m_metadataSkipped = true;
      return;
    }

  struct PacketMetadata::SmallItem item;
  item.next = m_head;
  item.prev = 0xffff;
  item.typeUid = uid;
  item.size = size;
  item.chunkUid = m_chunkUid;
  m_chunkUid++;
  uint16_t written = AddSmall (&item);
  UpdateHead (written);
}
void 
PacketMetadata::RemoveHeader (const Header &header, uint32_t size)
{
  uint32_t uid = header.GetInstanceTypeId ().GetUid () << 1;
  NS_LOG_FUNCTION (this << &header << size);
  NS_ASSERT (IsStateOk ());
  if (!m_enable) 
    {
      m_metadataSkipped = true;
      return;
    }
  struct PacketMetadata::SmallItem item;
  struct PacketMetadata::ExtraItem extraItem;
  uint32_t read = ReadItems (m_head, &item, &extraItem);
  if ((item.typeUid & 0xfffffffe) != uid ||
      item.size != size)
    {
      if (m_enableChecking)
        {
          NS_FATAL_ERROR ("Removing unexpected header.");
        }
      return;
    }
  else if (item.typeUid != uid &&
           (extraItem.fragmentStart != 0 ||
            extraItem.fragmentEnd != size))
    {
      if (m_enableChecking)
        {
          NS_FATAL_ERROR ("Removing incomplete header.");
        }
      return;
    }
  if (m_head + read == m_used)
    {
      m_used = m_head;
    }
  if (m_head == m_tail)
    {
      m_head = 0xffff;
      m_tail = 0xffff;
    }
  else
    {
      m_head = item.next;
    }
  NS_ASSERT (IsStateOk ());
}
void 
PacketMetadata::AddTrailer (const Trailer &trailer, uint32_t size)
{
  uint32_t uid = trailer.GetInstanceTypeId ().GetUid () << 1;
  NS_LOG_FUNCTION (this << &trailer << size);
  NS_ASSERT (IsStateOk ());
  if (!m_enable)
    {
      m_metadataSkipped = true;
      return;
    }
  struct PacketMetadata::SmallItem item;
  item.next = 0xffff;
  item.prev = m_tail;
  item.typeUid = uid;
  item.size = size;
  item.chunkUid = m_chunkUid;
  m_chunkUid++;
  uint16_t written = AddSmall (&item);
  UpdateTail (written);
  NS_ASSERT (IsStateOk ());
}
void 
PacketMetadata::RemoveTrailer (const Trailer &trailer, uint32_t size)
{
  uint32_t uid = trailer.GetInstanceTypeId ().GetUid () << 1;
  NS_LOG_FUNCTION (this << &trailer << size);
  NS_ASSERT (IsStateOk ());
  if (!m_enable) 
    {
      m_metadataSkipped = true;
      return;
    }
  struct PacketMetadata::SmallItem item;
  struct PacketMetadata::ExtraItem extraItem;
  uint32_t read = ReadItems (m_tail, &item, &extraItem);
  if ((item.typeUid & 0xfffffffe) != uid ||
      item.size != size)
    {
      if (m_enableChecking)
        {
          NS_FATAL_ERROR ("Removing unexpected trailer.");
        }
      return;
    }
  else if (item.typeUid != uid &&
           (extraItem.fragmentStart != 0 ||
            extraItem.fragmentEnd != size))
    {
      if (m_enableChecking)
        {
          NS_FATAL_ERROR ("Removing incomplete trailer.");
        }
      return;
    }
  if (m_tail + read == m_used)
    {
      m_used = m_tail;
    }
  if (m_head == m_tail)
    {
      m_head = 0xffff;
      m_tail = 0xffff;
    }
  else
    {
      m_tail = item.prev;
    }
  NS_ASSERT (IsStateOk ());
}
void
PacketMetadata::AddAtEnd (PacketMetadata const&o)
{
  NS_LOG_FUNCTION (this << &o);
  NS_ASSERT (IsStateOk ());
  if (!m_enable) 
    {
      m_metadataSkipped = true;
      return;
    }
  if (m_tail == 0xffff)
    {
      // We have no items so 'AddAtEnd' is 
      // equivalent to self-assignment.
      *this = o;
      NS_ASSERT (IsStateOk ());
      return;
    }
  if (o.m_head == 0xffff)
    {
      NS_ASSERT (o.m_tail == 0xffff);
      // we have nothing to append.
      return;
    }
  NS_ASSERT (m_head != 0xffff && m_tail != 0xffff);

  // We read the current tail because we are going to append
  // after this item.
  struct PacketMetadata::SmallItem tailItem;
  PacketMetadata::ExtraItem tailExtraItem;
  uint32_t tailSize = ReadItems (m_tail, &tailItem, &tailExtraItem);

  uint16_t current;
  struct PacketMetadata::SmallItem item;
  PacketMetadata::ExtraItem extraItem;
  o.ReadItems (o.m_head, &item, &extraItem);
  if (extraItem.packetUid == tailExtraItem.packetUid &&
      item.typeUid == tailItem.typeUid &&
      item.chunkUid == tailItem.chunkUid &&
      item.size == tailItem.size &&
      extraItem.fragmentStart == tailExtraItem.fragmentEnd)
    {
      /* If the previous tail came from the same header as
       * the next item we want to append to our array, then, 
       * we merge them and attempt to reuse the previous tail's
       * location.
       */
      tailExtraItem.fragmentEnd = extraItem.fragmentEnd;
      ReplaceTail (&tailItem, &tailExtraItem, tailSize);
      if (o.m_head == o.m_tail)
        {
          // there is only one item to append to self from other.
          return;
        }
      current = item.next;
    }
  else
    {
      current = o.m_head;
    }

  /* Now that we have merged our current tail with the head of the
   * next packet, we just append all items from the next packet
   * to the current packet.
   */
  while (current != 0xffff)
    {
      o.ReadItems (current, &item, &extraItem);
      uint16_t written = AddBig (0xffff, m_tail, &item, &extraItem);
      UpdateTail (written);
      if (current == o.m_tail)
        {
          break;
        }
      current = item.next;
    }
  NS_ASSERT (IsStateOk ());
}
void
PacketMetadata::AddPaddingAtEnd (uint32_t end)
{
  NS_LOG_FUNCTION (this << end);
  if (!m_enable)
    {
      m_metadataSkipped = true;
      return;
    }
}
void 
PacketMetadata::RemoveAtStart (uint32_t start)
{
  NS_LOG_FUNCTION (this << start);
  NS_ASSERT (IsStateOk ());
  if (!m_enable) 
    {
      m_metadataSkipped = true;
      return;
    }
  NS_ASSERT (m_data != 0);
  uint32_t leftToRemove = start;
  uint16_t current = m_head;
  while (current != 0xffff && leftToRemove > 0)
    {
      struct PacketMetadata::SmallItem item;
      PacketMetadata::ExtraItem extraItem;
      ReadItems (current, &item, &extraItem);
      uint32_t itemRealSize = extraItem.fragmentEnd - extraItem.fragmentStart;
      if (itemRealSize <= leftToRemove)
        {
          // remove from list.
          if (m_head == m_tail)
            {
              m_head = 0xffff;
              m_tail = 0xffff;
            }
          else
            {
              m_head = item.next;
            }
          leftToRemove -= itemRealSize;
        }
      else
        {
          // fragment the list item.
          PacketMetadata fragment (m_packetUid, 0);
          extraItem.fragmentStart += leftToRemove;
          leftToRemove = 0;
          uint16_t written = fragment.AddBig (0xffff, fragment.m_tail,
                                              &item, &extraItem);
          fragment.UpdateTail (written);
          while (current != 0xffff && current != m_tail)
            {
              current = item.next;
              ReadItems (current, &item, &extraItem);
              written = fragment.AddBig (0xffff, fragment.m_tail,
                                         &item, &extraItem);
              fragment.UpdateTail (written);
            }
          *this = fragment;
        }
      NS_ASSERT (item.size >= extraItem.fragmentEnd - extraItem.fragmentStart &&
                 extraItem.fragmentStart <= extraItem.fragmentEnd);
      if (current == m_tail)
        {
          break;
        }
      current = item.next;
    }
  NS_ASSERT (leftToRemove == 0);
  NS_ASSERT (IsStateOk ());
}
void 
PacketMetadata::RemoveAtEnd (uint32_t end)
{
  NS_LOG_FUNCTION (this << end);
  NS_ASSERT (IsStateOk ());
  if (!m_enable) 
    {
      m_metadataSkipped = true;
      return;
    }
  NS_ASSERT (m_data != 0);

  uint32_t leftToRemove = end;
  uint16_t current = m_tail;
  while (current != 0xffff && leftToRemove > 0)
    {
      struct PacketMetadata::SmallItem item;
      PacketMetadata::ExtraItem extraItem;
      ReadItems (current, &item, &extraItem);
      uint32_t itemRealSize = extraItem.fragmentEnd - extraItem.fragmentStart;
      if (itemRealSize <= leftToRemove)
        {
          // remove from list.
          if (m_head == m_tail)
            {
              m_head = 0xffff;
              m_tail = 0xffff;
            }
          else
            {
              m_tail = item.prev;
            }
          leftToRemove -= itemRealSize;
        }
      else
        {
          // fragment the list item.
          PacketMetadata fragment (m_packetUid, 0);
          NS_ASSERT (extraItem.fragmentEnd > leftToRemove);
          extraItem.fragmentEnd -= leftToRemove;
          leftToRemove = 0;
          uint16_t written = fragment.AddBig (fragment.m_head, 0xffff,
                                              &item, &extraItem);
          fragment.UpdateHead (written);
          while (current != 0xffff && current != m_head)
            {
              current = item.prev;
              ReadItems (current, &item, &extraItem);
              written = fragment.AddBig (fragment.m_head, 0xffff,
                                         &item, &extraItem);
              fragment.UpdateHead (written);
            }
          *this = fragment;
        }
      NS_ASSERT (item.size >= extraItem.fragmentEnd - extraItem.fragmentStart &&
                 extraItem.fragmentStart <= extraItem.fragmentEnd);
      if (current == m_head)
        {
          break;
        }
      current = item.prev;
    }
  NS_ASSERT (leftToRemove == 0);
  NS_ASSERT (IsStateOk ());
}
uint32_t
PacketMetadata::GetTotalSize (void) const
{
  NS_LOG_FUNCTION (this);
  uint32_t totalSize = 0;
  uint16_t current = m_head;
  uint16_t tail = m_tail;
  while (current != 0xffff)
    {
      struct PacketMetadata::SmallItem item;
      PacketMetadata::ExtraItem extraItem;
      ReadItems (current, &item, &extraItem);
      totalSize += extraItem.fragmentEnd - extraItem.fragmentStart;
      if (current == tail)
        {
          break;
        }
      NS_ASSERT (current != item.next);
      current = item.next;
    }
  return totalSize;
}

uint64_t 
PacketMetadata::GetUid (void) const
{
  NS_LOG_FUNCTION (this);
  return m_packetUid;
}
PacketMetadata::ItemIterator 
PacketMetadata::BeginItem (Buffer buffer) const
{
  NS_LOG_FUNCTION (this << &buffer);
  return ItemIterator (this, buffer);
}
PacketMetadata::ItemIterator::ItemIterator (const PacketMetadata *metadata, Buffer buffer)
  : m_metadata (metadata),
    m_buffer (buffer),
    m_current (metadata->m_head),
    m_offset (0),
    m_hasReadTail (false)
{
  NS_LOG_FUNCTION (this << metadata << &buffer);
}
bool
PacketMetadata::ItemIterator::HasNext (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_current == 0xffff)
    {
      return false;
    }
  if (m_hasReadTail)
    {
      return false;
    }
  return true;
}
PacketMetadata::Item
PacketMetadata::ItemIterator::Next (void)
{
  NS_LOG_FUNCTION (this);
  struct PacketMetadata::Item item;
  struct PacketMetadata::SmallItem smallItem;
  struct PacketMetadata::ExtraItem extraItem;
  m_metadata->ReadItems (m_current, &smallItem, &extraItem);
  if (m_current == m_metadata->m_tail)
    {
      m_hasReadTail = true;
    }
  m_current = smallItem.next;
  uint32_t uid = (smallItem.typeUid & 0xfffffffe) >> 1;
  item.tid.SetUid (uid);
  item.currentTrimedFromStart = extraItem.fragmentStart;
  item.currentTrimedFromEnd = extraItem.fragmentEnd - smallItem.size;
  item.currentSize = extraItem.fragmentEnd - extraItem.fragmentStart;
  if (extraItem.fragmentStart != 0 || extraItem.fragmentEnd != smallItem.size)
    {
      item.isFragment = true;
    }
  else
    {
      item.isFragment = false;
    }
  TypeId tid;
  tid.SetUid (uid);
  if (uid == 0)
    {
      item.type = PacketMetadata::Item::PAYLOAD;
    }
  else if (tid.IsChildOf (Header::GetTypeId ()))
    {
      item.type = PacketMetadata::Item::HEADER;
      if (!item.isFragment)
        {
          item.current = m_buffer.Begin ();
          item.current.Next (m_offset);
        }
    }
  else if (tid.IsChildOf (Trailer::GetTypeId ()))
    {
      item.type = PacketMetadata::Item::TRAILER;
      if (!item.isFragment)
        {
          item.current = m_buffer.End ();
          item.current.Prev (m_buffer.GetSize () - (m_offset + smallItem.size));
        }
    }
  else 
    {
      NS_ASSERT (false);
    }
  m_offset += extraItem.fragmentEnd - extraItem.fragmentStart;
  return item;
}

uint32_t 
PacketMetadata::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  uint32_t totalSize = 0;

  // add 8 bytes for the packet uid
  totalSize += 8;

  // if packet-metadata not enabled, total size
  // is simply 4-bytes for itself plus 8-bytes 
  // for packet uid
  if (!m_enable)
    {
      return totalSize;
    }

  struct PacketMetadata::SmallItem item;
  struct PacketMetadata::ExtraItem extraItem;
  uint32_t current = m_head;
  while (current != 0xffff)
    {
      ReadItems (current, &item, &extraItem);
      uint32_t uid = (item.typeUid & 0xfffffffe) >> 1;
      if (uid == 0)
        {
          totalSize += 4;
        }
      else
        {
          TypeId tid;
          tid.SetUid (uid);
          totalSize += 4 + tid.GetName ().size ();
        }
      totalSize += 1 + 4 + 2 + 4 + 4 + 8;
      if (current == m_tail)
        {
          break;
        }
      NS_ASSERT (current != item.next);
      current = item.next;
    }
  return totalSize;
}

uint32_t
PacketMetadata::Serialize (uint8_t* buffer, uint32_t maxSize) const
{
  NS_LOG_FUNCTION (this << &buffer << maxSize);
  uint8_t* start = buffer;

  buffer = AddToRawU64 (m_packetUid, start, buffer, maxSize);
  if (buffer == 0) 
    {
      return 0;
    }

  struct PacketMetadata::SmallItem item;
  struct PacketMetadata::ExtraItem extraItem;
  uint32_t current = m_head;
  while (current != 0xffff)
    {
      ReadItems (current, &item, &extraItem);
      NS_LOG_LOGIC ("bytesWritten=" << static_cast<uint32_t> (buffer - start) << ", typeUid="<<
                    item.typeUid << ", size="<<item.size<<", chunkUid="<<item.chunkUid<<
                    ", fragmentStart="<<extraItem.fragmentStart<<", fragmentEnd="<<
                    extraItem.fragmentEnd<< ", packetUid="<<extraItem.packetUid);

      uint32_t uid = (item.typeUid & 0xfffffffe) >> 1;
      if (uid != 0)
        {
          TypeId tid;
          tid.SetUid (uid);
          std::string uidString = tid.GetName ();
          uint32_t uidStringSize = uidString.size ();
          buffer = AddToRawU32 (uidStringSize, start, buffer, maxSize);
          if (buffer == 0) 
            {
              return 0;
            }
          buffer = AddToRaw (reinterpret_cast<const uint8_t *> (uidString.c_str ()), 
                             uidStringSize, start, buffer, maxSize);
          if (buffer == 0) 
            {
              return 0;
            }
        }
      else
        {
          buffer = AddToRawU32 (0, start, buffer, maxSize);
          if (buffer == 0) 
            {
              return 0;
            }
        }

      uint8_t isBig = item.typeUid & 0x1;
      buffer = AddToRawU8 (isBig, start, buffer, maxSize);
      if (buffer == 0) 
        {
          return 0;
        }

      buffer = AddToRawU32 (item.size, start, buffer, maxSize);
      if (buffer == 0) 
        {
          return 0;
        }

      buffer = AddToRawU16 (item.chunkUid, start, buffer, maxSize);
      if (buffer == 0) 
        {
          return 0;
        }

      buffer = AddToRawU32 (extraItem.fragmentStart, start, buffer, maxSize);
      if (buffer == 0) 
        {
          return 0;
        }

      buffer = AddToRawU32 (extraItem.fragmentEnd, start, buffer, maxSize);
      if (buffer == 0) 
        {
          return 0;
        }

      buffer = AddToRawU64 (extraItem.packetUid, start, buffer, maxSize);
      if (buffer == 0) 
        {
          return 0;
        }

      if (current == m_tail)
        {
          break;
        }

      NS_ASSERT (current != item.next);
      current = item.next;
    }

  NS_ASSERT (static_cast<uint32_t> (buffer - start) == maxSize);
  return 1;
}

uint32_t 
PacketMetadata::Deserialize (const uint8_t* buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << &buffer << size);
  const uint8_t* start = buffer;
  uint32_t desSize = size - 4;

  buffer = ReadFromRawU64 (m_packetUid, start, buffer, size);
  desSize -= 8;

  struct PacketMetadata::SmallItem item = {0};
  struct PacketMetadata::ExtraItem extraItem = {0};
  while (desSize > 0)
    {
      uint32_t uidStringSize = 0;
      buffer = ReadFromRawU32 (uidStringSize, start, buffer, size);
      desSize -= 4;
      uint32_t uid;
      if (uidStringSize == 0)
        {
          // uid zero for payload.
          uid = 0;
        }
      else
        {
          std::string uidString;
          for (uint32_t j = 0; j < uidStringSize; j++)
            {
              uint8_t ch = 0;
              buffer = ReadFromRawU8 (ch, start, buffer, size);
              uidString.push_back (ch);
              desSize--;
            }
          TypeId tid = TypeId::LookupByName (uidString);
          uid = tid.GetUid ();
        }
      uint8_t isBig = 0;
      buffer = ReadFromRawU8 (isBig, start, buffer, size);
      desSize--;
      item.typeUid = (uid << 1) | isBig;
      buffer = ReadFromRawU32 (item.size, start, buffer, size);
      desSize -= 4;
      buffer = ReadFromRawU16 (item.chunkUid, start, buffer, size);
      desSize -= 2;
      buffer = ReadFromRawU32 (extraItem.fragmentStart, start, buffer, size);
      desSize -= 4;
      buffer = ReadFromRawU32 (extraItem.fragmentEnd, start, buffer, size);
      desSize -= 4;
      buffer = ReadFromRawU64 (extraItem.packetUid, start, buffer, size);
      desSize -= 8;
      NS_LOG_LOGIC ("size=" << size << ", typeUid="<<item.typeUid <<
                    ", size="<<item.size<<", chunkUid="<<item.chunkUid<<
                    ", fragmentStart="<<extraItem.fragmentStart<<", fragmentEnd="<<
                    extraItem.fragmentEnd<< ", packetUid="<<extraItem.packetUid);
      uint32_t tmp = AddBig (0xffff, m_tail, &item, &extraItem);
      UpdateTail (tmp);
    }
  NS_ASSERT (desSize == 0);
  return (desSize !=0) ? 0 : 1;
}

uint8_t* 
PacketMetadata::AddToRawU8 (const uint8_t& data,
                            uint8_t* start,
                            uint8_t* current,
                            uint32_t maxSize)
{
  NS_LOG_FUNCTION (static_cast<uint32_t> (data) << &start << &current << maxSize);
  // First check buffer overflow
  if (static_cast<uint32_t> ((current + sizeof (uint8_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (current, &data, sizeof (uint8_t));
  return current + sizeof (uint8_t);
}

uint8_t* 
PacketMetadata::AddToRawU16 (const uint16_t& data,
                             uint8_t* start,
                             uint8_t* current,
                             uint32_t maxSize)
{
  NS_LOG_FUNCTION (data << &start << &current << maxSize);
  // First check buffer overflow
  if (static_cast<uint32_t> ((current + sizeof (uint16_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (current, &data, sizeof (uint16_t));
  return current + sizeof (uint16_t);
}

uint8_t* 
PacketMetadata::AddToRawU32 (const uint32_t& data,
                             uint8_t* start,
                             uint8_t* current,
                             uint32_t maxSize)
{
  NS_LOG_FUNCTION (data << &start << &current << maxSize);
  // First check buffer overflow
  if (static_cast<uint32_t> ((current + sizeof (uint32_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (current, &data, sizeof (uint32_t));
  return current + sizeof (uint32_t);
}

uint8_t* 
PacketMetadata::AddToRawU64 (const uint64_t& data,
                             uint8_t* start,
                             uint8_t* current,
                             uint32_t maxSize)
{
  NS_LOG_FUNCTION (data << &start << &current << maxSize);
  // First check buffer overflow
  if (static_cast<uint32_t> ((current + sizeof (uint64_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (current, &data, sizeof (uint64_t));
  return current + sizeof (uint64_t);
}

uint8_t* 
PacketMetadata::AddToRaw (const uint8_t* data,
                          uint32_t dataSize,
                          uint8_t* start,
                          uint8_t* current,
                          uint32_t maxSize)
{ 
  NS_LOG_FUNCTION (&data << dataSize << &start << &current << maxSize);
  // First check buffer overflow
  if (static_cast<uint32_t> ((current + dataSize - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (current, data, dataSize);
  return current + dataSize;
}

uint8_t* 
PacketMetadata::ReadFromRawU8 (uint8_t& data,
                               const uint8_t* start,
                               const uint8_t* current,
                               uint32_t maxSize)
{ 
  NS_LOG_FUNCTION (static_cast<uint32_t> (data) << &start << &current << maxSize);
  // First check buffer underflow
  if (static_cast<uint32_t> ((current + sizeof (uint8_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (&data, current, sizeof (uint8_t));
  return const_cast<uint8_t *> (current) + sizeof (uint8_t);
}

uint8_t* 
PacketMetadata::ReadFromRawU16 (uint16_t& data,
                                const uint8_t* start,
                                const uint8_t* current,
                                uint32_t maxSize)
{ 
  NS_LOG_FUNCTION (data << &start << &current << maxSize);
  // First check buffer underflow
  if (static_cast<uint32_t> ((current + sizeof (uint16_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (&data, current, sizeof (uint16_t));
  return const_cast<uint8_t *> (current) + sizeof (uint16_t);
}

uint8_t* 
PacketMetadata::ReadFromRawU32 (uint32_t& data,
                                const uint8_t* start,
                                const uint8_t* current,
                                uint32_t maxSize)
{ 
  NS_LOG_FUNCTION (data << &start << &current << maxSize);
  // First check buffer underflow
  if (static_cast<uint32_t> ((current + sizeof (uint32_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (&data, current, sizeof (uint32_t));
  return const_cast<uint8_t *> (current) + sizeof (uint32_t);
}

uint8_t* 
PacketMetadata::ReadFromRawU64 (uint64_t& data,
                                const uint8_t* start,
                                const uint8_t* current,
                                uint32_t maxSize)
{ 
  NS_LOG_FUNCTION (data << &start << &current << maxSize);
  // First check buffer underflow
  if ((uint32_t)((current + sizeof (uint64_t) - start)) > maxSize) 
    {
      return 0;
    }
  memcpy (&data, current, sizeof (uint64_t));
  return const_cast<uint8_t *> (current) + sizeof (uint64_t);
}


} // namespace ns3

