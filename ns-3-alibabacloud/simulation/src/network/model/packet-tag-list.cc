/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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

/**
\file   packet-tag-list.cc
\brief  Implements a linked list of Packet tags, including copy-on-write semantics.
*/

#include "packet-tag-list.h"
#include "tag-buffer.h"
#include "tag.h"
#include "ns3/fatal-error.h"
#include "ns3/log.h"
#include <cstring>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketTagList");

PacketTagList::TagData *
PacketTagList::CreateTagData (size_t dataSize)
{
  NS_ASSERT_MSG (dataSize
                 < std::numeric_limits<decltype(TagData::size)>::max (),
                 "Requested TagData size " << dataSize
                 << " exceeds maximum "
                 << std::numeric_limits<decltype(TagData::size)>::max () );

  void * p = std::malloc (sizeof (TagData) + dataSize - 1);
  // The matching frees are in RemoveAll and RemoveWriter

  TagData * tag = new (p) TagData;
  tag->size = dataSize;
  return tag;
}

bool
PacketTagList::COWTraverse (Tag & tag, PacketTagList::COWWriter Writer)
{
  TypeId tid = tag.GetInstanceTypeId ();
  NS_LOG_FUNCTION (this << tid);
  NS_LOG_INFO     ("looking for " << tid);

  // trivial case when list is empty
  if (m_next == 0)
    {
      return false;
    }

  bool found = false;

  struct TagData ** prevNext = &m_next; // previous node's next pointer
  struct TagData  * cur      =  m_next; // cursor to current node
  struct TagData  * it = 0;             // utility

  // Search from the head of the list until we find tid or a merge
  while (cur != 0)
    {
      if (cur->count > 1)
        {
          // found merge
          NS_LOG_INFO ("found initial merge before tid");
          break;
        }
      else if (cur->tid == tid)
        {
          NS_LOG_INFO ("found tid before initial merge, calling writer");
          found = (this->*Writer)(tag, true, cur, prevNext);
          break;
        }
      else
        {
          // no merge or tid found yet, move on
          prevNext = &cur->next;
          cur      =  cur->next;
        }
    }  // while !found && !cow

  // did we find it or run out of tags?
  if (cur == 0 || found)
    {
      NS_LOG_INFO ("returning after header with found: " << found);
      return found;
    }

  // From here on out, we have to copy the list
  // until we find tid, then link past it

  // Before we do all that work, let's make sure tid really exists
  for (it = cur; it != 0; it = it->next)
    {
      if (it->tid == tid)
        {
          break;
        }
    }
  if (it == 0)
    {
      // got to end of list without finding tid
      NS_LOG_INFO ("tid not found after first merge");
      return found;
    }

  // At this point cur is a merge, but untested for tid
  NS_ASSERT (cur != 0);
  NS_ASSERT (cur->count > 1);

  /*
     Walk the remainder of the list, copying, until we find tid
     As we put a copy of the cur node onto our list,
     we move the merge point down the list.

     Starting position                  End position
       T1 is a merge                     T1.count decremented
                                         T2 is a merge
                                         T1' is a copy of T1

          other                             other
               \                                 \
      Prev  ->  T1  ->  T2  -> ...                T1  ->  T2  -> ...
           /   /                                         /|
      pNext cur                         Prev  ->  T1' --/ |
                                                     /    |
                                                pNext   cur

     When we reach tid, we link past it, decrement count, and we're done.
  */

  // Should normally check for null cur pointer,
  // but since we know tid exists, we'll skip this test
  while ( /* cur && */ cur->tid != tid)
    {
      NS_ASSERT (cur != 0);
      NS_ASSERT (cur->count > 1);
      cur->count--;                       // unmerge cur
      struct TagData * copy = CreateTagData (cur->size);
      copy->tid = cur->tid;
      copy->count = 1;
      copy->size = cur->size;
      memcpy (copy->data, cur->data, copy->size);
      copy->next = cur->next;             // merge into tail
      copy->next->count++;                // mark new merge
      *prevNext = copy;                   // point prior list at copy
      prevNext = &copy->next;             // advance
      cur      =  copy->next;
    }
  // Sanity check:
  NS_ASSERT (cur != 0);                 // cur should be non-zero
  NS_ASSERT (cur->tid == tid);          // cur->tid should be tid
  NS_ASSERT (cur->count > 1);           // cur should be a merge

  // link around tid, removing it from our list
  found = (this->*Writer)(tag, false, cur, prevNext);
  return found;

}

bool
PacketTagList::Remove (Tag & tag)
{
  return COWTraverse (tag, &PacketTagList::RemoveWriter);
}

// COWWriter implementing Remove
bool
PacketTagList::RemoveWriter (Tag & tag, bool preMerge,
                             struct PacketTagList::TagData  * cur,
                             struct PacketTagList::TagData ** prevNext)
{
  NS_LOG_FUNCTION_NOARGS ();

  // found tid
  bool found = true;
  tag.Deserialize (TagBuffer (cur->data, cur->data + cur->size));
  *prevNext = cur->next;            // link around cur

  if (preMerge)
    {
      // found tid before first merge, so delete cur
      cur->~TagData ();
      std::free (cur);
    }
  else
    {
      // cur is always a merge at this point
      // unmerge cur, since we linked around it already
      cur->count--;
      if (cur->next != 0)
        {
          // there's a next, so make it a merge
          cur->next->count++;
        }
    }
  return found;
}

bool
PacketTagList::Replace (Tag & tag)
{
  bool found = COWTraverse (tag, &PacketTagList::ReplaceWriter);
  if (!found)
    {
      Add (tag);
    }
  return found;
}

// COWWriter implementing Replace
bool
PacketTagList::ReplaceWriter (Tag & tag, bool preMerge,
                              struct PacketTagList::TagData  * cur,
                              struct PacketTagList::TagData ** prevNext)
{
  NS_LOG_FUNCTION_NOARGS ();

  // found tid
  bool found = true;
  if (preMerge)
    {
      // found tid before first merge, so just rewrite
      tag.Serialize (TagBuffer (cur->data, cur->data + cur->size));
    }
  else
    {
      // cur is always a merge at this point
      // need to copy, replace, and link past cur
      cur->count--;                     // unmerge cur
      struct TagData * copy = CreateTagData (tag.GetSerializedSize ());
      copy->tid = tag.GetInstanceTypeId ();
      copy->count = 1;
      tag.Serialize (TagBuffer (copy->data, copy->data + copy->size));
      copy->next = cur->next;           // merge into tail
      if (copy->next != 0)
        {
          copy->next->count++;          // mark new merge
        }
      *prevNext = copy;                 // point prior list at copy
    }
  return found;
}

void
PacketTagList::Add (const Tag &tag) const
{
  NS_LOG_FUNCTION (this << tag.GetInstanceTypeId ());
  // ensure this id was not yet added
  for (struct TagData *cur = m_next; cur != 0; cur = cur->next)
    {
      NS_ASSERT_MSG (cur->tid != tag.GetInstanceTypeId (),
                     "Error: cannot add the same kind of tag twice.");
    }
  struct TagData * head = CreateTagData (tag.GetSerializedSize ());
  head->count = 1;
  head->next = 0;
  head->tid = tag.GetInstanceTypeId ();
  head->next = m_next;
  tag.Serialize (TagBuffer (head->data, head->data + head->size));

  const_cast<PacketTagList *> (this)->m_next = head;
}

bool
PacketTagList::Peek (Tag &tag) const
{
  NS_LOG_FUNCTION (this << tag.GetInstanceTypeId ());
  TypeId tid = tag.GetInstanceTypeId ();
  for (struct TagData *cur = m_next; cur != 0; cur = cur->next)
    {
      if (cur->tid == tid)
        {
          /* found tag */
          tag.Deserialize (TagBuffer (cur->data, cur->data + cur->size));
          return true;
        }
    }
  /* no tag found */
  return false;
}

const struct PacketTagList::TagData *
PacketTagList::Head (void) const
{
  return m_next;
}

uint32_t
PacketTagList::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION_NOARGS ();

  uint32_t size = 0;

  size = 4; // numberOfTags

  for (struct TagData *cur = m_next; cur != 0; cur = cur->next)
    {
      size += 4; // TagData -> size

      // TypeId hash; ensure size is multiple of 4 bytes
      uint32_t hashSize = (sizeof (TypeId::hash_t)+3) & (~3);
      size += hashSize;

      // TagData -> data; ensure size is multiple of 4 bytes
      uint32_t tagWordSize = (cur->size+3) & (~3);
      size += tagWordSize;
    }

  return size;
}

uint32_t
PacketTagList::Serialize (uint32_t* buffer, uint32_t maxSize) const
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

  for (struct TagData *cur = m_next; cur != 0; cur = cur->next)
    {
      if (size + 4 <= maxSize)
        {
          *p++ = cur->size;
          size += 4;
        }
      else
        {
          return 0;
        }

      NS_LOG_INFO("Serializing tag id " << cur->tid);

      // ensure size is multiple of 4 bytes for 4 byte boundaries
      uint32_t hashSize = (sizeof (TypeId::hash_t)+3) & (~3);
      if (size + hashSize <= maxSize)
        {
          TypeId::hash_t tid = cur->tid.GetHash ();
          memcpy (p, &tid, sizeof (TypeId::hash_t));
          p += hashSize / 4;
          size += hashSize;
        }
      else
        {
          return 0;
        }

      // ensure size is multiple of 4 bytes for 4 byte boundaries
      uint32_t tagWordSize = (cur->size+3) & (~3);
      if (size + tagWordSize <= maxSize)
        {
          memcpy (p, cur->data, cur->size);
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
PacketTagList::Deserialize (const uint32_t* buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << buffer << size);
  const uint32_t* p = buffer;
  uint32_t sizeCheck = size - 4;

  NS_ASSERT (sizeCheck >= 4);
  uint32_t numberOfTags = *p++;
  sizeCheck -= 4;

  NS_LOG_INFO("Deserializing number of tags " << numberOfTags);

  struct TagData * prevTag = 0;
  for (uint32_t i = 0; i < numberOfTags; ++i)
    {
      NS_ASSERT (sizeCheck >= 4);
      uint32_t tagSize = *p++;
      sizeCheck -= 4;

      uint32_t hashSize = (sizeof (TypeId::hash_t)+3) & (~3);
      NS_ASSERT (sizeCheck >= hashSize);
      TypeId::hash_t hash;
      memcpy (&hash, p, sizeof (TypeId::hash_t));
      p += hashSize / 4;
      sizeCheck -= hashSize;

      TypeId tid = TypeId::LookupByHash(hash);

      NS_LOG_INFO ("Deserializing tag of type " << tid);

      struct TagData * newTag = CreateTagData (tagSize);
      newTag->count = 1;
      newTag->next = 0;
      newTag->tid = tid;

      NS_ASSERT (sizeCheck >= tagSize);
      memcpy (newTag->data, p, tagSize);

      // ensure 4 byte boundary
      uint32_t tagWordSize = (tagSize+3) & (~3);
      p += tagWordSize / 4;
      sizeCheck -= tagWordSize;

      // Set link list pointers.
      if (i==0)
        {
          m_next = newTag;
        }
      else
        {
          prevTag->next = newTag;
        }

      prevTag = newTag;
    }

  NS_ASSERT (sizeCheck == 0);

  // return zero if buffer did not
  // contain a complete message
  return (sizeCheck != 0) ? 0 : 1;
}


} /* namespace ns3 */

