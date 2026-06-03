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
#ifndef PACKET_TAG_LIST_H
#define PACKET_TAG_LIST_H

/**
\file   packet-tag-list.h
\brief  Defines a linked list of Packet tags, including copy-on-write semantics.
*/

#include <stdint.h>
#include <ostream>
#include "ns3/type-id.h"
#include "ns3/atomic-counter.h"

namespace ns3 {

class Tag;

/**
 * \ingroup packet
 *
 * \brief List of the packet tags stored in a packet.
 *
 * This class is mostly private to the Packet implementation and users
 * should never have to access it directly.
 *
 * \internal
 *
 * The implementation of this class is a bit tricky.  Refer to this
 * diagram in the discussion that follows.
 *
 * \dot
 *    digraph {
 *        rankdir = "LR";
 *        clusterrank = local;
 *        node [ shape = record, fontname="FreeSans", fontsize="10" ];
 *        oth  [ label="<l> Other branch | <n> next | <c> ..." ];
 *        PTL1 [ label="<l> PacketTagList A | <n> m_next" , shape=Mrecord];
 *        PTL2 [ label="<l> PacketTagList B | <n> m_next" , shape=Mrecord];
 *        oth:n  -> T7:l ;
 *        PTL2:n -> T6:l ;
 *        PTL1:n -> T5:l ;
 *        T1   [ label="<l> T1  | <n> next | <c> count = 1" ];
 *        T2   [ label="<l> T2  | <n> next | <c> count = 1" ];
 *        T3   [ label="<l> T3  | <n> next | <c> count = 2" ];
 *        T4   [ label="<l> T4  | <n> next | <c> count = 1" ];
 *        T5   [ label="<l> T5  | <n> next | <c> count = 2" ];
 *        T6   [ label="<l> T6  | <n> next | <c> count = 1" ];
 *        T7   [ label="<l> T7  | <n> next | <c> count = 1" ];
 *        NULL [ label="0", shape = ellipse ];
 *        subgraph cluster_list {
 *          penwidth = 0;
 *          T6:n -> T5:l ;
 *          T5:n -> T4:l ;
 *          T4:n -> T3:l ;
 *          T7:n -> T3:l ;
 *          T3:n -> T2:l ;
 *          T2:n -> T1:l ;
 *          T1:n -> NULL ;
 *        };
 *      }
 * \enddot
 *
 *   - Tags are stored in serialized form in a tree of TagData
 *     structures. (<tt>T1-T7</tt> in the diagram)
 *
 *   - Each TagData points (\c next pointers in the diagram)
 *     toward the root of the tree, which is a null pointer.
 *
 *   - \c count is the number of incoming pointers to this
 *     TagData.  Branch points, where branches merge or join, have
 *     <tt>count \> 1</tt> (\c T3, \c T5); successive links along
 *     a branch have <tt>count = 1</tt> (\c T1, \c T2, \c T4, \c T6, \c T7).
 *
 *   - Each PacketTagList points to a specific TagData,
 *     which is the most recent Tag added to the packet. (<tt>T5-T7</tt>)
 *
 *   - Conceptually, therefore, each Packet has a PacketTagList which
 *     points to a singly-linked list of TagData.
 *
 * \par <b> Copy-on-write </b> is implemented as follows:
 *
 *   - #Add prepends the new tag to the list (growing that branch of the tree,
 *     as \c T6). This is a constant time operation, and does not affect
 *     any other #PacketTagList's, hence this is a \c const function.
 *
 *   - Copy constructor (PacketTagList(const PacketTagList & o))
 *     and assignment (#operator=(const PacketTagList & o))
 *     simply join the tree at the same place as the original
 *     PacketTagList \c o, incrementing the \c count.
 *     For assignment, the old branch is deleted, up to
 *     the first branch point, which has its \c count decremented.
 *     (PacketTagList \c B started as a copy of PacketTagList \c A,
 *     before \c T6 was added to \c B).
 *
 *   - #Remove and #Replace are a little tricky, depending on where the
 *     target tag is found relative to the first branch point:
 *     - \e Target before <em> the first branch point: </em> \n
 *       The target is just dealt with in place (linked around and deleted,
 *       in the case of #Remove; rewritten in the case of #Replace).
 *     - \e Target at or after <em> the first branch point: </em> \n
 *       The portion of the list between the first branch and the target is
 *       shared. This portion is copied before the #Remove or #Replace is
 *       performed.
 */
class PacketTagList 
{
public:
  /**
   * Tree node for sharing serialized tags.
   *
   * See PacketTagList for a discussion of the data structure.
   *
   * \internal
   * Unfortunately this has to be public, because
   * PacketTagIterator::Item::GetTag() needs the data and size values.
   * The Item nested class can't be forward declared, so friending isn't
   * possible.
   *
   * We use placement new so we can allocate enough room for the Tag
   * type which will be serialized into data.  See Object::Aggregates
   * for a similar construction.
   */
  struct TagData
  {
    struct TagData * next;      /**< Pointer to next in list */
#ifdef NS3_MTP
    AtomicCounter count;
#else
    uint32_t count;             /**< Number of incoming links */
#endif
    TypeId tid;                 /**< Type of the tag serialized into #data */
    uint32_t size;              /**< Size of the \c data buffer */
    uint8_t data[1];            /**< Serialization buffer */
  };  /* struct TagData */

  /**
   * Create a new PacketTagList.
   */
  inline PacketTagList ();
  /**
   * Copy constructor
   *
   * \param [in] o The PacketTagList to copy.
   *
   * This makes a light-weight copy by #RemoveAll, then
   * pointing to the same \ref TagData as \pname{o}.
   */
  inline PacketTagList (PacketTagList const &o);
  /**
   * Assignment
   *
   * \param [in] o The PacketTagList to copy.
   * \returns the copied object
   *
   * This makes a light-weight copy by #RemoveAll, then
   * pointing to the same \ref TagData as \pname{o}.
   */
  inline PacketTagList &operator = (PacketTagList const &o);
  /**
   * Destructor
   *
   * #RemoveAll's the tags up to the first merge.
   */
  inline ~PacketTagList ();

  /**
   * Add a tag to the head of this branch.
   *
   * \param [in] tag The tag to add
   */
  void Add (Tag const&tag) const;
  /**
   * Remove (the first instance of) tag from the list.
   *
   * \param [in,out] tag The tag type to remove.  If found,
   *          \pname{tag} is set to the value of the tag found.
   * \returns True if \pname{tag} is found, false otherwise.
   */
  bool Remove (Tag &tag);
  /**
   * Replace the value of a tag.
   *
   * \param [in] tag The tag type to replace.  To get the old
   *        value of the tag, use #Peek first.
   * \returns True if \pname{tag} is found, false otherwise.
   *        If \pname{tag} wasn't found, Add is performed instead (so
   *        the list is guaranteed to have the new tag value either way).
   */
  bool Replace (Tag &tag);
  /**
   * Find a tag and return its value.
   *
   * \param [in,out] tag The tag type to find.  If found,
   *          \pname{tag} is set to the value of the tag found.
   * \returns True if \pname{tag} is found, false otherwise.
   */
  bool Peek (Tag &tag) const;
  /**
   * Remove all tags from this list (up to the first merge).
   */
  inline void RemoveAll (void);
  /**
   * \returns pointer to head of tag list
   */
  const struct PacketTagList::TagData *Head (void) const;
  /**
   * Returns number of bytes required for packet serialization.
   *
   * \returns number of bytes required for packet serialization
   */
  uint32_t GetSerializedSize (void) const;
  /**
   * Serialize the tag list into a byte buffer.
   *
   * \param [in,out] buffer The byte buffer to which the tag list will be serialized
   * \param [in] maxSize Max The max size of the buffer for bounds checking
   *
   * \returns zero if complete tag list is not serialized
   */
  uint32_t Serialize (uint32_t* buffer, uint32_t maxSize) const;
  /**
   * Deserialize tag list from the provided buffer.
   *
   * \param [in] buffer The buffer to read from.
   * \param [in] size The number of bytes to deserialize.
   *
   * \returns zero if complete tag list is not deserialized
   */
  uint32_t Deserialize (const uint32_t* buffer, uint32_t size);

private:
  /**
   * Allocate and construct a TagData struct, sizing the data area
   * large enough to serialize dataSize bytes from a Tag.
   *
   * \param [in] dataSize The serialized size of the Tag.
   * \returns The newly constructed TagData object.
   */
  static
  TagData * CreateTagData (size_t dataSize);
  
  /**
   * Typedef of method function pointer for copy-on-write operations
   *
   * \param [in] tag The tag type to operate on.
   * \param [in] preMerge True if \pname{tag} was found before the first merge,
   *             false otherwise.
   * \param [in] cur Pointer to the tag.
   * \param [in] prevNext Pointer to the struct TagData.next pointer
   *          pointing to \pname{cur}.
   * \returns True if operation successful, false otherwise
   */
  typedef bool (PacketTagList::*COWWriter)
    (Tag       & tag, bool         preMerge,
    struct TagData * cur, struct TagData ** prevNext);
  /**
   * Traverse the list implementing copy-on-write, using \pname{Writer}.
   *
   * \param [in] tag The tag type to operate on.
   * \param [in] Writer The copy-on-write function to use.
   * \returns True if \pname{tag} found, false otherwise.
   */
  bool COWTraverse   (Tag & tag, PacketTagList::COWWriter Writer);
  /**
   * Copy-on-write implementing Remove.
   *
   * \param [in] tag The target tag type to remove.
   * \param [in] preMerge True if \pname{tag} was found before the first merge,
   *             false otherwise.
   * \param [in] cur Pointer to the tag.
   * \param [in] prevNext Pointer to the struct TagData.next pointer
   *          pointing to \pname{cur}.
   * \returns True, since tag will definitely be removed.
   */
  bool RemoveWriter  (Tag & tag, bool preMerge,
                      struct TagData * cur, struct TagData ** prevNext);
  /**
   * Copy-on-write implementing Replace
   *
   * \param [in] tag The target tag type to replace
   * \param [in] preMerge True if \pname{tag} was found before the first merge,
   *          false otherwise.
   * \param [in] cur Pointer to the tag
   * \param [in] prevNext Pointer to the struct TagData.next pointer
   *          pointing to \pname{cur}.
   * \returns True, since tag value will definitely be replaced.
   */
  bool ReplaceWriter (Tag & tag, bool preMerge,
                      struct TagData * cur, struct TagData ** prevNext);

  /**
   * Pointer to first \ref TagData on the list
   */
  struct TagData *m_next;
};

} // namespace ns3

/****************************************************
 *  Implementation of inline methods for performance
 ****************************************************/

namespace ns3 {

PacketTagList::PacketTagList ()
  : m_next ()
{
}

PacketTagList::PacketTagList (PacketTagList const &o)
  : m_next (o.m_next)
{
  if (m_next != 0)
    {
      m_next->count++;
    }
}

PacketTagList &
PacketTagList::operator = (PacketTagList const &o)
{
  // self assignment
  if (m_next == o.m_next) 
    {
      return *this;
    }
  RemoveAll ();
  m_next = o.m_next;
  if (m_next != 0) 
    {
      m_next->count++;
    }
  return *this;
}

PacketTagList::~PacketTagList ()
{
  RemoveAll ();
}

void
PacketTagList::RemoveAll (void)
{
  struct TagData *prev = 0;
  for (struct TagData *cur = m_next; cur != 0; cur = cur->next)
    {
      if (cur->count-- > 1) 
        {
          break;
        }
#ifdef NS3_MTP
      std::atomic_thread_fence (std::memory_order_acquire);
#endif
      if (prev != 0) 
        {
          prev->~TagData ();
          std::free (prev);
        }
      prev = cur;
    }
  if (prev != 0) 
    {
      prev->~TagData ();
      std::free (prev);
    }
  m_next = 0;
}

} // namespace ns3

#endif /* PACKET_TAG_LIST_H */
