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
#ifndef PACKET_METADATA_H
#define PACKET_METADATA_H

#include <stdint.h>
#include <vector>
#include <limits>
#include "ns3/callback.h"
#include "ns3/assert.h"
#include "ns3/type-id.h"
#include "ns3/atomic-counter.h"
#include "buffer.h"

namespace ns3 {

class Chunk;
class Buffer;
class Header;
class Trailer;

/**
 * \ingroup packet
 * \brief Handle packet metadata about packet headers and trailers
 *
 * This class is used by the Packet class to record every operation
 * performed on the packet's buffer. This class also provides
 * an implementation of the Packet::Print methods which uses
 * the metadata to analyse the content of the packet's buffer.
 *
 * To achieve this, this class maintains a linked list of so-called
 * "items", each of which represents a header or a trailer, or 
 * payload, or a fragment of any of these. Each item contains a "next"
 * and a "prev" field which point to the next and previous entries
 * in the linked list. The PacketMetadata class maintains a pair
 * of pointers to the head and the tail of the linked list.
 *
 * Each entry in the list also maintains:
 *   - its native size (the size it had when it was first added
 *     to the packet)
 *   - its type: identifies what kind of header, what kind of trailer,
 *     if it is payload or not
 *   - the uid of the packet to which it was first added
 *   - the start and end of the area represented by a fragment
 *     if it is one.
 *
 * This linked list is flattened in a byte buffer stored in
 * struct PacketMetadata::Data. Each entry of the linked list is
 * identified by an offset which identifies the first byte of the
 * entry from the start of the data buffer. The size of this data
 * buffer is 2^16-1 bytes maximum which somewhat limits the number
 * of entries which can be stored in this linked list but it is
 * quite unlikely to hit this limit in practice.
 *
 * Each item of the linked list is a variable-sized byte buffer
 * made of a number of fields. Some of these fields are stored
 * as fixed-size 32 bit integers, others as fixed-size 16 bit 
 * integers, and some others as variable-size 32-bit integers.
 * The variable-size 32 bit integers are stored using the uleb128
 * encoding.
 */
class PacketMetadata 
{
public:

  /**
   * \brief structure describing a packet metadata item
   */
  struct Item 
  {
    /// Type of data in the packet
    enum ItemType {
      PAYLOAD,  //!< Payload
      HEADER,   //!< Header
      TRAILER   //!< Trailer
    } type; //!< metadata type
    /**
     * true: this is a fragmented header, trailer, or, payload.
     * false: this is a whole header, trailer, or, payload.
     */
    bool isFragment;
    /**
     * TypeId of Header or Trailer. Valid only if type is
     * header or trailer.
     */
    TypeId tid;
    /**
     * size of item. If fragment, size of fragment. Otherwise,
     * size of original item. 
     */
    uint32_t currentSize;
    /**
     * how many bytes were trimmed from the start of a fragment.
     * if isFragment is true, this field is zero.
     */
    uint32_t currentTrimedFromStart;
    /**
     * how many bytes were trimmed from the end of a fragment.
     * if isFragment is true, this field is zero.
     */
    uint32_t currentTrimedFromEnd;
    /**
     * an iterator which can be fed to Deserialize. Valid only
     * if isFragment and isPayload are false.
     */
    Buffer::Iterator current;
  };

  /**
   * \brief Iterator class for metadata items.
   */
  class ItemIterator 
  {
public:
    /**
     * \brief Constructor
     * \param metadata a pointer to the metadata
     * \param buffer the buffer the metadata refers to
     */
    ItemIterator (const PacketMetadata *metadata, Buffer buffer);
    /**
     * \brief Checks if there is another metadata item
     * \returns true if there is another item
     */
    bool HasNext (void) const;
    /**
     * \brief Retrieve the next metadata item
     * \returns the next metadata item
     */
    Item Next (void);
private:
    const PacketMetadata *m_metadata; //!< pointer to the metadata
    Buffer m_buffer; //!< buffer the metadata refers to
    uint16_t m_current; //!< current position
    uint32_t m_offset; //!< offset
    bool m_hasReadTail; //!< true if the metadata tail has been read
  };

  /**
   * \brief Enable the packet metadata
   */
  static void Enable (void);
  /**
   * \brief Enable the packet metadata checking
   */
  static void EnableChecking (void);

  /**
   * \brief Constructor
   * \param uid packet uid
   * \param size size of the header
   */
  inline PacketMetadata (uint64_t uid, uint32_t size);
  /**
   * \brief Copy constructor
   * \param o the object to copy
   */
  inline PacketMetadata (PacketMetadata const &o);
  /**
   * \brief Basic assignment
   * \param o the object to copy
   * \return a copied object
   */
  inline PacketMetadata &operator = (PacketMetadata const& o);
  inline ~PacketMetadata ();

  /**
   * \brief Add an header
   * \param header header to add
   * \param size header serialized size
   */
  void AddHeader (Header const &header, uint32_t size);
  /**
   * \brief Remove an header
   * \param header header to remove
   * \param size header serialized size
   */
  void RemoveHeader (Header const &header, uint32_t size);

  /**
   * Add a trailer
   * \param trailer trailer to add
   * \param size trailer serialized size
   */
  void AddTrailer (Trailer const &trailer, uint32_t size);
  /**
   * Remove a trailer
   * \param trailer trailer to remove
   * \param size trailer serialized size
   */
  void RemoveTrailer (Trailer const &trailer, uint32_t size);

  /**
   * \brief Creates a fragment.
   *
   * \param start the amount of stuff to remove from the start
   * \param end the amount of stuff to remove from the end
   * \return the fragment's metadata
   *
   * Calling this method is equivalent to calling RemoveAtStart (start)
   * and then, RemoveAtEnd (end).
   */
  PacketMetadata CreateFragment (uint32_t start, uint32_t end) const;

  /**
   * \brief Add a metadata at the metadata start
   * \param o the metadata to add
   */
  void AddAtEnd (PacketMetadata const&o);
  /**
   * \brief Add some padding at the end
   * \param end size of padding
   */
  void AddPaddingAtEnd (uint32_t end);
  /**
   * \brief Remove a chunk of metadata at the metadata start
   * \param start the size of metadata to remove
   */
  void RemoveAtStart (uint32_t start);
  /**
   * \brief Remove a chunk of metadata at the metadata end
   * \param end the size of metadata to remove
   */
  void RemoveAtEnd (uint32_t end);

  /**
   * \brief Get the packet Uid
   * \return the packet Uid
   */
  uint64_t GetUid (void) const;

  /**
   * \brief Get the metadata serialized size
   * \return the seralized size
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \brief Initialize the item iterator to the buffer begin
   * \param buffer buffer to initialize.
   * \return the buffer iterator.
   */
  ItemIterator BeginItem (Buffer buffer) const;

  /**
   *  \brief Serialization to raw uint8_t*
   *  \param buffer the buffer to serialize to
   *  \param maxSize the maximum serialization size
   *  \return 1 on success, 0 on failure
   */
  uint32_t Serialize   (uint8_t* buffer, uint32_t maxSize) const;
  /**
   *  \brief Deserialization from raw uint8_t*
   *  \param buffer the buffer to deserialize from
   *  \param size the size
   *  \return 1 on success, 0 on failure
   */
  uint32_t Deserialize (const uint8_t* buffer, uint32_t size);

private:
  /**
   * \brief Helper for the raw serialization.
   *
   * \param data the buffer to write to
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* AddToRawU8 (const uint8_t& data,
                              uint8_t* start,
                              uint8_t* current,
                              uint32_t maxSize);

  /**
   * \brief Helper for the raw serialization.
   *
   * \param data the buffer to write to
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* AddToRawU16 (const uint16_t& data,
                               uint8_t* start,
                               uint8_t* current,
                               uint32_t maxSize);

  /**
   * \brief Helper for the raw serialization.
   *
   * \param data the buffer to write to
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* AddToRawU32 (const uint32_t& data,
                               uint8_t* start,
                               uint8_t* current,
                               uint32_t maxSize);

  /**
   * \brief Helper for the raw serialization.
   *
   * \param data the buffer to write to
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* AddToRawU64 (const uint64_t& data,
                               uint8_t* start,
                               uint8_t* current,
                               uint32_t maxSize);

  /**
   * \brief Helper for the raw serialization.
   *
   * \param data the buffer to write to
   * \param dataSize the data size to write to
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* AddToRaw (const uint8_t* data,
                            uint32_t dataSize,
                            uint8_t* start,
                            uint8_t* current,
                            uint32_t maxSize);

  /**
   * \brief Helper for the raw deserialization.
   *
   * \param data the buffer to read from
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* ReadFromRawU8 (uint8_t& data,
                                 const uint8_t* start,
                                 const uint8_t* current,
                                 uint32_t maxSize);

  /**
   * \brief Helper for the raw deserialization.
   *
   * \param data the buffer to read from
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* ReadFromRawU16 (uint16_t& data,
                                  const uint8_t* start,
                                  const uint8_t* current,
                                  uint32_t maxSize);

  /**
   * \brief Helper for the raw deserialization.
   *
   * \param data the buffer to read from
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* ReadFromRawU32 (uint32_t& data,
                                  const uint8_t* start,
                                  const uint8_t* current,
                                  uint32_t maxSize);

  /**
   * \brief Helper for the raw deserialization.
   *
   * \param data the buffer to read from
   * \param start start index
   * \param current current index
   * \param maxSize maximum size
   * \return updated current index
   */
  static uint8_t* ReadFromRawU64 (uint64_t& data,
                                  const uint8_t* start,
                                  const uint8_t* current,
                                  uint32_t maxSize);

  /**
   * the size of PacketMetadata::Data::m_data such that the total size
   * of PacketMetadata::Data is 16 bytes
   */ 
#define PACKET_METADATA_DATA_M_DATA_SIZE 8
  
  /**
   * Data structure
   */
  struct Data {
    /** number of references to this struct Data instance. */
#ifdef NS3_MTP
    AtomicCounter m_count;
#else
    uint32_t m_count;
#endif
    /** size (in bytes) of m_data buffer below */
    uint16_t m_size;
    /** max of the m_used field over all objects which
     * reference this struct Data instance */
    uint16_t m_dirtyEnd;
    /** variable-sized buffer of bytes */
    uint8_t m_data[PACKET_METADATA_DATA_M_DATA_SIZE]; 
  };
  /* Note that since the next and prev fields are 16 bit integers
     and since the value 0xffff is reserved to identify the 
     fact that the end or the start of the list is reached,
     only a limited number of elements can be stored in 
     a m_data byte buffer.
   */
  /**
   * \brief SmallItem structure
   */
  struct SmallItem {
    /** offset (in bytes) from start of m_data buffer
       to next element in linked list. value is 0xffff 
       if next element does not exist.
       stored as a fixed-size 16 bit integer.
    */
    uint16_t next;
    /** offset (in bytes) from start of m_data buffer
       to previous element in linked list. value is 0xffff 
       if previous element does not exist.
       stored as a fixed-size 16 bit integer.
     */
    uint16_t prev;
    /** the high 31 bits of this field identify the
       type of the header or trailer represented by 
       this item: the value zero represents payload.
       If the low bit of this uid is one, an ExtraItem
       structure follows this SmallItem structure.
       stored as a variable-size 32 bit integer.
     */
    uint32_t typeUid;
    /** the size (in bytes) of the header or trailer represented
       by this element.
       stored as a variable-size 32 bit integer.
     */
    uint32_t size;
    /** this field tries to uniquely identify each header or
       trailer _instance_ while the typeUid field uniquely
       identifies each header or trailer _type_. This field
       is used to test whether two items are equal in the sense 
       that they represent the same header or trailer instance.
       That equality test is based on the typeUid and chunkUid
       fields so, the likelihood that two header instances 
       share the same chunkUid _and_ typeUid is very small 
       unless they are really representations of the same header
       instance.
       stored as a fixed-size 16 bit integer.
     */
    uint16_t chunkUid;
  };

  /**
   * \brief ExtraItem structure
   */
  struct ExtraItem {
    /** offset (in bytes) from start of original header to
       the start of the fragment still present.
       stored as a variable-size 32 bit integer.
     */
    uint32_t fragmentStart;
    /** offset (in bytes) from start of original header to
       the end of the fragment still present.
       stored as a variable-size 32 bit integer.
     */
    uint32_t fragmentEnd;
    /** the packetUid of the packet in which this header or trailer
       was first added. It could be different from the m_packetUid
       field if the user has aggregated multiple packets into one.
       stored as a fixed-size 64 bit integer.
     */
    uint64_t packetUid;
  };

  /**
   * \brief Class to hold all the metadata
   */
  class DataFreeList : public std::vector<struct Data *>
  {
public:
    ~DataFreeList ();
  };

  friend DataFreeList::~DataFreeList ();
  /// Friend class
  friend class ItemIterator;

  PacketMetadata ();

  /**
   * \brief Add a SmallItem
   * \param item the SmallItem to add
   * \return added size
   */
  inline uint16_t AddSmall (const PacketMetadata::SmallItem *item);
  /**
   * \brief Add a "Big" Item (a SmallItem plus an ExtraItem)
   * \param head the head
   * \param tail the tail
   * \param item the SmallItem to add
   * \param extraItem the ExtraItem to add
   * \return added size
   */
  uint16_t AddBig (uint32_t head, uint32_t tail,
                   const PacketMetadata::SmallItem *item, 
                   const PacketMetadata::ExtraItem *extraItem);
  /**
   * \brief Replace the tail
   * \param item the item data to write
   * \param extraItem the extra item data to write
   * \param available the number of bytes which can
   *        be written without having to rewrite the buffer entirely.
   */
  void ReplaceTail (PacketMetadata::SmallItem *item, 
                    PacketMetadata::ExtraItem *extraItem,
                    uint32_t available);
  /**
   * \brief Update the head
   * \param written the used bytes
   */
  inline void UpdateHead (uint16_t written);
  /**
   * \brief Update the tail
   * \param written the used bytes
   */
  inline void UpdateTail (uint16_t written);

  /**
   * \brief Get the ULEB128 (Unsigned Little Endian Base 128) size
   * \param value the value
   * \returns the value's ULEB128 size
   */
  inline uint32_t GetUleb128Size (uint32_t value) const;
  /**
   * \brief Read a ULEB128 (Unsigned Little Endian Base 128) coded number
   * \param pBuffer the buffer to read from
   * \returns the value
   */
  uint32_t ReadUleb128 (const uint8_t **pBuffer) const;
  /**
   * \brief Append a 16-bit value to the buffer
   * \param value the value to add
   * \param buffer the buffer to write to
   */
  inline void Append16 (uint16_t value, uint8_t *buffer);
  /**
   * \brief Append a 32-bit value to the buffer
   * \param value the value to add
   * \param buffer the buffer to write to
   */
  inline void Append32 (uint32_t value, uint8_t *buffer);
  /**
   * \brief Append a value to the buffer
   * \param value the value to add
   * \param buffer the buffer to write to
   */
  inline void AppendValue (uint32_t value, uint8_t *buffer);
  /**
   * \brief Append a value to the buffer - extra
   *
   * This function is called by AppendValue
   *
   * \param value the value to add
   * \param buffer the buffer to write to
   */
  void AppendValueExtra (uint32_t value, uint8_t *buffer);

  /**
   * \brief Reserve space
   * \param n space to reserve
   */
  inline void Reserve (uint32_t n);
  /**
   * \brief Reserve space and make a metadata copy
   * \param n space to reserve
   */
  void ReserveCopy (uint32_t n);

  /**
   * \brief Get the total size used by the metadata
   * \return the metadata used size
   */
  uint32_t GetTotalSize (void) const;

  /**
   * \brief Read items
   * \param current the offset we should start reading the data from
   * \param item pointer to where we should store the data to return to the caller
   * \param extraItem pointer to where we should store the data to return to the caller
   * \returns the number of bytes read.
   */
  uint32_t ReadItems (uint16_t current, 
                      struct PacketMetadata::SmallItem *item,
                      struct PacketMetadata::ExtraItem *extraItem) const;
  /**
   * \brief Add an header
   * \param uid header's uid to add
   * \param size header serialized size
   */
  void DoAddHeader (uint32_t uid, uint32_t size);
  /**
   * \brief Check if the metadata state is ok
   * \returns true if the internal state is ok
   */
  bool IsStateOk (void) const;
  /**
   * \brief Check if the position is valid
   * \param pointer the position to check
   * \returns true if the position is valid
   */
  bool IsPointerOk (uint16_t pointer) const;
  /**
   * \brief Check if the position is valid
   * \param pointer the position to check
   * \returns true if the position is valid
   */
  bool IsSharedPointerOk (uint16_t pointer) const;

  /**
   * \brief Recycle the buffer memory
   * \param data the buffer data storage
   */
  static void Recycle (struct PacketMetadata::Data *data);
  /**
   * \brief Create a buffer data storage
   * \param size the storage size to create
   * \returns a pointer to the created buffer storage
   */
  static struct PacketMetadata::Data *Create (uint32_t size);
  /**
   * \brief Allocate a buffer data storage
   * \param n the storage size to create
   * \returns a pointer to the allocated buffer storage
   */
  static struct PacketMetadata::Data *Allocate (uint32_t n);
  /**
   * \brief Deallocate the buffer memory
   * \param data the buffer data storage
   */
  static void Deallocate (struct PacketMetadata::Data *data);

  static DataFreeList m_freeList; //!< the metadata data storage
  static bool m_enable; //!< Enable the packet metadata
  static bool m_enableChecking; //!< Enable the packet metadata checking

  /**
   * Set to true when adding metadata to a packet is skipped because
   * m_enable is false; used to detect enabling of metadata in the
   * middle of a simulation, which isn't allowed.
   */
  static bool m_metadataSkipped;

  static uint32_t m_maxSize; //!< maximum metadata size
  static uint16_t m_chunkUid; //!< Chunk Uid

  struct Data *m_data; //!< Metadata storage
  /*
     head -(next)-> tail
       ^             |
        \---(prev)---|
   */
  uint16_t m_head; //!< list head
  uint16_t m_tail; //!< list tail
  uint16_t m_used; //!< used portion
  uint64_t m_packetUid; //!< packet Uid
};

} // namespace ns3

namespace ns3 {

PacketMetadata::PacketMetadata (uint64_t uid, uint32_t size)
  : m_data (PacketMetadata::Create (10)),
    m_head (0xffff),
    m_tail (0xffff),
    m_used (0),
    m_packetUid (uid)
{
  memset (m_data->m_data, 0xff, 4);
  if (size > 0)
    {
      DoAddHeader (0, size);
    }
}
PacketMetadata::PacketMetadata (PacketMetadata const &o)
  : m_data (o.m_data),
    m_head (o.m_head),
    m_tail (o.m_tail),
    m_used (o.m_used),
    m_packetUid (o.m_packetUid)
{
  NS_ASSERT (m_data != 0);
  NS_ASSERT (m_data->m_count < std::numeric_limits<uint32_t>::max());
  m_data->m_count++;
}
PacketMetadata &
PacketMetadata::operator = (PacketMetadata const& o)
{
  if (m_data != o.m_data) 
    {
      // not self assignment
      NS_ASSERT (m_data != 0);
      if (m_data->m_count-- == 1) 
        {
          PacketMetadata::Recycle (m_data);
        }
      m_data = o.m_data;
      NS_ASSERT (m_data != 0);
      m_data->m_count++;
    }
  m_head = o.m_head;
  m_tail = o.m_tail;
  m_used = o.m_used;
  m_packetUid = o.m_packetUid;
  return *this;
}
PacketMetadata::~PacketMetadata ()
{
  NS_ASSERT (m_data != 0);
  if (m_data->m_count-- == 1) 
    {
      PacketMetadata::Recycle (m_data);
    }
}

} // namespace ns3


#endif /* PACKET_METADATA_H */
