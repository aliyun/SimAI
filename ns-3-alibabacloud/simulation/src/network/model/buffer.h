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
#ifndef BUFFER_H
#define BUFFER_H

#include <stdint.h>
#include <vector>
#include <ostream>
#include "ns3/atomic-counter.h"
#include "ns3/assert.h"

#ifndef NS3_MTP
#define BUFFER_FREE_LIST 1
#endif

namespace ns3 {

/**
 * \ingroup packet
 *
 * \brief automatically resized byte buffer
 *
 * This represents a buffer of bytes. Its size is
 * automatically adjusted to hold any data prepended
 * or appended by the user. Its implementation is optimized
 * to ensure that the number of buffer resizes is minimized,
 * by creating new Buffers of the maximum size ever used.
 * The correct maximum size is learned at runtime during use by 
 * recording the maximum size of each packet.
 *
 * \internal
 * The implementation of the Buffer class uses a COW (Copy On Write)
 * technique to ensure that the underlying data buffer which holds
 * the data bytes is shared among a lot of Buffer instances despite
 * data being added or removed from them.
 *
 * When multiple Buffer instances hold a reference to the same 
 * underlying BufferData object, they must be able to detect when
 * the operation they want to perform should trigger a copy of the
 * BufferData. If the BufferData::m_count field is one, it means that
 * there exist only one instance of Buffer which references the 
 * BufferData instance so, it is safe to modify it. It is also
 * safe to modify the content of a BufferData if the modification
 * falls outside of the "dirty area" defined by the BufferData.
 * In every other case, the BufferData must be copied before
 * being modified.
 *
 * To understand the way the Buffer::Add and Buffer::Remove methods
 * work, you first need to understand the "virtual offsets" used to
 * keep track of the content of buffers. Each Buffer instance
 * contains real data bytes in its BufferData instance but it also
 * contains "virtual zero data" which typically is used to represent
 * application-level payload. No memory is allocated to store the
 * zero bytes of application-level payload unless the user fragments
 * a Buffer: this application-level payload is kept track of with
 * a pair of integers which describe where in the buffer content
 * the "virtual zero area" starts and ends.
 *
 * \verbatim
 * ***: unused bytes
 * xxx: bytes "added" at the front of the zero area
 * ...: bytes "added" at the back of the zero area
 * 000: virtual zero bytes
 *
 * Real byte buffer:      |********xxxxxxxxxxxx.........*****|
 *                        |--------^ m_start
 *                        |-------------------^ m_zeroAreaStart
 *                        |-----------------------------^ m_end - (m_zeroAreaEnd - m_zeroAreaStart)
 * Virtual byte buffer:           |xxxxxxxxxxxx0000000000000.........|
 *                        |--------^ m_start
 *                        |--------------------^ m_zeroAreaStart
 *                        |---------------------------------^ m_zeroAreaEnd
 *                        |------------------------------------------^ m_end
 * \endverbatim
 *
 * A simple state invariant is that m_start <= m_zeroStart <= m_zeroEnd <= m_end
 */
class Buffer 
{
public:
  /**
   * \brief iterator in a Buffer instance
   */
  class Iterator 
  {
public:
    inline Iterator ();
    /**
     * go forward by one byte
     */
    inline void Next (void);
    /**
     * go backward by one byte
     */
    inline void Prev (void);
    /**
     * \param delta number of bytes to go forward
     */
    inline void Next (uint32_t delta);
    /**
     * \param delta number of bytes to go backward
     */
    inline void Prev (uint32_t delta);
    /**
     * \param o the second iterator
     * \return number of bytes included between the two iterators
     *
     * This method works only if the two iterators point
     * to the same underlying buffer. Debug builds ensure
     * this with an assert.
     */
    uint32_t GetDistanceFrom (Iterator const &o) const;

    /**
     * \return true if this iterator points to the end of the byte array.
     *     false otherwise.
     */
    bool IsEnd (void) const;
    /**
     * \return true if this iterator points to the start of the byte array.
     *     false otherwise.
     */
    bool IsStart (void) const;

    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by one byte.
     */
    inline void WriteU8 (uint8_t  data);
    /**
     * \param data data to write in buffer
     * \param len number of times data must be written in buffer
     *
     * Write the data in buffer len times and advance the iterator position
     * by len byte.
     */
    inline void WriteU8 (uint8_t data, uint32_t len);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by two bytes. The format of the data written in the byte
     * buffer is non-portable. We only ensure that readU16 will
     * return exactly what we wrote with writeU16 if the program
     * is run on the same machine.
     */
    void WriteU16 (uint16_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by four bytes. The format of the data written in the byte
     * buffer is non-portable. We only ensure that readU32 will
     * return exactly what we wrote with writeU32 if the program
     * is run on the same machine.
     */
    void WriteU32 (uint32_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by eight bytes. The format of the data written in the byte
     * buffer is non-portable. We only ensure that readU64 will
     * return exactly what we wrote with writeU64 if the program
     * is run on the same machine.
     */
    void WriteU64 (uint64_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by two bytes. The data is written in least significant byte order and the
     * input data is expected to be in host order.
     */
    void WriteHtolsbU16 (uint16_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by four bytes. The data is written in least significant byte order and the
     * input data is expected to be in host order.
     */
    void WriteHtolsbU32 (uint32_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by eight bytes. The data is written in least significant byte order and the
     * input data is expected to be in host order.
     */
    void WriteHtolsbU64 (uint64_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by two bytes. The data is written in network order and the
     * input data is expected to be in host order.
     */
    inline void WriteHtonU16 (uint16_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by four bytes. The data is written in network order and the
     * input data is expected to be in host order.
     */
    inline void WriteHtonU32 (uint32_t data);
    /**
     * \param data data to write in buffer
     *
     * Write the data in buffer and advance the iterator position
     * by eight bytes. The data is written in network order and the
     * input data is expected to be in host order.
     */
    void WriteHtonU64 (uint64_t data);
    /**
     * \param buffer a byte buffer to copy in the internal buffer.
     * \param size number of bytes to copy.
     *
     * Write the data in buffer and advance the iterator position
     * by size bytes.
     */
    void Write (uint8_t const*buffer, uint32_t size);
    /**
     * \param start the start of the data to copy
     * \param end the end of the data to copy
     *
     * Write the data delimited by start and end in internal buffer
     * and advance the iterator position by the number of bytes
     * copied.
     * The input iterators _must_ not point to the same Buffer as
     * we do to avoid overlapping copies. This is enforced
     * in debug builds by asserts.
     */
    void Write (Iterator start, Iterator end);

    /**
     * \return the byte read in the buffer.
     *
     * Read data, but do not advance the Iterator read.
     */
    inline uint8_t  PeekU8 (void);

    /**
     * \return the byte read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     */
    inline uint8_t  ReadU8 (void);
    /**
     * \return the two bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in the format written by writeU16.
     */
    inline uint16_t ReadU16 (void);
    /**
     * \return the four bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in the format written by writeU32.
     */
    uint32_t ReadU32 (void);
    /**
     * \return the eight bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in the format written by writeU64.
     */
    uint64_t ReadU64 (void);
    /**
     * \return the two bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in network format and returned in host format.
     */
    inline uint16_t ReadNtohU16 (void);
    /**
     * \return the four bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in network format and returned in host format.
     */
    inline uint32_t ReadNtohU32 (void);
    /**
     * \return the eight bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in network format and returned in host format.
     */
    uint64_t ReadNtohU64 (void);
    /**
     * \return the two bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in least significant byte format and returned in host format.
     */
    uint16_t ReadLsbtohU16 (void);
    /**
     * \return the four bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in least significant byte format and returned in host format.
     */
    uint32_t ReadLsbtohU32 (void);
    /**
     * \return the eight bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in least significant byte format and returned in host format.
     */
    uint64_t ReadLsbtohU64 (void);
    /**
     * \param buffer buffer to copy data into
     * \param size number of bytes to copy
     *
     * Copy size bytes of data from the internal buffer to the
     * input buffer and advance the Iterator by the number of
     * bytes read.
     */
    void Read (uint8_t *buffer, uint32_t size);

    /**
     * \param start start iterator of the buffer to copy data into
     * \param size  number of bytes to copy
     *
     * Copy size bytes of data from the internal buffer to the input buffer via
     * the provided iterator and advance the Iterator by the number of bytes
     * read.
     */
    inline void Read (Iterator start, uint32_t size);

    /**
     * \brief Calculate the checksum.
     * \param size size of the buffer.
     * \return checksum
     */
    uint16_t CalculateIpChecksum (uint16_t size);

    /**
     * \brief Calculate the checksum.
     * \param size size of the buffer.
     * \param initialChecksum initial value
     * \return checksum
     */
    uint16_t CalculateIpChecksum (uint16_t size, uint32_t initialChecksum);

    /**
     * \returns the size of the underlying buffer we are iterating
     */
    uint32_t GetSize (void) const;

    /**
     * \returns the size left to read of the underlying buffer we are iterating
     */
    uint32_t GetRemainingSize (void) const;

private:
    /// Friend class
    friend class Buffer;
    /**
     * Constructor - initializes the iterator to point to the buffer start
     *
     * \param buffer the buffer this iterator refers to
     */
    inline Iterator (Buffer const*buffer);
    /**
     * Constructor - initializes the iterator to point to the buffer end
     *
     * \param buffer the buffer this iterator refers to
     * \param dummy not used param
     */
    inline Iterator (Buffer const*buffer, bool dummy);
    /**
     * Initializes the iterator values
     *
     * \param buffer the buffer this iterator refers to
     */
    inline void Construct (const Buffer *buffer);
    /**
     * Checks that the [start, end) is not in the "virtual zero area".
     *
     * \param start start buffer position
     * \param end end buffer position
     * \returns true if [start, end) is not in the "virtual zero area".
     */
    bool CheckNoZero (uint32_t start, uint32_t end) const;
    /**
     * Checks that the buffer position is not in the "virtual zero area".
     *
     * \param i buffer position
     * \returns true if not in the "virtual zero area".
     */
    bool Check (uint32_t i) const;
    /**
     * \return the two bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in network format and returned in host format.
     *
     * \warning this is the slow version, please use ReadNtohU16 (void)
     */
    uint16_t SlowReadNtohU16 (void);
    /**
     * \return the four bytes read in the buffer.
     *
     * Read data and advance the Iterator by the number of bytes
     * read.
     * The data is read in network format and returned in host format.
     *
     * \warning this is the slow version, please use ReadNtohU32 (void)
     */
    uint32_t SlowReadNtohU32 (void);
    /**
     * \brief Returns an appropriate message indicating a read error
     * \returns the error message
     */
    std::string GetReadErrorMessage (void) const;
    /**
     * \brief Returns an appropriate message indicating a write error
     *
     * The message depends on the actual Buffer::Iterator status.
     *
     * \returns the error message
     */
    std::string GetWriteErrorMessage (void) const;

    /**
     * offset in virtual bytes from the start of the data buffer to the
     * start of the "virtual zero area".
     */
    uint32_t m_zeroStart;
    /**
     * offset in virtual bytes from the start of the data buffer to the
     * end of the "virtual zero area".
     */
    uint32_t m_zeroEnd;
    /**
     * offset in virtual bytes from the start of the data buffer to the
     * start of the data which can be read by this iterator
     */
    uint32_t m_dataStart;
    /**
     * offset in virtual bytes from the start of the data buffer to the
     * end of the data which can be read by this iterator
     */
    uint32_t m_dataEnd;
    /**
     * offset in virtual bytes from the start of the data buffer to the
     * current position represented by this iterator.
     */
    uint32_t m_current;
    /**
     * a pointer to the underlying byte buffer. All offsets are relative
     * to this pointer.
     */
    uint8_t *m_data;
  };

  /**
   * \return the number of bytes stored in this buffer.
   */
  inline uint32_t GetSize (void) const;

  /**
   * \return a pointer to the start of the internal 
   * byte buffer.
   *
   * The returned pointer points to an area of
   * memory which is ns3::Buffer::GetSize () bytes big.
   * Please, try to never ever use this method. It is really
   * evil and is present only for a few specific uses.
   */
  uint8_t const*PeekData (void) const;

  /**
   * \param start size to reserve
   *
   * Add bytes at the start of the Buffer. The
   * content of these bytes is undefined but debugging
   * builds initialize them to 0x33.
   * Any call to this method invalidates any Iterator
   * pointing to this Buffer.
   */
  void AddAtStart (uint32_t start);
  /**
   * \param end size to reserve
   *
   * Add bytes at the end of the Buffer. The
   * content of these bytes is undefined but debugging
   * builds initialize them to 0x33.
   * Any call to this method invalidates any Iterator
   * pointing to this Buffer.
   */
  void AddAtEnd (uint32_t end);

  /**
   * \param o the buffer to append to the end of this buffer.
   *
   * Add bytes at the end of the Buffer.
   * Any call to this method invalidates any Iterator
   * pointing to this Buffer.
   */
  void AddAtEnd (const Buffer &o);
  /**
   * \param start size to remove
   *
   * Remove bytes at the start of the Buffer.
   * Any call to this method invalidates any Iterator
   * pointing to this Buffer.
   */
  void RemoveAtStart (uint32_t start);
  /**
   * \param end size to remove
   *
   * Remove bytes at the end of the Buffer.
   * Any call to this method invalidates any Iterator
   * pointing to this Buffer.
   */
  void RemoveAtEnd (uint32_t end);

  /**
   * \param start offset from start of packet
   * \param length
   *
   * \return a fragment of size length starting at offset
   * start.
   */
  Buffer CreateFragment (uint32_t start, uint32_t length) const;

  /**
   * \return an Iterator which points to the
   * start of this Buffer.
   */
  inline Buffer::Iterator Begin (void) const;
  /**
   * \return an Iterator which points to the
   * end of this Buffer.
   */
  inline Buffer::Iterator End (void) const;

  /**
   * \brief Return the number of bytes required for serialization.
   * \return the number of bytes.
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \return zero if buffer not large enough
   * \param buffer points to serialization buffer
   * \param maxSize max number of bytes to write
   *
   * This buffer's contents are serialized into the raw 
   * character buffer parameter. Note: The zero length 
   * data is not copied entirely. Only the length of 
   * zero byte data is serialized.
   */
  uint32_t Serialize (uint8_t* buffer, uint32_t maxSize) const;

  /**
   * \return zero if a complete buffer is not deserialized
   * \param buffer points to buffer for deserialization
   * \param size number of bytes to deserialize
   *
   * The raw character buffer is deserialized and all the 
   * data is placed into this buffer.
   */
  uint32_t Deserialize (const uint8_t* buffer, uint32_t size);

  /** 
   * Copy the specified amount of data from the buffer to the given output stream.
   * 
   * @param os the output stream
   * @param size the maximum amount of bytes to copy. If zero, nothing is copied.
   */
  void CopyData (std::ostream *os, uint32_t size) const;

  /**
   * Copy the specified amount of data from the buffer to the given buffer.
   *
   * @param buffer the output buffer
   * @param size the maximum amount of bytes to copy. If zero, nothing is copied.
   * @returns the amount of bytes copied
   */
  uint32_t CopyData (uint8_t *buffer, uint32_t size) const;

  uint8_t* GetBuffer() const;
  /**
   * \brief Copy constructor
   * \param o the buffer to copy
   */
  inline Buffer (Buffer const &o);
  /**
   * \brief Assignment operator
   * \param o the buffer to copy
   * \return a reference to the buffer
   */
  Buffer &operator = (Buffer const &o);
  Buffer ();
  /**
   * \brief Constructor
   *
   * The buffer will be initialized with zeroes up to its size.
   *
   * \param dataSize the buffer size
   */
  Buffer (uint32_t dataSize);
  /**
   * \brief Constructor
   *
   * If initialize is set to true, the buffer will be initialized
   * with zeroes up to its size.
   *
   * \param dataSize the buffer size.
   * \param initialize initialize the buffer with zeroes.
   */
  Buffer (uint32_t dataSize, bool initialize);
  ~Buffer ();
private:
  /**
   * This data structure is variable-sized through its last member whose size
   * is determined at allocation time and stored in the m_size field.
   *
   * The so-called "dirty area" describes the area in the buffer which
   * has been reserved and used by a user. Multiple Buffer instances
   * may reference the same Buffer::Data object instance and may
   * reference different parts of the underlying byte buffer. The
   * "dirty area" is union of all the areas referenced by the Buffer
   * instances which reference the same BufferData instance.
   * New user data can be safely written only outside of the "dirty
   * area" if the reference count is higher than 1 (that is, if
   * more than one Buffer instance references the same BufferData).
   */
  struct Data
  {
    /**
     * The reference count of an instance of this data structure.
     * Each buffer which references an instance holds a count.
     */
#ifdef NS3_MTP
    AtomicCounter m_count;
#else
    uint32_t m_count;
#endif
    /**
     * the size of the m_data field below.
     */
    uint32_t m_size;
    /**
     * offset from the start of the m_data field below to the
     * start of the area in which user bytes were written.
     */
    uint32_t m_dirtyStart;
    /**
     * offset from the start of the m_data field below to the
     * end of the area in which user bytes were written.
     */
    uint32_t m_dirtyEnd;
    /**
     * The real data buffer holds _at least_ one byte.
     * Its real size is stored in the m_size field.
     */
    uint8_t m_data[1];
  };

  /**
   * \brief Create a full copy of the buffer, including
   * all the internal structures.
   *
   * \returns a copy of the buffer
   */
  Buffer CreateFullCopy (void) const;

  /**
   * \brief Transform a "Virtual byte buffer" into a "Real byte buffer"
   */
  void TransformIntoRealBuffer (void) const;
  /**
   * \brief Checks the internal buffer structures consistency
   *
   * Used only for debugging purposes.
   *
   * \returns true if the buffer status is consistent.
   */
  bool CheckInternalState (void) const;

  /**
   * \brief Initializes the buffer with a number of zeroes.
   *
   * \param zeroSize the zeroes size
   */
  void Initialize (uint32_t zeroSize);

  /**
   * \brief Get the buffer real size.
   * \warning The real size is the actual memory used by the buffer.
   * \returns the memory used by the buffer.
   */
  uint32_t GetInternalSize (void) const;

  /**
   * \brief Get the buffer end position.
   * \returns the buffer end index.
   */
  uint32_t GetInternalEnd (void) const;

  /**
   * \brief Recycle the buffer memory
   * \param data the buffer data storage
   */
  static void Recycle (struct Buffer::Data *data);
  /**
   * \brief Create a buffer data storage
   * \param size the storage size to create
   * \returns a pointer to the created buffer storage
   */
  static struct Buffer::Data *Create (uint32_t size);
  /**
   * \brief Allocate a buffer data storage
   * \param reqSize the storage size to create
   * \returns a pointer to the allocated buffer storage
   */
  static struct Buffer::Data *Allocate (uint32_t reqSize);
  /**
   * \brief Deallocate the buffer memory
   * \param data the buffer data storage
   */
  static void Deallocate (struct Buffer::Data *data);

  struct Data *m_data; //!< the buffer data storage

  /**
   * keep track of the maximum value of m_zeroAreaStart across
   * the lifetime of a Buffer instance. This variable is used
   * purely as a source of information for the heuristics which
   * decide on the position of the zero area in new buffers.
   * It is read from the Buffer destructor to update the global
   * heuristic data and these global heuristic data are used from
   * the Buffer constructor to choose an initial value for 
   * m_zeroAreaStart.
   */
  uint32_t m_maxZeroAreaStart;
  /**
   * location in a newly-allocated buffer where you should start
   * writing data. i.e., m_start should be initialized to this 
   * value.
   */
  static uint32_t g_recommendedStart;

  /**
   * offset to the start of the virtual zero area from the start
   * of m_data->m_data
   */
  uint32_t m_zeroAreaStart;
  /**
   * offset to the end of the virtual zero area from the start
   * of m_data->m_data
   */
  uint32_t m_zeroAreaEnd;
  /**
   * offset to the start of the data referenced by this Buffer
   * instance from the start of m_data->m_data
   */
  uint32_t m_start;
  /**
   * offset to the end of the data referenced by this Buffer
   * instance from the start of m_data->m_data
   */
  uint32_t m_end;

#ifdef BUFFER_FREE_LIST
  /// Container for buffer data
  typedef std::vector<struct Buffer::Data*> FreeList;
  /// Local static destructor structure
  struct LocalStaticDestructor 
  {
    ~LocalStaticDestructor ();
  };
  static uint32_t g_maxSize; //!< Max observed data size
  static FreeList *g_freeList; //!< Buffer data container
  static struct LocalStaticDestructor g_localStaticDestructor; //!< Local static destructor
#endif
};

} // namespace ns3

#include "ns3/assert.h"
#include <cstring>

namespace ns3 {

Buffer::Iterator::Iterator ()
  : m_zeroStart (0),
    m_zeroEnd (0),
    m_dataStart (0),
    m_dataEnd (0),
    m_current (0),
    m_data (0)
{
}
Buffer::Iterator::Iterator (Buffer const*buffer)
{
  Construct (buffer);
  m_current = m_dataStart;
}
Buffer::Iterator::Iterator (Buffer const*buffer, bool dummy)
{
  Construct (buffer);
  m_current = m_dataEnd;
}

void
Buffer::Iterator::Construct (const Buffer *buffer)
{
  m_zeroStart = buffer->m_zeroAreaStart;
  m_zeroEnd = buffer->m_zeroAreaEnd;
  m_dataStart = buffer->m_start;
  m_dataEnd = buffer->m_end;
  m_data = buffer->m_data->m_data;
}

void 
Buffer::Iterator::Next (void)
{
  NS_ASSERT (m_current + 1 <= m_dataEnd);
  m_current++;
}
void 
Buffer::Iterator::Prev (void)
{
  NS_ASSERT (m_current >= 1);
  m_current--;
}
void 
Buffer::Iterator::Next (uint32_t delta)
{
  NS_ASSERT (m_current + delta <= m_dataEnd);
  m_current += delta;
}
void 
Buffer::Iterator::Prev (uint32_t delta)
{
  NS_ASSERT (m_current >= delta);
  m_current -= delta;
}
void
Buffer::Iterator::WriteU8 (uint8_t data)
{
  NS_ASSERT_MSG (Check (m_current),
                 GetWriteErrorMessage ());

  if (m_current < m_zeroStart)
    {
      m_data[m_current] = data;
      m_current++;
    }
  else
    {
      m_data[m_current - (m_zeroEnd-m_zeroStart)] = data;
      m_current++;
    }
}

void 
Buffer::Iterator::WriteU8 (uint8_t  data, uint32_t len)
{
  NS_ASSERT_MSG (CheckNoZero (m_current, m_current + len),
                 GetWriteErrorMessage ());
  if (m_current <= m_zeroStart)
    {
      std::memset (&(m_data[m_current]), data, len);
      m_current += len;
    }
  else
    {
      uint8_t *buffer = &m_data[m_current - (m_zeroEnd-m_zeroStart)];
      std::memset (buffer, data, len);
      m_current += len;
    }
}

void 
Buffer::Iterator::WriteHtonU16 (uint16_t data)
{
  NS_ASSERT_MSG (CheckNoZero (m_current, m_current + 2),
                 GetWriteErrorMessage ());
  uint8_t *buffer;
  if (m_current + 2 <= m_zeroStart)
    {
      buffer = &m_data[m_current];
    }
  else
    {
      buffer = &m_data[m_current - (m_zeroEnd - m_zeroStart)];
    }
  buffer[0] = (data >> 8)& 0xff;
  buffer[1] = (data >> 0)& 0xff;
  m_current+= 2;
}

void 
Buffer::Iterator::WriteHtonU32 (uint32_t data)
{
  NS_ASSERT_MSG (CheckNoZero (m_current, m_current + 4),
                 GetWriteErrorMessage ());

  uint8_t *buffer;
  if (m_current + 4 <= m_zeroStart)
    {
      buffer = &m_data[m_current];
    }
  else
    {
      buffer = &m_data[m_current - (m_zeroEnd - m_zeroStart)];
    }
  buffer[0] = (data >> 24)& 0xff;
  buffer[1] = (data >> 16)& 0xff;
  buffer[2] = (data >> 8)& 0xff;
  buffer[3] = (data >> 0)& 0xff;
  m_current+= 4;
}

uint16_t 
Buffer::Iterator::ReadNtohU16 (void)
{
  uint8_t *buffer;
  if (m_current + 2 <= m_zeroStart)
    {
      buffer = &m_data[m_current];
    }
  else if (m_current >= m_zeroEnd)
    {
      buffer = &m_data[m_current - (m_zeroEnd - m_zeroStart)];
    }
  else
    {
      return SlowReadNtohU16 ();
    }
  uint16_t retval = 0;
  retval |= buffer[0];
  retval <<= 8;
  retval |= buffer[1];
  m_current += 2;
  return retval;
}

uint32_t 
Buffer::Iterator::ReadNtohU32 (void)
{
  uint8_t *buffer;
  if (m_current + 4 <= m_zeroStart)
    {
      buffer = &m_data[m_current];
    }
  else if (m_current >= m_zeroEnd)
    {
      buffer = &m_data[m_current - (m_zeroEnd - m_zeroStart)];
    }
  else
    {
      return SlowReadNtohU32 ();
    }
  uint32_t retval = 0;
  retval |= buffer[0];
  retval <<= 8;
  retval |= buffer[1];
  retval <<= 8;
  retval |= buffer[2];
  retval <<= 8;
  retval |= buffer[3];
  m_current += 4;
  return retval;
}

uint8_t
Buffer::Iterator::PeekU8 (void)
{
  NS_ASSERT_MSG (m_current >= m_dataStart &&
                 m_current < m_dataEnd,
                 GetReadErrorMessage ());

  if (m_current < m_zeroStart)
    {
      uint8_t data = m_data[m_current];
      return data;
    }
  else if (m_current < m_zeroEnd)
    {
      return 0;
    }
  else
    {
      uint8_t data = m_data[m_current - (m_zeroEnd-m_zeroStart)];
      return data;
    }
}

uint8_t
Buffer::Iterator::ReadU8 (void)
{
  uint8_t ret = PeekU8 ();
  m_current ++;
  return ret;
}

uint16_t 
Buffer::Iterator::ReadU16 (void)
{
  uint8_t byte0 = ReadU8 ();
  uint8_t byte1 = ReadU8 ();
  uint16_t data = byte1;
  data <<= 8;
  data |= byte0;

  return data;
}

void
Buffer::Iterator::Read (Buffer::Iterator start, uint32_t size)
{
  Buffer::Iterator end = *this;
  end.Next (size);
  
  start.Write (*this, end);
}


Buffer::Buffer (Buffer const&o)
  : m_data (o.m_data),
    m_maxZeroAreaStart (o.m_zeroAreaStart),
    m_zeroAreaStart (o.m_zeroAreaStart),
    m_zeroAreaEnd (o.m_zeroAreaEnd),
    m_start (o.m_start),
    m_end (o.m_end)
{
  m_data->m_count++;
  NS_ASSERT (CheckInternalState ());
}

uint32_t 
Buffer::GetSize (void) const
{
  return m_end - m_start;
}

Buffer::Iterator 
Buffer::Begin (void) const
{
  NS_ASSERT (CheckInternalState ());
  return Buffer::Iterator (this);
}
Buffer::Iterator 
Buffer::End (void) const
{
  NS_ASSERT (CheckInternalState ());
  return Buffer::Iterator (this, false);
}



} // namespace ns3

#endif /* BUFFER_H */
