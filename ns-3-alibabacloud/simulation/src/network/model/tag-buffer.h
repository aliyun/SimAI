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
#ifndef TAG_BUFFER_H
#define TAG_BUFFER_H

#include <stdint.h>

#define TAG_BUFFER_USE_INLINE 1

#ifdef TAG_BUFFER_USE_INLINE
#define TAG_BUFFER_INLINE inline
#else
#define TAG_BUFFER_INLINE
#endif

namespace ns3 {

/**
 * \ingroup packet
 *
 * \brief read and write tag data
 *
 * This class allows subclasses of the ns3::Tag base class
 * to serialize and deserialize their data through a stream-like
 * API. This class keeps track of the "current" point in the
 * buffer and advances that "current" point every time data is 
 * written. The in-memory format of the data written by 
 * this class is unspecified.
 *
 * If the user attempts to write more data in the buffer than 
 * he allocated with Tag::GetSerializedSize, he will trigger
 * an NS_ASSERT error.
 */
class TagBuffer
{
public:

  /**
   * \brief Constructor
   * \param start start position
   * \param end end position
   */
  TagBuffer (uint8_t *start, uint8_t *end);

  /**
   * \brief Trim some space from the end
   * \param trim space to remove
   */
  void TrimAtEnd (uint32_t trim);

  /**
   * \brief Copy the nternal structure of another TagBuffer
   * \param o the TagBuffer to copy from
   */
  void CopyFrom (TagBuffer o);

  /**
   * \param v the value to write
   *
   * Write one byte and advance the "current" point by one.
   */
  TAG_BUFFER_INLINE void WriteU8 (uint8_t v);
  /**
   * \param v the value to write
   *
   * Write two bytes and advance the "current" point by two.
   */
  TAG_BUFFER_INLINE void WriteU16 (uint16_t v);
  /**
   * \param v the value to write
   *
   * Write four bytes and advance the "current" point by four.
   */
  TAG_BUFFER_INLINE void WriteU32 (uint32_t v);
  /**
   * \param v the value to write
   *
   * Write eight bytes and advance the "current" point by eight.
   */
  void WriteU64 (uint64_t v);
  /**
   * \param v the value to write
   *
   * Write a double and advance the "current" point by the size of the
   * data written.
   */
  void WriteDouble (double v);
  /**
   * \param buffer a pointer to data to write
   * \param size the size of the data to write
   *
   * Write all the input data and advance the "current" point by the size of the
   * data written.
   */
  void Write (const uint8_t *buffer, uint32_t size);
  /**
   * \returns the value read
   *
   * Read one byte, advance the "current" point by one,
   * and return the value read.
   */
  TAG_BUFFER_INLINE uint8_t  ReadU8 (void);
  /**
   * \returns the value read
   *
   * Read two bytes, advance the "current" point by two,
   * and return the value read.
   */
  TAG_BUFFER_INLINE uint16_t ReadU16 (void);
  /**
   * \returns the value read
   *
   * Read four bytes, advance the "current" point by four,
   * and return the value read.
   */
  TAG_BUFFER_INLINE uint32_t ReadU32 (void);
  /**
   * \returns the value read
   *
   * Read eight bytes, advance the "current" point by eight,
   * and return the value read.
   */
  uint64_t ReadU64 (void);
  /**
   * \returns the value read
   *
   * Read a double, advance the "current" point by the size
   * of the data read, and, return the value read.
   */
  double ReadDouble (void);
  /**
   * \param buffer a pointer to the buffer where data should be
   * written.
   * \param size the number of bytes to read.
   *
   * Read the number of bytes requested, advance the "current"
   * point by the number of bytes read, return.
   */
  void Read (uint8_t *buffer, uint32_t size);
private:

  uint8_t *m_current; //!< current TagBuffer position
  uint8_t *m_end;     //!< end TagBuffer position
};

} // namespace ns3

#ifdef TAG_BUFFER_USE_INLINE

#include "ns3/assert.h"

namespace ns3 {

void 
TagBuffer::WriteU8 (uint8_t v)
{
  NS_ASSERT (m_current + 1 <= m_end);
  *m_current = v;
  m_current++;
}

void 
TagBuffer::WriteU16 (uint16_t data)
{
  WriteU8 ((data >> 0) & 0xff);
  WriteU8 ((data >> 8) & 0xff);
}
void 
TagBuffer::WriteU32 (uint32_t data)
{
  WriteU8 ((data >> 0) & 0xff);
  WriteU8 ((data >> 8) & 0xff);
  WriteU8 ((data >> 16) & 0xff);
  WriteU8 ((data >> 24) & 0xff);
}

uint8_t
TagBuffer::ReadU8 (void)
{
  NS_ASSERT (m_current + 1 <= m_end);
  uint8_t v;
  v = *m_current;
  m_current++;
  return v;
}

uint16_t 
TagBuffer::ReadU16 (void)
{
  uint8_t byte0 = ReadU8 ();
  uint8_t byte1 = ReadU8 ();
  uint16_t data = byte1;
  data <<= 8;
  data |= byte0;
  return data;
}
uint32_t 
TagBuffer::ReadU32 (void)
{
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

} // namespace ns3

#endif /* TAG_BUFFER_USE_INLINE */

#endif /* TAG_BUFFER_H */
