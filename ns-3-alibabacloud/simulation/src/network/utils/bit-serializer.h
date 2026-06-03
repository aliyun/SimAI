/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#ifndef BITSERIALIZER_H_
#define BITSERIALIZER_H_

#include <vector>

namespace ns3 {

/**
 * \ingroup packet
 *
 * \brief Bit serializer. See also \sa ns3::BitDeserializer
 *
 * This class helps converting a variable number, variable sized
 * number of bit-boundary fields to its final byte array representation.
 *
 * The typical use-case is:
 * \verbatim
   aaa: A field
   bbb: B field
   ccc: C field
   ddd: D field
   000: padding

   |aaaa abbb bbcc cdd0|
 \endverbatim
 *
 *
 * Padding can be automatically added at the end or at the start
 * of the byte blob to reach a multiple of 8 bits.
 * By default the padding is added at the end of the byte blob.
 *
 * This class should be used in two cases:
 *   - When the number of fields is large.
 *   - When the number of fields is variable.
 * In the other cases it's more convenient to use a simple bitmask.
 */

class BitSerializer
{
public:
  BitSerializer ();

  /**
   * Toggles the padding insertion policy.
   * \param padAtEnd true if the padding have to be inserted at the end of the bit blob.
   */
  void InsertPaddingAtEnd (bool padAtEnd);

  /**
   * Pushes a number of bits in the blob.
   * \param value the bits to be inserted.
   * \param significantBits Number of bits to insert.
   */
  void PushBits (uint64_t value, uint8_t significantBits);

  /**
   * Get the bytes representation of the blob.
   * Note that this operation  \b automatically add the
   * needed padding at the end (or start) of the blob.
   *
   * \warning {This operation clears the stored data.}
   *
   * \returns The byte representation of the blob.
   */
  std::vector<uint8_t> GetBytes ();

  /**
   * Get the bytes representation of the blob.
   * Note that this operation  \b automatically add the
   * needed padding at the end (or start) of the blob.
   *
   * \warning {This operation clears the stored data.}
   *
   * \param [out] buffer The buffer where to store the return data.
   * \param [in] size The size of the buffer.
   * \returns The number of bytes actually used in the buffer.
   */
  uint8_t GetBytes (uint8_t *buffer, uint32_t size);

private:
  /**
   * Add the padding at the start of the blob.
   */
  void PadAtStart ();

  /**
   * Add the padding at the end of the blob.
   */
  void PadAtEnd ();

  std::vector<bool> m_blob; //!< Blob of serialized bits.
  bool m_padAtEnd; //!< True if the padding must be added at the end of the blob.
};

} // namespace ns3

#endif /* BITSERIALIZER_H_ */
