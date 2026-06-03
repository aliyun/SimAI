/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Lluis Parcerisa <lparcerisa@cttc.cat>
 */

#ifndef ASN1_HEADER_H
#define ASN1_HEADER_H

#include "ns3/header.h"

#include <bitset>
#include <string>

namespace ns3 {

/**
 * This class has the purpose to encode Information Elements according
 * to ASN.1 syntax, as defined in ITU-T  X-691.
 * IMPORTANT: The encoding is done following the UNALIGNED variant.
 */
class Asn1Header : public Header
{
public:
  Asn1Header ();
  virtual ~Asn1Header ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator bIterator) const;

  // Inherited from ns3::Header base class
  // Pure virtual methods, to be implemented in child classes
  virtual uint32_t Deserialize (Buffer::Iterator bIterator) = 0;
  virtual void Print (std::ostream &os) const = 0;
    
  /**
   * This function serializes class attributes to m_serializationResult
   * local Buffer.  As ASN1 encoding produces a bitstream that does not have
   * a fixed length, this function is needed to store the result, so
   * its length can be retrieved with Header::GetSerializedSize() function.
   * This method is pure virtual in this class (needs to be implemented
   * in child classes) as the meaningful information elements are in
   * the subclasses.
   */
  virtual void PreSerialize (void) const = 0;

protected:
  mutable uint8_t m_serializationPendingBits; //!< pending bits
  mutable uint8_t m_numSerializationPendingBits; //!< number of pending bits
  mutable bool m_isDataSerialized; //!< true if data is serialized
  mutable Buffer m_serializationResult; //!< serialization result

  /**
   * Function to write in m_serializationResult, after resizing its size
   * \param octet bits to write
   */
  void WriteOctet (uint8_t octet) const;

  // Serialization functions

  /**
   * Serialize a bool
   * \param value value to serialize
   */
  void SerializeBoolean (bool value) const;
  /**
   * Serialize an Integer
   * \param n value to serialize
   * \param nmin min value to serialize
   * \param nmax max value to serialize
   */
  void SerializeInteger (int n, int nmin, int nmax) const;
  // void SerializeOctetstring (std::string s) const;
  /**
   * Serialize a Sequence
   * \param numElems element number to serialize
   * \param nMax max value to serialize
   * \param nMin min value to serialize
   */
  void SerializeSequenceOf (int numElems, int nMax, int nMin) const;
  /**
   * Serialize a Choice (set of options)
   * \param numOptions number of options
   * \param selectedOption selected option
   * \param isExtensionMarkerPresent true if extension mark is present
   */
  void SerializeChoice (int numOptions, int selectedOption,
                        bool isExtensionMarkerPresent) const;
  /**
   * Serialize an Enum
   * \param numElems number of elements in the enum
   * \param selectedElem selected element
   */
  void SerializeEnum (int numElems, int selectedElem) const;
  /**
   * Serialize nothing (null op)
   */
  void SerializeNull () const;
  /**
   * Finalizes an in progress serialization.
   */
  void FinalizeSerialization () const;

  /**
   * Serialize a bitset
   * \param data data to serialize
   */
  template <int N>
  void SerializeBitset (std::bitset<N> data) const;

  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  template <int N>
  void SerializeSequence (std::bitset<N> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<0> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<1> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<2> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<3> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<4> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<5> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<6> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<7> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<9> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<10> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;
  /**
   * Serialize a sequence
   * \param optionalOrDefaultMask Mask to serialize
   * \param isExtensionMarkerPresent true if Extension Marker is present
   */
  void SerializeSequence (std::bitset<11> optionalOrDefaultMask,
                          bool isExtensionMarkerPresent) const;

  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  template <int N>
  void SerializeBitstring (std::bitset<N> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<1> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<2> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<8> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<10> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<16> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<27> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<28> bitstring) const;
  /**
   * Serialize a bitstring
   * \param bitstring bitstring to serialize
   */
  void SerializeBitstring (std::bitset<32> bitstring) const;

  // Deserialization functions

  /**
   * Deserialize a bitset
   * \param data buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  template <int N>
  Buffer::Iterator DeserializeBitset (std::bitset<N> *data,
                                      Buffer::Iterator bIterator);
  /**
   * Deserialize a bitset
   * \param data buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitset (std::bitset<8> *data,
                                      Buffer::Iterator bIterator);

  /**
   * Deserialize a boolean
   * \param value buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBoolean (bool *value,
                                       Buffer::Iterator bIterator);
  /**
   * Deserialize an integer
   * \param n buffer to store the result
   * \param nmin min value to serialize
   * \param nmax max value to serialize
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeInteger (int *n, int nmin, int nmax,
                                       Buffer::Iterator bIterator);
  /**
   * Deserialize a Choice (set of options)
   * \param numOptions number of options
   * \param isExtensionMarkerPresent true if extension mark is present
   * \param selectedOption buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeChoice (int numOptions,
                                      bool isExtensionMarkerPresent,
                                      int *selectedOption,
                                      Buffer::Iterator bIterator);
  /**
   * Deserialize an Enum
   * \param numElems number of elements in the enum
   * \param selectedElem buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeEnum (int numElems, int *selectedElem,
                                    Buffer::Iterator bIterator);

  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  template <int N>
  Buffer::Iterator DeserializeSequence (std::bitset<N> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<0> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<1> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<2> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<3> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<4> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<5> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<6> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);

  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<7> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<9> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<10> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);
  /**
   * Deserialize a sequence
   * \param optionalOrDefaultMask buffer to store the result
   * \param isExtensionMarkerPresent true if Extension Marker is present
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequence (std::bitset<11> *optionalOrDefaultMask,
                                        bool isExtensionMarkerPresent,
                                        Buffer::Iterator bIterator);

  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
 template <int N>
  Buffer::Iterator DeserializeBitstring (std::bitset<N> *bitstring,
                                         Buffer::Iterator bIterator);
 /**
  * Deserialize a bitstring
  * \param bitstring buffer to store the result
  * \param bIterator buffer iterator
  * \returns the modified buffer iterator
  */
  Buffer::Iterator DeserializeBitstring (std::bitset<1> *bitstring,
                                         Buffer::Iterator bIterator);
  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitstring (std::bitset<2> *bitstring,
                                         Buffer::Iterator bIterator);
  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitstring (std::bitset<8> *bitstring,
                                         Buffer::Iterator bIterator);
  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitstring (std::bitset<10> *bitstring,
                                         Buffer::Iterator bIterator);
  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitstring (std::bitset<16> *bitstring,
                                         Buffer::Iterator bIterator);
  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitstring (std::bitset<27> *bitstring,
                                         Buffer::Iterator bIterator);
  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitstring (std::bitset<28> *bitstring,
                                         Buffer::Iterator bIterator);
  /**
   * Deserialize a bitstring
   * \param bitstring buffer to store the result
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeBitstring (std::bitset<32> *bitstring,
                                         Buffer::Iterator bIterator);

  /**
   * Deserialize nothing (null op)
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeNull (Buffer::Iterator bIterator);
  /**
   * Deserialize a Sequence
   * \param numElems buffer to store the result
   * \param nMax max value to serialize
   * \param nMin min value to serialize
   * \param bIterator buffer iterator
   * \returns the modified buffer iterator
   */
  Buffer::Iterator DeserializeSequenceOf (int *numElems, int nMax, int nMin,
                                          Buffer::Iterator bIterator);
};

} // namespace ns3

#endif // ASN1_HEADER_H

