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

#include "ns3/log.h"
#include "ns3/lte-asn1-header.h"

#include <stdio.h>
#include <sstream>
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Asn1Header");

NS_OBJECT_ENSURE_REGISTERED (Asn1Header);

TypeId
Asn1Header::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Asn1Header")
    .SetParent<Header> ()
    .SetGroupName("Lte")
  ;
  return tid;
}

TypeId
Asn1Header::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

Asn1Header::Asn1Header ()
{
  m_serializationPendingBits = 0x00;
  m_numSerializationPendingBits = 0;
  m_isDataSerialized = false;
}

Asn1Header::~Asn1Header ()
{
}

uint32_t
Asn1Header::GetSerializedSize (void) const
{
  if (!m_isDataSerialized)
    {
      PreSerialize ();
    }
  return m_serializationResult.GetSize ();
}

void Asn1Header::Serialize (Buffer::Iterator bIterator) const
{
  if (!m_isDataSerialized)
    {
      PreSerialize ();
    }
  bIterator.Write (m_serializationResult.Begin (),m_serializationResult.End ());
}

void Asn1Header::WriteOctet (uint8_t octet) const
{
  m_serializationResult.AddAtEnd (1);
  Buffer::Iterator bIterator = m_serializationResult.End ();
  bIterator.Prev ();
  bIterator.WriteU8 (octet);
}

template <int N>
void Asn1Header::SerializeBitset (std::bitset<N> data) const
{

  size_t dataSize = data.size ();
  uint8_t pendingBits = dataSize;
  uint8_t mask = 1;
  int j;

  // No extension marker (Clause 16.7 ITU-T X.691),
  // as 3GPP TS 36.331 does not use it in its IE's.

  // Clause 16.8 ITU-T X.691
  if (dataSize == 0)
    {
      return;
    }

  // Clause 16.9 ITU-T X.691
  // Clause 16.10 ITU-T X.691
  if (dataSize <= 65536)
    {
      // If there are bits pending to be processed,
      // append first bits in data to complete an octet.
      if (m_numSerializationPendingBits > 0)
        {
          mask = 0x80 >> m_numSerializationPendingBits;
          while (pendingBits > 0 && m_numSerializationPendingBits < 8)
            {
              m_serializationPendingBits |= (data[pendingBits - 1]) ? mask : 0;
              pendingBits--;
              m_numSerializationPendingBits++;
              mask = (mask >> 1) & (~mask);
            }

          if (m_numSerializationPendingBits >= 8)
            {
              WriteOctet (m_serializationPendingBits);
              m_numSerializationPendingBits = 0;
              m_serializationPendingBits = 0;
            }
        }

      while (pendingBits > 0)
        {
          mask = 1;
          j = 8;

          // If there are less than 8 remaining bits,
          // store it to m_serializationPendingBits.
          if (pendingBits < 8)
            {
              mask = 0x80;
              m_numSerializationPendingBits = pendingBits;
              while (pendingBits > 0)
                {
                  m_serializationPendingBits |= (data[pendingBits - 1]) ? mask : 0;
                  mask = (mask >> 1) & (~mask);
                  pendingBits--;
                }
            }

          // Write the data to buffer
          else
            {
              uint8_t octetToWrite = 0;
              for (; j > 0; j--)
                {
                  octetToWrite |= (data[pendingBits - j]) ? mask : 0;
                  mask = (mask << 1) & (~mask);
                }
              WriteOctet (octetToWrite);
              pendingBits -= 8;
            }
        }
    }

  // Clause 16.11 ITU-T X.691
  else
    {
      printf ("FRAGMENTATION NEEDED!\n");
    }
}

template <int N>
void Asn1Header::SerializeBitstring (std::bitset<N> data) const
{
  SerializeBitset<N> (data);
}


void Asn1Header::SerializeBitstring (std::bitset<1> data) const
{
  SerializeBitstring<1> (data);
}

void Asn1Header::SerializeBitstring (std::bitset<2> data) const
{
  SerializeBitstring<2> (data);
}

void Asn1Header::SerializeBitstring (std::bitset<8> data) const
{
  SerializeBitstring<8> (data);
}

void Asn1Header::SerializeBitstring (std::bitset<10> data) const
{
  SerializeBitstring<10> (data);
}

void Asn1Header::SerializeBitstring (std::bitset<16> data) const
{
  SerializeBitstring<16> (data);
}

void Asn1Header::SerializeBitstring (std::bitset<27> data) const
{
  SerializeBitstring<27> (data);
}

void Asn1Header::SerializeBitstring (std::bitset<28> data) const
{
  SerializeBitstring<28> (data);
}

void Asn1Header::SerializeBitstring (std::bitset<32> data) const
{
  SerializeBitstring<32> (data);
}

void Asn1Header::SerializeBoolean (bool value) const
{
  // Clause 12 ITU-T X.691
  std::bitset<1> val;
  (value) ? val.set () : val.reset ();
  SerializeBitset<1> (val);
}

template <int N>
void Asn1Header::SerializeSequence (std::bitset<N> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  if (isExtensionMarkerPresent)
    {
      // Extension marker present, but no extension
      SerializeBoolean (false);
    }
  SerializeBitstring<N> (optionalOrDefaultMask);
}

void Asn1Header::SerializeSequence (std::bitset<0> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<0> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<1> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<1> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<2> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<2> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<3> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<3> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<4> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<4> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<5> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<5> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<6> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<6> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<7> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<7> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<9> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<9> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<10> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<10> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequence (std::bitset<11> optionalOrDefaultMask, bool isExtensionMarkerPresent) const
{
  SerializeSequence<11> (optionalOrDefaultMask,isExtensionMarkerPresent);
}

void Asn1Header::SerializeSequenceOf (int numElems, int nMax, int nMin) const
{
  // Clause 20.6 ITU-T X.691
  SerializeInteger (numElems, nMin, nMax);
}

void Asn1Header::SerializeEnum (int numElems, int selectedElem) const
{
  // Clause 14 ITU-T X.691
  SerializeInteger (selectedElem, 0, numElems - 1);
}

void Asn1Header::SerializeChoice (int numOptions, int selectedOption, bool isExtensionMarkerPresent) const
{
  if(isExtensionMarkerPresent)
  {
    // Never extended attributes
    SerializeBoolean(false);
  }

  // Clause 23.4 ITU-T X.691
  if (numOptions < 2)
    {
      return;
    }

  SerializeInteger (selectedOption,0,numOptions - 1);
}

void Asn1Header::SerializeInteger (int n, int nmin, int nmax) const
{
  // The following is equivalent to:
  //  NS_ASSERT_MSG (nmin <= n && n <= nmax,
  //               "Integer " << n << " is outside range [" << nmin << ", " << nmax << "]");
  // This is a workaround to gcc-7 aggressive optimization, see #346, and can be dropped
  // once gcc-7 will not be anymore supported.
  long int nComp = nmin;
  nComp -= n;
  NS_ASSERT_MSG (nComp <= 0 && n <= nmax,
                 "Integer " << n << " is outside range [" << nmin << ", " << nmax << "]");

  // Clause 11.5.3 ITU-T X.691
  int range = nmax - nmin + 1;
  // Subtract nmin to n
  n -= nmin;

  // Clause 11.5.4 ITU-T X.691
  if (range <= 1)
    {
      return;
    }

  // Clause 11.5.6 ITU-T X.691
  int requiredBits = std::ceil (std::log (range) / std::log (2.0));

  switch (requiredBits)
    {
    case 1:
      SerializeBitset<1> (std::bitset<1> (n));
      break;
    case 2:
      SerializeBitset<2> (std::bitset<2> (n));
      break;
    case 3:
      SerializeBitset<3> (std::bitset<3> (n));
      break;
    case 4:
      SerializeBitset<4> (std::bitset<4> (n));
      break;
    case 5:
      SerializeBitset<5> (std::bitset<5> (n));
      break;
    case 6:
      SerializeBitset<6> (std::bitset<6> (n));
      break;
    case 7:
      SerializeBitset<7> (std::bitset<7> (n));
      break;
    case 8:
      SerializeBitset<8> (std::bitset<8> (n));
      break;
    case 9:
      SerializeBitset<9> (std::bitset<9> (n));
      break;
    case 10:
      SerializeBitset<10> (std::bitset<10> (n));
      break;
    case 11:
      SerializeBitset<11> (std::bitset<11> (n));
      break;
    case 12:
      SerializeBitset<12> (std::bitset<12> (n));
      break;
    case 13:
      SerializeBitset<13> (std::bitset<13> (n));
      break;
    case 14:
      SerializeBitset<14> (std::bitset<14> (n));
      break;
    case 15:
      SerializeBitset<15> (std::bitset<15> (n));
      break;
    case 16:
      SerializeBitset<16> (std::bitset<16> (n));
      break;
    case 17:
      SerializeBitset<17> (std::bitset<17> (n));
      break;
    case 18:
      SerializeBitset<18> (std::bitset<18> (n));
      break;
    case 19:
      SerializeBitset<19> (std::bitset<19> (n));
      break;
    case 20:
      SerializeBitset<20> (std::bitset<20> (n));
      break;
    default:
      {
        std::cout << "SerializeInteger " << requiredBits << " Out of range!!" << std::endl;
        exit (1);
      }
    }
}

void Asn1Header::SerializeNull () const
{
  // Clause 18 ITU-T X.691
  return;
}

void Asn1Header::FinalizeSerialization () const
{
  if (m_numSerializationPendingBits > 0)
    {
      m_numSerializationPendingBits = 0;
      SerializeBitset<8> (std::bitset<8> (m_serializationPendingBits));
    }
  m_isDataSerialized = true;
}

template <int N>
Buffer::Iterator Asn1Header::DeserializeBitset (std::bitset<N> *data, Buffer::Iterator bIterator)
{

  int bitsToRead = N;
  uint8_t mask;

  // Read bits from pending bits
  if (m_numSerializationPendingBits > 0)
    {
      while (bitsToRead > 0 && m_numSerializationPendingBits > 0)
        {
          data->set (bitsToRead - 1,(m_serializationPendingBits & 0x80) ? 1 : 0);
          bitsToRead--;
          m_numSerializationPendingBits--;
          m_serializationPendingBits = m_serializationPendingBits << 1;
        }
    }

  // Read bits from buffer
  while (bitsToRead > 0)
    {
      uint8_t octet = bIterator.ReadU8 ();
      // If 8 bits can be allocated to the bitset, set the bits
      if (bitsToRead >= 8)
        {
          mask = 0x80;
          for (int j = 0; j < 8; j++)
            {
              data->set (bitsToRead - 1,(octet & mask) ? 1 : 0);
              bitsToRead--;
              mask = mask >> 1;
            }
        }

      // Otherwise, we'll have to save the remaining bits
      else
        {
          mask = 0x80;
          m_numSerializationPendingBits = 8 - bitsToRead;
          m_serializationPendingBits = octet << bitsToRead;
          while (bitsToRead > 0)
            {
              data->set (bitsToRead - 1,(octet & mask) ? 1 : 0);
              bitsToRead--;
              mask = mask >> 1;
            }
        }
    }

  return bIterator;
}

template <int N>
Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<N> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitset<N> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<1> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<1> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<2> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<2> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<8> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<8> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<10> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<10> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<16> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<16> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<27> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<27> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<28> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<28> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBitstring (std::bitset<32> *data, Buffer::Iterator bIterator)
{
  return DeserializeBitstring<32> (data,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeBoolean (bool *value, Buffer::Iterator bIterator)
{
  std::bitset<1> readBit;
  bIterator = DeserializeBitset<1> (&readBit,bIterator);
  *value = (readBit[0] == 1);
  return bIterator;
}

Buffer::Iterator Asn1Header::DeserializeInteger (int *n, int nmin, int nmax, Buffer::Iterator bIterator)
{
  // Misusage check: Ensure nmax>nmin ...
  if (nmin > nmax)
    {
      int aux = nmin;
      nmin = nmax;
      nmax = aux;
    }

  int range = nmax - nmin + 1;

  if (range == 1)
    {
      return bIterator;
    }

  int requiredBits = std::ceil (std::log (range) / std::log (2.0));

  std::bitset<1> bitsRead1;
  std::bitset<2> bitsRead2;
  std::bitset<3> bitsRead3;
  std::bitset<4> bitsRead4;
  std::bitset<5> bitsRead5;
  std::bitset<6> bitsRead6;
  std::bitset<7> bitsRead7;
  std::bitset<8> bitsRead8;
  std::bitset<9> bitsRead9;
  std::bitset<10> bitsRead10;
  std::bitset<11> bitsRead11;
  std::bitset<12> bitsRead12;
  std::bitset<13> bitsRead13;
  std::bitset<14> bitsRead14;
  std::bitset<15> bitsRead15;
  std::bitset<16> bitsRead16;
  std::bitset<17> bitsRead17;
  std::bitset<18> bitsRead18;
  std::bitset<19> bitsRead19;
  std::bitset<20> bitsRead20;

  switch (requiredBits)
    {
    case 1:
      bIterator = DeserializeBitset<1> (&bitsRead1,bIterator);
      *n = (int)bitsRead1.to_ulong ();
      break;
    case 2:
      bIterator = DeserializeBitset<2> (&bitsRead2,bIterator);
      *n = (int)bitsRead2.to_ulong ();
      break;
    case 3:
      bIterator = DeserializeBitset<3> (&bitsRead3,bIterator);
      *n = (int)bitsRead3.to_ulong ();
      break;
    case 4:
      bIterator = DeserializeBitset<4> (&bitsRead4,bIterator);
      *n = (int)bitsRead4.to_ulong ();
      break;
    case 5:
      bIterator = DeserializeBitset<5> (&bitsRead5,bIterator);
      *n = (int)bitsRead5.to_ulong ();
      break;
    case 6:
      bIterator = DeserializeBitset<6> (&bitsRead6,bIterator);
      *n = (int)bitsRead6.to_ulong ();
      break;
    case 7:
      bIterator = DeserializeBitset<7> (&bitsRead7,bIterator);
      *n = (int)bitsRead7.to_ulong ();
      break;
    case 8:
      bIterator = DeserializeBitset<8> (&bitsRead8,bIterator);
      *n = (int)bitsRead8.to_ulong ();
      break;
    case 9:
      bIterator = DeserializeBitset<9> (&bitsRead9,bIterator);
      *n = (int)bitsRead9.to_ulong ();
      break;
    case 10:
      bIterator = DeserializeBitset<10> (&bitsRead10,bIterator);
      *n = (int)bitsRead10.to_ulong ();
      break;
    case 11:
      bIterator = DeserializeBitset<11> (&bitsRead11,bIterator);
      *n = (int)bitsRead11.to_ulong ();
      break;
    case 12:
      bIterator = DeserializeBitset<12> (&bitsRead12,bIterator);
      *n = (int)bitsRead12.to_ulong ();
      break;
    case 13:
      bIterator = DeserializeBitset<13> (&bitsRead13,bIterator);
      *n = (int)bitsRead13.to_ulong ();
      break;
    case 14:
      bIterator = DeserializeBitset<14> (&bitsRead14,bIterator);
      *n = (int)bitsRead14.to_ulong ();
      break;
    case 15:
      bIterator = DeserializeBitset<15> (&bitsRead15,bIterator);
      *n = (int)bitsRead15.to_ulong ();
      break;
    case 16:
      bIterator = DeserializeBitset<16> (&bitsRead16,bIterator);
      *n = (int)bitsRead16.to_ulong ();
      break;
    case 17:
      bIterator = DeserializeBitset<17> (&bitsRead17,bIterator);
      *n = (int)bitsRead17.to_ulong ();
      break;
    case 18:
      bIterator = DeserializeBitset<18> (&bitsRead18,bIterator);
      *n = (int)bitsRead18.to_ulong ();
      break;
    case 19:
      bIterator = DeserializeBitset<19> (&bitsRead19,bIterator);
      *n = (int)bitsRead19.to_ulong ();
      break;
    case 20:
      bIterator = DeserializeBitset<20> (&bitsRead20,bIterator);
      *n = (int)bitsRead20.to_ulong ();
      break;
    default:
      {
        std::cout << "SerializeInteger Out of range!!" << std::endl;
        exit (1);
      }
    }

  *n += nmin;

  return bIterator;
}

Buffer::Iterator Asn1Header::DeserializeChoice (int numOptions, bool isExtensionMarkerPresent, int *selectedOption, Buffer::Iterator bIterator)
{
  if (isExtensionMarkerPresent)
  {
    bool marker;
    bIterator = DeserializeBoolean (&marker,bIterator);
  }
  return DeserializeInteger (selectedOption,0,numOptions - 1,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeEnum (int numElems, int *selectedElem, Buffer::Iterator bIterator)
{
  return DeserializeInteger (selectedElem,0,numElems - 1,bIterator);
}

template <int N>
Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<N> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  if (isExtensionMarkerPresent)
    {
      bool dummy;
      bIterator = DeserializeBoolean (&dummy,bIterator);
    }
  bIterator = DeserializeBitset<N> (optionalOrDefaultMask,bIterator);
  return bIterator;
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<0> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<0> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<1> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<1> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<2> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<2> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<3> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<3> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<4> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<4> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<5> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<5> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<6> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<6> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<7> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<7> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<9> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<9> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<10> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<10> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeSequence (std::bitset<11> *optionalOrDefaultMask, bool isExtensionMarkerPresent, Buffer::Iterator bIterator)
{
  return DeserializeSequence<11> (optionalOrDefaultMask,isExtensionMarkerPresent,bIterator);
}

Buffer::Iterator Asn1Header::DeserializeNull (Buffer::Iterator bIterator)
{
  return bIterator;
}

Buffer::Iterator Asn1Header::DeserializeSequenceOf (int *numElems, int nMax, int nMin, Buffer::Iterator bIterator)
{
  return DeserializeInteger (numElems,nMin,nMax,bIterator);
}

} // namespace ns3
