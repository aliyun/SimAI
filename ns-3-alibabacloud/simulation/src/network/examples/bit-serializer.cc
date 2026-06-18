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

#include <iostream>
#include "ns3/bit-serializer.h"
#include "ns3/bit-deserializer.h"

using namespace ns3;

// The main purpose of the BitSerializer and bitDeserializer classes is to
// simplify the bit serialization and deserialization in headers, trailers
// and packet bodies.
//
// This is usually performed by using bit masks, which works great if the
// field delimiters are known in advance, and are in a fixed position.
// If the field (i.e., a group of bits) position is dependent from some
// other parameter, then the code is more complex.
// If the field boundary is not even a multiple of a byte, then the problem
// is even more complex.
//
// BitSerializer allows you to "push" bits into a temporary buffer, and then
// extract an array of uint8_t to be used in
// Buffer::Iterator:Write (uint8_t const *buffer, uint32_t size).
//
// Similarly, BitDeserializer can be initialized by an array of uint8_t,
// typically obtained by a Buffer::Iterator:Read (uint8_t *buffer, uint32_t size)
// and then "pop" bits from the underlying buffer.
//
// This example shows the basic operations.

int main ()
{

  BitSerializer testBitSerializer1;

  // add 7 bits - 0x55 (101 0101)
  testBitSerializer1.PushBits (0x55, 7);
  // add 3 bits - 0x7 (111)
  testBitSerializer1.PushBits (0x7, 3);
  // add 2 bits - 0x0 (00)
  testBitSerializer1.PushBits (0x0, 2);
  // The results is 1010 1011 1100.
  // Adding 4 bits of padding at the end the result is 0xabc0.

  std::vector<uint8_t> result = testBitSerializer1.GetBytes ();

  std::cout << "Result:    ";
  for (uint8_t i = 0; i < result.size (); i++)
    {
      std::cout << std::hex << int(result[i]) << " ";
    }
  std::cout << std::endl;
  std::cout << "Expecting: ab c0" << std::endl;

  // Here, instead of printing bits, you typically serialize them using
  // Buffer::Iterator:Write (uint8_t const *buffer, uint32_t size).
  //
  // In this case the number of bits pushed is not a multiple of a byte
  // so the class adds a padding at the end of the buffer.
  // This is the default behaviour.

  BitSerializer testBitSerializer2;

  // add 7 bits - 0x55 (101 0101)
  testBitSerializer2.PushBits (0x55, 7);
  // add 3 bits - 0x7 (111)
  testBitSerializer2.PushBits (0x7, 3);
  // add 2 bits - 0x0 (00)
  testBitSerializer2.PushBits (0x0, 2);

  // Change the class behaviour so to use a padding at the start of the buffer.
  testBitSerializer2.InsertPaddingAtEnd (false);
  // The results is 1010 1011 1100.
  // Adding 4 bits of padding at the start the result is 0xabc.

  result = testBitSerializer2.GetBytes ();

  std::cout << "Result:    ";
  for (uint8_t i = 0; i < result.size (); i++)
    {
      std::cout << std::hex << int(result[i]) << " ";
    }
  std::cout << std::endl;
  std::cout << "Expecting: a bc" << std::endl;

  // Here, instead of printing bits, you typically serialize them using
  // Buffer::Iterator:Write (uint8_t const *buffer, uint32_t size).
  //
  // In this case the number of bits pushed is not a multiple of a byte
  // so the class adds a padding at the start of the buffer.

  BitDeserializer testBitDeserializer;
  uint8_t test[2];
  test[0] = 0xab;
  test[1] = 0xc0;

  // Typically a BitDeserializer will be initialized by an array obtained by
  // Buffer::Iterator:Read (uint8_t *buffer, uint32_t size).

  testBitDeserializer.PushBytes (test, 2);
  uint16_t nibble1 = testBitDeserializer.GetBits (7);
  uint8_t nibble2 = testBitDeserializer.GetBits (3);
  uint8_t nibble3 = testBitDeserializer.GetBits (2);
//  if you deserialize too many bits you'll get an assert.
//  uint8_t errorNibble = testBitDeserializer.GetBits (6);

  std::cout << "Result:    " << std::hex << nibble1 << " " << +nibble2 << " " << +nibble3 << " " << std::endl;
  std::cout << "Expecting: 55 7 0" << std::endl;

  return 0;
}
