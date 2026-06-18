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
#include "bit-serializer.h"
#include "ns3/log.h"
#include "ns3/assert.h"
#include "ns3/abort.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BitSerializer");

BitSerializer::BitSerializer ()
{
  NS_LOG_FUNCTION (this);
  m_padAtEnd = true;
}

void BitSerializer::InsertPaddingAtEnd (bool padAtEnd)
{
  NS_LOG_FUNCTION (this);
  m_padAtEnd = padAtEnd;
}

void BitSerializer::PadAtStart ()
{
  NS_LOG_FUNCTION (this);

  uint8_t padding = 8 - (m_blob.size () % 8);

  m_blob.insert (m_blob.begin (), padding, false);
}

void BitSerializer::PadAtEnd ()
{
  uint8_t padding = 8 - (m_blob.size () % 8);

  m_blob.insert (m_blob.end (), padding, false);
}

void BitSerializer::PushBits (uint64_t value, uint8_t significantBits)
{
  NS_LOG_FUNCTION (this << value << +significantBits);

  uint64_t mask = 1;
  mask <<= significantBits - 1;

  for (uint8_t i = 0; i < significantBits; i++)
    {
      if (value & mask)
        {
          m_blob.push_back (true);
        }
      else
        {
          m_blob.push_back (false);
        }
      mask >>= 1;
    }
}

std::vector<uint8_t> BitSerializer::GetBytes ()
{
  NS_LOG_FUNCTION (this);

  std::vector<uint8_t> result;

  m_padAtEnd ? PadAtEnd () : PadAtStart ();

  for (auto it = m_blob.begin (); it != m_blob.end ();)
    {
      uint8_t tmp = 0;
      for (uint8_t i = 0; i < 8; ++i)
        {
          tmp <<= 1;
          tmp |= (*it & 1);
          it++;
        }
      result.push_back (tmp);
    }
  m_blob.clear ();
  return result;
}

uint8_t BitSerializer::GetBytes (uint8_t *buffer, uint32_t size)
{
  NS_LOG_FUNCTION (this << buffer << size);

  uint8_t resultLen = 0;

  m_padAtEnd ? PadAtEnd () : PadAtStart ();

  NS_ABORT_MSG_IF (m_blob.size () <= 8 * size, "Target buffer is too short, " << m_blob.size () / 8 << " bytes needed" );

  for (auto it = m_blob.begin (); it != m_blob.end ();)
    {
      uint8_t tmp = 0;
      for (uint8_t i = 0; i < 8; ++i)
        {
          tmp <<= 1;
          tmp |= (*it & 1);
          it++;
        }
      buffer[resultLen] = tmp;
      resultLen++;

    }
  m_blob.clear ();
  return resultLen;
}

} // namespace ns3
