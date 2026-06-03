/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdint.h>

namespace ns3 {

/**
 * \brief Convert a byte buffer to a string containing a hex representation
 * of the buffer.  Make the string pretty by adding a colon (':') between
 * the hex.
 *
 * \param buffer The input buffer to be converted.
 * \param len The length of the input buffer.
 * \returns A string containing a hex representation of the data in buffer.
 */
std::string
TapBufferToString (uint8_t *buffer, uint32_t len)
{
  std::ostringstream oss;
  //
  // Tell the stream to make hex characters, zero-filled
  //
  oss.setf (std::ios::hex, std::ios::basefield);
  oss.fill ('0');

  //
  // Loop through the buffer, separating the two-digit-wide hex bytes
  // with a colon.
  //
  for (uint8_t i = 0; i < len; i++)
    {
      oss << ":" << std::setw (2) << (uint32_t)buffer[i];
    }
  return oss.str ();
}

/**
 * \brief Convert string encoded by the inverse function (TapBufferToString)
 * back into a byte buffer.
 *
 * \param s The input string.
 * \param buffer The buffer to initialize with the converted bits.
 * \param len The length of the data that is valid in the buffer.
 * \returns True indicates a successful conversion.
 */
bool
TapStringToBuffer (std::string s, uint8_t *buffer, uint32_t *len)
{
  //
  // If the string was made by our inverse function, the string length must
  // be a multiple of three characters in length.  Use this fact to do a
  // quick reasonableness test.
  //
  if ((s.length () % 3) != 0)
    {
      return false;
    }

  std::istringstream iss;
  iss.str (s);

  uint8_t n = 0;

  while (iss.good ())
    {
      //
      // The first character in the "triplet" we're working on is always the
      // the ':' separator.  Read that into a char and make sure we're skipping
      // what we think we're skipping.
      //
      char c;
      iss.read (&c, 1);
      if (c != ':')
        {
          return false;
        }

      //
      // And then read in the real bits and convert them.
      //
      uint32_t tmp;
      iss >> std::hex >> tmp;
      buffer[n] = tmp;
      n++;
    }

  *len = n;
  return true;
}

} // namespace ns3
