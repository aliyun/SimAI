/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 INRIA
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
 */

#include "int64x64.h"
#include <stdint.h>
#include <iostream>
#include <iomanip>  // showpos
#include <sstream>
#include "assert.h"
#include "log.h"

/**
 * \file
 * \ingroup highprec
 * Implementation of the streaming input and output operators for
 * the ns3::int64x64_t type.
 */

namespace ns3 {

// Note:  Logging in this file is largely avoided due to the
// number of calls that are made to these functions and the possibility
// of causing recursions leading to stack overflow
NS_LOG_COMPONENT_DEFINE_MASK ("int64x64", LOG_PREFIX_TIME);

/**
 * \ingroup highprec
 * Print the high and low words of an int64x64 in hex, for debugging.
 *
 * \param [in] hi The high (integer) word.
 * \param [in] lo The low (fractional) work.
 */
#define HEXHILOW(hi, lo) \
  std::hex << std::setfill ('0') << std::right << " (0x"                \
           << std::setw (16) << hi << " "                               \
           << std::setw (16) << lo                                      \
           << std::dec << std::setfill (' ') << std::left << ")"


/**
 * \internal
 * This algorithm is exact to the precision requested, up to the full
 * 64 decimal digits required to exactly represent a 64-bit fraction.
 *
 * Proper rounding turns out to be surprisingly hard.
 * In `y.xxxx5|6`, where the `|` marks follows the last output digit,
 * rounding the `5|6` to `6|` is straightforward.  However,
 * rounding `y.xxx99|6` should result in `y.xx100|`.  Notice the
 * effect of rounding percolates to higher digits.
 * We accumulate the output digits in a string, then carry out
 * the rounding in the string directly.
 */
std::ostream &operator << (std::ostream &os, const int64x64_t &value)
{
  const bool negative = (value < 0);
  const int64x64_t absVal = (negative ? -value : value);

  int64_t hi = absVal.GetHigh ();

  // Save stream format flags
  std::size_t precision = static_cast<std::size_t> (os.precision ());
  std::ios_base::fmtflags ff = os.flags ();
  const bool floatfield = os.flags () & std::ios_base::floatfield;
  os << std::setw (1) << std::noshowpos;

  os << std::right << (negative ? "-" : "+");

  std::ostringstream oss;
  oss << hi << ".";  // collect the digits here so we can round properly


  int64x64_t low (0, absVal.GetLow ());
  std::size_t places = 0;    // Number of decimal places printed so far
  bool more = true;  // Should we print more digits?

  NS_LOG_LOGIC (std::endl
                << (floatfield ? " f" : "  ")
                << "[" << precision << "] " << hi << ". "
                << HEXHILOW (hi, low.GetLow ())
                );

  int64_t digit;
  do
    {
      low *= 10;
      digit = low.GetHigh ();
      NS_ASSERT_MSG ( (0 <= digit) && (digit <= 9),
                      "digit " << digit << " out of range [0,9] " <<
                      " streaming out " <<
                      HEXHILOW (value.GetHigh (), value.GetLow ()) );
      low -= digit;

      oss << std::setw (1) << digit;

      ++places;
      if (floatfield)
        {
          more = places < precision;
        }
      else  // default
        {
          // Full resolution is 20 decimal digits
          more = low.GetLow () && (places < 20);
        }

      NS_LOG_LOGIC ((more ? "+" : " ")
                    << (floatfield ? "f" : " ")
                    << "[" << places << "] " << digit
                    << HEXHILOW (low.GetHigh (), low.GetLow ())
                    << std::dec << std::setfill (' ' ) << std::left);

    }
  while (more);

  // Check if we need to round the last printed digit,
  // based on the first unprinted digit
  std::string digits = oss.str ();
  low *= 10;
  int64_t nextDigit = low.GetHigh ();
  if ( (nextDigit > 5) || ((nextDigit == 5) && (digit % 2 == 1)) )
    {
      // Walk backwards with the carry
      bool carry = true;
      for (std::string::reverse_iterator rit = digits.rbegin ();
           rit != digits.rend ();
           ++rit)
        {
          if (*rit == '.')  // Skip over the decimal point
            {
              continue;
            }

          ++(*rit);         // Add the carry
          if (*rit <= '9')  // Relies on character order...
            {
              carry = false;
              break;        // Carry complete
            }
          else
            {
              *rit = '0';     // Continue carry to next higher digit
            }
        }
      if (carry)            // If we still have a carry...
        {
          digits.insert (digits.begin (), '1');
        }
    }
  os << digits;

  os.flags (ff);  // Restore stream flags
  return os;
}

/**
 * \ingroup highprec
 * Read the integer portion of a number from a string containing
 * just the integral digits (no decimal point or fractional part).
 *
 * \param [in] str The string representation of the integral part
 *             of a number, with no fractional part or decimal point.
 * \returns    The integer.
 */
static uint64_t ReadHiDigits (std::string str)
{
  const char *buf = str.c_str ();
  uint64_t retval = 0;
  while (*buf != 0)
    {
      retval *= 10;
      retval += *buf - '0';
      buf++;
    }
  return retval;
}

/**
 * \ingroup highprec
 * Read the fractional part of a number from a string containing
 * just the decimal digits of the fractional part (no integral part
 * or decimal point).
 *
 * \param [in] str The string representation of the fractional part
 *             of a number, without integral part or decimal point.
 * \returns    The decimal portion of the input number.
 */
static uint64_t ReadLoDigits (std::string str)
{
  int64x64_t low;
  const int64x64_t round (0, 5);  // Round last place in division

  for (std::string::const_reverse_iterator rit = str.rbegin ();
       rit != str.rend ();
       ++rit)
    {
      int digit = *rit - '0';
      NS_ASSERT_MSG ( (0 <= digit) && (digit <= 9),
                      "digit " << digit << " out of range [0,9]" <<
                      " streaming in low digits \"" << str << "\"");
      low = (low + digit + round) / 10;
    }

  return low.GetLow ();
}

std::istream &operator >> (std::istream &is, int64x64_t &value)
{
  std::string str;

  is >> str;
  bool negative;
  // skip heading spaces
  std::string::size_type cur;
  cur = str.find_first_not_of (" ");
  std::string::size_type next;
  // first, remove the sign.
  next = str.find ("-", cur);
  if (next != std::string::npos)
    {
      negative = true;
      next++;
    }
  else
    {
      next = str.find ("+", cur);
      if (next != std::string::npos)
        {
          next++;
        }
      else
        {
          next = cur;
        }
      negative = false;
    }
  cur = next;
  int64_t hi;
  uint64_t lo;
  next = str.find (".", cur);
  if (next != std::string::npos)
    {
      hi = ReadHiDigits (str.substr (cur, next - cur));
      lo = ReadLoDigits (str.substr (next + 1, str.size () - (next + 1)));
    }
  else if (cur != std::string::npos)
    {
      hi = ReadHiDigits (str.substr (cur, str.size () - cur));
      lo = 0;
    }
  else
    {
      hi = 0;
      lo = 0;
    }

  value = int64x64_t (hi, lo);
  value = negative ? -value : value;

  return is;
}

} // namespace ns3
