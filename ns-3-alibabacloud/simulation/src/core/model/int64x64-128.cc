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

#include "int64x64-128.h"
#include "abort.h"
#include "assert.h"
#include "log.h"

/**
 * \file
 * \ingroup highprec
 * Implementation of the ns3::int64x64_t type using a native int128_t type.
 */

namespace ns3 {

// Note:  Logging in this file is largely avoided due to the
// number of calls that are made to these functions and the possibility
// of causing recursions leading to stack overflow
NS_LOG_COMPONENT_DEFINE ("int64x64-128");

/**
 * \ingroup highprec
 * Compute the sign of the result of multiplying or dividing
 * Q64.64 fixed precision operands.
 *
 * \param [in]  sa The signed value of the first operand.
 * \param [in]  sb The signed value of the second operand.
 * \param [out] ua The unsigned magnitude of the first operand.
 * \param [out] ub The unsigned magnitude of the second operand.
 * \returns \c true if the result will be negative.
 */
static inline
bool
output_sign (const int128_t sa,
             const int128_t sb,
             uint128_t & ua,
             uint128_t & ub)
{
  bool negA = sa < 0;
  bool negB = sb < 0;
  ua = negA ? -sa : sa;
  ub = negB ? -sb : sb;
  return negA != negB;
}

void
int64x64_t::Mul (const int64x64_t & o)
{
  uint128_t a, b;
  bool negative = output_sign (_v, o._v, a, b);
  uint128_t result = Umul (a, b);
  _v = negative ? -result : result;
}

uint128_t
int64x64_t::Umul (const uint128_t a, const uint128_t b)
{
  uint128_t aL = a & HP_MASK_LO;
  uint128_t bL = b & HP_MASK_LO;
  uint128_t aH = (a >> 64) & HP_MASK_LO;
  uint128_t bH = (b >> 64) & HP_MASK_LO;

  uint128_t result;
  uint128_t hiPart, loPart, midPart;
  uint128_t res1, res2;

  // Multiplying (a.h 2^64 + a.l) x (b.h 2^64 + b.l) =
  //			2^128 a.h b.h + 2^64*(a.h b.l+b.h a.l) + a.l b.l
  // get the low part a.l b.l
  // multiply the fractional part
  loPart = aL * bL;
  // compute the middle part 2^64*(a.h b.l+b.h a.l)
  midPart = aL * bH + aH * bL;
  // compute the high part 2^128 a.h b.h
  hiPart = aH * bH;
  // if the high part is not zero, put a warning
  NS_ABORT_MSG_IF ((hiPart & HP_MASK_HI) != 0,
                   "High precision 128 bits multiplication error: multiplication overflow.");

  // Adding 64-bit terms to get 128-bit results, with carries
  res1 = loPart >> 64;
  res2 = midPart & HP_MASK_LO;
  result = res1 + res2;

  res1 = midPart >> 64;
  res2 = hiPart & HP_MASK_LO;
  res1 += res2;
  res1 <<= 64;

  result += res1;

  return result;
}

void
int64x64_t::Div (const int64x64_t & o)
{
  uint128_t a, b;
  bool negative = output_sign (_v, o._v, a, b);
  int128_t result = Udiv (a, b);
  _v = negative ? -result : result;
}

uint128_t
int64x64_t::Udiv (const uint128_t a, const uint128_t b)
{

  uint128_t rem = a;
  uint128_t den = b;
  uint128_t quo = rem / den;
  rem = rem % den;
  uint128_t result = quo;

  // Now, manage the remainder
  const uint64_t DIGITS = 64;  // Number of fraction digits (bits) we need
  const uint128_t ZERO = 0;

  NS_ASSERT_MSG (rem < den,
                 "Remainder not less than divisor");

  uint64_t digis = 0;          // Number of digits we have already
  uint64_t shift = 0;          // Number we are going to get this round

  // Skip trailing zeros in divisor
  while ( (shift < DIGITS) && !(den & 0x1))
    {
      ++shift;
      den >>= 1;
    }

  while ( (digis < DIGITS) && (rem != ZERO) )
    {
      // Skip leading zeros in remainder
      while ( (digis + shift < DIGITS)
              && !(rem & HP128_MASK_HI_BIT))
        {
          ++shift;
          rem <<= 1;
        }

      // Cast off denominator bits if:
      //   Need more digits and
      //     LSB is zero or
      //     rem < den
      while ( (digis + shift < DIGITS)
              && ( !(den & 0x1) || (rem < den) ) )
        {
          ++shift;
          den >>= 1;
        }

      // Do the division
      quo = rem / den;
      rem = rem % den;

      // Add in the quotient as shift bits of the fraction
      result <<= shift;
      result += quo;

      digis += shift;
      shift = 0;
    }
  // Did we run out of remainder?
  if (digis < DIGITS)
    {
      shift = DIGITS - digis;
      result <<= shift;
    }

  return result;
}

void
int64x64_t::MulByInvert (const int64x64_t & o)
{
  bool negResult = _v < 0;
  uint128_t a = negResult ? -_v : _v;
  uint128_t result = UmulByInvert (a, o._v);

  _v = negResult ? -result : result;
}

uint128_t
int64x64_t::UmulByInvert (const uint128_t a, const uint128_t b)
{
  uint128_t result, ah, bh, al, bl;
  uint128_t hi, mid;
  ah = a >> 64;
  bh = b >> 64;
  al = a & HP_MASK_LO;
  bl = b & HP_MASK_LO;
  hi = ah * bh;
  mid = ah * bl + al * bh;
  mid >>= 64;
  result = hi + mid;
  return result;
}

int64x64_t
int64x64_t::Invert (const uint64_t v)
{
  NS_ASSERT (v > 1);
  uint128_t a;
  a = 1;
  a <<= 64;
  int64x64_t result;
  result._v = Udiv (a, v);
  int64x64_t tmp = int64x64_t (v, false);
  tmp.MulByInvert (result);
  if (tmp.GetHigh () != 1)
    {
      result._v += 1;
    }
  return result;
}

} // namespace ns3

