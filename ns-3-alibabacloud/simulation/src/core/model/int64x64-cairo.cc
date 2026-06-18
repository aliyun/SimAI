/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
#include "test.h"
#include "abort.h"
#include "assert.h"
#include "log.h"
#include <cmath>
#include <iostream>
#include "int64x64-cairo.h"

// Include directly to allow optimizations within this compilation unit.
extern "C" {
#include "cairo-wideint.c"
}

/**
 * \file
 * \ingroup highprec
 * Implementation of the ns3::int64x64_t type using the Cairo implementation.
 */

namespace ns3 {

// Note:  Logging in this file is largely avoided due to the
// number of calls that are made to these functions and the possibility
// of causing recursions leading to stack overflow
NS_LOG_COMPONENT_DEFINE ("int64x64-cairo");

/**
 * \ingroup highprec
 * Compute the sign of the result of multiplying or dividing
 * Q64.64 fixed precision operands.
 *
 * \param [in]  sa The signed value of the first operand.
 * \param [in]  sb The signed value of the second operand.
 * \param [out] ua The unsigned magnitude of the first operand.
 * \param [out] ub The unsigned magnitude of the second operand.
 * \returns True if the result will be negative.
 */
static inline
bool
output_sign (const cairo_int128_t sa,
             const cairo_int128_t sb,
             cairo_uint128_t & ua,
             cairo_uint128_t & ub)
{
  bool negA = _cairo_int128_negative (sa);
  bool negB = _cairo_int128_negative (sb);
  ua = _cairo_int128_to_uint128 (sa);
  ub = _cairo_int128_to_uint128 (sb);
  ua = negA ? _cairo_uint128_negate (ua) : ua;
  ub = negB ? _cairo_uint128_negate (ub) : ub;
  return (negA && !negB) || (!negA && negB);
}

void
int64x64_t::Mul (const int64x64_t & o)
{
  cairo_uint128_t a, b;
  bool sign = output_sign (_v, o._v, a, b);
  cairo_uint128_t result = Umul (a, b);
  _v = sign ? _cairo_uint128_negate (result) : result;
}

cairo_uint128_t
int64x64_t::Umul (const cairo_uint128_t a, const cairo_uint128_t b)
{
  cairo_uint128_t result;
  cairo_uint128_t hiPart, loPart, midPart;
  cairo_uint128_t res1, res2;

  // Multiplying (a.h 2^64 + a.l) x (b.h 2^64 + b.l) =
  //			2^128 a.h b.h + 2^64*(a.h b.l+b.h a.l) + a.l b.l
  // get the low part a.l b.l
  // multiply the fractional part
  loPart = _cairo_uint64x64_128_mul (a.lo, b.lo);
  // compute the middle part 2^64*(a.h b.l+b.h a.l)
  midPart = _cairo_uint128_add (_cairo_uint64x64_128_mul (a.lo, b.hi),
                                _cairo_uint64x64_128_mul (a.hi, b.lo));
  // compute the high part 2^128 a.h b.h
  hiPart = _cairo_uint64x64_128_mul (a.hi, b.hi);
  // if the high part is not zero, put a warning
  NS_ABORT_MSG_IF (hiPart.hi != 0,
                   "High precision 128 bits multiplication error: multiplication overflow.");

  // Adding 64-bit terms to get 128-bit results, with carries
  res1 = _cairo_uint64_to_uint128 (loPart.hi);
  res2 = _cairo_uint64_to_uint128 (midPart.lo);
  result  = _cairo_uint128_add (res1, res2);

  res1 = _cairo_uint64_to_uint128 (midPart.hi);
  res2 = _cairo_uint64_to_uint128 (hiPart.lo);
  res1 = _cairo_uint128_add (res1, res2);
  res1 = _cairo_uint128_lsl (res1, 64);

  result  = _cairo_uint128_add (result, res1);

  return result;
}

void
int64x64_t::Div (const int64x64_t & o)
{
  cairo_uint128_t a, b;
  bool sign = output_sign (_v, o._v, a, b);
  cairo_uint128_t result = Udiv (a, b);
  _v = sign ? _cairo_uint128_negate (result) : result;
}

cairo_uint128_t
int64x64_t::Udiv (const cairo_uint128_t a, const cairo_uint128_t b)
{
  cairo_uint128_t den = b;
  cairo_uquorem128_t qr = _cairo_uint128_divrem (a, b);
  cairo_uint128_t result = qr.quo;
  cairo_uint128_t rem = qr.rem;

  // Now, manage the remainder
  const uint64_t DIGITS = 64;  // Number of fraction digits (bits) we need
  const cairo_uint128_t ZERO = _cairo_uint32_to_uint128 ((uint32_t)0);

  NS_ASSERT_MSG (_cairo_uint128_lt (rem, den),
                 "Remainder not less than divisor");

  uint64_t digis = 0;          // Number of digits we have already
  uint64_t shift = 0;          // Number we are going to get this round

  // Skip trailing zeros in divisor
  while ( (shift < DIGITS) && !(den.lo & 0x1))
    {
      ++shift;
      den = _cairo_uint128_rsl (den, 1);
    }

  while ( (digis < DIGITS) && !(_cairo_uint128_eq (rem, ZERO)) )
    {
      // Skip leading zeros in remainder
      while ( (digis + shift < DIGITS)
              && !(rem.hi & HPCAIRO_MASK_HI_BIT) )
        {
          ++shift;
          rem = _cairo_int128_lsl (rem, 1);
        }

      // Cast off denominator bits if:
      //   Need more digits and
      //     LSB is zero or
      //     rem < den
      while ( (digis + shift < DIGITS)
              && ( !(den.lo & 0x1) || _cairo_uint128_lt (rem, den) ) )
        {
          ++shift;
          den = _cairo_uint128_rsl (den, 1);
        }

      // Do the division
      qr = _cairo_uint128_divrem (rem, den);

      // Add in the quotient as shift bits of the fraction
      result = _cairo_uint128_lsl (result, static_cast<int> (shift));
      result = _cairo_uint128_add (result, qr.quo);
      rem = qr.rem;
      digis += shift;
      shift = 0;
    }
  // Did we run out of remainder?
  if (digis < DIGITS)
    {
      shift = DIGITS - digis;
      result = _cairo_uint128_lsl (result, static_cast<int> (shift));
    }

  return result;
}

void
int64x64_t::MulByInvert (const int64x64_t & o)
{
  bool sign = _cairo_int128_negative (_v);
  cairo_uint128_t a = sign ? _cairo_int128_negate (_v) : _v;
  cairo_uint128_t result = UmulByInvert (a, o._v);

  _v = sign ? _cairo_int128_negate (result) : result;
}

cairo_uint128_t
int64x64_t::UmulByInvert (const cairo_uint128_t a, const cairo_uint128_t b)
{
  cairo_uint128_t result;
  cairo_uint128_t hi, mid;
  hi = _cairo_uint64x64_128_mul (a.hi, b.hi);
  mid = _cairo_uint128_add (_cairo_uint64x64_128_mul (a.hi, b.lo),
                            _cairo_uint64x64_128_mul (a.lo, b.hi));
  mid.lo = mid.hi;
  mid.hi = 0;
  result = _cairo_uint128_add (hi,mid);
  return result;
}

int64x64_t
int64x64_t::Invert (const uint64_t v)
{
  NS_ASSERT (v > 1);
  cairo_uint128_t a, factor;
  a.hi = 1;
  a.lo = 0;
  factor.hi = 0;
  factor.lo = v;
  int64x64_t result;
  result._v = Udiv (a, factor);
  int64x64_t tmp = int64x64_t (v, 0);
  tmp.MulByInvert (result);
  if (tmp.GetHigh () != 1)
    {
      cairo_uint128_t one = { 1, 0};
      result._v = _cairo_uint128_add (result._v, one);
    }
  return result;
}


} // namespace ns3
