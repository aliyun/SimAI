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

#ifndef INT64X64_H
#define INT64X64_H

#include "ns3/core-config.h"

// Order is important here, as it determines which implementation
// will generate doxygen API docs.  This order mimics the
// selection logic in CMakeLists.txt, so we generate docs from the
// implementation actually chosen by the configuration.
#if defined (INT64X64_USE_128) && !defined (PYTHON_SCAN)
#include "int64x64-128.h"
#elif defined (INT64X64_USE_CAIRO) && !defined (PYTHON_SCAN)
#include "int64x64-cairo.h"
#elif defined (INT64X64_USE_DOUBLE) || defined (PYTHON_SCAN)
#include "int64x64-double.h"
#endif

#include <iostream>

/**
 * \file
 * \ingroup highprec
 * Declaration of the ns3::int64x64_t type and associated operators.
 */

namespace ns3 {

/**
 * \ingroup core
 * \defgroup highprec High Precision Q64.64
 *
 * Functions and class for high precision Q64.64 fixed point arithmetic.
 *
 * A Q64.64 fixed precision number consists of:
 *
 *   Bits | Function
 *   ---- | --------
 *     1  | Sign bit
 *    63  | Integer portion
 *    64  | Fractional portion
 *
 * The `high` word consists of the sign bit and integer value;
 * the `low` word is the fractional part, unscaled.
 *
 * All standard arithmetic operations are supported:
 *
 *   Category    | Operators
 *   ----------- | ---------
 *   Computation | `+`, `+=`, `-`, `-=`, `*`, `*=`, `/`, `/=`
 *   Comparison  | `==`, `!=`, `<`, `<=`, `>`, `>=`
 *   Unary       | `+`, `-`, `!`
 */

/**
 * \ingroup highprec
 * \class int64x64_t
 *
 * High precision numerical type, implementing Q64.64 fixed precision.
 */


/**
 * \ingroup highprec
 * Addition operator.
 * \param [in] lhs Left hand argument
 * \param [in] rhs Right hand argument
 * \return The result of the operator.
 */
inline
int64x64_t operator + (const int64x64_t & lhs, const int64x64_t & rhs)
{
  int64x64_t tmp = lhs;
  tmp += rhs;
  return tmp;
}
/**
 * \ingroup highprec
 * Subtraction operator.
 * \param [in] lhs Left hand argument
 * \param [in] rhs Right hand argument
 * \return The result of the operator.
 */
inline
int64x64_t operator - (const int64x64_t & lhs, const int64x64_t & rhs)
{
  int64x64_t tmp = lhs;
  tmp -= rhs;
  return tmp;
}
/**
 * \ingroup highprec
 * Multiplication operator.
 * \param [in] lhs Left hand argument
 * \param [in] rhs Right hand argument
 * \return The result of the operator.
 */
inline
int64x64_t operator * (const int64x64_t & lhs, const int64x64_t & rhs)
{
  int64x64_t tmp = lhs;
  tmp *= rhs;
  return tmp;
}
/**
 * \ingroup highprec
 * Division operator.
 * \param [in] lhs Left hand argument
 * \param [in] rhs Right hand argument
 * \return The result of the operator.
 */
inline
int64x64_t operator / (const int64x64_t & lhs, const int64x64_t & rhs)
{
  int64x64_t tmp = lhs;
  tmp /= rhs;
  return tmp;
}
/**
 * \ingroup highprec
 * Inequality operator
 * \param [in] lhs Left hand argument
 * \param [in] rhs Right hand argument
 * \return The result of the operator.
 */
inline bool operator != (const int64x64_t & lhs, const int64x64_t & rhs)
{
  return !(lhs == rhs);
}
/**
 * \ingroup highprec
 * Less or equal operator.
 * \param [in] lhs Left hand argument
 * \param [in] rhs Right hand argument
 * \return The result of the operator.
 */
inline bool operator <= (const int64x64_t & lhs, const int64x64_t & rhs)
{
  return !(lhs > rhs);
}
/**
 * \ingroup highprec
 * Greater or equal operator.
 * \param [in] lhs Left hand argument
 * \param [in] rhs Right hand argument
 * \return The result of the operator.
 */
inline bool operator >= (const int64x64_t & lhs, const int64x64_t & rhs)
{
  return !(lhs < rhs);
}
/**
 * \ingroup highprec
 * Output streamer for int64x64_t.
 *
 * Values are printed with the following format flags
 * (independent of the stream flags):
 *   - `showpos`
 *   - `left`
 *
 * The stream `width` is ignored.  If `floatfield` is set,
 * `precision` decimal places are printed.  If `floatfield` is not set,
 * all digits of the fractional part are printed, up to the
 * representation limit of 20 digits; trailing zeros are omitted.
 *
 * \param [in,out] os The output stream.
 * \param [in] value The numerical value to print.
 * \returns The stream.
 */
std::ostream &operator << (std::ostream &os, const int64x64_t &value);
/**
 * \ingroup highprec
 * Input streamer for int64x64_t.
 *
 * \param [in,out] is The input stream.
 * \param [out] value The numerical value to set.
 * \returns The stream.
 */
std::istream &operator >> (std::istream &is, int64x64_t &value);

/**
 * \ingroup highprec
 * Absolute value.
 * \param [in] value The value to operate on.
 * \return The absolute value of \pname{value}.
 */
inline int64x64_t Abs (const int64x64_t &value)
{
  return (value < 0) ? -value : value;
}

/**
 * \ingroup highprec
 * Minimum.
 *
 * \param [in] a The first value.
 * \param [in] b The second value.
 * \return The smaller of the arguments.
 */
inline int64x64_t Min (const int64x64_t &a, const int64x64_t &b)
{
  return (a < b) ? a : b;
}
/**
 * \ingroup highprec
 * Maximum.
 *
 * \param [in] a The first value.
 * \param [in] b The second value.
 * \return The larger of the arguments.
 */
inline int64x64_t Max (const int64x64_t &a, const int64x64_t &b)
{
  return (a > b) ? a : b;
}

} // namespace ns3

#endif /* INT64X64_H */
