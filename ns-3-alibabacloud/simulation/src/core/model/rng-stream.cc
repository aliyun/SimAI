/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
//  Copyright (C) 2001  Pierre L'Ecuyer (lecuyer@iro.umontreal.ca)
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Modified for ns-3 by:
//   - Rajib Bhattacharjea<raj.b@gatech.edu>
//   - Mathieu Lacage <mathieu.lacage@gmail.com>
//

#include <cstdlib>
#include <iostream>
#include "rng-stream.h"
#include "fatal-error.h"
#include "log.h"

/**
 * \file
 * \ingroup rngimpl
 * ns3::RngStream and MRG32k3a implementations.
 */

namespace ns3 {

// Note:  Logging in this file is largely avoided due to the
// number of calls that are made to these functions and the possibility
// of causing recursions leading to stack overflow
NS_LOG_COMPONENT_DEFINE ("RngStream");

} // namespace ns3


/**
 * \ingroup rngimpl
 * @{
 */
/** Namespace for MRG32k3a implementation details. */
namespace MRG32k3a {

// *NS_CHECK_STYLE_OFF*

/** Type for 3x3 matrix of doubles. */
typedef double Matrix[3][3];

/** First component modulus, 2<sup>32</sup> - 209. */
const double m1   =       4294967087.0;

/** Second component modulus, 2<sup>32</sup> - 22853. */
const double m2   =       4294944443.0;

/** Normalization to obtain randoms on [0,1). */
const double norm =       1.0 / (m1 + 1.0);

/** First component multiplier of <i>n</i> - 2 value. */
const double a12  =       1403580.0;

/** First component multiplier of <i>n</i> - 3 value. */
const double a13n =       810728.0;

/** Second component multiplier of <i>n</i> - 1 value. */
const double a21  =       527612.0;

/** Second component multiplier of <i>n</i> - 3 value. */
const double a23n =       1370589.0;

/** Decomposition factor for computing a*s in less than 53 bits, 2<sup>17</sup> */
const double two17 =      131072.0;

/** IEEE-754 floating point precision, 2<sup>53</sup> */
const double two53 =      9007199254740992.0;

/** First component transition matrix. */
const Matrix A1p0 = {
  {       0.0,        1.0,       0.0 },
  {       0.0,        0.0,       1.0 },
  { -810728.0,  1403580.0,       0.0 }
};

/** Second component transition matrix. */
const Matrix A2p0 = {
  {        0.0,        1.0,       0.0 },
  {        0.0,        0.0,       1.0 },
  { -1370589.0,        0.0,  527612.0 }
};


//-------------------------------------------------------------------------
/**
 * Return (a*s + c) MOD m; a, s, c and m must be < 2^35
 *
 * This computes the result exactly, without exceeding the 53 bit
 * precision of doubles.
 *
 * \param [in] a First multiplicative argument.
 * \param [in] s Second multiplicative argument.
 * \param [in] c Additive argument.
 * \param [in] m Modulus.
 * \returns <tt>(a*s +c) MOD m</tt>
 */
double MultModM (double a, double s, double c, double m)
{
  double v;
  int32_t a1;

  v = a * s + c;

  if (v >= two53 || v <= -two53)
    {
      a1 = static_cast<int32_t> (a / two17);
      a -= a1 * two17;
      v  = a1 * s;
      a1 = static_cast<int32_t> (v / m);
      v -= a1 * m;
      v = v * two17 + a * s + c;
    }

  a1 = static_cast<int32_t> (v / m);
  /* in case v < 0)*/
  if ((v -= a1 * m) < 0.0)
    {
      return v += m;
    }
  else 
    {
      return v;
    }
}


//-------------------------------------------------------------------------
/**
 * Compute the vector v = A*s MOD m. Assume that -m < s[i] < m.
 * Works also when v = s.
 *
 * \param [in] A Matrix argument, 3x3.
 * \param [in] s Three component input vector.
 * \param [out] v Three component output vector.
 * \param [in] m Modulus.
 */
void MatVecModM (const Matrix A, const double s[3], double v[3],
                 double m)
{
  int i;
  double x[3];                 // Necessary if v = s

  for (i = 0; i < 3; ++i)
    {
      x[i] = MultModM (A[i][0], s[0], 0.0, m);
      x[i] = MultModM (A[i][1], s[1], x[i], m);
      x[i] = MultModM (A[i][2], s[2], x[i], m);
    }
  for (i = 0; i < 3; ++i)
    {
      v[i] = x[i];
    }
}


//-------------------------------------------------------------------------
/**
 * Compute the matrix C = A*B MOD m. Assume that -m < s[i] < m.
 * Note: works also if A = C or B = C or A = B = C.
 *
 * \param [in] A First matrix argument.
 * \param [in] B Second matrix argument.
 * \param [out] C Result matrix.
 * \param [in] m Modulus.
 */
void MatMatModM (const Matrix A, const Matrix B,
                 Matrix C, double m)
{
  int i, j;
  double V[3];
  Matrix W;

  for (i = 0; i < 3; ++i)
    {
      for (j = 0; j < 3; ++j)
        {
          V[j] = B[j][i];
        }
      MatVecModM (A, V, V, m);
      for (j = 0; j < 3; ++j)
        {
          W[j][i] = V[j];
        }
    }
  for (i = 0; i < 3; ++i)
    {
      for (j = 0; j < 3; ++j)
        {
          C[i][j] = W[i][j];
        }
    }
}


//-------------------------------------------------------------------------
/**
 * Compute the matrix B = (A^(2^e) Mod m);  works also if A = B. 
 *
 * \param [in] src Matrix input argument \c A.
 * \param [out] dst Matrix output \c B.
 * \param [in] m Modulus.
 * \param [in] e The exponent.
 */
void MatTwoPowModM (const Matrix src, Matrix dst, double m, int32_t e)
{
  int i, j;

  /* initialize: dst = src */
  for (i = 0; i < 3; ++i)
    {
      for (j = 0; j < 3; ++j)
        {
          dst[i][j] = src[i][j];
        }
    }
  /* Compute dst = src^(2^e) mod m */
  for (i = 0; i < e; i++)
    {
      MatMatModM (dst, dst, dst, m);
    }
}


//-------------------------------------------------------------------------
/**
 * Compute the matrix B = (A^n Mod m);  works even if A = B.
 *
 * \param [in] A Matrix input argument.
 * \param [out] B Matrix output.
 * \param [in] m Modulus.
 * \param [in] n Exponent.
 */
void MatPowModM (const double A[3][3], double B[3][3], double m, int32_t n)
{
  int i, j;
  double W[3][3];

  // initialize: W = A; B = I
  for (i = 0; i < 3; ++i)
    {
      for (j = 0; j < 3; ++j)
        {
          W[i][j] = A[i][j];
          B[i][j] = 0.0;
        }
    }
  for (j = 0; j < 3; ++j)
    {
      B[j][j] = 1.0;
    }

  // Compute B = A^n mod m using the binary decomposition of n
  while (n > 0)
    {
      if (n % 2)
        {
          MatMatModM (W, B, B, m);
        }
      MatMatModM (W, W, W, m);
      n /= 2;
    }
}

/**
 * The transition matrices of the two MRG components
 * (in matrix form), raised to all powers of 2 from 1 to 191
 */
struct Precalculated
{
  Matrix a1[190];  //!< First component transition matrix powers.
  Matrix a2[190];  //!< Second component transition matrix powers.
};

/**
 * Compute the transition matrices of the two MRG components
 * raised to all powers of 2 from 1 to 191.
 *
 * \returns The precalculated powers of the transition matrices.
 */
struct Precalculated PowerOfTwoConstants (void)
{
  struct Precalculated precalculated;
  for (int i = 0; i < 190; i++)
    {
      int power = i + 1;
      MatTwoPowModM (A1p0, precalculated.a1[i], m1, power);
      MatTwoPowModM (A2p0, precalculated.a2[i], m2, power);
    }
  return precalculated;
}
/**
 * Get the transition matrices raised to a power of 2.
 *
 * \param [in] n The power of 2.
 * \param [out] a1p The first transition matrix power.
 * \param [out] a2p The second transition matrix power.
 */
void PowerOfTwoMatrix (int n, Matrix a1p, Matrix a2p)
{
  static struct Precalculated constants = PowerOfTwoConstants ();
  for (int i = 0; i < 3; i ++)
    {
      for (int j = 0; j < 3; j++)
        {
          a1p[i][j] = constants.a1[n-1][i][j];
          a2p[i][j] = constants.a2[n-1][i][j];
        }
    }
}

} // namespace MRG32k3a

// *NS_CHECK_STYLE_ON*


namespace ns3 {

using namespace MRG32k3a;

double RngStream::RandU01 ()
{
  int32_t k;
  double p1, p2, u;

  /* Component 1 */
  p1 = a12 * m_currentState[1] - a13n * m_currentState[0];
  k = static_cast<int32_t> (p1 / m1);
  p1 -= k * m1;
  if (p1 < 0.0)
    {
      p1 += m1;
    }
  m_currentState[0] = m_currentState[1];
  m_currentState[1] = m_currentState[2];
  m_currentState[2] = p1;

  /* Component 2 */
  p2 = a21 * m_currentState[5] - a23n * m_currentState[3];
  k = static_cast<int32_t> (p2 / m2);
  p2 -= k * m2;
  if (p2 < 0.0)
    {
      p2 += m2;
    }
  m_currentState[3] = m_currentState[4];
  m_currentState[4] = m_currentState[5];
  m_currentState[5] = p2;

  /* Combination */
  u = ((p1 > p2) ? (p1 - p2) * norm : (p1 - p2 + m1) * norm);

  return u;
}

RngStream::RngStream (uint32_t seedNumber, uint64_t stream, uint64_t substream)
{
  if (seedNumber >= m1 || seedNumber >= m2 || seedNumber == 0)
    {
      NS_FATAL_ERROR ("invalid Seed " << seedNumber);
    }
  for (int i = 0; i < 6; ++i)
    {
      m_currentState[i] = seedNumber;
    }
  AdvanceNthBy (stream, 127, m_currentState);
  AdvanceNthBy (substream, 76, m_currentState);
}

RngStream::RngStream (const RngStream& r)
{
  for (int i = 0; i < 6; ++i)
    {
      m_currentState[i] = r.m_currentState[i];
    }
}

void
RngStream::AdvanceNthBy (uint64_t nth, int by, double state[6])
{
  Matrix matrix1,matrix2;
  for (int i = 0; i < 64; i++)
    {
      int nbit = 63 - i;
      int bit = (nth >> nbit) & 0x1;
      if (bit)
        {
          PowerOfTwoMatrix (by + nbit, matrix1, matrix2);
          MatVecModM (matrix1, state, state, m1);
          MatVecModM (matrix2, &state[3], &state[3], m2);
        }
    }
}

} // namespace ns3

/**@}*/  // \ingroup rngimpl
