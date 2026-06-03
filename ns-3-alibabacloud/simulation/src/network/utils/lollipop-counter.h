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


#ifndef LOLLIPOP_COUNTER_H
#define LOLLIPOP_COUNTER_H

#include <limits>
#include "ns3/abort.h"

namespace ns3 {

/**
 * \ingroup seq-counters
 * \class LollipopCounter
 *
 * \brief Template class implementing a Lollipop counter as defined in \RFC{8505}, \RFC{6550}, and [Perlman83].
 *
 * A Lollipop counter is a counter that solves initialization and
 * out-of-order problems often occurring in Internet protocols.
 *
 * The counter is split in two regions, an initializing region, and a circular region, having the same size.
 * Assuming a counter using an uint8_t (max value 255), values from 128 and greater
 * are used as a linear sequence to indicate a restart and bootstrap the counter, and the values less
 * than or equal to 127 are used as a circular sequence number space of
 * size 128 as mentioned in \RFC{1982}.
 *
 * In both regions, the comparison between two counters is allowed only if both counters are inside a
 * Sequence Window. The default value for the Sequence Window is equal to 2^N where N is half the number
 * of digits of the underlying type. For an uint8_t the Sequence Window is 16.
 *
 * The counter, by default, is initialized to the maximum counter value minus the Sequence Window plus one, e.g.,
 * in case of a uint8_t, to 240.
 *
 * This implementation extends the case presented in RFCs, allowing to use a
 * larger underlying type and to change the Sequence Window size.
 *
 * Warning: two Lollipop counters can be compared only if they are of the same type (same underlying type, and same Sequence Window).
 *
 * References:
 * [Perlman83] Perlman, R., "Fault-Tolerant Broadcast of Routing Information", North-Holland Computer Networks 7: pp. 395-405, DOI 10.1016/0376-5075(83)90034-X, 1983, <http://www.cs.illinois.edu/~pbg/courses/cs598fa09/readings/p83.pdf>.
 *
 * \tparam T \explicit The type being used for the counter.
 */
template <class T>
class LollipopCounter
{
public:
  /**
   * Builds a Lollipop counter with a default initial value.
   *
   * The Sequence Window is set to the default value.
   * The initial value is set to the maximum counter value minus the Sequence Window plus one.
   */
  LollipopCounter ()
  {
    NS_ABORT_MSG_UNLESS (std::is_unsigned<T>::value, "Lollipop counters must be defined on unsigned integer types");

    uint16_t numberofDigits = std::numeric_limits<T>::digits;
    m_sequenceWindow = 1 << (numberofDigits / 2);

    m_value = (m_maxValue - m_sequenceWindow) + 1;
  }

  /**
   * Builds a Lollipop counter with a specific initial value.
   *
   * The Sequence Window is set to the default value.
   *
   * \param val the initial value of the Lollipop Counter
   * \tparam T \deduced The type being used for the counter.
   */
  LollipopCounter (T val)
  {
    uint16_t numberofDigits = std::numeric_limits<T>::digits;
    m_sequenceWindow = 1 << (numberofDigits / 2);

    m_value = val;
  }

  /**
   * Assignment.
   *
   * \param [in] o Value to assign to this LollipopCounter.
   * \returns This LollipopCounter.
   */
  inline LollipopCounter & operator = (const LollipopCounter & o)
  {
    m_value = o.m_value;
    return *this;
  }

  /**
   * Resets the counter to its initial value.
   */
  void Reset ()
  {
    m_value = (m_maxValue - m_sequenceWindow) + 1;
  }

  /**
   * Set the Sequence Window Size and resets the counter.
   *
   * The sequence window is equal to 2^numberOfBits.
   * The counter is reset to maxValue - m_sequenceWindow +1, where
   * maxValue is the maximum number allowed by the underlying type.
   *
   * \param numberOfBits number of bits to use in the Sequence Window
   */
  void SetSequenceWindowSize (uint16_t numberOfBits)
  {
    uint16_t numberofDigits = std::numeric_limits<T>::digits;

    NS_ABORT_MSG_IF (numberOfBits >= numberofDigits, "The size of the Sequence Window should be less than the counter size (which is " << +m_maxValue << ")");

    m_sequenceWindow = 1 << numberOfBits;

    m_value = (m_maxValue - m_sequenceWindow) + 1;
  }

  /**
   * Checks if the counter is comparable with another counter (i.e., not desynchronized).
   *
   * If the absolute magnitude of difference of the two
   * sequence counters is greater than Sequence Window, then a
   * desynchronization has occurred and the two sequence
   * numbers are not comparable.
   *
   * Sequence Window is equal to 2^N where N is (by default) half the number
   * of digits of the underlying type.
   *
   * \param val counter to compare
   * \returns true if the counters are comparable.
   */
  bool IsComparable (const LollipopCounter &val) const
  {
    NS_ABORT_MSG_IF (m_sequenceWindow != val.m_sequenceWindow, "Can not compare two Lollipop Counters with different sequence windows");

    if ( (m_value <= m_circularRegion && val.m_value <= m_circularRegion)
         || (m_value > m_circularRegion && val.m_value > m_circularRegion) )
      {
        // They are desynchronized - comparison is impossible.
        T absDiff = AbsoluteMagnitudeOfDifference (val);
        if (absDiff > m_sequenceWindow)
          {
            return false;
          }
      }
    return true;
  }

  /**
   * Checks if a counter is in its starting region.
   *
   * \returns true if a counter is in its starting region.
   */
  bool IsInit () const
  {
    if ( m_value > m_circularRegion )
      {
        return true;
      }
    return false;
  }

  /**
   * Arithmetic operator equal-to
   * \param [in] lhs Left hand argument
   * \param [in] rhs Right hand argument
   * \return The result of the operator.
   */
  friend bool operator == (const LollipopCounter & lhs, const LollipopCounter & rhs)
  {

    NS_ABORT_MSG_IF (lhs.m_sequenceWindow != rhs.m_sequenceWindow, "Can not compare two Lollipop Counters with different sequence windows");

    if (lhs.m_value == rhs.m_value)
      {
        return true;
      }
    return false;
  }

  /**
   * Arithmetic operator greater-than
   * \param [in] lhs Left hand argument
   * \param [in] rhs Right hand argument
   * \return The result of the operator.
   */
  friend bool operator > (const LollipopCounter & lhs, const LollipopCounter & rhs)
  {
    NS_ABORT_MSG_IF (lhs.m_sequenceWindow != rhs.m_sequenceWindow, "Can not compare two Lollipop Counters with different sequence windows");

    if (lhs.m_value == rhs.m_value)
      {
        return false;
      }

    if ((lhs.m_value <= m_circularRegion && rhs.m_value <= m_circularRegion)
        || (lhs.m_value > m_circularRegion && rhs.m_value > m_circularRegion) )
      {
        // both counters are in the same region

        T absDiff = lhs.AbsoluteMagnitudeOfDifference (rhs);
        if (absDiff > lhs.m_sequenceWindow)
          {
            // They are desynchronized - comparison is impossible.
            // return false because we can not return anything else.
            return false;
          }

        // They are synchronized - comparison according to RFC1982.
        T serialRegion = ((m_circularRegion >> 1) + 1);
        return (((lhs.m_value < rhs.m_value) && ((rhs.m_value - lhs.m_value) > serialRegion))
                || ((lhs.m_value > rhs.m_value) && ((lhs.m_value - rhs.m_value) < serialRegion)) );
      }

    // One counter is in the "high" region and the other is in the in the "lower" region
    bool lhsIsHigher;
    T difference;

    if (lhs.m_value > m_circularRegion && rhs.m_value <= m_circularRegion)
      {
        lhsIsHigher = true;
        // this is guaranteed to be positive and between [1...m_lollipopMaxValue].
        difference = lhs.m_value - rhs.m_value;
      }
    else
      {
        lhsIsHigher = false;
        // this is guaranteed to be positive and between [1...m_lollipopMaxValue].
        difference = rhs.m_value - lhs.m_value;
      }

    T distance = (m_maxValue - difference) + 1; // this is guaranteed to be positive and between [1...m_lollipopMaxValue].
    if (distance > lhs.m_sequenceWindow)
      {
        if (lhsIsHigher)
          {
            return true;
          }
        else
          {
            return false;
          }
      }
    else
      {
        if (lhsIsHigher)
          {
            return false;
          }
        else
          {
            return true;
          }
      }

    // this should never be reached.
    return false;
  }

  /**
   * Arithmetic operator less-than
   * \param [in] lhs Left hand argument
   * \param [in] rhs Right hand argument
   * \return The result of the operator.
   */
  friend bool operator < (const LollipopCounter & lhs, const LollipopCounter & rhs)
  {
    if (!lhs.IsComparable (rhs))
      {
        return false;
      }

    if (lhs > rhs)
      {
        return false;
      }
    else if (lhs == rhs)
      {
        return false;
      }

    return true;
  }

  /**
   * Prefix increment operator
   * \param [in] val LollipopCounter to be incremented
   * \return The result of the Prefix increment.
   */
  friend LollipopCounter operator++ (LollipopCounter& val)   // prefix ++
  {
    val.m_value++;

    if ( val.m_value == val.m_circularRegion + 1 )
      {
        val.m_value = 0;
      }

    return val;
  }

  /**
   * Postfix increment operator
   * \param [in] val LollipopCounter to be incremented
   * \param [in] noop ignored argument (used to mark it as a postfix, blame c++).
   * \return The result of the Postfix increment.
   */
  friend LollipopCounter operator++ (LollipopCounter& val, int noop)  // postfix ++
  {
    LollipopCounter ans = val;
    ++(val);  // or just call operator++()
    return ans;
  }

  /**
   * Get the counter value.
   *
   * \return the counter value.
   */
  T GetValue () const
  {
    return m_value;
  }

  /**
   * Output streamer for LollipopCounter.
   *
   * \param [in,out] os The output stream.
   * \param [in] counter The LollipopCounter to print.
   * \returns The stream.
   */
  friend std::ostream &
  operator<< (std::ostream & os, LollipopCounter const & counter)
  {
    os << +counter.m_value;
    return os;
  }

private:

  /**
   * Compute the Absolute Magnitude Of Difference between two counters.
   *
   * The Absolute Magnitude Of Difference is considered to
   * be on a circular region, and it is represented by
   * the smallest circular distance between two numbers.
   *
   *  Arithmetic operator.
   *  \param [in] val Counter to compute the difference against
   *  \return The result of the difference.
   */
  T AbsoluteMagnitudeOfDifference (LollipopCounter const & val) const
  {
    // useless because it is computed always on counters on their respective regions.
    // Left (commented) for debugging purposes in case there is a code change.
    // NS_ASSERT_MSG ((m_value <= m_circularRegion && val.m_value <= m_circularRegion) ||
    //                (m_value > m_circularRegion && val.m_value > m_circularRegion),
    //                "Absolute Magnitude Of Difference can be computed only on two values in the circular region " << +m_value << " - " << +val.m_value);

    T absDiffDirect = std::max (m_value, val.m_value) - std::min (m_value, val.m_value);
    T absDiffWrapped = (std::min (m_value, val.m_value) + m_circularRegion + 1) - std::max (m_value, val.m_value);
    T absDiff = std::min (absDiffDirect, absDiffWrapped);
    return absDiff;
  }

  T m_value;          //!< Value of the Lollipop Counter.
  T m_sequenceWindow; //!< Sequence window used for comparing two counters.
  static constexpr T m_maxValue = std::numeric_limits<T>::max ();  //!< Maximum value of the counter.
  static constexpr T m_circularRegion = m_maxValue >> 1;           //!< Circular region of the counter.
};

/**
 * \ingroup seq-counters
 * 8 bit Lollipop Counter.
 */
typedef LollipopCounter<uint8_t> LollipopCounter8;
/**
 * \ingroup seq-counters
 * 16 bit Lollipop Counter.
 */
typedef LollipopCounter<uint16_t> LollipopCounter16;

} /* namespace ns3 */

#endif /* LOLLIPOP_COUNTER_H */
