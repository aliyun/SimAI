/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2008-2010 INESC Porto
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
// Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt> <gjcarneiro@gmail.com>
//

#ifndef NS3_SEQ_NUM_H
#define NS3_SEQ_NUM_H

#include <limits>
#include <iostream>
#include <stdint.h>

namespace ns3 {

/**
 * \ingroup network
 * \defgroup seq-counters Sequence Counter
 * \brief "sequence number" classes
 */

/**
 * \ingroup seq-counters
 * \brief Generic "sequence number" class
 *
 * This class can be used to handle sequence numbers.  In networking
 * protocols, sequence numbers are fixed precision integer numbers
 * that are used to order events relative to each other.  A sequence
 * number is expected to increase over time but, since it has a
 * limited number of bits, the number will "wrap around" from the
 * maximum value that can represented with the given number of bits
 * back to zero.  For this reason, comparison of two sequence numbers,
 * and subtraction, is non-trivial.  The SequenceNumber class behaves
 * like a number, with the usual arithmetic operators implemented, but
 * knows how to correctly compare and subtract sequence numbers.
 *
 * This is a templated class.  To use it you need to supply two
 * fundamental types as template parameters: NUMERIC_TYPE and
 * SIGNED_TYPE.  For instance, SequenceNumber<uint32_t, int32_t> gives
 * you a 32-bit sequence number, while SequenceNumber<uint16_t,
 * int16_t> is a 16-bit one.  For your convenience, these are
 * typedef'ed as SequenceNumber32 and SequenceNumber16, respectively.
 */
template<typename NUMERIC_TYPE, typename SIGNED_TYPE>
class SequenceNumber
{
public:
  SequenceNumber ()
    : m_value (0)
  {}

  /**
   * \brief Constructs a SequenceNumber with the given value
   * \param value the sequence number value
   */ 
  explicit SequenceNumber (NUMERIC_TYPE value)
    : m_value (value)
  {}

  /**
   * \brief Constructs a SequenceNumber from a copy
   * \param value sequence number to copy
   */
  SequenceNumber (SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> const &value)
    : m_value (value.m_value)
  {}

  /**
   * \brief Constructs a SequenceNumber from an assignment of given value
   * \param value sequence number to copy
   * \returns reference to the assignee
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& operator= (NUMERIC_TYPE value)
  {
    m_value = value;
    return *this;
  }

  /**
   * \brief Constructs a SequenceNumber from an assignment of another sequence number
   * \param value sequence number to copy
   * \returns reference to the assignee
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& operator= (SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> const &value)
  {
    m_value = value.m_value;
    return *this;
  }

#if 0
  // a SequenceNumber implicitly converts to a plain number, but not the other way around
  operator NUMERIC_TYPE () const
  {
    return m_value;
  }
#endif

  /**
   * \brief Extracts the numeric value of the sequence number
   * \returns the sequence number value
   */ 
  NUMERIC_TYPE GetValue () const
  {
    return m_value;
  }

  /**
   * \brief Prefix increment operator
   * \returns incremented sequence number
   */ 
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator++ ()
  {
    m_value++;
    return *this;
  }

  /**
   * \brief Postfix increment operator
   * \returns incremented sequence number
   */ 
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator++ (int)
  {
    SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> retval (m_value);
    m_value++;
    return retval;
  }

  /**
   * \brief Prefix decrement operator
   * \returns decremented sequence number
   */ 
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator-- ()
  {
    m_value--;
    return *this;
  }

   /**
   * \brief Postfix decrement operator
   * \returns decremented sequence number
   */ 
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator-- (int)
  {
    SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> retval (m_value);
    m_value--;
    return retval;
  }

  /**
   * \brief Plus equals operator
   * \param value value to add to sequence number
   * \returns incremented sequence number
   */ 
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& operator+= (SIGNED_TYPE value)
  {
    m_value += value;
    return *this;
  }
  
  /**
   * \brief Minus equals operator
   * \param value value to subtract from sequence number
   * \returns decremented sequence number
   */ 
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& operator-= (SIGNED_TYPE value)
  {
    m_value -= value;
    return *this;
  }

  /**
   * \brief Operator defining addition of two sequence numbers
   * \param other sequence number added to this
   * \returns sequence number representing sum
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator + (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    return SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> (m_value + other.m_value);
  }

  /**
   * \brief Addition operator for adding numeric value to sequence number
   * \param delta value to add to sequence number
   * \returns sequence number representing sum
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator + (SIGNED_TYPE delta) const
  {
    return SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> (m_value + delta);
  }

  /**
   * \brief Subtraction operator for subtracting numeric value from sequence number
   * \param delta value to subtract from sequence number
   * \returns sequence number representing difference
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator - (SIGNED_TYPE delta) const
  {
    return SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> (m_value - delta);
  }

  /**
   * \brief Subtraction operator for subtracting sequence number from sequence number
   * \param other sequence number to subtract from this sequence number
   * \returns numeric value representing the difference
   */
  SIGNED_TYPE operator - (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    static const NUMERIC_TYPE maxValue = std::numeric_limits<NUMERIC_TYPE>::max ();
    static const NUMERIC_TYPE halfMaxValue = std::numeric_limits<NUMERIC_TYPE>::max () / 2;
    if (m_value > other.m_value)
      {
        NUMERIC_TYPE diff = m_value - other.m_value;
        if (diff < halfMaxValue)
          {
            return static_cast<SIGNED_TYPE> (diff);
          }
        else
          {
            //      |------------|------------|
            //       ====                  ===
            //          ^                  ^
            //       other.m_value      m_value
            return -(static_cast<SIGNED_TYPE> (maxValue - m_value + 1 + other.m_value));
          }
      }
    else
      {
        NUMERIC_TYPE diff = other.m_value - m_value;
        if (diff < halfMaxValue)
          {
            //      |------------|------------|
            //          ========
            //          ^      ^
            //     m_value   other.m_value
            return -(static_cast<SIGNED_TYPE> (diff));
          }
        else
          {
            //      |------------|------------|
            //       ====                  ===
            //          ^                  ^
            //       m_value      other.m_value
            return static_cast<SIGNED_TYPE> (maxValue - other.m_value + 1 + m_value);
          }
      }
  }

  /**
   * Here is the critical part, how the comparison is made taking into
   * account wrap-around.  From RFC 3626:
   *
   * The sequence number S1 is said to be "greater than" the sequence
   * number S2 if:
   *      S1 > S2 AND S1 - S2 <= MAXVALUE/2 OR
   *      S2 > S1 AND S2 - S1 > MAXVALUE/2
   *
   * \param other sequence number to compare to this one
   * \returns true if this sequence number is greater than other
   */
  bool operator > (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    static const NUMERIC_TYPE halfMaxValue = std::numeric_limits<NUMERIC_TYPE>::max () / 2;

    return (((m_value > other.m_value) && (m_value - other.m_value) <= halfMaxValue)
            || ((other.m_value > m_value) && (other.m_value - m_value) > halfMaxValue));
  }

  /**
   * \brief Equality operator for comparing sequence number
   * \param other sequence number to compare to this sequence number
   * \returns true if the sequence numbers are equal
   */
  bool operator == (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    return (m_value == other.m_value);
  }

  /**
   * \brief Inequality operator for comparing sequence numbers
   * \param other sequence number to compare to this sequence number
   * \returns true if the sequence numbers are not equal
   */
  bool operator != (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    return (m_value != other.m_value);
  }

  /**
   * \brief Less than or equal operator for comparing sequence numbers
   * \param other sequence number to compare to this sequence number
   * \returns true if this sequence number is less than or equal to other
   */
  bool operator <= (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    return (!this->operator> (other));
  }

  /**
   * \brief Greater than or equal operator for comparing sequence numbers
   * \param other sequence number to compare to this sequence number
   * \returns true if this sequence number is greater than or equal to other
   */
  bool operator >= (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    return (this->operator> (other) || this->operator== (other));
  }

  /**
   * \brief Less than operator for comparing sequence numbers
   * \param other sequence number to compare to this sequence number
   * \returns true if this sequence number is less than other
   */
  bool operator < (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &other) const
  {
    return !this->operator> (other) && m_value != other.m_value;
  }

  /**
   * \brief For printing sequence number
   * \param os output stream
   * \param val sequence number to display
   * \returns output stream os
   */
  template<typename NUMERIC_TYPE2, typename SIGNED_TYPE2>
  friend std::ostream & operator<< (std::ostream& os, const SequenceNumber<NUMERIC_TYPE2, SIGNED_TYPE2> &val);

  /**
   * \brief For loading sequence number from input streams
   * \param is input stream
   * \param val sequence number to load
   * \returns input stream is
   */
  template<typename NUMERIC_TYPE2, typename SIGNED_TYPE2>
  friend std::istream & operator >> (std::istream &is, const SequenceNumber<NUMERIC_TYPE2, SIGNED_TYPE2> &val);

private: // unimplemented operators
  /**
   * \brief Plus equals operator - unimplemented
   * \param value value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& operator+= (SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> const &value);
  /**
   * \brief Minus equals operator - unimplemented
   * \param value value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& operator-= (SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> const &value);
  /**
   * \brief Multiplication operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator* (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Division operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator/ (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Modulo operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator% (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Logical NOT operator - unimplemented
   * \returns condition
   */
  bool operator ! () const;
  /**
   * \brief Logical AND operator - unimplemented
   * \param b value
   * \returns condition
   */
  bool operator && (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Logical OR operator - unimplemented
   * \param b value
   * \returns condition
   */
  bool operator || (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Bitwise NOT operator - unimplemented
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator~ () const;
  /**
   * \brief Bitwise AND operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator& (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Bitwise OR operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator| (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Bitwise XOR operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator^ (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Bitwise left shift operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator<< (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Bitwise right shift operator - unimplemented
   * \param b value
   * \returns sequence number
   */
  SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> operator>> (const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>& b) const;
  /**
   * \brief Indirection operator - unimplemented
   * \returns integer
   */
  int operator* ();
  //SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE>* operator& ();

private:
  NUMERIC_TYPE m_value; //!< Sequence number value
};


/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param val the value
 * \returns a reference to the stream
 */
template<typename NUMERIC_TYPE, typename SIGNED_TYPE>
std::ostream &
operator<< (std::ostream& os, const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &val)
{
  os << val.m_value;
  return os;
}


/**
 * \brief Stream extraction operator.
 *
 * \param is the stream
 * \param val the value
 * \returns a reference to the stream
 */
template<typename NUMERIC_TYPE, typename SIGNED_TYPE>
std::istream & operator >> (std::istream &is, const SequenceNumber<NUMERIC_TYPE, SIGNED_TYPE> &val)
{
  is >> val.m_value;
  return is;
}

/**
 * \ingroup seq-counters
 * 32 bit Sequence number.
 */
typedef SequenceNumber<uint32_t, int32_t> SequenceNumber32;
/**
 * \ingroup seq-counters
 * 16 bit Sequence number.
 */
typedef SequenceNumber<uint16_t, int16_t> SequenceNumber16;
/**
 * \ingroup seq-counters
 * 8 bit Sequence number.
 */
typedef SequenceNumber<uint8_t, int8_t> SequenceNumber8;

namespace TracedValueCallback {

/**
 * \ingroup seq-counters
 * TracedValue callback signature for SequenceNumber32
 *
 * \param [in] oldValue original value of the traced variable
 * \param [in] newValue new value of the traced variable
 */
typedef void (* SequenceNumber32)(SequenceNumber32 oldValue,
                                  SequenceNumber32 newValue);

}  // namespace TracedValueCallback

} // namespace ns3

#endif /* NS3_SEQ_NUM_H */


