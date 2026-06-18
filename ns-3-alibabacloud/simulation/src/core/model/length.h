/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Lawrence Livermore National Laboratory
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
 * Author: Mathew Bielejeski <bielejeski1@llnl.gov>
 */

#ifndef NS3_LENGTH_H_
#define NS3_LENGTH_H_

#include "attribute.h"
#include "attribute-helper.h"

#ifdef HAVE_BOOST
#include <boost/units/quantity.hpp>
#include <boost/units/systems/si.hpp>
#endif

#include <istream>
#include <limits>
#include <optional>
#include <ostream>
#include <string>

/**
 * \file
 * \ingroup length
 * Declaration of ns3::Length class
 */

/**
 * ns3 namespace
 */
namespace ns3 {

/**
 * \ingroup core
 * \defgroup length Length
 *
 * Management of lengths in real world units.
 *
 */

/**
 * \ingroup length
 *
 * Represents a length in meters
 *
 * The purpose of this class is to replace the use of raw numbers (ints, doubles)
 * that have implicit lengths with a class that represents lengths with an explicit
 * unit.  Using raw values with implicit units can lead to bugs when a value is
 * assumed to represent one unit but actually represents another.  For example,
 * assuming a value represents a length in meters when it in fact represents a
 * length in kilometers.
 *
 * The Length class prevents this confusion by storing values internally as
 * doubles representing lengths in Meters and providing conversion functions
 * to convert to/from other length units (kilometers, miles, etc.).
 *
 * ## Supported Units ##
 *
 * Conversion to and from meters is supported for the following units:
 *
 * ### Metric ###
 * - nanometer
 * - micrometer
 * - millimeter
 * - centimeter
 * - meter
 * - kilometer
 * - nautical mile
 *
 * ### US Customary ###
 * - inch
 * - foot
 * - yard
 * - mile
 *
 * ## Construction ##
 *
 * Length objects can be constructed in a number of different ways
 *
 * ### String Constructor ###
 *
 * The string constructor parses strings with the format \<number\>\<unit\> or
 * \<number\> \<unit\> and creates the equivalent Length value in meters.
 * \<unit\> can be the full name of the unit (nanometer, kilometer, foot, etc.) or the
 * abbreviated name (nm, km, ft, etc.).
 *
 * \code
 *   //construct lengths from strings
 *   Length foot ("1foot");
 *   Length cm ("1cm");
 *   Length mile ("1 mile");
 *   Length km ("1 km");
 *
 *   //nautical mile is special because it is two words
 *   Length nautmile ("1 nautical mile");
 *   Length nmi ("1 nmi");
 * \endcode
 *
 * ### Quantity Constructor ###
 *
 * Quantity is a class that describes a value and a unit type. For example, meter
 * is a unit while 5 meters is a quantity.
 * The Length constructor takes the quantity instance and converts it to the
 * equivalent value in meters.
 *
 * \code
 *   //construct a Length representing 2 kilometers
 *   Length::Quantity q (2, Length::Unit::Kilometer);
 *   Length km(q);
 * \endcode
 *
 * ### Value/Unit Constructor
 *
 * Two constructors are provided which take a value and a unit as separate
 * parameters.  The difference between the two constructors is that one takes
 * the unit as a string and the other takes the unit as a Length::Unit value.
 * An assertion is triggered if the string does not map to a valid
 * Length::Unit
 *
 * \code
 *   //These two contructors are equivalent
 *   Length l1 (1, "cm");
 *   Length l2 (1, Length::Unit::Centimeter);
 * \endcode
 *
 * ### boost::units
 *
 * If the boost::units library is discovered during ns-3 configuration an
 * additional constructor is enabled which allows constructing Length objects
 * from boost::unit quantities.
 *
 * \code
 *   //construct length from a boost::units quantitiy
 *   boost::units::quantity<boost::units::si::length> q
 *       = 5 * boost::units::si::meter;
 *   Length meters (q);
 * \endcode
 *
 * ## Arithmetic Operations ##
 *
 * The following arithmetic operations are supported:
 *
 * ### Addition ###
 *
 *   Addition is between two Length instances
 *
 * \code
 *   std::cout << Length(1, Length::Unit::Meter) + Length (2, Length::Unit::Meter);
 *   // output: "3 m"
 * \endcode
 *
 * ### Subtraction ###
 *
 *   Subtraction is between two Length instances
 *
 * \code
 *   std::cout << Length(3, Length::Unit::Meter) - Length (2, Length::Unit::Meter);
 *   // output: "1 m"
 * \endcode
 *
 * ### Multiplication ###
 *
 *   Multiplication is only supported between a Length and a unitless scalar value
 *
 * \code
 *   std::cout << Length(1, Length::Unit::Meter) * 5;  // output: "5 m"
 *
 *   std::cout << 5 * Length (1, Length::Unit::Meter);  // output: "5 m"
 * \endcode
 *
 *
 * ### Division ###
 *
 *   Division can be between a Length and a scalar or two Length objects.
 *
 *   Division between a Length and a scalar returns a new Length.
 *
 *   Division between two Length objects returns a unitless value.
 *
 * \code
 *   std::cout << Length(5, Length::Unit::Meter) / 5;  // output: "1 m"
 *
 *   std::cout << Length (5, Length::Unit::Meter) / Length (5, Length::Unit::Meter);
 *   // output: 1
 * \endcode
 *
 * ## Comparison Operations ##
 *
 * All the usual arithmetic comparison operations (=, !=, <, <=, >, >=) are supported.
 * There are two forms of comparison operators: free function and member function.
 * The free functions define =, !=, <, <=, >, >= and perform exact comparisons of
 * the underlying double values.  The Length class provides comparison
 * operator functions (IsEqual, IsLess, IsGreater, etc.) which accept an
 * additional tolerance value to control how much the underlying double values
 * must match when performing the comparison.
 *
 * \code
 *   //check for exact match
 *   Length l (5, Length::Unit::Meter);
 *
 *   bool match = (l == l);  // match is true
 *
 *   Length v1 (0.02, Length::Unit::Meter);
 *   Length v2 (0.022, Length::Unit::Meter);
 *
 *   match = (v1 == v2);  // match is false
 *
 *   double tolerance = 0.01;
 *   bool mostly_match = v1.IsEqual (v2, tolerance);  // mostly_match is true
 * \endcode
 *
 * ## Serialization ##
 *
 * The Length class supports serialization using the << and >> operators.
 * By default the output serialization is in meters. Use Length::As to output
 * the Length value in a different unit.
 *
 * \code
 *   Length m(5, Length::Unit::Meter);
 *
 *   std::cout << m << ", "
 *             << m.As(Length::Unit::Kilometer) << ", "
 *             << m.As(Length::Unit::Foot);
 *   //output: 5 m, 0.005 km, 16.4042 ft
 * \endcode
 */
class Length
{
public:
  /**
   * Units of length in various measurement systems that are supported by the
   * Length class
   */
  enum Unit : uint16_t
  {
    //Metric Units
    Nanometer = 1,  //!< 1e<sup>-9</sup> meters
    Micrometer,     //!< 1e<sup>-6</sup> meters
    Millimeter,     //!< 1e<sup>-3</sup> meters
    Centimeter,     //!< 1e<sup>-2</sup> meters
    Meter,          //!< Base length unit in metric system
    Kilometer,      //!< 1e<sup>3</sup> meters
    NauticalMile,   //!< 1,852 meters

    //US Customary Units
    Inch,           //!< 1/12 of a foot
    Foot,           //!< Base length unit in US customary system
    Yard,           //!< 3 feet
    Mile            //!< 5,280 feet
  };

  /**
   * An immutable class which represents a value in a specific length unit
   */
  class Quantity
  {
public:
    /**
     * Constructor
     *
     * \param value Length value
     * \param unit Length unit of the value
     */
    Quantity (double value, Length::Unit unit)
      : m_value (value),
        m_unit (unit)
    { }

    /**
     * Copy Constructor
     */
    Quantity (const Quantity&) = default;

    /**
     * Move Constructor
     */
    Quantity (Quantity&&) = default;

    /**
     * Destructor
     */
    ~Quantity () = default;

    /**
     * Copy Assignment Operator
     * \param [in] other The source to copy from.
     * \returns this.
     */
    Quantity& operator= (const Quantity& other) = default;

    /**
     * Move Assignment Operator
     * \param [in] other The source to move from.
     * \returns this.
     */
    Quantity& operator= (Quantity&& other) = default;

    /**
     * The value of the quantity
     *
     * \return The value of this quantity
     */
    double Value () const
    {
      return m_value;
    }

    /**
     * The unit of the quantity
     *
     * \return The unit of this quantity
     */
    Length::Unit Unit () const
    {
      return m_unit;
    }

private:
    double m_value;         //!< Value of the length
    Length::Unit m_unit;    //!< unit of length of the value
  };

  /**
   * Default tolerance value used for the member comparison functions (IsEqual,
   * IsLess, etc.)
   *
   * The default tolerance is set to epsilon which is defined as the difference
   * between 1.0 and the next value that can be represented by a double.
   */
  static constexpr double DEFAULT_TOLERANCE = std::numeric_limits<double>::epsilon ();

  /**
   * Attempt to construct a Length object from a value and a unit string
   *
   * \p unit can either be the full name of the unit (meter, kilometer, mile, etc.)
   * or the symbol of the unit (m, km, mi, etc.)
   *
   * This function will return false if \p unit does not map to a known type.
   *
   * \param value Numeric value of the new length
   * \param unit Unit that the value represents
   *
   * \return A std::optional object containing the Length object constructed from
   * the given value and unit, if the attempt to construct the Length object was
   * successful.
   */
  static std::optional<Length> TryParse (double value, const std::string& unit);

  /**
   * Default Constructor
   *
   * Initialize with a value of 0 meters.
   */
  Length ();

  /**
   * String Constructor
   *
   * Parses \p text and initializes the value with the parsed result.
   *
   * The expected format of \p text is \<number\> \<unit\> or \<number\>\<unit\>
   *
   * \param text Serialized length value
   */
  Length (const std::string& text);

  /**
   * Construct a Length object from a value and a unit string
   *
   * \p unit can either be the full name of the unit (meter, kilometer, mile, etc.)
   * or the symbol of the unit (m, km, mi, etc.)
   *
   * \warning NS_FATAL_ERROR is called if \p unit is not a valid unit string.
   * \warning Use Length::TryParse to parse potentially bad values without terminating.
   *
   * \param value Numeric value of the new length
   * \param unit Unit that the value represents
   */
  Length (double value, const std::string& unit);

  /**
   * Construct a Length object from a value and a unit
   *
   * \warning NS_FATAL_ERROR is called if \p unit is not valid.
   * \warning Use Length::TryParse to parse potentially bad values without terminating.
   *
   * \param value Numeric value of the new length
   * \param unit Length unit of the value
   */
  Length (double value, Length::Unit unit);

  /**
   * Construct a Length object from a Quantity
   *
   * \param quantity Quantity representing a length value and unit
   */
  Length (Quantity quantity);

#ifdef HAVE_BOOST_UNITS
  /**
   * Construct a Length object from a boost::units::quantity
   *
   * \note The boost::units:quantity must contain a unit that derives from
   * the length dimension.  Passing a quantity with a Unit that is not a length
   * unit will result in a compile time error
   *
   * \tparam U A boost::units length unit
   * \tparam T Numeric data type of the quantity value
   *
   * \param quantity A boost::units length quantity
   */
  template <class U, class T>
  explicit Length (boost::units::quantity<U, T> quantity);
#endif

  /**
   * Copy Constructor
   *
   * Initialize an object with the value from \p other.
   *
   * \param other Length object to copy
   */
  Length (const Length& other) = default;

  /**
   * Move Constructor
   *
   * Initialize an object with the value from \p other.
   *
   * After the move completes, \p other is left in an undefined but
   * useable state.
   *
   * \param other Length object to move
   */
  Length (Length&& other) = default;

  /**
   * Destructor
   */
  ~Length () = default;

  /**
   * Copy Assignment operator
   *
   * Replace the current value with the value from \p other
   *
   * \param other Length object to copy
   *
   * \return Reference to the updated object
   */
  Length& operator= (const Length& other) = default;

  /**
   * Move Assignment operator
   *
   * Replace the current value with the value from \p other
   * After the move, \p other is left in an undefined but valid state
   *
   * \param other Length object to move
   *
   * \return Reference to the updated object
   */
  Length& operator= (Length&& other) = default;

  /**
   * Assignment operator
   *
   * Replace the current value with the value from \p q
   *
   * \param q Quantity holding the value to assign
   *
   * \return Reference to the updated object
   */
  Length& operator= (const Length::Quantity& q);

  /**
   * Check if \p other is equal in value to this instance.
   *
   * \param other Value to compare against
   * \param tolerance Smallest difference allowed between the two
   * values to still be considered equal
   *
   * \return true if the absolute difference between lengths
   * is less than or equal to \p tolerance
   */
  bool IsEqual (const Length& other, double tolerance = DEFAULT_TOLERANCE) const;

  /**
   * Check if \p other is not equal in value to this instance.
   *
   * \param other Value to compare against
   * \param tolerance Smallest difference allowed between the two
   * values to still be considered equal
   *
   * \return true if the absolute difference between lengths
   * is greater than \p tolerance
   */
  bool IsNotEqual (const Length& other, double tolerance = DEFAULT_TOLERANCE) const;

  /**
   * Check if \p other is greater in value than this instance.
   *
   * \param other Value to compare against
   * \param tolerance Smallest difference allowed between the two
   * values to still be considered equal
   *
   * \return true if the values are not equal and \p other is
   * greater in value
   */
  bool IsLess (const Length& other, double tolerance = DEFAULT_TOLERANCE) const;

  /**
   * Check if \p other is greater or equal in value than this instance.
   *
   * \param other Value to compare against
   * \param tolerance Smallest difference allowed between the two
   * values to still be considered equal
   *
   * Equivalent to:
   * \code
   *   IsEqual(other, tolerance) || IsLess(other, tolerance)
   * \endcode
   *
   * \return true if the values are equal or \p other is
   * greater in value
   */
  bool IsLessOrEqual (const Length& other, double tolerance = DEFAULT_TOLERANCE) const;

  /**
   * Check if \p other is less in value than this instance.
   *
   * \param other Value to compare against
   * \param tolerance Smallest difference allowed between the two
   * values to still be considered equal
   *
   * Equivalent to:
   * \code
   *   !(IsLessOrEqual(other, tolerance))
   * \endcode
   *
   * \return true if the values are not equal and \p other is
   * less in value
   */
  bool IsGreater (const Length& other, double tolerance = DEFAULT_TOLERANCE) const;

  /**
   * Check if \p other is equal or less in value than this instance.
   *
   * \param other Value to compare against
   * \param tolerance Smallest difference allowed between the two
   * values to still be considered equal
   *
   * Equivalent to:
   * \code
   *   !IsLess(other, tolerance)
   * \endcode
   *
   * \return true if the values are equal or \p other is
   * less in value
   */
  bool IsGreaterOrEqual (const Length& other, double tolerance = DEFAULT_TOLERANCE) const;

  /**
   * Swap values with another object
   *
   * Swap the current value with the value in \p other.
   *
   * Equivalent to:
   * \code
   *   Length temp(*this);
   *   *this = other;
   *   other = temp;
   * \endcode
   *
   * \param other Length object to swap
   */
  void swap (Length& other);

  /**
   * Current length value
   *
   * Equivalent to:
   * \code
   *   As (Length::Unit::Meter).Value ()
   * \endcode
   * \return The current value, in meters
   */
  double GetDouble () const;

  /**
   * Create a Quantity in a specific unit from a Length
   *
   * Converts the current length value to the equivalent value specified by
   * \p unit and returns a Quantity object with the converted value and unit
   *
   * \param unit The desired unit of the returned Quantity
   *
   * \return A quantity representing the length in the requested unit
   */
  Quantity As (Unit unit) const;

private:
  double m_value;      //!< Length in meters
};  // class Length

ATTRIBUTE_HELPER_HEADER (Length);

/**
 * \ingroup length
 * \brief Return the symbol of the supplied unit
 *
 * The symbol of the unit is the shortened form of the unit name and is usually
 * two or three characters long
 *
 * \param unit The unit to symbolize
 *
 * \return String containing the symbol of \p unit
 */
std::string ToSymbol (Length::Unit unit);

/**
 * \ingroup length
 * \brief Return the name of the supplied unit
 *
 * The value returned by this function is the common name of \p unit. The output
 * is always lowercase.
 *
 * If \p plural is true, then the plural form of the common name is returned
 * instead.
 *
 * \param unit The unit to name
 * \param plural Boolean indicating if the returned string should contain the
 * plural form of the name
 *
 * \return String containing the full name of \p unit
 */
std::string ToName (Length::Unit unit, bool plural = false);

/**
 * \ingroup length
 * \brief Find the equivalent Length::Unit for a unit string
 *
 * The string value can be a symbol or name (plural or singular).
 *
 * The string comparison ignores case so strings like "NanoMeter", "centiMeter",
 * "METER" will all match the correct unit.
 *
 * Leading and trailing whitespace are trimmed from the string before searching
 * for a match.
 *
 * \param unitString String containing the symbol or name of a length unit
 *
 * \return A std::optional object containing a Length::Unit if a match for the
 * string could be found
 */
std::optional<Length::Unit> FromString (std::string unitString);

/**
 * \ingroup length
 * \brief Write a length value to an output stream.
 *
 * The output of the length is in meters.
 *
 * Equivalent to:
 * \code
 *   stream << l.As (Meter);
 * \endcode
 *
 * \param stream Output stream
 * \param l Length value to write to the stream
 *
 * \return Reference to the output stream
 */
std::ostream& operator<< (std::ostream& stream, const Length& l);

/**
 * \ingroup length
 * \brief Write a Quantity to an output stream.
 *
 * The data written to the output stream will have the format \<value\> \<symbol\>
 *
 * Equivalent to:
 * \code
 *   stream << q.Value () << ' ' << ToSymbol (q.Unit());
 * \endcode
 *
 * \param stream Output stream
 * \param q Quantity to write to the output stream
 *
 * \return Reference to the output stream
 */
std::ostream& operator<< (std::ostream& stream, const Length::Quantity& q);

/**
 * \ingroup length
 * \brief Write a Length::Unit to an output stream.
 *
 * Writes the name of \p unit to the output stream
 *
 * Equivalent to:
 * \code
 *   stream << ToName (unit);
 * \endcode
 *
 * \param stream Output stream
 * \param unit Length unit to output
 *
 * \return Reference to the output stream
 */
std::ostream& operator<< (std::ostream& stream, Length::Unit unit);

/**
 * \ingroup length
 * \brief Read a length value from an input stream.
 *
 * The expected format of the input is \<number\> \<unit\>
 * or \<number\>\<unit\>
 * This function calls NS_ABORT if the input stream does not contain
 * a valid length string
 *
 * \param stream Input stream
 * \param l Object where the deserialized value will be stored
 *
 * \return Reference to the input stream
 */
std::istream& operator>> (std::istream& stream, Length& l);

/**
 * \ingroup length
 * \brief Compare two length objects for equality.
 *
 * Equivalent to:
 * \code
 *   left.IsEqual(right, 0);
 * \endcode
 *
 * \param left Left length object
 * \param right Right length object
 *
 * \return true if \p left and \p right have the same value
 */
bool operator== (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Compare two length objects for inequality.
 *
 * Equivalent to:
 * \code
 *   left.IsNotEqual(right, 0);
 * \endcode
 *
 * \param left Left length object
 * \param right Right length object
 *
 * \return true if \p left and \p right do not have the same value
 */
bool operator!= (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Check if \p left has a value less than \p right
 *
 * Equivalent to:
 * \code
 *   left.IsLess(right, 0);
 * \endcode
 *
 * \param left Left length object
 * \param right Right length object
 *
 * \return true if \p left is less than \p right
 */
bool operator< (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Check if \p left has a value less than or equal to \p right
 *
 * Equivalent to:
 * \code
 *   left.IsLessOrEqual(right, 0);
 * \endcode
 *
 * \param left Left length object
 * \param right Right length object
 *
 * \return true if \p left is less than or equal to \p right
 */
bool operator<= (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Check if \p left has a value greater than \p right
 *
 * Equivalent to:
 * \code
 *   left.IsGreater(right, 0);
 * \endcode
 *
 * \param left Left length object
 * \param right Right length object
 *
 * \return true if \p left is greater than \p right
 */
bool operator> (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Check if \p left has a value greater than or equal to \p right
 *
 * Equivalent to:
 * \code
 *   left.IsGreaterOrEqual(right, 0);
 * \endcode
 *
 * \param left Left length object
 * \param right Right length object
 *
 * \return true if \p left is greater than or equal to \p right
 */
bool operator>= (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Add two length values together.
 *
 * Adds the values of \p left to \p right and returns a new
 * Length object containing the result.
 *
 * \param left A Length object
 * \param right A Length object
 *
 * \return A newly constructed Length object containing the
 * result of `left + right`.
 */
Length operator+ (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Subtract two length values.
 *
 * Subtracts the value of \p right from \p left and returns a
 * new Length object containing the result.
 *
 * \param left A Length object
 * \param right A Length object
 *
 * \return A newly constructed Length object containing the
 * result of `left - right`.
 */
Length operator- (const Length& left, const Length& right);

/**
 * \ingroup length
 * \brief Multiply a length value by a scalar
 *
 * Multiplies the value \p l by \p scalar and returns a new
 * Length object containing the result.
 *
 * \param l The Length object
 * \param scalar Multiplication factor
 *
 * \return A newly constructed Length object containing the result
 * of `l * scalar`.
 */
Length operator* (double scalar, const Length& l);
/**
 * \ingroup length
 * \brief Multiply a length value by a scalar
 *
 * Multiplies the value \p l by \p scalar and returns a new
 * Length object containing the result.
 *
 * \param l The Length object
 * \param scalar Multiplication factor
 *
 * \return A newly constructed Length object containing the result
 * of `l * scalar`.
 */
Length operator* (const Length& l, double scalar);

/**
 * \ingroup length
 * \brief Divide a length value by a scalar
 *
 * Divides the value \p left by \p scalar and returns a new
 * Length object containing the result.
 *
 * \p scalar must contain a non zero value.
 * NS_FATAL_ERROR is called if \p scalar is zero
 *
 * \param left Length value
 * \param scalar Multiplication factor
 *
 * \return A newly constructed Length object containing the result
 * of `left / scalar`.
 */
Length operator/ (const Length& left, double scalar);

/**
 * \ingroup length
 * \brief Divide a length value by another length value
 *
 * Divides the value \p numerator by the value \p denominator and
 * returns a scalar value containing the result.
 *
 * The return value will be NaN if \p denominator is 0.
 *
 * \param numerator The top value of the division
 * \param denominator The bottom value of the division
 *
 * \return A scalar value that is the result of `numerator / denominator` or
 * NaN if \p denominator is 0.
 */
double operator/ (const Length& numerator, const Length& denominator);

/**
 * \ingroup length
 * \brief Calculate how many times \p numerator can be split into \p denominator
 * sized pieces.
 *
 * If the result of `numerator / denominator` is not a whole number, the
 * result is rounded toward zero to the nearest whole number.
 * The amount remaining  after the division can be retrieved by passing a pointer
 * to a Length in \p remainder.  The remainder will be less than \p denominator
 * and have the same sign as \p numerator.
 *
 * NS_FATAL_ERROR is called if \p denominator is 0.
 *
 * \param numerator The value to split
 * \param denominator The length of each split
 * \param remainder Location to store the remainder
 *
 * \return The number of times \p numerator can be split into \p denominator
 * sized pieces, rounded down to the nearest whole number
 */
int64_t Div (const Length& numerator, const Length& denominator, Length* remainder = nullptr);

/**
 * \ingroup length
 * \brief Calculate the amount remaining after dividing two lengths
 *
 * The returned value will be less than \p denominator and have the same sign as
 * \p numerator.
 *
 * NS_FATAL_ERROR is called if \p denominator is 0.
 *
 * \param numerator The value to split
 * \param denominator The length of each split
 *
 * \return The amount remaining after splitting \p numerator into \p denominator
 * sized pieces.
 */
Length Mod (const Length& numerator, const Length& denominator);

/**
 * Construct a length from a value in the indicated unit.
 * @{
 * \ingroup length
 * \param value The numerical value.
 * \returns Length object.
 */
Length NanoMeters (double value);
Length MicroMeters (double value);
Length MilliMeters (double value);
Length CentiMeters (double value);
Length Meters (double value);
Length KiloMeters (double value);
Length NauticalMiles (double value);
Length Inches (double value);
Length Feet (double value);
Length Yards (double value);
Length Miles (double value);
/**@}*/

#ifdef HAVE_BOOST_UNITS
template <class U, class T>
Length::Length (boost::units::quantity<U, T> quantity)
  :   m_value (0)
{
  namespace bu = boost::units;
  using BoostMeters = bu::quantity<bu::si::length, double>;

  //convert value to meters
  m_value = static_cast<BoostMeters> (quantity).value ();
}
#endif

}  // namespace ns3

#endif  /* NS3_LENGTH_H_ */

