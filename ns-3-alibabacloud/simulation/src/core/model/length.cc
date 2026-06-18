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

#include "ns3/length.h"

#include "ns3/log.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <ratio>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

/**
 * \file
 * \ingroup length
 * ns3::Length implementation
 */

/**
 * \ingroup length
 * Unnamed namespace
 */
namespace {
/**
 * Helper function to scale an input value by a given ratio
 *
 * \tparam R a std::ratio
 *
 * \param value Input value to scale by R
 *
 * \return The result of value * R::num / R::den
 */
template<class R>
double ScaleValue (double value)
{
  return (value * R::num) / static_cast<double> (R::den);
}

/**
 * Convert a value in feet to the equivalent value in meters
 *
 * \param value Input value in feet
 *
 * \return Equivalent value in meters
 */
double FootToMeter (double value)
{
  return value * 0.3048;
}

/**
 * Convert a value in meters to the equivalent value in feet
 *
 * \param value Input value in meters
 *
 * \return Equivalent value in feet
 */
double MeterToFoot (double value)
{
  return value * 3.28084;
}

/**
 * Convert a value from a US Customary unit (inches, feet, yards etc.) to meters
 *
 * Value is scaled to feet then converted to meters
 *
 * \tparam R std::ratio needed to convert value to feet
 *
 * \param value Input value in some US Customary unit
 *
 * \return Equivalent value in meters
 */
template<class R>
double USToMeter (double value)
{
  return FootToMeter ( ScaleValue<R> (value) );
}

/**
 * Convert a value from meters to a US Customary unit (inches, feet, yards etc.)
 *
 * Value is converted to feet then scaled to the desired US Customary unit
 *
 * \tparam R std::ratio needed to convert feet to desired US customary unit
 *
 * \param value Input value in meters
 *
 * \return Equivalent value in a US customary unit
 */
template<class R>
double MeterToUS (double value)
{
  return ScaleValue<R> ( MeterToFoot (value) );
}

/**
 * Convert a value in one unit to the equivalent value in another unit
 *
 * \param value Length value in \p fromUnit units
 * \param fromUnit Unit of \p value
 * \param toUnit Target unit
 *
 * \return Result of converting value from \p fromUnit to \p toUnit
 */
double Convert (double value, ns3::Length::Unit fromUnit, ns3::Length::Unit toUnit)
{
  using Unit = ns3::Length::Unit;
  using Key = std::pair<Unit, Unit>;
  using Conversion = std::function<double (double)>;

  /**
   * Helper to generate hash values from pairs of Length::Units
   */
  struct KeyHash
  {
    std::size_t operator () (const Key& key) const noexcept
    {
      static_assert (sizeof(Unit) < sizeof(std::size_t),
                     "sizeof(Length::Unit) changed, it must be less than "
                     "sizeof(std::size_t)");

      int shift = sizeof(Unit) * 8;
      return static_cast<std::size_t> (key.first) << shift |
             static_cast<std::size_t> (key.second);
    }

  };

  using ConversionTable = std::unordered_map<Key, Conversion, KeyHash>;

  static ConversionTable CONVERSIONS {
    { {Unit::Nanometer, Unit::Meter}, ScaleValue<std::nano> },
    { {Unit::Meter, Unit::Nanometer}, ScaleValue<std::giga> },
    { {Unit::Micrometer, Unit::Meter}, ScaleValue<std::micro> },
    { {Unit::Meter, Unit::Micrometer}, ScaleValue<std::mega> },
    { {Unit::Millimeter, Unit::Meter}, ScaleValue<std::milli> },
    { {Unit::Meter, Unit::Millimeter}, ScaleValue<std::kilo> },
    { {Unit::Centimeter, Unit::Meter}, ScaleValue<std::centi> },
    { {Unit::Meter, Unit::Centimeter}, ScaleValue<std::hecto> },
    { {Unit::Meter, Unit::Meter}, ScaleValue<std::ratio<1,1> > },
    { {Unit::Kilometer, Unit::Meter}, ScaleValue<std::kilo> },
    { {Unit::Meter, Unit::Kilometer}, ScaleValue<std::milli> },
    { {Unit::NauticalMile, Unit::Meter}, ScaleValue<std::ratio<1852, 1> > },
    { {Unit::Meter, Unit::NauticalMile}, ScaleValue<std::ratio<1, 1852> > },
    { {Unit::Inch, Unit::Meter}, USToMeter<std::ratio<1, 12> > },
    { {Unit::Meter, Unit::Inch}, MeterToUS<std::ratio<12, 1> > },
    { {Unit::Foot, Unit::Meter}, FootToMeter },
    { {Unit::Meter, Unit::Foot}, MeterToFoot },
    { {Unit::Yard, Unit::Meter}, USToMeter<std::ratio<3, 1> > },
    { {Unit::Meter, Unit::Yard}, MeterToUS<std::ratio<1, 3> > },
    { {Unit::Mile, Unit::Meter}, USToMeter<std::ratio<5280, 1> > },
    { {Unit::Meter, Unit::Mile}, MeterToUS<std::ratio<1, 5280> > }
  };

  auto iter = CONVERSIONS.find ( Key {fromUnit, toUnit} );

  if (iter == CONVERSIONS.end ())
    {
      NS_FATAL_ERROR ("No conversion defined for " << fromUnit
                                                   << " -> " << toUnit);
    }

  return iter->second (value);
}

/**
 * Convert a Length::Quantity to the equivalent value in another unit
 *
 * \param from Quantity with the current value and unit
 * \param toUnit Target unit
 *
 * \return Result of converting the quantity value to the requested units
 */
double Convert (const ns3::Length::Quantity& from, ns3::Length::Unit toUnit)
{
  return Convert (from.Value (), from.Unit (), toUnit);
}

/**
 * Functor for hashing Length::Unit values
 *
 * This classes exists as a work around for a C++11 defect.  c++11 doesn't provide
 * a std::hash implementation for enums
 */
class EnumHash
{
public:
  /**
   * Produce a hash value for a Length::Unit
   *
   * \param u Length::Unit to hash
   *
   * \return Hash value for the Length::Unit
   */
  std::size_t operator () (ns3::Length::Unit u) const noexcept
  {
    return static_cast<std::size_t> (u);
  }
};

}   // unnamed namespace

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Length");

// Implement the attribute helper
ATTRIBUTE_HELPER_CPP (Length);

std::optional<Length>
Length::TryParse (double value, const std::string& unitString)
{
  NS_LOG_FUNCTION (value << unitString);

  auto unit = FromString (unitString);

  if (unit.has_value ())
    {
      return Length (value, *unit);
    }

  return std::nullopt;
}

Length::Length ()
  :   m_value (0)
{
  NS_LOG_FUNCTION (this);
}

Length::Length (const std::string& input)
  :   m_value (0)
{
  NS_LOG_FUNCTION (this << input);

  std::istringstream stream (input);

  stream >> *this;
}

Length::Length (double value, const std::string& unitString)
  :   m_value (0)
{
  NS_LOG_FUNCTION (this << value << unitString);

  auto unit = FromString (unitString);

  if (!unit.has_value ())
    {
      NS_FATAL_ERROR ("A Length object could not be constructed from the unit "
                      "string '" << unitString << "', because the string is not associated "
                      "with a Length::Unit entry");
    }

  m_value = Convert (value, *unit, Length::Unit::Meter);
}

Length::Length (double value, Length::Unit unit)
  :   m_value (0)
{
  NS_LOG_FUNCTION (this << value << unit);

  m_value = Convert (value, unit, Length::Unit::Meter);
}

Length::Length (Quantity quantity)
  : Length (quantity.Value (), quantity.Unit ())
{
  NS_LOG_FUNCTION (this << quantity);
}

Length&
Length::operator= (const Length::Quantity& q)
{
  NS_LOG_FUNCTION (this << q);

  m_value = Convert (q, Length::Unit::Meter);

  return *this;
}

bool
Length::IsEqual (const Length& other, double tolerance /*=DEFAULT_TOLERANCE*/) const
{
  NS_LOG_FUNCTION (this << m_value << other.m_value << tolerance);

  if ( m_value == other.m_value )
    {
      return true;
    }

  auto diff = std::abs (m_value - other.m_value);

  return diff <= tolerance;
}

bool
Length::IsNotEqual (const Length& other, double tolerance /*=DEFAULT_TOLERANCE*/) const
{
  NS_LOG_FUNCTION (this << m_value << other.m_value << tolerance);

  return !IsEqual (other, tolerance);
}

bool
Length::IsLess (const Length& other, double tolerance /*=DEFAULT_TOLERANCE*/) const
{
  NS_LOG_FUNCTION (this << m_value << other.m_value << tolerance);

  return m_value < other.m_value && IsNotEqual (other, tolerance);
}

bool
Length::IsLessOrEqual (const Length& other, double tolerance /*=DEFAULT_TOLERANCE*/) const
{
  NS_LOG_FUNCTION (this << m_value << other.m_value << tolerance);

  return m_value < other.m_value || IsEqual (other, tolerance);
}

bool
Length::IsGreater (const Length& other, double tolerance /*=DEFAULT_TOLERANCE*/) const
{
  NS_LOG_FUNCTION (this << m_value << other.m_value << tolerance);

  return !IsLessOrEqual (other, tolerance);
}

bool
Length::IsGreaterOrEqual (const Length& other, double tolerance /*=DEFAULT_TOLERANCE*/) const
{
  NS_LOG_FUNCTION (this << m_value << other.m_value << tolerance);

  return !IsLess (other, tolerance);
}

void
Length::swap (Length& other)
{
  using std::swap;

  swap (m_value, other.m_value);
}

double
Length::GetDouble () const
{
  return m_value;
}

Length::Quantity
Length::As (Length::Unit unit) const
{
  NS_LOG_FUNCTION (this << unit);

  double value = Convert (m_value, Length::Unit::Meter, unit);

  return Quantity (value, unit);
}

bool
operator== (const Length& left, const Length& right)
{
  return left.GetDouble () == right.GetDouble ();
}

bool
operator!= (const Length& left, const Length& right)
{
  return left.GetDouble () != right.GetDouble ();
}

bool
operator< (const Length& left, const Length& right)
{
  return left.GetDouble () < right.GetDouble ();
}

bool
operator<= (const Length& left, const Length& right)
{
  return left.GetDouble () <= right.GetDouble ();
}

bool
operator> (const Length& left, const Length& right)
{
  return left.GetDouble () > right.GetDouble ();
}

bool
operator>= (const Length& left, const Length& right)
{
  return left.GetDouble () >= right.GetDouble ();
}

Length
operator+ (const Length& left, const Length& right)
{
  double value = left.GetDouble () + right.GetDouble ();
  return Length ( value, Length::Unit::Meter );
}

Length
operator- (const Length& left, const Length& right)
{
  double value = left.GetDouble () - right.GetDouble ();
  return Length ( value, Length::Unit::Meter );
}

Length
operator* (const Length& left, double scalar)
{
  double value = left.GetDouble () * scalar;
  return Length ( value, Length::Unit::Meter );
}

Length
operator* (double scalar, const Length & right)
{
  return right * scalar;
}

Length
operator/ (const Length & left, double scalar)
{
  if (scalar == 0)
    {
      NS_FATAL_ERROR ("Attempted to divide Length by 0");
    }

  return left * (1.0 / scalar);
}

double
operator/ (const Length & numerator, const Length& denominator)
{
  if (denominator.GetDouble () == 0)
    {
      return std::numeric_limits<double>::quiet_NaN ();
    }

  return numerator.GetDouble () / denominator.GetDouble ();
}

int64_t
Div (const Length& numerator, const Length& denominator, Length* remainder)
{
  double value = numerator / denominator;

  if (std::isnan (value))
    {
      NS_FATAL_ERROR ("numerator / denominator return NaN");
    }

  if ( remainder )
    {
      double rem = std::fmod (numerator.GetDouble (), denominator.GetDouble ());
      *remainder = Length (rem, Length::Unit::Meter);
    }

  return static_cast<int64_t> (std::trunc (value));
}

Length
Mod (const Length& numerator, const Length& denominator)
{
  double rem = std::fmod (numerator.GetDouble (), denominator.GetDouble ());

  if (std::isnan (rem))
    {
      NS_FATAL_ERROR ("numerator / denominator return NaN");
    }

  return Length (rem, Length::Unit::Meter);
}

std::string
ToSymbol (Length::Unit unit)
{
  using StringTable = std::unordered_map<Length::Unit, std::string, EnumHash>;

  static const StringTable STRINGS {
    {Length::Unit::Nanometer, "nm"},
    {Length::Unit::Micrometer, "um"},
    {Length::Unit::Millimeter, "mm"},
    {Length::Unit::Centimeter, "cm"},
    {Length::Unit::Meter, "m"},
    {Length::Unit::Kilometer, "km"},
    {Length::Unit::NauticalMile, "nmi"},
    {Length::Unit::Inch, "in"},
    {Length::Unit::Foot, "ft"},
    {Length::Unit::Yard, "yd"},
    {Length::Unit::Mile, "mi"}
  };

  auto iter = STRINGS.find (unit);

  if (iter == STRINGS.end ())
    {
      NS_FATAL_ERROR ("A symbol could not be found for Length::Unit with value "
                      << EnumHash ()(unit));
    }

  return iter->second;
}

std::string
ToName (Length::Unit unit, bool plural /*=false*/)
{
  using Entry = std::tuple<std::string, std::string>;
  using StringTable = std::unordered_map<Length::Unit, Entry, EnumHash>;

  static const StringTable STRINGS {
    {Length::Unit::Nanometer,    Entry{"nanometer", "nanometers"}},
    {Length::Unit::Micrometer,   Entry{"micrometer", "micrometer"}},
    {Length::Unit::Millimeter,   Entry{"millimeter", "millimeters"}},
    {Length::Unit::Centimeter,   Entry{"centimeter", "centimeters"}},
    {Length::Unit::Meter,        Entry{"meter", "meters"}},
    {Length::Unit::Kilometer,    Entry{"kilometer", "kilometers"}},
    {Length::Unit::NauticalMile, Entry{"nautical mile", "nautical miles"}},
    {Length::Unit::Inch,         Entry{"inch", "inches"}},
    {Length::Unit::Foot,         Entry{"foot", "feet"}},
    {Length::Unit::Yard,         Entry{"yard", "yards"}},
    {Length::Unit::Mile,         Entry{"mile", "miles"}}
  };

  auto iter = STRINGS.find (unit);

  if (iter == STRINGS.end ())
    {
      NS_FATAL_ERROR ("A symbol could not be found for Length::Unit with value "
                      << EnumHash ()(unit));
    }

  if (plural)
    {
      return std::get<1> (iter->second);
    }

  return std::get<0> (iter->second);
}

std::optional<Length::Unit>
FromString (std::string unitString)
{
  using UnitTable = std::unordered_map<std::string, Length::Unit>;

  static const UnitTable UNITS {
    { "nm", Length::Unit::Nanometer },
    { "nanometer", Length::Unit::Nanometer },
    { "nanometers", Length::Unit::Nanometer },
    { "nanometre", Length::Unit::Nanometer },
    { "nanometres", Length::Unit::Nanometer },
    { "um", Length::Unit::Micrometer },
    { "micrometer", Length::Unit::Micrometer },
    { "micrometers", Length::Unit::Micrometer },
    { "micrometre", Length::Unit::Micrometer },
    { "micrometres", Length::Unit::Micrometer },
    { "mm", Length::Unit::Millimeter },
    { "millimeter", Length::Unit::Millimeter },
    { "millimeters", Length::Unit::Millimeter },
    { "millimetre", Length::Unit::Millimeter },
    { "millimetres", Length::Unit::Millimeter },
    { "cm", Length::Unit::Centimeter },
    { "centimeter", Length::Unit::Centimeter },
    { "centimeters", Length::Unit::Centimeter },
    { "centimetre", Length::Unit::Centimeter },
    { "centimetres", Length::Unit::Centimeter },
    { "m", Length::Unit::Meter },
    { "meter", Length::Unit::Meter },
    { "metre", Length::Unit::Meter },
    { "meters", Length::Unit::Meter },
    { "metres", Length::Unit::Meter },
    { "km", Length::Unit::Kilometer },
    { "kilometer", Length::Unit::Kilometer },
    { "kilometers", Length::Unit::Kilometer },
    { "kilometre", Length::Unit::Kilometer },
    { "kilometres", Length::Unit::Kilometer },
    { "nmi", Length::Unit::NauticalMile },
    { "nauticalmile", Length::Unit::NauticalMile },
    { "nauticalmiles", Length::Unit::NauticalMile },
    { "in", Length::Unit::Inch },
    { "inch", Length::Unit::Inch },
    { "inches", Length::Unit::Inch },
    { "ft", Length::Unit::Foot },
    { "foot", Length::Unit::Foot },
    { "feet", Length::Unit::Foot },
    { "yd", Length::Unit::Yard },
    { "yard", Length::Unit::Yard },
    { "yards", Length::Unit::Yard },
    { "mi", Length::Unit::Mile },
    { "mile", Length::Unit::Mile },
    { "miles", Length::Unit::Mile }
  };

  //function to trim whitespace and convert to lowercase in one pass
  static auto Normalize = [] (const std::string& str)
    {
      std::string output;
      output.reserve (str.size ());

      for (unsigned char c : str)
        {
          //this strips all spaces not just beg/end but is fine for our purposes
          if (std::isspace (c) )
            {
              continue;
            }

          output.push_back (std::tolower (c));
        }

      return output;
    };

  unitString = Normalize (unitString);

  auto iter = UNITS.find (unitString);

  if (iter != UNITS.end ())
    {
      return iter->second;
    }

  return std::nullopt;
}

std::ostream&
operator<< (std::ostream& stream, const Length& l)
{
  stream << l.As (Length::Unit::Meter);

  return stream;
}

std::ostream&
operator<< (std::ostream& stream, const Length::Quantity& q)
{
  stream << q.Value () << ' ' << ToSymbol (q.Unit ());

  return stream;
}

std::ostream&
operator<< (std::ostream& stream, Length::Unit unit)
{
  stream << ToName (unit);

  return stream;
}

/**
 * This function provides a string parsing method that does not rely
 * on istream, which has been found to have different behaviors in different
 * implementations.
 *
 * The input string can either contain a double (for example, "5.5") or
 * a double and a string with no space between them (for example, "5.5m")
 *
 * \param input The input string
 * \return A three element tuple containing the result of parsing the string.
 * The first tuple element is a boolean indicating whether the parsing succeeded
 * or failed.  The second element contains the value of the double that was
 * extracted from the string.  The third element was the unit symbol that was
 * extracted from the string.  If the input string did not have a unit symbol,
 * the third element will contain an empty string.
 */
std::tuple<bool, double, std::string>
ParseLengthString (const std::string& input)
{
    NS_LOG_FUNCTION (input);

    double value = 0;
    std::size_t pos = 0;
    std::string symbol;

    try
    {
        value = std::stod(input, &pos);
    }
    catch (const std::exception& e)
    {
        NS_LOG_ERROR ("Caught exception while parsing double: " << e.what());

        return std::make_tuple(false, 0, "");
    }

    //skip any whitespace between value and symbol
    while (pos < input.size () && std::isspace(input[pos]))
        ++pos;

    if (pos < input.size ())
    {
        NS_LOG_LOGIC ("String has value and symbol, extracting symbol");

        //input has a double followed by a string
        symbol = input.substr(pos);
    }

    return std::make_tuple(true, value, symbol);
}

std::istream&
operator>> (std::istream& stream, Length& l)
{
  bool success = false;
  double value = 0;
  std::string symbol;
  std::string temp;

  //configure stream to skip whitespace in case it was disabled
  auto origFlags = stream.flags ();
  std::skipws (stream);

  //Read the contents into a temporary string and parse it manually
  stream >> temp;

  std::tie(success, value, symbol) = ParseLengthString (temp);

  if (success && symbol.empty ())
  {
      NS_LOG_LOGIC ("Temp string only contained value, extracting unit symbol from stream");

      //temp only contained the double
      //still need to read the symbol from the stream
      stream >> symbol;
  }

  //special handling for nautical mile which is two words
  if (symbol == "nautical")
    {
      stream >> temp;

      if (!temp.empty ())
        {
          symbol.push_back (' ');
          symbol.append (temp);
        }
    }

  Length (value, symbol).swap (l);

  //restore original flags
  stream.flags (origFlags);

  return stream;
}

Length
NanoMeters (double value)
{
  return Length (value, Length::Unit::Nanometer);
}

Length
MicroMeters (double value)
{
  return Length (value, Length::Unit::Micrometer);
}

Length
MilliMeters (double value)
{
  return Length (value, Length::Unit::Millimeter);
}

Length
CentiMeters (double value)
{
  return Length (value, Length::Unit::Centimeter);
}

Length
Meters (double value)
{
  return Length (value, Length::Unit::Meter);
}

Length
KiloMeters (double value)
{
  return Length (value, Length::Unit::Kilometer);
}

Length
NauticalMiles (double value)
{
  return Length (value, Length::Unit::NauticalMile);
}

Length
Inches (double value)
{
  return Length (value, Length::Unit::Inch);
}

Length
Feet (double value)
{
  return Length (value, Length::Unit::Foot);
}

Length
Yards (double value)
{
  return Length (value, Length::Unit::Yard);
}

Length
Miles (double value)
{
  return Length (value, Length::Unit::Mile);
}

/**@}*/

}   // namespace ns3
