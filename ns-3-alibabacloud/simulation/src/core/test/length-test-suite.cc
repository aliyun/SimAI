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
 * Authors: Mathew Bielejeski<bielejeski1@llnl.gov>
 */

#include "ns3/length.h"
#include "ns3/log.h"
#include "ns3/object.h"
#include "ns3/string.h"
#include "ns3/test.h"

#ifdef HAVE_BOOST
#include <boost/units/base_units/us/foot.hpp>
#include <boost/units/systems/si.hpp>
#include <boost/units/systems/si/prefixes.hpp>
#endif

#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>

/**
 * \file
 * \ingroup length-tests
 * Length class tests.
 */

/**
 * \ingroup core-tests length
 * \defgroup length-tests Length test suite
 */

using namespace ns3;

/**
 * Save some typing by defining a short alias for Length::Unit
 */
using Unit = Length::Unit;

/**
 * Implements tests for the Length class
 */
class LengthTestCase : public TestCase
{
public:
  /**
   * Constructor
   */
  LengthTestCase ()
    :   TestCase ("length-tests")
  {}

  /**
   * Destructor
   */
  virtual ~LengthTestCase () = default;

protected:
  /**
   * Helper function to compare results with false
   *
   * \param condition The boolean condition to test
   * \param msg The message to print if the test fails
   */
  void AssertFalse (bool condition, std::string msg)
  {
    NS_TEST_ASSERT_MSG_EQ (condition, false, msg);
  }

  /**
   * Helper function to compare results with true
   *
   * \param condition The boolean condition to test
   * \param msg The message to print if the test fails
   */
  void AssertTrue (bool condition, std::string msg)
  {
    NS_TEST_ASSERT_MSG_EQ (condition, true, msg);
  }

private:
  /**
   * Test that a default constructed Length object has a value of 0
   */
  void TestDefaultLengthIsZero ();

  /**
   * Test that a Length object can be constructed from a Quantity object
   */
  void TestConstructLengthFromQuantity ();

  /**
   * Test that a Length object constructed from various SI units has the
   * correct value in meters
   */
  void TestConstructLengthFromSIUnits ();

  /**
   * Test that a Length object constructed from various US units has the
   * correct value in meters
   */
  void TestConstructLengthFromUSUnits ();

  /**
   * Test that the value from one length is copied to another
   * using the copy constructor.
   */
  void TestLengthCopyConstructor ();

  /**
   * Test that the value from one length is copied to another
   * using the move constructor.
   */
  void TestLengthMoveConstructor ();

  /**
   * Test that a length object can be constructed from a string
   * \param unitValue  //!< Value to test.
   * \param meterValue //!< Reference value [m].
   * \param tolerance  //!< Tolerance.
   * \param symbols    //!< Unit symbols.
   */
  void TestConstructLengthFromString (double unitValue,
                                      double meterValue,
                                      double tolerance,
                                      const std::initializer_list<std::string>& symbols);

  /**
   * Test that a length object can be constructed from a string
   * @{
   */
  void TestConstructLengthFromMeterString ();
  void TestConstructLengthFromNanoMeterString ();
  void TestConstructLengthFromMicroMeterString ();
  void TestConstructLengthFromMilliMeterString ();
  void TestConstructLengthFromCentiMeterString ();
  void TestConstructLengthFromKiloMeterString ();
  void TestConstructLengthFromNauticalMileString ();
  void TestConstructLengthFromInchString ();
  void TestConstructLengthFromFootString ();
  void TestConstructLengthFromYardString ();
  void TestConstructLengthFromMileString ();
  /** @} */

#ifdef HAVE_BOOST_UNITS
  /**
   * Test construction from boost::units
   * @{
   */
  void TestConstructLengthFromBoostUnits ();
  void TestConstructLengthFromBoostUnitsMeters ();
  void TestConstructLengthFromBoostUnitsKiloMeters ();
  void TestConstructLengthFromBoostUnitsFeet ();
  /** @} */
#endif

  /**
   * Test constructing length objects using the builder free functions
   * @{
   */
  void TestBuilderFreeFunctions ();
  /** @} */

  /**
   * Test the TryParse function returns false on bad input
   */
  void TestTryParseReturnsFalse ();

  /**
   * Test the TryParse function returns true on success
   */
  void TestTryParseReturnsTrue ();


  /**
   * Test that a length object can be updated by assignment from another
   * length object
   */
  void TestCopyAssignment ();

  /**
   * Test that a length object can be updated by assignment from a moved
   * length object
   */
  void TestMoveAssignment ();

  /**
   * Test that a length object can be updated by assignment from a quantity
   */
  void TestQuantityAssignment ();

  /**
   * Test member comparison operators
   * @{
   */
  void TestIsEqualReturnsTrue ();
  void TestIsEqualReturnsFalse ();
  void TestIsEqualWithToleranceReturnsTrue ();
  void TestIsEqualWithToleranceReturnsFalse ();
  void TestIsNotEqualReturnsTrue ();
  void TestIsNotEqualReturnsFalse ();
  void TestIsNotEqualWithToleranceReturnsTrue ();
  void TestIsNotEqualWithToleranceReturnsFalse ();
  void TestIsLessReturnsTrue ();
  void TestIsLessReturnsFalse ();
  void TestIsLessWithToleranceReturnsFalse ();
  void TestIsGreaterReturnsTrue ();
  void TestIsGreaterReturnsFalse ();
  void TestIsGreaterWithToleranceReturnsFalse ();
  /** @} */

  /**
   * Test writing length object to a stream produces the expected output
   */
  void TestOutputStreamOperator ();

  /**
   * Test reading length object from a stream produces the expected length
   * value
   */
  void TestInputStreamOperator ();

  /**
   * Generic function for testing serialization of a Length object in
   * various units
   *
   * \tparam T Type of the length unit that should be output during serialization
   *
   * \param l Length object to serialize
   * \param unit Unit that the length value will be converted to before serialization
   * \param expectedOutput Expected result of the serialization
   * \param context Included in the error message if the test fails
   */
  template<class T>
  void TestLengthSerialization (const Length& l,
                                const T& unit,
                                const std::string& expectedOutput,
                                const std::string& context);

  /**
   * Test serializing a length object to all of the supported unit types
   */
  void TestSerializeLengthWithUnit ();

  /**
   * Test free function comparison operators
   * @{
   */
  void TestOperatorEqualsReturnsTrue ();
  void TestOperatorEqualsReturnsFalse ();
  void TestOperatorNotEqualsReturnsTrue ();
  void TestOperatorNotEqualsReturnsFalse ();
  void TestOperatorLessThanReturnsTrue ();
  void TestOperatorLessThanReturnsFalse ();
  void TestOperatorLessOrEqualReturnsTrue ();
  void TestOperatorLessOrEqualReturnsFalse ();
  void TestOperatorGreaterThanReturnsTrue ();
  void TestOperatorGreaterThanReturnsFalse ();
  void TestOperatorGreaterOrEqualReturnsTrue ();
  void TestOperatorGreaterOrEqualReturnsFalse ();
  /** @} */

  /**
   * Test arithmetic operations
   * @{
   */
  void TestAddingTwoLengths ();
  void TestAddingLengthAndQuantity ();
  void TestAddingQuantityAndLength ();
  void TestSubtractingTwoLengths ();
  void TestSubtractingLengthAndQuantity ();
  void TestSubtractingQuantityAndLength ();
  void TestMultiplyLengthByScalar ();
  void TestMultiplyScalarByLength ();
  void TestDivideLengthByScalar ();
  void TestDivideLengthByLength ();
  void TestDivideLengthByLengthReturnsNaN ();
  /** @} */

  /**
   * Test Div function
   * @{
   */
  void TestDivReturnsCorrectResult ();
  void TestDivReturnsZeroRemainder ();
  void TestDivReturnsCorrectRemainder ();
  /** @} */

  /**
   * Test Mod function
   * @{
   */
  void TestModReturnsZero ();
  void TestModReturnsNonZero ();
  /** @} */

  virtual void DoRun ();
};

void
LengthTestCase::TestDefaultLengthIsZero ()
{
  Length l;

  NS_TEST_ASSERT_MSG_EQ (l.GetDouble (), 0, "Default value of Length is not 0");
}

void
LengthTestCase::TestConstructLengthFromQuantity ()
{
  const Length::Quantity VALUE (5.0, Unit::Meter);

  Length l (VALUE);

  NS_TEST_ASSERT_MSG_EQ (l.GetDouble (), VALUE.Value (),
                         "length constructed from meters has wrong value");
}

void
LengthTestCase::TestConstructLengthFromSIUnits ()
{
  using TestEntry = std::tuple<Length, std::string>;

  const double expectedMeters = 1;
  const std::initializer_list<TestEntry> inputs {
    std::make_tuple (Length (1e9, Unit::Nanometer), "nanometer"),
    std::make_tuple (Length (1e6, Unit::Micrometer), "micrometer"),
    std::make_tuple (Length (1e3, Unit::Millimeter), "millimeter"),
    std::make_tuple (Length (1e2, Unit::Centimeter), "centimeter"),
    std::make_tuple (Length (1e-3, Unit::Kilometer), "kilometer"),
    std::make_tuple (Length ( (1 / 1852.0), Unit::NauticalMile), "nautical_mile")
  };

  for (const TestEntry& entry : inputs )
    {
      const Length& l = std::get<0> (entry);
      const std::string& context = std::get<1> (entry);

      NS_TEST_ASSERT_MSG_EQ (l.GetDouble (), expectedMeters,
                             context << ": constructed length from SI unit has wrong value");
    }
}

void
LengthTestCase::TestConstructLengthFromUSUnits ()
{
  using TestEntry = std::tuple<Length, std::string>;

  const double expectedMeters = 0.3048;
  const double tolerance = 0.0001;

  const std::initializer_list<TestEntry> inputs {
    std::make_tuple (Length (12.0, Unit::Inch), "inch"),
    std::make_tuple (Length (1.0, Unit::Foot), "foot"),
    std::make_tuple (Length ((1 / 3.0), Unit::Yard), "yard"),
    std::make_tuple (Length ((1 / 5280.0), Unit::Mile), "mile"),
  };

  for (const TestEntry& entry : inputs )
    {
      const Length& l = std::get<0> (entry);
      const std::string& context = std::get<1> (entry);

      NS_TEST_ASSERT_MSG_EQ_TOL (l.GetDouble (), expectedMeters, tolerance,
                                 "constructed length from US unit (" << context << ") has wrong value");
    }
}

void
LengthTestCase::TestLengthCopyConstructor ()
{
  const double value = 5;
  Length original (value, Unit::Meter);

  Length copy (original);

  NS_TEST_ASSERT_MSG_EQ (copy.GetDouble (), original.GetDouble (),
                         "copy constructed length has wrong value");
}

void
LengthTestCase::TestLengthMoveConstructor ()
{
  const double value = 5;
  Length original (value, Unit::Meter);

  Length copy (std::move (original));

  NS_TEST_ASSERT_MSG_EQ (copy.GetDouble (), value,
                         "move constructed length has wrong value");
}

void
LengthTestCase::TestConstructLengthFromString (double unitValue,
                                               double meterValue,
                                               double tolerance,
                                               const std::initializer_list<std::string>& symbols)
{
  const std::array<std::string, 2> SEPARATORS {{"", " "}};

  for (const std::string& symbol : symbols)
    {
      for ( const std::string& separator : SEPARATORS )
        {
          std::ostringstream stream;

          stream << unitValue << separator << symbol;

          Length l (stream.str ());

          std::ostringstream msg;
          msg << "string constructed length has wrong value: '" << stream.str () << "'";

          NS_TEST_ASSERT_MSG_EQ_TOL (l.GetDouble (), meterValue, tolerance, msg.str ());
        }
    }
}

void
LengthTestCase::TestConstructLengthFromMeterString ()
{
  const double value = 5;

  TestConstructLengthFromString (value, value, 0,
                                 {"m", "meter", "meters", "metre", "metres"});
}

void
LengthTestCase::TestConstructLengthFromNanoMeterString ()
{
  const double value = 5;
  const double expectedValue = 5e-9;

  TestConstructLengthFromString (value, expectedValue, 0,
                                 {"nm", "nanometer", "nanometers",
                                  "nanometre", "nanometres"});
}

void
LengthTestCase::TestConstructLengthFromMicroMeterString ()
{
  const double value = 5;
  const double expectedValue = 5e-6;
  const double tolerance = 1e-7;

  TestConstructLengthFromString (value, expectedValue, tolerance,
                                 {"um", "micrometer", "micrometers",
                                  "micrometre", "micrometres"});
}

void
LengthTestCase::TestConstructLengthFromMilliMeterString ()
{
  const double value = 5;
  const double expectedValue = 5e-3;
  const double tolerance = 1e-4;

  TestConstructLengthFromString (value, expectedValue, tolerance,
                                 {"mm", "millimeter", "millimeters",
                                  "millimetre", "millimetres"});
}

void
LengthTestCase::TestConstructLengthFromCentiMeterString ()
{
  const double value = 5;
  const double expectedValue = 5e-2;
  const double tolerance = 1e-3;

  TestConstructLengthFromString (value, expectedValue, tolerance,
                                 {"cm", "centimeter", "centimeters",
                                  "centimetre", "centimetres"});
}

void
LengthTestCase::TestConstructLengthFromKiloMeterString ()
{
  const double value = 5;
  const double expectedValue = 5e3;

  TestConstructLengthFromString (value, expectedValue, 0,
                                 {"km", "kilometer", "kilometers",
                                  "kilometre", "kilometres"});
}

void
LengthTestCase::TestConstructLengthFromNauticalMileString ()
{
  const double value = 5;
  const double expectedValue = 9260;

  TestConstructLengthFromString (value, expectedValue, 0,
                                 {"nmi", "nautical mile", "nautical miles"});
}
void
LengthTestCase::TestConstructLengthFromInchString ()
{
  const double value = 5;
  const double expectedValue = 0.127;
  const double tolerance = 1e-4;

  TestConstructLengthFromString (value, expectedValue, tolerance,
                                 {"in", "inch", "inches"});
}

void
LengthTestCase::TestConstructLengthFromFootString ()
{
  const double value = 5;
  const double expectedValue = 1.524;
  const double tolerance = 1e-4;

  TestConstructLengthFromString (value, expectedValue, tolerance,
                                 {"ft", "foot", "feet"});
}

void
LengthTestCase::TestConstructLengthFromYardString ()
{
  const double value = 5;
  const double expectedValue = 4.572;
  const double tolerance = 1e-4;

  TestConstructLengthFromString (value, expectedValue, tolerance,
                                 {"yd", "yard", "yards"});
}

void
LengthTestCase::TestConstructLengthFromMileString ()
{
  const double value = 5;
  const double expectedValue = 8046.72;
  const double tolerance = 1e-3;

  TestConstructLengthFromString (value, expectedValue, tolerance,
                                 {"mi", "mile", "miles"});
}

#ifdef HAVE_BOOST_UNITS
void
LengthTestCase::TestConstructLengthFromBoostUnits ()
{
  TestConstructLengthFromBoostUnitsMeters ();
  TestConstructLengthFromBoostUnitsKiloMeters ();
  TestConstructLengthFromBoostUnitsFeet ();
}

void
LengthTestCase::TestConstructLengthFromBoostUnitsMeters ()
{
  namespace bu = boost::units;

  auto meters = 5 * bu::si::meter;

  Length l (meters);

  NS_TEST_ASSERT_MSG_EQ (l.GetDouble (), meters.value (),
                         "Construction from boost::units meters produced "
                         "incorrect value");
}

void
LengthTestCase::TestConstructLengthFromBoostUnitsKiloMeters ()
{
  namespace bu = boost::units;
  auto kilometer = bu::si::kilo * bu::si::meter;

  const double expectedValue = 5000;
  auto quantity = 5 * kilometer;

  Length l (quantity);

  NS_TEST_ASSERT_MSG_EQ (l.GetDouble (), expectedValue,
                         "Construction from boost::units kilometers produced "
                         "incorrect value");
}

void
LengthTestCase::TestConstructLengthFromBoostUnitsFeet ()
{
  namespace bu = boost::units;

  bu::us::foot_base_unit::unit_type Foot;

  const double expectedValue = 3.048;
  auto feet = 10 * Foot;

  Length l (feet);

  NS_TEST_ASSERT_MSG_EQ_TOL (l.GetDouble (), expectedValue, 0.001,
                             "Construction from boost::units foot produced "
                             "incorrect value");
}
#endif

void
LengthTestCase::TestBuilderFreeFunctions ()
{
  using Builder = std::function<Length (double)>;

  double inputValue = 10;

  std::map<Unit, Builder> TESTDATA{
    {Unit::Nanometer, NanoMeters},
    {Unit::Micrometer, MicroMeters},
    {Unit::Millimeter, MilliMeters},
    {Unit::Centimeter, CentiMeters},
    {Unit::Meter, Meters},
    {Unit::Kilometer, KiloMeters},
    {Unit::NauticalMile, NauticalMiles},
    {Unit::Inch, Inches},
    {Unit::Foot, Feet},
    {Unit::Yard, Yards},
    {Unit::Mile, Miles}
  };

  for (auto& entry : TESTDATA)
    {
      Length expected (inputValue, entry.first);

      Length output = entry.second (inputValue);

      NS_TEST_ASSERT_MSG_EQ (output, expected,
                             "The builder free function for " << entry.first <<
                             " did not create a Length with the correct value");
    }
}

void
LengthTestCase::TestTryParseReturnsFalse ()
{
  auto l = Length::TryParse (1, "");

  AssertFalse (l.has_value (), "TryParse returned true on bad input");
}

void
LengthTestCase::TestTryParseReturnsTrue ()
{
  using TestInput = std::pair<double, std::string>;
  using TestArgs = std::pair<double, double>;
  std::map<TestInput, TestArgs> tests{
    {{5, "m"}, {5, 0}},
    {{5, " m"}, {5, 0}},
    {{5, "kilometer"}, {5e3, 0}},
    {{5, " kilometer"}, {5e3, 0}}
  };

  for (auto& entry : tests)
    {
      TestInput input = entry.first;
      TestArgs args = entry.second;

      auto l = Length::TryParse (input.first, input.second);

      AssertTrue (l.has_value (), "TryParse returned false when expecting true");

      std::stringstream stream;
      stream << "Parsing input (" << input.first << ", " << input.second
             << ") returned the wrong value";

      NS_TEST_ASSERT_MSG_EQ_TOL (l->GetDouble (), args.first, args.second, stream.str ());
    }

}

void
LengthTestCase::TestCopyAssignment ()
{
  const double value = 5;

  Length original (value, Unit::Meter);

  Length copy;
  copy = original;

  NS_TEST_ASSERT_MSG_EQ (copy.GetDouble (), original.GetDouble (),
                         "copy assignment failed");
}

void
LengthTestCase::TestMoveAssignment ()
{
  const double value = 5;

  Length original (value, Unit::Meter);

  Length copy;
  copy = std::move (original);

  NS_TEST_ASSERT_MSG_EQ (copy.GetDouble (), value,
                         "move assignment failed");
}

void
LengthTestCase::TestQuantityAssignment ()
{
  Length::Quantity input (5, Unit::Kilometer);

  Length l;
  Length expected (input);

  l = input;

  NS_TEST_ASSERT_MSG_EQ (l, expected,
                         "quantity assignment failed");
}

void
LengthTestCase::TestIsEqualReturnsTrue ()
{
  const double value = 5;
  Length one (value, Unit::Meter);
  Length two (one);

  AssertTrue (one.IsEqual (two), "IsEqual returned false for equal lengths");
}

void
LengthTestCase::TestIsEqualReturnsFalse ()
{
  const double value = 5;
  Length one (value, Unit::Meter);
  Length two ( value, Unit::Foot );

  AssertFalse (one.IsEqual (two), "IsEqual returned true for unequal lengths");
}

void
LengthTestCase::TestIsEqualWithToleranceReturnsTrue ()
{
  const double value = 5;
  const double tolerance = 0.1;

  Length one (value, Unit::Meter);
  Length two ( (value + 0.1), Unit::Meter);

  AssertTrue (one.IsEqual (two, tolerance),
              "IsEqual returned false for almost equal lengths");
}

void
LengthTestCase::TestIsEqualWithToleranceReturnsFalse ()
{
  const double value = 5;
  const double tolerance = 0.01;

  Length one (value, Unit::Meter);
  Length two ( (value + 0.1), Unit::Meter);

  AssertFalse (one.IsEqual (two, tolerance),
               "IsEqual returned true for almost equal lengths");
}

void
LengthTestCase::TestIsNotEqualReturnsTrue ()
{
  const double value = 5;

  Length one (value, Unit::Meter);
  Length two ( (value + 0.1), Unit::Meter);

  AssertTrue (one.IsNotEqual (two),
              "IsNotEqual returned false for not equal lengths");
}

void
LengthTestCase::TestIsNotEqualReturnsFalse ()
{
  const double value = 5;

  Length one (value, Unit::Meter);
  Length two ( one );

  AssertFalse (one.IsNotEqual (two),
               "IsNotEqual returned true for equal lengths");
}

void
LengthTestCase::TestIsNotEqualWithToleranceReturnsTrue ()
{
  const double tolerance = 0.001;

  Length one ( 5.01, Unit::Meter);
  Length two ( 5.02, Unit::Meter);

  AssertTrue (one.IsNotEqual (two, tolerance),
              "IsNotEqual with tolerance returned false for not equal lengths");
}

void
LengthTestCase::TestIsNotEqualWithToleranceReturnsFalse ()
{
  const double tolerance = 0.01;

  Length one ( 5.01, Unit::Meter);
  Length two ( 5.02, Unit::Meter);

  AssertFalse (one.IsNotEqual (two, tolerance),
               "IsNotEqual with tolerance returned true for not equal lengths");
}

void
LengthTestCase::TestIsLessReturnsTrue ()
{
  const double value = 5;

  Length one (value, Unit::Meter);
  Length two ( (value + 0.1), Unit::Meter);

  AssertTrue (one.IsLess (two),
              "IsLess returned false for non equal lengths");
}

void
LengthTestCase::TestIsLessReturnsFalse ()
{
  const double value = 5;

  Length one (value, Unit::Meter);
  Length two ( one );

  AssertFalse (one.IsLess (two),
               "IsLess returned true for equal lengths");
}

void
LengthTestCase::TestIsLessWithToleranceReturnsFalse ()
{
  const double tolerance = 0.01;

  Length one ( 5.1234, Unit::Meter );
  Length two ( 5.1278, Unit::Meter );

  AssertFalse (one.IsLess (two, tolerance),
               "IsLess with tolerance returned true");
}

void
LengthTestCase::TestIsGreaterReturnsTrue ()
{
  Length one (2.0, Unit::Meter);
  Length two (1.0, Unit::Meter);

  AssertTrue (one.IsGreater (two),
              "IsGreater returned false");
}

void
LengthTestCase::TestIsGreaterReturnsFalse ()
{
  Length one (2.0, Unit::Meter);
  Length two (1.0, Unit::Meter);

  AssertFalse (two.IsGreater (one),
               "IsGreater returned true");
}

void
LengthTestCase::TestIsGreaterWithToleranceReturnsFalse ()
{
  const double tolerance = 0.01;

  Length one (5.1234, Unit::Meter);
  Length two (5.1278, Unit::Meter);

  AssertFalse (two.IsGreater (one, tolerance),
               "IsGreater returned true");
}

void
LengthTestCase::TestOutputStreamOperator ()
{
  Length l (1.0, Unit::Meter);

  std::stringstream stream;

  stream << l;

  NS_TEST_ASSERT_MSG_EQ (stream.str (), "1 m",
                         "unexpected output from operator<<");
}

void
LengthTestCase::TestInputStreamOperator ()
{
  const double value = 5;

  Length l;

  std::stringstream stream;

  stream << value << "m";

  stream >> l;

  NS_TEST_ASSERT_MSG_EQ (l.GetDouble (), value,
                         "unexpected length from operator>>");
}

template<class T>
void
LengthTestCase::TestLengthSerialization (const Length& l,
                                         const T& unit,
                                         const std::string& expectedOutput,
                                         const std::string& context)
{
  const std::string msg = context + ": unexpected output when serializing length";

  std::ostringstream stream;

  stream << std::fixed
         << std::setprecision (5)
         << l.As (unit);

  NS_TEST_ASSERT_MSG_EQ (stream.str (), expectedOutput, msg);
}

void
LengthTestCase::TestSerializeLengthWithUnit ()
{
  Length l (1.0, Unit::Meter);

  TestLengthSerialization (l, Unit::Nanometer, "1000000000.00000 nm", "nanometers");
  TestLengthSerialization (l, Unit::Micrometer, "1000000.00000 um", "micrometers");
  TestLengthSerialization (l, Unit::Millimeter, "1000.00000 mm", "millimeters");
  TestLengthSerialization (l, Unit::Centimeter, "100.00000 cm", "centimeters");
  TestLengthSerialization (l, Unit::Meter, "1.00000 m", "meters");
  TestLengthSerialization (l, Unit::Kilometer, "0.00100 km", "kilometers");
  TestLengthSerialization (l, Unit::NauticalMile, "0.00054 nmi", "nautical_mile");
  TestLengthSerialization (l, Unit::Inch, "39.37008 in", "inches");
  TestLengthSerialization (l, Unit::Foot, "3.28084 ft", "feet");
  TestLengthSerialization (l, Unit::Yard, "1.09361 yd", "yards");
  TestLengthSerialization (l, Unit::Mile, "0.00062 mi", "miles");
}

void
LengthTestCase::TestOperatorEqualsReturnsTrue ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Meter );

  AssertTrue ( one == two,
               "operator== returned false for equal lengths");
}

void
LengthTestCase::TestOperatorEqualsReturnsFalse ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer );

  AssertFalse ( one == two,
                "operator== returned true for non equal lengths");
}

void
LengthTestCase::TestOperatorNotEqualsReturnsTrue ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);

  AssertTrue ( one != two,
               "operator!= returned false for non equal lengths");

}

void
LengthTestCase::TestOperatorNotEqualsReturnsFalse ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Meter );

  AssertFalse ( one != two,
                "operator!= returned true for equal lengths");

}

void
LengthTestCase::TestOperatorLessThanReturnsTrue ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);

  AssertTrue ( one < two,
               "operator< returned false for smaller length");
}

void
LengthTestCase::TestOperatorLessThanReturnsFalse ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);

  AssertFalse ( two < one,
                "operator< returned true for larger length");
}

void
LengthTestCase::TestOperatorLessOrEqualReturnsTrue ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);
  Length three ( one );

  AssertTrue ( one <= two,
               "operator<= returned false for smaller length");

  AssertTrue ( one <= three,
               "operator<= returned false for equal lengths");
}

void
LengthTestCase::TestOperatorLessOrEqualReturnsFalse ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);

  AssertFalse ( two <= one,
                "operator<= returned true for larger length");
}

void
LengthTestCase::TestOperatorGreaterThanReturnsTrue ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);

  AssertTrue ( two > one,
               "operator> returned false for larger length");
}

void
LengthTestCase::TestOperatorGreaterThanReturnsFalse ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);

  AssertFalse ( one > two,
                "operator> returned true for smaller length");
}

void
LengthTestCase::TestOperatorGreaterOrEqualReturnsTrue ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);
  Length three ( one );

  AssertTrue ( two >= one,
               "operator>= returned false for larger length");

  AssertTrue ( one >= three,
               "operator>= returned false for equal lengths");
}

void
LengthTestCase::TestOperatorGreaterOrEqualReturnsFalse ()
{
  const double value = 5;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Kilometer);

  AssertFalse ( one >= two,
                "operator>= returned true for smaller length");
}

void
LengthTestCase::TestAddingTwoLengths ()
{
  const double value = 1;
  const double expectedOutput = 2;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Meter );

  Length result = one + two;

  NS_TEST_ASSERT_MSG_EQ ( one.GetDouble (), value,
                          "operator+ modified first operand");
  NS_TEST_ASSERT_MSG_EQ ( two.GetDouble (), value,
                          "operator+ modified second operand");
  NS_TEST_ASSERT_MSG_EQ ( result.GetDouble (), expectedOutput,
                          "operator+ returned incorrect value");
}

void
LengthTestCase::TestAddingLengthAndQuantity ()
{
  const double value = 1;
  const double expectedOutput = 2;

  Length one ( value, Unit::Meter );

  Length result = one + Length::Quantity (value, Unit::Meter);

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator+ modified first operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator+ returned incorrect value");
}

void
LengthTestCase::TestAddingQuantityAndLength ()
{
  const double value = 1;
  const double expectedOutput = 2;

  Length one ( value, Unit::Meter );

  Length result = Length::Quantity (value, Unit::Meter) + one;

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator+ modified first operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator+ returned incorrect value");
}

void
LengthTestCase::TestSubtractingTwoLengths ()
{
  const double value = 1;
  const double expectedOutput = 0;

  Length one ( value, Unit::Meter );
  Length two ( value, Unit::Meter );

  Length result = one - two;

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator- modified first operand");
  NS_TEST_ASSERT_MSG_EQ (two.GetDouble (), value,
                         "operator- modified second operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator- returned incorrect value");
}

void
LengthTestCase::TestSubtractingLengthAndQuantity ()
{
  const double value = 1;
  const double expectedOutput = 0;

  Length one ( value, Unit::Meter );

  Length result = one - Length::Quantity ( value, Unit::Meter);

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator- modified first operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator- returned incorrect value");
}

void
LengthTestCase::TestSubtractingQuantityAndLength ()
{
  const double value = 1;
  const double expectedOutput = 0;

  Length one ( value, Unit::Meter );

  Length result = Length::Quantity (value, Unit::Meter) - one;

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator- modified second operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator- returned incorrect value");
}

void
LengthTestCase::TestMultiplyLengthByScalar ()
{
  const double value = 1;
  const double scalar = 5;
  const double expectedOutput = value * scalar;

  Length one ( value, Unit::Meter );
  Length result = one * scalar;

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator* modified first operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator* returned incorrect value");
}

void
LengthTestCase::TestMultiplyScalarByLength ()
{
  const double value = 1;
  const double scalar = 5;
  const double expectedOutput = value * scalar;

  Length one ( value, Unit::Meter );
  Length result = scalar * one;

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator* modified second operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator* returned incorrect value");
}

void
LengthTestCase::TestDivideLengthByScalar ()
{
  const double value = 10;
  const double scalar = 5;
  const double expectedOutput = value / scalar;

  Length one ( value, Unit::Meter );
  Length result = one / scalar;

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), value,
                         "operator/ modified first operand");
  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedOutput,
                         "operator/ returned incorrect value");
}

void
LengthTestCase::TestDivideLengthByLength ()
{
  const double valueOne = 100;
  const double valueTwo = 2;
  const double expectedOutput = valueOne / valueTwo;

  Length one ( valueOne, Unit::Meter );
  Length two ( valueTwo, Unit::Meter );

  double result = one / two;

  NS_TEST_ASSERT_MSG_EQ (one.GetDouble (), valueOne,
                         "operator/ modified first operand");
  NS_TEST_ASSERT_MSG_EQ (two.GetDouble (), valueTwo,
                         "operator/ modified second operand");
  NS_TEST_ASSERT_MSG_EQ (result, expectedOutput,
                         "operator/ returned incorrect value");

}

void
LengthTestCase::TestDivideLengthByLengthReturnsNaN ()
{
  const double value = 1;

  Length one ( value, Unit::Meter );
  Length two;

  double result = one / two;

  AssertTrue ( std::isnan (result),
               "operator/ did not return NaN when dividing by zero");
}

void
LengthTestCase::TestDivReturnsCorrectResult ()
{
  const double topValue = 100;
  const double bottomValue = 20;
  const int64_t expectedOutput = 5;

  Length numerator (topValue, Unit::Meter);
  Length denominator (bottomValue, Unit::Meter);

  auto result = Div (numerator, denominator);

  NS_TEST_ASSERT_MSG_EQ (result, expectedOutput,
                         "Div() returned an incorrect value");
}

void
LengthTestCase::TestDivReturnsZeroRemainder ()
{
  const double topValue = 100;
  const double bottomValue = 20;
  const int64_t expectedOutput = 5;
  const int64_t expectedRemainder = 0;

  Length numerator (topValue, Unit::Meter);
  Length denominator (bottomValue, Unit::Meter);
  Length remainder;

  auto result = Div (numerator, denominator, &remainder);

  NS_TEST_ASSERT_MSG_EQ (result, expectedOutput,
                         "Div() returned an incorrect value");
  NS_TEST_ASSERT_MSG_EQ (remainder.GetDouble (), expectedRemainder,
                         "Div() returned an incorrect remainder");
}

void
LengthTestCase::TestDivReturnsCorrectRemainder ()
{
  const double topValue = 110;
  const double bottomValue = 20;
  const int64_t expectedOutput = 5;
  const int64_t expectedRemainder = 10;

  Length numerator (topValue, Unit::Meter);
  Length denominator (bottomValue, Unit::Meter);
  Length remainder;

  auto result = Div (numerator, denominator, &remainder);

  NS_TEST_ASSERT_MSG_EQ (result, expectedOutput,
                         "Div() returned an incorrect value");
  NS_TEST_ASSERT_MSG_EQ (remainder.GetDouble (), expectedRemainder,
                         "Div() returned an incorrect remainder");
}

void
LengthTestCase::TestModReturnsZero ()
{
  Length numerator (10, Unit::Meter);
  Length denominator (2, Unit::Meter);

  auto result = Mod (numerator, denominator);

  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), 0,
                         "Mod() returned a non zero value");
}

void
LengthTestCase::TestModReturnsNonZero ()
{
  Length numerator (14, Unit::Meter);
  Length denominator (3, Unit::Meter);
  const double expectedValue = 2;

  auto result = Mod (numerator, denominator);

  NS_TEST_ASSERT_MSG_EQ (result.GetDouble (), expectedValue,
                         "Mod() returned the wrong value");
}

void
LengthTestCase::DoRun ()
{
  TestDefaultLengthIsZero ();

  TestConstructLengthFromQuantity ();

  TestConstructLengthFromSIUnits ();

  TestConstructLengthFromUSUnits ();

  TestLengthCopyConstructor ();

  TestLengthMoveConstructor ();

  TestConstructLengthFromMeterString ();
  TestConstructLengthFromNanoMeterString ();
  TestConstructLengthFromMicroMeterString ();
  TestConstructLengthFromMilliMeterString ();
  TestConstructLengthFromCentiMeterString ();
  TestConstructLengthFromKiloMeterString ();
  TestConstructLengthFromNauticalMileString ();
  TestConstructLengthFromInchString ();
  TestConstructLengthFromFootString ();
  TestConstructLengthFromYardString ();
  TestConstructLengthFromMileString ();

#ifdef HAVE_BOOST_UNITS
  TestConstructLengthFromBoostUnits ();
#endif

  TestBuilderFreeFunctions ();

  TestTryParseReturnsFalse ();
  TestTryParseReturnsTrue ();

  TestCopyAssignment ();
  TestMoveAssignment ();
  TestQuantityAssignment ();

  TestIsEqualReturnsTrue ();
  TestIsEqualReturnsFalse ();
  TestIsEqualWithToleranceReturnsTrue ();
  TestIsEqualWithToleranceReturnsFalse ();
  TestIsNotEqualReturnsTrue ();
  TestIsNotEqualReturnsFalse ();
  TestIsNotEqualWithToleranceReturnsTrue ();
  TestIsNotEqualWithToleranceReturnsFalse ();
  TestIsLessReturnsTrue ();
  TestIsLessReturnsFalse ();
  TestIsLessWithToleranceReturnsFalse ();
  TestIsGreaterReturnsTrue ();
  TestIsGreaterReturnsFalse ();
  TestIsGreaterWithToleranceReturnsFalse ();

  TestOutputStreamOperator ();

  TestSerializeLengthWithUnit ();

  TestOperatorEqualsReturnsTrue ();
  TestOperatorEqualsReturnsFalse ();
  TestOperatorNotEqualsReturnsTrue ();
  TestOperatorNotEqualsReturnsFalse ();
  TestOperatorLessThanReturnsTrue ();
  TestOperatorLessThanReturnsFalse ();
  TestOperatorLessOrEqualReturnsTrue ();
  TestOperatorLessOrEqualReturnsFalse ();
  TestOperatorGreaterThanReturnsTrue ();
  TestOperatorGreaterThanReturnsFalse ();
  TestOperatorGreaterOrEqualReturnsTrue ();
  TestOperatorGreaterOrEqualReturnsFalse ();

  TestAddingTwoLengths ();
  TestAddingLengthAndQuantity ();
  TestAddingQuantityAndLength ();
  TestSubtractingTwoLengths ();
  TestSubtractingLengthAndQuantity ();
  TestSubtractingQuantityAndLength ();
  TestMultiplyLengthByScalar ();
  TestMultiplyScalarByLength ();
  TestDivideLengthByScalar ();
  TestDivideLengthByLength ();
  TestDivideLengthByLengthReturnsNaN ();

  TestDivReturnsCorrectResult ();
  TestDivReturnsZeroRemainder ();
  TestDivReturnsCorrectRemainder ();

  TestModReturnsZero ();
  TestModReturnsNonZero ();
}

/**
 * \ingroup length-tests
 *
 * Test case for LengthValue attribute
 */
class LengthValueTestCase : public TestCase
{
public:
  /**
   * Default Constructor
   */
  LengthValueTestCase ()
    :   TestCase ("length-value-tests")
  {}

  /**
   * Destructor
   */
  virtual ~LengthValueTestCase ()
  {}

private:
    /// Class with Length attribute
    class TestObject : public Object
    {
    public:
        /**
         * \brief Get the type ID.
         * \return The object TypeId.
         */
        static TypeId GetTypeId ();

        TestObject ()
            :   m_length ()
        {}

        virtual ~TestObject ()
        {}

    private:
        Length m_length; //!< Length object
    };

private:
    /**
     * Test that a LengthValue can be constructed from a Length instance
     */
    void TestAttributeConstructor ();

    /**
     * Test that a LengthValue can be serialized to a string
     */
    void TestAttributeSerialization ();

    /**
     * Test that a LengthValue can be deserialized from a string
     */
    void TestAttributeDeserialization ();

    /**
     * Test that a LengthValue works as an attribute
     */
    void TestObjectAttribute ();

    /**
     * Test that a StringValue is converted to LengthValue
     */
    void TestSetAttributeUsingStringValue ();

    // Inherited function
    virtual void DoRun ();
};

TypeId
LengthValueTestCase::TestObject::GetTypeId ()
{
    static TypeId tid = TypeId ("LengthValueTestCase::TestObject")
        .SetParent<Object> ()
        .SetGroupName ("Test")
        .AddConstructor<TestObject> ()
        .AddAttribute ("Length",
                        "Length value",
                        LengthValue (),
                        MakeLengthAccessor (&TestObject::m_length),
                        MakeLengthChecker ())
        ;

    return tid;
}

void
LengthValueTestCase::TestAttributeConstructor ()
{
  Length l = KiloMeters (2);
  LengthValue value (l);

  NS_TEST_ASSERT_MSG_EQ (value.Get (), l, "Length attribute has wrong value");
}

void
LengthValueTestCase::TestAttributeSerialization ()
{
  Ptr<const AttributeChecker> checker = MakeLengthChecker ();

  Length l = KiloMeters (2);
  LengthValue value (l);

  std::string output = value.SerializeToString (checker);

  NS_TEST_ASSERT_MSG_EQ (output, "2000 m",
                         "Length attribute serialization has wrong output");
}

void
LengthValueTestCase::TestAttributeDeserialization ()
{
  Ptr<const AttributeChecker> checker = MakeLengthChecker ();

  Length l = KiloMeters (2);
  std::ostringstream stream;
  stream << l;

  LengthValue value;
  bool result = value.DeserializeFromString (stream.str (), checker);

  NS_TEST_ASSERT_MSG_EQ (result, true,
                         "Length attribute deserialization failed");
  NS_TEST_ASSERT_MSG_EQ (value.Get (), l,
                         "Length attribute has wrong value after deserialization");
}

void
LengthValueTestCase::TestObjectAttribute ()
{
    Length expected (5, Unit::Kilometer);
    Ptr<TestObject> obj = CreateObject<TestObject> ();

    obj->SetAttribute ("Length", LengthValue (expected));

    LengthValue val;
    obj->GetAttribute ("Length", val);

    NS_TEST_ASSERT_MSG_EQ (val.Get (), expected,
                            "Length attribute does not have expected value");
}

void
LengthValueTestCase::TestSetAttributeUsingStringValue ()
{
    Length expected (5, Unit::Kilometer);
    Ptr<TestObject> obj = CreateObject<TestObject> ();

    std::stringstream stream;
    stream << expected.As (Unit::Kilometer);

    obj->SetAttribute ("Length", StringValue (stream.str()));

    LengthValue val;
    obj->GetAttribute ("Length", val);

    NS_TEST_ASSERT_MSG_EQ (val.Get (), expected,
                            "Length attribute does not have expected value");
}

void
LengthValueTestCase::DoRun ()
{
    TestAttributeConstructor ();
    TestAttributeSerialization ();
    TestAttributeDeserialization ();
    TestObjectAttribute ();
    TestSetAttributeUsingStringValue ();
}

/**
 * \ingroup length-tests
 * The Test Suite that runs the test case
 */
class LengthTestSuite : public TestSuite
{
public:
  /**
   * Default Constructor
   */
  LengthTestSuite ();
};

LengthTestSuite::LengthTestSuite ()
  : TestSuite ("length")
{
  AddTestCase ( new LengthTestCase (), TestCase::QUICK );
  AddTestCase ( new LengthValueTestCase (), TestCase::QUICK );
}

/**
 * LengthTestSuite instance
 */
static LengthTestSuite gLengthTestSuite;

