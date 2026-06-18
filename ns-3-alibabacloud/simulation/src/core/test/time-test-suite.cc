/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
 * Copyright (c) 2007 Emmanuelle Laprise
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
 * TimeStep support by Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <tuple>

#include "ns3/nstime.h"
#include "ns3/int64x64.h"
#include "ns3/test.h"

using namespace ns3;
/**
 * \ingroup core-tests
 * \brief time simple test case, Checks the basic operations on time
 */
class TimeSimpleTestCase : public TestCase
{
public:
  /**
   * \brief constructor for TimeSimpleTestCase.
   */
  TimeSimpleTestCase ();

private:
  /**
   * \brief setup function for TimeSimpleTestCase.
   */
  virtual void DoSetup (void);

  /**
   * \brief Runs the Simple Time test case.
   */
  virtual void DoRun (void);

  /**
   * \brief Tests the Time Operations.
   */
  virtual void DoTimeOperations (void);

  /**
   * \brief Does the tear down for TimeSimpleTestCase.
   */
  virtual void DoTeardown (void);

  /**
   * Helper function to handle boilerplate code for multiplication tests
   *
   * \tparam T type of multiplication value
   *  
   * \param t Time value to multiply
   * \param expected Expected result of the multiplication
   * \param val Value to multiply by
   * \param msg Error message to print if test fails
   */
  template<typename T>
  void TestMultiplication (Time t, Time expected, T val, const std::string& msg); 

  /**
   * Test multiplying a Time instance by various integer types
   */
  void TestMultiplicationByIntegerTypes ();

  /**
   * Test multiplying a Time instance by various decimal types
   */
  void TestMultiplicationByDecimalTypes ();

  /**
   * Helper function to handle boilerplate code for division tests
   *
   * \tparam T type of division value
   *
   * \param t Time value to divide 
   * \param expected Expected result of the divsion 
   * \param val Value to divide by
   * \param msg Error message to print if test fails
   */
  template<typename T>
  void TestDivision(Time t, Time expected, T val, const std::string& msg);

  /**
   * Test dividing a Time instance by various integer types
   */
  void TestDivisionByIntegerTypes ();

  /**
   * Test dividing a Time instance by various decimal types
   */
  void TestDivisionByDecimalTypes ();
};

TimeSimpleTestCase::TimeSimpleTestCase ()
  : TestCase ("Sanity check of common time operations")
{}

void
TimeSimpleTestCase::DoSetup (void)
{}

void TimeSimpleTestCase::DoTimeOperations (void)
{
  // Test Multiplication
  constexpr long long oneSec =  1000000000;  // conversion to default nanoseconds

  // Time in seconds
  ns3::Time t1 = Time (125LL * oneSec);
  ns3::Time t2 = Time (2000LL * oneSec);

  std::cout << "Testing Time Subtraction \n";

  NS_TEST_ASSERT_MSG_EQ ( (t2 - t1).GetSeconds (), 1875, "Time Subtraction");

  // Test Multiplication Operations:
  std::cout << "Testing Time Multiplication \n";

  TestMultiplicationByIntegerTypes ();
  TestMultiplicationByDecimalTypes ();

  // Test Division Operations:
  std::cout << "Testing Time Division \n";

  TestDivisionByIntegerTypes ();
  TestDivisionByDecimalTypes ();

  std::cout << "Testing modulo division \n";

  t1 = Time (101LL * oneSec);
  NS_TEST_ASSERT_MSG_EQ ( (t2 % t1).GetSeconds (), 81, "Remainder Operation (2000 % 101 = 81)" );
  NS_TEST_ASSERT_MSG_EQ ( Div (t2,t1), 19, "Modular Divison");
  NS_TEST_ASSERT_MSG_EQ ( Rem (t2,t1).GetSeconds (), 81, "Remainder Operation (2000 % 101 = 81)" );

}

void
TimeSimpleTestCase::DoRun (void)
{
  NS_TEST_ASSERT_MSG_EQ_TOL (Years (1.0).GetYears (), 1.0, Years (1).GetYears (),
                             "is 1 really 1 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Years (10.0).GetYears (), 10.0, Years (1).GetYears (),
                             "is 10 really 10 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Days (1.0).GetDays (), 1.0, Days (1).GetDays (),
                             "is 1 really 1 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Days (10.0).GetDays (), 10.0, Days (1).GetDays (),
                             "is 10 really 10 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Hours (1.0).GetHours (), 1.0, Hours (1).GetHours (),
                             "is 1 really 1 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Hours (10.0).GetHours (), 10.0, Hours (1).GetHours (),
                             "is 10 really 10 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Minutes (1.0).GetMinutes (), 1.0, Minutes (1).GetMinutes (),
                             "is 1 really 1 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Minutes (10.0).GetMinutes (), 10.0, Minutes (1).GetMinutes (),
                             "is 10 really 10 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Seconds (1.0).GetSeconds (), 1.0, TimeStep (1).GetSeconds (),
                             "is 1 really 1 ?");
  NS_TEST_ASSERT_MSG_EQ_TOL (Seconds (10.0).GetSeconds (), 10.0, TimeStep (1).GetSeconds (),
                             "is 10 really 10 ?");
  NS_TEST_ASSERT_MSG_EQ (MilliSeconds (1).GetMilliSeconds (), 1,
                         "is 1ms really 1ms ?");
  NS_TEST_ASSERT_MSG_EQ (MicroSeconds (1).GetMicroSeconds (), 1,
                         "is 1us really 1us ?");

  DoTimeOperations ();

#if 0
  Time ns = NanoSeconds (1);
  ns.GetNanoSeconds ();
  NS_TEST_ASSERT_MSG_EQ (NanoSeconds (1).GetNanoSeconds (), 1,
                         "is 1ns really 1ns ?");
  NS_TEST_ASSERT_MSG_EQ (PicoSeconds (1).GetPicoSeconds (), 1,
                         "is 1ps really 1ps ?");
  NS_TEST_ASSERT_MSG_EQ (FemtoSeconds (1).GetFemtoSeconds (), 1,
                         "is 1fs really 1fs ?");
#endif

  Time ten = NanoSeconds (10);
  int64_t tenValue = ten.GetInteger ();
  Time::SetResolution (Time::PS);
  int64_t tenKValue = ten.GetInteger ();
  NS_TEST_ASSERT_MSG_EQ (tenValue * 1000, tenKValue,
                         "change resolution to PS");
}

void
TimeSimpleTestCase::DoTeardown (void)
{}

template<typename T>
void 
TimeSimpleTestCase::TestMultiplication (Time t, Time expected, T val, 
                                       const std::string& msg) 
                      
{
  using TestEntry = std::tuple<Time, std::string>;
  std::array<TestEntry, 2> TESTS{ 
      std::make_tuple (t * val, "Test Time * value: "),
      std::make_tuple (val * t, "Test Time * value: ")
  };
  
  for (auto test: TESTS )
  {
    std::string errMsg = std::get<1> (test) + msg;
  
    NS_TEST_ASSERT_MSG_EQ (std::get<0> (test), expected, errMsg);
  }
}

void
TimeSimpleTestCase::TestMultiplicationByIntegerTypes ()
{
  int sec = 125;
  int scale = 100; 

  Time t = Seconds (sec);
  Time expected = Time (t.GetTimeStep () * scale); 

  TestMultiplication (t, expected, static_cast<char> (scale), "Multiplication by char");
  TestMultiplication (t, expected, static_cast<unsigned char> (scale), 
                     "Multiplication by unsigned char");
  TestMultiplication (t, expected, static_cast<short> (scale), "Multiplication by short");
  TestMultiplication (t, expected, static_cast<unsigned short> (scale), 
                     "Multiplication by unsigned short");
  TestMultiplication (t, expected, static_cast<int> (scale), "Multiplication by int");
  TestMultiplication (t, expected, static_cast<unsigned int> (scale), 
                     "Multiplication by unsigned int");
  TestMultiplication (t, expected, static_cast<long> (scale), "Multiplication by long");
  TestMultiplication (t, expected, static_cast<unsigned long> (scale), 
                     "Multiplication by unsigned long");
  TestMultiplication (t, expected, static_cast<long long> (scale), 
                     "Multiplication by long long");
  TestMultiplication (t, expected, static_cast<unsigned long long> (scale), 
                     "Multiplication by unsigned long long");
  TestMultiplication (t, expected, static_cast<std::size_t> (scale), 
                     "Multiplication by size_t");

  int64x64_t scale64 = 100;
  TestMultiplication (t, expected, scale64, "Multiplication by int64x64_t");
}

void
TimeSimpleTestCase::TestMultiplicationByDecimalTypes ()
{
  float sec = 150.0;
  float scale = 100.2; 

  Time t = Seconds (sec);
  Time expected = Time (t.GetDouble () * scale); 

  TestMultiplication (t, expected, scale, "Multiplication by float");
  TestMultiplication (t, expected, static_cast<double> (scale), 
                     "Multiplication by double");
}

template<typename T>
void 
TimeSimpleTestCase::TestDivision (Time t, Time expected, T val, const std::string& msg)
{
  Time result = t / val;

  NS_TEST_ASSERT_MSG_EQ (result, expected, msg);
}

void 
TimeSimpleTestCase::TestDivisionByIntegerTypes()
{
  int sec = 2000;
  int scale = 100; 

  Time t = Seconds (sec);
  Time expected = Time (t.GetTimeStep () / scale);

  TestDivision (t, expected, static_cast<char> (scale), "Division by char");
  TestDivision (t, expected, static_cast<unsigned char> (scale), 
                     "Division by unsigned char");
  TestDivision (t, expected, static_cast<short> (scale), "Division by short");
  TestDivision (t, expected, static_cast<unsigned short> (scale), 
                     "Division by unsigned short");
  TestDivision (t, expected, static_cast<int> (scale), "Division by int");
  TestDivision (t, expected, static_cast<unsigned int> (scale), 
                     "Division by unsigned int");
  TestDivision (t, expected, static_cast<long> (scale), "Division by long");
  TestDivision (t, expected, static_cast<unsigned long> (scale), 
                     "Division by unsigned long");
  TestDivision (t, expected, static_cast<long long> (scale), 
                     "Division by long long");
  TestDivision (t, expected, static_cast<unsigned long long> (scale), 
                     "Division by unsigned long long");
  TestDivision (t, expected, static_cast<std::size_t> (scale), 
                     "Division by size_t");

  int64x64_t scale64 = 100;
  TestDivision (t, expected, scale64, "Division by int64x64_t");
}

void 
TimeSimpleTestCase::TestDivisionByDecimalTypes ()
{
  float sec = 200.0;
  float scale = 0.2; 

  Time t = Seconds (sec);
  Time expected = t / int64x64_t (scale);

  TestDivision (t, expected, scale, "Division by float");
  TestDivision (t, expected, static_cast<double> (scale), 
                     "Division by double");
}


/**
 * \ingroup core-tests
 * \brief  time-tests Time with Sign test case
 */
class TimeWithSignTestCase : public TestCase
{
public:
  /**
   * \brief constructor for TimeWithSignTestCase.
   */
  TimeWithSignTestCase ();

private:
  /**
   * \brief DoSetup for TimeWithSignTestCase.
   */
  virtual void DoSetup (void);

  /**
   * \brief DoRun for TimeWithSignTestCase.
   */
  virtual void DoRun (void);

  /**
   * \brief DoTeardown for TimeWithSignTestCase.
   */
  virtual void DoTeardown (void);
};

TimeWithSignTestCase::TimeWithSignTestCase ()
  : TestCase ("Checks times that have plus or minus signs")
{}

void
TimeWithSignTestCase::DoSetup (void)
{}

void
TimeWithSignTestCase::DoRun (void)
{
  Time timePositive          ("+1000.0");
  Time timePositiveWithUnits ("+1000.0ms");

  Time timeNegative          ("-1000.0");
  Time timeNegativeWithUnits ("-1000.0ms");

  NS_TEST_ASSERT_MSG_EQ_TOL (timePositive.GetSeconds (),
                             +1000.0,
                             1.0e-8,
                             "Positive time not parsed correctly.");

  NS_TEST_ASSERT_MSG_EQ_TOL (timePositiveWithUnits.GetSeconds (),
                             +1.0,
                             1.0e-8,
                             "Positive time with units not parsed correctly.");

  NS_TEST_ASSERT_MSG_EQ_TOL (timeNegative.GetSeconds (),
                             -1000.0,
                             1.0e-8,
                             "Negative time not parsed correctly.");

  NS_TEST_ASSERT_MSG_EQ_TOL (timeNegativeWithUnits.GetSeconds (),
                             -1.0,
                             1.0e-8,
                             "Negative time with units not parsed correctly.");
}


void
TimeWithSignTestCase::DoTeardown (void)
{}

/**
 * \ingroup core-tests
 * \brief Input output Test Case for Time
 */
class TimeInputOutputTestCase : public TestCase
{
public:
  /**
   * \brief Constructor for TimeInputOutputTestCase.
   */
  TimeInputOutputTestCase ();

private:
  /**
   * \brief DoRun for TimeInputOutputTestCase.
   */ 
  virtual void DoRun (void);
  /**
   * \brief Check roundtrip from/to string.
   * \param str Time input check.
   */
  void Check (const std::string & str);

  /**
   * \brief Check autoscaling output using Time::As()
   * \param t Time instance.
   * \param expect Expected string output with Time::As() autoscaling.
   */
  void CheckAs (const Time t, const std::string expect);
};

TimeInputOutputTestCase::TimeInputOutputTestCase ()
  : TestCase ("Input,output from,to strings")
{}

void
TimeInputOutputTestCase::Check (const std::string & str)
{
  std::stringstream ss (str);
  Time time;
  ss >> time;
  ss << time;
  bool pass = (str == ss.str ());

  std::cout << GetParent ()->GetName () << " InputOutput:    "
            << (pass ? "pass " : "FAIL ")
            << "\"" << str << "\"";
  if (!pass)
    {
      std::cout << ", got " << ss.str ();
    }
  std::cout << std::endl;
  NS_TEST_EXPECT_MSG_EQ (ss.str (), str, "round trip conversion from/to string");
}

void
TimeInputOutputTestCase::CheckAs (const Time t, const std::string expect)
{
  std::stringstream ss;
  ss << std::fixed << std::setprecision (6) << t.As ();
  std::string str;
  ss >> str;
  bool pass = (str == expect);

  std::cout << GetParent ()->GetName () << " InputOutput:As: "
            << (pass ? "pass " : "FAIL ")
            << "\"" << expect << "\"";
  if (!pass)
    {
      std::cout << ", got " << str;
    }
  std::cout << std::endl;
  NS_TEST_EXPECT_MSG_EQ (str, expect, "Time::As() autoscaling");

}

void
TimeInputOutputTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " InputOutput: " << GetName ()
            << std::endl;

  Check ("2ns");
  Check ("+3.1us");
  Check ("-4.2ms");
  Check ("5.3s");
  Check ("6.4min");
  Check ("7.5h");
  Check ("8.6d");
  Check ("10.8y");

  Time t (3.141592654e9);  // Pi seconds

  std::cout << GetParent ()->GetName () << " InputOutput: "
            << "example: raw:   " << t
            << std::endl;

  std::cout << GetParent ()->GetName () << " InputOutput: "
            << std::fixed << std::setprecision (9)
            << "example: in s:  " << t.As (Time::S)
            << std::endl;

  std::cout << GetParent ()->GetName () << " InputOutput: "
            << std::setprecision (6)
            << "example: in ms: " << t.As (Time::MS)
            << std::endl;

  std::cout << GetParent ()->GetName () << " InputOutput: "
            << "example: Get ns: " << t.GetNanoSeconds ()
            << std::endl;

  std::cout << GetParent ()->GetName () << " InputOutput: "
            << "example: auto scale: \n";
  CheckAs (t * 1e-9, "+3.000000ns");
  CheckAs (t * 1e-8, "+31.000000ns");
  CheckAs (t * 1e-7, "+314.000000ns");
  CheckAs (t * 1e-6, "+3.142000us");
  CheckAs (t * 1e-5, "+31.416000us");
  CheckAs (t * 1e-4, "+314.159000us");
  CheckAs (t * 1e-3, "+3.141593ms");
  CheckAs (t * 1e-2, "+31.415927ms");
  CheckAs (t * 1e-1, "+314.159265ms");
  CheckAs (t * 1e-0, "+3.141593s");
  CheckAs (t * 1e+1, "+31.415927s");
  CheckAs (t * 1e+2, "+5.235988min");
  CheckAs (t * 1e+3, "+52.359878min");
  CheckAs (t * 1e+4, "+8.726646h");
  CheckAs (t * 1e+5, "+3.636103d");
  CheckAs (t * 1e+6, "+36.361026d");
  CheckAs (t * 1e+7, "+363.610261d");
  CheckAs (t * 1e+8, "+9.961925y");
}

/**
* \ingroup core-tests
* \brief   Time test Suite.  Runs the appropriate test cases for time
*/
static class TimeTestSuite : public TestSuite
{
public:
  TimeTestSuite ()
    : TestSuite ("time", UNIT)
  {
    AddTestCase (new TimeWithSignTestCase (), TestCase::QUICK);
    AddTestCase (new TimeInputOutputTestCase (), TestCase::QUICK);
    // This should be last, since it changes the resolution
    AddTestCase (new TimeSimpleTestCase (), TestCase::QUICK);
  }
}
/** \brief Member variable for time test suite */
g_timeTestSuite;
