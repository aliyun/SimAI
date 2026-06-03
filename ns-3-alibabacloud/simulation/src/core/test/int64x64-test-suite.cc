/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 INRIA
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

#include "ns3/int64x64.h"
#include "ns3/test.h"
#include "ns3/valgrind.h"  // Bug 1882

#include <cfloat>   // FLT_RADIX,...
#include <cmath>    // fabs, round
#include <iomanip>
#include <limits>   // numeric_limits<>::epsilon ()

using namespace ns3;

namespace ns3 {

namespace int64x64 {

namespace test {

/**
 * \file
 * \ingroup int64x64-tests
 * int64x46 test suite
 */

/**
 * \ingroup core-tests
 * \defgroup int64x64-tests int64x64 tests
 */


/**
 * \ingroup int64x64-tests
 * 
 * Pretty printer for test cases.
 */
class Printer
{
public:
  /**
   * Construct from high and low words of Q64.64 representation.
   *
   * \param [in] high The integer portion.
   * \param [in] low The fractional portion.
   */
  Printer (const int64_t high, const uint64_t low)
    : m_haveInt (false),
      m_value (0),
      m_high (high),
      m_low (low)
  { }

  /**
   * Construct from an \c int64x64_t Q64.64 value.
   *
   * \param [in] value The value.
   */
  Printer (const int64x64_t value)
    : m_haveInt (true),
      m_value (value),
      m_high (value.GetHigh ()),
      m_low (value.GetLow ())
  { }

private:
  /**
   * Output streamer, the main reason for this class.
   *
   * \param [in] os The stream.
   * \param [in] p The value to print.
   * \returns The stream.
   */
  friend std::ostream & operator << (std::ostream & os, const Printer & p);

  bool       m_haveInt;  /**< Do we have a full int64x64_t value? */
  int64x64_t m_value;    /**< The int64x64_t value. */
  int64_t    m_high;     /**< The high (integer) word. */
  uint64_t   m_low;      /**< The low (fractional) word. */
};

std::ostream & operator << (std::ostream & os, const Printer & p)
{
  if (p.m_haveInt)
    {
      os << std::fixed << std::setprecision (22)
         << p.m_value;

    }

  os << std::hex  << std::setfill ('0')
     << " (0x" << std::setw (16) << p.m_high
     << " 0x"  << std::setw (16) << p.m_low  << ")"
     << std::dec  << std::setfill (' ');
  return os;
}

/**
 * \ingroup int64x64-tests
 * 
 * Test: manipulate the high and low part of every number.
 */
class Int64x64HiLoTestCase : public TestCase
{
public:
  Int64x64HiLoTestCase ();
  virtual void DoRun (void);
  /**
   * Check the high and low parts for correctness.
   * \param hi The high part of the int64x64_t.
   * \param lo The low part of the int64x64_t.
   */
  void Check (const int64_t hi, const uint64_t lo);
};

Int64x64HiLoTestCase::Int64x64HiLoTestCase ()
  : TestCase ("Manipulate the high and low part of every number")
{}

void
Int64x64HiLoTestCase::Check (const int64_t hi, const uint64_t lo)
{
  uint64_t tolerance = 0;
  if (int64x64_t::implementation == int64x64_t::ld_impl)
    {
      // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
      tolerance = 1;
    }

  int64x64_t value = int64x64_t (hi,lo);
  uint64_t vLow = value.GetLow ();
  bool pass = ( (value.GetHigh () == hi)
                && ( (Max (vLow, lo) - Min (vLow, lo)) <= tolerance)
                );

  std::cout << GetParent ()->GetName () << " Check: "
            << (pass ? "pass " : "FAIL ")
            << Printer (value) << " from" << Printer (hi, lo)
            << std::endl;

  NS_TEST_EXPECT_MSG_EQ (value.GetHigh (), hi,
                         "High part does not match for hi:" << hi
                                                            << " lo: " << lo);
  NS_TEST_EXPECT_MSG_EQ_TOL ((int64_t)vLow, (int64_t)lo, (int64_t)tolerance,
                             "Low part does not match for hi: " << hi
                                                                << " lo: " << lo);
}

void
Int64x64HiLoTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Check: " << GetName ()
            << std::endl;

  uint64_t low = 1;
  if (int64x64_t::implementation == int64x64_t::ld_impl)
    {
      // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
      low = static_cast<uint64_t> (HP_MAX_64 * std::numeric_limits<long double>::epsilon ());
    }

  Check ( 0, 0);
  Check ( 0, low);
  Check ( 0, 0xffffffffffffffffULL - low);

  Check ( 1, 0);
  Check ( 1, low);
  Check ( 1, 0xffffffffffffffffULL - low);

  Check (-1, 0);
  Check (-1, low);
  Check (-1, 0xffffffffffffffffULL - low);
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: check GetInt and Round.
 */
class Int64x64IntRoundTestCase : public TestCase
{
public:
  Int64x64IntRoundTestCase (void);
  virtual void DoRun (void);
  /**
   * Check the int64x64 value for correctness.
   * \param value The int64x64_t value.
   * \param expectInt The expected integer value.
   * \param expectRnd TThe expected rounding value.
   */
  void Check (const int64x64_t value,
              const int64_t expectInt,
              const int64_t expectRnd);
};

Int64x64IntRoundTestCase::Int64x64IntRoundTestCase (void)
  : TestCase ("Check GetInt and Round")
{}

void
Int64x64IntRoundTestCase::Check (const int64x64_t value,
                                 const int64_t expectInt,
                                 const int64_t expectRnd)
{
  int64_t vInt = value.GetInt ();
  int64_t vRnd = value.Round ();

  bool pass = (vInt == expectInt) && (vRnd == expectRnd);
  std::cout << GetParent ()->GetName () << " Check: "
            << (pass ? "pass " : "FAIL ")
            << value
            << " (int)-> " << std::setw (2) << vInt << " (expected: " << std::setw (2) << expectInt
            << "), (rnd)-> " << std::setw (2) << vRnd << " (expected " << std::setw (2) << expectRnd
            << ")"
            << std::endl;

  NS_TEST_EXPECT_MSG_EQ (vInt, expectInt,
                         "Truncation to int failed");
  NS_TEST_EXPECT_MSG_EQ (vRnd, expectRnd,
                         "Rounding to int failed.");
}

void
Int64x64IntRoundTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Check: " << GetName ()
            << std::endl;

  // Trivial cases
  Check ( 0,    0,  0);
  Check ( 1,    1,  1);
  Check (-1,   -1, -1);

  // Both should move toward zero
  Check ( 2.4,  2,  2);
  Check (-2.4, -2, -2);

  // GetInt should move toward zero; Round should move away
  Check ( 3.6,  3,  4);
  Check (-3.6, -3, -4);
  // Boundary case
  Check ( 4.5,  4,  5);
  Check (-4.5, -4, -5);
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: parse int64x64_t numbers as strings.
 */
class Int64x64InputTestCase : public TestCase
{
public:
  Int64x64InputTestCase ();
  virtual void DoRun (void);
  /**
   * Check the iont64x64 for correctness.
   * \param str String representation of a number.
   * \param hi The expected high part of the int64x64_t.
   * \param lo The expected low part of the int64x64_t.
   * \param tolerance The allowed tolerance.
   */
  void Check (const std::string & str,
              const int64_t hi, const uint64_t lo,
              const int64_t tolerance = 0);
};
Int64x64InputTestCase::Int64x64InputTestCase ()
  : TestCase ("Parse int64x64_t numbers as strings")
{}
void
Int64x64InputTestCase::Check (const std::string & str,
                              const int64_t hi, const uint64_t lo,
                              const int64_t tolerance /* = 0 */)

{
  std::istringstream iss;
  iss.str (str);
  int64x64_t value;
  iss >> value;

  std::string input = "\"" + str + "\"";
  uint64_t vLow = value.GetLow ();
  bool pass = ( (value.GetHigh () == hi)
                && ( Max (vLow, lo) - Min (vLow, lo) <= tolerance)
                );

  std::cout << GetParent ()->GetName () << " Input: "
            << (pass ? "pass " : "FAIL ")
            << std::left << std::setw (28) << input << std::right
            << Printer (value)
            << " expected: " << Printer (hi, lo) << " +/- " << tolerance
            << std::endl;

  NS_TEST_EXPECT_MSG_EQ (value.GetHigh (), hi,
                         "High parts do not match for input string \""
                         << str << "\"");
  NS_TEST_EXPECT_MSG_EQ_TOL ((int64_t)value.GetLow (), (int64_t)lo, tolerance,
                             "Low parts do not match for input string \""
                             << str << "\"");
}
void
Int64x64InputTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Input: " << GetName ()
            << std::endl;

  int64_t tolerance = 0;
  if (int64x64_t::implementation == int64x64_t::ld_impl)
    {
      // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
      tolerance = 2;
    }

  Check ("1", 1, 0);
  Check ("+1", 1, 0);
  Check ("-1", -1, 0);
  Check ("1.0", 1, 0);
  Check ("+1.0", 1, 0);
  Check ("001.0", 1, 0);
  Check ("+001.0", 1, 0);
  Check ("020.0", 20, 0);
  Check ("+020.0", 20, 0);
  Check ("1.0000000", 1, 0);
  Check ("-1.0", -1, 0, tolerance);
  Check ("-1.0000", -1, 0, tolerance);
  Check (" 1.000000000000000000054",  1, 1, tolerance);
  Check ("-1.000000000000000000054", (int64_t)-2, (uint64_t)-1, tolerance);
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: roundtrip int64x64_t numbers as strings.
 * 
 * Prints an int64x64_t and read it back.
 */
class Int64x64InputOutputTestCase : public TestCase
{
public:
  Int64x64InputOutputTestCase ();
  virtual void DoRun (void);
  /**
   * Check the iont64x64 for correctness.
   * \param str String representation of a number.
   * \param tolerance The allowed tolerance.
   */
  void Check (const std::string & str,
              const int64_t tolerance = 0);
};
Int64x64InputOutputTestCase::Int64x64InputOutputTestCase ()
  : TestCase ("Roundtrip int64x64_t numbers as strings")
{}
void
Int64x64InputOutputTestCase::Check (const std::string & str,
                                    const int64_t tolerance /* = 0 */)
{
  std::stringstream iss (str);
  int64x64_t expect;
  iss >> expect;

  std::stringstream oss;
  oss << std::scientific << std::setprecision (21) << expect;
  int64x64_t value;
  oss >> value;

  bool pass = Abs (value - expect) <= int64x64_t (0, tolerance + 1);

  std::string input  = "\"" + str + "\"";
  std::string output = "\"" + oss.str () + "\"";

  if (pass)
    {
      std::cout << GetParent ()->GetName () << " InputOutput: "
                << (pass ? "pass " : "FAIL ")
                << " in:  " << std::left << std::setw (28) << input
                << " out: " << std::left << std::setw (28) << output
                << std::right
                << std::endl;
    }
  else
    {
      std::cout << GetParent ()->GetName () << " InputOutput: "
                << (pass ? "pass " : "FAIL ")
                << " in:  " << std::left << std::setw (28) << input
                << std::right << Printer (expect)
                << std::endl;
      std::cout << GetParent ()->GetName ()
                << std::setw (19) << " "
                << " out: " << std::left << std::setw (28) << output
                << std::right << Printer (value)
                << std::endl;
    }

  NS_TEST_EXPECT_MSG_EQ_TOL (value, expect, int64x64_t (0, tolerance),
                             "Converted string does not match expected string");
}

void
Int64x64InputOutputTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " InputOutput: " << GetName ()
            << std::endl;

  int64_t tolerance = 0;
  if (int64x64_t::implementation == int64x64_t::ld_impl)
    {
      // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
      tolerance = 1;
    }

  Check ("+1.000000000000000000000");
  Check ("+20.000000000000000000000");
  Check ("+0.000000000000000000000", tolerance);
  Check ("-1.000000000000000000000", tolerance);
  Check ("+1.084467440737095516158", tolerance);
  Check ("-2.084467440737095516158", tolerance);
  Check ("+3.184467440737095516179", tolerance);
  Check ("-4.184467440737095516179", tolerance);
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: basic arithmetic operations.
 */
class Int64x64ArithmeticTestCase : public TestCase
{
public:
  Int64x64ArithmeticTestCase ();
  virtual void DoRun (void);
  /**
   * Check the int64x64 for correctness.
   * \param test The test number.
   * \param value The actual value.
   * \param expect The expected value.
   * \param tolerance The allowed tolerance.
   */
  void Check (const int        test,
              const int64x64_t value, const int64x64_t expect,
              const int64x64_t tolerance = int64x64_t (0,0));
};

Int64x64ArithmeticTestCase::Int64x64ArithmeticTestCase ()
  : TestCase ("Basic arithmetic operations")
{}
void
Int64x64ArithmeticTestCase::Check (const int test,
                                   const int64x64_t value,
                                   const int64x64_t expect,
                                   const int64x64_t tolerance)
{
  bool pass = Abs (value - expect) <= tolerance;

  std::cout << GetParent ()->GetName () << " Arithmetic: "
            << (pass ? "pass " : "FAIL ")
            << test << ": " << value << " == " << expect
            << " (+/- " << tolerance << ")"
            << std::endl;

  NS_TEST_ASSERT_MSG_EQ_TOL ( value, expect, tolerance,
                              "Arithmetic failure in test case " << test);
}

void
Int64x64ArithmeticTestCase::DoRun (void)
{
  const int64x64_t tol1 (0, 1);
  const int64x64_t zero (0, 0);
  const int64x64_t one  (1, 0);
  const int64x64_t two  (2, 0);
  const int64x64_t thre (3, 0);

  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Arithmetic: " << GetName ()
            << std::endl;

  Check ( 0,   zero  -   zero,   zero  );
  Check ( 1,   zero  -   one,   -one   );
  Check ( 2,   one   -   one,    zero  );
  Check ( 3,   one   -   two,   -one   );
  Check ( 4,   one   - (-one ),   two   );
  Check ( 5, (-one ) - (-two ),   one   );
  Check ( 6, (-one ) -   two,   -thre  );

  Check ( 7,   zero  +   zero,   zero  );
  Check ( 8,   zero  +   one,    one   );
  Check ( 9,   one   +   one,    two   );
  Check (10,   one   +   two,    thre  );
  Check (11,   one   + (-one ),   zero  );
  Check (12, (-one ) + (-two ),  -thre  );
  Check (13, (-one ) +   two,    one   );

  Check (14,   zero  *   zero,   zero  );
  Check (15,   zero  *   one,    zero  );
  Check (16,   zero  * (-one ),   zero  );
  Check (17,   one   *   one,    one   );
  Check (18,   one   * (-one ),  -one   );
  Check (19, (-one ) * (-one ),   one   );

  Check (20,  (two  * thre ) / thre, two  );

  const int64x64_t frac  = int64x64_t (0, 0xc000000000000000ULL);  // 0.75
  const int64x64_t fplf2 = frac + frac * frac;  // 1.3125

  Check (21,   frac,              0.75);
  Check (22,   fplf2,             1.3125);

  const int64x64_t zerof = zero + frac;
  const int64x64_t onef  = one  + frac;
  const int64x64_t twof  = two  + frac;
  const int64x64_t thref = thre + frac;

  Check (23,   zerof,             frac);


  Check (24,   zerof -   zerof,   zero );
  Check (25,   zerof -   onef,   -one  );
  Check (26,   onef  -   onef,    zero );
  Check (27,   onef  -   twof,   -one  );
  Check (28,   onef  - (-onef),   twof  + frac  );
  Check (29, (-onef) - (-twof),   one  );
  Check (30, (-onef) -   twof,   -thref - frac  );

  Check (31,   zerof +   zerof,   zerof + frac  );
  Check (32,   zerof +   onef,    onef  + frac  );
  Check (33,   onef  +   onef,    twof  + frac  );
  Check (34,   onef  +   twof,    thref + frac  );
  Check (35,   onef  + (-onef),   zero  );
  Check (36, (-onef) + (-twof),  -thref - frac  );
  Check (37, (-onef) +   twof,    one   );

  Check (38,   zerof *   zerof,   frac  * frac  );
  Check (39,   zero  *   onef,    zero  );
  Check (40,   zerof *   one,     frac  );

  Check (41,   zerof *   onef,    fplf2 );
  Check (42,   zerof * (-onef),  -fplf2 );
  Check (43,   onef  *   onef,    onef  + fplf2 );
  Check (44,   onef  * (-onef),  -onef  - fplf2 );
  Check (45, (-onef) * (-onef),   onef  + fplf2 );


  // Multiplication followed by division is exact:
  Check (46, (two  * thre ) / thre, two  );
  Check (47, (twof * thref) / thref, twof );

  // Division followed by multiplication loses a bit or two:
  Check (48, (two  / thre)  * thre,  two, 2 * tol1 );
  Check (49, (twof / thref) * thref, twof, 3 * tol1 );

  // The example below shows that we really do not lose
  // much precision internally: it is almost always the
  // final conversion which loses precision.
  Check (50, (int64x64_t (2000000000) / int64x64_t (3)) * int64x64_t (3),
         int64x64_t (1999999999, 0xfffffffffffffffeULL));

  // Check special values
  Check (51,  int64x64_t (0, 0x159fa87f8aeaad21ULL) * 10,
         int64x64_t (0, 0xd83c94fb6d2ac34aULL));

}



/**
 * \ingroup int64x64-tests
 * 
 * Test case for bug 455.
 * 
 * See \bugid{455}
 */
class Int64x64Bug455TestCase : public TestCase
{
public:
  Int64x64Bug455TestCase ();
  virtual void DoRun (void);
  /**
   * Check the int64x64 for correctness.
   * \param result The actual value.
   * \param expect The expected value.
   * \param msg The error message to print.
   */
  void Check (const double result, const double expect,
              const std::string & msg);
};

Int64x64Bug455TestCase::Int64x64Bug455TestCase ()
  : TestCase ("Test case for bug 455")
{}
void
Int64x64Bug455TestCase::Check (const double result, const double expect,
                               const std::string & msg)
{
  bool pass = result == expect;

  std::cout << GetParent ()->GetName () << " Bug 455: "
            << (pass ? "pass " : "FAIL ")
            << "res: "  << result
            << " exp: " << expect
            << ": " << msg
            << std::endl;

  NS_TEST_ASSERT_MSG_EQ (result, expect, msg);
}

void
Int64x64Bug455TestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Bug 455: " << GetName ()
            << std::endl;

  int64x64_t a = int64x64_t (0.1);
  a /= int64x64_t (1.25);
  Check (a.GetDouble (), 0.08, "The original testcase");

  a = int64x64_t (0.5);
  a *= int64x64_t (5);
  Check (a.GetDouble (), 2.5, "Simple test for multiplication");

  a = int64x64_t (-0.5);
  a *= int64x64_t (5);
  Check (a.GetDouble (), -2.5, "Test sign, first operation negative");

  a = int64x64_t (-0.5);
  a *= int64x64_t (-5);
  Check (a.GetDouble (), 2.5, "both operands negative");

  a = int64x64_t (0.5);
  a *= int64x64_t (-5);
  Check (a.GetDouble (), -2.5, "only second operand negative");
}



/**
 * \ingroup int64x64-tests
 * 
 * Test case for bug 455.
 * 
 * See \bugid{863}
 */
class Int64x64Bug863TestCase : public TestCase
{
public:
  Int64x64Bug863TestCase ();
  virtual void DoRun (void);
  /**
   * Check the int64x64 for correctness.
   * \param result The actual value.
   * \param expect The expected value.
   * \param msg The error message to print.
   */
  void Check (const double result, const double expect,
              const std::string & msg);
};

Int64x64Bug863TestCase::Int64x64Bug863TestCase ()
  : TestCase ("Test case for bug 863")
{}
void
Int64x64Bug863TestCase::Check (const double result, const double expect,
                               const std::string & msg)
{
  bool pass = result == expect;

  std::cout << GetParent ()->GetName () << " Bug 863: "
            << (pass ? "pass " : "FAIL ")
            << "res: "  << result
            << " exp: " << expect
            << ": " << msg
            << std::endl;

  NS_TEST_ASSERT_MSG_EQ (result, expect, msg);
}

void
Int64x64Bug863TestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Bug 863: " << GetName ()
            << std::endl;

  int64x64_t a = int64x64_t (0.9);
  a /= int64x64_t (1);
  Check (a.GetDouble (), 0.9, "The original testcase");

  a = int64x64_t (0.5);
  a /= int64x64_t (0.5);
  Check (a.GetDouble (), 1.0, "Simple test for division");

  a = int64x64_t (-0.5);
  Check (a.GetDouble (), -0.5, "Check that we actually convert doubles correctly");

  a /= int64x64_t (0.5);
  Check (a.GetDouble (), -1.0, "first argument negative");

  a = int64x64_t (0.5);
  a /= int64x64_t (-0.5);
  Check (a.GetDouble (), -1.0, "second argument negative");

  a = int64x64_t (-0.5);
  a /= int64x64_t (-0.5);
  Check (a.GetDouble (), 1.0, "both arguments negative");
}


/**
 * \ingroup int64x64-tests
 * 
 * Test case for bug 455.
 * 
 * See \bugid{1786}
 */
class Int64x64Bug1786TestCase : public TestCase
{
public:
  Int64x64Bug1786TestCase ();
  virtual void DoRun (void);
  /**
   * Check the int64x64 for correctness.
   * \param low The actual low value.
   * \param value The expected low part printed value.
   * \param tolerance The allowed tolerance.
   */
  void Check (const uint64_t low, const std::string & value,
              const int64_t tolerance = 0);
};

Int64x64Bug1786TestCase::Int64x64Bug1786TestCase ()
  : TestCase ("Test case for bug 1786")
{}
void
Int64x64Bug1786TestCase::Check (const uint64_t low,
                                const std::string & str,
                                const int64_t tolerance /* = 0 */)
{
  int64x64_t value (0, low);
  std::ostringstream oss;
  oss << std::scientific << std::setprecision (22) << value;

  if (tolerance == 0)
    {
      bool pass = oss.str () == str;

      std::cout << GetParent ()->GetName () << " Bug 1786: "
                << (pass ? "pass " : "FAIL ")
                << "    0x" << std::hex << std::setw (16) << low << std::dec
                << " = "    << oss.str ();
      if (!pass)
        {
          std::cout << ", expected " << str;
        }
      std::cout << std::endl;

      NS_TEST_EXPECT_MSG_EQ (oss.str (), str,
                             "Fraction string not correct");
    }
  else
    {
      // No obvious way to implement a tolerance on the strings

      std::cout << GetParent ()->GetName () << " Bug 1786: "
                << "skip "
                << "    0x" << std::hex << std::setw (16) << low << std::dec
                << " = "    << oss.str ()
                << ", expected " << str
                << std::endl;

    }
}
void
Int64x64Bug1786TestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " But 1786: " << GetName ()
            << std::endl;

  int64_t tolerance = 0;
  if (int64x64_t::implementation == int64x64_t::ld_impl)
    {
      // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
      tolerance = 1;
    }

  // Some of these values differ from the DoubleTestCase
  // by one count in the last place
  // because operator<< truncates the last output digit,
  // instead of rounding.
  Check (                 1ULL, "+0.0000000000000000000542");
  Check (                 2ULL, "+0.0000000000000000001084");
  Check (                 3ULL, "+0.0000000000000000001626");
  Check (                 4ULL, "+0.0000000000000000002168");
  Check (                 5ULL, "+0.0000000000000000002710");
  Check (                 6ULL, "+0.0000000000000000003253");
  Check (                 7ULL, "+0.0000000000000000003795");
  Check (                 8ULL, "+0.0000000000000000004337");
  Check (                 9ULL, "+0.0000000000000000004879");
  Check (               0xAULL, "+0.0000000000000000005421");
  Check (               0xFULL, "+0.0000000000000000008132");
  Check (              0xF0ULL, "+0.0000000000000000130104");
  Check (             0xF00ULL, "+0.0000000000000002081668");
  Check (            0xF000ULL, "+0.0000000000000033306691");
  Check (           0xF0000ULL, "+0.0000000000000532907052");
  Check (          0xF00000ULL, "+0.0000000000008526512829");
  Check (         0xF000000ULL, "+0.0000000000136424205266");
  Check (        0xF0000000ULL, "+0.0000000002182787284255");
  Check (       0xF00000000ULL, "+0.0000000034924596548080");
  Check (      0xF000000000ULL, "+0.0000000558793544769287");
  Check (     0xF0000000000ULL, "+0.0000008940696716308594");
  Check (    0xF00000000000ULL, "+0.0000143051147460937500");
  Check (   0xF000000000000ULL, "+0.0002288818359375000000");
  Check (  0xF0000000000000ULL, "+0.0036621093750000000000");
  Check ( 0xF00000000000000ULL, "+0.0585937500000000000000");
  std::cout << std::endl;
  Check (0x7FFFFFFFFFFFFFFDULL, "+0.4999999999999999998374", tolerance);
  Check (0x7FFFFFFFFFFFFFFEULL, "+0.4999999999999999998916", tolerance);
  Check (0x7FFFFFFFFFFFFFFFULL, "+0.4999999999999999999458", tolerance);
  Check (0x8000000000000000ULL, "+0.5000000000000000000000");
  Check (0x8000000000000001ULL, "+0.5000000000000000000542", tolerance);
  Check (0x8000000000000002ULL, "+0.5000000000000000001084", tolerance);
  Check (0x8000000000000003ULL, "+0.5000000000000000001626", tolerance);
  std::cout << std::endl;
  Check (0xF000000000000000ULL, "+0.9375000000000000000000");
  Check (0xFF00000000000000ULL, "+0.9960937500000000000000");
  Check (0xFFF0000000000000ULL, "+0.9997558593750000000000");
  Check (0xFFFF000000000000ULL, "+0.9999847412109375000000");
  Check (0xFFFFF00000000000ULL, "+0.9999990463256835937500");
  Check (0xFFFFFF0000000000ULL, "+0.9999999403953552246094");
  Check (0xFFFFFFF000000000ULL, "+0.9999999962747097015381");
  Check (0xFFFFFFFF00000000ULL, "+0.9999999997671693563461");
  Check (0xFFFFFFFFF0000000ULL, "+0.9999999999854480847716");
  Check (0xFFFFFFFFFF000000ULL, "+0.9999999999990905052982");
  Check (0xFFFFFFFFFFF00000ULL, "+0.9999999999999431565811");
  Check (0xFFFFFFFFFFFF0000ULL, "+0.9999999999999964472863");
  Check (0xFFFFFFFFFFFFF000ULL, "+0.9999999999999997779554");
  Check (0xFFFFFFFFFFFFFF00ULL, "+0.9999999999999999861222");
  Check (0xFFFFFFFFFFFFFFF0ULL, "+0.9999999999999999991326");
  Check (0xFFFFFFFFFFFFFFF5ULL, "+0.9999999999999999994037", tolerance);
  Check (0xFFFFFFFFFFFFFFF6ULL, "+0.9999999999999999994579", tolerance);
  Check (0xFFFFFFFFFFFFFFF7ULL, "+0.9999999999999999995121", tolerance);
  Check (0xFFFFFFFFFFFFFFF8ULL, "+0.9999999999999999995663", tolerance);
  Check (0xFFFFFFFFFFFFFFF9ULL, "+0.9999999999999999996205", tolerance);
  Check (0xFFFFFFFFFFFFFFFAULL, "+0.9999999999999999996747", tolerance);
  Check (0xFFFFFFFFFFFFFFFBULL, "+0.9999999999999999997289", tolerance);
  Check (0xFFFFFFFFFFFFFFFCULL, "+0.9999999999999999997832", tolerance);
  Check (0xFFFFFFFFFFFFFFFDULL, "+0.9999999999999999998374", tolerance);
  Check (0xFFFFFFFFFFFFFFFEULL, "+0.9999999999999999998916", tolerance);
  Check (0xFFFFFFFFFFFFFFFFULL, "+0.9999999999999999999458", tolerance);
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: basic compare operations.
 */
class Int64x64CompareTestCase : public TestCase
{
public:
  Int64x64CompareTestCase ();
  virtual void DoRun (void);

  /**
   * Check the int64x64 for correctness.
   * \param result The actual value.
   * \param expect The expected value.
   * \param msg The error mesage to print.
   */
  void Check (const bool result, const bool expect,
              const std::string & msg);
};
Int64x64CompareTestCase::Int64x64CompareTestCase ()
  : TestCase ("Basic compare operations")
{}
void
Int64x64CompareTestCase::Check (const bool result, const bool expect,
                                const std::string & msg)
{
  bool pass = result == expect;

  std::cout << GetParent ()->GetName () << " Compare: "
            << (pass ? "pass " : "FAIL ")
            << msg
            << std::endl;

  NS_TEST_ASSERT_MSG_EQ (result, expect, msg);
}

void
Int64x64CompareTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Compare: " << GetName ()
            << std::endl;

  const int64x64_t zero ( 0, 0);
  const int64x64_t one  ( 1, 0);
  const int64x64_t two  ( 2, 0);
  const int64x64_t mone (-1, 0);
  const int64x64_t mtwo (-2, 0);
  const int64x64_t frac  = int64x64_t (0, 0xc000000000000000ULL);  // 0.75
  const int64x64_t zerof = zero + frac;
  const int64x64_t onef  = one  + frac;
  const int64x64_t monef = mone - frac;
  const int64x64_t mtwof = mtwo - frac;

  Check ( zerof    == zerof, true,  "equality, zero");
  Check ( onef     == onef,  true,  "equality, positive");
  Check ( mtwof    == mtwof, true,  "equality, negative");
  Check ( zero     == one,   false, "equality false, zero");
  Check ( one      == two,   false, "equality false, unsigned");
  Check ( one      == mone,  false, "equality false, signed");
  Check ( onef     == one,   false, "equality false, fraction");
  std::cout << std::endl;

  Check ( zerof    != zerof, false, "inequality, zero");
  Check ( onef     != onef,  false, "inequality, positive");
  Check ( mtwof    != mtwof, false, "inequality, negative");
  Check ( zero     != one,   true,  "inequality true, zero");
  Check ( one      != two,   true,  "inequality true, unsigned");
  Check ( one      != mone,  true,  "inequality true, signed");
  Check ( onef     != one,   true,  "inequality true, fraction");
  std::cout << std::endl;

  Check ( zerof    <  onef,  true,  "less, zerof");
  Check ( zero     <  zerof, true,  "less, zero");
  Check ( one      <  onef,  true,  "less, positive");
  Check ( monef    <  mone,  true,  "less, negative");
  Check ( onef     <  one,   false, "less, false, positive");
  Check ( mtwo     <  mtwof, false, "less, false, negative");
  std::cout << std::endl;

  Check ( zerof    <= zerof, true,  "less equal, equal, zerof");
  Check ( zero     <= zerof, true,  "less equal, less, zero");
  Check ( onef     <= onef,  true,  "less equal, equal, positive");
  Check ( monef    <= mone,  true,  "less equal, less, negative");
  Check ( onef     <= one,   false, "less equal, false, positive");
  Check ( mtwo     <= mtwof, false, "less equal, false, negative");
  std::cout << std::endl;

  Check ( onef     >  zerof, true,  "greater, zerof");
  Check ( zerof    >  zero,  true,  "greater, zero");
  Check ( onef     >  one,   true,  "greater, positive");
  Check ( mone     >  monef, true,  "greater, negative");
  Check ( one      >  onef,  false, "greater, false, positive");
  Check ( mtwof    >  mtwo,  false, "greater, false, negative");
  std::cout << std::endl;

  Check ( zerof    >= zerof, true,  "greater equal, equal, zerof");
  Check ( zerof    >= zero,  true,  "greater equal, greater, zero");
  Check ( onef     >= onef,  true,  "greater equal, equal, positive");
  Check ( mone     >= monef, true,  "greater equal, greater, negative");
  Check ( one      >= onef,  false, "greater equal, false, positive");
  Check ( mtwof    >= mtwo,  false, "greater equal, false, negative");
  std::cout << std::endl;

  Check ( zero     == false, true,  "zero   == false");
  Check ( one      == true,  true,  "one    == true");
  Check ( zerof    != false, true,  "zerof  != false");
  Check ( (!zero)  == true,  true,  "!zero  == true");
  Check ( (!zerof) == false, true,  "!zerof == false");
  Check ( (!one)   == false, true,  "!one   == false");
  Check ( (+onef)  == onef,  true, "unary positive");
  Check ( (-onef)  == monef, true, "unary negative");
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: Invert and MulByInvert.
 */
class Int64x64InvertTestCase : public TestCase
{
public:
  Int64x64InvertTestCase ();
  virtual void DoRun (void);
  /**
   * Check the int64x64 for correctness.
   * \param factor The factor used to invert the number.
   */
  void Check (const int64_t factor);
  /**
   * Check the int64x64 for correctness.
   * \param factor The factor used to invert the number.
   * \param result The value.
   * \param expect The expected value.
   * \param msg The error mesage to print.
   * \param tolerance The allowed tolerance.
   */
  void CheckCase (const uint64_t factor,
                  const int64x64_t result, const int64x64_t expect,
                  const std::string & msg,
                  const double tolerance = 0);
};

Int64x64InvertTestCase::Int64x64InvertTestCase ()
  : TestCase ("Invert and MulByInvert")
{}
void
Int64x64InvertTestCase::CheckCase (const uint64_t factor,
                                   const int64x64_t result,
                                   const int64x64_t expect,
                                   const std::string & msg,
                                   const double tolerance /* = 0 */)
{
  bool pass = Abs (result - expect) <= tolerance;

  std::cout << GetParent ()->GetName () << " Invert: ";

  if (pass)
    {
      std::cout << "pass:  " << factor << ": ";

    }
  else
    {
      std::cout << "FAIL:  " << factor << ": "
                << "(res: " << result
                << " exp: " << expect
                << " tol: " << tolerance << ")  ";
    }
  std::cout << msg
            << std::endl;

  NS_TEST_ASSERT_MSG_EQ_TOL (result, expect, int64x64_t (tolerance), msg);
}

void
Int64x64InvertTestCase::Check (const int64_t factor)
{
  const int64x64_t one (1, 0);
  const int64x64_t factorI = one / int64x64_t (factor);

  const int64x64_t a = int64x64_t::Invert (factor);
  int64x64_t b = int64x64_t (factor);

  double tolerance = 0;
  if (int64x64_t::implementation == int64x64_t::ld_impl)
    {
      // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
      tolerance = 0.000000000000000001L;
    }

  b.MulByInvert (a);
  CheckCase (factor, b, one, "x * x^-1 == 1", tolerance);

  int64x64_t c = int64x64_t (1);
  c.MulByInvert (a);
  CheckCase (factor, c, factorI, "1 * x^-1 == 1 / x");

  int64x64_t d = int64x64_t (1);
  d /= (int64x64_t (factor));
  CheckCase (factor, d, c, "1/x == x^-1");

  int64x64_t e = int64x64_t (-factor);
  e.MulByInvert (a);
  CheckCase (factor, e, -one, "-x * x^-1 == -1", tolerance);
}

void
Int64x64InvertTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Invert: " << GetName ()
            << std::endl;

  Check (2);
  Check (3);
  Check (4);
  Check (5);
  Check (6);
  Check (10);
  Check (99);
  Check (100);
  Check (1000);
  Check (10000);
  Check (100000);
  Check (100000);
  Check (1000000);
  Check (10000000);
  Check (100000000);
  Check (1000000000);
  Check (10000000000LL);
  Check (100000000000LL);
  Check (1000000000000LL);
  Check (10000000000000LL);
  Check (100000000000000LL);
  Check (1000000000000000LL);
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: construct from floating point.
 */
class Int64x64DoubleTestCase : public TestCase
{
public:
  Int64x64DoubleTestCase ();
  virtual void DoRun (void);

  /**
   * Check the int64x64 for correctness.
   * \param intPart The expected integer part value of the int64x64.
   */
  void Check (const int64_t intPart);
  /**
   * Check the int64x64 for correctness.
   * \param dec The integer part of the value to test.
   * \param frac The fractional part of the value to test.x
   * \param intPart The expected integer part value of the int64x64.
   * \param lo The expected low part value of the int64x64.
   */
  void Check (const long double dec,
              const long double frac,
              const int64_t intPart,
              const uint64_t lo);

private:

  /**
   * Compute a multiplier to match the mantissa size on this platform
   *
   * Since we will store the fractional part of a double
   * in the low word (64 bits) of our Q64.64
   * the most mantissa bits we can take advantage of is
   *
   *     EFF_MANT_DIG = std::min (64, LDBL_MANT_DIG)
   *
   * We have to bound this for platforms with LDBL_MANT_DIG > 64.
   *
   * The number of "missing" bits in the mantissa is
   *
   *     MISS_MANT_DIG = 64 - EFF_MANT_DIG = std::max (0, 64 - LDBL_MANT_DIG)
   *
   * This will lie in the closed interval [0, 64]
   */
  static constexpr int MISS_MANT_DIG = std::max (0, 64 - LDBL_MANT_DIG);

  /**
   * The smallest low word we expect to get from a conversion.
   *
   *     MIN_LOW = 2^MISS_MANT_DIG
   *
   * which will be in [1, 2^64].
   */
  static constexpr long double MIN_LOW = 1 << MISS_MANT_DIG;

  /**
   * Smallest mantissa we expect to convert to a non-zero low word.
   *
   *     MIN_MANT = MIN_LOW / 2^64
   *              = 2^(MISS_MANT_DIG - 64)
   *              = 2^(-EFF_MANT_DIG)
   *
   * We scale and round this value to match the
   * hard-coded fractional values in Check(intPart)
   * which have 22 decimal digits.
   * 
   * Since we use std::round() which isn't constexpr,
   * just declare this const and initialize below.
   */
  static const long double MIN_MANT;


  // Member variables
  long double m_last;       //!< The last value tested.
  int64x64_t  m_deltaMax;   //!< The maximum observed difference between expected and computed values.
  int         m_deltaCount; //!< The number of times a delta was recorded.
};


/* static */
const long double
Int64x64DoubleTestCase::MIN_MANT =
  std::round ( 1e22 / std::pow (2.0L, std::min (64, LDBL_MANT_DIG))) / 1e22;

Int64x64DoubleTestCase::Int64x64DoubleTestCase ()
  : TestCase ("Construct from floating point."),
    m_last {0},
    m_deltaMax {0},
    m_deltaCount {0}
{}

void
Int64x64DoubleTestCase::Check (const long double dec,
                               const long double frac,
                               const int64_t intPart,
                               const uint64_t lo)
{
  // 1.  The double value we're going to convert
  long double value = dec + frac;

  // 2.  The expected value of the conversion
  int64x64_t expect (intPart, lo);

  // 1a, 2a.  Handle lower-precision architectures by scaling up the fractional part
  // We assume MISS_MANT_DIG is much less than 64, MIN_MANT much less than 0.5
  // Could check lo < MIN_LOW instead...

  /*
    This approach works for real values with mantissa very near zero,
    but isn't ideal.  For values near 0.5, say, the low order bits
    are completely lost, since they exceed the precision of the 
    double representation.  This shows up on M1 and ARM architectures
    as the x.5... values all skipped, because they are indistinguishable 
    from x.5 exactly.

    A more involved alternative would be to separate the
    "frac" and "low" values in the caller.  Then the underflow
    rescaling could be applied to the low bits only, 
    before adding to the frac part.
    
    To do this the signature of this function would have to be
       Check (cld dec, cld frac, int64_t intPart, int64_t low);
                                                  ^- Note this signed
    The caller Check (intPart) would look like

      Check (v, 0.0L, intPart,  0x0LL);
      Check (v, 0.0L, intPart,  0x1LL);
      Check (v, 0.0L, intPart,  0x2LL);
      ...
      Check (v, 0.5L, intPart, -0xFLL);
      Check (v, 0.5L, intPart, -0xELL);
      ...
      Check (v, 0.5L, intPart,  0x0LL);
      Check (v, 0.5L, intPart,  0x1LL);

    Here we would construct value as
      long double lowLd = (double)low / std::pow(2.0L, 64);
      value = dec + frac + lowLd;

    For underflow cases:
      value = dec + frac + std::max::(lowLd, MIN_MANT);
  */


  bool under = false;
  if (frac && (frac < MIN_MANT))
    {
      under = true;
      value = dec + std::max(frac * MIN_LOW, MIN_MANT);
      expect = int64x64_t (intPart, lo * MIN_LOW);
    }

  // 3.  The actual value of the conversion
  const int64x64_t result = int64x64_t (value);

  // 4.  Absolute error in the conversion
  const int64x64_t delta = Abs (result - expect);

  // Mark repeats (no change in input floating value) as "skip" (but not integers)
  const bool skip = (frac && (value == m_last));
  // Save the value to detect unchanged values next time
  m_last = value;

  // 5.  Tolerance for the test, scaled to the magnitude of value
  // Tolerance will be computed from the value, epsilon and margin
  int64x64_t tolerance;

  // Default epsilon
  long double epsilon = std::numeric_limits<long double>::epsilon ();

  // A few cases need extra tolerance
  // If you add cases please thoroughly document the configuration
  long double margin = 0;

  if (int64x64_t::implementation == int64x64_t::ld_impl)
    {
      // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
      margin = 1.0;
    }
  if (RUNNING_ON_VALGRIND)
    {
      // Valgrind uses 64-bit doubles for long doubles
      // See ns-3 bug 1882
      // Need non-zero margin to ensure final tolerance is non-zero
      margin = 1.0;
      epsilon = std::numeric_limits<double>::epsilon ();
    }

  // Final tolerance amount
  tolerance = std::max (1.0L, std::fabs (value)) * epsilon  +  margin * epsilon;

  // 6.  Is the conversion acceptably close to the expected value?
  const bool pass = delta <= tolerance;

  // 7.  Show the result of this check

  // Save stream format flags
  std::ios_base::fmtflags ff = std::cout.flags ();
  std::cout << std::fixed << std::setprecision (22);

  std::cout << GetParent ()->GetName () << " Double: "
            << (skip ? "skip " : (pass ? "pass " : "FAIL "))
            << std::showpos << value << " == "
            << Printer (result)
            << (under ? " (underflow)" : "")
            << std::endl;

  if ( delta )
    {
      // There was a difference, show the expected value
      std::cout << GetParent ()->GetName ()
                << std::left  << std::setw (43) << "         expected"
                << std::right << Printer (expect)
                << std::endl;

      if (delta == tolerance)
        {
          // Short form: show the delta, and note it equals the tolerance
          std::cout << GetParent ()->GetName ()
                    << std::left  << std::setw (43) << "         delta = tolerance"
                    << std::right << Printer (delta)
                    << std::endl;
        }
      else
        {
          // Long form, show both delta and tolerance
          std::cout << GetParent ()->GetName ()
                    << std::left  << std::setw (43) << "         delta"
                    << std::right << Printer (delta)
                    << std::endl;
          std::cout << GetParent ()->GetName ()
                    << std::left  << std::setw (43) << "         tolerance"
                    << std::right << Printer (tolerance)
                    << " eps: " << epsilon << ", margin: " << margin
                    << std::endl;
        }

      // Record number and max delta
      ++m_deltaCount;

      if ( delta > m_deltaMax )
        {
          m_deltaMax = delta;
        }
    }

  // Report pass/fail
  NS_TEST_ASSERT_MSG_EQ_TOL (result, expect, tolerance,
                             "int64x64_t (long double) failed");
  std::cout.flags (ff);

}

void
Int64x64DoubleTestCase::Check (const int64_t intPart)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Double: "
            << "integer: " << intPart
            << std::endl;
  // Reset last value for new intPart
  m_last = intPart;
  // Save current number and max delta, so we can report max from just this intPart
  int64x64_t  deltaMaxPrior = m_deltaMax;
  m_deltaMax = 0;
  int deltaCountPrior = m_deltaCount;
  m_deltaCount = 0;

  // Nudging the integer part eliminates deltas around 0
  long double v = intPart;

  Check (v, 0.0L,                      intPart,                0x0ULL);
  Check (v, 0.0000000000000000000542L, intPart,                0x1ULL);
  Check (v, 0.0000000000000000001084L, intPart,                0x2ULL);
  Check (v, 0.0000000000000000001626L, intPart,                0x3ULL);
  Check (v, 0.0000000000000000002168L, intPart,                0x4ULL);
  Check (v, 0.0000000000000000002711L, intPart,                0x5ULL);
  Check (v, 0.0000000000000000003253L, intPart,                0x6ULL);
  Check (v, 0.0000000000000000003795L, intPart,                0x7ULL);
  Check (v, 0.0000000000000000004337L, intPart,                0x8ULL);
  Check (v, 0.0000000000000000004879L, intPart,                0x9ULL);
  Check (v, 0.0000000000000000005421L, intPart,                0xAULL);
  Check (v, 0.0000000000000000005963L, intPart,                0xBULL);
  Check (v, 0.0000000000000000006505L, intPart,                0xCULL);
  Check (v, 0.0000000000000000007047L, intPart,                0xDULL);
  Check (v, 0.0000000000000000007589L, intPart,                0xEULL);
  Check (v, 0.0000000000000000008132L, intPart,                0xFULL);
  Check (v, 0.0000000000000000130104L, intPart,               0xF0ULL);
  Check (v, 0.0000000000000002081668L, intPart,              0xF00ULL);
  Check (v, 0.0000000000000033306691L, intPart,             0xF000ULL);
  Check (v, 0.0000000000000532907052L, intPart,            0xF0000ULL);
  Check (v, 0.0000000000008526512829L, intPart,           0xF00000ULL);
  Check (v, 0.0000000000136424205266L, intPart,          0xF000000ULL);
  Check (v, 0.0000000002182787284255L, intPart,         0xF0000000ULL);
  Check (v, 0.0000000034924596548080L, intPart,        0xF00000000ULL);
  Check (v, 0.0000000558793544769287L, intPart,       0xF000000000ULL);
  Check (v, 0.0000008940696716308594L, intPart,      0xF0000000000ULL);
  Check (v, 0.0000143051147460937500L, intPart,     0xF00000000000ULL);
  Check (v, 0.0002288818359375000000L, intPart,    0xF000000000000ULL);
  Check (v, 0.0036621093750000000000L, intPart,   0xF0000000000000ULL);
  Check (v, 0.0585937500000000000000L, intPart,  0xF00000000000000ULL);
  std::cout << std::endl;
  Check (v, 0.4999999999999999991326L, intPart, 0x7FFFFFFFFFFFFFF0ULL);
  Check (v, 0.4999999999999999991868L, intPart, 0x7FFFFFFFFFFFFFF1ULL);
  Check (v, 0.4999999999999999992411L, intPart, 0x7FFFFFFFFFFFFFF2ULL);
  Check (v, 0.4999999999999999992953L, intPart, 0x7FFFFFFFFFFFFFF3ULL);
  Check (v, 0.4999999999999999993495L, intPart, 0x7FFFFFFFFFFFFFF4ULL);
  Check (v, 0.4999999999999999994037L, intPart, 0x7FFFFFFFFFFFFFF5ULL);
  Check (v, 0.4999999999999999994579L, intPart, 0x7FFFFFFFFFFFFFF6ULL);
  Check (v, 0.4999999999999999995121L, intPart, 0x7FFFFFFFFFFFFFF7ULL);
  Check (v, 0.4999999999999999995663L, intPart, 0x7FFFFFFFFFFFFFF8ULL);
  Check (v, 0.4999999999999999996205L, intPart, 0x7FFFFFFFFFFFFFF9ULL);
  Check (v, 0.4999999999999999996747L, intPart, 0x7FFFFFFFFFFFFFFAULL);
  Check (v, 0.4999999999999999997289L, intPart, 0x7FFFFFFFFFFFFFFBULL);
  Check (v, 0.4999999999999999997832L, intPart, 0x7FFFFFFFFFFFFFFCULL);
  Check (v, 0.4999999999999999998374L, intPart, 0x7FFFFFFFFFFFFFFDULL);
  Check (v, 0.4999999999999999998916L, intPart, 0x7FFFFFFFFFFFFFFEULL);
  Check (v, 0.4999999999999999999458L, intPart, 0x7FFFFFFFFFFFFFFFULL);
  Check (v, 0.5000000000000000000000L, intPart, 0x8000000000000000ULL);
  Check (v, 0.5000000000000000000542L, intPart, 0x8000000000000001ULL);
  Check (v, 0.5000000000000000001084L, intPart, 0x8000000000000002ULL);
  Check (v, 0.5000000000000000001626L, intPart, 0x8000000000000003ULL);
  Check (v, 0.5000000000000000002168L, intPart, 0x8000000000000004ULL);
  Check (v, 0.5000000000000000002711L, intPart, 0x8000000000000005ULL);
  Check (v, 0.5000000000000000003253L, intPart, 0x8000000000000006ULL);
  Check (v, 0.5000000000000000003795L, intPart, 0x8000000000000007ULL);
  Check (v, 0.5000000000000000004337L, intPart, 0x8000000000000008ULL);
  Check (v, 0.5000000000000000004879L, intPart, 0x8000000000000009ULL);
  Check (v, 0.5000000000000000005421L, intPart, 0x800000000000000AULL);
  Check (v, 0.5000000000000000005963L, intPart, 0x800000000000000BULL);
  Check (v, 0.5000000000000000006505L, intPart, 0x800000000000000CULL);
  Check (v, 0.5000000000000000007047L, intPart, 0x800000000000000DULL);
  Check (v, 0.5000000000000000007589L, intPart, 0x800000000000000EULL);
  Check (v, 0.5000000000000000008132L, intPart, 0x800000000000000FULL);
  std::cout << std::endl;
  Check (v, 0.9375000000000000000000L, intPart, 0xF000000000000000ULL);
  Check (v, 0.9960937500000000000000L, intPart, 0xFF00000000000000ULL);
  Check (v, 0.9997558593750000000000L, intPart, 0xFFF0000000000000ULL);
  Check (v, 0.9999847412109375000000L, intPart, 0xFFFF000000000000ULL);
  Check (v, 0.9999990463256835937500L, intPart, 0xFFFFF00000000000ULL);
  Check (v, 0.9999999403953552246094L, intPart, 0xFFFFFF0000000000ULL);
  Check (v, 0.9999999962747097015381L, intPart, 0xFFFFFFF000000000ULL);
  Check (v, 0.9999999997671693563461L, intPart, 0xFFFFFFFF00000000ULL);
  Check (v, 0.9999999999854480847716L, intPart, 0xFFFFFFFFF0000000ULL);
  Check (v, 0.9999999999990905052982L, intPart, 0xFFFFFFFFFF000000ULL);
  Check (v, 0.9999999999999431565811L, intPart, 0xFFFFFFFFFFF00000ULL);
  Check (v, 0.9999999999999964472863L, intPart, 0xFFFFFFFFFFFF0000ULL);
  Check (v, 0.9999999999999997779554L, intPart, 0xFFFFFFFFFFFFF000ULL);
  Check (v, 0.9999999999999999861222L, intPart, 0xFFFFFFFFFFFFFF00ULL);
  Check (v, 0.9999999999999999991326L, intPart, 0xFFFFFFFFFFFFFFF0ULL);
  Check (v, 0.9999999999999999991868L, intPart, 0xFFFFFFFFFFFFFFF1ULL);
  Check (v, 0.9999999999999999992411L, intPart, 0xFFFFFFFFFFFFFFF2ULL);
  Check (v, 0.9999999999999999992943L, intPart, 0xFFFFFFFFFFFFFFF3ULL);
  Check (v, 0.9999999999999999993495L, intPart, 0xFFFFFFFFFFFFFFF4ULL);
  Check (v, 0.9999999999999999994037L, intPart, 0xFFFFFFFFFFFFFFF5ULL);
  Check (v, 0.9999999999999999994579L, intPart, 0xFFFFFFFFFFFFFFF6ULL);
  Check (v, 0.9999999999999999995121L, intPart, 0xFFFFFFFFFFFFFFF7ULL);
  Check (v, 0.9999999999999999995663L, intPart, 0xFFFFFFFFFFFFFFF8ULL);
  Check (v, 0.9999999999999999996205L, intPart, 0xFFFFFFFFFFFFFFF9ULL);
  Check (v, 0.9999999999999999996747L, intPart, 0xFFFFFFFFFFFFFFFAULL);
  Check (v, 0.9999999999999999997289L, intPart, 0xFFFFFFFFFFFFFFFBULL);
  Check (v, 0.9999999999999999997832L, intPart, 0xFFFFFFFFFFFFFFFCULL);
  Check (v, 0.9999999999999999998374L, intPart, 0xFFFFFFFFFFFFFFFDULL);
  Check (v, 0.9999999999999999998916L, intPart, 0xFFFFFFFFFFFFFFFEULL);
  Check (v, 0.9999999999999999999458L, intPart, 0xFFFFFFFFFFFFFFFFULL);

  std::cout << GetParent ()->GetName () << " Double: "
            << "integer:"  << std::setw (4) << intPart
            << ": deltas:" << std::setw (4) << m_deltaCount
            << ", max:   " << Printer (m_deltaMax)
            << std::endl;

  // Add the count, max from this intPart to the grand totals
  m_deltaCount += deltaCountPrior;
  m_deltaMax = Max (m_deltaMax, deltaMaxPrior);
}

void
Int64x64DoubleTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Double: " << GetName ()
            << std::endl;

  // Save stream format flags
  std::ios_base::fmtflags ff = std::cout.flags ();

  std::cout << GetParent ()->GetName () << " Double: "
            <<                    "FLT_RADIX:     " << FLT_RADIX
            << "\n                 LDBL_MANT_DIG: " << LDBL_MANT_DIG
            << "\n                 MISS_MANT_DIG: " << MISS_MANT_DIG
            << "\n                 MIN_LOW:       " << Printer (MIN_LOW)
            << " (" << std::hexfloat << MIN_LOW << ")" << std::defaultfloat
            << "\n                 MIN_MANT:      " << Printer (MIN_MANT)
            << std::endl;

  std::cout << std::scientific << std::setprecision (21);

  Check (-2);
  Check (-1);
  Check ( 0);
  Check ( 1);
  Check ( 2);

  std::cout << GetParent ()->GetName () << " Double: "
            << "Total deltas:" << std::setw (7) << m_deltaCount
            << ", max delta:  " << Printer (m_deltaMax)
            << std::endl;

  std::cout.flags (ff);
}


/**
 * \ingroup int64x64-tests
 * 
 * Test: print the implementation
 */
class Int64x64ImplTestCase : public TestCase
{
public:
  Int64x64ImplTestCase ();
  virtual void DoRun (void);
};

Int64x64ImplTestCase::Int64x64ImplTestCase ()
  : TestCase ("Print the implementation")
{}

void
Int64x64ImplTestCase::DoRun (void)
{
  std::cout << std::endl;
  std::cout << GetParent ()->GetName () << " Impl: " << GetName ()
            << std::endl;


  std::cout << "int64x64_t::implementation: ";
  switch (int64x64_t::implementation)
    {
      /* *NS_CHECK_STYLE_OFF* */
    case (int64x64_t::int128_impl) : std::cout << "int128_impl"; break;
    case (int64x64_t::cairo_impl)  : std::cout << "cairo_impl";  break;
    case (int64x64_t::ld_impl)     : std::cout << "ld_impl";     break;
    default :                        std::cout << "unknown!";
      /* *NS_CHECK_STYLE_ON* */
    }
  std::cout << std::endl;

#if defined (INT64X64_USE_CAIRO) && !defined (PYTHON_SCAN)
  std::cout << "cairo_impl64:  " << cairo_impl64 << std::endl;
  std::cout << "cairo_impl128: " << cairo_impl128 << std::endl;
#endif

  if (RUNNING_ON_VALGRIND != 0)
    {
      std::cout << "Running with valgrind" << std::endl;
    }

}

/**
 * \ingroup int64x64-tests
 * \internal
 *
 * The int64x64 Test Suite.
 * 
 * Some of these tests are a little unusual for ns-3 in that they
 * are sensitive to implementation, specifically the resolution
 * of the double and long double implementations.
 *
 * To handle this, where needed we define a tolerance to use in the
 * test comparisons.  If you need to increase the tolerance,
 * please append the system and compiler version.  For example:
 *
 * \code
 *   // Darwin 12.5.0 (Mac 10.8.5) g++ 4.2.1
 *   tolerance = 1;
 *   // System Foo gcc 3.9
 *   tolerance = 3;
 * \endcode
 */
class Int64x64TestSuite : public TestSuite
{
public:
  Int64x64TestSuite ()
    : TestSuite ("int64x64", UNIT)
  {
    AddTestCase (new Int64x64ImplTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64HiLoTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64IntRoundTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64ArithmeticTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64CompareTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64InputTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64InputOutputTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64Bug455TestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64Bug863TestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64Bug1786TestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64InvertTestCase (), TestCase::QUICK);
    AddTestCase (new Int64x64DoubleTestCase (), TestCase::QUICK);
  }
};

static Int64x64TestSuite g_int64x64TestSuite; //!< Static variable for test initialization

}  // namespace test

}  // namespace int64x64

}  // namespace ns3

