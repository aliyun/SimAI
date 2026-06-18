/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#ifndef NS3_TEST_H
#define NS3_TEST_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <limits>
#include <stdint.h>

#include "system-wall-clock-ms.h"

/**
 * \file
 * \ingroup testing
 * \brief ns3::TestCase, ns3::TestSuite, ns3::TestRunner declarations,
 * and \c NS_TEST_ASSERT macro definitions.
 */

/**
 * \ingroup core
 * \defgroup testing Testing
 * \brief Tools to define and execute unit tests.
 *
 * This module lists the normal Testing API.  Most of these
 * macros forward to the implementation macros in testingimpl.
 * You should generally use these macros only.
 */
/**
 * \ingroup testing
 * \defgroup testingimpl Testing Implementation
 * \brief Internal implementation of the Testing system.
 */

namespace ns3 {

/** Namespace for test files, TestCases and TestSuites. */
namespace tests {} // namespace tests

//
// Note on below macros:
//
// When multiple statements are used in a macro, they should be bound
// together in a loop syntactically, so the macro can appear safely
// inside if clauses or other places that expect a single statement or
// a statement block.  The "strange" do while construct is a generally
// expected best practice for defining a robust macro.
//

/**
 * \ingroup testing
 * \brief Check if we should assert on errors, and do so
 */
#define ASSERT_ON_FAILURE                                               \
  do {                                                                  \
      if (MustAssertOnFailure ())                                       \
        {                                                               \
          *(volatile int *)0 = 0;                                       \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 * \brief If we shouldn't continue on errors, return
 */
#define CONTINUE_ON_FAILURE                                             \
  do {                                                                  \
      if (!MustContinueOnFailure ())                                    \
        {                                                               \
          return;                                                       \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 * \brief If we shouldn't continue on errors, return test status
 */
#define CONTINUE_ON_FAILURE_RETURNS_BOOL                                \
  do {                                                                  \
      if (!MustContinueOnFailure ())                                    \
        {                                                               \
          return IsStatusFailure ();                                    \
        }                                                               \
    } while (false)



// ===========================================================================
// Test for equality (generic version)
// ===========================================================================

/**
 * \ingroup testing
 *
 * \brief Test that an actual and expected (limit) value are equal and
 * report and abort if not.
 *
 * Check to see if the expected (limit) value is equal to the actual
 * value found in a test case.  If the two values are equal nothing
 * happens, but if the comparison fails, an error is reported in a
 * consistent way and the execution of the current test case is
 * aborted.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 * NS_TEST_ASSERT_MSG_EQ (result, true,
 *      "cannot open file " << filename << " in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the expected value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 *
 * \warning Do not use this macro if you are comparing floating point
 * numbers (float or double) as it is unlikely to do what you expect.
 * Use NS_TEST_ASSERT_MSG_EQ_TOL instead.
 */
#define NS_TEST_ASSERT_MSG_EQ(actual, limit, msg)                       \
  do {                                                                  \
      if (!((actual) == (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) == " +  \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE;                                          \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual and expected (limit) value are equal and
 * report and abort if not.
 *
 * Check to see if the expected (limit) value is equal to the actual
 * value found in a test case.  If the two values are equal nothing
 * happens, but if the comparison fails, an error is reported in a
 * consistent way and the execution of the current test case is
 * aborted.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 * NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (result, true,
 *      "cannot open file " << filename << " in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the expected value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 *
 * \warning Do not use this macro if you are comparing floating point
 * numbers (float or double) as it is unlikely to do what you expect.
 * Use NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL_TOL instead.
 *
 * This function returns a Boolean value.
 *
 */
#define NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL(actual, limit, msg)          \
  do {                                                                  \
      if (!((actual) == (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) == " +  \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE_RETURNS_BOOL;                             \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual and expected (limit) value are equal and
 * report if not.
 *
 * Check to see if the expected (lmit) value is equal to the actual
 * value found in a test case.  If the two values are equal nothing
 * happens, but if the comparison fails, an error is reported in a
 * consistent way.  EXPECT* macros do not return if an error is
 * detected.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 * NS_TEST_EXPECT_MSG_EQUAL (result, true,
 *      "cannot open file " << filename << " in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the expected value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 *
 * \warning Do not use this macro if you are comparing floating point
 * numbers (float or double) as it is unlikely to do what you expect.
 * Use NS_TEST_EXPECT_MSG_EQ_TOL instead.
 */
#define NS_TEST_EXPECT_MSG_EQ(actual, limit, msg)                       \
  do {                                                                  \
      if (!((actual) == (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) == " +  \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
        }                                                               \
    } while (false)


// ===========================================================================
// Test for equality with a provided tolerance (use for floating point
// comparisons -- both float and double)
// ===========================================================================

/**
 * \ingroup testing
 *
 * \brief Test that actual and expected (limit) values are equal to
 * plus or minus some tolerance and report and abort if not.
 *
 * Check to see if the expected (limit) value is equal to the actual
 * value found in a test case to some tolerance.  This is not the same
 * thing as asking if two floating point are equal to within some
 * epsilon, but is useful for that case.  This assertion is geared
 * toward more of a measurement problem.  Consider measuring a
 * physical rod of some kind that you have ordered.  You need to
 * determine if it is "good."  You want to measure the rod to an
 * arbitrary precision of sixteen significant figures, you will
 * measure the rod to determine if its length is within the tolerances
 * you provided.  For example, 12.00 inches plus or minus .005 inch
 * may be just fine.
 *
 * In ns-3, you might want to measure a signal to noise ratio and
 * check to see if the answer is what you expect.  If you naively
 * measure (double)1128.93 and compare this number with a constant
 * 1128.93 you are almost certainly going to have your test fail
 * because of floating point rounding errors.  We provide a floating
 * point comparison function ns3::TestDoubleIsEqual() but you will
 * probably quickly find that is not what you want either.  It may
 * turn out to be the case that when you measured an SNR that printed
 * as 1128.93, what was actually measured was something more like
 * 1128.9287653857625442 for example.  Given that the double epsilon
 * is on the order of 0.0000000000000009, you would need to provide
 * sixteen significant figures of expected value for this kind of test
 * to pass even with a typical test for floating point "approximate
 * equality."  That is clearly not required or desired.  You really
 * want to be able to provide 1128.93 along with a tolerance just like
 * you provided 12 inches +- 0.005 inch above.
 *
 * This assertion is designed for real measurements by taking into
 * account measurement tolerances.  By doing so it also automatically
 * compensates for floating point rounding errors.  If you really want
 * to check floating point equality down to the
 * numeric_limits<double>::epsilon () range, consider using
 * ns3::TestDoubleIsEqual().
 *
 * \note Mixing signed and unsigned types can lead to misleading
 * results.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 *   NS_TEST_ASSERT_MSG_EQ_TOL (snr, 1128.93, 0.005,
 *                              "wrong snr (" << snr << ") in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the expected value of the test.
 * \param [in] tol Tolerance of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_ASSERT_MSG_EQ_TOL(actual, limit, tol, msg)              \
  do {                                                                  \
      if ((actual) > (limit) + (tol) || (actual) < (limit) - (tol))     \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit << " +- " << tol;                        \
          std::ostringstream condStream;                                \
          condStream << #actual << " (actual) < " << #limit             \
                     << " (limit) + " << #tol << " (tol) && "           \
                     << #actual << " (actual) > " << #limit             \
                     << " (limit) - " << #tol << " (tol)";              \
          ReportTestFailure (condStream.str (), actualStream.str (),    \
                             limitStream.str (), msgStream.str (),      \
                             __FILE__, __LINE__);                       \
          CONTINUE_ON_FAILURE;                                          \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that actual and expected (limit) values are equal to
 * plus or minus some tolerance and report and abort if not.
 *
 * Check to see if the expected (limit) value is equal to the actual
 * value found in a test case to some tolerance.  This is not the same
 * thing as asking if two floating point are equal to within some
 * epsilon, but is useful for that case.  This assertion is geared
 * toward more of a measurement problem.  Consider measuring a
 * physical rod of some kind that you have ordered.  You need to
 * determine if it is "good."  You want to measure the rod to an
 * arbitrary precision of sixteen significant figures, you will
 * measure the rod to determine if its length is within the tolerances
 * you provided.  For example, 12.00 inches plus or minus .005 inch
 * may be just fine.
 *
 * In ns-3, you might want to measure a signal to noise ratio and
 * check to see if the answer is what you expect.  If you naively
 * measure (double)1128.93 and compare this number with a constant
 * 1128.93 you are almost certainly going to have your test fail
 * because of floating point rounding errors.  We provide a floating
 * point comparison function ns3::TestDoubleIsEqual() but you will
 * probably quickly find that is not what you want either.  It may
 * turn out to be the case that when you measured an SNR that printed
 * as 1128.93, what was actually measured was something more like
 * 1128.9287653857625442 for example.  Given that the double epsilon
 * is on the order of 0.0000000000000009, you would need to provide
 * sixteen significant figures of expected value for this kind of test
 * to pass even with a typical test for floating point "approximate
 * equality."  That is clearly not required or desired.  You really
 * want to be able to provide 1128.93 along with a tolerance just like
 * you provided 12 inches +- 0.005 inch above.
 *
 * This assertion is designed for real measurements by taking into
 * account measurement tolerances.  By doing so it also automatically
 * compensates for floating point rounding errors.  If you really want
 * to check floating point equality down to the
 * numeric_limits<double>::epsilon () range, consider using
 * ns3::TestDoubleIsEqual().
 *
 * \note Mixing signed and unsigned types can lead to misleading
 * results.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 *   NS_TEST_ASSERT_MSG_EQ_TOL_RETURNS_BOOL (snr, 1128.93, 0.005,
 *                                           "wrong snr (" << snr << ") in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the expected value of the test.
 * \param [in] tol Tolerance of the test.
 * \param [in] msg Message that is output if the test does not pass.
 *
 * This function returns a Boolean value.
 *
 */
#define NS_TEST_ASSERT_MSG_EQ_TOL_RETURNS_BOOL(actual, limit, tol, msg) \
  do {                                                                  \
      if ((actual) > (limit) + (tol) || (actual) < (limit) - (tol))     \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit << " +- " << tol;                        \
          std::ostringstream condStream;                                \
          condStream << #actual << " (actual) < " << #limit             \
                     << " (limit) + " << #tol << " (tol) && "           \
                     << #actual << " (actual) > " << #limit             \
                     << " (limit) - " << #tol << " (tol)";              \
          ReportTestFailure (condStream.str (), actualStream.str (),    \
                             limitStream.str (), msgStream.str (),      \
                             __FILE__, __LINE__);                       \
          CONTINUE_ON_FAILURE_RETURNS_BOOL;                             \
        }                                                               \
    } while (false)


/**
 * \ingroup testing
 *
 * \brief Test that actual and expected (limit) values are equal to
 * plus or minus some tolerance and report if not.
 *
 * Check to see if the expected (limit) value is equal to the actual
 * value found in a test case to some tolerance.  This is not the same
 * thing as asking if two floating point are equal to within some
 * epsilon, but is useful for that case.  This assertion is geared
 * toward more of a measurement problem.  Consider measuring a
 * physical rod of some kind that you have ordered.  You need to
 * determine if it is "good."  You want to measure the rod to an
 * arbitrary precision of sixteen significant figures, you will
 * measure the rod to determine if its length is within the tolerances
 * you provided.  For example, 12.00 inches plus or minus .005 inch
 * may be just fine.
 *
 * In ns-3, you might want to measure a signal to noise ratio and
 * check to see if the answer is what you expect.  If you naively
 * measure (double)1128.93 and compare this number with a constant
 * 1128.93 you are almost certainly going to have your test fail
 * because of floating point rounding errors.  We provide a floating
 * point comparison function ns3::TestDoubleIsEqual() but you will
 * probably quickly find that is not what you want either.  It may
 * turn out to be the case that when you measured an SNR that printed
 * as 1128.93, what was actually measured was something more like
 * 1128.9287653857625442 for example.  Given that the double epsilon
 * is on the order of 0.0000000000000009, you would need to provide
 * sixteen significant figures of expected value for this kind of test
 * to pass even with a typical test for floating point "approximate
 * equality."  That is clearly not required or desired.  You really
 * want to be able to provide 1128.93 along with a tolerance just like
 * you provided 12 inches +- 0.005 inch above.
 *
 * This assertion is designed for real measurements by taking into
 * account measurement tolerances.  By doing so it also automatically
 * compensates for floating point rounding errors.  If you really want
 * to check floating point equality down to the
 * numeric_limits<double>::epsilon () range, consider using
 * ns3::TestDoubleIsEqual().
 *
 * \note Mixing signed and unsigned types can lead to misleading
 * results.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 *   NS_TEST_EXPECT_MSG_EQ_TOL (snr, 1128.93, 0.005,
 *                              "wrong snr (" << snr << ") in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the expected value of the test.
 * \param [in] tol Tolerance of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_EXPECT_MSG_EQ_TOL(actual, limit, tol, msg)              \
  do {                                                                  \
      if ((actual) > (limit) + (tol) || (actual) < (limit) - (tol))     \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit << " +- " << tol;                        \
          std::ostringstream condStream;                                \
          condStream << #actual << " (actual) < " << #limit             \
                     << " (limit) + " << #tol << " (tol) && "           \
                     << #actual << " (actual) > " << #limit             \
                     << " (limit) - " << #tol << " (tol)";              \
          ReportTestFailure (condStream.str (), actualStream.str (),    \
                             limitStream.str (), msgStream.str (),      \
                             __FILE__, __LINE__);                       \
        }                                                               \
    } while (false)

// ===========================================================================
// Test for inequality
// ===========================================================================

/**
 * \ingroup testing
 *
 * \brief Test that an actual and expected (limit) value are not equal
 * and report and abort if not.
 *
 * Check to see if the expected (limit) value is not equal to the
 * actual value found in a test case.  If the two values are not equal
 * nothing happens, but if the comparison fails, an error is reported
 * in a consistent way and the execution of the current test case is
 * aborted.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 * NS_TEST_ASSERT_MSG_NE (result, false,
 *      "cannot open file " << filename << " in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the value that actual is tested against.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_ASSERT_MSG_NE(actual, limit, msg)                       \
  do {                                                                  \
      if (!((actual) != (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) != " +  \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE;                                          \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual and expected (limit) value are not equal
 * and report and abort if not.
 *
 * Check to see if the expected (limit) value is not equal to the
 * actual value found in a test case.  If the two values are equal
 * nothing happens, but if the comparison fails, an error is reported
 * in a consistent way and the execution of the current test case is
 * aborted.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 * NS_TEST_ASSERT_MSG_NE_RETURNS_BOOL (result, false,
 *      "cannot open file " << filename << " in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the expected value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 *
 * This function returns a Boolean value.
 *
 */
#define NS_TEST_ASSERT_MSG_NE_RETURNS_BOOL(actual, limit, msg)          \
  do {                                                                  \
      if (!((actual) != (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) != " +  \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE_RETURNS_BOOL;                             \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual and expected (limit) value are not equal
 * and report if not.
 *
 * Check to see if the expected (limit) value is not equal to the
 * actual value found in a test case.  If the two values are not equal
 * nothing happens, but if the comparison fails, an error is reported
 * in a consistent way.  EXPECT* macros do not return if an error is
 * detected.
 *
 * The message is interpreted as a stream, for example:
 *
 * \code
 * NS_TEST_EXPECT_MSG_NE (result, false,
 *      "cannot open file " << filename << " in test");
 * \endcode
 *
 * is legal.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the value that actual is tested against.
 * \param [in] msg Message that is output if the test does not pass.
 *
 * \warning Do not use this macro if you are comparing floating point
 * numbers (float or double).  Use NS_TEST_EXPECT_MSG_FLNE instead.
 */
#define NS_TEST_EXPECT_MSG_NE(actual, limit, msg) \
  do {                                                                  \
      if (!((actual) != (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) != " +  \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
        }                                                               \
    } while (false)

// ===========================================================================
// Test for less than relation
// ===========================================================================

/**
 * \ingroup testing
 *
 * \brief Test that an actual value is less than a limit and report
 * and abort if not.
 *
 * Check to see if the actual value found in a test case is less than
 * the limit value.  If the actual value is lesser nothing happens,
 * but if the check fails, an error is reported in a consistent way
 * and the execution of the current test case is aborted.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_ASSERT_MSG_LT(actual, limit, msg)                       \
  do {                                                                  \
      if (!((actual) < (limit)))                                        \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) < " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE;                                          \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual value is less than or equal to a limit
 * and report and abort if not.
 *
 * Check to see if the actual value found in a test case is less than
 * or equal to the limit value.  If the actual value is lesser or
 * equal nothing happens, but if the check fails, an error is reported
 * in a consistent way and the execution of the current test case is
 * aborted.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_ASSERT_MSG_LT_OR_EQ(actual, limit, msg)                 \
  do {                                                                  \
      if (!((actual) <= (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) < " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE;                                          \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual value is less than a limit and report if
 * not.
 *
 * Check to see if the actual value found in a test case is less than
 * the limit value.  If the actual value is lesser nothing happens,
 * but if the check fails, an error is reported in a consistent way.
 * EXPECT* macros do not return if an error is detected.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_EXPECT_MSG_LT(actual, limit, msg)                       \
  do {                                                                  \
      if (!((actual) < (limit)))                                        \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) < " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
        }                                                               \
    } while (false)


/**
 * \ingroup testing
 *
 * \brief Test that an actual value is less than or equal to a limit
 * and report if not.
 *
 * Check to see if the actual value found in a test case is less than
 * or equal to the limit value.  If the actual value is lesser or
 * equal nothing happens, but if the check fails, an error is reported
 * in a consistent way.  EXPECT* macros do not return if an error is
 * detected.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_EXPECT_MSG_LT_OR_EQ(actual, limit, msg) \
  do {                                                                  \
      if (!((actual) <= (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) < " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
        }                                                               \
    } while (false)

// ===========================================================================
// Test for greater than relation
// ===========================================================================

/**
 * \ingroup testing
 *
 * \brief Test that an actual value is greater than a limit and report
 * and abort if not.
 *
 * Check to see if the actual value found in a test case is greater
 * than the limit value.  If the actual value is greater nothing
 * happens, but if the check fails, an error is reported in a
 * consistent way and the execution of the current test case is
 * aborted.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_ASSERT_MSG_GT(actual, limit, msg)                       \
  do {                                                                  \
      if (!((actual) > (limit)))                                        \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) > " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE;                                          \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual value is greater than or equal to a
 * limit and report and abort if not.
 *
 * Check to see if the actual value found in a test case is greater
 * than or equal to the limit value.  If the actual value is greater
 * nothing happens, but if the check fails, an error is reported in a
 * consistent way and the execution of the current test case is
 * aborted.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_ASSERT_MSG_GT_OR_EQ(actual, limit, msg)                 \
  do {                                                                  \
      if (!((actual) >= (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) > " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
          CONTINUE_ON_FAILURE;                                          \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual value is greater than a limit and report
 * if not.
 *
 * Check to see if the actual value found in a test case is greater
 * than the limit value.  If the actual value is greater nothing
 * happens, but if the check fails, an error is reported in a
 * consistent way.  EXPECT* macros do not return if an error is
 * detected.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_EXPECT_MSG_GT(actual, limit, msg)                       \
  do {                                                                  \
      if (!((actual) > (limit)))                                        \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) > " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 *
 * \brief Test that an actual value is greater than or equal to limit
 * and report if not.
 *
 * Check to see if the actual value found in a test case is greater
 * than or equal to the limit value.  If the actual value is greater
 * nothing happens, but if the check fails, an error is reported in a
 * consistent way.  EXPECT* macros do not return if an error is
 * detected.
 *
 * The message is interpreted as a stream.
 *
 * \param [in] actual Expression for the actual value found during the test.
 * \param [in] limit Expression for the limit value of the test.
 * \param [in] msg Message that is output if the test does not pass.
 */
#define NS_TEST_EXPECT_MSG_GT_OR_EQ(actual, limit, msg)                 \
  do {                                                                  \
      if (!((actual) >= (limit)))                                       \
        {                                                               \
          ASSERT_ON_FAILURE;                                            \
          std::ostringstream msgStream;                                 \
          msgStream << msg;                                             \
          std::ostringstream actualStream;                              \
          actualStream << actual;                                       \
          std::ostringstream limitStream;                               \
          limitStream << limit;                                         \
          ReportTestFailure (std::string (#actual) + " (actual) > " +   \
                             std::string (#limit) + " (limit)",         \
                             actualStream.str (), limitStream.str (),   \
                             msgStream.str (), __FILE__, __LINE__);     \
        }                                                               \
    } while (false)

/**
 * \ingroup testing
 * \brief Compare two double precision floating point numbers and
 * declare them equal if they are within some epsilon of each other.
 *
 * Approximate comparison of floating point numbers near equality is
 * trickier than one may expect and is well-discussed in the
 * literature.  Basic strategies revolve around a suggestion by Knuth
 * to compare the floating point numbers as binary integers, supplying
 * a maximum difference between them .  This max difference is
 * specified in Units in the Last Place (ulps) or a floating point
 * epsilon.
 *
 * This routine is based on the GNU Scientific Library function
 * gsl_fcmp.
 *
 * \param [in] a The first of double precision floating point
 *               numbers to compare
 * \param [in] b The second of double precision floating point
 *               numbers to compare
 * \param [in] epsilon The tolerance to use in the comparison.
 * \returns Returns \c true if the doubles are equal to a precision
 *          defined by epsilon
 */
bool TestDoubleIsEqual (const double a, const double b,
                        const double epsilon = std::numeric_limits<double>::epsilon ());

class TestRunnerImpl;

/**
 * \ingroup testing
 *
 * \brief encapsulates test code
 *
 * To allow a new test to be run within the ns-3 test framework, users
 * need to create subclasses of this base class, override the DoRun
 * method, and use the NS_TEST_* macros within DoRun.
 *
 * \see sample-test-suite.cc
 */
class TestCase
{
public:
  /** \brief How long the test takes to execute. */
  enum TestDuration
  {
    QUICK         = 1,  //!< Fast test.
    EXTENSIVE     = 2,  //!< Medium length test.
    TAKES_FOREVER = 3   //!< Very long running test.
  };

  /**
   *  Destructor
   */
  virtual ~TestCase ();

  // Delete copy constructor and assignment operator to avoid misuse
  TestCase (const TestCase &) = delete;
  TestCase & operator = (const TestCase &) = delete;

  /**
   * \return The name of this test
   */
  std::string GetName (void) const;

protected:
  /**
   * \brief Constructor.
   *
   * \param [in] name The name of the new TestCase created
   */
  TestCase (std::string name);

  /**
   * \brief Add an individual child TestCase to this test suite.
   *
   * \param [in] testCase Pointer to the TestCase object to be added.
   * \param [in] duration Amount of time this test takes to execute
   *             (defaults to QUICK).
   */
  void AddTestCase (TestCase *testCase, TestDuration duration = QUICK);

  /**
   * \brief Set the data directory where reference trace files can be
   * found.
   *
   * \param [in] directory The directory where the test data is
   * located
   *
   * In general, this method is invoked as SetDataDir
   * (NS_TEST_SOURCEDIR); However, if a module contains a test
   * directory with subdirectories (e.g. src/mesh/test), and the test
   * data (e.g. pcap traces) is located in one of these
   * subdirectories, then the variable NS_TEST_SOURCEDIR may not work
   * and the user may want to explicitly pass in a directory string.
   *
   * Note that NS_TEST_SOURCEDIR is set in src/CMakeLists.txt for each module
   */
  void SetDataDir (std::string directory);

  /**
   * \brief Check if any tests failed.
   *
   * \return \c true if any of the tests have failed, \c false otherwise.
   */
  bool IsStatusFailure (void) const;
  /**
   * \brief Check if all tests passed.
   *
   * \return \c true if the tests have succeeded, \c false otherwise.
   */
  bool IsStatusSuccess (void) const;

  /**
   * \brief Get the parent of this TestCsse.
   *
   * \return A pointer to the parent of this test.
   */
  TestCase * GetParent () const;

  /**
   * \name Internal Interface
   * These methods are the interface used by test macros and should not
   * be used directly by normal test code.
   * @{
   */
  /**
   * \brief Log the failure of this TestCase.
   *
   * \param [in] cond The test condition.
   * \param [in] actual Actual value of the test.
   * \param [in] limit Expected value of the test.
   * \param [in] message Message indicating the type of failure.
   * \param [in] file The file where the test failed.
   * \param [in] line The line number in \pname{file} where the test failed.
   */
  void ReportTestFailure (std::string cond, std::string actual,
                          std::string limit, std::string message,
                          std::string file, int32_t line);
  /**
   * \brief Check if this run should assert on failure.
   *
   * \return \c true if we should assert on failure.
   */
  bool MustAssertOnFailure (void) const;
  /**
   * \brief Check if this run should continue on failure.
   *
   * \return \c true if we should continue on failure.
   */
  bool MustContinueOnFailure (void) const;
  /**
   * \brief Construct the full path to a file in the data directory.
   *
   * The data directory is configured by SetDataDirectory().
   *
   * \param [in] filename The bare (no path) file name
   * \return The full path to \pname{filename} in the data directory
   */
  std::string CreateDataDirFilename (std::string filename);
  /**
   * \brief Construct the full path to a file in a temporary directory.
   *
   *  If the TestRunner is invoked with "--update-data", this will be
   *  the data directory instead.
   *
   * \param [in] filename The bare (no path) file name
   * \return The full path to \pname{filename} in the temporary directory.
   */
  std::string CreateTempDirFilename (std::string filename);
  /**@}*/

private:

  /** Needs access to the TestCase data members. */
  friend class TestRunnerImpl;

  /**
   * \brief Implementation to do any local setup required for this
   * TestCase.
   *
   * Subclasses should override this method to perform any costly
   * per-test setup before DoRun is invoked.
   */
  virtual void DoSetup (void);

  /**
   * \brief Implementation to actually run this TestCase.
   *
   * Subclasses should override this method to conduct their tests.
   */
  virtual void DoRun (void) = 0;

  /**
   * \brief Implementation to do any local setup required for this
   * TestCase.
   *
   * Subclasses should override this method to perform any costly
   * per-test teardown
   */
  virtual void DoTeardown (void);

  // methods called by TestRunnerImpl
  /**
   * \brief Actually run this TestCase
   *
   * \param [in] runner The test runner implementation.
   */
  void Run (TestRunnerImpl *runner);
  /** \copydoc IsStatusFailure() */
  bool IsFailed (void) const;

  /**
   * \ingroup testingimpl
   * \brief Container for results from a TestCase.
   */
  struct Result;

  TestCase *m_parent;                   //!< Pointer to my parent TestCase
  std::vector<TestCase *> m_children;   //!< Vector of my children
  std::string m_dataDir;                //!< My data directory
  TestRunnerImpl *m_runner;             //!< Pointer to the TestRunner
  struct Result *m_result;              //!< Results data
  std::string m_name;                   //!< TestCase name
  enum TestDuration m_duration;         //!< TestCase duration
};

/**
 * \ingroup testing
 *
 * \brief A suite of tests to run.
 *
 * \see sample-test-suite.cc
 */
class TestSuite : public TestCase
{
public:
  /**
   * \enum Type
   * \brief Type of test.
   */
  enum Type
  {
    ALL = 0,    //!<
    UNIT,       //!< This test suite implements a Unit Test
    SYSTEM,     //!< This test suite implements a System Test
    EXAMPLE,    //!< This test suite implements an Example Test
    PERFORMANCE //!< This test suite implements a Performance Test
  };

  /**
   * \brief Construct a new test suite.
   *
   * \param [in] name The name of the test suite.
   * \param [in] type The TestType of the test suite (defaults to UNIT test).
   */
  TestSuite (std::string name, Type type = UNIT);

  /**
   * \brief get the kind of test this test suite implements
   *
   * \returns The Type of the suite.
   */
  TestSuite::Type GetTestType (void);

private:
  // Inherited
  virtual void DoRun (void);

  TestSuite::Type m_type;               //!< Type of this TestSuite
};

/**
 * \ingroup testingimpl
 *
 * \brief A runner to execute tests.
 */
class TestRunner
{
public:
  /**
   * Run the requested suite of tests,
   * according to the given command line arguments.
   *
   * \param [in] argc The number of elements in \pname{argv}
   * \param [in] argv The vector of command line arguments
   * \returns Success status
   */
  static int Run (int argc, char *argv[]);
};

/**
 * \ingroup testing
 *
 * \brief A simple way to store test vectors (for stimulus or from responses)
 */
template <typename T>
class TestVectors
{
public:
  /**
   * Constructor
   */
  TestVectors ();
  /**
   * Virtual destructor
   */
  virtual ~TestVectors ();

  // Delete copy constructor and assignment operator to avoid misuse
  TestVectors (const TestVectors &) = delete;
  TestVectors & operator = (const TestVectors &) = delete;

  /**
   * \brief Set the expected length of this vector.
   *
   * \param [in] reserve The number of entries to reserve
   */
  void Reserve (uint32_t reserve);

  /**
   * \param [in] vector The test vector to add
   *
   * \returns The new test vector index
   */
  std::size_t Add (T vector);

  /**
   * \brief Get the total number of test vectors.
   * \return The number of test vectors
   */
  std::size_t GetN (void) const;
  /**
   * \brief Get the i'th test vector
   * \param [in] i The requested vector index
   * \return The requested vector
   */
  T Get (std::size_t i) const;

private:
  typedef std::vector<T> TestVector;    //!< Container type
  TestVector m_vectors;                 //!< The list of test vectors
};

template <typename T>
TestVectors<T>::TestVectors ()
  : m_vectors ()
{}

template <typename T>
void
TestVectors<T>::Reserve (uint32_t reserve)
{
  m_vectors.reserve (reserve);
}

template <typename T>
TestVectors<T>::~TestVectors ()
{}

template <typename T>
std::size_t
TestVectors<T>::Add (T vector)
{
  std::size_t index = m_vectors.size ();
  m_vectors.push_back (vector);
  return index;
}

template <typename T>
std::size_t
TestVectors<T>::GetN (void) const
{
  return m_vectors.size ();
}

template <typename T>
T
TestVectors<T>::Get (std::size_t i) const
{
  NS_ABORT_MSG_UNLESS (m_vectors.size () > i, "TestVectors::Get(): Bad index");
  return m_vectors[i];
}

} // namespace ns3

#endif /* NS3_TEST_H */
