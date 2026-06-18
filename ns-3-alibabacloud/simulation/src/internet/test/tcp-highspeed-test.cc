/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello, <natale.patriciello@gmail.com>
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

#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-highspeed.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpHighSpeedTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the congestion avoidance increment on TcpHighSpeed
 */
class TcpHighSpeedIncrementTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param segmentSize Segment size.
   * \param name Test description.
   */
  TcpHighSpeedIncrementTest (uint32_t cWnd, uint32_t segmentSize,
                             const std::string &name);

private:
  virtual void DoRun (void);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_segmentSize; //!< Segment size.
  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

TcpHighSpeedIncrementTest::TcpHighSpeedIncrementTest (uint32_t cWnd,
                                                      uint32_t segmentSize,
                                                      const std::string &name)
  : TestCase (name),
  m_cWnd (cWnd),
  m_segmentSize (segmentSize)
{
}

void
TcpHighSpeedIncrementTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;

  Ptr<TcpHighSpeed> cong = CreateObject <TcpHighSpeed> ();

  uint32_t segCwnd = m_cWnd / m_segmentSize;
  uint32_t coeffA = TcpHighSpeed::TableLookupA (segCwnd);

  // Each received ACK weight is "coeffA". To see an increase of 1 MSS, we need
  // to ACK at least segCwnd/coeffA ACK.

  cong->IncreaseWindow (m_state, (segCwnd / coeffA) + 1);

  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), m_cWnd + m_segmentSize,
                         "CWnd has not increased");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the congestion avoidance decrement on TcpHighSpeed
 */
class TcpHighSpeedDecrementTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param segmentSize Segment size.
   * \param name Test description.
   */
  TcpHighSpeedDecrementTest (uint32_t cWnd, uint32_t segmentSize,
                             const std::string &name);

private:
  virtual void DoRun (void);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_segmentSize; //!< Segment size.
  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

TcpHighSpeedDecrementTest::TcpHighSpeedDecrementTest (uint32_t cWnd,
                                                      uint32_t segmentSize,
                                                      const std::string &name)
  : TestCase (name),
  m_cWnd (cWnd),
  m_segmentSize (segmentSize)
{
}

void
TcpHighSpeedDecrementTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;

  Ptr<TcpHighSpeed> cong = CreateObject <TcpHighSpeed> ();

  uint32_t segCwnd = m_cWnd / m_segmentSize;
  double coeffB = 1.0 - TcpHighSpeed::TableLookupB (segCwnd);

  uint32_t ret = cong->GetSsThresh (m_state, m_state->m_cWnd);

  uint32_t ssThHS = std::max (2.0, segCwnd * coeffB);

  NS_TEST_ASSERT_MSG_EQ (ret / m_segmentSize, ssThHS,
                         "HighSpeed decrement fn not used");
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TcpHighSpeed Congestion window values to test.
 */
static const struct HighSpeedImportantValues
{
  unsigned int cwnd; //!< Congestion window.
  unsigned int md;   //!< Currently unused.
} highSpeedImportantValues[] = {
  {     38,  128, /*  0.50 */ },
  {    118,  112, /*  0.44 */ },
  {    221,  104, /*  0.41 */ },
  {    347,   98, /*  0.38 */ },
  {    495,   93, /*  0.37 */ },
  {    663,   89, /*  0.35 */ },
  {    851,   86, /*  0.34 */ },
  {   1058,   83, /*  0.33 */ },
  {   1284,   81, /*  0.32 */ },
  {   1529,   78, /*  0.31 */ },
  {   1793,   76, /*  0.30 */ },
  {   2076,   74, /*  0.29 */ },
  {   2378,   72, /*  0.28 */ },
  {   2699,   71, /*  0.28 */ },
  {   3039,   69, /*  0.27 */ },
  {   3399,   68, /*  0.27 */ },
  {   3778,   66, /*  0.26 */ },
  {   4177,   65, /*  0.26 */ },
  {   4596,   64, /*  0.25 */ },
  {   5036,   62, /*  0.25 */ },
  {   5497,   61, /*  0.24 */ },
  {   5979,   60, /*  0.24 */ },
  {   6483,   59, /*  0.23 */ },
  {   7009,   58, /*  0.23 */ },
  {   7558,   57, /*  0.22 */ },
  {   8130,   56, /*  0.22 */ },
  {   8726,   55, /*  0.22 */ },
  {   9346,   54, /*  0.21 */ },
  {   9991,   53, /*  0.21 */ },
  {  10661,   52, /*  0.21 */ },
  {  11358,   52, /*  0.20 */ },
  {  12082,   51, /*  0.20 */ },
  {  12834,   50, /*  0.20 */ },
  {  13614,   49, /*  0.19 */ },
  {  14424,   48, /*  0.19 */ },
  {  15265,   48, /*  0.19 */ },
  {  16137,   47, /*  0.19 */ },
  {  17042,   46, /*  0.18 */ },
  {  17981,   45, /*  0.18 */ },
  {  18955,   45, /*  0.18 */ },
  {  19965,   44, /*  0.17 */ },
  {  21013,   43, /*  0.17 */ },
  {  22101,   43, /*  0.17 */ },
  {  23230,   42, /*  0.17 */ },
  {  24402,   41, /*  0.16 */ },
  {  25618,   41, /*  0.16 */ },
  {  26881,   40, /*  0.16 */ },
  {  28193,   39, /*  0.16 */ },
  {  29557,   39, /*  0.15 */ },
  {  30975,   38, /*  0.15 */ },
  {  32450,   38, /*  0.15 */ },
  {  33986,   37, /*  0.15 */ },
  {  35586,   36, /*  0.14 */ },
  {  37253,   36, /*  0.14 */ },
  {  38992,   35, /*  0.14 */ },
  {  40808,   35, /*  0.14 */ },
  {  42707,   34, /*  0.13 */ },
  {  44694,   33, /*  0.13 */ },
  {  46776,   33, /*  0.13 */ },
  {  48961,   32, /*  0.13 */ },
  {  51258,   32, /*  0.13 */ },
  {  53677,   31, /*  0.12 */ },
  {  56230,   30, /*  0.12 */ },
  {  58932,   30, /*  0.12 */ },
  {  61799,   29, /*  0.12 */ },
  {  64851,   28, /*  0.11 */ },
  {  68113,   28, /*  0.11 */ },
  {  71617,   27, /*  0.11 */ },
  {  75401,   26, /*  0.10 */ },
  {  79517,   26, /*  0.10 */ },
  {  84035,   25, /*  0.10 */ },
  {  89053,   24, /*  0.10 */ },
}; //!< List of data to be tested.

#define HIGHSPEED_VALUES_N 71

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP HighSpeed TestSuite
 */
class TcpHighSpeedTestSuite : public TestSuite
{
public:
  TcpHighSpeedTestSuite () : TestSuite ("tcp-highspeed-test", UNIT)
  {
    std::stringstream ss;

    for (uint32_t i = 0; i < HIGHSPEED_VALUES_N; ++i)
      {
        ss << highSpeedImportantValues[i].cwnd;
        AddTestCase (new TcpHighSpeedIncrementTest (highSpeedImportantValues[i].cwnd, 1,
                                                    "Highspeed increment test on cWnd " + ss.str ()),
                     TestCase::QUICK);
        AddTestCase (new TcpHighSpeedIncrementTest (highSpeedImportantValues[i].cwnd * 536, 536,
                                                    "Highspeed increment test on cWnd " + ss.str ()),
                     TestCase::QUICK);
        AddTestCase (new TcpHighSpeedIncrementTest (highSpeedImportantValues[i].cwnd * 1446, 1446,
                                                    "Highspeed increment test on cWnd " + ss.str ()),
                     TestCase::QUICK);
        AddTestCase (new TcpHighSpeedDecrementTest (highSpeedImportantValues[i].cwnd, 1,
                                                    "Highspeed Decrement test on cWnd " + ss.str ()),
                     TestCase::QUICK);
        AddTestCase (new TcpHighSpeedDecrementTest (highSpeedImportantValues[i].cwnd * 536, 536,
                                                    "Highspeed Decrement test on cWnd " + ss.str ()),
                     TestCase::QUICK);
        AddTestCase (new TcpHighSpeedDecrementTest (highSpeedImportantValues[i].cwnd * 1446, 1446,
                                                    "Highspeed Decrement test on cWnd " + ss.str ()),
                     TestCase::QUICK);
        ss.flush ();
      }
  }
};

static TcpHighSpeedTestSuite g_tcpHighSpeedTest; //!< Static variable for test initialization

