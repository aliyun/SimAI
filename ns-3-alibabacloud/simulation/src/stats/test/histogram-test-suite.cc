/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2009 INESC Porto
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
// Author: Pedro Fortuna  <pedro.fortuna@inescporto.pt> <pedro.fortuna@gmail.com>
//

#include "ns3/histogram.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup stats-tests
 *
 * \brief Histogram Test
 */
class HistogramTestCase : public ns3::TestCase {
private:
public:
  HistogramTestCase ();
  virtual void DoRun (void);


};

HistogramTestCase::HistogramTestCase ()
  : ns3::TestCase ("Histogram")
{
}


void
HistogramTestCase::DoRun (void)
{
  Histogram h0 (3.5);
  // Testing floating-point bin widths
  {
    for (int i=1; i <= 10; i++)
      { 
        h0.AddValue (3.4);
      }

    for (int i=1; i <= 5; i++)
      {
        h0.AddValue (3.6);
      }

    NS_TEST_EXPECT_MSG_EQ_TOL (h0.GetBinWidth (0),  3.5, 1e-6, "");
    NS_TEST_EXPECT_MSG_EQ (h0.GetNBins (),  2, "");
    NS_TEST_EXPECT_MSG_EQ_TOL (h0.GetBinStart (1),  3.5, 1e-6, "");
    NS_TEST_EXPECT_MSG_EQ (h0.GetBinCount (0),  10, "");
    NS_TEST_EXPECT_MSG_EQ (h0.GetBinCount (1),  5, "");
  }

  {
    // Testing bin expansion
    h0.AddValue (74.3);
    NS_TEST_EXPECT_MSG_EQ (h0.GetNBins (), 22, "");
    NS_TEST_EXPECT_MSG_EQ (h0.GetBinCount (21), 1, "");
  }
}

/**
 * \ingroup stats-tests
 *
 * \brief Histogram TestSuite
 */
class HistogramTestSuite : public TestSuite
{
public:
  HistogramTestSuite ();
};

HistogramTestSuite::HistogramTestSuite ()
  : TestSuite ("histogram", UNIT)
{
  AddTestCase (new HistogramTestCase, TestCase::QUICK);
}

static HistogramTestSuite g_HistogramTestSuite; //!< Static variable for test initialization
