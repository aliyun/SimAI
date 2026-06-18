/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 */

#include "ns3/test.h"
#include "pmp-regression.h"
#include "hwmp-reactive-regression.h"
#include "hwmp-proactive-regression.h"
#include "hwmp-simplest-regression.h"
#include "hwmp-target-flags-regression.h"

using namespace ns3;

/**
 * \ingroup dot11s-test
 * \ingroup tests
 *
 * \brief Dot11s Regression Suite
 */
class Dot11sRegressionSuite : public TestSuite
{
public:
  Dot11sRegressionSuite () : TestSuite ("devices-mesh-dot11s-regression", SYSTEM) 
  {
    // We do not use NS_TEST_SOURCEDIR variable here since mesh/test has 
    // subdirectories
    SetDataDir (std::string ("src/mesh/test/dot11s"));
    AddTestCase (new PeerManagementProtocolRegressionTest, TestCase::QUICK);
    AddTestCase (new HwmpSimplestRegressionTest, TestCase::QUICK);
    AddTestCase (new HwmpReactiveRegressionTest, TestCase::QUICK);
    AddTestCase (new HwmpProactiveRegressionTest, TestCase::QUICK);
    AddTestCase (new HwmpDoRfRegressionTest, TestCase::QUICK);
  }
} g_dot11sRegressionSuite; ///< the test suite
