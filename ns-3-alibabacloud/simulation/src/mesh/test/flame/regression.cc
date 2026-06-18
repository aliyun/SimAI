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
 * Authors: Pavel Boyko <boyko@iitp.ru>
 */

#include "ns3/test.h"
#include "flame-regression.h"

using namespace ns3;

/**
 * \ingroup flame-test
 * \ingroup tests
 *
 * \brief Flame Regression Suite
 */
class FlameRegressionSuite : public TestSuite
{
public:
  FlameRegressionSuite () : TestSuite ("devices-mesh-flame-regression", SYSTEM) 
  {
    // We do not use NS_TEST_SOURCEDIR variable here since mesh/test has 
    // subdirectories
    SetDataDir (std::string ("src/mesh/test/flame"));
    AddTestCase (new FlameRegressionTest, TestCase::QUICK);
  }
} g_flameRegressionSuite; ///< the test suite
