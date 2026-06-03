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

#include "hello-regression-test.h"
#include "tc-regression-test.h"
#include "bug780-test.h"

using namespace ns3;
using namespace olsr;

/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * Various olsr regression tests
 */
class RegressionTestSuite : public TestSuite
{
public:
  RegressionTestSuite () : TestSuite ("routing-olsr-regression", SYSTEM)
  {
    SetDataDir (NS_TEST_SOURCEDIR);
    AddTestCase (new HelloRegressionTest, TestCase::QUICK);
    AddTestCase (new TcRegressionTest, TestCase::QUICK);
    AddTestCase (new Bug780Test, TestCase::QUICK);
  }
};

static RegressionTestSuite g_olsrRegressionTestSuite; //!< Static variable for test initialization

