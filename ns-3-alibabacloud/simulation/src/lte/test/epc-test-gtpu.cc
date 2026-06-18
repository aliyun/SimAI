/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Jaume Nin <jaume.nin@cttc.cat>
 */


#include "ns3/log.h"
#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/epc-gtpu-header.h"

#include "epc-test-gtpu.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("EpcGtpuTest");

/**
 * TestSuite
 */

EpsGtpuTestSuite::EpsGtpuTestSuite ()
  : TestSuite ("epc-gtpu", SYSTEM)
{
  AddTestCase (new EpsGtpuHeaderTestCase (), TestCase::QUICK);
}

static EpsGtpuTestSuite epsGtpuTestSuite;


/**
 * TestCase
 */

EpsGtpuHeaderTestCase::EpsGtpuHeaderTestCase ()
  : TestCase ("Check header coding and decoding")
{
  NS_LOG_INFO ("Creating EpsGtpuHeaderTestCase");
}

EpsGtpuHeaderTestCase::~EpsGtpuHeaderTestCase ()
{
}

void
EpsGtpuHeaderTestCase::DoRun (void)
{
  LogLevel logLevel = (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_LEVEL_ALL);

  LogComponentEnable ("EpcGtpuTest", logLevel);
  GtpuHeader h1;
  h1.SetExtensionHeaderFlag (true);
  h1.SetLength (1234);
  h1.SetMessageType (123);
  h1.SetNPduNumber (123);
  h1.SetNPduNumberFlag (true);
  h1.SetNextExtensionType (123);
  h1.SetProtocolType (true);
  h1.SetSequenceNumber (1234);
  h1.SetSequenceNumberFlag (true);
  h1.SetTeid (1234567);
  h1.SetVersion (123);

  Packet p;
  GtpuHeader h2;
  p.AddHeader (h1);
  p.RemoveHeader (h2);

  NS_TEST_ASSERT_MSG_EQ (h1, h2, "Wrong value!");
}

