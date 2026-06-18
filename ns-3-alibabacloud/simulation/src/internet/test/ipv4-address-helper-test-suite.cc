/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
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

#include "ns3/test.h"
#include "ns3/ipv4-address-generator.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/simulator.h"

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 network allocator helper Test
 */
class NetworkAllocatorHelperTestCase : public TestCase
{
public:
  NetworkAllocatorHelperTestCase ();
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

NetworkAllocatorHelperTestCase::NetworkAllocatorHelperTestCase ()
  : TestCase ("Make sure the network allocator part is working on some common network prefixes.")
{
}

void
NetworkAllocatorHelperTestCase::DoTeardown (void)
{
  Ipv4AddressGenerator::Reset ();
  Simulator::Destroy ();
}
void
NetworkAllocatorHelperTestCase::DoRun (void)
{
  Ipv4Address address;
  Ipv4Address network;
  Ipv4AddressHelper h;

  h.SetBase ("1.0.0.0", "255.0.0.0");
  network = h.NewNetwork ();
  NS_TEST_EXPECT_MSG_EQ (network, Ipv4Address ("2.0.0.0"), "100");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("2.0.0.1"), "101");

  h.SetBase ("0.1.0.0", "255.255.0.0");
  network = h.NewNetwork ();
  NS_TEST_EXPECT_MSG_EQ (network, Ipv4Address ("0.2.0.0"), "102");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.2.0.1"), "103");

  h.SetBase ("0.0.1.0", "255.255.255.0");
  network = h.NewNetwork ();
  NS_TEST_EXPECT_MSG_EQ (network, Ipv4Address ("0.0.2.0"), "104");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.0.2.1"), "105");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 address allocator helper Test
 */
class AddressAllocatorHelperTestCase : public TestCase
{
public:
  AddressAllocatorHelperTestCase ();
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

AddressAllocatorHelperTestCase::AddressAllocatorHelperTestCase ()
  : TestCase ("Make sure the address allocator part is working")
{
}

void
AddressAllocatorHelperTestCase::DoTeardown (void)
{
  Ipv4AddressGenerator::Reset ();
  Simulator::Destroy ();
}

void
AddressAllocatorHelperTestCase::DoRun (void)
{
  Ipv4Address network;
  Ipv4Address address;
  Ipv4AddressHelper h;

  h.SetBase ("1.0.0.0", "255.0.0.0", "0.0.0.3");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("1.0.0.3"), "200");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("1.0.0.4"), "201");

  h.SetBase ("0.1.0.0", "255.255.0.0", "0.0.0.3");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.1.0.3"), "202");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.1.0.4"), "203");

  h.SetBase ("0.0.1.0", "255.255.255.0", "0.0.0.3");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.0.1.3"), "204");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.0.1.4"), "205");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 reset allocator helper Test
 */
class ResetAllocatorHelperTestCase : public TestCase
{
public:
  ResetAllocatorHelperTestCase ();
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

ResetAllocatorHelperTestCase::ResetAllocatorHelperTestCase ()
  : TestCase ("Make sure the reset to base behavior is working")
{
}

void
ResetAllocatorHelperTestCase::DoRun (void)
{
  Ipv4Address network;
  Ipv4Address address;
  Ipv4AddressHelper h;

  //
  // We're going to use some of the same addresses allocated above, 
  // so reset the Ipv4AddressGenerator to make it forget we did.
  //

  h.SetBase ("1.0.0.0", "255.0.0.0", "0.0.0.3");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("1.0.0.3"), "301");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("1.0.0.4"), "302");
  network = h.NewNetwork ();
  NS_TEST_EXPECT_MSG_EQ (network, Ipv4Address ("2.0.0.0"), "303");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("2.0.0.3"), "304");

  h.SetBase ("0.1.0.0", "255.255.0.0", "0.0.0.3");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.1.0.3"), "305");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.1.0.4"), "306");
  network = h.NewNetwork ();
  NS_TEST_EXPECT_MSG_EQ (network, Ipv4Address ("0.2.0.0"), "307");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.2.0.3"), "308");

  h.SetBase ("0.0.1.0", "255.255.255.0", "0.0.0.3");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.0.1.3"), "309");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.0.1.4"), "310");
  network = h.NewNetwork ();
  NS_TEST_EXPECT_MSG_EQ (network, Ipv4Address ("0.0.2.0"), "311");
  address = h.NewAddress ();
  NS_TEST_EXPECT_MSG_EQ (address, Ipv4Address ("0.0.2.3"), "312");
}

void
ResetAllocatorHelperTestCase::DoTeardown (void)
{
  Ipv4AddressGenerator::Reset ();
  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 address helper Test
 */
class IpAddressHelperTestCasev4 : public TestCase
{
public:
  IpAddressHelperTestCasev4 ();
  virtual ~IpAddressHelperTestCasev4 ();

private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

IpAddressHelperTestCasev4::IpAddressHelperTestCasev4 ()
  : TestCase ("IpAddressHelper Ipv4 test case (similar to IPv6)")
{
}

IpAddressHelperTestCasev4::~IpAddressHelperTestCasev4 ()
{
}

void
IpAddressHelperTestCasev4::DoRun (void)
{
  Ipv4AddressHelper ip1;
  Ipv4Address ipAddr1;
  ipAddr1 = ip1.NewAddress ();
  // Ipv4AddressHelper that is unconfigured
  NS_TEST_ASSERT_MSG_EQ (ipAddr1, Ipv4Address ("255.255.255.255"), "Ipv4AddressHelper failure");

  ip1.SetBase ("192.168.0.0", "255.255.255.0");
  ipAddr1 = ip1.NewAddress ();
  NS_TEST_ASSERT_MSG_EQ (ipAddr1, Ipv4Address ("192.168.0.1"), "Ipv4AddressHelper failure");
  ipAddr1 = ip1.NewAddress ();
  NS_TEST_ASSERT_MSG_EQ (ipAddr1, Ipv4Address ("192.168.0.2"), "Ipv4AddressHelper failure");
  ip1.NewNetwork ();
  ipAddr1 = ip1.NewAddress ();
  NS_TEST_ASSERT_MSG_EQ (ipAddr1, Ipv4Address ("192.168.1.1"), "Ipv4AddressHelper failure");
  ip1.NewNetwork ();  // 192.168.2
  ip1.NewNetwork ();  // 192.168.3
  ip1.NewNetwork ();  // 192.168.4
  ipAddr1 = ip1.NewAddress (); // 4.1
  ipAddr1 = ip1.NewAddress (); // 4.2
  ipAddr1 = ip1.NewAddress (); // 4.3
  NS_TEST_ASSERT_MSG_EQ (ipAddr1, Ipv4Address ("192.168.4.3"), "Ipv4AddressHelper failure");

  // reset base to start at 192.168.0.100
  ip1.SetBase ("192.168.0.0", "255.255.255.0", "0.0.0.100");
  ipAddr1 = ip1.NewAddress ();
  NS_TEST_ASSERT_MSG_EQ (ipAddr1, Ipv4Address ("192.168.0.100"), "Ipv4AddressHelper failure");
  
  // rollover
  ip1.SetBase ("192.168.0.0", "255.255.255.0", "0.0.0.254");
  ipAddr1 = ip1.NewAddress (); // .254
  NS_TEST_ASSERT_MSG_EQ (ipAddr1, Ipv4Address ("192.168.0.254"), "Ipv4AddressHelper failure");
  // The below will overflow and assert, so it is commented out
  // ipAddr1 = ip1.NewAddress (); // .255

  // create with arguments
  Ipv4AddressHelper ip2 = Ipv4AddressHelper ("192.168.1.0", "255.255.255.0", "0.0.0.1");
  // duplicate assignment
  ip2.NewNetwork ();  // 192.168.2
  ip2.NewNetwork ();  // 192.168.3
  ip2.NewNetwork ();  // 192.168.4
  // Uncomment below, and 192.168.4.1 will crash since it was allocated above
  // ipAddr1 = ip2.NewAddress (); // 4.1

}

void
IpAddressHelperTestCasev4::DoTeardown (void)
{
  Ipv4AddressGenerator::Reset ();
  Simulator::Destroy ();
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Address Helper TestSuite
 */
class Ipv4AddressHelperTestSuite : public TestSuite
{
public:
  Ipv4AddressHelperTestSuite ();
private:
};

Ipv4AddressHelperTestSuite::Ipv4AddressHelperTestSuite ()
  : TestSuite ("ipv4-address-helper", UNIT)
{
  AddTestCase (new NetworkAllocatorHelperTestCase (), TestCase::QUICK);
  AddTestCase (new AddressAllocatorHelperTestCase (), TestCase::QUICK);
  AddTestCase (new ResetAllocatorHelperTestCase (), TestCase::QUICK);
  AddTestCase (new IpAddressHelperTestCasev4 (), TestCase::QUICK);
}

static Ipv4AddressHelperTestSuite g_ipv4AddressHelperTestSuite; //!< Static variable for test initialization


