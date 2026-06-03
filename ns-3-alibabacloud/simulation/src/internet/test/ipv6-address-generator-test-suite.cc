/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
 * Copyright (c) 2011 Atishay Jain
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
#include "ns3/ipv6-address-generator.h"
#include "ns3/simulation-singleton.h"

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 network number allocator Test
 */
class NetworkNumber6AllocatorTestCase : public TestCase
{
public:
  NetworkNumber6AllocatorTestCase ();
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

NetworkNumber6AllocatorTestCase::NetworkNumber6AllocatorTestCase ()
  : TestCase ("Make sure the network number allocator is working on some of network prefixes.")
{
}
void
NetworkNumber6AllocatorTestCase::DoTeardown (void)
{
  Ipv6AddressGenerator::Reset ();
}
void
NetworkNumber6AllocatorTestCase::DoRun (void)
{
  Ipv6Address network;

  Ipv6AddressGenerator::Init (Ipv6Address ("1::0:0:0"), Ipv6Prefix ("FFFF::0"),
                              Ipv6Address ("::"));
  network = Ipv6AddressGenerator::GetNetwork (Ipv6Prefix ("FFFF::0"));
  NS_TEST_EXPECT_MSG_EQ (network, Ipv6Address ("1::0:0:0"), "network should equal the initialized network for given prefix");
  network = Ipv6AddressGenerator::NextNetwork (Ipv6Prefix ("FFFF::0"));
  NS_TEST_EXPECT_MSG_EQ (network, Ipv6Address ("2::0:0:0"), "network should equal next network");

  Ipv6AddressGenerator::Init (Ipv6Address ("0:1::0:0"),
                              Ipv6Prefix ("FFFF:FFFF::0"), Ipv6Address ("::"));
  network = Ipv6AddressGenerator::GetNetwork (Ipv6Prefix ("FFFF:FFFF::0"));
  NS_TEST_EXPECT_MSG_EQ (network, Ipv6Address ("0:1::0"), "network should equal the initialized network for given prefix");
  network = Ipv6AddressGenerator::NextNetwork (Ipv6Prefix ("FFFF:FFFF::0"));
  NS_TEST_EXPECT_MSG_EQ (network, Ipv6Address ("0:2::0"), "network should equal next network");

  Ipv6AddressGenerator::Init (Ipv6Address ("0:0:1::0"),
                              Ipv6Prefix ("FFFF:FFFF:FFFF::0"), Ipv6Address ("::0"));
  network = Ipv6AddressGenerator::GetNetwork (Ipv6Prefix ("FFFF:FFFF:FFFF::0"));
  NS_TEST_EXPECT_MSG_EQ (network, Ipv6Address ("0:0:1::0"), "network should equal the initialized network for given prefix");
  network = Ipv6AddressGenerator::NextNetwork (Ipv6Prefix ("FFFF:FFFF:FFFF::0"));
  NS_TEST_EXPECT_MSG_EQ (network, Ipv6Address ("0:0:2::0"), "network should equal next network");

}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 address allocator Test
 */
class AddressAllocator6TestCase : public TestCase
{
public:
  AddressAllocator6TestCase ();
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

AddressAllocator6TestCase::AddressAllocator6TestCase ()
  : TestCase ("Sanity check on allocation of addresses")
{
}

void
AddressAllocator6TestCase::DoRun (void)
{
  Ipv6Address address;

  Ipv6AddressGenerator::Init (Ipv6Address ("2001::0"), Ipv6Prefix (64));
  address = Ipv6AddressGenerator::GetNetwork (Ipv6Prefix (64));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001::0"), "address should equal the initialized address for given prefix");
  Ipv6AddressGenerator::NextNetwork (Ipv6Prefix (64));
  address = Ipv6AddressGenerator::GetNetwork (Ipv6Prefix (64));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0:0:1::0"), "address should equal the initialized address for given prefix");
  address = Ipv6AddressGenerator::GetAddress (Ipv6Prefix (64));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0:0:1::1"), "address should equal the initialized address for given prefix");
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (64));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0:0:1::1"), "address should equal the initialized address for given prefix");
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (64));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0:0:1::2"), "address should equal the initialized address for given prefix");

  Ipv6AddressGenerator::Init (Ipv6Address ("1::"), Ipv6Prefix ("FFFF::"),
                              Ipv6Address ("::3"));
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (16));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("1::3"), "address should equal initialized address for given prefix");
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (16));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("1::4"), "address should equal next address");

}

void
AddressAllocator6TestCase::DoTeardown (void)
{
  Ipv6AddressGenerator::Reset ();
  Simulator::Destroy ();
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 network number and address allocator Test
 */
class NetworkAndAddress6TestCase : public TestCase
{
public:
  NetworkAndAddress6TestCase ();
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

NetworkAndAddress6TestCase::NetworkAndAddress6TestCase ()
  : TestCase ("Make sure Network and address allocation play together.")
{
}

void
NetworkAndAddress6TestCase::DoTeardown (void)
{
  Ipv6AddressGenerator::Reset ();
  Simulator::Destroy ();
}

void
NetworkAndAddress6TestCase::DoRun (void)
{
  Ipv6Address address;
  Ipv6Address network;

  Ipv6AddressGenerator::Init (Ipv6Address ("3::"), Ipv6Prefix ("FFFF::"),
                              Ipv6Address ("::3"));
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (16));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("3::3"), "address should equal initialized address for given prefix");
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (16));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("3::4"), "address should equal next address for given prefix");

  network = Ipv6AddressGenerator::NextNetwork (Ipv6Prefix ("FFFF::"));
  NS_TEST_EXPECT_MSG_EQ (network, Ipv6Address ("4::0"), "address should equal next address for given prefix");
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (16));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("4::3"), "address should equal next address for given prefix");

}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 example of an address generator Test
 */
class ExampleAddress6GeneratorTestCase : public TestCase
{
public:
  ExampleAddress6GeneratorTestCase ();
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

ExampleAddress6GeneratorTestCase::ExampleAddress6GeneratorTestCase ()
  : TestCase ("A typical real-world example")
{
}

void
ExampleAddress6GeneratorTestCase::DoTeardown (void)
{
  Ipv6AddressGenerator::Reset ();
}

void
ExampleAddress6GeneratorTestCase::DoRun (void)
{
  Ipv6Address address;

  Ipv6AddressGenerator::Init (Ipv6Address ("2001:0AB8::"),
                              Ipv6Prefix ("FFFF:FFFF:FFFF::0"), Ipv6Address ("::3"));
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (48));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0AB8::0:3"), "address should equal initialized address for given prefix");
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (48));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0AB8::0:4"), "address should equal next address for given prefix");
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (48));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0AB8::0:5"), "address should equal next address for given prefix");
  //
  // Allocate the next network based on the prefix passed in, which should
  // be 2001:0AB0:0001
  //
  Ipv6AddressGenerator::NextNetwork (Ipv6Prefix ("FFFF:FFFF:FFFF::0"));
  //
  // reset first address to be allocated back to ::0:3
  //
  Ipv6AddressGenerator::InitAddress (Ipv6Address ("::3"),
                                     Ipv6Prefix (48));
  //
  // The first address we should get is the network and address ORed
  //
  address = Ipv6AddressGenerator::NextAddress (Ipv6Prefix (48));
  NS_TEST_EXPECT_MSG_EQ (address, Ipv6Address ("2001:0AB8:1::3"), "address should equal initialized address for given prefix");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 address collision Test
 */
class AddressCollision6TestCase : public TestCase
{
public:
  AddressCollision6TestCase ();
private:
  void DoRun (void);
  void DoTeardown (void);
};

AddressCollision6TestCase::AddressCollision6TestCase ()
  : TestCase ("Make sure that the address collision logic works.")
{
}

void
AddressCollision6TestCase::DoTeardown (void)
{
  Ipv6AddressGenerator::Reset ();
  Simulator::Destroy ();
}
void
AddressCollision6TestCase::DoRun (void)
{
  Ipv6AddressGenerator::AddAllocated ("0::0:5");
  Ipv6AddressGenerator::AddAllocated ("0::0:10");
  Ipv6AddressGenerator::AddAllocated ("0::0:15");
  Ipv6AddressGenerator::AddAllocated ("0::0:20");

  Ipv6AddressGenerator::AddAllocated ("0::0:4");
  Ipv6AddressGenerator::AddAllocated ("0::0:3");
  Ipv6AddressGenerator::AddAllocated ("0::0:2");
  Ipv6AddressGenerator::AddAllocated ("0::0:1");

  Ipv6AddressGenerator::AddAllocated ("0::0:6");
  Ipv6AddressGenerator::AddAllocated ("0::0:7");
  Ipv6AddressGenerator::AddAllocated ("0::0:8");
  Ipv6AddressGenerator::AddAllocated ("0::0:9");

  Ipv6AddressGenerator::AddAllocated ("0::0:11");
  Ipv6AddressGenerator::AddAllocated ("0::0:12");
  Ipv6AddressGenerator::AddAllocated ("0::0:13");
  Ipv6AddressGenerator::AddAllocated ("0::0:14");

  Ipv6AddressGenerator::AddAllocated ("0::0:19");
  Ipv6AddressGenerator::AddAllocated ("0::0:18");
  Ipv6AddressGenerator::AddAllocated ("0::0:17");
  Ipv6AddressGenerator::AddAllocated ("0::0:16");

  Ipv6AddressGenerator::TestMode ();
  bool added = Ipv6AddressGenerator::AddAllocated ("0::0:21");
  NS_TEST_EXPECT_MSG_EQ (added, true, "address should get allocated");

  added = Ipv6AddressGenerator::AddAllocated ("0::0:4");
  NS_TEST_EXPECT_MSG_EQ (added, false, "address should not get allocated");

  added = Ipv6AddressGenerator::AddAllocated ("0::0:9");
  NS_TEST_EXPECT_MSG_EQ (added, false, "address should not get allocated");

  added = Ipv6AddressGenerator::AddAllocated ("0::0:16");
  NS_TEST_EXPECT_MSG_EQ (added, false, "address should not get allocated");

  added = Ipv6AddressGenerator::AddAllocated ("0::0:21");
  NS_TEST_EXPECT_MSG_EQ (added, false, "address should not get allocated");
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 address generator TestSuite
 */
class Ipv6AddressGeneratorTestSuite : public TestSuite
{
public:
  Ipv6AddressGeneratorTestSuite ()
    : TestSuite ("ipv6-address-generator")
  {
    AddTestCase (new NetworkNumber6AllocatorTestCase (), TestCase::QUICK);
    AddTestCase (new AddressAllocator6TestCase (), TestCase::QUICK);
    AddTestCase (new NetworkAndAddress6TestCase (), TestCase::QUICK);
    AddTestCase (new ExampleAddress6GeneratorTestCase (), TestCase::QUICK);
    AddTestCase (new AddressCollision6TestCase (), TestCase::QUICK);
  }
};

static Ipv6AddressGeneratorTestSuite g_ipv6AddressGeneratorTestSuite; //!< Static variable for test initialization
