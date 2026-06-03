/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
#include "ns3/ipv6-address.h"

using namespace ns3;

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * Ipv6Address unit tests.
 */
class Ipv6AddressTestCase : public TestCase
{
public:
  Ipv6AddressTestCase ();
  virtual ~Ipv6AddressTestCase ();

private:
  virtual void DoRun (void);
};

Ipv6AddressTestCase::Ipv6AddressTestCase ()
  : TestCase ("serialization code")
{
}

Ipv6AddressTestCase::~Ipv6AddressTestCase ()
{
}

void
Ipv6AddressTestCase::DoRun (void)
{
  Ipv6Address ip = Ipv6Address ("2001:db8::1");
  uint8_t ipBytes[16];
  ip.Serialize (ipBytes);
  NS_TEST_ASSERT_MSG_EQ (ipBytes[0], 0x20, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[1], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[2], 0x0d, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[3], 0xb8, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[15], 1, "Failed string conversion");

  ip = Ipv6Address ("2001:db8:1::1");
  ip.Serialize (ipBytes);
  NS_TEST_ASSERT_MSG_EQ (ipBytes[0], 0x20, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[1], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[2], 0x0d, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[3], 0xb8, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[5], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[15], 1, "Failed string conversion");

  // Zero padding
  ip = Ipv6Address ("2001:0db8:0001::1");
  ip.Serialize (ipBytes);
  NS_TEST_ASSERT_MSG_EQ (ipBytes[0], 0x20, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[1], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[2], 0x0d, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[3], 0xb8, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[5], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[15], 1, "Failed string conversion");

  ip = Ipv6Address ("2001:db8:0:1::1");
  ip.Serialize (ipBytes);
  NS_TEST_ASSERT_MSG_EQ (ipBytes[0], 0x20, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[1], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[2], 0x0d, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[3], 0xb8, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[7], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[15], 1, "Failed string conversion");

  ip = Ipv6Address ("2001:db8:0:1:0:0:0:1");
  ip.Serialize (ipBytes);
  NS_TEST_ASSERT_MSG_EQ (ipBytes[0], 0x20, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[1], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[2], 0x0d, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[3], 0xb8, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[7], 0x01, "Failed string conversion");
  NS_TEST_ASSERT_MSG_EQ (ipBytes[15], 1, "Failed string conversion");

  // Please add more tests below

}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Ipv6Address TestSuite
 *
 */
class Ipv6AddressTestSuite : public TestSuite
{
public:
  Ipv6AddressTestSuite ();
};

Ipv6AddressTestSuite::Ipv6AddressTestSuite ()
  : TestSuite ("ipv6-address", UNIT)
{
  AddTestCase (new Ipv6AddressTestCase, TestCase::QUICK);
}

static Ipv6AddressTestSuite ipv6AddressTestSuite; //!< Static variable for test initialization

