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
 *
 * Author: Fabian Mauchle <fabian.mauchle@hsr.ch>
 */

#include "ns3/test.h"
#include "ns3/ipv6-extension-header.h"
#include "ns3/ipv6-option-header.h"

using namespace ns3;

// ===========================================================================
// An empty option field must be filled with pad1 or padN header so theshape
// extension header's size is a multiple of 8.
//
// 0                                                              31
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |  Extension Destination Header |                               |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+          PadN Header          +
// |                                                               |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// ===========================================================================

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 extensions Test: Empty option field.
 */
class TestEmptyOptionField : public TestCase
{
public:
  TestEmptyOptionField () : TestCase ("TestEmptyOptionField") {}

  virtual void DoRun ()
  {
    Ipv6ExtensionDestinationHeader header;
    NS_TEST_EXPECT_MSG_EQ (header.GetSerializedSize () % 8, 0, "length of extension header is not a multiple of 8");

    Buffer buf;
    buf.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buf.Begin ());

    const uint8_t* data = buf.PeekData ();
    NS_TEST_EXPECT_MSG_EQ (*(data+2), 1, "padding is missing"); //expecting a padN header
  }
};

// ===========================================================================
// An option without alignment requirement must not be padded
//
// 0                                                              31
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |  Extension Destination Header | OptionWithoutAlignmentHeader..|
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |..OptionWithoutAlignmentHeader |          PadN Header          |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// ===========================================================================

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 extensions Test: Option without alignment.
 */
class OptionWithoutAlignmentHeader : public Ipv6OptionHeader
{
public:
  static const uint8_t TYPE = 42; //!< Option type.

  virtual uint32_t GetSerializedSize () const
  {
    return 4;
  }

  virtual void Serialize (Buffer::Iterator start) const
  {
    start.WriteU8 (TYPE);
    start.WriteU8 (GetSerializedSize ()-2);
    start.WriteU16 (0);
  }
};


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 extensions Test: Test the option without alignment.
 */
class TestOptionWithoutAlignment : public TestCase
{
public:
  TestOptionWithoutAlignment () : TestCase ("TestOptionWithoutAlignment") {}

  virtual void DoRun ()
  {
    Ipv6ExtensionDestinationHeader header;
    OptionWithoutAlignmentHeader optionHeader;
    header.AddOption (optionHeader);


    NS_TEST_EXPECT_MSG_EQ (header.GetSerializedSize () % 8, 0, "length of extension header is not a multiple of 8");

    Buffer buf;
    buf.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buf.Begin ());

    const uint8_t* data = buf.PeekData ();
    NS_TEST_EXPECT_MSG_EQ (*(data+2), OptionWithoutAlignmentHeader::TYPE, "option without alignment is not first in header field");
  }
};

// ===========================================================================
// An option with alignment requirement must be padded accordingly (padding to
// a total size multiple of 8 is allowed)
//
// 0                                                              31
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |  Extension Destination Header |          PadN Header          |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |                   OptionWithAlignmentHeader                   |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |          PadN Header          |                               |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +
// |                   Ipv6OptionJumbogramHeader                   |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// ===========================================================================

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 extensions Test: Option with alignment.
 */
class OptionWithAlignmentHeader : public Ipv6OptionHeader
{
public:
  static const uint8_t TYPE = 73; //!< Option Type.

  virtual uint32_t GetSerializedSize () const
  {
    return 4;
  }

  virtual void Serialize (Buffer::Iterator start) const
  {
    start.WriteU8 (TYPE);
    start.WriteU8 (GetSerializedSize ()-2);
    start.WriteU16 (0);
  }

  virtual Alignment GetAlignment () const
  {
    return (Alignment){ 4,0};
  }
};


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 extensions Test: Test the option with alignment.
 */
class TestOptionWithAlignment : public TestCase
{
public:
  TestOptionWithAlignment () : TestCase ("TestOptionWithAlignment") {}

  virtual void DoRun ()
  {
    Ipv6ExtensionDestinationHeader header;
    OptionWithAlignmentHeader optionHeader;
    header.AddOption (optionHeader);
    Ipv6OptionJumbogramHeader jumboHeader; //has an alignment of 4n+2
    header.AddOption (jumboHeader);

    NS_TEST_EXPECT_MSG_EQ (header.GetSerializedSize () % 8, 0, "length of extension header is not a multiple of 8");

    Buffer buf;
    buf.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buf.Begin ());

    const uint8_t* data = buf.PeekData ();
    NS_TEST_EXPECT_MSG_EQ (*(data+2), 1, "padding is missing"); //expecting a padN header
    NS_TEST_EXPECT_MSG_EQ (*(data+4), OptionWithAlignmentHeader::TYPE, "option with alignment is not padded correctly");
    NS_TEST_EXPECT_MSG_EQ (*(data+8), 1, "padding is missing"); //expecting a padN header
    NS_TEST_EXPECT_MSG_EQ (*(data+10), jumboHeader.GetType (), "option with alignment is not padded correctly");
  }
};

// ===========================================================================
// An option with an alignment that exactly matches the gap must not be padded
// (padding to a total size multiple of 8 is allowed)
//
// 0                                                              31
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |  Extension Destination Header |                               |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +
// |                   Ipv6OptionJumbogramHeader                   |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |                   OptionWithAlignmentHeader                   |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |                           PadN Header                         |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// ===========================================================================

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 extensions Test: Test an option already aligned.
 */
class TestFulfilledAlignment : public TestCase
{
public:
  TestFulfilledAlignment () : TestCase ("TestCorrectAlignment") {}

  virtual void DoRun ()
  {
    Ipv6ExtensionDestinationHeader header;
    Ipv6OptionJumbogramHeader jumboHeader; //has an alignment of 4n+2
    header.AddOption (jumboHeader);
    OptionWithAlignmentHeader optionHeader;
    header.AddOption (optionHeader);

    NS_TEST_EXPECT_MSG_EQ (header.GetSerializedSize () % 8, 0, "length of extension header is not a multiple of 8");

    Buffer buf;
    buf.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buf.Begin ());

    const uint8_t* data = buf.PeekData ();
    NS_TEST_EXPECT_MSG_EQ (*(data+2), jumboHeader.GetType (), "option with fulfilled alignment is padded anyway");
    NS_TEST_EXPECT_MSG_EQ (*(data+8), OptionWithAlignmentHeader::TYPE, "option with fulfilled alignment is padded anyway");
  }
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 extensions TestSuite.
 */
class Ipv6ExtensionHeaderTestSuite : public TestSuite
{
public:
  Ipv6ExtensionHeaderTestSuite ()
    : TestSuite ("ipv6-extension-header", UNIT)
  {
    AddTestCase (new TestEmptyOptionField, TestCase::QUICK);
    AddTestCase (new TestOptionWithoutAlignment, TestCase::QUICK);
    AddTestCase (new TestOptionWithAlignment, TestCase::QUICK);
    AddTestCase (new TestFulfilledAlignment, TestCase::QUICK);

  }
};

static Ipv6ExtensionHeaderTestSuite ipv6ExtensionHeaderTestSuite; //!< Static variable for test initialization
