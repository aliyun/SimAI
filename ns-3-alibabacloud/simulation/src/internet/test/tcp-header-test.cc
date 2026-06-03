/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
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

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include "ns3/test.h"
#include "ns3/core-module.h"
#include "ns3/tcp-header.h"
#include "ns3/buffer.h"
#include "ns3/tcp-option-rfc793.h"

using namespace ns3;

#define GET_RANDOM_UINT32(RandomVariable) \
  static_cast<uint32_t> (RandomVariable->GetInteger (0, UINT32_MAX))

#define GET_RANDOM_UINT16(RandomVariable) \
  static_cast<uint16_t> (RandomVariable->GetInteger (0, UINT16_MAX))

#define GET_RANDOM_UINT8(RandomVariable) \
  static_cast<uint8_t> (RandomVariable->GetInteger (0, UINT8_MAX))

#define GET_RANDOM_UINT6(RandomVariable) \
  static_cast<uint8_t> (RandomVariable->GetInteger (0, UINT8_MAX >> 2))



/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP header Get/Set test.
 */
class TcpHeaderGetSetTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param name Test description.
   */
  TcpHeaderGetSetTestCase (std::string name);
protected:
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

};

TcpHeaderGetSetTestCase::TcpHeaderGetSetTestCase (std::string name) : TestCase (name)
{
}

void TcpHeaderGetSetTestCase::DoRun (void)
{
  uint16_t sourcePort;        // Source port
  uint16_t destinationPort;   // Destination port
  SequenceNumber32 sequenceNumber;  // Sequence number
  SequenceNumber32 ackNumber;       // ACK number
  uint8_t flags;              // Flags (really a uint6_t)
  uint16_t windowSize;        // Window size
  uint16_t urgentPointer;     // Urgent pointer
  TcpHeader header;
  Buffer buffer;

  Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
  for (uint32_t i = 0; i < 1000; ++i)
    {
      sourcePort = GET_RANDOM_UINT16 (x);
      destinationPort = GET_RANDOM_UINT16 (x);
      sequenceNumber = SequenceNumber32 (GET_RANDOM_UINT32 (x));
      ackNumber = SequenceNumber32 (GET_RANDOM_UINT32 (x));
      flags = GET_RANDOM_UINT6 (x);
      windowSize = GET_RANDOM_UINT16 (x);
      urgentPointer = GET_RANDOM_UINT16 (x);

      header.SetSourcePort (sourcePort);
      header.SetDestinationPort (destinationPort);
      header.SetSequenceNumber (sequenceNumber);
      header.SetAckNumber (ackNumber);
      header.SetFlags (flags);
      header.SetWindowSize (windowSize);
      header.SetUrgentPointer (urgentPointer);

      NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 5, "TcpHeader without option is"
                             " not 5 word");

      buffer.AddAtStart (header.GetSerializedSize ());
      header.Serialize (buffer.Begin ());

      NS_TEST_ASSERT_MSG_EQ (sourcePort, header.GetSourcePort (),
                             "Different source port found");
      NS_TEST_ASSERT_MSG_EQ (destinationPort, header.GetDestinationPort (),
                             "Different destination port found");
      NS_TEST_ASSERT_MSG_EQ (sequenceNumber, header.GetSequenceNumber (),
                             "Different sequence number found");
      NS_TEST_ASSERT_MSG_EQ (ackNumber, header.GetAckNumber (),
                             "Different ack number found");
      NS_TEST_ASSERT_MSG_EQ (flags, header.GetFlags (),
                             "Different flags found");
      NS_TEST_ASSERT_MSG_EQ (windowSize, header.GetWindowSize (),
                             "Different window size found");
      NS_TEST_ASSERT_MSG_EQ (urgentPointer, header.GetUrgentPointer (),
                             "Different urgent pointer found");

      NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 5, "TcpHeader without option is"
                             " not 5 word");

      TcpHeader copyHeader;

      copyHeader.Deserialize (buffer.Begin ());

      NS_TEST_ASSERT_MSG_EQ (sourcePort, copyHeader.GetSourcePort (),
                             "Different source port found in deserialized header");
      NS_TEST_ASSERT_MSG_EQ (destinationPort, copyHeader.GetDestinationPort (),
                             "Different destination port found in deserialized header");
      NS_TEST_ASSERT_MSG_EQ (sequenceNumber, copyHeader.GetSequenceNumber (),
                             "Different sequence number found in deserialized header");
      NS_TEST_ASSERT_MSG_EQ (ackNumber, copyHeader.GetAckNumber (),
                             "Different ack number found in deserialized header");
      NS_TEST_ASSERT_MSG_EQ (flags, copyHeader.GetFlags (),
                             "Different flags found in deserialized header");
      NS_TEST_ASSERT_MSG_EQ (windowSize, copyHeader.GetWindowSize (),
                             "Different window size found in deserialized header");
      NS_TEST_ASSERT_MSG_EQ (urgentPointer, copyHeader.GetUrgentPointer (),
                             "Different urgent pointer found in deserialized header");
    }
}

void TcpHeaderGetSetTestCase::DoTeardown (void)
{
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP header with RFC793 Options test.
 */
class TcpHeaderWithRFC793OptionTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param name Test description.
   */
  TcpHeaderWithRFC793OptionTestCase (std::string name);

private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  /**
   * \brief Check an header with only one kind of option.
   */
  void OneOptionAtTime ();
  /**
   * \brief Check an header for the correct padding.
   */
  void CheckNoPadding ();
  /**
   * \brief Check the correct header deserialization.
   */
  void CheckCorrectDeserialize ();
};


TcpHeaderWithRFC793OptionTestCase::TcpHeaderWithRFC793OptionTestCase (std::string name)
  : TestCase (name)
{

}

void
TcpHeaderWithRFC793OptionTestCase::DoRun ()
{
  OneOptionAtTime ();
  CheckNoPadding ();
  CheckCorrectDeserialize ();
}

void
TcpHeaderWithRFC793OptionTestCase::CheckCorrectDeserialize ()
{
  TcpHeader source, destination;
  TcpOptionNOP temp;
  Buffer buffer;
  buffer.AddAtStart (40);

  Buffer::Iterator i = buffer.Begin ();
  source.AppendOption (&temp);

  source.Serialize (i);

  i.ReadU8 ();
  i.WriteU8 (59);

  i = buffer.Begin ();
  destination.Deserialize (i);

  NS_TEST_ASSERT_MSG_EQ (destination.HasOption (59), false, "Kind 59 registered");
}

void
TcpHeaderWithRFC793OptionTestCase::CheckNoPadding ()
{
  {
    TcpOptionNOP oNop1, oNop2, oNop3, oNop4;
    TcpHeader header;
    Buffer buffer;

    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 5, "TcpHeader without option is"
                           " not 5 word");
    header.AppendOption (&oNop1);
    header.AppendOption (&oNop2);
    header.AppendOption (&oNop3);
    header.AppendOption (&oNop4);

    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 6, "Four byte added as option "
                           "are not a word");
    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (), 24, "Four byte added as option "
                           "are not a word");

    buffer.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buffer.Begin ());

    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (),
                           buffer.GetSize (), "Header not correctly serialized");

    // Inserted 4 byte NOP, no padding should be present
    Buffer::Iterator i = buffer.Begin ();
    i.Next (20);

    for (uint32_t j = 0; j < 4; ++j)
      {
        std::stringstream ss;
        ss << j;
        uint8_t value = i.ReadU8 ();
        NS_TEST_ASSERT_MSG_EQ (value, TcpOption::NOP,
                               "NOP not present at position " + ss.str ());
      }
  }
}

void
TcpHeaderWithRFC793OptionTestCase::OneOptionAtTime ()
{
  {
    TcpOptionEnd oEnd;
    TcpHeader header;
    Buffer buffer;

    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 5, "TcpHeader without option is"
                           " not 5 word");
    header.AppendOption (&oEnd);
    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 5, "Length has changed also for"
                           " END option");
    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (), 20, "Length has changed also for"
                           " END option");


    buffer.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buffer.Begin ());

    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (),
                           buffer.GetSize (), "Header not correctly serialized");
  }

  {
    TcpOptionNOP oNop;
    TcpHeader header;
    Buffer buffer;

    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 5, "TcpHeader without option is"
                           " not 5 word");
    header.AppendOption (&oNop);
    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 6, "NOP option not handled correctly");
    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (), 24, "Different length found for"
                           "NOP option");

    buffer.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buffer.Begin ());

    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (),
                           buffer.GetSize (), "Header not correctly serialized");

    // Inserted only 1 byte NOP, and so implementation should pad; so
    // the other 3 bytes should be END, PAD, PAD (n.b. PAD is same as END)
    Buffer::Iterator i = buffer.Begin ();
    i.Next (20);

    uint8_t value = i.ReadU8 ();
    NS_TEST_ASSERT_MSG_EQ (value, TcpOption::NOP, "NOP not present at byte 1");
    value = i.ReadU8 ();
    NS_TEST_ASSERT_MSG_EQ (value, TcpOption::END, "END not present at byte 2");
    value = i.ReadU8 ();
    NS_TEST_ASSERT_MSG_EQ (value, TcpOption::END, "pad not present at byte 3");
    value = i.ReadU8 ();
    NS_TEST_ASSERT_MSG_EQ (value, TcpOption::END, "pad not present at byte 4");
  }

  {
    TcpOptionMSS oMSS;
    oMSS.SetMSS (50);
    TcpHeader header, dest;
    Buffer buffer;

    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 5, "TcpHeader without option is"
                           " not 5 word");
    header.AppendOption (&oMSS);
    NS_TEST_ASSERT_MSG_EQ (header.GetLength (), 6, "MSS option not handled correctly");
    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (), 24, "Different length found for"
                           "MSS option");

    buffer.AddAtStart (header.GetSerializedSize ());
    header.Serialize (buffer.Begin ());

    NS_TEST_ASSERT_MSG_EQ (header.GetSerializedSize (),
                           buffer.GetSize (), "Header not correctly serialized");

    dest.Deserialize (buffer.Begin ());
    NS_TEST_ASSERT_MSG_EQ (header.HasOption (TcpOption::MSS),
                           true, "MSS option not correctly serialized");
    NS_TEST_ASSERT_MSG_EQ (header.GetOptionLength (), oMSS.GetSerializedSize (),
                           "MSS Option not counted in the total");
  }
}


void
TcpHeaderWithRFC793OptionTestCase::DoTeardown ()
{
  uint32_t foo;
  foo = UINT32_MAX;
  foo--;

}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP header Flags to Striing test.
 */
class TcpHeaderFlagsToString : public TestCase
{
public:
  /**
   * Constructor.
   * \param name Test description.
   */
  TcpHeaderFlagsToString (std::string name);

private:
  virtual void DoRun (void);
};

TcpHeaderFlagsToString::TcpHeaderFlagsToString (std::string name)
  : TestCase (name)
{
}

void
TcpHeaderFlagsToString::DoRun (void)
{
  std::string str, target;
  str = TcpHeader::FlagsToString (0x0);
  target = "";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x1);
  target = "FIN";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x2);
  target = "SYN";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x4);
  target = "RST";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x8);
  target = "PSH";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x10);
  target = "ACK";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x20);
  target = "URG";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x40);
  target = "ECE";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x80);
  target = "CWR";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x3);
  target = "FIN|SYN";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0x5);
  target = "FIN|RST";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0xff);
  target = "FIN|SYN|RST|PSH|ACK|URG|ECE|CWR";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
  str = TcpHeader::FlagsToString (0xff, ":");
  target = "FIN:SYN:RST:PSH:ACK:URG:ECE:CWR";
  NS_TEST_ASSERT_MSG_EQ (str, target, "str " << str <<  " does not equal target " << target);
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP header TestSuite
 */
class TcpHeaderTestSuite : public TestSuite
{
public:
  TcpHeaderTestSuite ()
    : TestSuite ("tcp-header", UNIT)
  {
    AddTestCase (new TcpHeaderGetSetTestCase ("GetSet test cases"), TestCase::QUICK);
    AddTestCase (new TcpHeaderWithRFC793OptionTestCase ("Test for options in RFC 793"), TestCase::QUICK);
    AddTestCase (new TcpHeaderFlagsToString ("Test flags to string function"), TestCase::QUICK);
  }

};

static TcpHeaderTestSuite g_TcpHeaderTestSuite; //!< Static variable for test initialization

