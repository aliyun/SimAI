/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INESC Porto
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
 * Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt>
 */

#include "ns3/test.h"
#include "ns3/olsr-header.h"
#include "ns3/packet.h"


using namespace ns3;

/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * Check Emf olsr time conversion
 */
class OlsrEmfTestCase : public TestCase
{
public:
  OlsrEmfTestCase ();
  virtual void DoRun (void);
};

OlsrEmfTestCase::OlsrEmfTestCase ()
  : TestCase ("Check Emf olsr time conversion")
{
}
void
OlsrEmfTestCase::DoRun (void)
{
  for (int time = 1; time <= 30; time++)
    {
      uint8_t emf = olsr::SecondsToEmf (time);
      double seconds = olsr::EmfToSeconds (emf);
      NS_TEST_ASSERT_MSG_EQ ((seconds < 0 || std::fabs (seconds - time) > 0.1), false,
                             "100");
    }
}


/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * Check Mid olsr messages
 */
class OlsrMidTestCase : public TestCase
{
public:
  OlsrMidTestCase ();
  virtual void DoRun (void);
};

OlsrMidTestCase::OlsrMidTestCase ()
  : TestCase ("Check Mid olsr messages")
{
}
void
OlsrMidTestCase::DoRun (void)
{
  Packet packet;

  {
    olsr::PacketHeader hdr;
    olsr::MessageHeader msg1;
    olsr::MessageHeader::Mid &mid1 = msg1.GetMid ();
    olsr::MessageHeader msg2;
    olsr::MessageHeader::Mid &mid2 = msg2.GetMid ();

    // MID message #1
    {
      std::vector<Ipv4Address> &addresses = mid1.interfaceAddresses;
      addresses.clear ();
      addresses.push_back (Ipv4Address ("1.2.3.4"));
      addresses.push_back (Ipv4Address ("1.2.3.5"));
    }

    msg1.SetTimeToLive (255);
    msg1.SetOriginatorAddress (Ipv4Address ("11.22.33.44"));
    msg1.SetVTime (Seconds (9));
    msg1.SetMessageSequenceNumber (7);

    // MID message #2
    {
      std::vector<Ipv4Address> &addresses = mid2.interfaceAddresses;
      addresses.clear ();
      addresses.push_back (Ipv4Address ("2.2.3.4"));
      addresses.push_back (Ipv4Address ("2.2.3.5"));
    }

    msg2.SetTimeToLive (254);
    msg2.SetOriginatorAddress (Ipv4Address ("12.22.33.44"));
    msg2.SetVTime (Seconds (10));
    msg2.SetMessageType (olsr::MessageHeader::MID_MESSAGE);
    msg2.SetMessageSequenceNumber (7);

    // Build an OLSR packet header
    hdr.SetPacketLength (hdr.GetSerializedSize () + msg1.GetSerializedSize () + msg2.GetSerializedSize ());
    hdr.SetPacketSequenceNumber (123);


    // Now add all the headers in the correct order
    packet.AddHeader (msg2);
    packet.AddHeader (msg1);
    packet.AddHeader (hdr);
  }

  {
    olsr::PacketHeader hdr;
    packet.RemoveHeader (hdr);
    NS_TEST_ASSERT_MSG_EQ (hdr.GetPacketSequenceNumber (), 123, "200");
    uint32_t sizeLeft = hdr.GetPacketLength () - hdr.GetSerializedSize ();
    {
      olsr::MessageHeader msg1;

      packet.RemoveHeader (msg1);

      NS_TEST_ASSERT_MSG_EQ (msg1.GetTimeToLive (),  255, "201");
      NS_TEST_ASSERT_MSG_EQ (msg1.GetOriginatorAddress (), Ipv4Address ("11.22.33.44"), "202");
      NS_TEST_ASSERT_MSG_EQ (msg1.GetVTime (), Seconds (9), "203");
      NS_TEST_ASSERT_MSG_EQ (msg1.GetMessageType (), olsr::MessageHeader::MID_MESSAGE, "204");
      NS_TEST_ASSERT_MSG_EQ (msg1.GetMessageSequenceNumber (), 7, "205");

      olsr::MessageHeader::Mid &mid1 = msg1.GetMid ();
      NS_TEST_ASSERT_MSG_EQ (mid1.interfaceAddresses.size (), 2, "206");
      NS_TEST_ASSERT_MSG_EQ (*mid1.interfaceAddresses.begin (), Ipv4Address ("1.2.3.4"), "207");

      sizeLeft -= msg1.GetSerializedSize ();
      NS_TEST_ASSERT_MSG_EQ ((sizeLeft > 0), true, "208");
    }
    {
      // now read the second message
      olsr::MessageHeader msg2;

      packet.RemoveHeader (msg2);

      NS_TEST_ASSERT_MSG_EQ (msg2.GetTimeToLive (),  254, "209");
      NS_TEST_ASSERT_MSG_EQ (msg2.GetOriginatorAddress (), Ipv4Address ("12.22.33.44"), "210");
      NS_TEST_ASSERT_MSG_EQ (msg2.GetVTime (), Seconds (10), "211");
      NS_TEST_ASSERT_MSG_EQ (msg2.GetMessageType (), olsr::MessageHeader::MID_MESSAGE, "212");
      NS_TEST_ASSERT_MSG_EQ (msg2.GetMessageSequenceNumber (), 7, "213");

      olsr::MessageHeader::Mid mid2 = msg2.GetMid ();
      NS_TEST_ASSERT_MSG_EQ (mid2.interfaceAddresses.size (), 2, "214");
      NS_TEST_ASSERT_MSG_EQ (*mid2.interfaceAddresses.begin (), Ipv4Address ("2.2.3.4"), "215");

      sizeLeft -= msg2.GetSerializedSize ();
      NS_TEST_ASSERT_MSG_EQ (sizeLeft, 0, "216");
    }
  }
}


/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * Check Hello olsr messages
 */
class OlsrHelloTestCase : public TestCase
{
public:
  OlsrHelloTestCase ();
  virtual void DoRun (void);
};

OlsrHelloTestCase::OlsrHelloTestCase ()
  : TestCase ("Check Hello olsr messages")
{
}
void
OlsrHelloTestCase::DoRun (void)
{
  Packet packet;
  olsr::MessageHeader msgIn;
  olsr::MessageHeader::Hello &helloIn = msgIn.GetHello ();

  helloIn.SetHTime (Seconds (7));
  helloIn.willingness = 66;

  {
    olsr::MessageHeader::Hello::LinkMessage lm1;
    lm1.linkCode = 2;
    lm1.neighborInterfaceAddresses.push_back (Ipv4Address ("1.2.3.4"));
    lm1.neighborInterfaceAddresses.push_back (Ipv4Address ("1.2.3.5"));
    helloIn.linkMessages.push_back (lm1);

    olsr::MessageHeader::Hello::LinkMessage lm2;
    lm2.linkCode = 3;
    lm2.neighborInterfaceAddresses.push_back (Ipv4Address ("2.2.3.4"));
    lm2.neighborInterfaceAddresses.push_back (Ipv4Address ("2.2.3.5"));
    helloIn.linkMessages.push_back (lm2);
  }

  packet.AddHeader (msgIn);

  olsr::MessageHeader msgOut;
  packet.RemoveHeader (msgOut);
  olsr::MessageHeader::Hello &helloOut = msgOut.GetHello ();

  NS_TEST_ASSERT_MSG_EQ (helloOut.GetHTime (), Seconds (7), "300");
  NS_TEST_ASSERT_MSG_EQ (helloOut.willingness, 66, "301");
  NS_TEST_ASSERT_MSG_EQ (helloOut.linkMessages.size (), 2, "302");

  NS_TEST_ASSERT_MSG_EQ (helloOut.linkMessages[0].linkCode, 2, "303");
  NS_TEST_ASSERT_MSG_EQ (helloOut.linkMessages[0].neighborInterfaceAddresses[0],
                         Ipv4Address ("1.2.3.4"), "304");
  NS_TEST_ASSERT_MSG_EQ (helloOut.linkMessages[0].neighborInterfaceAddresses[1],
                         Ipv4Address ("1.2.3.5"), "305");

  NS_TEST_ASSERT_MSG_EQ (helloOut.linkMessages[1].linkCode, 3, "306");
  NS_TEST_ASSERT_MSG_EQ (helloOut.linkMessages[1].neighborInterfaceAddresses[0],
                         Ipv4Address ("2.2.3.4"), "307");
  NS_TEST_ASSERT_MSG_EQ (helloOut.linkMessages[1].neighborInterfaceAddresses[1],
                         Ipv4Address ("2.2.3.5"), "308");

  NS_TEST_ASSERT_MSG_EQ (packet.GetSize (), 0, "All bytes in packet were not read");

}

/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * Check Tc olsr messages
 */
class OlsrTcTestCase : public TestCase
{
public:
  OlsrTcTestCase ();
  virtual void DoRun (void);
};

OlsrTcTestCase::OlsrTcTestCase ()
  : TestCase ("Check Tc olsr messages")
{
}
void
OlsrTcTestCase::DoRun (void)
{
  Packet packet;
  olsr::MessageHeader msgIn;
  olsr::MessageHeader::Tc &tcIn = msgIn.GetTc ();

  tcIn.ansn = 0x1234;
  tcIn.neighborAddresses.push_back (Ipv4Address ("1.2.3.4"));
  tcIn.neighborAddresses.push_back (Ipv4Address ("1.2.3.5"));
  packet.AddHeader (msgIn);

  olsr::MessageHeader msgOut;
  packet.RemoveHeader (msgOut);
  olsr::MessageHeader::Tc &tcOut = msgOut.GetTc ();

  NS_TEST_ASSERT_MSG_EQ (tcOut.ansn, 0x1234, "400");
  NS_TEST_ASSERT_MSG_EQ (tcOut.neighborAddresses.size (), 2, "401");

  NS_TEST_ASSERT_MSG_EQ (tcOut.neighborAddresses[0],
                         Ipv4Address ("1.2.3.4"), "402");
  NS_TEST_ASSERT_MSG_EQ (tcOut.neighborAddresses[1],
                         Ipv4Address ("1.2.3.5"), "403");

  NS_TEST_ASSERT_MSG_EQ (packet.GetSize (), 0, "404");

}

/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * Check Hna olsr messages
 */
class OlsrHnaTestCase : public TestCase
{
public:
  OlsrHnaTestCase ();
  virtual void DoRun (void);
};

OlsrHnaTestCase::OlsrHnaTestCase ()
  : TestCase ("Check Hna olsr messages")
{
}

void
OlsrHnaTestCase::DoRun (void)
{
  Packet packet;
  olsr::MessageHeader msgIn;
  olsr::MessageHeader::Hna &hnaIn = msgIn.GetHna ();

  hnaIn.associations.push_back ((olsr::MessageHeader::Hna::Association)
                                { Ipv4Address ("1.2.3.4"), Ipv4Mask ("255.255.255.0")});
  hnaIn.associations.push_back ((olsr::MessageHeader::Hna::Association)
                                { Ipv4Address ("1.2.3.5"), Ipv4Mask ("255.255.0.0")});
  packet.AddHeader (msgIn);

  olsr::MessageHeader msgOut;
  packet.RemoveHeader (msgOut);
  olsr::MessageHeader::Hna &hnaOut = msgOut.GetHna ();

  NS_TEST_ASSERT_MSG_EQ (hnaOut.associations.size (), 2, "500");

  NS_TEST_ASSERT_MSG_EQ (hnaOut.associations[0].address,
                         Ipv4Address ("1.2.3.4"), "501");
  NS_TEST_ASSERT_MSG_EQ (hnaOut.associations[0].mask,
                         Ipv4Mask ("255.255.255.0"), "502");

  NS_TEST_ASSERT_MSG_EQ (hnaOut.associations[1].address,
                         Ipv4Address ("1.2.3.5"), "503");
  NS_TEST_ASSERT_MSG_EQ (hnaOut.associations[1].mask,
                         Ipv4Mask ("255.255.0.0"), "504");

  NS_TEST_ASSERT_MSG_EQ (packet.GetSize (), 0, "All bytes in packet were not read");

}


/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * Check olsr header messages
 */
class OlsrTestSuite : public TestSuite
{
public:
  OlsrTestSuite ();
};

OlsrTestSuite::OlsrTestSuite ()
  : TestSuite ("routing-olsr-header", UNIT)
{
  AddTestCase (new OlsrHnaTestCase (), TestCase::QUICK);
  AddTestCase (new OlsrTcTestCase (), TestCase::QUICK);
  AddTestCase (new OlsrHelloTestCase (), TestCase::QUICK);
  AddTestCase (new OlsrMidTestCase (), TestCase::QUICK);
  AddTestCase (new OlsrEmfTestCase (), TestCase::QUICK);
}

static OlsrTestSuite g_olsrTestSuite; //!< Static variable for test initialization


