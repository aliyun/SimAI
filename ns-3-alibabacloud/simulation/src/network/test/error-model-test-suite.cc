/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 University of Washington
 * Copyright (c) 2013 ResiliNets, ITTC, University of Kansas 
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

/* BurstErrorModel additions
 *
 * Author: Truc Anh N. Nguyen   <annguyen@ittc.ku.edu>
 *         ResiliNets Research Group   http://wiki.ittc.ku.edu/resilinets
 *         James P.G. Sterbenz <jpgs@ittc.ku.edu>, director 
 */

#include "ns3/test.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-channel.h"
#include "ns3/address.h"
#include "ns3/mac48-address.h"
#include "ns3/packet.h"
#include "ns3/callback.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/error-model.h"
#include "ns3/pointer.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/queue.h"

using namespace ns3;

static void SendPacket (int num, Ptr<NetDevice> device, Address& addr)
{
  for (int i = 0; i < num; i++)
    {
      Ptr<Packet> pkt = Create<Packet> (1000);  // 1000 dummy bytes of data
      device->Send (pkt, addr, 0);
    }
}

// Two nodes, two devices, one channel
static void BuildSimpleTopology (Ptr<Node> a, Ptr<Node> b, Ptr<SimpleNetDevice> input, Ptr<SimpleNetDevice> output, Ptr<SimpleChannel> channel)
{
  ObjectFactory queueFactory;
  queueFactory.SetTypeId("ns3::DropTailQueue<Packet>");
  queueFactory.Set("MaxSize", StringValue("100000p")); // Much larger than we need
  Ptr<Queue<Packet> > queueA = queueFactory.Create<Queue<Packet> > ();
  Ptr<Queue<Packet> > queueB = queueFactory.Create<Queue<Packet> > ();

  input->SetQueue(queueA);
  output->SetQueue(queueB);
  a->AddDevice (input);
  b->AddDevice (output);
  input->SetAddress (Mac48Address::Allocate ());
  input->SetChannel (channel);
  input->SetNode (a);
  output->SetChannel (channel);
  output->SetNode (b);
  output->SetAddress (Mac48Address::Allocate ());
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * ErrorModel unit tests.
 */
class ErrorModelSimple : public TestCase
{
public:
  ErrorModelSimple ();
  virtual ~ErrorModelSimple ();

private:
  virtual void DoRun (void);
  /**
   * Receive form a NetDevice
   * \param nd The NetDevice.
   * \param p The received packet.
   * \param protocol The protocol received.
   * \param addr The sender address.
   * \return True on success.
   */
  bool Receive (Ptr<NetDevice> nd, Ptr<const Packet> p, uint16_t protocol, const Address& addr);
  /**
   * Register a Drop
   * \param p The dropped packet
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_count; //!< The received packets counter.
  uint32_t m_drops; //!< The dropped packets counter.
};

// Add some help text to this case to describe what it is intended to test
ErrorModelSimple::ErrorModelSimple ()
  : TestCase ("ErrorModel and PhyRxDrop trace for SimpleNetDevice"), m_count (0), m_drops (0)
{
}

ErrorModelSimple::~ErrorModelSimple ()
{
}

bool 
ErrorModelSimple::Receive (Ptr<NetDevice> nd, Ptr<const Packet> p, uint16_t protocol, const Address& addr)
{
  m_count++;
  return true;
}

void 
ErrorModelSimple::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

void
ErrorModelSimple::DoRun (void)
{
  // Set some arbitrary deterministic values
  RngSeedManager::SetSeed (7);
  RngSeedManager::SetRun (2);

  Ptr<Node> a = CreateObject<Node> ();
  Ptr<Node> b = CreateObject<Node> ();

  Ptr<SimpleNetDevice> input = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleNetDevice> output = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleChannel> channel = CreateObject<SimpleChannel> ();
  BuildSimpleTopology (a, b, input, output, channel);

  output->SetReceiveCallback (MakeCallback (&ErrorModelSimple::Receive, this));
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable> ();
  // Set this variable to a specific stream 
  uv->SetStream (50);

  Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
  em->SetRandomVariable (uv);
  em->SetAttribute ("ErrorRate", DoubleValue (0.001));
  em->SetAttribute ("ErrorUnit", StringValue ("ERROR_UNIT_PACKET"));

  // The below hooks will cause drops and receptions to be counted
  output->SetAttribute ("ReceiveErrorModel", PointerValue (em));
  output->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&ErrorModelSimple::DropEvent, this));

  // Send 10000 packets
  Simulator::Schedule (Seconds (0), &SendPacket, 10000, input, output->GetAddress ());

  Simulator::Run ();
  Simulator::Destroy ();

  // For this combination of values, we expect about 1 packet in 1000 to be
  // dropped.  For this specific RNG stream, we see 9991 receptions and 9 drops
  NS_TEST_ASSERT_MSG_EQ (m_count, 9991, "Wrong number of receptions.");
  NS_TEST_ASSERT_MSG_EQ (m_drops, 9, "Wrong number of drops.");
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * BurstErrorModel unit tests.
 */
class BurstErrorModelSimple : public TestCase
{
public:
  BurstErrorModelSimple ();
  virtual ~BurstErrorModelSimple ();

private:
  virtual void DoRun (void);
  /**
   * Receive form a NetDevice
   * \param nd The NetDevice.
   * \param p The received packet.
   * \param protocol The protocol received.
   * \param addr The sender address.
   * \return True on success.
   */
  bool Receive (Ptr<NetDevice> nd, Ptr<const Packet> p, uint16_t protocol, const Address& addr);
  /**
   * Register a Drop
   * \param p The dropped packet
   */
  void DropEvent (Ptr<const Packet> p);

  uint32_t m_count; //!< The received packets counter.
  uint32_t m_drops; //!< The dropped packets counter.
};

// Add some help text to this case to describe what it is intended to test
BurstErrorModelSimple::BurstErrorModelSimple ()
  : TestCase ("ErrorModel and PhyRxDrop trace for SimpleNetDevice"), m_count (0), m_drops (0)
{
}

BurstErrorModelSimple::~BurstErrorModelSimple ()
{
}

bool
BurstErrorModelSimple::Receive (Ptr<NetDevice> nd, Ptr<const Packet> p, uint16_t protocol, const Address& addr)
{
  m_count++;
  return true;
}

void
BurstErrorModelSimple::DropEvent (Ptr<const Packet> p)
{
  m_drops++;
}

void
BurstErrorModelSimple::DoRun (void)
{
  // Set some arbitrary deterministic values
  RngSeedManager::SetSeed (5);
  RngSeedManager::SetRun (8);

  Ptr<Node> a = CreateObject<Node> ();
  Ptr<Node> b = CreateObject<Node> ();

  Ptr<SimpleNetDevice> input = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleNetDevice> output = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleChannel> channel = CreateObject<SimpleChannel> ();
  BuildSimpleTopology (a, b, input, output, channel);

  output->SetReceiveCallback (MakeCallback (&BurstErrorModelSimple::Receive, this));
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable> ();
  // Set this variable to a specific stream
  uv->SetStream (50);

  Ptr<BurstErrorModel> em = CreateObject<BurstErrorModel> ();
  em->SetRandomVariable (uv);
  em->SetAttribute ("ErrorRate", DoubleValue (0.01));

  // Assign the underlying error model random variables to specific streams
  em->AssignStreams (51);

  // The below hooks will cause drops and receptions to be counted
  output->SetAttribute ("ReceiveErrorModel", PointerValue (em));
  output->TraceConnectWithoutContext ("PhyRxDrop", MakeCallback (&BurstErrorModelSimple::DropEvent, this));

  // Send 10000 packets
  Simulator::Schedule (Seconds (0), &SendPacket, 10000, input, output->GetAddress ());

  Simulator::Run ();
  Simulator::Destroy ();

  // With the burst error rate to be 0.01 and the burst size to be from 1 to 4,
  // we expect about 2.5 packets being dropped every 1000 packets.
  // That means for 10000 packets, we expect a total of about 250 packet drops.
  // For this specific RNG seed, we see 9740 receptions and 260 drops.
  NS_TEST_ASSERT_MSG_EQ (m_count, 9740, "Wrong number of receptions.");
  NS_TEST_ASSERT_MSG_EQ (m_drops, 260 , "Wrong number of drops.");
}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief ErrorModel TestSuite
 *
 *  This is the start of an error model test suite.  For starters, this is
 *  just testing that the SimpleNetDevice is working but this can be
 *  extended to many more test cases in the future
 */
class ErrorModelTestSuite : public TestSuite
{
public:
  ErrorModelTestSuite ();
};

ErrorModelTestSuite::ErrorModelTestSuite ()
  : TestSuite ("error-model", UNIT)
{
  AddTestCase (new ErrorModelSimple, TestCase::QUICK);
  AddTestCase (new BurstErrorModelSimple, TestCase::QUICK);
}

// Do not forget to allocate an instance of this TestSuite
static ErrorModelTestSuite errorModelTestSuite; //!< Static variable for test initialization
