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
#include "ns3/aodv-neighbor.h"
#include "ns3/aodv-packet.h"
#include "ns3/aodv-rqueue.h"
#include "ns3/aodv-rtable.h"
#include "ns3/ipv4-route.h"

namespace ns3 {
namespace aodv {

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for neighbors
 */
struct NeighborTest : public TestCase
{
  NeighborTest () : TestCase ("Neighbor"),
                    neighbor (0)
  {
  }
  virtual void DoRun ();
  /**
   * Handler test function
   * \param addr the IPv4 address of the neighbor
   */
  void Handler (Ipv4Address addr);
  /// Check timeout function 1
  void CheckTimeout1 ();
  /// Check timeout function 2
  void CheckTimeout2 ();
  /// Check timeout function 3
  void CheckTimeout3 ();
  /// The Neighbors
  Neighbors * neighbor;
};

void
NeighborTest::Handler (Ipv4Address addr)
{
}

void
NeighborTest::CheckTimeout1 ()
{
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.2.3.4")), true, "Neighbor exists");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.1.1.1")), true, "Neighbor exists");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("2.2.2.2")), true, "Neighbor exists");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("3.3.3.3")), true, "Neighbor exists");
}
void
NeighborTest::CheckTimeout2 ()
{
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.2.3.4")), false, "Neighbor doesn't exist");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.1.1.1")), false, "Neighbor doesn't exist");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("2.2.2.2")), false, "Neighbor doesn't exist");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("3.3.3.3")), true, "Neighbor exists");
}
void
NeighborTest::CheckTimeout3 ()
{
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.2.3.4")), false, "Neighbor doesn't exist");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.1.1.1")), false, "Neighbor doesn't exist");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("2.2.2.2")), false, "Neighbor doesn't exist");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("3.3.3.3")), false, "Neighbor doesn't exist");
}

void
NeighborTest::DoRun ()
{
  Neighbors nb (Seconds (1));
  neighbor = &nb;
  neighbor->SetCallback (MakeCallback (&NeighborTest::Handler, this));
  neighbor->Update (Ipv4Address ("1.2.3.4"), Seconds (1));
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.2.3.4")), true, "Neighbor exists");
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("4.3.2.1")), false, "Neighbor doesn't exist");
  neighbor->Update (Ipv4Address ("1.2.3.4"), Seconds (10));
  NS_TEST_EXPECT_MSG_EQ (neighbor->IsNeighbor (Ipv4Address ("1.2.3.4")), true, "Neighbor exists");
  NS_TEST_EXPECT_MSG_EQ (neighbor->GetExpireTime (Ipv4Address ("1.2.3.4")), Seconds (10), "Known expire time");
  NS_TEST_EXPECT_MSG_EQ (neighbor->GetExpireTime (Ipv4Address ("4.3.2.1")), Seconds (0), "Known expire time");
  neighbor->Update (Ipv4Address ("1.1.1.1"), Seconds (5));
  neighbor->Update (Ipv4Address ("2.2.2.2"), Seconds (10));
  neighbor->Update (Ipv4Address ("3.3.3.3"), Seconds (20));

  Simulator::Schedule (Seconds (2), &NeighborTest::CheckTimeout1, this);
  Simulator::Schedule (Seconds (15), &NeighborTest::CheckTimeout2, this);
  Simulator::Schedule (Seconds (30), &NeighborTest::CheckTimeout3, this);
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Type header test case
 */
struct TypeHeaderTest : public TestCase
{
  TypeHeaderTest () : TestCase ("AODV TypeHeader")
  {
  }
  virtual void DoRun ()
  {
    TypeHeader h (AODVTYPE_RREQ);
    NS_TEST_EXPECT_MSG_EQ (h.IsValid (), true, "Default header is valid");
    NS_TEST_EXPECT_MSG_EQ (h.Get (), AODVTYPE_RREQ, "Default header is RREQ");

    Ptr<Packet> p = Create<Packet> ();
    p->AddHeader (h);
    TypeHeader h2 (AODVTYPE_RREP);
    uint32_t bytes = p->RemoveHeader (h2);
    NS_TEST_EXPECT_MSG_EQ (bytes, 1, "Type header is 1 byte long");
    NS_TEST_EXPECT_MSG_EQ (h, h2, "Round trip serialization works");
  }
};

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for RREQ
 */
struct RreqHeaderTest : public TestCase
{
  RreqHeaderTest () : TestCase ("AODV RREQ")
  {
  }
  virtual void DoRun ()
  {
    RreqHeader h (/*flags*/ 0, /*reserved*/ 0, /*hopCount*/ 6, /*requestID*/ 1, /*dst*/ Ipv4Address ("1.2.3.4"),
                            /*dstSeqNo*/ 40, /*origin*/ Ipv4Address ("4.3.2.1"), /*originSeqNo*/ 10);
    NS_TEST_EXPECT_MSG_EQ (h.GetGratuitousRrep (), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDestinationOnly (), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetHopCount (), 6, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetId (), 1, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDst (), Ipv4Address ("1.2.3.4"), "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDstSeqno (), 40, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetOrigin (), Ipv4Address ("4.3.2.1"), "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetOriginSeqno (), 10, "trivial");

    h.SetGratuitousRrep (true);
    NS_TEST_EXPECT_MSG_EQ (h.GetGratuitousRrep (), true, "trivial");
    h.SetDestinationOnly (true);
    NS_TEST_EXPECT_MSG_EQ (h.GetDestinationOnly (), true, "trivial");
    h.SetUnknownSeqno (true);
    NS_TEST_EXPECT_MSG_EQ (h.GetUnknownSeqno (), true, "trivial");
    h.SetDst (Ipv4Address ("1.1.1.1"));
    NS_TEST_EXPECT_MSG_EQ (h.GetDst (), Ipv4Address ("1.1.1.1"), "trivial");
    h.SetDstSeqno (5);
    NS_TEST_EXPECT_MSG_EQ (h.GetDstSeqno (), 5, "trivial");
    h.SetHopCount (7);
    NS_TEST_EXPECT_MSG_EQ (h.GetHopCount (), 7, "trivial");
    h.SetId (55);
    NS_TEST_EXPECT_MSG_EQ (h.GetId (), 55, "trivial");
    h.SetOrigin (Ipv4Address ("4.4.4.4"));
    NS_TEST_EXPECT_MSG_EQ (h.GetOrigin (), Ipv4Address ("4.4.4.4"), "trivial");
    h.SetOriginSeqno (23);
    NS_TEST_EXPECT_MSG_EQ (h.GetOriginSeqno (), 23, "trivial");

    Ptr<Packet> p = Create<Packet> ();
    p->AddHeader (h);
    RreqHeader h2;
    uint32_t bytes = p->RemoveHeader (h2);
    NS_TEST_EXPECT_MSG_EQ (bytes, 23, "RREP is 23 bytes long");
    NS_TEST_EXPECT_MSG_EQ (h, h2, "Round trip serialization works");

  }
};

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for RREP
 */
struct RrepHeaderTest : public TestCase
{
  RrepHeaderTest () : TestCase ("AODV RREP")
  {
  }
  virtual void DoRun ()
  {
    RrepHeader h (/*prefixSize*/ 0, /*hopCount*/ 12, /*dst*/ Ipv4Address ("1.2.3.4"), /*dstSeqNo*/ 2,
                                 /*origin*/ Ipv4Address ("4.3.2.1"), /*lifetime*/ Seconds (3));
    NS_TEST_EXPECT_MSG_EQ (h.GetPrefixSize (), 0, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetHopCount (), 12, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDst (), Ipv4Address ("1.2.3.4"), "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDstSeqno (), 2, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetOrigin (), Ipv4Address ("4.3.2.1"), "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetLifeTime (), Seconds (3), "trivial");
    h.SetDst (Ipv4Address ("1.1.1.1"));
    NS_TEST_EXPECT_MSG_EQ (h.GetDst (), Ipv4Address ("1.1.1.1"), "trivial");
    h.SetDstSeqno (123);
    NS_TEST_EXPECT_MSG_EQ (h.GetDstSeqno (), 123, "trivial");
    h.SetOrigin (Ipv4Address ("4.4.4.4"));
    NS_TEST_EXPECT_MSG_EQ (h.GetOrigin (), Ipv4Address ("4.4.4.4"), "trivial");
    h.SetLifeTime (MilliSeconds (1200));
    NS_TEST_EXPECT_MSG_EQ (h.GetLifeTime (), MilliSeconds (1200), "trivial");
    h.SetAckRequired (true);
    NS_TEST_EXPECT_MSG_EQ (h.GetAckRequired (), true, "trivial");
    h.SetAckRequired (false);
    NS_TEST_EXPECT_MSG_EQ (h.GetAckRequired (), false, "trivial");
    h.SetPrefixSize (2);
    NS_TEST_EXPECT_MSG_EQ (h.GetPrefixSize (), 2, "trivial");
    h.SetHopCount (15);
    NS_TEST_EXPECT_MSG_EQ (h.GetHopCount (), 15, "trivial");

    h.SetHello (Ipv4Address ("10.0.0.2"), 9, Seconds (15));
    NS_TEST_EXPECT_MSG_EQ (h.GetDst (), h.GetOrigin (), "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDstSeqno (), 9, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetLifeTime (), Seconds (15), "trivial");

    Ptr<Packet> p = Create<Packet> ();
    p->AddHeader (h);
    RrepHeader h2;
    uint32_t bytes = p->RemoveHeader (h2);
    NS_TEST_EXPECT_MSG_EQ (bytes, 19, "RREP is 19 bytes long");
    NS_TEST_EXPECT_MSG_EQ (h, h2, "Round trip serialization works");
  }
};

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for RREP-ACK
 */
struct RrepAckHeaderTest : public TestCase
{
  RrepAckHeaderTest () : TestCase ("AODV RREP-ACK")
  {
  }
  virtual void DoRun ()
  {
    RrepAckHeader h;
    Ptr<Packet> p = Create<Packet> ();
    p->AddHeader (h);
    RrepAckHeader h2;
    uint32_t bytes = p->RemoveHeader (h2);
    NS_TEST_EXPECT_MSG_EQ (bytes, 1, "ACK is 1 byte long");
    NS_TEST_EXPECT_MSG_EQ (h, h2, "Round trip serialization works");
  }
};

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for RERR
 */
struct RerrHeaderTest : public TestCase
{
  RerrHeaderTest () : TestCase ("AODV RERR")
  {
  }
  virtual void DoRun ()
  {
    RerrHeader h;
    h.SetNoDelete (true);
    NS_TEST_EXPECT_MSG_EQ (h.GetNoDelete (), true, "trivial");
    Ipv4Address dst = Ipv4Address ("1.2.3.4");
    NS_TEST_EXPECT_MSG_EQ (h.AddUnDestination (dst, 12), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDestCount (), 1, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.AddUnDestination (dst, 13), true, "trivial");
    Ipv4Address dst2 = Ipv4Address ("4.3.2.1");
    NS_TEST_EXPECT_MSG_EQ (h.AddUnDestination (dst2, 12), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetDestCount (), 2, "trivial");

    Ptr<Packet> p = Create<Packet> ();
    p->AddHeader (h);
    RerrHeader h2;
    uint32_t bytes = p->RemoveHeader (h2);
    NS_TEST_EXPECT_MSG_EQ (bytes, h.GetSerializedSize (), "(De)Serialized size match");
    NS_TEST_EXPECT_MSG_EQ (h, h2, "Round trip serialization works");
  }
};

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for AODV routing table entry
 */
struct QueueEntryTest : public TestCase
{
  QueueEntryTest () : TestCase ("QueueEntry")
  {
  }
  /**
   * Unicast test function
   * \param route the IPv4 route
   * \param packet the packet
   * \param header the IPv4 header
   */
  void Unicast (Ptr<Ipv4Route> route, Ptr<const Packet> packet, const Ipv4Header & header)
  {
  }
  /**
   * Error test function
   * \param p The packet
   * \param h The header
   * \param e the socket error
   */
  void Error (Ptr<const Packet> p, const Ipv4Header & h, Socket::SocketErrno e)
  {
  }
  /**
   * Unicast 2 testfunction
   * \param route The IPv4 route
   * \param packet The packet
   * \param header The header
   */
  void Unicast2 (Ptr<Ipv4Route> route, Ptr<const Packet> packet, const Ipv4Header & header)
  {
  }
  /**
   * Error2 test function
   * \param p The packet
   * \param h The header
   * \param e the socket error
   */
  void Error2 (Ptr<const Packet> p, const Ipv4Header & h, Socket::SocketErrno e)
  {
  }
  virtual void DoRun ()
  {
    Ptr<const Packet> packet = Create<Packet> ();
    Ipv4Header h;
    h.SetDestination (Ipv4Address ("1.2.3.4"));
    h.SetSource (Ipv4Address ("4.3.2.1"));
    Ipv4RoutingProtocol::UnicastForwardCallback ucb = MakeCallback (&QueueEntryTest::Unicast, this);
    Ipv4RoutingProtocol::ErrorCallback ecb = MakeCallback (&QueueEntryTest::Error, this);
    QueueEntry entry (packet, h, ucb, ecb, Seconds (1));
    NS_TEST_EXPECT_MSG_EQ (h.GetDestination (),  entry.GetIpv4Header ().GetDestination (), "trivial");
    NS_TEST_EXPECT_MSG_EQ (h.GetSource (),  entry.GetIpv4Header ().GetSource (), "trivial");
    NS_TEST_EXPECT_MSG_EQ (ucb.IsEqual (entry.GetUnicastForwardCallback ()), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (ecb.IsEqual (entry.GetErrorCallback ()), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (entry.GetExpireTime (), Seconds (1), "trivial");
    NS_TEST_EXPECT_MSG_EQ (entry.GetPacket (), packet, "trivial");
    entry.SetExpireTime (Seconds (3));
    NS_TEST_EXPECT_MSG_EQ (entry.GetExpireTime (), Seconds (3), "trivial");
    Ipv4Header h2;
    h2.SetDestination (Ipv4Address ("1.1.1.1"));
    entry.SetIpv4Header (h2);
    NS_TEST_EXPECT_MSG_EQ (entry.GetIpv4Header ().GetDestination (), Ipv4Address ("1.1.1.1"), "trivial");
    Ipv4RoutingProtocol::UnicastForwardCallback ucb2 = MakeCallback (&QueueEntryTest::Unicast2, this);
    Ipv4RoutingProtocol::ErrorCallback ecb2 = MakeCallback (&QueueEntryTest::Error2, this);
    entry.SetErrorCallback (ecb2);
    NS_TEST_EXPECT_MSG_EQ (ecb2.IsEqual (entry.GetErrorCallback ()), true, "trivial");
    entry.SetUnicastForwardCallback (ucb2);
    NS_TEST_EXPECT_MSG_EQ (ucb2.IsEqual (entry.GetUnicastForwardCallback ()), true, "trivial");
  }
};
//-----------------------------------------------------------------------------
/// Unit test for RequestQueue
struct AodvRqueueTest : public TestCase
{
  AodvRqueueTest () : TestCase ("Rqueue"),
                      q (64, Seconds (30))
  {
  }
  virtual void DoRun ();
  /**
   * Unicast test function
   * \param route the IPv4 route
   * \param packet the packet
   * \param header the IPv4 header
   */
  void Unicast (Ptr<Ipv4Route> route, Ptr<const Packet> packet, const Ipv4Header & header)
  {
  }
  /**
   * Error test function
   * \param p The packet
   * \param h The header
   * \param e the socket error
   */
  void Error (Ptr<const Packet> p, const Ipv4Header & h, Socket::SocketErrno e)
  {
  }
  /// Check size limit function
  void CheckSizeLimit ();
  /// Check timeout function
  void CheckTimeout ();

  /// Request queue
  RequestQueue q;
};

void
AodvRqueueTest::DoRun ()
{
  NS_TEST_EXPECT_MSG_EQ (q.GetMaxQueueLen (), 64, "trivial");
  q.SetMaxQueueLen (32);
  NS_TEST_EXPECT_MSG_EQ (q.GetMaxQueueLen (), 32, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.GetQueueTimeout (), Seconds (30), "trivial");
  q.SetQueueTimeout (Seconds (10));
  NS_TEST_EXPECT_MSG_EQ (q.GetQueueTimeout (), Seconds (10), "trivial");

  Ptr<const Packet> packet = Create<Packet> ();
  Ipv4Header h;
  h.SetDestination (Ipv4Address ("1.2.3.4"));
  h.SetSource (Ipv4Address ("4.3.2.1"));
  Ipv4RoutingProtocol::UnicastForwardCallback ucb = MakeCallback (&AodvRqueueTest::Unicast, this);
  Ipv4RoutingProtocol::ErrorCallback ecb = MakeCallback (&AodvRqueueTest::Error, this);
  QueueEntry e1 (packet, h, ucb, ecb, Seconds (1));
  q.Enqueue (e1);
  q.Enqueue (e1);
  q.Enqueue (e1);
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("1.2.3.4")), true, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("1.1.1.1")), false, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 1, "trivial");
  q.DropPacketWithDst (Ipv4Address ("1.2.3.4"));
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("1.2.3.4")), false, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 0, "trivial");

  h.SetDestination (Ipv4Address ("2.2.2.2"));
  QueueEntry e2 (packet, h, ucb, ecb, Seconds (1));
  q.Enqueue (e1);
  q.Enqueue (e2);
  Ptr<Packet> packet2 = Create<Packet> ();
  QueueEntry e3 (packet2, h, ucb, ecb, Seconds (1));
  NS_TEST_EXPECT_MSG_EQ (q.Dequeue (Ipv4Address ("3.3.3.3"), e3), false, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.Dequeue (Ipv4Address ("2.2.2.2"), e3), true, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("2.2.2.2")), false, "trivial");
  q.Enqueue (e2);
  q.Enqueue (e3);
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 2, "trivial");
  Ptr<Packet> packet4 = Create<Packet> ();
  h.SetDestination (Ipv4Address ("1.2.3.4"));
  QueueEntry e4 (packet4, h, ucb, ecb, Seconds (20));
  q.Enqueue (e4);
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 3, "trivial");
  q.DropPacketWithDst (Ipv4Address ("1.2.3.4"));
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 1, "trivial");

  CheckSizeLimit ();

  Ipv4Header header2;
  Ipv4Address dst2 ("1.2.3.4");
  header2.SetDestination (dst2);

  Simulator::Schedule (q.GetQueueTimeout () + Seconds (1), &AodvRqueueTest::CheckTimeout, this);

  Simulator::Run ();
  Simulator::Destroy ();
}

void
AodvRqueueTest::CheckSizeLimit ()
{
  Ptr<Packet> packet = Create<Packet> ();
  Ipv4Header header;
  Ipv4RoutingProtocol::UnicastForwardCallback ucb = MakeCallback (&AodvRqueueTest::Unicast, this);
  Ipv4RoutingProtocol::ErrorCallback ecb = MakeCallback (&AodvRqueueTest::Error, this);
  QueueEntry e1 (packet, header, ucb, ecb, Seconds (1));

  for (uint32_t i = 0; i < q.GetMaxQueueLen (); ++i)
    {
      q.Enqueue (e1);
    }
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 2, "trivial");

  for (uint32_t i = 0; i < q.GetMaxQueueLen (); ++i)
    {
      q.Enqueue (e1);
    }
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 2, "trivial");
}

void
AodvRqueueTest::CheckTimeout ()
{
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 0, "Must be empty now");
}

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for AODV routing table entry
 */
struct AodvRtableEntryTest : public TestCase
{
  AodvRtableEntryTest () : TestCase ("RtableEntry")
  {
  }
  virtual void DoRun ()
  {
    Ptr<NetDevice> dev;
    Ipv4InterfaceAddress iface;
    RoutingTableEntry rt (/*output device*/ dev, /*dst*/ Ipv4Address ("1.2.3.4"), /*validSeqNo*/ true, /*seqNo*/ 10,
                                            /*interface*/ iface, /*hop*/ 5, /*next hop*/ Ipv4Address ("3.3.3.3"), /*lifetime*/ Seconds (10));
    NS_TEST_EXPECT_MSG_EQ (rt.GetOutputDevice (), dev, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetDestination (), Ipv4Address ("1.2.3.4"), "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetValidSeqNo (), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetSeqNo (), 10, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetInterface (), iface, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetHop (), 5, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetNextHop (), Ipv4Address ("3.3.3.3"), "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetLifeTime (), Seconds (10), "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetFlag (), VALID, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetRreqCnt (), 0, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.IsPrecursorListEmpty (), true, "trivial");

    Ptr<NetDevice> dev2;
    Ipv4InterfaceAddress iface2;
    rt.SetOutputDevice (dev2);
    NS_TEST_EXPECT_MSG_EQ (rt.GetOutputDevice (), dev2, "trivial");
    rt.SetInterface (iface2);
    NS_TEST_EXPECT_MSG_EQ (rt.GetInterface (), iface2, "trivial");
    rt.SetValidSeqNo (false);
    NS_TEST_EXPECT_MSG_EQ (rt.GetValidSeqNo (), false, "trivial");
    rt.SetFlag (INVALID);
    NS_TEST_EXPECT_MSG_EQ (rt.GetFlag (), INVALID, "trivial");
    rt.SetFlag (IN_SEARCH);
    NS_TEST_EXPECT_MSG_EQ (rt.GetFlag (), IN_SEARCH, "trivial");
    rt.SetHop (12);
    NS_TEST_EXPECT_MSG_EQ (rt.GetHop (), 12, "trivial");
    rt.SetLifeTime (Seconds (1));
    NS_TEST_EXPECT_MSG_EQ (rt.GetLifeTime (), Seconds (1), "trivial");
    rt.SetNextHop (Ipv4Address ("1.1.1.1"));
    NS_TEST_EXPECT_MSG_EQ (rt.GetNextHop (), Ipv4Address ("1.1.1.1"), "trivial");
    rt.SetUnidirectional (true);
    NS_TEST_EXPECT_MSG_EQ (rt.IsUnidirectional (), true, "trivial");
    rt.SetBlacklistTimeout (Seconds (7));
    NS_TEST_EXPECT_MSG_EQ (rt.GetBlacklistTimeout (), Seconds (7), "trivial");
    rt.SetRreqCnt (2);
    NS_TEST_EXPECT_MSG_EQ (rt.GetRreqCnt (), 2, "trivial");
    rt.IncrementRreqCnt ();
    NS_TEST_EXPECT_MSG_EQ (rt.GetRreqCnt (), 3, "trivial");
    rt.Invalidate (Seconds (13));
    NS_TEST_EXPECT_MSG_EQ (rt.GetFlag (), INVALID, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetLifeTime (), Seconds (13), "trivial");
    rt.SetLifeTime (MilliSeconds (100));
    NS_TEST_EXPECT_MSG_EQ (rt.GetLifeTime (), MilliSeconds (100), "trivial");
    Ptr<Ipv4Route> route = rt.GetRoute ();
    NS_TEST_EXPECT_MSG_EQ (route->GetDestination (), Ipv4Address ("1.2.3.4"), "trivial");

    NS_TEST_EXPECT_MSG_EQ (rt.InsertPrecursor (Ipv4Address ("10.0.0.1")), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.IsPrecursorListEmpty (), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.InsertPrecursor (Ipv4Address ("10.0.0.2")), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.InsertPrecursor (Ipv4Address ("10.0.0.2")), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.LookupPrecursor (Ipv4Address ("10.0.0.3")), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.LookupPrecursor (Ipv4Address ("10.0.0.1")), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.DeletePrecursor (Ipv4Address ("10.0.0.2")), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.LookupPrecursor (Ipv4Address ("10.0.0.2")), false, "trivial");
    std::vector<Ipv4Address> prec;
    rt.GetPrecursors (prec);
    NS_TEST_EXPECT_MSG_EQ (prec.size (), 1, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.InsertPrecursor (Ipv4Address ("10.0.0.4")), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.DeletePrecursor (Ipv4Address ("10.0.0.5")), false, "trivial");
    rt.GetPrecursors (prec);
    NS_TEST_EXPECT_MSG_EQ (prec.size (), 2, "trivial");
    rt.DeleteAllPrecursors ();
    NS_TEST_EXPECT_MSG_EQ (rt.IsPrecursorListEmpty (), true, "trivial");
    rt.GetPrecursors (prec);
    NS_TEST_EXPECT_MSG_EQ (prec.size (), 2, "trivial");
    Simulator::Destroy ();
  }
};

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief Unit test for AODV routing table
 */
struct AodvRtableTest : public TestCase
{
  AodvRtableTest () : TestCase ("Rtable")
  {
  }
  virtual void DoRun ()
  {
    RoutingTable rtable (Seconds (2));
    NS_TEST_EXPECT_MSG_EQ (rtable.GetBadLinkLifetime (), Seconds (2), "trivial");
    rtable.SetBadLinkLifetime (Seconds (1));
    NS_TEST_EXPECT_MSG_EQ (rtable.GetBadLinkLifetime (), Seconds (1), "trivial");
    Ptr<NetDevice> dev;
    Ipv4InterfaceAddress iface;
    RoutingTableEntry rt (/*output device*/ dev, /*dst*/ Ipv4Address ("1.2.3.4"), /*validSeqNo*/ true, /*seqNo*/ 10,
                                            /*interface*/ iface, /*hop*/ 5, /*next hop*/ Ipv4Address ("1.1.1.1"), /*lifetime*/ Seconds (10));
    NS_TEST_EXPECT_MSG_EQ (rtable.AddRoute (rt), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.AddRoute (rt), false, "trivial");
    RoutingTableEntry rt2 (/*output device*/ dev, /*dst*/ Ipv4Address ("4.3.2.1"), /*validSeqNo*/ false, /*seqNo*/ 0,
                                             /*interface*/ iface, /*hop*/ 15, /*next hop*/ Ipv4Address ("1.1.1.1"), /*lifetime*/ Seconds (1));
    NS_TEST_EXPECT_MSG_EQ (rtable.AddRoute (rt2), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.LookupRoute (rt2.GetDestination (), rt), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt2.GetDestination (), rt.GetDestination (), "trivial");
    rt.SetHop (20);
    rt.InsertPrecursor (Ipv4Address ("10.0.0.3"));
    NS_TEST_EXPECT_MSG_EQ (rtable.Update (rt), true, "trivial");
    RoutingTableEntry rt3;
    NS_TEST_EXPECT_MSG_EQ (rtable.LookupRoute (Ipv4Address ("10.0.0.1"), rt), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.Update (rt3), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.SetEntryState (Ipv4Address ("10.0.0.1"), INVALID), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.SetEntryState (Ipv4Address ("1.2.3.4"), IN_SEARCH), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.DeleteRoute (Ipv4Address ("5.5.5.5")), false, "trivial");
    RoutingTableEntry rt4 (/*output device*/ dev, /*dst*/ Ipv4Address ("5.5.5.5"), /*validSeqNo*/ false, /*seqNo*/ 0,
                                             /*interface*/ iface, /*hop*/ 15, /*next hop*/ Ipv4Address ("1.1.1.1"), /*lifetime*/ Seconds (-10));
    NS_TEST_EXPECT_MSG_EQ (rtable.AddRoute (rt4), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.SetEntryState (Ipv4Address ("5.5.5.5"), INVALID), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.LookupRoute (Ipv4Address ("5.5.5.5"), rt), false, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.MarkLinkAsUnidirectional (Ipv4Address ("1.2.3.4"), Seconds (2)), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.LookupRoute (Ipv4Address ("1.2.3.4"), rt), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.IsUnidirectional (), true, "trivial");
    rt.SetLifeTime (Seconds (-5));
    NS_TEST_EXPECT_MSG_EQ (rtable.Update (rt), true, "trivial");
    std::map<Ipv4Address, uint32_t> unreachable;
    rtable.GetListOfDestinationWithNextHop (Ipv4Address ("1.1.1.1"), unreachable);
    NS_TEST_EXPECT_MSG_EQ (unreachable.size (), 2, "trivial");
    unreachable.insert (std::make_pair (Ipv4Address ("4.3.2.1"), 3));
    rtable.InvalidateRoutesWithDst (unreachable);
    NS_TEST_EXPECT_MSG_EQ (rtable.LookupRoute (Ipv4Address ("4.3.2.1"), rt), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rt.GetFlag (), INVALID, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.DeleteRoute (Ipv4Address ("1.2.3.4")), true, "trivial");
    NS_TEST_EXPECT_MSG_EQ (rtable.DeleteRoute (Ipv4Address ("1.2.3.4")), false, "trivial");
    Simulator::Destroy ();
  }
};

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief AODV test suite
 */
class AodvTestSuite : public TestSuite
{
public:
  AodvTestSuite () : TestSuite ("routing-aodv", UNIT)
  {
    AddTestCase (new NeighborTest, TestCase::QUICK);
    AddTestCase (new TypeHeaderTest, TestCase::QUICK);
    AddTestCase (new RreqHeaderTest, TestCase::QUICK);
    AddTestCase (new RrepHeaderTest, TestCase::QUICK);
    AddTestCase (new RrepAckHeaderTest, TestCase::QUICK);
    AddTestCase (new RerrHeaderTest, TestCase::QUICK);
    AddTestCase (new QueueEntryTest, TestCase::QUICK);
    AddTestCase (new AodvRqueueTest, TestCase::QUICK);
    AddTestCase (new AodvRtableEntryTest, TestCase::QUICK);
    AddTestCase (new AodvRtableTest, TestCase::QUICK);
  }
} g_aodvTestSuite; ///< the test suite

}  // namespace aodv
}  // namespace ns3
