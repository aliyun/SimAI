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
 * Author: Pavel Boyko <boyko@iitp.ru>
 */
#include "ns3/test.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/mgt-headers.h"
#include "ns3/dot11s-mac-header.h"
#include "ns3/hwmp-rtable.h"
#include "ns3/peer-link-frame.h"
#include "ns3/ie-dot11s-peer-management.h"

using namespace ns3;
using namespace dot11s;

/**
 * \ingroup mesh-test
 * \defgroup dot11s-test dot11s sub-module tests
 */


/**
 * \ingroup dot11s-test
 * \ingroup tests
 *
 * \brief Built-in self test for MeshHeader
 */
struct MeshHeaderTest : public TestCase
{
  MeshHeaderTest () :
    TestCase ("Dot11sMeshHeader roundtrip serialization")
  {
  }
  void DoRun ();
};

void
MeshHeaderTest::DoRun ()
{
  {
    MeshHeader a;
    a.SetAddressExt (3);
    a.SetAddr4 (Mac48Address ("11:22:33:44:55:66"));
    a.SetAddr5 (Mac48Address ("11:00:33:00:55:00"));
    a.SetAddr6 (Mac48Address ("00:22:00:44:00:66"));
    a.SetMeshTtl (122);
    a.SetMeshSeqno (321);
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader (a);
    MeshHeader b;
    packet->RemoveHeader (b);
    NS_TEST_ASSERT_MSG_EQ (a, b, "Mesh header roundtrip serialization works, 3 addresses");
  }
  {
    MeshHeader a;
    a.SetAddressExt (2);
    a.SetAddr5 (Mac48Address ("11:00:33:00:55:00"));
    a.SetAddr6 (Mac48Address ("00:22:00:44:00:66"));
    a.SetMeshTtl (122);
    a.SetMeshSeqno (321);
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader (a);
    MeshHeader b;
    packet->RemoveHeader (b);
    NS_TEST_ASSERT_MSG_EQ (a, b, "Mesh header roundtrip serialization works, 2 addresses");
  }
  {
    MeshHeader a;
    a.SetAddressExt (1);
    a.SetAddr4 (Mac48Address ("11:22:33:44:55:66"));
    a.SetMeshTtl (122);
    a.SetMeshSeqno (321);
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader (a);
    MeshHeader b;
    packet->RemoveHeader (b);
    NS_TEST_ASSERT_MSG_EQ (a, b, "Mesh header roundtrip serialization works, 1 address");
  }
}

/**
 * \ingroup mesh-test
 * \ingroup tests
 *
 * \brief Unit test for HwmpRtable
 */
class HwmpRtableTest : public TestCase
{
public:
  HwmpRtableTest ();
  virtual void DoRun ();

private:
  /// Test Add apth and lookup path;
  void TestLookup ();

  /// Test add path and try to lookup after entry has expired
  void TestAddPath ();
  /// Test add path and try to lookup after entry has expired
  void TestExpire ();

  /// Test add precursors and find precursor list in rtable
  void TestPrecursorAdd ();
  /// Test add precursors and find precursor list in rtable
  void TestPrecursorFind ();

private:
  Mac48Address dst; ///< destination address
  Mac48Address hop; ///< hop address
  uint32_t iface; ///< interface
  uint32_t metric; ///< metric
  uint32_t seqnum; ///< sequence number
  Time expire; ///< expiration time
  Ptr<HwmpRtable> table; ///< tab;e
  std::vector<Mac48Address> precursors; ///< precursors
};

HwmpRtableTest::HwmpRtableTest () :
  TestCase ("HWMP routing table"), 
  dst ("01:00:00:01:00:01"), 
  hop ("01:00:00:01:00:03"),
  iface (8010), 
  metric (10), 
  seqnum (1), 
  expire (Seconds (10))
{
  precursors.push_back (Mac48Address ("00:10:20:30:40:50"));
  precursors.push_back (Mac48Address ("00:11:22:33:44:55"));
  precursors.push_back (Mac48Address ("00:01:02:03:04:05"));
}

void
HwmpRtableTest::TestLookup ()
{
  HwmpRtable::LookupResult correct (hop, iface, metric, seqnum);

  // Reactive path
  table->AddReactivePath (dst, hop, iface, metric, expire, seqnum);
  NS_TEST_EXPECT_MSG_EQ ((table->LookupReactive (dst) == correct), true, "Reactive lookup works");
  table->DeleteReactivePath (dst);
  NS_TEST_EXPECT_MSG_EQ (table->LookupReactive (dst).IsValid (), false, "Reactive lookup works");

  // Proactive
  table->AddProactivePath (metric, dst, hop, iface, expire, seqnum);
  NS_TEST_EXPECT_MSG_EQ ((table->LookupProactive () == correct), true, "Proactive lookup works");
  table->DeleteProactivePath (dst);
  NS_TEST_EXPECT_MSG_EQ (table->LookupProactive ().IsValid (), false, "Proactive lookup works");
}

void
HwmpRtableTest::TestAddPath ()
{
  table->AddReactivePath (dst, hop, iface, metric, expire, seqnum);
  table->AddProactivePath (metric, dst, hop, iface, expire, seqnum);
}

void
HwmpRtableTest::TestExpire ()
{
  // this is assumed to be called when path records are already expired
  HwmpRtable::LookupResult correct (hop, iface, metric, seqnum);
  NS_TEST_EXPECT_MSG_EQ ((table->LookupReactiveExpired (dst) == correct), true, "Reactive expiration works");
  NS_TEST_EXPECT_MSG_EQ ((table->LookupProactiveExpired () == correct), true, "Proactive expiration works");

  NS_TEST_EXPECT_MSG_EQ (table->LookupReactive (dst).IsValid (), false, "Reactive expiration works");
  NS_TEST_EXPECT_MSG_EQ (table->LookupProactive ().IsValid (), false, "Proactive expiration works");
}

void
HwmpRtableTest::TestPrecursorAdd ()
{
  for (std::vector<Mac48Address>::const_iterator i = precursors.begin (); i != precursors.end (); i++)
    {
      table->AddPrecursor (dst, iface, *i, Seconds (100));
      // Check that duplicates are filtered
      table->AddPrecursor (dst, iface, *i, Seconds (100));
    }
}

void
HwmpRtableTest::TestPrecursorFind ()
{
  HwmpRtable::PrecursorList precursorList = table->GetPrecursors (dst);
  NS_TEST_EXPECT_MSG_EQ (precursors.size (), precursorList.size (), "Precursors size works");
  for (unsigned i = 0; i < precursors.size (); i++)
    {
      NS_TEST_EXPECT_MSG_EQ (precursorList[i].first, iface, "Precursors lookup works");
      NS_TEST_EXPECT_MSG_EQ (precursorList[i].second, precursors[i], "Precursors lookup works");
    }
}

void
HwmpRtableTest::DoRun ()
{
  table = CreateObject<HwmpRtable> ();

  Simulator::Schedule (Seconds (0), &HwmpRtableTest::TestLookup, this);
  Simulator::Schedule (Seconds (1), &HwmpRtableTest::TestAddPath, this);
  Simulator::Schedule (Seconds (2), &HwmpRtableTest::TestPrecursorAdd, this);
  Simulator::Schedule (expire + Seconds (2), &HwmpRtableTest::TestExpire, this);
  Simulator::Schedule (expire + Seconds (3), &HwmpRtableTest::TestPrecursorFind, this);

  Simulator::Run ();
  Simulator::Destroy ();
}
//-----------------------------------------------------------------------------
/// Built-in self test for PeerLinkFrameStart
struct PeerLinkFrameStartTest : public TestCase
{
  PeerLinkFrameStartTest () :
    TestCase ("PeerLinkFrames (open, confirm, close) unit tests")
  {
  }
  virtual void DoRun ();
};

void
PeerLinkFrameStartTest::DoRun ()
{
  {
    PeerLinkOpenStart a;
    PeerLinkOpenStart::PlinkOpenStartFields fields;
    fields.capability = 0;
    fields.meshId = IeMeshId ("qwertyuiop");
    a.SetPlinkOpenStart (fields);
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader (a);
    PeerLinkOpenStart b;
    packet->RemoveHeader (b);
    NS_TEST_EXPECT_MSG_EQ (a, b, "PEER_LINK_OPEN works");
  }
  {
    PeerLinkConfirmStart a;
    PeerLinkConfirmStart::PlinkConfirmStartFields fields;
    fields.capability = 0;
    fields.aid = 1234;
    a.SetPlinkConfirmStart (fields);
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader (a);
    PeerLinkConfirmStart b;
    packet->RemoveHeader (b);
    NS_TEST_EXPECT_MSG_EQ (a, b, "PEER_LINK_CONFIRM works");
  }
  {
    PeerLinkCloseStart a;
    PeerLinkCloseStart::PlinkCloseStartFields fields;
    fields.meshId = IeMeshId ("qqq");
    a.SetPlinkCloseStart (fields);
    Ptr<Packet> packet = Create<Packet> ();
    packet->AddHeader (a);
    PeerLinkCloseStart b;
    packet->RemoveHeader (b);
    NS_TEST_EXPECT_MSG_EQ (a, b, "PEER_LINK_CLOSE works");
  }
}
/**
 * \ingroup mesh-test
 * \ingroup tests
 *
 * \brief Dot11s Test Suite
 */
class Dot11sTestSuite : public TestSuite
{
public:
  Dot11sTestSuite ();
};

Dot11sTestSuite::Dot11sTestSuite ()
  : TestSuite ("devices-mesh-dot11s", UNIT)
{
  AddTestCase (new MeshHeaderTest, TestCase::QUICK);
  AddTestCase (new HwmpRtableTest, TestCase::QUICK);
  AddTestCase (new PeerLinkFrameStartTest, TestCase::QUICK);
}

static Dot11sTestSuite g_dot11sTestSuite; ///< the test suite
