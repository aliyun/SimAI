/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Yufei Cheng
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
 * Author: Yufei Cheng   <yfcheng@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture),
 * NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI),
 * US Department of Defense (DoD), and ITTC at The University of Kansas.
 */

#include <vector>
#include "ns3/ptr.h"
#include "ns3/boolean.h"
#include "ns3/test.h"
#include "ns3/ipv4-route.h"
#include "ns3/mesh-helper.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/ipv4-address-helper.h"

#include "ns3/dsr-fs-header.h"
#include "ns3/dsr-option-header.h"
#include "ns3/dsr-rreq-table.h"
#include "ns3/dsr-rcache.h"
#include "ns3/dsr-rsendbuff.h"
#include "ns3/dsr-main-helper.h"
#include "ns3/dsr-helper.h"

using namespace ns3;
using namespace dsr;

// -----------------------------------------------------------------------------
/**
 * \ingroup dsr
 * \defgroup dsr-test DSR routing module tests
 */


/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrFsHeaderTest
 * \brief Unit test for DSR Fixed Size Header
 */
class DsrFsHeaderTest : public TestCase
{
public:
  DsrFsHeaderTest ();
  ~DsrFsHeaderTest ();
  virtual void
  DoRun (void);
};
DsrFsHeaderTest::DsrFsHeaderTest ()
  : TestCase ("DSR Fixed size Header")
{
}
DsrFsHeaderTest::~DsrFsHeaderTest ()
{
}
void
DsrFsHeaderTest::DoRun ()
{
  dsr::DsrRoutingHeader header;
  dsr::DsrOptionRreqHeader rreqHeader;
  header.AddDsrOption (rreqHeader); // has an alignment of 4n+0

  NS_TEST_EXPECT_MSG_EQ (header.GetSerializedSize () % 2, 0, "length of routing header is not a multiple of 4");
  Buffer buf;
  buf.AddAtStart (header.GetSerializedSize ());
  header.Serialize (buf.Begin ());

  const uint8_t* data = buf.PeekData ();
  NS_TEST_EXPECT_MSG_EQ (*(data + 8), rreqHeader.GetType (), "expect the rreqHeader after fixed size header");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrRreqHeaderTest
 * \brief Unit test for RREQ
 */
class DsrRreqHeaderTest : public TestCase
{
public:
  DsrRreqHeaderTest ();
  ~DsrRreqHeaderTest ();
  virtual void
  DoRun (void);
};
DsrRreqHeaderTest::DsrRreqHeaderTest ()
  : TestCase ("DSR RREQ")
{
}
DsrRreqHeaderTest::~DsrRreqHeaderTest ()
{
}
void
DsrRreqHeaderTest::DoRun ()
{
  dsr::DsrOptionRreqHeader h;
  std::vector<Ipv4Address> nodeList;
  nodeList.push_back (Ipv4Address ("1.1.1.0"));
  nodeList.push_back (Ipv4Address ("1.1.1.1"));
  nodeList.push_back (Ipv4Address ("1.1.1.2"));

  h.SetTarget (Ipv4Address ("1.1.1.3"));
  NS_TEST_EXPECT_MSG_EQ (h.GetTarget (), Ipv4Address ("1.1.1.3"), "trivial");
  h.SetNodesAddress (nodeList);
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (0), Ipv4Address ("1.1.1.0"), "trivial");
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (1), Ipv4Address ("1.1.1.1"), "trivial");
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (2), Ipv4Address ("1.1.1.2"), "trivial");
  h.SetId (1);
  NS_TEST_EXPECT_MSG_EQ (h.GetId (), 1, "trivial");

  Ptr<Packet> p = Create<Packet> ();
  dsr::DsrRoutingHeader header;
  header.AddDsrOption (h);
  p->AddHeader (header);
  p->RemoveAtStart (8);
  dsr::DsrOptionRreqHeader h2;
  h2.SetNumberAddress (3);
  uint32_t bytes = p->RemoveHeader (h2);
  NS_TEST_EXPECT_MSG_EQ (bytes, 20, "Total RREP is 20 bytes long");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrRrepHeaderTest
 * \brief Unit test for RREP
 */
class DsrRrepHeaderTest : public TestCase
{
public:
  DsrRrepHeaderTest ();
  ~DsrRrepHeaderTest ();
  virtual void
  DoRun (void);
};
DsrRrepHeaderTest::DsrRrepHeaderTest ()
  : TestCase ("DSR RREP")
{
}
DsrRrepHeaderTest::~DsrRrepHeaderTest ()
{
}
void
DsrRrepHeaderTest::DoRun ()
{
  dsr::DsrOptionRrepHeader h;

  std::vector<Ipv4Address> nodeList;
  nodeList.push_back (Ipv4Address ("1.1.1.0"));
  nodeList.push_back (Ipv4Address ("1.1.1.1"));
  nodeList.push_back (Ipv4Address ("1.1.1.2"));
  h.SetNodesAddress (nodeList);
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (0), Ipv4Address ("1.1.1.0"), "trivial");
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (1), Ipv4Address ("1.1.1.1"), "trivial");
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (2), Ipv4Address ("1.1.1.2"), "trivial");

  Ptr<Packet> p = Create<Packet> ();
  dsr::DsrRoutingHeader header;
  header.AddDsrOption (h);
  p->AddHeader (header);
  p->RemoveAtStart (8);
  dsr::DsrOptionRrepHeader h2;
  h2.SetNumberAddress (3);
  uint32_t bytes = p->RemoveHeader (h2);
  NS_TEST_EXPECT_MSG_EQ (bytes, 16, "Total RREP is 16 bytes long");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrSRHeaderTest
 * \brief Unit test for Source Route
 */
class DsrSRHeaderTest : public TestCase
{
public:
  DsrSRHeaderTest ();
  ~DsrSRHeaderTest ();
  virtual void
  DoRun (void);
};
DsrSRHeaderTest::DsrSRHeaderTest ()
  : TestCase ("DSR Source Route")
{
}
DsrSRHeaderTest::~DsrSRHeaderTest ()
{
}
void
DsrSRHeaderTest::DoRun ()
{
  dsr::DsrOptionSRHeader h;
  std::vector<Ipv4Address> nodeList;
  nodeList.push_back (Ipv4Address ("1.1.1.0"));
  nodeList.push_back (Ipv4Address ("1.1.1.1"));
  nodeList.push_back (Ipv4Address ("1.1.1.2"));
  h.SetNodesAddress (nodeList);
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (0), Ipv4Address ("1.1.1.0"), "trivial");
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (1), Ipv4Address ("1.1.1.1"), "trivial");
  NS_TEST_EXPECT_MSG_EQ (h.GetNodeAddress (2), Ipv4Address ("1.1.1.2"), "trivial");

  h.SetSalvage (1);
  NS_TEST_EXPECT_MSG_EQ (h.GetSalvage (), 1, "trivial");
  h.SetSegmentsLeft (2);
  NS_TEST_EXPECT_MSG_EQ (h.GetSegmentsLeft (), 2, "trivial");

  Ptr<Packet> p = Create<Packet> ();
  dsr::DsrRoutingHeader header;
  header.AddDsrOption (h);
  p->AddHeader (header);
  p->RemoveAtStart (8);
  dsr::DsrOptionSRHeader h2;
  h2.SetNumberAddress (3);
  uint32_t bytes = p->RemoveHeader (h2);
  NS_TEST_EXPECT_MSG_EQ (bytes, 16, "Total RREP is 16 bytes long");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrRerrHeaderTest
 * \brief Unit test for RERR
 */
class DsrRerrHeaderTest : public TestCase
{
public:
  DsrRerrHeaderTest ();
  ~DsrRerrHeaderTest ();
  virtual void
  DoRun (void);
};
DsrRerrHeaderTest::DsrRerrHeaderTest ()
  : TestCase ("DSR RERR")
{
}
DsrRerrHeaderTest::~DsrRerrHeaderTest ()
{
}
void
DsrRerrHeaderTest::DoRun ()
{
  dsr::DsrOptionRerrUnreachHeader h;
  h.SetErrorSrc (Ipv4Address ("1.1.1.0"));
  NS_TEST_EXPECT_MSG_EQ (h.GetErrorSrc (), Ipv4Address ("1.1.1.0"), "trivial");
  h.SetErrorDst (Ipv4Address ("1.1.1.1"));
  NS_TEST_EXPECT_MSG_EQ (h.GetErrorDst (), Ipv4Address ("1.1.1.1"), "trivial");
  h.SetSalvage (1);
  NS_TEST_EXPECT_MSG_EQ (h.GetSalvage (), 1, "trivial");
  h.SetUnreachNode (Ipv4Address ("1.1.1.2"));
  NS_TEST_EXPECT_MSG_EQ (h.GetUnreachNode (), Ipv4Address ("1.1.1.2"), "trivial");

  Ptr<Packet> p = Create<Packet> ();
  dsr::DsrRoutingHeader header;
  header.AddDsrOption (h);
  p->AddHeader (header);
  p->RemoveAtStart (8);
  dsr::DsrOptionRerrUnreachHeader h2;
  uint32_t bytes = p->RemoveHeader (h2);
  NS_TEST_EXPECT_MSG_EQ (bytes, 20, "Total RREP is 20 bytes long");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrAckReqHeaderTest
 * \brief Unit test for ACK-REQ
 */
class DsrAckReqHeaderTest : public TestCase
{
public:
  DsrAckReqHeaderTest ();
  ~DsrAckReqHeaderTest ();
  virtual void
  DoRun (void);
};
DsrAckReqHeaderTest::DsrAckReqHeaderTest ()
  : TestCase ("DSR Ack Req")
{
}
DsrAckReqHeaderTest::~DsrAckReqHeaderTest ()
{
}
void
DsrAckReqHeaderTest::DoRun ()
{
  dsr::DsrOptionAckReqHeader h;

  h.SetAckId (1);
  NS_TEST_EXPECT_MSG_EQ (h.GetAckId (), 1, "trivial");

  Ptr<Packet> p = Create<Packet> ();
  dsr::DsrRoutingHeader header;
  header.AddDsrOption (h);
  p->AddHeader (header);
  p->RemoveAtStart (8);
  p->AddHeader (header);
  dsr::DsrOptionAckReqHeader h2;
  p->RemoveAtStart (8);
  uint32_t bytes = p->RemoveHeader (h2);
  NS_TEST_EXPECT_MSG_EQ (bytes, 4, "Total RREP is 4 bytes long");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrAckHeaderTest
 * \brief Unit test for ACK
 */
class DsrAckHeaderTest : public TestCase
{
public:
  DsrAckHeaderTest ();
  ~DsrAckHeaderTest ();
  virtual void
  DoRun (void);
};
DsrAckHeaderTest::DsrAckHeaderTest ()
  : TestCase ("DSR ACK")
{
}
DsrAckHeaderTest::~DsrAckHeaderTest ()
{
}
void
DsrAckHeaderTest::DoRun ()
{
  dsr::DsrOptionAckHeader h;

  h.SetRealSrc (Ipv4Address ("1.1.1.0"));
  NS_TEST_EXPECT_MSG_EQ (h.GetRealSrc (), Ipv4Address ("1.1.1.0"), "trivial");
  h.SetRealDst (Ipv4Address ("1.1.1.1"));
  NS_TEST_EXPECT_MSG_EQ (h.GetRealDst (), Ipv4Address ("1.1.1.1"), "trivial");
  h.SetAckId (1);
  NS_TEST_EXPECT_MSG_EQ (h.GetAckId (), 1, "trivial");

  Ptr<Packet> p = Create<Packet> ();
  dsr::DsrRoutingHeader header;
  header.AddDsrOption (h);
  p->AddHeader (header);
  p->RemoveAtStart (8);
  p->AddHeader (header);
  dsr::DsrOptionAckHeader h2;
  p->RemoveAtStart (8);
  uint32_t bytes = p->RemoveHeader (h2);
  NS_TEST_EXPECT_MSG_EQ (bytes, 12, "Total RREP is 12 bytes long");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrCacheEntryTest
 * \brief Unit test for DSR route cache entry
 */
class DsrCacheEntryTest : public TestCase
{
public:
  DsrCacheEntryTest ();
  ~DsrCacheEntryTest ();
  virtual void
  DoRun (void);
};
DsrCacheEntryTest::DsrCacheEntryTest ()
  : TestCase ("DSR ACK")
{
}
DsrCacheEntryTest::~DsrCacheEntryTest ()
{
}
void
DsrCacheEntryTest::DoRun ()
{
  Ptr<dsr::DsrRouteCache> rcache = CreateObject<dsr::DsrRouteCache> ();
  std::vector<Ipv4Address> ip;
  ip.push_back (Ipv4Address ("0.0.0.0"));
  ip.push_back (Ipv4Address ("0.0.0.1"));
  Ipv4Address dst = Ipv4Address ("0.0.0.1");
  dsr::DsrRouteCacheEntry entry (ip, dst, Seconds (1));
  NS_TEST_EXPECT_MSG_EQ (entry.GetVector ().size (), 2, "trivial");
  NS_TEST_EXPECT_MSG_EQ (entry.GetDestination (), Ipv4Address ("0.0.0.1"), "trivial");
  NS_TEST_EXPECT_MSG_EQ (entry.GetExpireTime (), Seconds (1), "trivial");

  entry.SetExpireTime (Seconds (3));
  NS_TEST_EXPECT_MSG_EQ (entry.GetExpireTime (), Seconds (3), "trivial");
  entry.SetDestination (Ipv4Address ("1.1.1.1"));
  NS_TEST_EXPECT_MSG_EQ (entry.GetDestination (), Ipv4Address ("1.1.1.1"), "trivial");
  ip.push_back (Ipv4Address ("0.0.0.2"));
  entry.SetVector (ip);
  NS_TEST_EXPECT_MSG_EQ (entry.GetVector ().size (), 3, "trivial");

  NS_TEST_EXPECT_MSG_EQ (rcache->AddRoute (entry), true, "trivial");

  std::vector<Ipv4Address> ip2;
  ip2.push_back (Ipv4Address ("1.1.1.0"));
  ip2.push_back (Ipv4Address ("1.1.1.1"));
  Ipv4Address dst2 = Ipv4Address ("1.1.1.1");
  dsr::DsrRouteCacheEntry entry2 (ip2, dst2, Seconds (2));
  dsr::DsrRouteCacheEntry newEntry;
  NS_TEST_EXPECT_MSG_EQ (rcache->AddRoute (entry2), true, "trivial");
  NS_TEST_EXPECT_MSG_EQ (rcache->LookupRoute (dst2, newEntry), true, "trivial");
  NS_TEST_EXPECT_MSG_EQ (rcache->DeleteRoute (Ipv4Address ("2.2.2.2")), false, "trivial");

  NS_TEST_EXPECT_MSG_EQ (rcache->DeleteRoute (Ipv4Address ("1.1.1.1")), true, "trivial");
  NS_TEST_EXPECT_MSG_EQ (rcache->DeleteRoute (Ipv4Address ("1.1.1.1")), false, "trivial");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrSendBuffTest
 * \brief Unit test for Send Buffer
 */
class DsrSendBuffTest : public TestCase
{
public:
  DsrSendBuffTest ();
  ~DsrSendBuffTest ();
  virtual void
  DoRun (void);
  /// Check size limit function
  void CheckSizeLimit ();
  /// Check timeout function
  void CheckTimeout ();

  dsr::DsrSendBuffer q; ///< send buffer
};
DsrSendBuffTest::DsrSendBuffTest ()
  : TestCase ("DSR SendBuff"),
    q ()
{
}
DsrSendBuffTest::~DsrSendBuffTest ()
{
}
void
DsrSendBuffTest::DoRun ()
{
  q.SetMaxQueueLen (32);
  NS_TEST_EXPECT_MSG_EQ (q.GetMaxQueueLen (), 32, "trivial");
  q.SetSendBufferTimeout (Seconds (10));
  NS_TEST_EXPECT_MSG_EQ (q.GetSendBufferTimeout (), Seconds (10), "trivial");

  Ptr<const Packet> packet = Create<Packet> ();
  Ipv4Address dst1 = Ipv4Address ("0.0.0.1");
  dsr::DsrSendBuffEntry e1 (packet, dst1, Seconds (1));
  q.Enqueue (e1);
  q.Enqueue (e1);
  q.Enqueue (e1);
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("0.0.0.1")), true, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("1.1.1.1")), false, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 1, "trivial");
  q.DropPacketWithDst (Ipv4Address ("0.0.0.1"));
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("0.0.0.1")), false, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 0, "trivial");

  Ipv4Address dst2 = Ipv4Address ("0.0.0.2");
  dsr::DsrSendBuffEntry e2 (packet, dst2, Seconds (1));
  q.Enqueue (e1);
  q.Enqueue (e2);
  Ptr<Packet> packet2 = Create<Packet> ();
  dsr::DsrSendBuffEntry e3 (packet2, dst2, Seconds (1));
  NS_TEST_EXPECT_MSG_EQ (q.Dequeue (Ipv4Address ("0.0.0.3"), e3), false, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.Dequeue (Ipv4Address ("0.0.0.2"), e3), true, "trivial");
  NS_TEST_EXPECT_MSG_EQ (q.Find (Ipv4Address ("0.0.0.2")), false, "trivial");
  q.Enqueue (e2);
  q.Enqueue (e3);
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 2, "trivial");
  Ptr<Packet> packet4 = Create<Packet> ();
  Ipv4Address dst4 = Ipv4Address ("0.0.0.4");
  dsr::DsrSendBuffEntry e4 (packet4, dst4, Seconds (20));
  q.Enqueue (e4);
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 3, "trivial");
  q.DropPacketWithDst (Ipv4Address ("0.0.0.4"));
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 2, "trivial");

  CheckSizeLimit ();

  Simulator::Schedule (q.GetSendBufferTimeout () + Seconds (1), &DsrSendBuffTest::CheckTimeout, this);

  Simulator::Run ();
  Simulator::Destroy ();
}
void
DsrSendBuffTest::CheckSizeLimit ()
{
  Ptr<Packet> packet = Create<Packet> ();
  Ipv4Address dst;
  dsr::DsrSendBuffEntry e1 (packet, dst, Seconds (1));

  for (uint32_t i = 0; i < q.GetMaxQueueLen (); ++i)
    {
      q.Enqueue (e1);
    }
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 3, "trivial");

  for (uint32_t i = 0; i < q.GetMaxQueueLen (); ++i)
    {
      q.Enqueue (e1);
    }
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 3, "trivial");
}
void
DsrSendBuffTest::CheckTimeout ()
{
  NS_TEST_EXPECT_MSG_EQ (q.GetSize (), 0, "Must be empty now");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrRreqTableTest
 * \brief Unit test for DSR routing table entry
 */
class DsrRreqTableTest : public TestCase
{
public:
  DsrRreqTableTest ();
  ~DsrRreqTableTest ();
  virtual void
  DoRun (void);
};
DsrRreqTableTest::DsrRreqTableTest ()
  : TestCase ("DSR RreqTable")
{
}
DsrRreqTableTest::~DsrRreqTableTest ()
{
}
void
DsrRreqTableTest::DoRun ()
{
  dsr::RreqTableEntry rt;

  rt.m_reqNo = 2;
  NS_TEST_EXPECT_MSG_EQ (rt.m_reqNo, 2, "trivial");
}
// -----------------------------------------------------------------------------
/**
 * \ingroup dsr-test
 * \ingroup tests
 *
 * \class DsrTestSuite
 * \brief DSR test suite
 */
class DsrTestSuite : public TestSuite
{
public:
  DsrTestSuite () : TestSuite ("routing-dsr", UNIT)
  {
    AddTestCase (new DsrFsHeaderTest, TestCase::QUICK);
    AddTestCase (new DsrRreqHeaderTest, TestCase::QUICK);
    AddTestCase (new DsrRrepHeaderTest, TestCase::QUICK);
    AddTestCase (new DsrSRHeaderTest, TestCase::QUICK);
    AddTestCase (new DsrRerrHeaderTest, TestCase::QUICK);
    AddTestCase (new DsrAckReqHeaderTest, TestCase::QUICK);
    AddTestCase (new DsrAckHeaderTest, TestCase::QUICK);
    AddTestCase (new DsrCacheEntryTest, TestCase::QUICK);
    AddTestCase (new DsrSendBuffTest, TestCase::QUICK);
  }
} g_dsrTestSuite;
