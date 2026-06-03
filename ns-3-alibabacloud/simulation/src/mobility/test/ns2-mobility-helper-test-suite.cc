/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
 *               2009,2010 Contributors
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Contributors: Thomas Waldecker <twaldecker@rocketmail.com>
 *               Martín Giachino <martin.giachino@gmail.com>
 *
 * Brief description: Implementation of a ns2 movement trace file reader.
 *
 * This implementation is based on the ns2 movement documentation of ns2
 * as described in http://www.isi.edu/nsnam/ns/doc/node174.html
 *
 * Valid trace files use the following ns2 statements:
 *
 * $node set X_ x1
 * $node set Y_ y1
 * $node set Z_ z1
 * $ns at $time $node setdest x2 y2 speed
 * $ns at $time $node set X_ x1
 * $ns at $time $node set Y_ Y1
 * $ns at $time $node set Z_ Z1
 *
 */

#include <algorithm>
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/constant-velocity-mobility-model.h"
#include "ns3/test.h"
#include "ns3/node-container.h"
#include "ns3/names.h"
#include "ns3/config.h"
#include "ns3/ns2-mobility-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ns2-mobility-helper-test-suite");

// -----------------------------------------------------------------------------
// Testing
// -----------------------------------------------------------------------------
bool AreVectorsEqual (Vector const & actual, Vector const & limit, double tol)
{
  if (actual.x > limit.x + tol || actual.x < limit.x - tol)
    {
      return false;
    }
  if (actual.y > limit.y + tol || actual.y < limit.y - tol)
    {
      return false;
    }
  if (actual.z > limit.z + tol || actual.z < limit.z - tol)
    {
      return false;
    }
  return true;
}

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief Every test case is supposed to:
 *  1. Generate short mobility trace file
 *  2. Read it back using Ns2MobilityHelper
 *  3. Check initial node positions and speeds.
 *  4. Run simulation listening for all CourseChange events and compare actual mobility with the reference
 */
class Ns2MobilityHelperTest : public TestCase
{
public:
  /// Single record in mobility reference
  struct ReferencePoint
  {
    std::string node;       ///< node ID as string, e.g. "1"
    Time time;              ///< timestamp
    Vector pos;             ///< reference position
    Vector vel;             ///< reference velocity

    /**
     * Constructor
     *
     * \param id reference ID
     * \param t time
     * \param p position
     * \param v velocity
     */
    ReferencePoint (std::string const & id, Time t, Vector const & p, Vector const & v)
      : node (id),
        time (t),
        pos (p),
        vel (v)
    {
    }
    /**
     * Less-than operator - used to sort by timestamp
     * \param o object to compare to
     * \returns true if the timestamp of the 1nd operand is less than the other one's
     */
    bool operator< (ReferencePoint const & o) const
    {
      return (time < o.time);
    }
  };
  /**
   * Create new test case. To make it useful SetTrace () and AddReferencePoint () must be called
   *
   * \param name        Short description
   * \param timeLimit   Test time limit
   * \param nodes       Number of nodes used in the test trace, 1 by default
   */
  Ns2MobilityHelperTest (std::string const & name, Time timeLimit, uint32_t nodes = 1)
    : TestCase (name),
      m_timeLimit (timeLimit),
      m_nodeCount (nodes),
      m_nextRefPoint (0)
  {
  }
  /// Empty
  virtual ~Ns2MobilityHelperTest ()
  {
  }
  /**
   * Set NS-2 trace to read as single large string (don't forget to add \\n and quote \"'s)
   *  \param trace the mobility trace
   */
  void SetTrace (std::string const & trace)
  {
    m_trace = trace;
  }
  /**
   * Add next reference point
   * \param r reference point to add
   */
  void AddReferencePoint (ReferencePoint const & r)
  {
    m_reference.push_back (r);
  }
  /**
   * Add next reference point
   * \param id reference point id
   * \param sec reference point ime (in seconds)
   * \param p reference point position
   * \param v reference point velocity
   */
  void AddReferencePoint (const char * id, double sec, Vector const & p, Vector const & v)
  {
    AddReferencePoint (ReferencePoint (id, Seconds (sec), p, v));
  }

private:
  /// Test time limit
  Time m_timeLimit;
  /// Number of nodes used in the test
  uint32_t m_nodeCount;
  /// Trace as string
  std::string m_trace;
  /// Reference mobility
  std::vector<ReferencePoint> m_reference;
  /// Next reference point to be checked
  size_t m_nextRefPoint;
  /// TMP trace file name
  std::string m_traceFile;

private:
  /**
   * Dump NS-2 trace to tmp file
   * \return true on error.
   */
  bool WriteTrace ()
  {
    m_traceFile = CreateTempDirFilename ("Ns2MobilityHelperTest.tcl");
    std::ofstream of (m_traceFile.c_str ());
    NS_TEST_ASSERT_MSG_EQ_RETURNS_BOOL (of.is_open (), true, "Need to write tmp. file");
    of << m_trace;
    of.close ();
    return false; // no errors
  }
  /// Create and name nodes
  void CreateNodes ()
  {
    NodeContainer nodes;
    nodes.Create (m_nodeCount);
    for (uint32_t i = 0; i < m_nodeCount; ++i)
      {
        std::ostringstream os;
        os << i;
        Names::Add (os.str (), nodes.Get (i));
      }
  }
  /**
   * Check that all initial positions are correct
   * \return true on error.
   */
  bool CheckInitialPositions ()
  {
    std::stable_sort (m_reference.begin (), m_reference.end ());
    while (m_nextRefPoint < m_reference.size () && m_reference[m_nextRefPoint].time == Seconds (0))
      {
        ReferencePoint const & rp = m_reference[m_nextRefPoint];
        Ptr<Node> node = Names::Find<Node> (rp.node);
        NS_TEST_ASSERT_MSG_NE_RETURNS_BOOL (node, 0, "Can't find node with id " << rp.node);
        Ptr<MobilityModel> mob = node->GetObject<MobilityModel> ();
        NS_TEST_ASSERT_MSG_NE_RETURNS_BOOL (mob, 0, "Can't find mobility for node " << rp.node);

        double tol = 0.001;
        NS_TEST_EXPECT_MSG_EQ (AreVectorsEqual (mob->GetPosition (), rp.pos, tol), true, "Initial position mismatch for node " << rp.node);
        NS_TEST_EXPECT_MSG_EQ (AreVectorsEqual (mob->GetVelocity (), rp.vel, tol), true, "Initial velocity mismatch for node " << rp.node);

        m_nextRefPoint++;
      }
    return IsStatusFailure ();
  }
  /**
   * Listen for course change events
   * \param context event context
   * \param mobility a pointer to the mobility model
   */
  void CourseChange (std::string context, Ptr<const MobilityModel> mobility)
  {
    Time time = Simulator::Now ();
    Ptr<Node> node = mobility->GetObject<Node> ();
    NS_ASSERT (node);
    std::string id = Names::FindName (node);
    NS_ASSERT (!id.empty ());
    Vector pos = mobility->GetPosition ();
    Vector vel = mobility->GetVelocity ();

    NS_TEST_EXPECT_MSG_LT (m_nextRefPoint, m_reference.size (), "Not enough reference points");
    if (m_nextRefPoint >= m_reference.size ())
      {
        return;
      }

    ReferencePoint const & ref = m_reference [m_nextRefPoint++];
    NS_TEST_EXPECT_MSG_EQ (time, ref.time, "Time mismatch");
    NS_TEST_EXPECT_MSG_EQ (id, ref.node, "Node ID mismatch at time " << time.GetSeconds () << " s");

    double tol = 0.001;
    NS_TEST_EXPECT_MSG_EQ (AreVectorsEqual (pos, ref.pos, tol), true, "Position mismatch at time " << time.GetSeconds () << " s for node " << id);
    NS_TEST_EXPECT_MSG_EQ (AreVectorsEqual (vel, ref.vel, tol), true, "Velocity mismatch at time " << time.GetSeconds () << " s for node " << id);
  }

  void DoSetup ()
  {
    CreateNodes ();
  }

  void DoTeardown ()
  {
    Names::Clear ();
    Simulator::Destroy ();
  }

  /// Go
  void DoRun ()
  {
    NS_TEST_ASSERT_MSG_EQ (m_trace.empty (), false, "Need trace");
    NS_TEST_ASSERT_MSG_EQ (m_reference.empty (), false, "Need reference");

    if (WriteTrace ())
      {
        return;
      }
    Ns2MobilityHelper mobility (m_traceFile);
    mobility.Install ();
    if (CheckInitialPositions ())
      {
        return;
      }
    Config::Connect ("/NodeList/*/$ns3::MobilityModel/CourseChange",
                     MakeCallback (&Ns2MobilityHelperTest::CourseChange, this));
    Simulator::Stop (m_timeLimit);
    Simulator::Run ();
  }
};

/**
 * \ingroup mobility-test
 * \ingroup tests
 *
 * \brief The test suite
 */
class Ns2MobilityHelperTestSuite : public TestSuite
{
public:
  Ns2MobilityHelperTestSuite () : TestSuite ("mobility-ns2-trace-helper", UNIT)
  {
    SetDataDir (NS_TEST_SOURCEDIR);

    // to be used as temporary variable for test cases.
    // Note that test suite takes care of deleting all test cases.
    Ns2MobilityHelperTest * t (0);

    // Initial position
    t = new Ns2MobilityHelperTest ("initial position", Seconds (1));
    t->SetTrace ("$node_(0) set X_ 1.0\n"
                 "$node_(0) set Y_ 2.0\n"
                 "$node_(0) set Z_ 3.0\n"
                 );
    t->AddReferencePoint ("0", 0, Vector (1, 2, 3), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Check parsing comments, empty lines and no EOF at the end of file
    t = new Ns2MobilityHelperTest ("comments", Seconds (1));
    t->SetTrace ("# comment\n"
                 "\n\n" // empty lines
                 "$node_(0) set X_ 1.0 # comment \n"
                 "$node_(0) set Y_ 2.0 ### \n"
                 "$node_(0) set Z_ 3.0 # $node_(0) set Z_ 3.0\n"
                 "#$node_(0) set Z_ 100 #"
                 );
    t->AddReferencePoint ("0", 0, Vector (1, 2, 3), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Simple setdest. Arguments are interpreted as x, y, speed by default
    t = new Ns2MobilityHelperTest ("simple setdest", Seconds (10));
    t->SetTrace ("$ns_ at 1.0 \"$node_(0) setdest 25 0 5\"");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (0, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 1, Vector (0, 0, 0), Vector (5, 0, 0));
    t->AddReferencePoint ("0", 6, Vector (25, 0, 0), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Several set and setdest. Arguments are interpreted as x, y, speed by default
    t = new Ns2MobilityHelperTest ("square setdest", Seconds (6));
    t->SetTrace ("$node_(0) set X_ 0.0\n"
                 "$node_(0) set Y_ 0.0\n"
                 "$ns_ at 1.0 \"$node_(0) setdest 5  0  5\"\n"
                 "$ns_ at 2.0 \"$node_(0) setdest 5  5  5\"\n"
                 "$ns_ at 3.0 \"$node_(0) setdest 0  5  5\"\n"
                 "$ns_ at 4.0 \"$node_(0) setdest 0  0  5\"\n"
                 );
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (0, 0, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 1, Vector (0, 0, 0), Vector (5,  0, 0));
    t->AddReferencePoint ("0", 2, Vector (5, 0, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 2, Vector (5, 0, 0), Vector (0,  5, 0));
    t->AddReferencePoint ("0", 3, Vector (5, 5, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 3, Vector (5, 5, 0), Vector (-5, 0, 0));
    t->AddReferencePoint ("0", 4, Vector (0, 5, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 4, Vector (0, 5, 0), Vector (0, -5, 0));
    t->AddReferencePoint ("0", 5, Vector (0, 0, 0), Vector (0,  0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Copy of previous test case but with the initial positions at
    // the end of the trace rather than at the beginning.
    //
    // Several set and setdest. Arguments are interpreted as x, y, speed by default
    t = new Ns2MobilityHelperTest ("square setdest (initial positions at end)", Seconds (6));
    t->SetTrace ("$ns_ at 1.0 \"$node_(0) setdest 15  10  5\"\n"
                 "$ns_ at 2.0 \"$node_(0) setdest 15  15  5\"\n"
                 "$ns_ at 3.0 \"$node_(0) setdest 10  15  5\"\n"
                 "$ns_ at 4.0 \"$node_(0) setdest 10  10  5\"\n"
                 "$node_(0) set X_ 10.0\n"
                 "$node_(0) set Y_ 10.0\n"
                 );
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (10, 10, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 1, Vector (10, 10, 0), Vector (5,  0, 0));
    t->AddReferencePoint ("0", 2, Vector (15, 10, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 2, Vector (15, 10, 0), Vector (0,  5, 0));
    t->AddReferencePoint ("0", 3, Vector (15, 15, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 3, Vector (15, 15, 0), Vector (-5, 0, 0));
    t->AddReferencePoint ("0", 4, Vector (10, 15, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 4, Vector (10, 15, 0), Vector (0, -5, 0));
    t->AddReferencePoint ("0", 5, Vector (10, 10, 0), Vector (0,  0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Scheduled set position
    t = new Ns2MobilityHelperTest ("scheduled set position", Seconds (2));
    t->SetTrace ("$ns_ at 1.0 \"$node_(0) set X_ 10\"\n"
                 "$ns_ at 1.0 \"$node_(0) set Z_ 10\"\n"
                 "$ns_ at 1.0 \"$node_(0) set Y_ 10\"");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 1, Vector (10, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 1, Vector (10, 0, 10), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 1, Vector (10, 10, 10), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Malformed lines
    t = new Ns2MobilityHelperTest ("malformed lines", Seconds (2));
    t->SetTrace ("$node() set X_ 1 # node id is not present\n"
                 "$node # incoplete line\"\n"
                 "$node this line is not correct\n"
                 "$node_(0) set X_ 1 # line OK \n"
                 "$node_(0) set Y_ 2 # line OK \n"
                 "$node_(0) set Z_ 3 # line OK \n"
                 "$ns_ at  \"$node_(0) setdest 4 4 4\" # time not present\n"
                 "$ns_ at 1 \"$node_(0) setdest 2 2 1   \" # line OK \n");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (1, 2, 3), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 1, Vector (1, 2, 3), Vector (1, 0, 0));
    t->AddReferencePoint ("0", 2, Vector (2, 2, 3), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Non possible values
    t = new Ns2MobilityHelperTest ("non possible values", Seconds (2));
    t->SetTrace ("$node_(0) set X_ 1 # line OK \n"
                 "$node_(0) set Y_ 2 # line OK \n"
                 "$node_(0) set Z_ 3 # line OK \n"
                 "$node_(-22) set Y_ 3 # node id not correct\n"
                 "$node_(3.3) set Y_ 1111 # node id not correct\n"
                 "$ns_ at sss \"$node_(0) setdest 5 5 5\" # time is not a number\n"
                 "$ns_ at 1 \"$node_(0) setdest 2 2 1\" # line OK \n"
                 "$ns_ at 1 \"$node_(0) setdest 2 2 -1\" # negative speed is not correct\n"
                 "$ns_ at 1 \"$node_(0) setdest 2 2 sdfs\"    # speed is not a number\n"
                 "$ns_ at 1 \"$node_(0) setdest 2 2 s232dfs\" # speed is not a number\n"
                 "$ns_ at 1 \"$node_(0) setdest 233 2.. s232dfs\"   # more than one non numbers\n"
                 "$ns_ at -12 \"$node_(0) setdest 11 22 33\" # time should not be negative\n");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (1, 2, 3), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 1, Vector (1, 2, 3), Vector (1, 0, 0));
    t->AddReferencePoint ("0", 2, Vector (2, 2, 3), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);

    // More than one node
    t = new Ns2MobilityHelperTest ("few nodes, combinations of set and setdest", Seconds (10), 3);
    t->SetTrace ("$node_(0) set X_ 1.0\n"
                 "$node_(0) set Y_ 2.0\n"
                 "$node_(0) set Z_ 3.0\n"
                 "$ns_ at 1.0 \"$node_(1) setdest 25 0 5\"\n"
                 "$node_(2) set X_ 0.0\n"
                 "$node_(2) set Y_ 0.0\n"
                 "$ns_ at 1.0 \"$node_(2) setdest 5  0  5\"\n"
                 "$ns_ at 2.0 \"$node_(2) setdest 5  5  5\"\n"
                 "$ns_ at 3.0 \"$node_(2) setdest 0  5  5\"\n"
                 "$ns_ at 4.0 \"$node_(2) setdest 0  0  5\"\n");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (1, 2, 3), Vector (0, 0, 0));
    t->AddReferencePoint ("1", 0, Vector (0, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("1", 1, Vector (0, 0, 0), Vector (5, 0, 0));
    t->AddReferencePoint ("1", 6, Vector (25, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("2", 0, Vector (0, 0, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("2", 1, Vector (0, 0, 0), Vector (5,  0, 0));
    t->AddReferencePoint ("2", 2, Vector (5, 0, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("2", 2, Vector (5, 0, 0), Vector (0,  5, 0));
    t->AddReferencePoint ("2", 3, Vector (5, 5, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("2", 3, Vector (5, 5, 0), Vector (-5, 0, 0));
    t->AddReferencePoint ("2", 4, Vector (0, 5, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("2", 4, Vector (0, 5, 0), Vector (0, -5, 0));
    t->AddReferencePoint ("2", 5, Vector (0, 0, 0), Vector (0,  0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Test for Speed == 0, that acts as stop the node.
    t = new Ns2MobilityHelperTest ("setdest with speed cero", Seconds (10));
    t->SetTrace ("$ns_ at 1.0 \"$node_(0) setdest 25 0 5\"\n"
                 "$ns_ at 7.0 \"$node_(0) setdest 11  22  0\"\n");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (0, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 1, Vector (0, 0, 0), Vector (5, 0, 0));
    t->AddReferencePoint ("0", 6, Vector (25, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 7, Vector (25, 0, 0), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);


    // Test negative positions
    t = new Ns2MobilityHelperTest ("test negative positions", Seconds (10));
    t->SetTrace ("$node_(0) set X_ -1.0\n"
                 "$node_(0) set Y_ 0\n"
                 "$ns_ at 1.0 \"$node_(0) setdest 0 0 1\"\n"
                 "$ns_ at 2.0 \"$node_(0) setdest 0  -1  1\"\n");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (-1, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 1, Vector (-1, 0, 0), Vector (1, 0, 0));
    t->AddReferencePoint ("0", 2, Vector (0, 0, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 2, Vector (0, 0, 0), Vector (0, -1, 0));
    t->AddReferencePoint ("0", 3, Vector (0, -1, 0), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);

    // Sqare setdest with values in the form 1.0e+2
    t = new Ns2MobilityHelperTest ("Foalt numbers in 1.0e+2 format", Seconds (6));
    t->SetTrace ("$node_(0) set X_ 0.0\n"
                 "$node_(0) set Y_ 0.0\n"
                 "$ns_ at 1.0 \"$node_(0) setdest 1.0e+2  0       1.0e+2\"\n"
                 "$ns_ at 2.0 \"$node_(0) setdest 1.0e+2  1.0e+2  1.0e+2\"\n"
                 "$ns_ at 3.0 \"$node_(0) setdest 0       1.0e+2  1.0e+2\"\n"
                 "$ns_ at 4.0 \"$node_(0) setdest 0       0       1.0e+2\"\n");
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (0, 0, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 1, Vector (0, 0, 0), Vector (100,  0, 0));
    t->AddReferencePoint ("0", 2, Vector (100, 0, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 2, Vector (100, 0, 0), Vector (0,  100, 0));
    t->AddReferencePoint ("0", 3, Vector (100, 100, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 3, Vector (100, 100, 0), Vector (-100, 0, 0));
    t->AddReferencePoint ("0", 4, Vector (0, 100, 0), Vector (0, 0, 0));
    t->AddReferencePoint ("0", 4, Vector (0, 100, 0), Vector (0, -100, 0));
    t->AddReferencePoint ("0", 5, Vector (0, 0, 0), Vector (0,  0, 0));
    AddTestCase (t, TestCase::QUICK);
    t = new Ns2MobilityHelperTest ("Bug 1219 testcase", Seconds (16));
    t->SetTrace ("$node_(0) set X_ 0.0\n"
                 "$node_(0) set Y_ 0.0\n"
                 "$ns_ at 1.0 \"$node_(0) setdest 0  10       1\"\n"
                 "$ns_ at 6.0 \"$node_(0) setdest 0  -10       1\"\n"
                 );
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (0, 0, 0), Vector (0,  0, 0));
    t->AddReferencePoint ("0", 1, Vector (0, 0, 0), Vector (0,  1, 0));
    t->AddReferencePoint ("0", 6, Vector (0, 5, 0), Vector (0,  -1, 0));
    t->AddReferencePoint ("0", 16, Vector (0, -10, 0), Vector (0, 0, 0));
    AddTestCase (t, TestCase::QUICK);
    t = new Ns2MobilityHelperTest ("Bug 1059 testcase", Seconds (16));
    t->SetTrace ("$node_(0) set X_ 10.0\r\n"
                 "$node_(0) set Y_ 0.0\r\n"
                 );
    //                     id  t  position         velocity
    t->AddReferencePoint ("0", 0, Vector (10, 0, 0), Vector (0,  0, 0));
    AddTestCase (t, TestCase::QUICK);
    t = new Ns2MobilityHelperTest ("Bug 1301 testcase", Seconds (16));
    t->SetTrace ("$node_(0) set X_ 10.0\n"
                 "$node_(0) set Y_ 0.0\n"
                 "$ns_ at 1.0 \"$node_(0) setdest 10  0       1\"\n"
                 );
    //                     id  t  position         velocity
    // Moving to the current position must change nothing. No NaN
    // speed must be.
    t->AddReferencePoint ("0", 0, Vector (10, 0, 0), Vector (0,  0, 0));
    AddTestCase (t, TestCase::QUICK);

    t = new Ns2MobilityHelperTest ("Bug 1316 testcase", Seconds (1000));
    t->SetTrace ("$node_(0) set X_ 350.00000000000000\n"
                 "$node_(0) set Y_ 50.00000000000000\n"
                 "$ns_ at 50.00000000000000  \"$node_(0) setdest 400.00000000000000 50.00000000000000 1.00000000000000\"\n"
                 "$ns_ at 150.00000000000000 \"$node_(0) setdest 400.00000000000000 150.00000000000000 4.00000000000000\"\n"
                 "$ns_ at 300.00000000000000 \"$node_(0) setdest 250.00000000000000 150.00000000000000 3.00000000000000\"\n"
                 "$ns_ at 350.00000000000000 \"$node_(0) setdest 250.00000000000000 50.00000000000000 1.00000000000000\"\n"
                 "$ns_ at 600.00000000000000 \"$node_(0) setdest 250.00000000000000 1050.00000000000000 2.00000000000000\"\n"
                 "$ns_ at 900.00000000000000 \"$node_(0) setdest 300.00000000000000 650.00000000000000 2.50000000000000\"\n"
                 );
    t->AddReferencePoint ("0", 0.000, Vector (350.000, 50.000, 0.000), Vector (0.000, 0.000, 0.000));
    t->AddReferencePoint ("0", 50.000, Vector (350.000, 50.000, 0.000), Vector (1.000, 0.000, 0.000));
    t->AddReferencePoint ("0", 100.000, Vector (400.000, 50.000, 0.000), Vector (0.000, 0.000, 0.000));
    t->AddReferencePoint ("0", 150.000, Vector (400.000, 50.000, 0.000), Vector (0.000, 4.000, 0.000));
    t->AddReferencePoint ("0", 175.000, Vector (400.000, 150.000, 0.000), Vector (0.000, 0.000, 0.000));
    t->AddReferencePoint ("0", 300.000, Vector (400.000, 150.000, 0.000), Vector (-3.000, 0.000, 0.000));
    t->AddReferencePoint ("0", 350.000, Vector (250.000, 150.000, 0.000), Vector (0.000, 0.000, 0.000));
    t->AddReferencePoint ("0", 350.000, Vector (250.000, 150.000, 0.000), Vector (0.000, -1.000, 0.000));
    t->AddReferencePoint ("0", 450.000, Vector (250.000,  50.000, 0.000), Vector (0.000, 0.000, 0.000));
    t->AddReferencePoint ("0", 600.000, Vector (250.000,  50.000, 0.000), Vector (0.000, 2.000, 0.000));
    t->AddReferencePoint ("0", 900.000, Vector (250.000,  650.000, 0.000), Vector (2.500, 0.000, 0.000));
    t->AddReferencePoint ("0", 920.000, Vector (300.000,  650.000, 0.000), Vector (0.000, 0.000, 0.000));
    AddTestCase (t, TestCase::QUICK);

  }
} g_ns2TransmobilityHelperTestSuite; ///< the test suite
