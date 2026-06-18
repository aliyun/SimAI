/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2008-2010 INESC Porto
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt> <gjcarneiro@gmail.com>
//

#include "ns3/sequence-number.h"
#include "ns3/test.h"
#include "ns3/object.h"
#include "ns3/traced-value.h"
#include "ns3/trace-source-accessor.h"

using namespace ns3;

namespace {


/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Test object using sequence numbers
 *
 * \note Class internal to sequence-number-test-suite.cc
 */
class SequenceNumberTestObj : public Object
{
  /// Test traced sequence number.
  TracedValue<SequenceNumber32> m_testTracedSequenceNumber;

public:

  SequenceNumberTestObj () 
  {
    m_testTracedSequenceNumber = SequenceNumber32 (0);
  }

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void)
  {
    static TypeId tid = TypeId ("ns3::SequenceNumberTestObj")
      .SetParent<Object> ()
      .AddTraceSource ("TestTracedSequenceNumber",
                       "A traceable sequence number",
                       MakeTraceSourceAccessor (&SequenceNumberTestObj::m_testTracedSequenceNumber),
                       "ns3::SequenceNumber32TracedValueCallback")
      .AddConstructor<SequenceNumberTestObj> ()
    ;
    return tid;
  }

  TypeId GetInstanceTypeId (void) const
  {
    return GetTypeId ();
  }

  /// Increment the sequence number.
  void IncSequenceNumber ()
  {
    m_testTracedSequenceNumber += 1;
  }
};

}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Sequence Number Unit Test
 */
class SequenceNumberTestCase : public TestCase
{
  SequenceNumber32 m_oldval;  //!< Old value
  SequenceNumber32 m_newval;  //!< New value

  /**
   * Sequence number tracker
   * \param oldval Old value
   * \param newval New value
   */
  void SequenceNumberTracer (SequenceNumber32 oldval, SequenceNumber32 newval);

public:

  SequenceNumberTestCase ();
  virtual ~SequenceNumberTestCase ();
  virtual void DoRun (void);
};

SequenceNumberTestCase::SequenceNumberTestCase ()
  : TestCase ("Sequence number test case")
{
  m_oldval = 0;
  m_newval = 0;
}

SequenceNumberTestCase::~SequenceNumberTestCase ()
{
}

void
SequenceNumberTestCase::SequenceNumberTracer (SequenceNumber32 oldval, SequenceNumber32 newval)
{
  m_oldval = oldval;
  m_newval = newval;
}

void SequenceNumberTestCase::DoRun (void)
{
#define SEQ_TEST_ASSERT_EQUAL(a,b) NS_TEST_ASSERT_MSG_EQ (a,b, "foo")
#define SEQ_TEST_ASSERT(a) NS_TEST_ASSERT_MSG_EQ (bool(a), true, "foo")

  {
    SequenceNumber32 num1 (3), num2 (5);
    uint32_t value;

    value = (num1 + num2).GetValue ();
    SEQ_TEST_ASSERT_EQUAL (value, 8);

    num1 += num2.GetValue ();
    SEQ_TEST_ASSERT_EQUAL (num1, SequenceNumber32 (8));

    ++num1;
    SEQ_TEST_ASSERT_EQUAL (num1, SequenceNumber32 (9));

    --num1;
    SEQ_TEST_ASSERT_EQUAL (num1, SequenceNumber32 (8));

    num1++;
    SEQ_TEST_ASSERT_EQUAL (num1, SequenceNumber32 (9));

    num1--;
    SEQ_TEST_ASSERT_EQUAL (num1, SequenceNumber32 (8));

  }

  {
    SequenceNumber16 num1 (60900), num2 (5), num3 (10000);

    SEQ_TEST_ASSERT (num1 == num1);

    SEQ_TEST_ASSERT (num2 != num1);

    SEQ_TEST_ASSERT (num3 > num2);
    SEQ_TEST_ASSERT (num3 >= num2);
    SEQ_TEST_ASSERT (num1 < num3);
    SEQ_TEST_ASSERT (num1 <= num3);

    SEQ_TEST_ASSERT (num1 < num2);
    SEQ_TEST_ASSERT (num1 <= num2);
    SEQ_TEST_ASSERT (num2 > num1);
    SEQ_TEST_ASSERT (num2 >= num1);

    SEQ_TEST_ASSERT (num1+num2 > num1);
    SEQ_TEST_ASSERT (num1+num2 >= num1);
    SEQ_TEST_ASSERT (num1 < num1+num2);
    SEQ_TEST_ASSERT (num1 <= num1+num2);

    SEQ_TEST_ASSERT (num1 < num1+num3);
    SEQ_TEST_ASSERT (num1 <= num1+num3);
    SEQ_TEST_ASSERT (num1+num3 > num1);
    SEQ_TEST_ASSERT (num1+num3 >= num1);
  }

  {
    SEQ_TEST_ASSERT_EQUAL ((SequenceNumber16 (1000) + SequenceNumber16 (6000)) - SequenceNumber16 (1000), 6000);
    SEQ_TEST_ASSERT_EQUAL ((SequenceNumber16 (60000) + SequenceNumber16 (6000)) - SequenceNumber16 (60000), 6000);
    SEQ_TEST_ASSERT_EQUAL (SequenceNumber16 (1000) - SequenceNumber16 (6000), -5000);
    SEQ_TEST_ASSERT_EQUAL ((SequenceNumber16 (60000) + SequenceNumber16 (1000)) - SequenceNumber16 (65000), -4000);
  }

  {
    SequenceNumber32 num1 (3);

    SEQ_TEST_ASSERT_EQUAL (num1 + 10, SequenceNumber32 (13));
    num1 += -1;
    SEQ_TEST_ASSERT_EQUAL (num1, SequenceNumber32 (2));

    SEQ_TEST_ASSERT_EQUAL (num1 - (num1 - 100), 100);
  }

  {
    Ptr<SequenceNumberTestObj> obj = CreateObject<SequenceNumberTestObj> ();
    obj->TraceConnectWithoutContext ("TestTracedSequenceNumber", MakeCallback (&SequenceNumberTestCase::SequenceNumberTracer, this));
    obj->IncSequenceNumber ();
    SEQ_TEST_ASSERT_EQUAL (m_oldval, SequenceNumber32 (0));
    SEQ_TEST_ASSERT_EQUAL (m_newval, SequenceNumber32 (1));
    obj->Dispose ();
  }

}

/**
 * \ingroup network-test
 * \ingroup tests
 *
 * \brief Sequence Number TestSuite
 */
class SequenceNumberTestSuite : public TestSuite
{
public:
  SequenceNumberTestSuite ()
    : TestSuite ("sequence-number", UNIT) 
  {
    AddTestCase (new SequenceNumberTestCase (), TestCase::QUICK);
  }
};

static SequenceNumberTestSuite g_seqNumTests; //!< Static variable for test initialization
