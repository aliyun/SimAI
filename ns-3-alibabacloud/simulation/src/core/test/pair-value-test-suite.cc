/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Caliola Engineering, LLC.
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
 * Author: Jared Dulmage <jared.dulmage@caliola.com>
 */

#include <ns3/test.h>
#include <ns3/log.h>
#include <ns3/pair.h>
#include <ns3/double.h>
#include <ns3/integer.h>
#include <ns3/string.h>
#include <ns3/ptr.h>
#include <ns3/object.h>
#include <ns3/type-id.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <utility>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("PairTestSuite");

/**
 * \file
 * \ingroup pair-tests
 * Pairs tests.
 */

/**
 * \ingroup core-tests
 * \defgroup pair-tests Pairs tests
 */

/**
 * \ingroup pair-tests
 * 
 * Object holding pairs of values.
 */
class PairObject : public Object
{
public:
  PairObject ();
  virtual ~PairObject ();

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static
  TypeId GetTypeId ();

  friend std::ostream &operator <<(std::ostream &os, const PairObject &obj);

private:
  std::pair <std::string, std::string> m_stringPair;  //!< A string pair.
  std::pair <double, int> m_doubleIntPair;             //!< A pair of double, int.
};

PairObject::PairObject ()
{

}

PairObject::~PairObject ()
{

}

TypeId
PairObject::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::PairObject")
    .SetParent<Object> ()
    .SetGroupName("Test")
    .AddConstructor<PairObject> ()
    .AddAttribute ("StringPair", "Pair: string, string",
                   PairValue <StringValue, StringValue> (),
                   MakePairAccessor <StringValue, StringValue> (&PairObject::m_stringPair),
                   MakePairChecker<StringValue, StringValue> (MakeStringChecker (), MakeStringChecker ()))
    .AddAttribute ("DoubleIntPair", "Pair: double int",
                   // the container value container differs from the underlying object
                   PairValue <DoubleValue, IntegerValue> (),
                   MakePairAccessor <DoubleValue, IntegerValue> (&PairObject::m_doubleIntPair),
                   MakePairChecker<DoubleValue, IntegerValue> (MakeDoubleChecker<double> (), MakeIntegerChecker<int> ()))
    ;
  return tid;
}

/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] obj The PairObject.
 * \returns The reference to the output stream.
 */
std::ostream &
operator << (std::ostream &os, const PairObject &obj)
{
  os << "StringPair = { " << obj.m_stringPair << " } ";
  os << "DoubleIntPair = { " << obj.m_doubleIntPair << " }";
  return os;
}

/**
 * \ingroup pair-tests
 * 
 * Pair test - Test instantiation, initialization, access.
 */
class PairValueTestCase : public TestCase
{
public:
  PairValueTestCase ();
  virtual ~PairValueTestCase () {}

private:
  virtual void DoRun ();
};

PairValueTestCase::PairValueTestCase ()
  : TestCase ("test instantiation, initialization, access")
{

}

void
PairValueTestCase::DoRun ()
{
 {
    std::pair<const int, double> ref = {1, 2.4};

    PairValue<IntegerValue, DoubleValue> ac (ref);

    std::pair<const int, double> rv = ac.Get ();
    NS_TEST_ASSERT_MSG_EQ (rv, ref, "Attribute value does not equal original");
  }

  {
    std::pair<const std::string, double> ref = {"hello", 3.14};

    PairValue<StringValue, DoubleValue> ac;
    ac.Set (ref);
    std::pair<const std::string, double> rv = ac.Get ();
    NS_TEST_ASSERT_MSG_EQ (rv, ref, "Attribute value does not equal original");
  }

}

/**
 * \ingroup pair-tests
 * 
 * Pair test - test setting through attribute interface.
 */
class PairValueSettingsTestCase : public TestCase
{
public:
  PairValueSettingsTestCase ();

  void DoRun ();
};

PairValueSettingsTestCase::PairValueSettingsTestCase ()
  : TestCase ("test setting through attribute interface")
{}

void
PairValueSettingsTestCase::DoRun ()
{
  auto p = CreateObject <PairObject> ();
  p->SetAttribute ("StringPair", PairValue<StringValue, StringValue> (std::make_pair ("hello", "world")));
  p->SetAttribute ("DoubleIntPair", PairValue<DoubleValue, IntegerValue> (std::make_pair (3.14, 31)));

  std::ostringstream oss;
  oss << *p;
  
  std::ostringstream ref;
  ref << "StringPair = { (hello,world) } DoubleIntPair = { (3.14,31) }";

  NS_TEST_ASSERT_MSG_EQ ((oss.str ()), (ref.str ()), "Pairs not correctly set");
}

/**
 * \ingroup pair-tests
 *  
 * \brief The pair-value Test Suite.
 */
class PairValueTestSuite : public TestSuite
{
  public:
    PairValueTestSuite ();
};

PairValueTestSuite::PairValueTestSuite ()
  : TestSuite ("pair-value-test-suite", UNIT)
{
  AddTestCase (new PairValueTestCase (), TestCase::QUICK);
  AddTestCase (new PairValueSettingsTestCase (), TestCase::QUICK);
}

static PairValueTestSuite g_pairValueTestSuite; //!< Static variable for test initialization
