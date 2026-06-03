/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include <ns3/test.h>
#include <ns3/log.h>
#include <ns3/tuple.h>
#include <ns3/double.h>
#include <ns3/uinteger.h>
#include <ns3/string.h>
#include <ns3/ptr.h>
#include <ns3/object.h>
#include <ns3/enum.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <utility>
#include <tuple>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TupleTestSuite");

/** Object with attribute values storing tuples */
class TupleObject : public Object
{
public:
  /**
   * Test enum type
   */
  enum TupleTestEnum
  {
    VALUE1,
    VALUE2,
    VALUE3
  };

  TupleObject ();
  virtual ~TupleObject ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  // NOTE EnumValue::Get() return an int, so the tuple element type must be an int
  // in place of the enum type
  using Tuple1Value = TupleValue<StringValue, StringValue, EnumValue>;  //!< Tuple1 attribute value
  using Tuple1 = Tuple1Value::result_type;                              //!< tuple of values
  using Tuple1Pack = Tuple1Value::value_type;                           //!< tuple of attribute values

  using Tuple2 = std::tuple <double, uint16_t, std::string>;            //!< Tuple2 typedef

  /**
   * Set tuple1
   * \param tuple tuple value
   */
  void SetTuple1 (const Tuple1& tuple);
  /**
   * Get tuple1
   * \return tuple1
   */
  Tuple1 GetTuple1 (void) const;
  /**
   * Set tuple2
   * \param tuple tuple value
   */
  void SetTuple2 (const Tuple2& tuple);
  /**
   * Get tuple2
   * \return tuple2
   */
  Tuple2 GetTuple2 (void) const;

private:
  Tuple1 m_tuple1;     //!< first tuple
  Tuple2 m_tuple2;     //!< second tuple
};

TypeId
TupleObject::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::TupleObject")
    .SetParent<Object> ()
    .SetGroupName("Test")
    .AddConstructor<TupleObject> ()
    .AddAttribute ("StringStringEnumTuple", "Tuple1: string, string, enum",
                   MakeTupleValue<Tuple1Pack> (Tuple1 {"Hey", "Jude", TupleObject::VALUE1}),
                   MakeTupleAccessor<Tuple1Pack> (&TupleObject::m_tuple1),
                   MakeTupleChecker<Tuple1Pack> (MakeStringChecker (), MakeStringChecker (),
                                            MakeEnumChecker (TupleObject::VALUE1, "VALUE1",
                                                             TupleObject::VALUE2, "VALUE2")))
    .AddAttribute ("DoubleUintStringTuple", "Tuple2: double, uint16_t, string",
                   TupleValue <DoubleValue, UintegerValue, StringValue> ({6.022, 23, "Avogadro"}),
                   MakeTupleAccessor <DoubleValue, UintegerValue, StringValue> (&TupleObject::SetTuple2,
                                                                                &TupleObject::GetTuple2),
                   MakeTupleChecker<DoubleValue, UintegerValue, StringValue> (MakeDoubleChecker<double> (1.0, 10.0),
                                                                              MakeUintegerChecker<int> (1, 30),
                                                                              MakeStringChecker ()))
    ;
  return tid;
}

TupleObject::TupleObject ()
{
}

TupleObject::~TupleObject ()
{
}

void
TupleObject::SetTuple1 (const Tuple1& tuple)
{
  m_tuple1 = tuple;
}

TupleObject::Tuple1
TupleObject::GetTuple1 (void) const
{
  return m_tuple1;
}

void
TupleObject::SetTuple2 (const Tuple2& tuple)
{
  m_tuple2 = tuple;
}

TupleObject::Tuple2
TupleObject::GetTuple2 (void) const
{
  return m_tuple2;
}


/** Test instantiation, initialization, access */
class TupleValueTestCase : public TestCase
{
public:
  TupleValueTestCase ();
  virtual ~TupleValueTestCase () {}

private:
  virtual void DoRun ();
};

TupleValueTestCase::TupleValueTestCase ()
  : TestCase ("test TupleValue attribute value")
{
}

void
TupleValueTestCase::DoRun ()
{
  auto tupleObject = CreateObject <TupleObject> ();

  // Test that default values have been assigned to tuple 1
  auto t1 = tupleObject->GetTuple1 ();
  NS_TEST_ASSERT_MSG_EQ ((std::get<0> (t1) == "Hey"), true, "First element of tuple 1 not correctly set");
  NS_TEST_ASSERT_MSG_EQ ((std::get<1> (t1) == "Jude"), true, "Second element of tuple 1 not correctly set");
  NS_TEST_ASSERT_MSG_EQ (std::get<2> (t1), (int)(TupleObject::VALUE1), "Third element of tuple 1 not correctly set");

  // Test that default values have been assigned to tuple 2
  auto t2 = tupleObject->GetTuple2 ();
  NS_TEST_ASSERT_MSG_EQ (std::get<0> (t2), 6.022, "First element of tuple 2 not correctly set");
  NS_TEST_ASSERT_MSG_EQ (std::get<1> (t2), 23, "Second element of tuple 2 not correctly set");
  NS_TEST_ASSERT_MSG_EQ ((std::get<2> (t2) == "Avogadro"), true, "Third element of tuple 2 not correctly set");

  // Test that we can correctly set and get new values for tuple 1
  bool ret1 = tupleObject->SetAttributeFailSafe ("StringStringEnumTuple",
                                                 MakeTupleValue<TupleObject::Tuple1Pack> (TupleObject::Tuple1 {"Norwegian", "Wood", TupleObject::VALUE2}));
  NS_TEST_ASSERT_MSG_EQ (ret1, true, "Setting valid values to tuple 1 failed");

  TupleValue<StringValue, StringValue, EnumValue> tupleValue1;
  ret1 = tupleObject->GetAttributeFailSafe ("StringStringEnumTuple", tupleValue1);
  NS_TEST_ASSERT_MSG_EQ (ret1, true, "Getting values for tuple 1 failed");

  t1 = tupleValue1.Get ();
  NS_TEST_ASSERT_MSG_EQ ((std::get<0> (t1) == "Norwegian"), true, "First element of tuple 1 not correctly set");
  NS_TEST_ASSERT_MSG_EQ ((std::get<1> (t1) == "Wood"), true, "Second element of tuple 1 not correctly set");
  NS_TEST_ASSERT_MSG_EQ (std::get<2> (t1), (int)(TupleObject::VALUE2), "Third element of tuple 1 not correctly set");

  // Test that we can correctly set and get new values for tuple 2
  bool ret2 = tupleObject->SetAttributeFailSafe ("DoubleUintStringTuple",
                                                 TupleValue <DoubleValue, UintegerValue, StringValue> ({8.987, 9, "Coulomb"}));
  NS_TEST_ASSERT_MSG_EQ (ret2, true, "Setting valid values to tuple 2 failed");

  TupleValue <DoubleValue, UintegerValue, StringValue> tupleValue2;
  ret2 = tupleObject->GetAttributeFailSafe ("DoubleUintStringTuple", tupleValue2);
  NS_TEST_ASSERT_MSG_EQ (ret2, true, "Getting values for tuple 2 failed");

  t2 = tupleValue2.Get ();
  NS_TEST_ASSERT_MSG_EQ (std::get<0> (t2), 8.987, "First element of tuple 2 not correctly set");
  NS_TEST_ASSERT_MSG_EQ (std::get<1> (t2), 9, "Second element of tuple 2 not correctly set");
  NS_TEST_ASSERT_MSG_EQ ((std::get<2> (t2) == "Coulomb"), true, "Third element of tuple 2 not correctly set");

  // Test that we can set tuple 1 from string
  ret1 = tupleObject->SetAttributeFailSafe ("StringStringEnumTuple", StringValue ("{Come, Together, VALUE1}"));
  NS_TEST_ASSERT_MSG_EQ (ret1, true, "Setting valid values to tuple 1 failed");

  t1 = tupleObject->GetTuple1 ();
  NS_TEST_ASSERT_MSG_EQ ((std::get<0> (t1) == "Come"), true, "First element of tuple 1 not correctly set");
  NS_TEST_ASSERT_MSG_EQ ((std::get<1> (t1) == "Together"), true, "Second element of tuple 1 not correctly set");
  NS_TEST_ASSERT_MSG_EQ (std::get<2> (t1), (int)(TupleObject::VALUE1), "Third element of tuple 1 not correctly set");

  // Test that we can set tuple 2 from string
  ret2 = tupleObject->SetAttributeFailSafe ("DoubleUintStringTuple", StringValue ("{2.99, 8, LightSpeed}"));
  NS_TEST_ASSERT_MSG_EQ (ret2, true, "Setting valid values to tuple 2 failed");

  t2 = tupleObject->GetTuple2 ();
  NS_TEST_ASSERT_MSG_EQ (std::get<0> (t2), 2.99, "First element of tuple 2 not correctly set");
  NS_TEST_ASSERT_MSG_EQ (std::get<1> (t2), 8, "Second element of tuple 2 not correctly set");
  NS_TEST_ASSERT_MSG_EQ ((std::get<2> (t2) == "LightSpeed"), true, "Third element of tuple 2 not correctly set");

  // Test that setting invalid values fails
  ret1 = tupleObject->SetAttributeFailSafe ("StringStringEnumTuple",
                                            TupleValue<StringValue, StringValue> ({"Get", "Back"}));
  NS_TEST_ASSERT_MSG_EQ (ret1, false, "Too few values");
  NS_TEST_ASSERT_MSG_EQ ((tupleObject->GetTuple1 () == std::make_tuple ("Come", "Together", (int)(TupleObject::VALUE1))), true,
                         "Tuple modified after failed assignment");

  ret1 = tupleObject->SetAttributeFailSafe ("StringStringEnumTuple",
                                            MakeTupleValue<TupleObject::Tuple1Pack> (TupleObject::Tuple1 {"Get", "Back", TupleObject::VALUE3}));
  NS_TEST_ASSERT_MSG_EQ (ret1, false, "Invalid enum value");
  NS_TEST_ASSERT_MSG_EQ ((tupleObject->GetTuple1 () == std::make_tuple ("Come", "Together", (int)(TupleObject::VALUE1))), true,
                         "Tuple modified after failed assignment");

  ret2 = tupleObject->SetAttributeFailSafe ("DoubleUintStringTuple",
                                            TupleValue <DoubleValue, UintegerValue, StringValue, StringValue> ({4.83, 14, "Josephson", "constant"}));
  NS_TEST_ASSERT_MSG_EQ (ret2, false, "Too many values");
  NS_TEST_ASSERT_MSG_EQ ((tupleObject->GetTuple2 () == std::make_tuple (2.99, 8, "LightSpeed")), true,
                         "Tuple modified after failed assignment");

  ret2 = tupleObject->SetAttributeFailSafe ("DoubleUintStringTuple",
                                            TupleValue <DoubleValue, UintegerValue, StringValue> ({48.3, 13, "Josephson"}));
  NS_TEST_ASSERT_MSG_EQ (ret2, false, "Double value out of range");
  NS_TEST_ASSERT_MSG_EQ ((tupleObject->GetTuple2 () == std::make_tuple (2.99, 8, "LightSpeed")), true,
                         "Tuple modified after failed assignment");

  ret2 = tupleObject->SetAttributeFailSafe ("DoubleUintStringTuple",
                                            TupleValue <DoubleValue, UintegerValue, StringValue> ({4.83, 130, "Josephson"}));
  NS_TEST_ASSERT_MSG_EQ (ret2, false, "Uinteger value out of range");
  NS_TEST_ASSERT_MSG_EQ ((tupleObject->GetTuple2 () == std::make_tuple (2.99, 8, "LightSpeed")), true,
                         "Tuple modified after failed assignment");


}


/** Test suite */
class TupleValueTestSuite : public TestSuite
{
  public:
    TupleValueTestSuite ();
};

TupleValueTestSuite::TupleValueTestSuite ()
  : TestSuite ("tuple-value-test-suite", UNIT)
{
  AddTestCase (new TupleValueTestCase (), TestCase::QUICK);
}

static TupleValueTestSuite g_tupleValueTestSuite; //!< Static variable for test initialization
