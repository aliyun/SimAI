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
#include <ns3/attribute-container.h>
#include <ns3/attribute-container-accessor-helper.h>
#include <ns3/pair.h>
#include <ns3/double.h>
#include <ns3/integer.h>
#include <ns3/string.h>
#include <ns3/ptr.h>
#include <ns3/object.h>
#include <ns3/type-id.h>

#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <utility>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("AttributeContainerTestSuite");

/**
 * \file
 * \ingroup attribute-tests
 * Attribute container test suite
 */

/**
 * \ingroup attribute-tests
 * Attribute container object.
 */
class AttributeContainerObject : public Object
{
public:
  AttributeContainerObject ();
  virtual ~AttributeContainerObject ();

  /**
   * Reverses the list of doubles.
   */
  void ReverseList ();

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */  
  static
  TypeId GetTypeId ();

  /**
   * \brief Stream insertion operator.
   *
   * \param [in] os The reference to the output stream.
   * \param [in] obj The AttributeContainer object.
   * \returns The reference to the output stream.
   */
  friend std::ostream &operator <<(std::ostream &os, const AttributeContainerObject &obj);

private:
  std::list<double> m_doublelist; //!< List of doubles.
  std::vector<int> m_intvec;      //!< Vector of ints.
  // TODO(jared): need PairValue attributevalue to handle std::pair elements
  std::map<std::string, int> m_map; //!< Map of <std::string, int>.
};

AttributeContainerObject::AttributeContainerObject ()
{

}

AttributeContainerObject::~AttributeContainerObject ()
{

}

TypeId
AttributeContainerObject::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::AttributeContainerObject")
    .SetParent<Object> ()
    .SetGroupName("Test")
    .AddConstructor<AttributeContainerObject> ()
    .AddAttribute ("DoubleList", "List of doubles",
                   AttributeContainerValue <DoubleValue> (),
                   MakeAttributeContainerAccessor <DoubleValue> (&AttributeContainerObject::m_doublelist),
                   MakeAttributeContainerChecker<DoubleValue> (MakeDoubleChecker<double> ()))
    .AddAttribute ("IntegerVector", "Vector of integers",
                   // the container value container differs from the underlying object
                   AttributeContainerValue <IntegerValue> (),
                   MakeAttributeContainerAccessor <IntegerValue> (&AttributeContainerObject::m_intvec),
                   MakeAttributeContainerChecker<IntegerValue> (MakeIntegerChecker<int> ()))
    .AddAttribute ("MapStringInt", "Map of strings to ints",
                   // the container value container differs from the underlying object
                   AttributeContainerValue <PairValue <StringValue, IntegerValue> > (),
                   MakeAttributeContainerAccessor <PairValue <StringValue, IntegerValue> > (&AttributeContainerObject::m_map),
                   MakeAttributeContainerChecker<PairValue <StringValue, IntegerValue> > (
                   MakePairChecker<StringValue, IntegerValue> (MakeStringChecker (), MakeIntegerChecker<int> ())))
    ;
  return tid;
}

void
AttributeContainerObject::ReverseList ()
{
  m_doublelist.reverse ();
  std::vector<int> tmp;
  std::copy_backward (m_intvec.begin (), m_intvec.end (), tmp.begin ());
  m_intvec = tmp;
}

std::ostream &
operator << (std::ostream &os, const AttributeContainerObject &obj)
{
  os << "AttributeContainerObject: ";
  bool first = true;
  for (auto d: obj.m_doublelist)
    {
      if (!first) os << ", ";
      os << d;
      first = false;
    }
  return os;
}

/**
 * \ingroup attribute-tests
 *  
 * This function handles mixed constness and compatible, yet
 * distinct numerical classes (like int and long)
 * \param x The left operand.
 * \param y The right operand.
 * \return true if the pairs have the same numerical values.
 */
template <class A, class B, class C, class D>
bool
operator ==(const std::pair<A, B> &x, const std::pair<C, D> &y)
{
  return x.first == y.first && x.second == y.second;
}

/**
 * \ingroup attribute-tests
 *  
 * Test AttributeContainer instantiation, initialization, access
 */
class AttributeContainerTestCase : public TestCase
{
public:
  AttributeContainerTestCase ();
  virtual ~AttributeContainerTestCase () {}

private:
  virtual void DoRun ();
};

AttributeContainerTestCase::AttributeContainerTestCase ()
  : TestCase ("test instantiation, initialization, access")
{

}

void
AttributeContainerTestCase::DoRun ()
{
 {
    std::list<double> ref = {1.0, 2.1, 3.145269};

    AttributeContainerValue<DoubleValue> ac (ref);

    NS_TEST_ASSERT_MSG_EQ (ref.size (), ac.GetN (), "Container size mismatch");
    auto aciter = ac.Begin ();
    for (auto rend = ref.end (),
              riter= ref.begin (); riter != rend; ++riter)
      {
        NS_TEST_ASSERT_MSG_NE (true, (aciter == ac.End ()), "AC iterator reached end");
        NS_TEST_ASSERT_MSG_EQ (*riter, (*aciter)->Get (), "Incorrect value");
        ++aciter;
      }
    NS_TEST_ASSERT_MSG_EQ (true, (aciter == ac.End ()), "AC iterator did not reach end");
  }

  {
    std::vector<int> ref = {-2, 3, 10, -1042};

    AttributeContainerValue<IntegerValue> ac (ref.begin (), ref.end ());

    NS_TEST_ASSERT_MSG_EQ (ref.size (), ac.GetN (), "Container size mismatch");
    auto aciter = ac.Begin ();
    for (auto rend = ref.end (),
              riter= ref.begin (); riter != rend; ++riter)
      {
        NS_TEST_ASSERT_MSG_NE (true, (aciter == ac.End ()), "AC iterator reached end");
        NS_TEST_ASSERT_MSG_EQ (*riter, (*aciter)->Get (), "Incorrect value");
        ++aciter;
      }
    NS_TEST_ASSERT_MSG_EQ (true, (aciter == ac.End ()), "AC iterator did not reach end");
  }

  {
    auto ref = {"one", "two", "three"};
    AttributeContainerValue<StringValue> ac (ref.begin (), ref.end ());

    NS_TEST_ASSERT_MSG_EQ (3, ac.GetN (), "Container size mismatch");
    auto aciter = ac.Begin ();
    for (auto v: ref)
      {
        NS_TEST_ASSERT_MSG_NE (true, (aciter == ac.End ()), "AC iterator reached end");
        NS_TEST_ASSERT_MSG_EQ (v, (*aciter)->Get (), "Incorrect value");
        ++aciter;
      }
    NS_TEST_ASSERT_MSG_EQ (true, (aciter == ac.End ()), "AC iterator did not reach end");
  }

  {
    auto ref = {"one", "two", "three"};
    AttributeContainerValue<StringValue, std::vector> ac (ref);

    NS_TEST_ASSERT_MSG_EQ (3, ac.GetN (), "Container size mismatch");
    auto aciter = ac.Begin ();
    for (auto v: ref)
      {
        NS_TEST_ASSERT_MSG_NE (true, (aciter == ac.End ()), "AC iterator reached end");
        NS_TEST_ASSERT_MSG_EQ (v, (*aciter)->Get (), "Incorrect value");
        ++aciter;
      }
    NS_TEST_ASSERT_MSG_EQ (true, (aciter == ac.End ()), "AC iterator did not reach end");
  }

  {
    // use int64_t which is default for IntegerValue
    std::map<std::string, int64_t> ref = { {"one", 1}, {"two", 2}, {"three", 3}};
    AttributeContainerValue<PairValue <StringValue, IntegerValue> > ac (ref);

    NS_TEST_ASSERT_MSG_EQ (3, ac.GetN (), "Container size mismatch");
    auto aciter = ac.Begin ();
    for (auto v: ref)
      {
        NS_TEST_ASSERT_MSG_NE (true, (aciter == ac.End ()), "AC iterator reached end");
        NS_TEST_ASSERT_MSG_EQ (v, (*aciter)->Get (), "Incorrect value");
        ++aciter;
      }
    NS_TEST_ASSERT_MSG_EQ (true, (aciter == ac.End ()), "AC iterator did not reach end");

  }
}

/**
 * \ingroup attribute-tests
 *  
 * Attribute serialization and deserialization TestCase.
 */
class AttributeContainerSerializationTestCase : public TestCase
{
public:
  AttributeContainerSerializationTestCase ();
  virtual ~AttributeContainerSerializationTestCase () {}

private:
  virtual void DoRun (void);
};

AttributeContainerSerializationTestCase::AttributeContainerSerializationTestCase ()
  : TestCase ("test serialization and deserialization")
{

}

void
AttributeContainerSerializationTestCase::DoRun (void)
{
  {
    // notice embedded spaces
    std::string doubles = "1.0001, 20.53, -102.3";

    AttributeContainerValue<DoubleValue> attr;
    auto checker = MakeAttributeContainerChecker (attr);
    auto acchecker = DynamicCast<AttributeContainerChecker> (checker);
    acchecker->SetItemChecker (MakeDoubleChecker<double> ());
    NS_TEST_ASSERT_MSG_EQ (attr.DeserializeFromString (doubles, checker), true, "Deserialize failed");
    NS_TEST_ASSERT_MSG_EQ (attr.GetN (), 3, "Incorrect container size");

    std::string reserialized = attr.SerializeToString (checker);
    std::string canonical = doubles;
    canonical.erase (std::remove (canonical.begin (), canonical.end (), ' '), canonical.end ());
    NS_TEST_ASSERT_MSG_EQ (reserialized, canonical, "Reserialization failed");
  }

  {
    // notice embedded spaces
    std::string ints = "1, 2, -3, -4";

    AttributeContainerValue<IntegerValue> attr;
    auto checker = MakeAttributeContainerChecker (attr);
    auto acchecker = DynamicCast<AttributeContainerChecker> (checker);
    acchecker->SetItemChecker (MakeIntegerChecker<int> ());
    NS_TEST_ASSERT_MSG_EQ (attr.DeserializeFromString (ints, checker), true, "Deserialize failed");
    NS_TEST_ASSERT_MSG_EQ (attr.GetN (), 4, "Incorrect container size");

    std::string reserialized = attr.SerializeToString (checker);
    std::string canonical = ints;
    canonical.erase (std::remove (canonical.begin (), canonical.end (), ' '), canonical.end ());
    NS_TEST_ASSERT_MSG_EQ (reserialized, canonical, "Reserialization failed");
  }

  {
    std::string strings = "this is a sentence with words";

    AttributeContainerValue<StringValue> attr (' ');
    auto checker = MakeAttributeContainerChecker (attr);
    auto acchecker = DynamicCast<AttributeContainerChecker> (checker);
    acchecker->SetItemChecker (MakeStringChecker ());
    NS_TEST_ASSERT_MSG_EQ (attr.DeserializeFromString (strings, checker), true, "Deserialize failed");
    NS_TEST_ASSERT_MSG_EQ (attr.GetN (), 6, "Incorrect container size");

    std::string reserialized = attr.SerializeToString (checker);
    std::string canonical = strings;
    NS_TEST_ASSERT_MSG_EQ (reserialized, canonical, "Reserialization failed");
  }

  {
    std::string pairs = "one 1,two 2,three 3";
    AttributeContainerValue<PairValue<StringValue, IntegerValue> > attr;
    auto checker = MakeAttributeContainerChecker (attr);
    auto acchecker = DynamicCast<AttributeContainerChecker> (checker);
    acchecker->SetItemChecker (MakePairChecker <StringValue, IntegerValue> (
                                MakeStringChecker (), MakeIntegerChecker<int> ()));
    NS_TEST_ASSERT_MSG_EQ (attr.DeserializeFromString (pairs, checker), true, "Deserialization failed");
    NS_TEST_ASSERT_MSG_EQ (attr.GetN (), 3, "Incorrect container size");

    std::string reserialized = attr.SerializeToString (checker);
    std::string canonical = pairs;
    NS_TEST_ASSERT_MSG_EQ (reserialized, canonical, "Reserealization failed");

  }
}

/**
 * \ingroup attribute-tests
 *  
 * Attribute set and get TestCase.
 */
class AttributeContainerSetGetTestCase : public TestCase
{
public:
  AttributeContainerSetGetTestCase ();
  virtual ~AttributeContainerSetGetTestCase () {}

private:
  virtual void DoRun (void);
};

AttributeContainerSetGetTestCase::AttributeContainerSetGetTestCase ()
  : TestCase ("test attribute set and get")
{

}

void
AttributeContainerSetGetTestCase::DoRun (void)
{
  Ptr<AttributeContainerObject> obj = CreateObject<AttributeContainerObject> ();
  {
    std::ostringstream oss;
    oss << *obj;
    NS_TEST_ASSERT_MSG_EQ (oss.str (), "AttributeContainerObject: ", "DoubleList initialized incorrectly");
  }

  std::list<double> doubles = {1.1, 2.22, 3.333};
  obj->SetAttribute ("DoubleList", AttributeContainerValue<DoubleValue> (doubles));
  {
    std::ostringstream oss;
    oss << *obj;
    NS_TEST_ASSERT_MSG_EQ (oss.str (), "AttributeContainerObject: 1.1, 2.22, 3.333", "DoubleList incorrectly set");
  }

  obj->ReverseList ();
  {
    std::ostringstream oss;
    oss << *obj;
    NS_TEST_ASSERT_MSG_EQ (oss.str (), "AttributeContainerObject: 3.333, 2.22, 1.1", "DoubleList incorrectly reversed");

    // NOTE: changing the return container here too!
    AttributeContainerValue<DoubleValue> value;
    obj->GetAttribute ("DoubleList", value);
    NS_TEST_ASSERT_MSG_EQ (doubles.size (), value.GetN (), "AttributeContainerValue wrong size");

    AttributeContainerValue<DoubleValue>::result_type doublevec = value.Get ();
    NS_TEST_ASSERT_MSG_EQ (doubles.size (), doublevec.size (), "DoublesVec wrong size");
    auto iter = doubles.rbegin ();
    for (auto d: doublevec)
      {
        NS_TEST_ASSERT_MSG_EQ (d, *iter, "Incorrect value in doublesvec");
        ++iter;
      }
  }

  std::vector<int> ints = {-1, 0, 1, 2, 3};
  // NOTE: here the underlying attribute container type differs from the actual container
  obj->SetAttribute ("IntegerVector", AttributeContainerValue<IntegerValue> (ints));

  {
    // NOTE: changing the container here too!
    AttributeContainerValue<IntegerValue> value;
    obj->GetAttribute ("IntegerVector", value);
    NS_TEST_ASSERT_MSG_EQ (ints.size (), value.GetN (), "AttributeContainerValue wrong size");

    AttributeContainerValue<IntegerValue>::result_type intlist = value.Get ();
    NS_TEST_ASSERT_MSG_EQ (ints.size (), intlist.size (), "Intvec wrong size");
    auto iter = ints.begin ();
    for (auto d: intlist)
      {
        NS_TEST_ASSERT_MSG_EQ (d, *iter, "Incorrect value in intvec");
        ++iter;
      }
  }

  std::map<std::string, int> map = { {"one", 1}, {"two", 2}, {"three", 3}};
  obj->SetAttribute ("MapStringInt", AttributeContainerValue<PairValue <StringValue, IntegerValue> > (map));

  {
    AttributeContainerValue<PairValue<StringValue, IntegerValue> > value;
    obj->GetAttribute ("MapStringInt", value);
    NS_TEST_ASSERT_MSG_EQ (map.size (), value.GetN (), "AttributeContainerValue wrong size");

    // could possibly make custom assignment operator to make assignment statement work
    std::map<std::string, int> mapstrint;
    auto lst = value.Get ();
    for (auto l: lst) mapstrint[l.first] = l.second;

    NS_TEST_ASSERT_MSG_EQ (map.size (), mapstrint.size (), "mapstrint wrong size");
    auto iter = map.begin ();
    for (auto v: mapstrint)
      {
        NS_TEST_ASSERT_MSG_EQ (v, *iter, "Incorrect value in mapstrint");
        ++iter;
      }
  }
}

/**
 * \ingroup attribute-tests
 *  
 * Attribute attribute container TestCase.
 */
class AttributeContainerTestSuite : public TestSuite
{
  public:
    AttributeContainerTestSuite ();
};

AttributeContainerTestSuite::AttributeContainerTestSuite ()
  : TestSuite ("attribute-container-test-suite", UNIT)
{
  AddTestCase (new AttributeContainerTestCase (), TestCase::QUICK);
  AddTestCase (new AttributeContainerSerializationTestCase (), TestCase::QUICK);
  AddTestCase (new AttributeContainerSetGetTestCase (), TestCase::QUICK);
}

static AttributeContainerTestSuite g_attributeContainerTestSuite; //!< Static variable for test initialization
