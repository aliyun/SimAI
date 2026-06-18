/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Lawrence Livermore National Laboratory
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include <iostream>
#include <iomanip>
#include <ctime>

#include "ns3/integer.h"
#include "ns3/double.h"
#include "ns3/object.h"
#include "ns3/traced-value.h"
#include "ns3/type-id.h"
#include "ns3/test.h"
#include "ns3/log.h"

using namespace std;

using namespace ns3;

/// Const string used to build the test name.
const std::string suite ("type-id: ");

/**
 * \file
 * \ingroup typeid-tests
 * TypeId test suite
 */

/**
 * \ingroup core-tests
 * \defgroup typeid-tests TypeId class tests
 */

/**
 * \ingroup typeid-tests
 * 
 * Test for uniqueness of all TypeIds.
 */
class UniqueTypeIdTestCase : public TestCase
{
public:
  UniqueTypeIdTestCase ();
  virtual ~UniqueTypeIdTestCase ();

private:
  virtual void DoRun (void);
  enum
  {
    HashChainFlag = 0x80000000
  };
};

UniqueTypeIdTestCase::UniqueTypeIdTestCase ()
  : TestCase ("Check uniqueness of all TypeIds")
{}

UniqueTypeIdTestCase::~UniqueTypeIdTestCase ()
{}

void
UniqueTypeIdTestCase::DoRun (void)
{
  cout << suite << endl;
  cout << suite << GetName () << endl;

  // Use same custom hasher as TypeId
  ns3::Hasher hasher = ns3::Hasher ( Create<Hash::Function::Murmur3> () );

  uint32_t nids = TypeId::GetRegisteredN ();

  cout << suite << "UniqueTypeIdTestCase: nids: " << nids << endl;
  cout << suite << "TypeId list:" << endl;
  cout << suite << "TypeId  Chain  hash          Name" << endl;

  for (uint16_t i = 0; i < nids; ++i)
    {
      const TypeId tid = TypeId::GetRegistered (i);
      cout << suite << "" << std::setw (6) << tid.GetUid ();
      if (tid.GetHash () & HashChainFlag)
        {
          cout << "  chain";
        }
      else
        {
          cout << "       ";
        }
      cout << "  0x"     << std::setfill ('0') << std::hex << std::setw (8)
           << tid.GetHash () << std::dec << std::setfill (' ')
           << "    "     << tid.GetName ()
           << endl;

      NS_TEST_ASSERT_MSG_EQ (tid.GetUid (),
                             TypeId::LookupByName (tid.GetName ()).GetUid (),
                             "LookupByName returned different TypeId for "
                             << tid.GetName ());

      // Mask off HashChainFlag in this test, since tid might have been chained
      NS_TEST_ASSERT_MSG_EQ ((tid.GetHash () & (~HashChainFlag)),
                             (hasher.clear ().GetHash32 (tid.GetName ()) & (~HashChainFlag)),
                             "TypeId .hash and Hash32 (.name) unequal for "
                             << tid.GetName ());

      NS_TEST_ASSERT_MSG_EQ (tid.GetUid (),
                             TypeId::LookupByHash (tid.GetHash ()).GetUid (),
                             "LookupByHash returned different TypeId for "
                             << tid.GetName ());

    }

  cout << suite << "<-- end TypeId list -->" << endl;
}


/**
 * \ingroup typeid-tests
 * 
 * Collision test.
 */
class CollisionTestCase : public TestCase
{
public:
  CollisionTestCase ();
  virtual ~CollisionTestCase ();

private:
  virtual void DoRun (void);
  enum
  {
    HashChainFlag = 0x80000000
  };
};

CollisionTestCase::CollisionTestCase ()
  : TestCase ("Check behavior when type names collide")
{}

CollisionTestCase::~CollisionTestCase ()
{}

void
CollisionTestCase::DoRun (void)
{
  cout << suite << endl;
  cout << suite << GetName () << endl;

  // Register two types whose hashes collide, in alphabetical order
  // Murmur3 collision from /usr/share/dict/web2
  string t1Name = "daemon";
  string t2Name = "unerring";
  cout << suite << "creating colliding types "
       << "'" << t1Name << "', '" << t2Name << "'"
       << " in alphabetical order:"
       << endl;
  TypeId t1 (t1Name);
  TypeId t2 (t2Name);

  // Check that they are alphabetical: t1 name < t2 name
  NS_TEST_ASSERT_MSG_EQ ( (t1.GetHash () & HashChainFlag), 0,
                          "First and lesser TypeId has HashChainFlag set");
  cout << suite << "collision: first,lesser  not chained: OK" << endl;

  NS_TEST_ASSERT_MSG_NE ( (t2.GetHash () & HashChainFlag), 0,
                          "Second and greater TypeId does not have HashChainFlag set");
  cout << suite << "collision: second,greater    chained: OK" << endl;


  // Register colliding types in reverse alphabetical order
  // Murmur3 collision from /usr/share/dict/web2
  string t3Name = "trigonon";
  string t4Name = "seriation";
  cout << suite << "creating colliding types "
       << "'" << t3Name << "', '" << t4Name << "'"
       << " in reverse alphabetical order:"
       << endl;
  TypeId t3 (t3Name.c_str ());
  TypeId t4 (t4Name.c_str ());

  // Check that they are alphabetical: t3 name > t4 name
  NS_TEST_ASSERT_MSG_NE ( (t3.GetHash () & HashChainFlag), 0,
                          "First and greater TypeId does not have HashChainFlag set");
  cout << suite << "collision: first,greater     chained: OK" << endl;

  NS_TEST_ASSERT_MSG_EQ ( (t4.GetHash () & HashChainFlag), 0,
                          "Second and lesser TypeId has HashChainFlag set");
  cout << suite << "collision: second,lesser not chained: OK" << endl;

  /** TODO Extra credit:  register three types whose hashes collide
   *
   *  None found in /usr/share/dict/web2
   */

}


/**
 * \ingroup typeid-tests
 * 
 * Class used to test deprecated Attributes.
 */
class DeprecatedAttribute : public Object
{
private:
  // float m_obsAttr;  // this is obsolete, no trivial forwarding
  // int m_oldAttr;  // this has become m_attr
  int m_attr; //!< An attribute to test deprecation.

  // TracedValue<int> m_obsTrace;  // this is obsolete, no trivial forwarding
  // TracedValue<double> m_oldTrace;  // this has become m_trace
  TracedValue<double> m_trace;  //!< A TracedValue to test deprecation.

public:
  DeprecatedAttribute ()
    : m_attr (0)
  {
  }
  virtual ~DeprecatedAttribute ()
  {}

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void)
  {
    static TypeId tid = TypeId ("DeprecatedAttribute")
      .SetParent<Object> ()

      // The new attribute
      .AddAttribute ("attribute",
                     "the Attribute",
                     IntegerValue (1),
                     MakeIntegerAccessor (&DeprecatedAttribute::m_attr),
                     MakeIntegerChecker<int> ())
      // The old deprecated attribute
      .AddAttribute ("oldAttribute",
                     "the old attribute",
                     IntegerValue (1),
                     MakeIntegerAccessor (&DeprecatedAttribute::m_attr),
                     MakeIntegerChecker<int> (),
                     TypeId::DEPRECATED,
                     "use 'attribute' instead")
      // Obsolete attribute, as an example
      .AddAttribute ("obsoleteAttribute",
                     "the obsolete attribute",
                     EmptyAttributeValue (),
                     MakeEmptyAttributeAccessor (),
                     MakeEmptyAttributeChecker (),
                     TypeId::OBSOLETE,
                     "refactor to use 'attribute'")

      // The new trace source
      .AddTraceSource ("trace",
                       "the TraceSource",
                       MakeTraceSourceAccessor (&DeprecatedAttribute::m_trace),
                       "ns3::TracedValueCallback::Double")
      // The old trace source
      .AddTraceSource ("oldTrace",
                       "the old trace source",
                       MakeTraceSourceAccessor (&DeprecatedAttribute::m_trace),
                       "ns3::TracedValueCallback::Double",
                       TypeId::DEPRECATED,
                       "use 'trace' instead")
      // Obsolete trace source, as an example
      .AddTraceSource ("obsoleteTraceSource",
                       "the obsolete trace source",
                       MakeEmptyTraceSourceAccessor (),
                       "ns3::TracedValueCallback::Void",
                       TypeId::OBSOLETE,
                       "refactor to use 'trace'");

    return tid;
  }

};


/**
 * \ingroup typeid-tests
 * 
 * Check deprecated Attributes and TraceSources.
 */
class DeprecatedAttributeTestCase : public TestCase
{
public:
  DeprecatedAttributeTestCase ();
  virtual ~DeprecatedAttributeTestCase ();

private:
  virtual void DoRun (void);

};

DeprecatedAttributeTestCase::DeprecatedAttributeTestCase ()
  : TestCase ("Check deprecated Attributes and TraceSources")
{}

DeprecatedAttributeTestCase::~DeprecatedAttributeTestCase ()
{}

void
DeprecatedAttributeTestCase::DoRun (void)
{
  cerr << suite << endl;
  cerr << suite << GetName () << endl;

  TypeId tid = DeprecatedAttribute::GetTypeId ();
  cerr << suite << "DeprecatedAttribute TypeId: " << tid.GetUid () << endl;

  //  Try the lookups
  struct TypeId::AttributeInformation ainfo;
  NS_TEST_ASSERT_MSG_EQ (tid.LookupAttributeByName ("attribute", &ainfo), true,
                         "lookup new attribute");
  cerr << suite << "lookup new attribute:"
       << (ainfo.supportLevel == TypeId::SUPPORTED ? "supported" : "error")
       << endl;

  NS_TEST_ASSERT_MSG_EQ (tid.LookupAttributeByName ("oldAttribute", &ainfo), true,
                         "lookup old attribute");
  cerr << suite << "lookup old attribute:"
       << (ainfo.supportLevel == TypeId::DEPRECATED ? "deprecated" : "error")
       << endl;


  struct TypeId::TraceSourceInformation tinfo;
  Ptr<const TraceSourceAccessor> acc;
  acc = tid.LookupTraceSourceByName ("trace", &tinfo);
  NS_TEST_ASSERT_MSG_NE (acc, 0, "lookup new trace source");
  cerr << suite << "lookup new trace source:"
       << (tinfo.supportLevel == TypeId::SUPPORTED ? "supported" : "error")
       << endl;

  acc = tid.LookupTraceSourceByName ("oldTrace", &tinfo);
  NS_TEST_ASSERT_MSG_NE (acc, 0, "lookup old trace source");
  cerr << suite << "lookup old trace source:"
       << (tinfo.supportLevel == TypeId::DEPRECATED ? "deprecated" : "error")
       << endl;
}


/**
 * \ingroup typeid-tests
 * 
 * Performance test: measure average lookup time.
 */
class LookupTimeTestCase : public TestCase
{
public:
  LookupTimeTestCase ();
  virtual ~LookupTimeTestCase ();

private:
  void DoRun (void);
  void DoSetup (void);
  /**
   * Report the performance test results.
   * \param how How the TypeId is searched (name or hash).
   * \param delta The time required for the lookup.
   */
  void Report (const std::string how, const uint32_t delta) const;

  enum
  {
    REPETITIONS = 100000
  };
};

LookupTimeTestCase::LookupTimeTestCase ()
  : TestCase ("Measure average lookup time")
{}

LookupTimeTestCase::~LookupTimeTestCase ()
{}

void
LookupTimeTestCase::DoRun (void)
{
  cout << suite << endl;
  cout << suite << GetName () << endl;

  uint32_t nids = TypeId::GetRegisteredN ();

  int start = clock ();
  for (uint32_t j = 0; j < REPETITIONS; ++j)
    {
      for (uint16_t i = 0; i < nids; ++i)
        {
          const TypeId tid = TypeId::GetRegistered (i);
          const TypeId sid = TypeId::LookupByName (tid.GetName ());
        }
    }
  int stop = clock ();
  Report ("name", stop - start);

  start = clock ();
  for (uint32_t j = 0; j < REPETITIONS; ++j)
    {
      for (uint16_t i = 0; i < nids; ++i)
        {
          const TypeId tid = TypeId::GetRegistered (i);
          const TypeId sid = TypeId::LookupByHash (tid.GetHash ());
        }
    }
  stop = clock ();
  Report ("hash", stop - start);

}

void
LookupTimeTestCase::DoSetup (void)
{
  uint32_t nids = TypeId::GetRegisteredN ();

  cout << suite << "Lookup time: reps: " << REPETITIONS
       << ", num TypeId's: " << nids
       << endl;

}

void
LookupTimeTestCase::Report (const std::string how,
                            const uint32_t    delta) const
{
  double nids = TypeId::GetRegisteredN ();
  double reps = nids * REPETITIONS;

  double per = 1E6 * double(delta) / (reps * double(CLOCKS_PER_SEC));

  cout << suite << "Lookup time: by " << how << ": "
       << "ticks: " << delta
       << "\tper: "   << per
       << " microsec/lookup"
       << endl;
}


/**
 * \ingroup typeid-tests
 * 
 * TypeId test suites.
 */
class TypeIdTestSuite : public TestSuite
{
public:
  TypeIdTestSuite ();
};

TypeIdTestSuite::TypeIdTestSuite ()
  : TestSuite ("type-id", UNIT)
{
  // Turn on logging, so we see the result of collisions
  LogComponentEnable ("TypeId", ns3::LogLevel (LOG_ERROR | LOG_PREFIX_FUNC));

  // If the CollisionTestCase is performed before the
  // UniqueIdTestCase, the artificial collisions added by
  // CollisionTestCase will show up in the list of TypeIds
  // as chained.
  AddTestCase (new UniqueTypeIdTestCase, QUICK);
  AddTestCase (new CollisionTestCase, QUICK);
  AddTestCase (new DeprecatedAttributeTestCase, QUICK);
}

/// Static variable for test initialization.
static TypeIdTestSuite g_TypeIdTestSuite; 


/**
 * \ingroup typeid-tests
 * 
 * TypeId performance test suites.
 */
class TypeIdPerformanceSuite : public TestSuite
{
public:
  TypeIdPerformanceSuite ();
};

TypeIdPerformanceSuite::TypeIdPerformanceSuite ()
  : TestSuite ("type-id-perf", PERFORMANCE)
{
  AddTestCase (new LookupTimeTestCase, QUICK);
}

/// Static variable for test initialization.
static TypeIdPerformanceSuite g_TypeIdPerformanceSuite;
