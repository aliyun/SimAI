/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#include "ns3/type-traits.h"
#include "ns3/test.h"

/**
 * \file
 * \ingroup core-tests
 * \ingroup object
 * \ingroup object-tests
 * TypeTraits test suite.
 */

namespace ns3 {

namespace tests {


/**
 * \ingroup object-tests
 *  Type traits test
 */
class TypeTraitsTestCase : public TestCase
{
public:
  /** Constructor. */
  TypeTraitsTestCase ();
  /** Destructor. */
  virtual ~TypeTraitsTestCase ()
  {}

private:
  virtual void DoRun (void);
};

TypeTraitsTestCase::TypeTraitsTestCase (void)
  : TestCase ("Check type traits")
{}

void
TypeTraitsTestCase::DoRun (void)
{
  NS_TEST_ASSERT_MSG_EQ
    (TypeTraits<void (TypeTraitsTestCase::*)(void)>::IsPointerToMember, 1,
    "Check pointer to member function (void)");
  NS_TEST_ASSERT_MSG_EQ
    (TypeTraits<void (TypeTraitsTestCase::*)(void) const>::IsPointerToMember, 1,
    "Check pointer to member function (void) const");
  NS_TEST_ASSERT_MSG_EQ
    (TypeTraits<void (TypeTraitsTestCase::*)(int)>::IsPointerToMember, 1,
    "Check pointer to member function (int)");
  NS_TEST_ASSERT_MSG_EQ
    (TypeTraits<void (TypeTraitsTestCase::*)(int) const>::IsPointerToMember, 1,
    "Check pointer to member function (int) const");
  NS_TEST_ASSERT_MSG_EQ
    (TypeTraits<void (TypeTraitsTestCase::*)(void) const>::PointerToMemberTraits::nArgs, 0,
    "Check number of arguments for pointer to member function (void) const");
  NS_TEST_ASSERT_MSG_EQ
    (TypeTraits<void (TypeTraitsTestCase::*)(int) const>::PointerToMemberTraits::nArgs, 1,
    "Check number of arguments for pointer to member function (int) const");
}

/**
 * \ingroup object-tests
 *  Type traits test suite
 */
class TypeTraitsTestSuite : public TestSuite
{
public:
  /** Constructor. */
  TypeTraitsTestSuite ();
};

TypeTraitsTestSuite::TypeTraitsTestSuite ()
  : TestSuite ("type-traits")
{
  AddTestCase (new TypeTraitsTestCase);
}

/**
 * \ingroup object-tests
 * TypeTraitsTestSuite instance variable.
 */
static TypeTraitsTestSuite g_typeTraitsTestSuite;


}    // namespace tests

}  // namespace ns3
