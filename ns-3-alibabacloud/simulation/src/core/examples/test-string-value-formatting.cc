/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Tom Henderson
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

#include "ns3/core-module.h"

/**
 * \defgroup string-value-formatting
 * Check that StringValue parses complex values correctly.
 */

/**
 * \file
 * \ingroup core-examples
 * \ingroup string-value-formatting
 * Check that StringValue parses complex values correctly.
 * \todo This should really be turned into a TestSuite
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TestStringValueFormatting");

namespace {

/**
 * \ingroup string-value-formatting
 *
 * StringValue formatting example test object.
 *
 * We use an attribute containing a pointer to a random variable
 * to stress StringValue.
 */
class FormattingTestObject : public Object
{
public:
  /**
   * \brief Register this type and get the TypeId.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /** Default constructor */
  FormattingTestObject ();
  /**
   * Get the test variable
   * \returns the test variable
   */
  Ptr<RandomVariableStream> GetTestVariable (void) const;

private:
  Ptr<RandomVariableStream> m_testVariable; //!< The test variable
};

NS_OBJECT_ENSURE_REGISTERED (FormattingTestObject);

TypeId
FormattingTestObject::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FormattingTestObject")
    .SetParent<Object> ()
    .AddConstructor<FormattingTestObject> ()
    .AddAttribute ("OnTime", "A RandomVariableStream used to pick the duration of the 'On' state.",
                   StringValue ("ns3::ConstantRandomVariable[Constant=1.0]"),
                   MakePointerAccessor (&FormattingTestObject::m_testVariable),
                   MakePointerChecker <RandomVariableStream>())
  ;
  return tid;
}

FormattingTestObject::FormattingTestObject ()
{}

Ptr<RandomVariableStream>
FormattingTestObject::GetTestVariable (void) const
{
  return m_testVariable;
}

/**
 * \ingroup string-value-formatting
 *
 * StringValue formatting example test helper class.
 */
class FormattingTestObjectHelper
{
public:
  /** Default constructor */
  FormattingTestObjectHelper ();
  /**
   * Set an attribute by name
   * \param name the attribute
   * \param value the attribute value
   */
  void SetAttribute (std::string name, const AttributeValue &value);
  /**
   * Create an Object as configured by SetAttribute
   * \returns the newly created Object
   */
  Ptr<Object> CreateFromFactory (void);

private:
  ObjectFactory m_factory; //!< Object factory
};

FormattingTestObjectHelper::FormattingTestObjectHelper ()
{
  m_factory.SetTypeId (FormattingTestObject::GetTypeId ());
}

void
FormattingTestObjectHelper::SetAttribute (std::string name, const AttributeValue &value)
{
  m_factory.Set (name, value);
}

Ptr<Object>
FormattingTestObjectHelper::CreateFromFactory (void)
{
  return m_factory.Create ();
}

}  // unnamed namespace


int
main (int argc, char *argv[])
{
  // CreateObject parsing
  Ptr<FormattingTestObject> obj = CreateObject<FormattingTestObject> ();
  obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable"));
  obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=0.]"));
  obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=0.|Max=1.]"));
  obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=50.|Max=100.]"));

  Ptr<RandomVariableStream> rvStream = obj->GetTestVariable ();
  // Either GetObject () or DynamicCast may be used to get subclass pointer
  Ptr<UniformRandomVariable> uniformStream = rvStream->GetObject<UniformRandomVariable> ();
  NS_ASSERT (uniformStream);

  // Check that the last setting of Min to 50 and Max to 100 worked
  DoubleValue val;
  uniformStream->GetAttribute ("Min", val);
  NS_ASSERT_MSG (val.Get () == 50, "Minimum not set to 50");
  uniformStream->GetAttribute ("Max", val);
  NS_ASSERT_MSG (val.Get () == 100, "Maximum not set to 100");


  // The following malformed values should result in an error exit
  // if uncommented

  // Attribute doesn't exist
  //obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[A=0.]"));
  // Missing right bracket
  //obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=0."));
  // Comma delimiter fails
  //obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=0.,Max=1.]"));
  // Incomplete specification
  //obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=0.|Max=]"));
  // Incomplete specification
  //obj->SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=0.|Max]"));

  // ObjectFactory parsing
  FormattingTestObjectHelper formattingHelper;
  formattingHelper.SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=30.|Max=60.0]"));
  // Attribute doesn't exist
  //formattingHelper.SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[A=0.]"));
  // Missing right bracket
  //formattingHelper.SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=30."));
  // Comma delimiter fails
  //formattingHelper.SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=30.,Max=60.]"));
  // Incomplete specification
  //formattingHelper.SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=30.|Max=]"));
  // Incomplete specification
  //formattingHelper.SetAttribute ("OnTime", StringValue ("ns3::UniformRandomVariable[Min=30.|Max]"));

  // verify that creation occurs correctly
  Ptr<Object> outputObj = formattingHelper.CreateFromFactory ();
  Ptr<FormattingTestObject> fto = DynamicCast<FormattingTestObject> (outputObj);
  NS_ASSERT_MSG (fto, "object creation failed");
  rvStream = fto->GetTestVariable ();
  uniformStream = rvStream->GetObject<UniformRandomVariable> ();
  NS_ASSERT (uniformStream);
  // Check that the last setting of Min to 30 and Max to 60 worked
  uniformStream->GetAttribute ("Min", val);
  NS_ASSERT_MSG (val.Get () == 30, "Minimum not set to 30");
  uniformStream->GetAttribute ("Max", val);
  NS_ASSERT_MSG (val.Get () == 60, "Maximum not set to 60");
}
