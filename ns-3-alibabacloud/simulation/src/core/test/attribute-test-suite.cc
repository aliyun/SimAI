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

#include "ns3/test.h"
#include "ns3/object.h"
#include "ns3/boolean.h"
#include "ns3/integer.h"
#include "ns3/uinteger.h"
#include "ns3/config.h"
#include "ns3/enum.h"
#include "ns3/string.h"
#include "ns3/random-variable-stream.h"
#include "ns3/double.h"
#include "ns3/object-vector.h"
#include "ns3/object-map.h"
#include "ns3/traced-value.h"
#include "ns3/callback.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/pointer.h"
#include "ns3/object-factory.h"
#include "ns3/nstime.h"

using namespace ns3;

namespace ns3 {

/**
 * \file
 * \ingroup attribute-tests
 * Attribute test suite
 */

/**
 * \ingroup core-tests
 * \defgroup attribute-tests Attribute tests
 */

/**
 * \ingroup attribute-tests
 * 
 * Test class for TracedValue callbacks attributes.
 * \see attribute_ValueClassTest
 */
class ValueClassTest
{
public:
  ValueClassTest ()
  {}

  /**
   * TracedValue callback signature for ValueClassTest
   *
   * \param [in] oldValue original value of the traced variable
   * \param [in] newValue new value of the traced variable
   */
  typedef void (* TracedValueCallback)(const ValueClassTest oldValue,
                                       const ValueClassTest newValue);
};

/**
 * Operator not equal.
 * \param a The left operand.
 * \param b The right operand.
 * \return always true.
 */
bool operator != ([[maybe_unused]] const ValueClassTest &a, [[maybe_unused]] const ValueClassTest &b)
{
  return true;
}
/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] v The ValueClassTest object.
 * \returns The reference to the output stream.
 */
std::ostream & operator << (std::ostream &os, [[maybe_unused]] ValueClassTest v)
{
  return os;
}
/**
 * \brief Stream extraction operator.
 *
 * \param [in] is The reference to the input stream.
 * \param [out] v The ValueClassTest object.
 * \returns The reference to the input stream.
 */
std::istream & operator >> (std::istream &is, [[maybe_unused]] ValueClassTest &v)
{
  return is;
}

ATTRIBUTE_HELPER_HEADER (ValueClassTest);
ATTRIBUTE_HELPER_CPP (ValueClassTest);

} // end of ns3 namespace

/**
 * \ingroup attribute-tests
 * 
 * Simple class derived from ns3::Object, used to check attribute constructors.
 */
class Derived : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void)
  {
    static TypeId tid = TypeId ("ns3::Derived")
      .AddConstructor<Derived> ()
      .SetParent<Object> ()
    ;
    return tid;
  }
  Derived ()
  {}
};

NS_OBJECT_ENSURE_REGISTERED (Derived);

/**
 * \ingroup attribute-tests
 * 
 * Class used to check attributes.
 */
class AttributeObjectTest : public Object
{
public:
  /// Test enumerator.
  enum Test_e
  {
    TEST_A, //!< Test value A.
    TEST_B, //!< Test value B.
    TEST_C  //!< Test value C.
  };
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void)
  {
    static TypeId tid = TypeId ("ns3::AttributeObjectTest")
      .AddConstructor<AttributeObjectTest> ()
      .SetParent<Object> ()
      .HideFromDocumentation ()
      .AddAttribute ("TestBoolName", "help text",
                     BooleanValue (false),
                     MakeBooleanAccessor (&AttributeObjectTest::m_boolTest),
                     MakeBooleanChecker ())
      .AddAttribute ("TestBoolA", "help text",
                     BooleanValue (false),
                     MakeBooleanAccessor (&AttributeObjectTest::DoSetTestA,
                                          &AttributeObjectTest::DoGetTestA),
                     MakeBooleanChecker ())
      .AddAttribute ("TestInt16", "help text",
                     IntegerValue (-2),
                     MakeIntegerAccessor (&AttributeObjectTest::m_int16),
                     MakeIntegerChecker<int16_t> ())
      .AddAttribute ("TestInt16WithBounds", "help text",
                     IntegerValue (-2),
                     MakeIntegerAccessor (&AttributeObjectTest::m_int16WithBounds),
                     MakeIntegerChecker<int16_t> (-5, 10))
      .AddAttribute ("TestInt16SetGet", "help text",
                     IntegerValue (6),
                     MakeIntegerAccessor (&AttributeObjectTest::DoSetInt16,
                                          &AttributeObjectTest::DoGetInt16),
                     MakeIntegerChecker<int16_t> ())
      .AddAttribute ("TestUint8", "help text",
                     UintegerValue (1),
                     MakeUintegerAccessor (&AttributeObjectTest::m_uint8),
                     MakeUintegerChecker<uint8_t> ())
      .AddAttribute ("TestEnum", "help text",
                     EnumValue (TEST_A),
                     MakeEnumAccessor (&AttributeObjectTest::m_enum),
                     MakeEnumChecker (TEST_A, "TestA",
                                      TEST_B, "TestB",
                                      TEST_C, "TestC"))
      .AddAttribute ("TestEnumSetGet", "help text",
                     EnumValue (TEST_B),
                     MakeEnumAccessor (&AttributeObjectTest::DoSetEnum,
                                       &AttributeObjectTest::DoGetEnum),
                     MakeEnumChecker (TEST_A, "TestA",
                                      TEST_B, "TestB",
                                      TEST_C, "TestC"))
      .AddAttribute ("TestRandom", "help text",
                     StringValue ("ns3::ConstantRandomVariable[Constant=1.0]"),
                     MakePointerAccessor (&AttributeObjectTest::m_random),
                     MakePointerChecker <RandomVariableStream>())
      .AddAttribute ("TestFloat", "help text",
                     DoubleValue (-1.1),
                     MakeDoubleAccessor (&AttributeObjectTest::m_float),
                     MakeDoubleChecker<float> ())
      .AddAttribute ("TestVector1", "help text",
                     ObjectVectorValue (),
                     MakeObjectVectorAccessor (&AttributeObjectTest::m_vector1),
                     MakeObjectVectorChecker<Derived> ())
      .AddAttribute ("TestVector2", "help text",
                     ObjectVectorValue (),
                     MakeObjectVectorAccessor (&AttributeObjectTest::DoGetVectorN,
                                               &AttributeObjectTest::DoGetVector),
                     MakeObjectVectorChecker<Derived> ())
      .AddAttribute ("TestMap1", "help text",
                     ObjectMapValue (),
                     MakeObjectMapAccessor (&AttributeObjectTest::m_map1),
                     MakeObjectMapChecker<Derived> ())
      .AddAttribute ("IntegerTraceSource1", "help text",
                     IntegerValue (-2),
                     MakeIntegerAccessor (&AttributeObjectTest::m_intSrc1),
                     MakeIntegerChecker<int8_t> ())
      .AddAttribute ("IntegerTraceSource2", "help text",
                     IntegerValue (-2),
                     MakeIntegerAccessor (&AttributeObjectTest::DoSetIntSrc,
                                          &AttributeObjectTest::DoGetIntSrc),
                     MakeIntegerChecker<int8_t> ())
      .AddAttribute ("UIntegerTraceSource", "help text",
                     UintegerValue (2),
                     MakeUintegerAccessor (&AttributeObjectTest::m_uintSrc),
                     MakeIntegerChecker<uint8_t> ())
      .AddAttribute ("DoubleTraceSource", "help text",
                     DoubleValue (2),
                     MakeDoubleAccessor (&AttributeObjectTest::m_doubleSrc),
                     MakeDoubleChecker<double> ())
      .AddAttribute ("BoolTraceSource", "help text",
                     BooleanValue (false),
                     MakeBooleanAccessor (&AttributeObjectTest::m_boolSrc),
                     MakeBooleanChecker ())
      .AddAttribute ("EnumTraceSource", "help text",
                     EnumValue (false),
                     MakeEnumAccessor (&AttributeObjectTest::m_enumSrc),
                     MakeEnumChecker (TEST_A, "TestA"))
      .AddAttribute ("ValueClassSource", "help text",
                     ValueClassTestValue (ValueClassTest ()),
                     MakeValueClassTestAccessor (&AttributeObjectTest::m_valueSrc),
                     MakeValueClassTestChecker ())
      .AddTraceSource ("Source1", "help test",
                       MakeTraceSourceAccessor (&AttributeObjectTest::m_intSrc1),
                       "ns3::TracedValueCallback::Int8")
      .AddTraceSource ("Source2", "help text",
                       MakeTraceSourceAccessor (&AttributeObjectTest::m_cb),
                       "ns3::AttributeObjectTest::NumericTracedCallback")
      .AddTraceSource ("ValueSource", "help text",
                       MakeTraceSourceAccessor (&AttributeObjectTest::m_valueSrc),
                       "ns3::ValueClassTest::TracedValueCallback")
      .AddAttribute ("Pointer", "help text",
                     PointerValue (),
                     MakePointerAccessor (&AttributeObjectTest::m_ptr),
                     MakePointerChecker<Derived> ())
      .AddAttribute ("PointerInitialized", "help text",
                     StringValue ("ns3::Derived"),
                     MakePointerAccessor (&AttributeObjectTest::m_ptrInitialized),
                     MakePointerChecker<Derived> ())
      .AddAttribute ("PointerInitialized2", "help text",
                     StringValue ("ns3::Derived[]"),
                     MakePointerAccessor (&AttributeObjectTest::m_ptrInitialized2),
                     MakePointerChecker<Derived> ())
      .AddAttribute ("Callback", "help text",
                     CallbackValue (),
                     MakeCallbackAccessor (&AttributeObjectTest::m_cbValue),
                     MakeCallbackChecker ())
      .AddAttribute ("TestTimeWithBounds", "help text",
                     TimeValue (Seconds (-2)),
                     MakeTimeAccessor (&AttributeObjectTest::m_timeWithBounds),
                     MakeTimeChecker (Seconds (-5), Seconds (10)))
      .AddAttribute ("TestDeprecated", "help text",
                     BooleanValue (false),
                     MakeBooleanAccessor (&AttributeObjectTest::m_boolTestDeprecated),
                     MakeBooleanChecker (),
                     TypeId::DEPRECATED,
                     "DEPRECATED test working.")
    ;

    return tid;
  }

  AttributeObjectTest (void)
  {
  }

  virtual ~AttributeObjectTest (void)
  {}

  /// Add an object to the first vector.
  void AddToVector1 (void)
  {
    m_vector1.push_back (CreateObject<Derived> ());
  }
  /// Add an object to the second vector.
  void AddToVector2 (void)
  {
    m_vector2.push_back (CreateObject<Derived> ());
  }

  /**
   * Adds an object to the first map.
   * \param i The index to assign to the object.
   */
  void AddToMap1 (uint32_t i)
  {
    m_map1.insert (std::pair <uint32_t, Ptr<Derived> > (i, CreateObject<Derived> ()));
  }

  /**
   * Invoke the m_cb callback.
   * \param a The first argument of the callback.
   * \param b The second argument of the callback.
   * \param c The third argument of the callback.
   */
  void InvokeCb (double a, int b, float c)
  {
    m_cb (a,b,c);
  }

  /**
   * Invoke the m_cbValue callback.
   * \param a The argument of the callback.
   */
  void InvokeCbValue (int8_t a)
  {
    if (!m_cbValue.IsNull ())
      {
        m_cbValue (a);
      }
  }

private:
  /**
   * Set the m_boolTestA value.
   * \param v The value to set.
   */
  void DoSetTestA (bool v)
  {
    m_boolTestA = v;
  }
  /**
   * Get the m_boolTestA value.
   * \return the value of m_boolTestA.
   */
  bool DoGetTestA (void) const
  {
    return m_boolTestA;
  }
  /**
   * Get the m_int16SetGet value.
   * \return the value of m_int16SetGet.
   */
  int16_t DoGetInt16 (void) const
  {
    return m_int16SetGet;
  }
  /**
   * Set the m_int16SetGet value.
   * \param v The value to set.
   */
  void DoSetInt16 (int16_t v)
  {
    m_int16SetGet = v;
  }
  /**
   * Get the length of m_vector2.
   * \return the vector size.
   */
  std::size_t DoGetVectorN (void) const
  {
    return m_vector2.size ();
  }
  /**
   * Get the i-th item of m_vector2.
   * \param i The index of the element to get.
   * \return i-th item of m_vector2.
   */
  Ptr<Derived> DoGetVector (std::size_t i) const
  {
    return m_vector2[i];
  }
  /**
   * Set the m_intSrc2 value.
   * \param v The value to set.
   * \return true.
   */
  bool DoSetIntSrc (int8_t v)
  {
    m_intSrc2 = v;
    return true;
  }
  /**
   * Get the m_intSrc2 value.
   * \return the value of m_intSrc2.
   */
  int8_t DoGetIntSrc (void) const
  {
    return m_intSrc2;
  }
  /**
   * Set the m_enumSetGet value.
   * \param v The value to set.
   * \return true.
   */
  bool DoSetEnum (Test_e v)
  {
    m_enumSetGet = v;
    return true;
  }
  /**
   * Get the m_enumSetGet value.
   * \return the value of m_enumSetGet.
   */
  Test_e DoGetEnum (void) const
  {
    return m_enumSetGet;
  }

  bool m_boolTestA;   //!< Boolean test A.
  bool m_boolTest;    //!< Boolean test.
  bool m_boolTestDeprecated;  //!< Boolean test deprecated.
  int16_t m_int16;            //!< 16-bit integer.
  int16_t m_int16WithBounds;  //!< 16-bit integer with bounds.
  int16_t m_int16SetGet;      //!< 16-bit integer set-get.
  uint8_t m_uint8;            //!< 8-bit integer.
  float m_float;              //!< float.
  enum Test_e m_enum;         //!< Enum.
  enum Test_e m_enumSetGet;   //!< Enum set-get.
  Ptr<RandomVariableStream> m_random;   //!< Random number generator.
  std::vector<Ptr<Derived> > m_vector1; //!< First vector of derived objects.
  std::vector<Ptr<Derived> > m_vector2; //!< Second vector of derived objects.
  std::map <uint32_t, Ptr<Derived> > m_map1;   //!< Map of uint32_t, derived objects.
  Callback<void,int8_t> m_cbValue;      //!< Callback accepting an integer.
  TracedValue<int8_t> m_intSrc1;        //!< First int8_t Traced value.
  TracedValue<int8_t> m_intSrc2;        //!< Second int8_t Traced value.

  /// Traced callbacks for (double, int, float) values.
  typedef void (* NumericTracedCallback) (double, int, float);
  TracedCallback<double, int, float> m_cb;  //!< TracedCallback (double, int, float).
  TracedValue<ValueClassTest> m_valueSrc;   //!< ValueClassTest Traced value.
  Ptr<Derived> m_ptr;               //!< Pointer to Derived class.
  Ptr<Derived> m_ptrInitialized;    //!< Pointer to Derived class.
  Ptr<Derived> m_ptrInitialized2;   //!< Pointer to Derived class.
  TracedValue<uint8_t> m_uintSrc;   //!< uint8_t Traced value.
  TracedValue<enum Test_e> m_enumSrc; //!< enum Traced value.
  TracedValue<double> m_doubleSrc;  //!< double Traced value.
  TracedValue<bool> m_boolSrc;      //!< bool Traced value.
  Time m_timeWithBounds;            //!< Time with bounds
};

NS_OBJECT_ENSURE_REGISTERED (AttributeObjectTest);

/**
 * \ingroup attribute-tests
 * 
 * \brief Test case template used for generic Attribute Value types -- used to make
 * sure that Attributes work as expected.
 */
template <typename T>
class AttributeTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  AttributeTestCase (std::string description);
  virtual ~AttributeTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Check the attribute path and value.
   * \param p The object to test.
   * \param attributeName The attribute name.
   * \param expectedString The expected attribute name.
   * \param expectedValue The expected attribute value.
   * \return true if everything is as expected.
   */
  bool CheckGetCodePaths (Ptr<Object> p, std::string attributeName, std::string expectedString, T expectedValue);
};

template <typename T>
AttributeTestCase<T>::AttributeTestCase (std::string description)
  : TestCase (description)
{}

template <typename T>
AttributeTestCase<T>::~AttributeTestCase ()
{}

template <typename T> bool
AttributeTestCase<T>::CheckGetCodePaths (
  Ptr<Object> p,
  std::string attributeName,
  std::string expectedString,
  T expectedValue)
{
  StringValue stringValue;
  T actualValue;

  //
  // Get an Attribute value through its StringValue representation.
  //
  bool ok1 = p->GetAttributeFailSafe (attributeName.c_str (), stringValue);
  bool ok2 = stringValue.Get () == expectedString;

  //
  // Get the existing boolean value through its particular type representation.
  //
  bool ok3 = p->GetAttributeFailSafe (attributeName.c_str (), actualValue);
  bool ok4 = expectedValue.Get () == actualValue.Get ();

  return ok1 && ok2 && ok3 && ok4;
}

// ===========================================================================
// The actual Attribute type test cases are specialized for each Attribute type
// ===========================================================================
template <> void
AttributeTestCase<BooleanValue>::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // Set the default value of the BooleanValue and create an object.  The new
  // default value should stick.
  //
  Config::SetDefault ("ns3::AttributeObjectTest::TestBoolName", StringValue ("true"));
  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  ok = CheckGetCodePaths (p, "TestBoolName", "true", BooleanValue (true));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by default value");

  std::string expected ("Attribute 'TestDeprecated' is deprecated: DEPRECATED test working.\n");
  // Temporarily redirect std::cerr to a stringstream
  std::stringstream buffer;
  std::streambuf *oldBuffer = std::cerr.rdbuf (buffer.rdbuf());
  // Cause the deprecation warning to be sent to the stringstream
  Config::SetDefault ("ns3::AttributeObjectTest::TestDeprecated", BooleanValue (true));

  // Compare the obtained actual string with the expected string.
  NS_TEST_ASSERT_MSG_EQ (buffer.str (), expected, "Deprecated attribute not working");
  // Restore cerr to its original stream buffer
  std::cerr.rdbuf (oldBuffer);



  //
  // Set the default value of the BooleanValue the other way and create an object.
  // The new default value should stick.
  //
  Config::SetDefaultFailSafe ("ns3::AttributeObjectTest::TestBoolName", StringValue ("false"));

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  ok = CheckGetCodePaths (p, "TestBoolName", "false", BooleanValue (false));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not et properly by default value");

  //
  // Set the BooleanValue Attribute to true via SetAttributeFailSafe path.
  //
  ok = p->SetAttributeFailSafe ("TestBoolName", StringValue ("true"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() \"TestBoolName\" to true");

  ok = CheckGetCodePaths (p, "TestBoolName", "true", BooleanValue (true));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via StringValue");

  //
  // Set the BooleanValue to false via SetAttributeFailSafe path.
  //
  ok = p->SetAttributeFailSafe ("TestBoolName", StringValue ("false"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() \"TestBoolName\" to false");

  ok = CheckGetCodePaths (p, "TestBoolName", "false", BooleanValue (false));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via StringValue");

  //
  // Create an object using
  //
  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // The previous object-based tests checked access directly.  Now check through
  // setter and getter.  The code here looks the same, but the underlying
  // attribute is declared differently in the object.  First make sure we can set
  // to true.
  //
  ok = p->SetAttributeFailSafe ("TestBoolA", StringValue ("true"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() a boolean value to true");

  ok = CheckGetCodePaths (p, "TestBoolA", "true", BooleanValue (true));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() (getter/setter) via StringValue");

  //
  // Now Set the BooleanValue to false via the setter.
  //
  ok = p->SetAttributeFailSafe ("TestBoolA", StringValue ("false"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() a boolean value to false");

  ok = CheckGetCodePaths (p, "TestBoolA", "false", BooleanValue (false));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() (getter/setter) via StringValue");
}

template <> void
AttributeTestCase<IntegerValue>::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // When the object is first created, the Attribute should have the default
  // value.
  //
  ok = CheckGetCodePaths (p, "TestInt16", "-2", IntegerValue (-2));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by default value");

  //
  // Set the Attribute to a negative value through a StringValue.
  //
  ok = p->SetAttributeFailSafe ("TestInt16", StringValue ("-5"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via StringValue to -5");

  ok = CheckGetCodePaths (p, "TestInt16", "-5", IntegerValue (-5));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via StringValue");

  //
  // Set the Attribute to a positive value through a StringValue.
  //
  ok = p->SetAttributeFailSafe ("TestInt16", StringValue ("+2"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via StringValue to +2");

  ok = CheckGetCodePaths (p, "TestInt16", "2", IntegerValue (2));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via StringValue");

  //
  // Set the Attribute to the most negative value of the signed 16-bit range.
  //
  ok = p->SetAttributeFailSafe ("TestInt16", StringValue ("-32768"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via StringValue to -32768");

  ok = CheckGetCodePaths (p, "TestInt16", "-32768", IntegerValue (-32768));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() (most negative) via StringValue");

  //
  // Try to set the Attribute past the most negative value of the signed 16-bit
  // range and make sure the underlying attribute is unchanged.
  //
  ok = p->SetAttributeFailSafe ("TestInt16", StringValue ("-32769"));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via StringValue to -32769");

  ok = CheckGetCodePaths (p, "TestInt16", "-32768", IntegerValue (-32768));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");

  //
  // Set the Attribute to the most positive value of the signed 16-bit range.
  //
  ok = p->SetAttributeFailSafe ("TestInt16", StringValue ("32767"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via StringValue to 32767");

  ok = CheckGetCodePaths (p, "TestInt16", "32767", IntegerValue (32767));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() (most positive) via StringValue");

  //
  // Try to set the Attribute past the most positive value of the signed 16-bit
  // range and make sure the underlying attribute is unchanged.
  //
  ok = p->SetAttributeFailSafe ("TestInt16", StringValue ("32768"));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via StringValue to 32768");

  ok = CheckGetCodePaths (p, "TestInt16", "32767", IntegerValue (32767));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");

  //
  // Attributes can have limits other than the intrinsic limits of the
  // underlying data types.  These limits are specified in the Object.
  //
  ok = p->SetAttributeFailSafe ("TestInt16WithBounds", IntegerValue (10));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to 10");

  ok = CheckGetCodePaths (p, "TestInt16WithBounds", "10", IntegerValue (10));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() (positive limit) via StringValue");

  //
  // Set the Attribute past the positive limit.
  //
  ok = p->SetAttributeFailSafe ("TestInt16WithBounds", IntegerValue (11));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via IntegerValue to 11");

  ok = CheckGetCodePaths (p, "TestInt16WithBounds", "10", IntegerValue (10));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");

  //
  // Set the Attribute at the negative limit.
  //
  ok = p->SetAttributeFailSafe ("TestInt16WithBounds", IntegerValue (-5));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to -5");

  ok = CheckGetCodePaths (p, "TestInt16WithBounds", "-5", IntegerValue (-5));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() (negative limit) via StringValue");

  //
  // Set the Attribute past the negative limit.
  //
  ok = p->SetAttributeFailSafe ("TestInt16WithBounds", IntegerValue (-6));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via IntegerValue to -6");

  ok = CheckGetCodePaths (p, "TestInt16WithBounds", "-5", IntegerValue (-5));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");
}

template <> void
AttributeTestCase<UintegerValue>::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // When the object is first created, the Attribute should have the default
  // value.
  //
  ok = CheckGetCodePaths (p, "TestUint8", "1", UintegerValue (1));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by default value");

  //
  // Set the Attribute to zero.
  //
  ok = p->SetAttributeFailSafe ("TestUint8", UintegerValue (0));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() to 0");

  ok = CheckGetCodePaths (p, "TestUint8", "0", UintegerValue (0));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via StringValue");

  //
  // Set the Attribute to the most positive value of the unsigned 8-bit range.
  //
  ok = p->SetAttributeFailSafe ("TestUint8", UintegerValue (255));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() to 255");

  ok = CheckGetCodePaths (p, "TestUint8", "255", UintegerValue (255));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() (positive limit) via UintegerValue");

  //
  // Try and set the Attribute past the most positive value of the unsigned
  // 8-bit range.
  //
  ok = p->SetAttributeFailSafe ("TestUint8", UintegerValue (256));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() to 256");

  ok = CheckGetCodePaths (p, "TestUint8", "255", UintegerValue (255));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");

  //
  // Set the Attribute to the most positive value of the unsigned 8-bit range
  // through a StringValue.
  //
  ok = p->SetAttributeFailSafe ("TestUint8", StringValue ("255"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via StringValue to 255");

  ok = CheckGetCodePaths (p, "TestUint8", "255", UintegerValue (255));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via StringValue");

  //
  // Try and set the Attribute past the most positive value of the unsigned
  // 8-bit range through a StringValue.
  //
  ok = p->SetAttributeFailSafe ("TestUint8", StringValue ("256"));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via StringValue to 256");

  ok = CheckGetCodePaths (p, "TestUint8", "255", UintegerValue (255));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");

  //
  // Try to set the Attribute to a negative StringValue.
  //
  ok = p->SetAttributeFailSafe ("TestUint8", StringValue ("-1"));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via StringValue to -1");
}

template <> void
AttributeTestCase<DoubleValue>::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // When the object is first created, the Attribute should have the default
  // value.
  //
  ok = CheckGetCodePaths (p, "TestFloat", "-1.1", DoubleValue ((float)-1.1));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by default value");

  //
  // Set the Attribute.
  //
  ok = p->SetAttributeFailSafe ("TestFloat", DoubleValue ((float)2.3));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() to 2.3");

  ok = CheckGetCodePaths (p, "TestFloat", "2.3", DoubleValue ((float)2.3));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via DoubleValue");
}

template <> void
AttributeTestCase<EnumValue>::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // When the object is first created, the Attribute should have the default
  // value.
  //
  ok = CheckGetCodePaths (p, "TestEnum", "TestA", EnumValue (AttributeObjectTest::TEST_A));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by default value");

  //
  // Set the Attribute using the EnumValue type.
  //
  ok = p->SetAttributeFailSafe ("TestEnum", EnumValue (AttributeObjectTest::TEST_C));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() to TEST_C");

  ok = CheckGetCodePaths (p, "TestEnum", "TestC", EnumValue (AttributeObjectTest::TEST_C));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via EnumValue");

  //
  // When the object is first created, the Attribute should have the default
  // value.
  //
  ok = CheckGetCodePaths (p, "TestEnumSetGet", "TestB", EnumValue (AttributeObjectTest::TEST_B));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by default value");

  //
  // Set the Attribute using the EnumValue type.
  //
  ok = p->SetAttributeFailSafe ("TestEnumSetGet", EnumValue (AttributeObjectTest::TEST_C));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() to TEST_C");

  ok = CheckGetCodePaths (p, "TestEnumSetGet", "TestC", EnumValue (AttributeObjectTest::TEST_C));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via EnumValue");

  //
  // Set the Attribute using the StringValue type.
  //
  ok = p->SetAttributeFailSafe ("TestEnum", StringValue ("TestB"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() to TEST_B");

  ok = CheckGetCodePaths (p, "TestEnum", "TestB", EnumValue (AttributeObjectTest::TEST_B));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe() via StringValue");

  //
  // Try to set the Attribute to a bogus enum using the StringValue type
  // throws a fatal error.
  //
  //  ok = p->SetAttributeFailSafe ("TestEnum", StringValue ("TestD"));
  //  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() to TEST_D"); //

  ok = CheckGetCodePaths (p, "TestEnum", "TestB", EnumValue (AttributeObjectTest::TEST_B));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");

  //
  // Try to set the Attribute to a bogus enum using an integer implicit conversion
  // and make sure the underlying value doesn't change.
  //
  ok = p->SetAttributeFailSafe ("TestEnum", EnumValue (5));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() to 5");

  ok = CheckGetCodePaths (p, "TestEnum", "TestB", EnumValue (AttributeObjectTest::TEST_B));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");
}

template <> void
AttributeTestCase<TimeValue>::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  // The test vectors assume ns resolution
  Time::SetResolution (Time::NS);

  //
  // Set value
  //
  ok = p->SetAttributeFailSafe ("TestTimeWithBounds", TimeValue (Seconds (5)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via TimeValue to 5s");

  ok = CheckGetCodePaths (p, "TestTimeWithBounds", "+5e+09ns", TimeValue (Seconds (5)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe(5s) via TimeValue");

  ok = p->SetAttributeFailSafe ("TestTimeWithBounds", StringValue ("3s"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via TimeValue to 3s");

  ok = CheckGetCodePaths (p, "TestTimeWithBounds", "+3e+09ns", TimeValue (Seconds (3)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe(3s) via StringValue");


  //
  // Attributes can have limits other than the intrinsic limits of the
  // underlying data types.  These limits are specified in the Object.
  //

  //
  // Set the Attribute at the positive limit
  //
  ok = p->SetAttributeFailSafe ("TestTimeWithBounds", TimeValue (Seconds (10)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via TimeValue to 10s");

  ok = CheckGetCodePaths (p, "TestTimeWithBounds", "+1e+10ns", TimeValue (Seconds (10)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe(10s [positive limit]) via StringValue");

  //
  // Set the Attribute past the positive limit.
  //
  ok = p->SetAttributeFailSafe ("TestTimeWithBounds", TimeValue (Seconds (11)));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via TimeValue to 11s [greater than positive limit]");

  ok = CheckGetCodePaths (p, "TestTimeWithBounds", "+1e+10ns", TimeValue (Seconds (10)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");

  //
  // Set the Attribute at the negative limit.
  //
  ok = p->SetAttributeFailSafe ("TestTimeWithBounds", TimeValue (Seconds (-5)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via TimeValue to -5s");

  ok = CheckGetCodePaths (p, "TestTimeWithBounds", "-5e+09ns", TimeValue (Seconds (-5)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Attribute not set properly by SetAttributeFailSafe(-5s [negative limit]) via StringValue");

  //
  // Set the Attribute past the negative limit.
  //
  ok = p->SetAttributeFailSafe ("TestTimeWithBounds", TimeValue (Seconds (-6)));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via TimeValue to -6s");

  ok = CheckGetCodePaths (p, "TestTimeWithBounds", "-5e+09ns", TimeValue (Seconds (-5)));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Error in SetAttributeFailSafe() but value changes");
}

/**
 * \ingroup attribute-tests
 *  
 * Test the Attributes of type RandomVariableStream.
 */
class RandomVariableStreamAttributeTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  RandomVariableStreamAttributeTestCase (std::string description);
  virtual ~RandomVariableStreamAttributeTestCase ()
  {}

  /**
   * Invoke the m_cbValue.
   * \param a The value to use on the callback.
   */
  void InvokeCbValue (int8_t a)
  {
    if (!m_cbValue.IsNull ())
      {
        m_cbValue (a);
      }
  }

private:
  virtual void DoRun (void);

  /// Callback used in the test.
  Callback<void,int8_t> m_cbValue;

  /**
   * Function called when the callback is used.
   * \param a The value of the callback.
   */
  void NotifyCallbackValue (int8_t a)
  {
    m_gotCbValue = a;
  }

  int16_t m_gotCbValue; //!< Value used to verify that the callback has been invoked.
};

RandomVariableStreamAttributeTestCase::RandomVariableStreamAttributeTestCase (std::string description)
  : TestCase (description)
{}

void
RandomVariableStreamAttributeTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // Try to set a UniformRandomVariable
  //
  ok = p->SetAttributeFailSafe ("TestRandom", StringValue ("ns3::UniformRandomVariable[Min=0.|Max=1.]"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() a UniformRandomVariable");

  //
  // Try to set a <snicker> ConstantRandomVariable
  //
  ok = p->SetAttributeFailSafe ("TestRandom", StringValue ("ns3::ConstantRandomVariable[Constant=1.0]"));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() a ConstantRandomVariable");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief Test case for Object Vector Attributes.
 * 
 * Generic nature is pretty much lost here, so we just break the class out.
 */
class ObjectVectorAttributeTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  ObjectVectorAttributeTestCase (std::string description);
  virtual ~ObjectVectorAttributeTestCase ()
  {}

private:
  virtual void DoRun (void);
};

ObjectVectorAttributeTestCase::ObjectVectorAttributeTestCase (std::string description)
  : TestCase (description)
{}

void
ObjectVectorAttributeTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  ObjectVectorValue vector;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // When the object is first created, the Attribute should have no items in
  // the vector.
  //
  p->GetAttribute ("TestVector1", vector);
  NS_TEST_ASSERT_MSG_EQ (vector.GetN (), 0, "Initial count of ObjectVectorValue \"TestVector1\" should be zero");

  //
  // Adding to the attribute shouldn't affect the value we already have.
  //
  p->AddToVector1 ();
  NS_TEST_ASSERT_MSG_EQ (vector.GetN (), 0, "Initial count of ObjectVectorValue \"TestVector1\" should still be zero");

  //
  // Getting the attribute again should update the value.
  //
  p->GetAttribute ("TestVector1", vector);
  NS_TEST_ASSERT_MSG_EQ (vector.GetN (), 1, "ObjectVectorValue \"TestVector1\" should be incremented");

  //
  // Get the Object pointer from the value.
  //
  Ptr<Object> a = vector.Get (0);
  NS_TEST_ASSERT_MSG_NE (a, 0, "Ptr<Object> from VectorValue \"TestVector1\" is zero");

  //
  // Adding to the attribute shouldn't affect the value we already have.
  //
  p->AddToVector1 ();
  NS_TEST_ASSERT_MSG_EQ (vector.GetN (), 1, "Count of ObjectVectorValue \"TestVector1\" should still be one");

  //
  // Getting the attribute again should update the value.
  //
  p->GetAttribute ("TestVector1", vector);
  NS_TEST_ASSERT_MSG_EQ (vector.GetN (), 2, "ObjectVectorValue \"TestVector1\" should be incremented");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief Test case for Object Map Attributes.
 */
class ObjectMapAttributeTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  ObjectMapAttributeTestCase (std::string description);
  virtual ~ObjectMapAttributeTestCase ()
  {}

private:
  virtual void DoRun (void);
};

ObjectMapAttributeTestCase::ObjectMapAttributeTestCase (std::string description)
  : TestCase (description)
{}

void
ObjectMapAttributeTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  ObjectMapValue map;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // When the object is first created, the Attribute should have no items in
  // the vector.
  //
  p->GetAttribute ("TestMap1", map);
  NS_TEST_ASSERT_MSG_EQ (map.GetN (), 0, "Initial count of ObjectVectorValue \"TestMap1\" should be zero");

  //
  // Adding to the attribute shouldn't affect the value we already have.
  //
  p->AddToMap1 (1);
  NS_TEST_ASSERT_MSG_EQ (map.GetN (), 0, "Initial count of ObjectVectorValue \"TestMap1\" should still be zero");

  //
  // Getting the attribute again should update the value.
  //
  p->GetAttribute ("TestMap1", map);
  NS_TEST_ASSERT_MSG_EQ (map.GetN (), 1, "ObjectVectorValue \"TestMap1\" should be incremented");

  //
  // Get the Object pointer from the value.
  //
  Ptr<Object> a = map.Get (1);
  NS_TEST_ASSERT_MSG_NE (a, 0, "Ptr<Object> from VectorValue \"TestMap1\" is zero");

  //
  // Adding to the attribute shouldn't affect the value we already have.
  //
  p->AddToMap1 (2);
  NS_TEST_ASSERT_MSG_EQ (map.GetN (), 1, "Count of ObjectVectorValue \"TestMap1\" should still be one");

  //
  // Getting the attribute again should update the value.
  //
  p->GetAttribute ("TestMap1", map);
  NS_TEST_ASSERT_MSG_EQ (map.GetN (), 2, "ObjectVectorValue \"TestMap1\" should be incremented");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief Trace sources with value semantics can be used like Attributes, 
 * make sure we can use them that way.
 */
class IntegerTraceSourceAttributeTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  IntegerTraceSourceAttributeTestCase (std::string description);
  virtual ~IntegerTraceSourceAttributeTestCase ()
  {}

private:
  virtual void DoRun (void);
};

IntegerTraceSourceAttributeTestCase::IntegerTraceSourceAttributeTestCase (std::string description)
  : TestCase (description)
{}

void
IntegerTraceSourceAttributeTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  IntegerValue iv;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // When the object is first created, the Attribute should have the default
  // value.
  //
  p->GetAttribute ("IntegerTraceSource1", iv);
  NS_TEST_ASSERT_MSG_EQ (iv.Get (), -2, "Attribute not set properly by default value");

  //
  // Set the Attribute to a positive value through an IntegerValue.
  //
  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (5));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to 5");

  p->GetAttribute ("IntegerTraceSource1", iv);
  NS_TEST_ASSERT_MSG_EQ (iv.Get (), 5, "Attribute not set properly by SetAttributeFailSafe() via IntegerValue");

  //
  // Limits should work.
  //
  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (127));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to 127");

  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (128));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via IntegerValue to 128");

  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (-128));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to -128");

  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (-129));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via IntegerValue to -129");

  //
  // When the object is first created, the Attribute should have the default
  // value.
  //
  p->GetAttribute ("IntegerTraceSource2", iv);
  NS_TEST_ASSERT_MSG_EQ (iv.Get (), -2, "Attribute not set properly by default value");

  //
  // Set the Attribute to a positive value through an IntegerValue.
  //
  ok = p->SetAttributeFailSafe ("IntegerTraceSource2", IntegerValue (5));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to 5");

  p->GetAttribute ("IntegerTraceSource2", iv);
  NS_TEST_ASSERT_MSG_EQ (iv.Get (), 5, "Attribute not set properly by SetAttributeFailSafe() via IntegerValue");

  //
  // Limits should work.
  //
  ok = p->SetAttributeFailSafe ("IntegerTraceSource2", IntegerValue (127));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to 127");

  ok = p->SetAttributeFailSafe ("IntegerTraceSource2", IntegerValue (128));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via IntegerValue to 128");

  ok = p->SetAttributeFailSafe ("IntegerTraceSource2", IntegerValue (-128));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to -128");

  ok = p->SetAttributeFailSafe ("IntegerTraceSource2", IntegerValue (-129));
  NS_TEST_ASSERT_MSG_EQ (ok, false, "Unexpectedly could SetAttributeFailSafe() via IntegerValue to -129");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief Trace sources used like Attributes must also work as trace sources, 
 * make sure we can use them that way.
 */
class IntegerTraceSourceTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  IntegerTraceSourceTestCase (std::string description);
  virtual ~IntegerTraceSourceTestCase ()
  {}

private:
  virtual void DoRun (void);

  /**
   * Notify the call of source 1.
   * \param old First value.
   * \param n Second value.
   */
  void NotifySource1 ([[maybe_unused]] int8_t old, int8_t n)
  {
    m_got1 = n;
  }
  int64_t m_got1; //!< Value used to verify that source 1 was called.
};

IntegerTraceSourceTestCase::IntegerTraceSourceTestCase (std::string description)
  : TestCase (description)
{}

void
IntegerTraceSourceTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // Check to make sure changing an Attribute value triggers a trace callback
  // that sets a member variable.
  //
  m_got1 = 1234;

  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (-1));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to -1");

  //
  // Source1 is declared as a TraceSourceAccessor to m_intSrc1.  This m_intSrc1
  // is also declared as an Integer Attribute.  We just checked to make sure we
  // could set it using an IntegerValue through its IntegerTraceSource1 "persona."
  // We should also be able to hook a trace source to the underlying variable.
  //
  ok = p->TraceConnectWithoutContext ("Source1", MakeCallback (&IntegerTraceSourceTestCase::NotifySource1, this));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not TraceConnectWithoutContext() \"Source1\" to NodifySource1()");

  //
  // When we set the IntegerValue that now underlies both the Integer Attribute
  // and the trace source, the trace should fire and call NotifySource1 which
  // will set m_got1 to the new value.
  //
  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (0));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to 0");

  NS_TEST_ASSERT_MSG_EQ (m_got1, 0, "Hitting a TracedValue does not cause trace callback to be called");

  //
  // Now disconnect from the trace source and ensure that the trace callback
  // is not called if the trace source is hit.
  //
  ok = p->TraceDisconnectWithoutContext ("Source1", MakeCallback (&IntegerTraceSourceTestCase::NotifySource1, this));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not TraceConnectWithoutContext() \"Source1\" to NodifySource1()");

  ok = p->SetAttributeFailSafe ("IntegerTraceSource1", IntegerValue (1));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() via IntegerValue to 1");

  NS_TEST_ASSERT_MSG_EQ (m_got1, 0, "Hitting a TracedValue after disconnect still causes callback");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief Trace sources used like Attributes must also work as trace sources, 
 * make sure we can use them that way.
 */
class TracedCallbackTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  TracedCallbackTestCase (std::string description);
  virtual ~TracedCallbackTestCase ()
  {}

private:
  virtual void DoRun (void);

  /**
   * Notify the call of source 2.
   * \param a First value.
   * \param b Second value.
   * \param c Third value.
   */
  void NotifySource2 (double a, [[maybe_unused]] int b, [[maybe_unused]] float c)
  {
    m_got2 = a;
  }

  double m_got2; //!< Value used to verify that source 2 was called.
};

TracedCallbackTestCase::TracedCallbackTestCase (std::string description)
  : TestCase (description)
{}

void
TracedCallbackTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // Initialize the
  //
  m_got2 = 4.3;

  //
  // Invoke the callback that lies at the heart of this test.  We have a
  // method InvokeCb() that just executes m_cb().  The variable m_cb is
  // declared as a TracedCallback<double, int, float>.  This kind of beast
  // is like a callback but can call a list of targets.  This list should
  // be empty so nothing should happen now.  Specifically, m_got2 shouldn't
  // have changed.
  //
  p->InvokeCb (1.0, -5, 0.0);
  NS_TEST_ASSERT_MSG_EQ (m_got2, 4.3, "Invoking a newly created TracedCallback results in an unexpected callback");

  //
  // Now, wire the TracedCallback up to a trace sink.  This sink will just set
  // m_got2 to the first argument.
  //
  ok = p->TraceConnectWithoutContext ("Source2", MakeCallback (&TracedCallbackTestCase::NotifySource2, this));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not TraceConnectWithoutContext() to NotifySource2");

  //
  // Now if we invoke the callback, the trace source should fire and m_got2
  // should be set in the trace sink.
  //
  p->InvokeCb (1.0, -5, 0.0);
  NS_TEST_ASSERT_MSG_EQ (m_got2, 1.0, "Invoking TracedCallback does not result in trace callback");

  //
  // Now, disconnect the trace sink and see what happens when we invoke the
  // callback again.  Of course, the trace should not happen and m_got2
  // should remain unchanged.
  //
  ok = p->TraceDisconnectWithoutContext ("Source2", MakeCallback (&TracedCallbackTestCase::NotifySource2, this));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not TraceDisconnectWithoutContext() from NotifySource2");

  p->InvokeCb (-1.0, -5, 0.0);
  NS_TEST_ASSERT_MSG_EQ (m_got2, 1.0, "Invoking disconnected TracedCallback unexpectedly results in trace callback");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief Smart pointers (Ptr) are central to our architecture, so they 
 * must work as attributes.
 */
class PointerAttributeTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  PointerAttributeTestCase (std::string description);
  virtual ~PointerAttributeTestCase ()
  {}

private:
  virtual void DoRun (void);

  /**
   * Notify the call of source 2.
   * \param a First value.
   * \param b Second value.
   * \param c Third value.
   */
  void NotifySource2 (double a, [[maybe_unused]] int b, [[maybe_unused]] float c)
  {
    m_got2 = a;
  }

  double m_got2; //!< Value used to verify that source 2 was called.
};

PointerAttributeTestCase::PointerAttributeTestCase (std::string description)
  : TestCase (description)
{}

void
PointerAttributeTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // We have declared a PointerValue Attribute named "Pointer" with a pointer
  // checker of type Derived.  This means that we should be able to pull out
  // a Ptr<Derived> with the initial value (which is 0).
  //
  PointerValue ptr;
  p->GetAttribute ("Pointer", ptr);
  Ptr<Derived> derived = ptr.Get<Derived> ();
  NS_TEST_ASSERT_MSG_EQ (derived, 0, "Unexpectedly found non-null pointer in newly initialized PointerValue Attribute");

  //
  // Now, lets create an Object of type Derived and set the local Ptr to point
  // to that object.  We can then set the PointerValue Attribute to that Ptr.
  //
  derived = Create<Derived> ();
  ok = p->SetAttributeFailSafe ("Pointer", PointerValue (derived));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() a PointerValue of the correct type");

  //
  // Pull the value back out of the Attribute and make sure it points to the
  // correct object.
  //
  p->GetAttribute ("Pointer", ptr);
  Ptr<Derived> stored = ptr.Get<Derived> ();
  NS_TEST_ASSERT_MSG_EQ (stored, derived, "Retrieved Attribute does not match stored PointerValue");

  //
  // We should be able to use the Attribute Get() just like GetObject<type>,
  // So see if we can get a Ptr<Object> out of the Ptr<Derived> we stored.
  // This should be a pointer to the same physical memory since its the
  // same object.
  //
  p->GetAttribute ("Pointer", ptr);
  Ptr<Object> storedBase = ptr.Get<Object> ();
  NS_TEST_ASSERT_MSG_EQ (storedBase, stored, "Retrieved Ptr<Object> does not match stored Ptr<Derived>");

  //
  // If we try to Get() something that is unrelated to what we stored, we should
  // retrieve a 0.
  //
  p->GetAttribute ("Pointer", ptr);
  Ptr<AttributeObjectTest> x = ptr.Get<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_EQ (x, 0, "Unexpectedly retrieved unrelated Ptr<type> from stored Ptr<Derived>");

  //
  // Test whether the initialized pointers from two different objects
  // point to different Derived objects
  //
  p->GetAttribute ("PointerInitialized", ptr);
  Ptr<Derived> storedPtr = ptr.Get<Derived> ();
  Ptr<AttributeObjectTest> p2 = CreateObject<AttributeObjectTest> ();
  PointerValue ptr2;
  p2->GetAttribute ("PointerInitialized", ptr2);
  Ptr<Derived> storedPtr2 = ptr2.Get<Derived> ();
  NS_TEST_ASSERT_MSG_NE (storedPtr, storedPtr2, "ptr and ptr2 both have PointerInitialized pointing to the same object");
  PointerValue ptr3;
  p2->GetAttribute ("PointerInitialized", ptr3);
  Ptr<Derived> storedPtr3 = ptr3.Get<Derived> ();
  NS_TEST_ASSERT_MSG_NE (storedPtr, storedPtr3, "ptr and ptr3 both have PointerInitialized pointing to the same object");

  //
  // Test whether object factory creates the objects properly
  //
  ObjectFactory factory;
  factory.SetTypeId ("ns3::AttributeObjectTest");
  factory.Set ("PointerInitialized", StringValue ("ns3::Derived"));
  Ptr<AttributeObjectTest> aotPtr = factory.Create ()->GetObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (aotPtr, 0, "Unable to factory.Create() a AttributeObjectTest");
  Ptr<AttributeObjectTest> aotPtr2 = factory.Create ()->GetObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (aotPtr2, 0, "Unable to factory.Create() a AttributeObjectTest");
  NS_TEST_ASSERT_MSG_NE (aotPtr, aotPtr2, "factory object not creating unique objects");
  PointerValue ptr4;
  aotPtr->GetAttribute ("PointerInitialized", ptr4);
  Ptr<Derived> storedPtr4 = ptr4.Get<Derived> ();
  PointerValue ptr5;
  aotPtr2->GetAttribute ("PointerInitialized", ptr5);
  Ptr<Derived> storedPtr5 = ptr5.Get<Derived> ();
  NS_TEST_ASSERT_MSG_NE (storedPtr4, storedPtr5, "aotPtr and aotPtr2 are unique, but their Derived member is not");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief Test the Attributes of type CallbackValue.
 */
class CallbackValueTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * \param description The TestCase description.
   */
  CallbackValueTestCase (std::string description);
  virtual ~CallbackValueTestCase ()
  {}

  /**
   * Function to invoke the callback.
   * \param a The value.
   */
  void InvokeCbValue (int8_t a)
  {
    if (!m_cbValue.IsNull ())
      {
        m_cbValue (a);
      }
  }

private:
  virtual void DoRun (void);

  Callback<void,int8_t> m_cbValue; //!< The callback

  /**
   * Function invoked when the callback is fired.
   * \param a The value.
   */
  void NotifyCallbackValue (int8_t a)
  {
    m_gotCbValue = a;
  }

  int16_t m_gotCbValue; //!< Value used to verify that source 2 was called.
};

CallbackValueTestCase::CallbackValueTestCase (std::string description)
  : TestCase (description)
{}

void
CallbackValueTestCase::DoRun (void)
{
  Ptr<AttributeObjectTest> p;
  bool ok;

  p = CreateObject<AttributeObjectTest> ();
  NS_TEST_ASSERT_MSG_NE (p, 0, "Unable to CreateObject");

  //
  // The member variable m_cbValue is declared as a Callback<void, int8_t>.  The
  // Attribute named "Callback" also points to m_cbValue and allows us to set the
  // callback using that Attribute.
  //
  // NotifyCallbackValue is going to be the target of the callback and will just set
  // m_gotCbValue to its single parameter.  This will be the parameter from the
  // callback invocation.  The method InvokeCbValue() just invokes the m_cbValue
  // callback if it is non-null.
  //
  m_gotCbValue = 1;

  //
  // If we invoke the callback (which has not been set) nothing should happen.
  // Further, nothing should happen when we initialize the callback (it shouldn't
  // accidentally fire).
  //
  p->InvokeCbValue (2);
  CallbackValue cbValue = MakeCallback (&CallbackValueTestCase::NotifyCallbackValue, this);

  NS_TEST_ASSERT_MSG_EQ (m_gotCbValue, 1, "Callback unexpectedly fired");

  ok = p->SetAttributeFailSafe ("Callback", cbValue);
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() a CallbackValue");

  //
  // Now that the callback has been set, invoking it should set m_gotCbValue.
  //
  p->InvokeCbValue (2);
  NS_TEST_ASSERT_MSG_EQ (m_gotCbValue, 2, "Callback Attribute set by CallbackValue did not fire");

  ok = p->SetAttributeFailSafe ("Callback", CallbackValue (MakeNullCallback<void,int8_t> ()));
  NS_TEST_ASSERT_MSG_EQ (ok, true, "Could not SetAttributeFailSafe() a null CallbackValue");

  //
  // If the callback has been set to a null callback, it should no longer fire.
  //
  p->InvokeCbValue (3);
  NS_TEST_ASSERT_MSG_EQ (m_gotCbValue, 2, "Callback Attribute set to null callback unexpectedly fired");
}

/**
 * \ingroup attribute-tests
 *  
 * \brief The attributes Test Suite.
 */
class AttributesTestSuite : public TestSuite
{
public:
  AttributesTestSuite ();
};

AttributesTestSuite::AttributesTestSuite ()
  : TestSuite ("attributes", UNIT)
{
  AddTestCase (new AttributeTestCase<BooleanValue> ("Check Attributes of type BooleanValue"), TestCase::QUICK);
  AddTestCase (new AttributeTestCase<IntegerValue> ("Check Attributes of type IntegerValue"), TestCase::QUICK);
  AddTestCase (new AttributeTestCase<UintegerValue> ("Check Attributes of type UintegerValue"), TestCase::QUICK);
  AddTestCase (new AttributeTestCase<DoubleValue> ("Check Attributes of type DoubleValue"), TestCase::QUICK);
  AddTestCase (new AttributeTestCase<EnumValue> ("Check Attributes of type EnumValue"), TestCase::QUICK);
  AddTestCase (new AttributeTestCase<TimeValue> ("Check Attributes of type TimeValue"), TestCase::QUICK);
  AddTestCase (new RandomVariableStreamAttributeTestCase ("Check Attributes of type RandomVariableStream"), TestCase::QUICK);
  AddTestCase (new ObjectVectorAttributeTestCase ("Check Attributes of type ObjectVectorValue"), TestCase::QUICK);
  AddTestCase (new ObjectMapAttributeTestCase ("Check Attributes of type ObjectMapValue"), TestCase::QUICK);
  AddTestCase (new PointerAttributeTestCase ("Check Attributes of type PointerValue"), TestCase::QUICK);
  AddTestCase (new CallbackValueTestCase ("Check Attributes of type CallbackValue"), TestCase::QUICK);
  AddTestCase (new IntegerTraceSourceAttributeTestCase ("Ensure TracedValue<uint8_t> can be set like IntegerValue"), TestCase::QUICK);
  AddTestCase (new IntegerTraceSourceTestCase ("Ensure TracedValue<uint8_t> also works as trace source"), TestCase::QUICK);
  AddTestCase (new TracedCallbackTestCase ("Ensure TracedCallback<double, int, float> works as trace source"), TestCase::QUICK);
}

static AttributesTestSuite g_attributesTestSuite; //!< Static variable for test initialization
