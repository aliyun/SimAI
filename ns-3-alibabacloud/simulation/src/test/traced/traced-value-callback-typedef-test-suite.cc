/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Lawrence Livermore National Laboratory
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
 * Author:  Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include "ns3/test.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"  // SequenceNumber32

using namespace ns3;

/**
 * \file
 * \ingroup system-tests-traced
 * 
 * TracedValueCallback tests to verify that they work with different types
 * of classes - it tests bool, double, various types of integers types, 
 * Time, and SequenceNumber32.
 */

namespace {

/**
 * \ingroup system-tests-traced
 * 
 * \name Stringify the known TracedValue type names.
 */
/**
 * Generic template for unknown classes.
 * 
 * \returns The \c TracedValueCallback type name, or "unknown" for unknown classes.
 */
template <typename T> inline
std::string TypeName (void) { return "unknown"; }
/** @{ */
template <> inline std::string TypeName <bool>     (void) { return "Bool"    ; }
template <> inline std::string TypeName <int8_t>   (void) { return "Int8_t"  ; }
template <> inline std::string TypeName <int16_t>  (void) { return "Int16_t" ; }
template <> inline std::string TypeName <int32_t>  (void) { return "Int32_t" ; }
template <> inline std::string TypeName <int64_t>  (void) { return "Int64_t" ; }
template <> inline std::string TypeName <uint8_t>  (void) { return "Uint8_t" ; }
template <> inline std::string TypeName <uint16_t> (void) { return "Uint16_t"; }
template <> inline std::string TypeName <uint32_t> (void) { return "Uint32_t"; }
template <> inline std::string TypeName <uint64_t> (void) { return "Uint64_t"; }
template <> inline std::string TypeName <double>   (void) { return "Double"  ; }
template <> inline std::string TypeName <Time>     (void) { return "Time"    ; }
template <> inline std::string TypeName <SequenceNumber32> (void) { return "SequenceNumber32" ; }
/** @} */


/**
 * \ingroup system-tests-traced
 * 
 * Result of callback test.
 *
 * Since the sink function is outside the invoking class,
 * which in this case is TracedValueCallbackTestCase, we can't use
 * the test macros directly.  Instead, we cache the result
 * in the \c g_Result global value, then inspect it
 * in the TracedValueCallbackTestCase::CheckType method.
 */
std::string g_Result = "";
  

/**
 * \ingroup system-tests-traced
 * 
 * Template for TracedValue sink functions.
 *
 * This generates a sink function for any underlying type.
 *
 * \tparam T \explicit The type of the value being traced.
 *        Since the point of this template is to create a
 *        sink function, the template type must be given explicitly.
 * \param [in] oldValue The original value.
 * \param [in] newValue The new value.
 */
template <typename T>
void TracedValueCbSink (T oldValue, T newValue)
{
  std::cout << ": "
            << (int64_t)oldValue << " -> "
            << (int64_t)newValue
            << std::endl;
  if (oldValue != 0)
    g_Result = "oldValue should be 0";
  else if (newValue != 1)
    g_Result = "newValue should be 1";

}  // TracedValueCbSink<>()
  
/**
 * \ingroup system-tests-traced
 * 
 * TracedValueCbSink specialization for Time.
 * \param oldValue The old value,
 * \param newValue The new value.
 */
template <>
void TracedValueCbSink<Time> (Time oldValue, Time newValue)
{
  TracedValueCbSink <int64_t> (oldValue.GetInteger (),
                               newValue.GetInteger ());
}
/**
 * \ingroup system-tests-traced
 * 
 * TracedValueCbSink specialization for SequenceNumber32.
 * \param oldValue The old value,
 * \param newValue The new value.
 */
template <>  
void TracedValueCbSink<SequenceNumber32> (SequenceNumber32 oldValue,
                                          SequenceNumber32 newValue)
{
  TracedValueCbSink <int64_t> (oldValue.GetValue (), newValue.GetValue ());
}


}  // unnamed namespace


/**
 * \ingroup system-tests-traced
 *
 * \brief TracedValueCallback Test Case
 */
class TracedValueCallbackTestCase : public TestCase
{
public:
  TracedValueCallbackTestCase ();
  virtual ~TracedValueCallbackTestCase () {}

private:

  /**
   * A class to check that the callback function typedef will
   * actually connect to the TracedValue.
   */
  template <typename T>
  class CheckTvCb : public Object
  {
    /// Traced value.
    TracedValue<T> m_value;

  public:
    /** Constructor. */
    CheckTvCb (void)  : m_value (0)  { }
    
    /**
     * \brief Register this type.
     * \return The object TypeId. 
     */
    static TypeId GetTypeId (void)
    {
      static TypeId tid =
        TypeId ("CheckTvCb<" + TypeName<T>() + ">")
        .SetParent <Object> ()
        .AddTraceSource ("value",
                         "A value being traced.",
                         MakeTraceSourceAccessor (&CheckTvCb<T>::m_value),
                         ("ns3::TracedValueCallback::" + TypeName<T>()) )
        ;
      return tid;
    }  // GetTypeId ()

    /**
     * Check the sink function against the actual TracedValue invocation.
     *
     * We connect the TracedValue to the sink.  If the types
     * aren't compatible, the connection will fail.
     *
     * Just to make sure, we increment the TracedValue,
     * which calls the sink.
     * 
     * \param cb Callback.
     */
    template <typename U>
    void Invoke (U cb)
    {
      bool ok = TraceConnectWithoutContext ("value", MakeCallback (cb));
      std::cout << GetTypeId () << ": "
                << (ok ? "connected " : "failed to connect ")
                << GetTypeId ().GetTraceSource (0).callback
        ;
      // The endl is in the sink function.
      
      if (ok)
        // Odd form here is to accommodate the uneven operator support
        // of Time and SequenceNumber32.
        m_value = m_value + (T) 1; 
      else
        {
          // finish the line started above
          std::cout << std::endl;

          // and log the error
          g_Result = "failed to connect callback";
        }

    }  // Invoke()

  };  // class CheckTvCb<T>
  

  /**
   * Check the TracedValue typedef against TracedValueCbSink<T>.
   *
   * We instantiate a sink function of type \c U, initialized to
   * TracedValueCbSink<T>.  If this compiles, we've proved the
   * sink function and the typedef agree.
   *
   * \tparam T \explicit The base type.
   * \tparam U \explicit The TracedValueCallback sink typedef type.
   */
  template <typename T, typename U>
  void CheckType (void)
  {
    U sink = TracedValueCbSink<T>;
    CreateObject<CheckTvCb<T> > ()->Invoke (sink);

    NS_TEST_ASSERT_MSG_EQ (g_Result, "", g_Result);
    g_Result = "";
    
  }  // CheckType<>()
    
  virtual void DoRun (void);

};

TracedValueCallbackTestCase::TracedValueCallbackTestCase ()
  : TestCase ("Check basic TracedValue callback operation")
{
}

void
TracedValueCallbackTestCase::DoRun (void)
{
  CheckType< bool,     TracedValueCallback::Bool   > ();
  CheckType< int8_t,   TracedValueCallback::Int8   > ();
  CheckType< int16_t,  TracedValueCallback::Int16  > ();
  CheckType< int32_t,  TracedValueCallback::Int32  > ();
  CheckType< int64_t,  TracedValueCallback::Int64  > ();
  CheckType< uint8_t,  TracedValueCallback::Uint8  > ();
  CheckType< uint16_t, TracedValueCallback::Uint16 > ();
  CheckType< uint32_t, TracedValueCallback::Uint32 > ();
  CheckType< uint64_t, TracedValueCallback::Uint64 > ();
  CheckType< double,   TracedValueCallback::Double > ();
  CheckType< Time,     TracedValueCallback::Time   > ();
  CheckType< SequenceNumber32, TracedValueCallback::SequenceNumber32 > ();
}

/**
 * \ingroup system-tests-traced
 *
 * \brief TracedValueCallback TestSuite
 */
class TracedValueCallbackTestSuite : public TestSuite
{
public:
  TracedValueCallbackTestSuite ();
};

TracedValueCallbackTestSuite::TracedValueCallbackTestSuite ()
  : TestSuite ("traced-value-callback", UNIT)
{
  AddTestCase (new TracedValueCallbackTestCase, TestCase::QUICK);
}

/// Static variable for test initialization
static TracedValueCallbackTestSuite tracedValueCallbackTestSuite;
