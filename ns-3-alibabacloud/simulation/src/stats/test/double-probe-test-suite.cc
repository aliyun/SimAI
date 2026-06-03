/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

// Include a header file from your module to test.
#include "ns3/double-probe.h"
#include "ns3/test.h"
#include "ns3/random-variable-stream.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/traced-value.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/object.h"
#include "ns3/type-id.h"
#include "ns3/names.h"

using namespace ns3;

/**
 * \ingroup stats-tests
 *
 * \brief Simple data emitter to check that a probe receives data.
 */
class SampleEmitter : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  SampleEmitter ()
  {
    m_var = CreateObject<ExponentialRandomVariable> ();
  }
  virtual ~SampleEmitter ()
  {
  }
  /// Start emission of data.
  void Start ()
  {
    Reschedule ();
  }
  /// Reschedule a report \sa Report
  void Reschedule ()
  {
    m_time = m_var->GetValue ();
    Simulator::Schedule (Seconds (m_time), &SampleEmitter::Report, this);
    m_time += Simulator::Now ().GetSeconds ();
  }
  /// \return the time delta of the next report.
  double GetTime ()
  {
    return m_time;
  }
  /// \return a random variable, different for each reschedule.
  double GetValue ()
  {
    return aux;
  }
private:
  /// Reports a new value and reschedules \sa Reschedule
  void Report ()
  {
    aux = m_var->GetValue ();
    m_trace = aux;
    Reschedule ();
  }
  Ptr<ExponentialRandomVariable> m_var; //!< Random value generator.
  double m_time;  //!< Delta time between reschedules.
  TracedValue<double> m_trace; //!< Trace
  double aux; //!< Emitted value.
};


TypeId
SampleEmitter::GetTypeId (void)
{
  static TypeId tid = TypeId ("SampleEmitter")
    .SetParent<Object> ()
    .AddTraceSource ("Emitter", "XX",
                     MakeTraceSourceAccessor (&SampleEmitter::m_trace),
                     "ns3::TracedValueCallback::Double")
  ;
  return tid;
}


/**
 * \ingroup stats-tests
 *
 * \brief DoubleProbe class - Test case for connecting and receiving data.
 */
class ProbeTestCase1 : public TestCase
{
public:
  ProbeTestCase1 ();
  virtual ~ProbeTestCase1 ();

private:
  virtual void DoRun (void);

  /**
   * Trace sink.
   * \param context Trace context
   * \param oldValue Old value
   * \param newValue New value
   */
  void TraceSink (std::string context, double oldValue, double newValue);
  uint32_t m_objectProbed;  //!< Number of probes by Object
  uint32_t m_pathProbed;    //!< Number of probed by Path
  Ptr<SampleEmitter> m_s;   //!< Sample emitter pointer
};

ProbeTestCase1::ProbeTestCase1 ()
  : TestCase ("basic probe test case"),
    m_objectProbed (0),
    m_pathProbed (0)
{
}

ProbeTestCase1::~ProbeTestCase1 ()
{
}

void
ProbeTestCase1::TraceSink (std::string context, double oldValue, double newValue)
{
  NS_TEST_ASSERT_MSG_GT (Simulator::Now (), Seconds (100), "Probed a value outside of the time window");
  NS_TEST_ASSERT_MSG_LT (Simulator::Now (), Seconds (200), "Probed a value outside of the time window");

  NS_TEST_ASSERT_MSG_EQ_TOL (m_s->GetValue (), newValue, 0.00001, "Value probed different than value in the variable");

  if (context == "testProbe")
    {
      m_objectProbed++;
    }
  else if (context == "testProbe2")
    {
      m_pathProbed++;
    }
}

void
ProbeTestCase1::DoRun (void)
{
  // Defer creation of this until here because it is a random variable
  m_s = CreateObject<SampleEmitter> ();
  // Test that all instances of probe data are between time window specified
  // Check also that probes can be hooked to sources by Object and by path

  Ptr<DoubleProbe> p = CreateObject<DoubleProbe> ();
  p->SetName ("testProbe");

  Simulator::Schedule (Seconds (1), &SampleEmitter::Start, m_s);
  p->SetAttribute ("Start", TimeValue (Seconds (100.0)));
  p->SetAttribute ("Stop", TimeValue (Seconds (200.0)));
  Simulator::Stop (Seconds (300));

  // Register our emitter object so we can fetch it by using the Config
  // namespace
  Names::Add ("/Names/SampleEmitter", m_s);

  // Hook probe to the emitter.
  p->ConnectByObject ("Emitter", m_s);

  // Hook our test function to the probe trace source
  p->TraceConnect ("Output", p->GetName (), MakeCallback (&ProbeTestCase1::TraceSink, this));

  // Repeat but hook the probe to the object this time using the Config
  // name set above
  Ptr<DoubleProbe> p2 = CreateObject<DoubleProbe> ();
  p2->SetName ("testProbe2");
  p2->SetAttribute ("Start", TimeValue (Seconds (100.0)));
  p2->SetAttribute ("Stop", TimeValue (Seconds (200.0)));

  // Hook probe to the emitter.
  p2->ConnectByPath ("/Names/SampleEmitter/Emitter");

  // Hook our test function to the  probe trace source
  p2->TraceConnect ("Output", p2->GetName (), MakeCallback (&ProbeTestCase1::TraceSink, this));

  Simulator::Run ();

  // Check that each trace sink was called
  NS_TEST_ASSERT_MSG_GT (m_objectProbed, 0, "Trace sink for object probe never called");
  NS_TEST_ASSERT_MSG_GT (m_pathProbed, 0, "Trace sink for path probe never called");
  Simulator::Destroy ();
}


/**
 * \ingroup stats-tests
 *
 * \brief DoubleProbe class TestSuite
 */
class ProbeTestSuite : public TestSuite
{
public:
  ProbeTestSuite ();
};

ProbeTestSuite::ProbeTestSuite ()
  : TestSuite ("double-probe", UNIT)
{
  AddTestCase (new ProbeTestCase1, TestCase::QUICK);
}

 /// Static variable for test initialization
static ProbeTestSuite probeTestSuite;

