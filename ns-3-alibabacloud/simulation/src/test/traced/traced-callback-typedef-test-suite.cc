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

#include <iostream>
#include <sstream>
#include <set>
#include <string>
#include "ns3/test.h"
#include "ns3/core-module.h"
#include "ns3/dsr-module.h"       // DsrOPtionSRHeader
#include "ns3/internet-module.h"  // Ipv4, Ipv4L3Protocol, Ipv4PacketProbe
                                  // Ipv6L3Protocol, Ipv6PacketProbe
#include "ns3/lr-wpan-mac.h"      // LrWpanMac
#include "ns3/lte-module.h"       // PhyReceptionStatParameters,
                                  // PhyTransmissionStatParameters,
                                  // LteUePowerControl
#include "ns3/mesh-module.h"      // PeerManagementProtocol
#include "ns3/mobility-module.h"  // MobilityModel
#include "ns3/network-module.h"   // Packet, PacketBurst
#include "ns3/olsr-module.h"      // olsr::RoutingProtocol
#include "ns3/sixlowpan-module.h" // SixLowPanNetDevice
#include "ns3/spectrum-module.h"  // SpectrumValue
#include "ns3/stats-module.h"     // TimeSeriesAdapter
#include "ns3/uan-module.h"       // UanPhy
#include "ns3/wifi-phy-state-helper.h"
#include "ns3/wifi-mac-header.h"

using namespace ns3;

/**
 * \file
 * \ingroup system-tests-traced
 * 
 * TracedCallback tests to verify if they are called with
 * the right type and number of arguments.
 */


/**
 * \ingroup system-tests-traced
 * 
 * TracedCallback Testcase.
 * 
 * This test verifies that the TracedCallback is called with
 * the right type and number of arguments.
 */
class TracedCallbackTypedefTestCase : public TestCase
{
public:
  TracedCallbackTypedefTestCase ();
  virtual ~TracedCallbackTypedefTestCase ()
  {
  }

  /**
   * Number of arguments passed to callback.
   *
   * Since the sink function is outside the invoking class we can't use
   * the test macros directly.  Instead, we cache success
   * in the \c m_nArgs public value, then inspect it
   * in the CheckType() method.
   */
  static std::size_t m_nArgs;

private:

  /** Callback checkers. */
  template <typename... Ts>
  class Checker;

  virtual void DoRun (void);

};  // TracedCallbackTypedefTestCase

/*
  --------------------------------------------------------------------
  Support functions and classes
  --------------------------------------------------------------------
*/

namespace {

/** 
 * \ingroup system-tests-traced
 * 
 * Record typedefs which are identical to previously declared.
 * \return a container of strings representing the duplicates.
 */
std::set<std::string>
Duplicates (void)
{
  std::set<std::string> dupes;

  dupes.insert ("LteRlc::NotifyTxTracedCallback");
  dupes.insert ("LteRlc::ReceiveTracedCallback");
  dupes.insert ("LteUeRrc::ImsiCidRntiTracedCallback");
  dupes.insert ("LteUeRrc::MibSibHandoverTracedCallback");
  dupes.insert ("WifiPhyStateHelper::RxEndErrorTracedCallback");

  return dupes;
}

/**
 * \ingroup system-tests-traced
 * 
 * Container for duplicate types.
 */
std::set<std::string> g_dupes = Duplicates ();


/**
 * \ingroup system-tests-traced
 * 
 * Stringify the known TracedCallback type names.
 *
 * \tparam T \explicit The typedef name.
 * \param [in] N The number of arguments expected.
 * \returns The \c TracedCallback type name.
 */
template <typename T>
inline
std::string TypeName (int N)
{
  return "unknown";
}

/**
 * \ingroup system-tests-traced
 * 
 * Returns a sting representing the type of a class.
*/
#define TYPENAME(T)               \
  template <> \
  inline std::string  \
  TypeName < T >     (int N)      \
  {                               \
    std::stringstream ss;         \
    ss << # T << "(" << N << ")"; \
    return ss.str ();             \
  }

/**
 * \ingroup system-tests-traced
 * 
 * \name Stringify known typename.
*/
/**
 * @{
 * \brief Stringify a known typename
 */
TYPENAME (dsr::DsrOptionSRHeader::TracedCallback);
TYPENAME (EpcUeNas::StateTracedCallback);
TYPENAME (Ipv4L3Protocol::DropTracedCallback);
TYPENAME (Ipv4L3Protocol::SentTracedCallback);
TYPENAME (Ipv4L3Protocol::TxRxTracedCallback);
TYPENAME (Ipv6L3Protocol::DropTracedCallback);
TYPENAME (Ipv6L3Protocol::SentTracedCallback);
TYPENAME (Ipv6L3Protocol::TxRxTracedCallback);
TYPENAME (LrWpanMac::SentTracedCallback);
TYPENAME (LrWpanMac::StateTracedCallback);
TYPENAME (LrWpanPhy::StateTracedCallback);
TYPENAME (LteEnbMac::DlSchedulingTracedCallback);
TYPENAME (LteEnbMac::UlSchedulingTracedCallback);
TYPENAME (LteEnbPhy::ReportInterferenceTracedCallback);
TYPENAME (LteEnbPhy::ReportUeSinrTracedCallback);
TYPENAME (LteEnbRrc::ConnectionHandoverTracedCallback);
TYPENAME (LteEnbRrc::HandoverStartTracedCallback);
TYPENAME (LteEnbRrc::NewUeContextTracedCallback);
TYPENAME (LteEnbRrc::ReceiveReportTracedCallback);
TYPENAME (LtePdcp::PduRxTracedCallback);
TYPENAME (LtePdcp::PduTxTracedCallback);
TYPENAME (LteUePhy::StateTracedCallback);
TYPENAME (LteUePhy::RsrpSinrTracedCallback);
TYPENAME (LteUeRrc::CellSelectionTracedCallback);
TYPENAME (LteUeRrc::StateTracedCallback);
TYPENAME (Mac48Address::TracedCallback);
TYPENAME (MobilityModel::TracedCallback);
TYPENAME (olsr::RoutingProtocol::PacketTxRxTracedCallback);
TYPENAME (olsr::RoutingProtocol::TableChangeTracedCallback);
TYPENAME (Packet::AddressTracedCallback);
TYPENAME (Packet::Mac48AddressTracedCallback);
TYPENAME (Packet::SinrTracedCallback);
TYPENAME (Packet::SizeTracedCallback);
TYPENAME (Packet::TracedCallback);
TYPENAME (PacketBurst::TracedCallback);
TYPENAME (dot11s::PeerManagementProtocol::LinkOpenCloseTracedCallback);
TYPENAME (PhyReceptionStatParameters::TracedCallback);
TYPENAME (PhyTransmissionStatParameters::TracedCallback);
TYPENAME (SixLowPanNetDevice::DropTracedCallback);
TYPENAME (SixLowPanNetDevice::RxTxTracedCallback);
TYPENAME (SpectrumChannel::LossTracedCallback);
TYPENAME (SpectrumValue::TracedCallback);
TYPENAME (TimeSeriesAdaptor::OutputTracedCallback);
TYPENAME (UanMac::PacketModeTracedCallback);
TYPENAME (UanMacCw::QueueTracedCallback);
TYPENAME (UanMacRc::QueueTracedCallback);
TYPENAME (UanNetDevice::RxTxTracedCallback);
TYPENAME (UanPhy::TracedCallback);
TYPENAME (UeManager::StateTracedCallback);
TYPENAME (WifiMacHeader::TracedCallback);
TYPENAME (WifiPhyStateHelper::RxOkTracedCallback);
TYPENAME (WifiPhyStateHelper::StateTracedCallback);
TYPENAME (WifiPhyStateHelper::TxTracedCallback);
TYPENAME (WifiRemoteStationManager::PowerChangeTracedCallback);
TYPENAME (WifiRemoteStationManager::RateChangeTracedCallback);
/** @} */
#undef TYPENAME


/**
 * \ingroup system-tests-traced
 * 
 * Log that a callback was invoked.
 *
 * We can't actually do anything with any of the arguments,
 * but the fact we got called is what's important.
 *
 * \param [in] N The number of arguments passed to the callback.
 */
void SinkIt (std::size_t N)
{
  std::cout << "with " << N << " args." << std::endl;
  TracedCallbackTypedefTestCase::m_nArgs = N;
}

/**
 * \ingroup system-tests-traced
 * 
 * Sink functions.
 */
template <typename... Ts>
class TracedCbSink
{
public:
  /**
   * \brief Sink function, called by a TracedCallback.
   * \tparam Ts prameters of the TracedCallback.
   */
  static void Sink (Ts...)
  {
    const std::size_t n = sizeof...(Ts);
    SinkIt (n);
  }
};


}  // unnamed namespace


/*
  --------------------------------------------------------------------
  Class TracedCallbackTypedefTestCase implementation

  We put the template implementations here to break a dependency cycle
  from the Checkers() to TracedCbSink<> to SinkIt()
  --------------------------------------------------------------------
*/

std::size_t TracedCallbackTypedefTestCase::m_nArgs = 0;

template <typename... Ts>
class TracedCallbackTypedefTestCase::Checker : public Object
{
  /// TracedCallback to be called.
  TracedCallback<Ts...> m_cb;
public:
  Checker  () {};
  virtual ~Checker () {};
  
  /// Arguments of the TracedCallback.
  std::tuple<typename TypeTraits<Ts>::BaseType...> m_items;

  /// Number of arguments of the TracedCallback.
  const std::size_t m_nItems = sizeof...(Ts);

  /**
   * Invoke a TracedCallback.
   */
  template <typename U>
  void Invoke (void)
  {
    U sink = TracedCbSink<Ts...>::Sink;
    Callback<void, Ts...> cb = MakeCallback (sink);

    std::cout << TypeName<U> (m_nItems) << " invoked ";
    m_cb.ConnectWithoutContext (cb);
    std::apply(m_cb, m_items);
    Cleanup ();
  }
  
  /**
   * Cleanup the test.
   */
  void Cleanup ()
  {
    if (m_nArgs == 0)
      {
        std::cout << std::endl;
      }
    NS_ASSERT_MSG (m_nArgs && m_nArgs == m_nItems, "failed, m_nArgs: " << m_nArgs << " N: " << m_nItems);
    m_nArgs = 0;
  }

};


TracedCallbackTypedefTestCase::TracedCallbackTypedefTestCase ()
  : TestCase ("Check basic TracedCallback operation")
{
}

/**
 * \ingroup system-tests-traced
 * 
 * Check the TracedCallback duplicate by checking if it maches the TracedCallback
 * it is supposed to be equal to.
 */
#define DUPE(U, T1)                                                     \
  if (g_dupes.find ( # U ) == g_dupes.end ()) {                           \
      NS_TEST_ASSERT_MSG_NE (0, 1,                                        \
                             "expected to find " <<  # U << " in dupes."); }  \
  if (TypeName<U> (0) == TypeName<T1> (0)) {                              \
      std::cout << # U << " matches " << # T1  << std::endl; }              \
  else                                                                  \
    NS_TEST_ASSERT_MSG_EQ                                               \
      (TypeName<U> (0), TypeName<T1> (0),                               \
      "the typedef " << # U <<                                         \
      " used to match the typedef " << # T1 <<                         \
      " but no longer does.  Please add a new CHECK call.")


/**
 * \ingroup system-tests-traced
 * 
 * Check the TracedCallback by calling its Invoke function.
 */
#define CHECK(U, ...)                                    \
  CreateObject< Checker< __VA_ARGS__ > > ()->Invoke<U> ()


void
TracedCallbackTypedefTestCase::DoRun (void)
{

  CHECK (dsr::DsrOptionSRHeader::TracedCallback,
         const dsr::DsrOptionSRHeader &);

  CHECK (EpcUeNas::StateTracedCallback,
         EpcUeNas::State, EpcUeNas::State);

  CHECK (Ipv4L3Protocol::DropTracedCallback,
         const Ipv4Header &, Ptr<const Packet>,
         Ipv4L3Protocol::DropReason, Ptr<Ipv4>, uint32_t );

  CHECK (Ipv4L3Protocol::SentTracedCallback,
         const Ipv4Header &, Ptr<const Packet>, uint32_t);

  CHECK (Ipv4L3Protocol::TxRxTracedCallback,
         Ptr<const Packet>, Ptr<Ipv4>, uint32_t);

  CHECK (Ipv6L3Protocol::DropTracedCallback,
         const Ipv6Header &, Ptr<const Packet>,
         Ipv6L3Protocol::DropReason, Ptr<Ipv6>, uint32_t);

  CHECK (Ipv6L3Protocol::SentTracedCallback,
         const Ipv6Header &, Ptr<const Packet>, uint32_t);

  CHECK (Ipv6L3Protocol::TxRxTracedCallback,
         Ptr<const Packet>, Ptr<Ipv6>, uint32_t);

  CHECK (LrWpanMac::SentTracedCallback,
         Ptr<const Packet>, uint8_t, uint8_t);

  CHECK (LrWpanMac::StateTracedCallback,
         LrWpanMacState, LrWpanMacState);

  CHECK (LrWpanPhy::StateTracedCallback,
         Time, LrWpanPhyEnumeration, LrWpanPhyEnumeration);

  CHECK (LteEnbMac::DlSchedulingTracedCallback,
         uint32_t, uint32_t, uint16_t,
         uint8_t, uint16_t, uint8_t, uint16_t, uint8_t);

  CHECK (LteEnbMac::UlSchedulingTracedCallback,
         uint32_t, uint32_t, uint16_t, uint8_t, uint16_t);

  CHECK (LteEnbPhy::ReportUeSinrTracedCallback,
         uint16_t, uint16_t, double, uint8_t);

  CHECK (LteEnbPhy::ReportInterferenceTracedCallback,
         uint16_t, Ptr<SpectrumValue>);

  CHECK (LteEnbRrc::ConnectionHandoverTracedCallback,
         uint64_t, uint16_t, uint16_t);

  CHECK (LteEnbRrc::HandoverStartTracedCallback,
         uint64_t, uint16_t, uint16_t, uint16_t);

  CHECK (LteEnbRrc::NewUeContextTracedCallback,
         uint16_t, uint16_t);

  CHECK (LteEnbRrc::ReceiveReportTracedCallback,
         uint64_t, uint16_t, uint16_t, LteRrcSap::MeasurementReport);

  CHECK (LtePdcp::PduRxTracedCallback,
         uint16_t, uint8_t, uint32_t, uint64_t);

  CHECK (LtePdcp::PduTxTracedCallback,
         uint16_t, uint8_t, uint32_t);

  DUPE  (LteRlc::NotifyTxTracedCallback, LtePdcp::PduTxTracedCallback);

  DUPE  (LteRlc::ReceiveTracedCallback, LtePdcp::PduRxTracedCallback);

  CHECK (LteUePhy::RsrpSinrTracedCallback,
         uint16_t, uint16_t, double, double, uint8_t);

  CHECK (LteUePhy::StateTracedCallback,
         uint16_t, uint16_t, LteUePhy::State, LteUePhy::State);

  CHECK (LteUeRrc::CellSelectionTracedCallback,
         uint64_t, uint16_t);

  DUPE  (LteUeRrc::ImsiCidRntiTracedCallback, LteEnbRrc::ConnectionHandoverTracedCallback);

  DUPE   (LteUeRrc::MibSibHandoverTracedCallback, LteEnbRrc::HandoverStartTracedCallback);

  CHECK (LteUeRrc::StateTracedCallback,
         uint64_t, uint16_t, uint16_t, LteUeRrc::State, LteUeRrc::State);

  CHECK (Mac48Address::TracedCallback,
         Mac48Address);

  CHECK (MobilityModel::TracedCallback,
         Ptr<const MobilityModel>);

  CHECK (olsr::RoutingProtocol::PacketTxRxTracedCallback,
         const olsr::PacketHeader &, const olsr::MessageList &);

  CHECK (olsr::RoutingProtocol::TableChangeTracedCallback,
         uint32_t);

  CHECK (Packet::AddressTracedCallback,
         Ptr<const Packet>, const Address &);

  CHECK (Packet::Mac48AddressTracedCallback,
         Ptr<const Packet>, Mac48Address);

  CHECK (Packet::SinrTracedCallback,
         Ptr<const Packet>, double);

  CHECK (Packet::SizeTracedCallback,
         uint32_t, uint32_t);

  CHECK (Packet::TracedCallback,
         Ptr<const Packet>);

  CHECK (PacketBurst::TracedCallback,
         Ptr<const PacketBurst>);

  CHECK (dot11s::PeerManagementProtocol::LinkOpenCloseTracedCallback,
         Mac48Address, Mac48Address);

  CHECK (PhyReceptionStatParameters::TracedCallback,
         PhyReceptionStatParameters);

  CHECK (PhyTransmissionStatParameters::TracedCallback,
         PhyTransmissionStatParameters);

  CHECK (SixLowPanNetDevice::DropTracedCallback,
         SixLowPanNetDevice::DropReason, Ptr<const Packet>,
         Ptr<SixLowPanNetDevice>, uint32_t);

  CHECK (SixLowPanNetDevice::RxTxTracedCallback,
         Ptr<const Packet>, Ptr<SixLowPanNetDevice>, uint32_t);

  CHECK (SpectrumChannel::LossTracedCallback,
         Ptr<const SpectrumPhy>, Ptr<const SpectrumPhy>, double);

  CHECK (SpectrumValue::TracedCallback,
         Ptr<SpectrumValue>);

  CHECK (TimeSeriesAdaptor::OutputTracedCallback,
         double, double);

  CHECK (UanMac::PacketModeTracedCallback,
         Ptr<const Packet>, UanTxMode);

  CHECK (UanMacCw::QueueTracedCallback,
         Ptr<const Packet>, uint16_t);

  CHECK (UanMacRc::QueueTracedCallback,
         Ptr<const Packet>, uint32_t);

  CHECK (UanNetDevice::RxTxTracedCallback,
         Ptr<const Packet>, Mac8Address);

  CHECK (UanPhy::TracedCallback,
         Ptr<const Packet>, double, UanTxMode);

  CHECK (UeManager::StateTracedCallback,
         uint64_t, uint16_t, uint16_t, UeManager::State, UeManager::State);

  CHECK (WifiMacHeader::TracedCallback,
         const WifiMacHeader &);

  CHECK (WifiPhyStateHelper::RxEndErrorTracedCallback,
         Ptr<const Packet>, double);

  CHECK (WifiPhyStateHelper::RxOkTracedCallback,
         Ptr<const Packet>, double, WifiMode, WifiPreamble);

  CHECK (WifiPhyStateHelper::StateTracedCallback,
         Time, Time, WifiPhyState);

  CHECK (WifiPhyStateHelper::TxTracedCallback,
         Ptr<const Packet>, WifiMode, WifiPreamble, uint8_t);

  CHECK (WifiRemoteStationManager::PowerChangeTracedCallback,
         double, double, Mac48Address);

  CHECK (WifiRemoteStationManager::RateChangeTracedCallback,
         DataRate, DataRate, Mac48Address);
}

/**
 * \ingroup system-tests-traced
 *
 * \brief TracedCallback typedef TestSuite
 */
class TracedCallbackTypedefTestSuite : public TestSuite
{
public:
  TracedCallbackTypedefTestSuite ();
};

TracedCallbackTypedefTestSuite::TracedCallbackTypedefTestSuite ()
  : TestSuite ("traced-callback-typedef", SYSTEM)
{
  AddTestCase (new TracedCallbackTypedefTestCase, TestCase::QUICK);
}

/// Static variable for test initialization
static TracedCallbackTypedefTestSuite tracedCallbackTypedefTestSuite;
