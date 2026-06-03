/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#ifndef WIMAX_SS_NET_DEVICE_H
#define WIMAX_SS_NET_DEVICE_H

#include "ns3/event-id.h"
#include "wimax-net-device.h"
#include "ns3/nstime.h"
#include "wimax-mac-header.h"
#include "ns3/uinteger.h"
#include "ns3/ss-service-flow-manager.h"
#include "ipcs-classifier.h"

namespace ns3 {

class Node;
class OfdmDlBurstProfile;
class OfdmUlBurstProfile;
class SSScheduler;
class SSLinkManager;
class SsServiceFlowManager;
class IpcsClassifier;

/**
 * \ingroup wimax
 * SubscriberStationNetDevice subclass of WimaxNetDevice
 */
class SubscriberStationNetDevice : public WimaxNetDevice
{
public:
  /// State enumeration
  enum State
  {
    SS_STATE_IDLE,
    SS_STATE_SCANNING,
    SS_STATE_SYNCHRONIZING,
    SS_STATE_ACQUIRING_PARAMETERS,
    SS_STATE_WAITING_REG_RANG_INTRVL, // regular ranging interval
    SS_STATE_WAITING_INV_RANG_INTRVL, // invited ranging interval
    SS_STATE_WAITING_RNG_RSP,
    SS_STATE_ADJUSTING_PARAMETERS,
    SS_STATE_REGISTERED,
    SS_STATE_TRANSMITTING, // currently not being used anywhere
    SS_STATE_STOPPED
  };

  /// EventType enumeration
  enum EventType
  {
    EVENT_NONE,
    EVENT_WAIT_FOR_RNG_RSP,
    EVENT_DL_MAP_SYNC_TIMEOUT,
    EVENT_LOST_DL_MAP,
    EVENT_LOST_UL_MAP,
    EVENT_DCD_WAIT_TIMEOUT,
    EVENT_UCD_WAIT_TIMEOUT,
    EVENT_RANG_OPP_WAIT_TIMEOUT
  };

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  SubscriberStationNetDevice (void);
  /**
   * Constructor
   *
   * \param node the node
   * \param phy the wimax phy
   */
  SubscriberStationNetDevice (Ptr<Node> node, Ptr<WimaxPhy> phy);
  ~SubscriberStationNetDevice (void);

  /**
   * \brief initializes the net device and sets the parameters to the default values
   */
  void InitSubscriberStationNetDevice (void);
  /**
   * \param lostDlMapInterval time since last received DL-MAP message before downlink synchronization is considered lost
   */
  void SetLostDlMapInterval (Time lostDlMapInterval);
  /**
   * \returns the time since last received DL-MAP message before downlink synchronization is considered lost
   */
  Time GetLostDlMapInterval (void) const;
  /**
   * \param lostUlMapInterval Time since last received UL-MAP before uplink synchronization is considered lost
   */
  void SetLostUlMapInterval (Time lostUlMapInterval);
  /**
   * \returns the time since last received UL-MAP before uplink synchronization is considered lost
   */
  Time GetLostUlMapInterval (void) const;
  /**
   * \param maxDcdInterval Maximum time between transmission of DCD messages
   */
  void SetMaxDcdInterval (Time maxDcdInterval);
  /**
   * \returns the maximum time between transmission of DCD messages
   */
  Time GetMaxDcdInterval (void) const;
  /**
   * \param maxUcdInterval Maximum time between transmission of UCD messages
   */
  void SetMaxUcdInterval (Time maxUcdInterval);
  /**
   * \returns the maximum time between transmission of UCD messages
   */
  Time GetMaxUcdInterval (void) const;
  /**
   * \param interval1 Wait for DCD timeout value
   */
  void SetIntervalT1 (Time interval1);
  /**
   * \returns the wait for DCD timeout
   */
  Time GetIntervalT1 (void) const;
  /**
   * \param interval2 the wait for broadcast ranging timeout, i.e., wait for initial ranging opportunity
   */
  void SetIntervalT2 (Time interval2);
  /**
   * \returns the wait for broadcast ranging timeout, i.e., wait for initial ranging opportunity
   */
  Time GetIntervalT2 (void) const;
  /**
   * \param interval3 the ranging Response reception timeout following the transmission of a ranging request
   */
  void SetIntervalT3 (Time interval3);
  /**
   * \returns the ranging Response reception timeout following the transmission of a ranging request
   */
  Time GetIntervalT3 (void) const;
  /**
   * \param interval7 the wait for DSA/DSC/DSD Response timeout
   */
  void SetIntervalT7 (Time interval7);
  /**
   * \returns the wait for DSA/DSC/DSD Response timeout
   */
  Time GetIntervalT7 (void) const;
  /**
   * \param interval12 the Wait for UCD descriptor timeout
   */
  void SetIntervalT12 (Time interval12);
  /**
   * \returns the wait for UCD descriptor timeout
   */
  Time GetIntervalT12 (void) const;
  /**
   * \param interval20 the Time the SS searches for preambles on a given channel
   */
  void SetIntervalT20 (Time interval20);
  /**
   * \returns the Time the SS searches for preambles on a given channel
   */
  Time GetIntervalT20 (void) const;
  /**
   * \param interval21 the time the SS searches for (decodable) DL-MAP on a given channel
   */
  void SetIntervalT21 (Time interval21);
  /**
   * \returns the time the SS searches for (decodable) DL-MAP on a given channel
   */
  Time GetIntervalT21 (void) const;
  /**
   * \param maxContentionRangingRetries the Number of retries on contention Ranging Requests
   */
  void SetMaxContentionRangingRetries (uint8_t maxContentionRangingRetries);
  /**
   * \returns the Number of retries on contention Ranging Requests
   */
  uint8_t GetMaxContentionRangingRetries (void) const;
  /**
   *  \param basicConnection the basic connection to be used
   */
  void SetBasicConnection (Ptr<WimaxConnection> basicConnection);
  /**
   * \return the basic connection currently in use
   */
  Ptr<WimaxConnection> GetBasicConnection (void) const;
  /**
   * \param primaryConnection the primary connection to be used
   */
  void SetPrimaryConnection (Ptr<WimaxConnection> primaryConnection);
  /**
   * \returns the primary connection currently used
   */
  Ptr<WimaxConnection> GetPrimaryConnection (void) const;
  /**
   * \returns the basic CID
   */
  Cid GetBasicCid (void) const;
  /**
   * \returns the primary CID
   */
  Cid GetPrimaryCid (void) const;

  /**
   * \brief Set the most efficient modulation and coding scheme (MCS) supported by the device
   * \param modulationType the most robust MCS supported
   */
  void SetModulationType (WimaxPhy::ModulationType modulationType);
  /**
   * \returns the most efficient modulation and coding scheme (MCS) supported by the device
   */
  WimaxPhy::ModulationType GetModulationType (void) const;
  /**
   * \param areManagementConnectionsAllocated true if the management connections are allocated, false otherwise
   */
  void SetAreManagementConnectionsAllocated (bool areManagementConnectionsAllocated);
  /**
   * \returns true if the management connections are allocated, false otherwise
   */
  bool GetAreManagementConnectionsAllocated (void) const;
  /**
   * \param areServiceFlowsAllocated true if the service flows are allocated, false otherwise
   */
  void SetAreServiceFlowsAllocated (bool areServiceFlowsAllocated);
  /**
   * \returns true if the service flows are allocated, false otherwise
   */
  bool GetAreServiceFlowsAllocated (void) const;
  /**
   * \return the scheduler installed on the device
   */
  Ptr<SSScheduler> GetScheduler (void) const;
  /**
   * \param ssScheduler the scheduler to be installed on the device
   */
  void SetScheduler (Ptr<SSScheduler> ssScheduler);
  /**
   * \returns true if the device has at least one active service flow, false otherwise
   */
  bool HasServiceFlows (void) const;
  /**
   * \brief Enqueue a packet into a connection queue
   * \param packet the packet to be enqueued
   * \param hdrType the mac header type to be appended to the packet
   * \param connection the connection to be used
   * \returns true if successful
   */
  bool Enqueue (Ptr<Packet> packet, const MacHeaderType &hdrType, Ptr<WimaxConnection> connection);
  /**
   * \brief Sends a burst on the uplink frame
   * \param uiuc theOfdmUlBurstProfile
   * \param nrSymbols number of symbols
   * \param connection connection to use
   * \param packetType optional HeaderType (default HEADER_TYPE_GENERIC)
   */
  void SendBurst (uint8_t uiuc,
                  uint16_t nrSymbols,
                  Ptr<WimaxConnection> connection,
                  MacHeaderType::HeaderType packetType = MacHeaderType::HEADER_TYPE_GENERIC);

  /**
   * \brief Start the device
   */
  void Start (void);
  /**
   * \brief Stop the device
   */
  void Stop (void);

  /**
   * \brief adds a new service flow
   * \param sf pointer to service flow to add
   */
  void AddServiceFlow (ServiceFlow *sf);
  /**
   * \brief adds a new service flow
   * \param sf service flow to add
   */
  void AddServiceFlow (ServiceFlow sf);
  /**
   * \brief Set timer.  If in stopped state, the EventId passed in the first 
   * argument will be cancelled.  If not, the second parameter will be set
   * to the value of the first parameter.  
   *
   * \param eventId EventId to cancel or to map to the second argument
   * \param event EventId is set to first argument if not in stopped state
   */
  void SetTimer (EventId eventId, EventId &event);
  /**
   * \returns true if the SS is registered to a BS, false otherwise
   */
  bool IsRegistered (void) const;
  /**
   * \brief Get time to allocation
   * \param deferTime defer time
   * \returns the time to allocation
   */
  Time GetTimeToAllocation (Time deferTime);

  Ptr<SSLinkManager> m_linkManager; ///< link manager
  /**
   * \returns a pointer to the CS packet classifier
   */
  Ptr<IpcsClassifier> GetIpcsClassifier () const;
  /**
   * \brief Sets the packet classifier to be used
   * \param classifier the classifier to use
   */
  void SetIpcsPacketClassifier (Ptr<IpcsClassifier> classifier);
  /**
   * \returns a pointer to the link manager currently used
   */
  Ptr<SSLinkManager> GetLinkManager (void) const;
  /**
   * \brief sets the link manager to be used
   * \param linkManager link manager to use
   */
  void SetLinkManager (Ptr<SSLinkManager> linkManager);
  /**
   * \returns a pointer to the service flow manager installed on the device
   */
  Ptr<SsServiceFlowManager> GetServiceFlowManager (void) const;
  /**
   * \brief Sets the service flow manager to be installed on the device
   * \param sfm service flow manager to be used
   */
  void SetServiceFlowManager (Ptr<SsServiceFlowManager> sfm);

  /// Callback for logging packets on ASCII traces
  typedef Callback<void, std::string, Ptr<const Packet> > AsciiTraceCallback;

  /**
   * Set the Enqueue callback for ASCII tracing
   * \param cb callback for ASCII tracing
   */
  void SetAsciiTxQueueEnqueueCallback (AsciiTraceCallback cb);

  /**
   * Set the Dequeue callback for ASCII tracing
   * \param cb callback for ASCII tracing
   */
  void SetAsciiTxQueueDequeueCallback (AsciiTraceCallback cb);

  /**
   * Set the Drop callback for ASCII tracing
   * \param cb callback for ASCII tracing
   */
  void SetAsciiTxQueueDropCallback (AsciiTraceCallback cb);

private:
  /**
   * Get default lost DL map interval
   * \returns the default lost DL map interval
   */
  static Time GetDefaultLostDlMapInterval ();

  void DoDispose (void);
  /**
   * Send a packet
   * \param packet the packet to send
   * \param source the source MAC address
   * \param dest the destination MAC address
   * \param protocolNumber the protocol number
   * \returns true if successful
   */
  bool DoSend (Ptr<Packet> packet, const Mac48Address &source, const Mac48Address &dest, uint16_t protocolNumber);
  /**
   * Receive a packet
   * \param packet the packet received
   */
  void DoReceive (Ptr<Packet> packet);

  /**
   * Process DL map
   * \param dlmap the DL map
   */
  void ProcessDlMap (const DlMap &dlmap);
  /**
   * Process UL map
   * \param ulmap the UL map
   */
  void ProcessUlMap (const UlMap &ulmap);
  /**
   * Process DCD
   * \param dcd the DCD
   */
  void ProcessDcd (const Dcd &dcd);
  /**
   * Process UCD
   * \param ucd the UCD
   */
  void ProcessUcd (const Ucd &ucd);

  // parameters defined in Table 342
  Time m_lostDlMapInterval; ///< in milliseconds, time since last received DL-MAP before downlink synchronization is considered lost, maximum is 600
  Time m_lostUlMapInterval; ///< in milliseconds, time since last received UL-MAP before uplink synchronization is considered lost, maximum is 600
  Time m_maxDcdInterval; ///< in seconds, maximum time between transmission of DCD messages
  Time m_maxUcdInterval; ///< in seconds, maximum time between transmission of UCD messages
  Time m_intervalT1; ///< in seconds, wait for DCD timeout
  Time m_intervalT2; ///< in seconds, wait for broadcast ranging timeout, i.e., wait for initial ranging opportunity
  Time m_intervalT3; ///< in milliseconds, ranging Response reception timeout following the transmission of a ranging request
  Time m_intervalT7; ///< in seconds, wait for DSA/DSC/DSD Response timeout
  Time m_intervalT12; ///< in seconds, wait for UCD descriptor
  Time m_intervalT20; ///< in seconds, time the SS searches for preambles on a given channel
  Time m_intervalT21; ///< in seconds, time the SS searches for (decodable) DL-MAP on a given channel
  uint8_t m_maxContentionRangingRetries; ///< maximum contention ranging retries

  // parameters obtained from DL-MAP
  uint8_t m_dcdCount; ///< DCD count
  Mac48Address m_baseStationId; ///< base station ID

  // parameters obtained from UL-MAP
  uint8_t m_ucdCount; ///< UCD count
  double m_allocationStartTime; ///< allocation start time

  // to keep the number of DL-MAP/UL-MAP IEs found in the last DL-MAP/U-MAP messages
  uint16_t m_nrDlMapElements; ///< number DL Map elements
  uint16_t m_nrUlMapElements; ///< number UL Map elements

  Ptr<WimaxConnection> m_basicConnection; ///< basic connection
  Ptr<WimaxConnection> m_primaryConnection; ///< primary connection

  EventId m_lostDlMapEvent; ///< lost DL map event
  EventId m_lostUlMapEvent; ///< lost UL map event
  EventId m_dcdWaitTimeoutEvent; ///< DCD wait timeout event
  EventId m_ucdWaitTimeoutEvent; ///< UCD wait timeout event
  EventId m_rangOppWaitTimeoutEvent; ///< range opp wait timeout event

  uint32_t m_nrDlMapRecvd; ///< number DL map received
  uint32_t m_nrUlMapRecvd; ///< number UL map received
  uint32_t m_nrDcdRecvd; ///< number DCD received
  uint32_t m_nrUcdRecvd; ///< number UCD received

  OfdmDlBurstProfile *m_dlBurstProfile; ///< DL burst profile
  OfdmUlBurstProfile *m_ulBurstProfile; ///< UL burst profile

  /*represents the (least robust) modulation type of the SS which it then requests in RNG-REQ and if accepted by BS uses it for receiving and
   transmitting. currently it is set by user in simulation script, shall actually be determined based on SS's distance, power, signal etc*/
  WimaxPhy::ModulationType m_modulationType; ///< modulation type

  bool m_areManagementConnectionsAllocated; ///< are management connections allocated
  bool m_areServiceFlowsAllocated; ///< are service flows allocated

  Ptr<SSScheduler> m_scheduler; ///< the scheduler
  Ptr<SsServiceFlowManager> m_serviceFlowManager; ///< the service flow manager
  Ptr<IpcsClassifier> m_classifier; ///< the classifier

  TracedCallback<Ptr<const Packet>, Mac48Address, const Cid &> m_traceSSRx; ///< trace SS receive callback

  /**
   * The trace source fired when packets come into the "top" of the device
   * at the L3/L2 transition, before being queued for transmission.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_ssTxTrace;

  /**
   * The trace source fired when packets coming into the "top" of the device
   * are dropped at the MAC layer during transmission.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_ssTxDropTrace;

  /**
   * The trace source fired for packets successfully received by the device
   * immediately before being forwarded up to higher layers (at the L2/L3
   * transition).  This is a promiscuous trace.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_ssPromiscRxTrace;

  /**
   * The trace source fired for packets successfully received by the device
   * immediately before being forwarded up to higher layers (at the L2/L3
   * transition).  This is a non- promiscuous trace.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_ssRxTrace;

  /**
   * The trace source fired when packets coming into the "top" of the device
   * are dropped at the MAC layer during reception.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_ssRxDropTrace;

  /// Bound callback to perform ASCII logging for Enqueue events
  AsciiTraceCallback m_asciiTxQueueEnqueueCb;
  /// Bound callback to perform ASCII logging for Dequeue events
  AsciiTraceCallback m_asciiTxQueueDequeueCb;
  /// Bound callback to perform ASCII logging for Drop events
  AsciiTraceCallback m_asciiTxQueueDropCb;
};

} // namespace ns3

#endif /* WIMAX_SS_NET_DEVICE_H */
