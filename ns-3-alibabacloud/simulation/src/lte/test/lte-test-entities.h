/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#ifndef LTE_TEST_ENTITIES_H
#define LTE_TEST_ENTITIES_H

#include "ns3/simulator.h"
#include "ns3/test.h"

#include "ns3/lte-mac-sap.h"
#include "ns3/lte-rlc-sap.h"
#include "ns3/lte-pdcp-sap.h"

#include "ns3/net-device.h"
#include <ns3/epc-enb-s1-sap.h>

namespace ns3 {

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This class implements a testing RRC entity
 */
class LteTestRrc : public Object
{
    /// allow LtePdcpSpecificLtePdcpSapUser<LteTestRrc> class friend access
    friend class LtePdcpSpecificLtePdcpSapUser<LteTestRrc>;
//   friend class EnbMacMemberLteEnbCmacSapProvider;
//   friend class EnbMacMemberLteMacSapProvider<LteTestMac>;
//   friend class EnbMacMemberFfMacSchedSapUser;
//   friend class EnbMacMemberFfMacCschedSapUser;
//   friend class EnbMacMemberLteEnbPhySapUser;

  public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
    static TypeId GetTypeId (void);

    LteTestRrc (void);
    virtual ~LteTestRrc (void);
    virtual void DoDispose (void);


    /**
    * \brief Set the PDCP SAP provider
    * \param s a pointer to the PDCP SAP provider
    */
    void SetLtePdcpSapProvider (LtePdcpSapProvider* s);
    /**
    * \brief Get the PDCP SAP user
    * \return a pointer to the SAP user of the RLC
    */
    LtePdcpSapUser* GetLtePdcpSapUser (void);

    /// Start function
    void Start ();
    /// Stop function
    void Stop ();

    /**
    * \brief Send data function
    * \param at the time to send
    * \param dataToSend the data to send
    */
    void SendData (Time at, std::string dataToSend);
    /**
    * \brief Get data received function
    * \returns the received data string
    */
    std::string GetDataReceived (void);

    // Stats
    /**
    * \brief Get the transmit PDUs
    * \return the number of transmit PDUS
    */
    uint32_t GetTxPdus (void);
    /**
    * \brief Get the transmit bytes
    * \return the number of bytes transmitted
    */
    uint32_t GetTxBytes (void);
    /**
    * \brief Get the receive PDUs
    * \return the number of receive PDUS
    */
    uint32_t GetRxPdus (void);
    /**
    * \brief Get the receive bytes
    * \return the number of bytes received
    */
    uint32_t GetRxBytes (void);

    /**
    * \brief Get the last transmit time
    * \return the time of the last transmit
    */
    Time GetTxLastTime (void);
    /**
    * \brief Get the last receive time
    * \return the time of the last receive
    */
    Time GetRxLastTime (void);

    /**
    * \brief Set the arrival time
    * \param arrivalTime the arrival time
    */
    void SetArrivalTime (Time arrivalTime);
    /**
    * \brief Set the PDU size
    * \param pduSize the PDU size
    */
    void SetPduSize (uint32_t pduSize);

    /**
    * \brief Set the device
    * \param device the device
    */
    void SetDevice (Ptr<NetDevice> device);

  private:
    /**
     * Interface forwarded by LtePdcpSapUser
     * \param params the LtePdcpSapUser::ReceivePdcpSduParameters
     */
    virtual void DoReceivePdcpSdu (LtePdcpSapUser::ReceivePdcpSduParameters params);

    LtePdcpSapUser* m_pdcpSapUser; ///< PDCP SAP user
    LtePdcpSapProvider* m_pdcpSapProvider; ///< PDCP SAP provider

    std::string m_receivedData; ///< the received data

    uint32_t m_txPdus; ///< number of transmit PDUs
    uint32_t m_txBytes; ///< number of transmit bytes
    uint32_t m_rxPdus; ///< number of receive PDUs
    uint32_t m_rxBytes; ///< number of receive bytes
    Time     m_txLastTime; ///< last transmit time
    Time     m_rxLastTime; ///< last reeive time

    EventId m_nextPdu; ///< next PDU event
    Time m_arrivalTime; ///< next arrival time
    uint32_t m_pduSize; ///< PDU size

    Ptr<NetDevice> m_device; ///< the device
};

/////////////////////////////////////////////////////////////////////

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This class implements a testing PDCP entity
 */
class LteTestPdcp : public Object
{
  /// allow LteRlcSpecificLteRlcSapUser<LteTestPdcp> class friend access
  friend class LteRlcSpecificLteRlcSapUser<LteTestPdcp>;
  
  public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
    static TypeId GetTypeId (void);

    LteTestPdcp (void);
    virtual ~LteTestPdcp (void);
    virtual void DoDispose (void);


    /**
    * \brief Set the RLC SAP provider
    * \param s a pointer to the RLC SAP provider
    */
    void SetLteRlcSapProvider (LteRlcSapProvider* s);
    /**
    * \brief Get the RLC SAP user
    * \return a pointer to the SAP user of the RLC
    */
    LteRlcSapUser* GetLteRlcSapUser (void);

    /// Start function
    void Start ();

    /**
    * \brief Send data function
    * \param time the time to send
    * \param dataToSend the data to send
    */
    void SendData (Time time, std::string dataToSend);
    /**
    * \brief Get data received function
    * \returns the received data string
    */
    std::string GetDataReceived (void);

  private:
    /**
     * Interface forwarded by LteRlcSapUser
     * \param p the PDCP PDU packet received
     */
    virtual void DoReceivePdcpPdu (Ptr<Packet> p);

    LteRlcSapUser* m_rlcSapUser; ///< RLC SAP user
    LteRlcSapProvider* m_rlcSapProvider; ///< RLC SAP provider

    std::string m_receivedData; ///< the received data 
};

/////////////////////////////////////////////////////////////////////

/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief This class implements a testing loopback MAC layer
 */
class LteTestMac : public Object
{
//   friend class EnbMacMemberLteEnbCmacSapProvider;
    /// allow EnbMacMemberLteMacSapProvider<LteTestMac> class friend access
    friend class EnbMacMemberLteMacSapProvider<LteTestMac>;
//   friend class EnbMacMemberFfMacSchedSapUser;
//   friend class EnbMacMemberFfMacCschedSapUser;
//   friend class EnbMacMemberLteEnbPhySapUser;

  public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
    static TypeId GetTypeId (void);

    LteTestMac (void);
    virtual ~LteTestMac (void);
    virtual void DoDispose (void);

    /**
    * \brief Set the device function
    * \param device the device
    */
    void SetDevice (Ptr<NetDevice> device);

    /**
    * \brief Send transmit opportunity function
    * \param time the time
    * \param bytes the number of bytes
    */
    void SendTxOpportunity (Time time, uint32_t bytes);
    /**
    * \brief Get data received function
    * \returns the received data string
    */
    std::string GetDataReceived (void);

    /**
    * \brief the Receive function
    * \param nd the device
    * \param p the packet
    * \param protocol the protocol
    * \param addr the address
    * \returns true if successful
    */
    bool Receive (Ptr<NetDevice> nd, Ptr<const Packet> p, uint16_t protocol, const Address& addr);

    /**
     * \brief Set the MAC SAP user
     * \param s a pointer to the MAC SAP user
     */
    void SetLteMacSapUser (LteMacSapUser* s);
    /**
     * \brief Get the MAC SAP provider
     * \return a pointer to the SAP provider of the MAC
     */
    LteMacSapProvider* GetLteMacSapProvider (void);

    /**
     * \brief Set the other side of the MAC Loopback
     * \param s a pointer to the other side of the MAC loopback
     */
    void SetLteMacLoopback (Ptr<LteTestMac> s);

    /**
     * \brief Set PDCP header present function
     * \param present true iif PDCP header present
     */
    void SetPdcpHeaderPresent (bool present);

    /**
     * \brief Set RLC header type
     * \param rlcHeaderType the RLC header type
     */
    void SetRlcHeaderType (uint8_t rlcHeaderType);

    /// RCL Header Type enumeration
    typedef enum {
      UM_RLC_HEADER = 0,
      AM_RLC_HEADER = 1,
    } RlcHeaderType_t; ///< the RLC header type

    /**
     * Set transmit opportunity mode
     * \param mode the transmit opportunity mode
     */
    void SetTxOpportunityMode (uint8_t mode);

    /// Transmit opportunity mode enumeration
    typedef enum {
      MANUAL_MODE     = 0,
      AUTOMATIC_MODE  = 1,
      RANDOM_MODE     = 2
    } TxOpportunityMode_t; ///< transmit opportunity mode

    /**
     * Set transmit opportunity time
     * \param txOppTime the transmit opportunity time
     */
    void SetTxOppTime (Time txOppTime);
    /**
     * Set transmit opportunity time
     * \param txOppSize the transmit opportunity size
     */
    void SetTxOppSize (uint32_t txOppSize);

    // Stats
    /**
    * \brief Get the transmit PDUs
    * \return the number of transmit PDUS
    */
    uint32_t GetTxPdus (void);
    /**
    * \brief Get the transmit bytes
    * \return the number of bytes transmitted
    */
    uint32_t GetTxBytes (void);
    /**
    * \brief Get the receive PDUs
    * \return the number of receive PDUS
    */
    uint32_t GetRxPdus (void);
    /**
    * \brief Get the receive bytes
    * \return the number of bytes received
    */
    uint32_t GetRxBytes (void);

  private:
    // forwarded from LteMacSapProvider
    /**
     * Transmit PDU
     * \param params LteMacSapProvider::TransmitPduParameters
     */
    void DoTransmitPdu (LteMacSapProvider::TransmitPduParameters params);
    /**
     * Report buffer status function
     * \param params LteMacSapProvider::ReportBufferStatusParameters
     */
    void DoReportBufferStatus (LteMacSapProvider::ReportBufferStatusParameters params);

    LteMacSapProvider* m_macSapProvider; ///< MAC SAP provider
    LteMacSapUser* m_macSapUser; ///< MAC SAP user
    Ptr<LteTestMac> m_macLoopback; ///< MAC loopback

    std::string m_receivedData; ///< the received data string

    uint8_t m_rlcHeaderType; ///< RLC header type
    bool m_pdcpHeaderPresent; ///< PDCP header present?
    uint8_t m_txOpportunityMode; ///< transmit opportunity mode

    Ptr<NetDevice> m_device; ///< the device

    // TxOpportunity configuration
    EventId m_nextTxOpp; ///< next transmit opportunity event
    Time m_txOppTime; ///< transmit opportunity time
    uint32_t m_txOppSize; ///< transmit opportunity size
    std::list<EventId> m_nextTxOppList; ///< next transmit opportunity list

    // Stats
    uint32_t m_txPdus; ///< the number of transmit PDUs
    uint32_t m_txBytes; ///< the number of transmit bytes
    uint32_t m_rxPdus; ///< the number of receive PDUs
    uint32_t m_rxBytes; ///< the number of receive bytes

};



/**
 * \ingroup lte-test
 * \ingroup tests
 *
 * \brief RRC stub providing a testing S1 SAP user to be used with the EpcEnbApplication
 */
class EpcTestRrc : public Object
{
  /// allow MemberEpcEnbS1SapUser<EpcTestRrc> class friend access
  friend class MemberEpcEnbS1SapUser<EpcTestRrc>;

public:
  EpcTestRrc ();
  virtual ~EpcTestRrc ();

  // inherited from Object
  virtual void DoDispose (void);
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /** 
   * Set the S1 SAP Provider
   * 
   * \param s the S1 SAP Provider
   */
  void SetS1SapProvider (EpcEnbS1SapProvider* s);

  /** 
   * 
   * \return the S1 SAP user
   */
  EpcEnbS1SapUser* GetS1SapUser ();

private:

  // S1 SAP methods
  /**
   * Initial context setup request
   * \param params EpcEnbS1SapUser::InitialContextSetupRequestParameters
   */
  void DoInitialContextSetupRequest (EpcEnbS1SapUser::InitialContextSetupRequestParameters params);
  /**
   * Data radio bearer setup request
   * \param params EpcEnbS1SapUser::DataRadioBearerSetupRequestParameters
   */
  void DoDataRadioBearerSetupRequest (EpcEnbS1SapUser::DataRadioBearerSetupRequestParameters params);
  /**
   * Path switch request acknowledge function
   * \param params EpcEnbS1SapUser::PathSwitchRequestAcknowledgeParameters
   */
  void DoPathSwitchRequestAcknowledge (EpcEnbS1SapUser::PathSwitchRequestAcknowledgeParameters params);  
  
  EpcEnbS1SapProvider* m_s1SapProvider; ///< S1 SAP provider
  EpcEnbS1SapUser* m_s1SapUser; ///< S1 SAP user
  

};


} // namespace ns3

#endif /* LTE_TEST_MAC_H */
