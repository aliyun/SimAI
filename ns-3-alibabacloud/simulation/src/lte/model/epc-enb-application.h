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
 * Author: Jaume Nin <jnin@cttc.cat>
 *         Nicola Baldo <nbaldo@cttc.cat>
 */

#ifndef EPC_ENB_APPLICATION_H
#define EPC_ENB_APPLICATION_H

#include <ns3/address.h>
#include <ns3/socket.h>
#include <ns3/virtual-net-device.h>
#include <ns3/traced-callback.h>
#include <ns3/callback.h>
#include <ns3/ptr.h>
#include <ns3/object.h>
#include <ns3/lte-common.h>
#include <ns3/application.h>
#include <ns3/eps-bearer.h>
#include <ns3/epc-enb-s1-sap.h>
#include <ns3/epc-s1ap-sap.h>
#include <map>

namespace ns3 {
class EpcEnbS1SapUser;
class EpcEnbS1SapProvider;


/**
 * \ingroup lte
 *
 * This application is installed inside eNBs and provides the bridge functionality for user data plane packets between the radio interface and the S1-U interface.
 */
class EpcEnbApplication : public Application
{

  /// allow MemberEpcEnbS1SapProvider<EpcEnbApplication> class friend access
  friend class MemberEpcEnbS1SapProvider<EpcEnbApplication>;
  /// allow MemberEpcS1apSapEnb<EpcEnbApplication> class friend access
  friend class MemberEpcS1apSapEnb<EpcEnbApplication>;


public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
protected:
  void DoDispose (void);

public:
  
  

  /** 
   * Constructor
   * 
   * \param lteSocket the socket to be used to send/receive IPv4 packets to/from the LTE radio interface
   * \param lteSocket6 the socket to be used to send/receive IPv6 packets to/from the LTE radio interface
   * \param cellId the identifier of the eNB
   */
  EpcEnbApplication (Ptr<Socket> lteSocket, Ptr<Socket> lteSocket6, uint16_t cellId);

  /**
   * Add a S1-U interface to the eNB
   *
   * \param s1uSocket the socket to be used to send/receive packets to/from the S1-U interface connected with the SGW
   * \param enbS1uAddress the IPv4 address of the S1-U interface of this eNB
   * \param sgwS1uAddress the IPv4 address at which this eNB will be able to reach its SGW for S1-U communications
   */
  void AddS1Interface (Ptr<Socket> s1uSocket, Ipv4Address enbS1uAddress, Ipv4Address sgwS1uAddress);


  /**
   * Destructor
   * 
   */
  virtual ~EpcEnbApplication (void);


  /** 
   * Set the S1 SAP User
   * 
   * \param s the S1 SAP User
   */
  void SetS1SapUser (EpcEnbS1SapUser * s);

  /** 
   * 
   * \return the S1 SAP Provider
   */
  EpcEnbS1SapProvider* GetS1SapProvider ();

  /** 
   * Set the MME side of the S1-AP SAP 
   * 
   * \param s the MME side of the S1-AP SAP 
   */
  void SetS1apSapMme (EpcS1apSapMme * s);

  /** 
   * 
   * \return the ENB side of the S1-AP SAP 
   */
  EpcS1apSapEnb* GetS1apSapEnb ();
 
  /** 
   * Method to be assigned to the recv callback of the LTE socket. It is called when the eNB receives a data packet from the radio interface that is to be forwarded to the SGW.
   * 
   * \param socket pointer to the LTE socket
   */
  void RecvFromLteSocket (Ptr<Socket> socket);


  /** 
   * Method to be assigned to the recv callback of the S1-U socket. It is called when the eNB receives a data packet from the SGW that is to be forwarded to the UE.
   * 
   * \param socket pointer to the S1-U socket
   */
  void RecvFromS1uSocket (Ptr<Socket> socket);

  /**
   * TracedCallback signature for data Packet reception event.
   *
   * \param [in] packet The data packet sent from the internet.
   */
  typedef void (* RxTracedCallback)
    (Ptr<Packet> packet);


  /**
   * EPS flow ID structure
   */
  struct EpsFlowId_t
  {
    uint16_t  m_rnti; ///< RNTI
    uint8_t   m_bid; ///< Bid, the EPS Bearer IDentifier

  public:
    EpsFlowId_t ();
    /**
     * Constructor
     *
     * \param a RNTI
     * \param b bid
     */
    EpsFlowId_t (const uint16_t a, const uint8_t b);

    /**
     * Comparison operator
     *
     * \param a first application
     * \param b second application
     * \returns true is the applications are "equal"
     */
    friend bool operator == (const EpsFlowId_t &a, const EpsFlowId_t &b);
    /**
     * Less than operator
     *
     * \param a first application
     * \param b second application
     * \returns true is the applications are "equal"
     */
    friend bool operator < (const EpsFlowId_t &a, const EpsFlowId_t &b);
  };


private:

  // ENB S1 SAP provider methods
  /**
   * Initial UE message function
   * \param imsi the IMSI
   * \param rnti the RNTI
   */
  void DoInitialUeMessage (uint64_t imsi, uint16_t rnti);
  /**
   * Path switch request function
   * \param params PathSwitchRequestParameters
   */
  void DoPathSwitchRequest (EpcEnbS1SapProvider::PathSwitchRequestParameters params);
  /**
   * UE Context Release function
   * \param rnti the RNTI
   */
  void DoUeContextRelease (uint16_t rnti);
  
  // S1-AP SAP ENB methods
  /**
   * Initial Context Setup Request 
   * \param mmeUeS1Id the MME UE S1 ID
   * \param enbUeS1Id the ENB UE S1 ID
   * \param erabToBeSetupList the ERAB setup list
   */
  void DoInitialContextSetupRequest (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<EpcS1apSapEnb::ErabToBeSetupItem> erabToBeSetupList);
  /**
   * Path Switch Request Acknowledge 
   * \param mmeUeS1Id the MME UE S1 ID
   * \param enbUeS1Id the ENB UE S1 ID
   * \param cgi the CGI
   * \param erabToBeSwitchedInUplinkList the ERAB switched in uplink list
   */
  void DoPathSwitchRequestAcknowledge (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t cgi, std::list<EpcS1apSapEnb::ErabSwitchedInUplinkItem> erabToBeSwitchedInUplinkList);

  /** 
   * \brief This function accepts bearer id corresponding to a particular UE and schedules indication of bearer release towards MME
   * \param imsi maps to mmeUeS1Id
   * \param rnti maps to enbUeS1Id
   * \param bearerId Bearer Identity which is to be de-activated
   */
  void DoReleaseIndication (uint64_t imsi, uint16_t rnti, uint8_t bearerId);


  /**
   * Send a packet to the UE via the LTE radio interface of the eNB
   * 
   * \param packet t
   * \param rnti maps to enbUeS1Id
   * \param bid the EPS Bearer IDentifier
   */
  void SendToLteSocket (Ptr<Packet> packet, uint16_t rnti, uint8_t bid);


  /** 
   * Send a packet to the SGW via the S1-U interface
   * 
   * \param packet packet to be sent
   * \param teid the Tunnel Enpoint IDentifier
   */
  void SendToS1uSocket (Ptr<Packet> packet, uint32_t teid);


  /** 
   * internal method used for the actual setup of the S1 Bearer
   * 
   * \param teid the Tunnel Endpoint IDentifier
   * \param rnti maps to enbUeS1Id
   * \param bid the S1-U Bearer IDentifier
   */
  void SetupS1Bearer (uint32_t teid, uint16_t rnti, uint8_t bid);

  /**
   * raw packet socket to send and receive the packets to and from the LTE radio interface
   */
  Ptr<Socket> m_lteSocket;

  /**
   * raw packet socket to send and receive the packets to and from the LTE radio interface
   */
  Ptr<Socket> m_lteSocket6;

  /**
   * UDP socket to send and receive GTP-U the packets to and from the S1-U interface
   */
  Ptr<Socket> m_s1uSocket;

  /**
   * address of the eNB for S1-U communications
   */
  Ipv4Address m_enbS1uAddress;

  /**
   * address of the SGW which terminates all S1-U tunnels
   */
  Ipv4Address m_sgwS1uAddress;

  /**
   * map of maps telling for each RNTI and BID the corresponding  S1-U TEID
   * 
   */
  std::map<uint16_t, std::map<uint8_t, uint32_t> > m_rbidTeidMap;  

  /**
   * map telling for each S1-U TEID the corresponding RNTI,BID
   * 
   */
  std::map<uint32_t, EpsFlowId_t> m_teidRbidMap;
 
  /**
   * UDP port to be used for GTP
   */
  uint16_t m_gtpuUdpPort;

  /**
   * Provider for the S1 SAP 
   */
  EpcEnbS1SapProvider* m_s1SapProvider;

  /**
   * User for the S1 SAP 
   */
  EpcEnbS1SapUser* m_s1SapUser;

  /**
   * MME side of the S1-AP SAP
   * 
   */
  EpcS1apSapMme* m_s1apSapMme;

  /**
   * ENB side of the S1-AP SAP
   * 
   */
  EpcS1apSapEnb* m_s1apSapEnb;

  /**
   * UE context info
   * 
   */
  std::map<uint64_t, uint16_t> m_imsiRntiMap;

  uint16_t m_cellId; ///< cell ID

  /**
   * \brief Callback to trace RX (reception) data packets from LTE Socket.
   */ 
  TracedCallback<Ptr<Packet> > m_rxLteSocketPktTrace;

  /**
   * \brief Callback to trace RX (reception) data packets from S1-U Socket.
   */ 
  TracedCallback<Ptr<Packet> > m_rxS1uSocketPktTrace;
};

} //namespace ns3

#endif /* EPC_ENB_APPLICATION_H */

