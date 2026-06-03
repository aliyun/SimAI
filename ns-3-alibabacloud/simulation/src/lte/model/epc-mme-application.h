/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017-2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

#ifndef EPC_MME_APPLICATION_H
#define EPC_MME_APPLICATION_H

#include "ns3/application.h"
#include "ns3/socket.h"
#include "ns3/epc-s1ap-sap.h"

namespace ns3 {

/**
 * \ingroup lte
 *
 * This application implements the Mobility Management Entity (MME) according to
 * the 3GPP TS 23.401 document.
 *
 * This Application implements the MME side of the S1-MME interface between
 * the MME node and the eNB nodes and the MME side of the S11 interface between
 * the MME node and the SGW node. It supports the following functions and messages:
 *
 *  - Bearer management functions including dedicated bearer establishment
 *  - NAS signalling
 *  - Tunnel Management messages
 *
 * Others functions enumerated in section 4.4.2 of 3GPP TS 23.401 are not supported.
 */
class EpcMmeApplication : public Application
{
  /// allow MemberEpcS1apSapMme<EpcMme> class friend access
  friend class MemberEpcS1apSapMme<EpcMmeApplication>;

public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  /** Constructor */
  EpcMmeApplication ();

  /** Destructor */
  virtual ~EpcMmeApplication ();

  /**
   *
   * \return the MME side of the S1-AP SAP
   */
  EpcS1apSapMme* GetS1apSapMme ();


  /**
   * Add a new SGW to the MME
   *
   * \param sgwS11Addr IPv4 address of the SGW S11 interface
   * \param mmeS11Addr IPv4 address of the MME S11 interface
   * \param mmeS11Socket socket of the MME S11 interface
   */
  void AddSgw (Ipv4Address sgwS11Addr, Ipv4Address mmeS11Addr, Ptr<Socket> mmeS11Socket);

  /**
   * Add a new eNB to the MME
   *
   * \param ecgi E-UTRAN Cell Global ID, the unique identifier of the eNodeB
   * \param enbS1UAddr IPv4 address of the eNB for S1-U communications
   * \param enbS1apSap the eNB side of the S1-AP SAP
   */
  void AddEnb (uint16_t ecgi, Ipv4Address enbS1UAddr, EpcS1apSapEnb* enbS1apSap);

  /**
   * Add a new UE to the MME. This is the equivalent of storing the UE
   * credentials before the UE is ever turned on.
   *
   * \param imsi the unique identifier of the UE
   */
  void AddUe (uint64_t imsi);

  /**
   * Add an EPS bearer to the list of bearers to be activated for this UE.
   * The bearer will be activated when the UE enters the ECM
   * connected state.
   *
   * \param imsi UE identifier
   * \param tft traffic flow template of the bearer
   * \param bearer QoS characteristics of the bearer
   * \returns bearer ID
   */
  uint8_t AddBearer (uint64_t imsi, Ptr<EpcTft> tft, EpsBearer bearer);


private:
  // S1-AP SAP MME forwarded methods

  /**
   * Process the S1 Initial UE Message received from an eNB
   * \param mmeUeS1Id the MME UE S1 ID
   * \param enbUeS1Id the ENB UE S1 ID
   * \param imsi the IMSI
   * \param ecgi the ECGI
   */
  void DoInitialUeMessage (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, uint64_t imsi, uint16_t ecgi);

  /**
   * Process the S1 Initial Context Setup Response received from an eNB
   * \param mmeUeS1Id the MME UE S1 ID
   * \param enbUeS1Id the ENB UE S1 ID
   * \param erabSetupList the ERAB setup list
   */
  void DoInitialContextSetupResponse (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<EpcS1apSapMme::ErabSetupItem> erabSetupList);

  /**
   * Process the S1 Path Switch Request received from an eNB
   * \param mmeUeS1Id the MME UE S1 ID
   * \param enbUeS1Id the ENB UE S1 ID
   * \param cgi the CGI
   * \param erabToBeSwitchedInDownlinkList the ERAB to be switched in downlink list
   */
  void DoPathSwitchRequest (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t cgi, std::list<EpcS1apSapMme::ErabSwitchedInDownlinkItem> erabToBeSwitchedInDownlinkList);

  /**
   * Process ERAB Release Indication received from an eNB
   * \param mmeUeS1Id the MME UE S1 ID
   * \param enbUeS1Id the ENB UE S1 ID
   * \param erabToBeReleaseIndication the ERAB to be release indication list
   */
  void DoErabReleaseIndication (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<EpcS1apSapMme::ErabToBeReleasedIndication> erabToBeReleaseIndication);


  // Methods to read/process GTP-C messages of the S11 interface

  /**
   * Reads the S11 messages from a socket
   * \param socket the socket
   */
  void RecvFromS11Socket (Ptr<Socket> socket);

  /**
   * Process GTP-C Create Session Response message
   * \param header the GTP-C header
   * \param packet the packet containing the message
   */
  void DoRecvCreateSessionResponse (GtpcHeader &header, Ptr<Packet> packet);

  /**
   * Process GTP-C Modify Bearer Response message
   * \param header the GTP-C header
   * \param packet the packet containing the message
   */
  void DoRecvModifyBearerResponse (GtpcHeader &header, Ptr<Packet> packet);

  /**
   * Process GTP-C Delete Bearer Request message
   * \param header the GTP-C header
   * \param packet the packet containing the message
   */
  void DoRecvDeleteBearerRequest (GtpcHeader &header, Ptr<Packet> packet);


  /**
   * Hold info on an EPS bearer to be activated
   */
  struct BearerInfo
  {
    Ptr<EpcTft> tft;  ///< traffic flow template
    EpsBearer bearer; ///< bearer QOS characteristics
    uint8_t bearerId; ///< bearer ID
  };

  /**
   * Hold info on a UE
   */
  struct UeInfo : public SimpleRefCount<UeInfo>
  {
    uint64_t imsi; ///< UE identifier
    uint64_t mmeUeS1Id; ///< mmeUeS1Id
    uint16_t enbUeS1Id; ///< enbUeS1Id
    uint16_t cellId; ///< cell ID
    uint16_t bearerCounter; ///< bearer counter
    std::list<BearerInfo> bearersToBeActivated; ///< list of bearers to be activated
  };

  /**
   * UeInfo stored by IMSI
   */
  std::map<uint64_t, Ptr<UeInfo> > m_ueInfoMap;

  /**
   * \brief This Function erases all contexts of bearer from MME side
   * \param ueInfo UE information pointer
   * \param epsBearerId Bearer Id which need to be removed corresponding to UE
   */
  void RemoveBearer (Ptr<UeInfo> ueInfo, uint8_t epsBearerId);

  /**
   * Hold info on an ENB
   */
  struct EnbInfo : public SimpleRefCount<EnbInfo>
  {
    uint16_t gci; ///< GCI
    Ipv4Address s1uAddr; ///< IP address of the S1-U interface
    EpcS1apSapEnb* s1apSapEnb; ///< EpcS1apSapEnb
  };

  /**
   * EnbInfo stored by EGCI
   */
  std::map<uint16_t, Ptr<EnbInfo> > m_enbInfoMap;


  EpcS1apSapMme* m_s1apSapMme; ///< EpcS1apSapMme

  Ptr<Socket> m_s11Socket; ///< Socket to send/receive messages in the S11 interface
  Ipv4Address m_mmeS11Addr; ///< IPv4 address of the MME S11 interface
  Ipv4Address m_sgwS11Addr; ///< IPv4 address of the SGW S11 interface
  uint16_t m_gtpcUdpPort; ///< UDP port for GTP-C protocol. Fixed by the standard to port 2123
};

} // namespace ns3

#endif // EPC_MME_APPLICATION_H
