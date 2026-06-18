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

#ifndef EPC_SGW_APPLICATION_H
#define EPC_SGW_APPLICATION_H

#include "ns3/application.h"
#include "ns3/address.h"
#include "ns3/socket.h"
#include "ns3/epc-gtpc-header.h"

namespace ns3 {

/**
 * \ingroup lte
 *
 * This application implements the Serving Gateway Entity (SGW)
 * according to the 3GPP TS 23.401 document.
 *
 * This Application implements the SGW side of the S5 interface between
 * the SGW node and the PGW node and the SGW side of the S11 interface between
 * the SGW node and the MME node hosts. It supports the following functions and messages:
 *
 *  - S5 connectivity (i.e. GTPv2-C signalling and GTP-U data plane)
 *  - Bearer management functions including dedicated bearer establishment
 *  - UL and DL bearer binding
 *  - Tunnel Management messages
 *
 * Others functions enumerated in section 4.4.3.2 of 3GPP TS 23.401 are not supported.
 */
class EpcSgwApplication : public Application
{
public:

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  /**
   * Constructor that binds callback methods of sockets.
   *
   * \param s1uSocket socket used to send/receive GTP-U packets to/from the eNBs
   * \param s5Addr IPv4 address of the S5 interface
   * \param s5uSocket socket used to send/receive GTP-U packets to/from the PGW
   * \param s5cSocket socket used to send/receive GTP-C packets to/from the PGW
   */
  EpcSgwApplication (const Ptr<Socket> s1uSocket, Ipv4Address s5Addr,
                     const Ptr<Socket> s5uSocket, const Ptr<Socket> s5cSocket);

  /** Destructor */
  virtual ~EpcSgwApplication (void);


  /**
   * Let the SGW be aware of an MME
   *
   * \param mmeS11Addr the address of the MME
   * \param s11Socket the socket to send/receive messages from the MME
   */
  void AddMme (Ipv4Address mmeS11Addr, Ptr<Socket> s11Socket);

  /**
   * Let the SGW be aware of a PGW
   *
   * \param pgwAddr the address of the PGW
   */
  void AddPgw (Ipv4Address pgwAddr);

  /**
   * Let the SGW be aware of a new eNB 
   *
   * \param cellId the cell identifier
   * \param enbAddr the address of the eNB
   * \param sgwAddr the address of the SGW
   */
  void AddEnb (uint16_t cellId, Ipv4Address enbAddr, Ipv4Address sgwAddr);


private:
  /**
   * Method to be assigned to the recv callback of the S11 socket.
   * It is called when the SGW receives a control packet from the MME.
   *
   * \param socket pointer to the S11 socket
   */
  void RecvFromS11Socket (Ptr<Socket> socket);

  /**
   * Method to be assigned to the recv callback of the S5-U socket.
   * It is called when the SGW receives a data packet from the PGW
   * that is to be forwarded to an eNB.
   *
   * \param socket pointer to the S5-U socket
   */
  void RecvFromS5uSocket (Ptr<Socket> socket);

  /**
   * Method to be assigned to the recv callback of the S5-C socket.
   * It is called when the SGW receives a control packet from the PGW.
   *
   * \param socket pointer to the S5-C socket
   */
  void RecvFromS5cSocket (Ptr<Socket> socket);

  /**
   * Method to be assigned to the recv callback of the S1-U socket.
   * It is called when the SGW receives a data packet from the eNB
   * that is to be forwarded to the PGW.
   *
   * \param socket pointer to the S1-U socket
   */
  void RecvFromS1uSocket (Ptr<Socket> socket);

  /**
   * Send a data packet to the PGW via the S5 interface
   *
   * \param packet packet to be sent
   * \param pgwAddr the address of the PGW
   * \param teid the Tunnel Enpoint Identifier
   */
  void SendToS5uSocket (Ptr<Packet> packet, Ipv4Address pgwAddr, uint32_t teid);

  /**
   * Send a data packet to an eNB via the S1-U interface
   *
   * \param packet packet to be sent
   * \param enbS1uAddress the address of the eNB
   * \param teid the Tunnel Enpoint IDentifier
   */
  void SendToS1uSocket (Ptr<Packet> packet, Ipv4Address enbS1uAddress, uint32_t teid);


  // Process messages received from the MME

  /**
   * Process GTP-C Create Session Request message
   * \param packet the packet containing the message
   */
  void DoRecvCreateSessionRequest (Ptr<Packet> packet);

  /**
   * Process GTP-C Modify Bearer Request message
   * \param packet the packet containing the message
   */
  void DoRecvModifyBearerRequest (Ptr<Packet> packet);

  /**
   * Process GTP-C Delete Bearer Command message
   * \param packet the packet containing the message
   */
  void DoRecvDeleteBearerCommand (Ptr<Packet> packet);

  /**
   * Process GTP-C Delete Bearer Response message
   * \param packet the packet containing the message
   */
  void DoRecvDeleteBearerResponse (Ptr<Packet> packet);


  // Process messages received from the PGW

  /**
   * Process GTP-C Create Session Response message
   * \param packet the packet containing the message
   */
  void DoRecvCreateSessionResponse (Ptr<Packet> packet);

  /**
   * Process GTP-C Modify Bearer Response message
   * \param packet the packet containing the message
   */
  void DoRecvModifyBearerResponse (Ptr<Packet> packet);

  /**
   * Process GTP-C Delete Bearer Request message
   * \param packet the packet containing the message
   */
  void DoRecvDeleteBearerRequest (Ptr<Packet> packet);


  /**
   * SGW address in the S5 interface
   */
  Ipv4Address m_s5Addr;

  /**
   * MME address in the S11 interface
   */
  Ipv4Address m_mmeS11Addr;

  /**
    * UDP socket to send/receive control messages to/from the S11 interface
    */
  Ptr<Socket> m_s11Socket;

  /**
   * PGW address in the S5 interface
   */
  Ipv4Address m_pgwAddr;

 /**
  * UDP socket to send/receive GTP-U packets to/from the S5 interface
  */
  Ptr<Socket> m_s5uSocket;

 /**
  * UDP socket to send/receive GTP-C packets to/from the S5 interface
  */
  Ptr<Socket> m_s5cSocket;

 /**
  * UDP socket to send/receive GTP-U packets to/from the S1-U interface
  */
  Ptr<Socket> m_s1uSocket;

  /**
   * UDP port to be used for GTP-U
   */
  uint16_t m_gtpuUdpPort;

  /**
   * UDP port to be used for GTP-C
   */
  uint16_t m_gtpcUdpPort;

  /**
   * TEID count
   */
  uint32_t m_teidCount;


  /// EnbInfo structure
  struct EnbInfo
  {
    Ipv4Address enbAddr; ///< eNB address
    Ipv4Address sgwAddr; ///< SGW address
  };

  /**
   * Map for eNB info by cell ID
   */
  std::map<uint16_t, EnbInfo> m_enbInfoByCellId;

  /**
   * Map for eNB address by TEID
   */
  std::map<uint32_t, Ipv4Address> m_enbByTeidMap;

  /**
   * MME S11 FTEID by SGW S5C TEID
   */
  std::map<uint32_t, GtpcHeader::Fteid_t> m_mmeS11FteidBySgwS5cTeid;
};

} // namespace ns3

#endif // EPC_SGW_APPLICATION_H
