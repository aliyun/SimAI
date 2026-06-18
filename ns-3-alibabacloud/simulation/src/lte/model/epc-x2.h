/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

#ifndef EPC_X2_H
#define EPC_X2_H

#include "ns3/socket.h"
#include "ns3/callback.h"
#include "ns3/ptr.h"
#include "ns3/object.h"

#include "ns3/epc-x2-sap.h"

#include <map>

namespace ns3 {


/**
 * X2IfaceInfo
 */
class X2IfaceInfo : public SimpleRefCount<X2IfaceInfo>
{
public:
  /**
   * Constructor
   *
   * \param remoteIpAddr remote IP address
   * \param localCtrlPlaneSocket control plane socket
   * \param localUserPlaneSocket user plane socket
   */
  X2IfaceInfo (Ipv4Address remoteIpAddr, Ptr<Socket> localCtrlPlaneSocket, Ptr<Socket> localUserPlaneSocket);
  virtual ~X2IfaceInfo (void);

  /**
   * Assignment operator
   * \param value value to assign
   * \returns X2IfaceInfo& 
   */
  X2IfaceInfo& operator= (const X2IfaceInfo &value);

public:
  Ipv4Address   m_remoteIpAddr; ///< remote IP address
  Ptr<Socket>   m_localCtrlPlaneSocket; ///< local control plane socket
  Ptr<Socket>   m_localUserPlaneSocket; ///< local user plane socket
};


/**
 * X2CellInfo
 */
class X2CellInfo : public SimpleRefCount<X2CellInfo>
{
public:
  /**
   * Constructor
   *
   * \param localCellIds local cell IDs
   * \param remoteCellIds remote cell IDs
   */
  X2CellInfo (std::vector<uint16_t> localCellIds, std::vector<uint16_t> remoteCellIds);
  virtual ~X2CellInfo (void);

  /**
   * Assignment operator
   * \param value value to assign
   * \returns X2CellInfo&  
   */
  X2CellInfo& operator= (const X2CellInfo &value);

public:
  std::vector<uint16_t> m_localCellIds; ///< local cell IDs
  std::vector<uint16_t> m_remoteCellIds; ///< remote cell IDs
};


/**
 * \ingroup lte
 *
 * This entity is installed inside an eNB and provides the functionality for the X2 interface
 */
class EpcX2 : public Object
{
  /// allow EpcX2SpecificEpcX2SapProvider<EpcX2> class friend access
  friend class EpcX2SpecificEpcX2SapProvider<EpcX2>;

public:
  /** 
   * Constructor
   */
  EpcX2 ();

  /**
   * Destructor
   */
  virtual ~EpcX2 (void);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual void DoDispose (void);


  /**
   * \param s the X2 SAP User to be used by this EPC X2 entity
   */
  void SetEpcX2SapUser (EpcX2SapUser * s);

  /**
   * \return the X2 SAP Provider interface offered by this EPC X2 entity
   */
  EpcX2SapProvider* GetEpcX2SapProvider ();


  /**
   * Add an X2 interface to this EPC X2 entity
   * \param enb1CellId the cell ID of the current eNodeB
   * \param enb1X2Address the address of the current eNodeB
   * \param enb2CellIds the cell IDs of the neighbouring eNodeB
   * \param enb2X2Address the address of the neighbouring eNodeB
   */
  void AddX2Interface (uint16_t enb1CellId, Ipv4Address enb1X2Address,
                       std::vector<uint16_t> enb2CellIds, Ipv4Address enb2X2Address);


  /** 
   * Method to be assigned to the recv callback of the X2-C (X2 Control Plane) socket.
   * It is called when the eNB receives a packet from the peer eNB of the X2-C interface
   * 
   * \param socket socket of the X2-C interface
   */
  void RecvFromX2cSocket (Ptr<Socket> socket);

  /** 
   * Method to be assigned to the recv callback of the X2-U (X2 User Plane) socket.
   * It is called when the eNB receives a packet from the peer eNB of the X2-U interface
   * 
   * \param socket socket of the X2-U interface
   */
  void RecvFromX2uSocket (Ptr<Socket> socket);


protected:
  // Interface provided by EpcX2SapProvider
  /**
   * Send handover request function
   * \param params the send handover request parameters
   */
  virtual void DoSendHandoverRequest (EpcX2SapProvider::HandoverRequestParams params);
  /**
   * Send handover request ack function
   * \param params the send handover request ack parameters
   */
  virtual void DoSendHandoverRequestAck (EpcX2SapProvider::HandoverRequestAckParams params);
  /**
   * Send handover preparation failure function
   * \param params the handover preparation failure parameters
   */
  virtual void DoSendHandoverPreparationFailure (EpcX2SapProvider::HandoverPreparationFailureParams params);
  /**
   * Send SN status transfer function
   * \param params the SN status transfer parameters 
   */
  virtual void DoSendSnStatusTransfer (EpcX2SapProvider::SnStatusTransferParams params);
  /**
   * Send UE context release function
   * \param params the UE context release parameters
   */
  virtual void DoSendUeContextRelease (EpcX2SapProvider::UeContextReleaseParams params);
  /**
   * Send load information function
   * \param params the send load information parameters
   */
  virtual void DoSendLoadInformation (EpcX2SapProvider::LoadInformationParams params);
  /**
   * Send resource status update function
   * \param params the send resource status update parameters
   */
  virtual void DoSendResourceStatusUpdate (EpcX2SapProvider::ResourceStatusUpdateParams params);
  /**
   * Send UE data function
   *
   * \param params EpcX2SapProvider::UeDataParams
   */
  virtual void DoSendUeData (EpcX2SapProvider::UeDataParams params);

  EpcX2SapUser* m_x2SapUser; ///< X2 SAP user
  EpcX2SapProvider* m_x2SapProvider; ///< X2 SAP provider


private:

  /**
   * Map the targetCellId to the corresponding (sourceSocket, remoteIpAddr) to be used
   * to send the X2 message
   */
  std::map < uint16_t, Ptr<X2IfaceInfo> > m_x2InterfaceSockets;

  /**
   * Map the localSocket (the one receiving the X2 message) 
   * to the corresponding (sourceCellId, targetCellId) associated with the X2 interface
   */
  std::map < Ptr<Socket>, Ptr<X2CellInfo> > m_x2InterfaceCellIds;

  /**
   * UDP ports to be used for the X2-C interface
   */
  uint16_t m_x2cUdpPort;
  /**
   * UDP ports to be used for the X2-U interface
   */
  uint16_t m_x2uUdpPort;

};

} //namespace ns3

#endif // EPC_X2_H
