/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#ifndef SERVICE_FLOW_H
#define SERVICE_FLOW_H

#include <stdint.h>
#include "wimax-phy.h"
#include "wimax-connection.h"
#include "cs-parameters.h"

namespace ns3 {

class ServiceFlowRecord;
class WimaxConnection;

/**
 * \ingroup wimax
 * This class implements service flows as described by the IEEE-802.16 standard
 */
class ServiceFlow
{
public:
  /// Direction enumeration
  enum Direction
  {
    SF_DIRECTION_DOWN,
    SF_DIRECTION_UP
  };

  /// Type enumeration
  enum Type
  {
    SF_TYPE_PROVISIONED,
    SF_TYPE_ADMITTED,
    SF_TYPE_ACTIVE
  };

  /// section 11.13.11 Service flow scheduling type, page 701
  enum SchedulingType
  {
    SF_TYPE_NONE = 0,
    SF_TYPE_UNDEF = 1,
    SF_TYPE_BE = 2,
    SF_TYPE_NRTPS = 3,
    SF_TYPE_RTPS = 4,
    SF_TYPE_UGS = 6,
    SF_TYPE_ALL = 255
  };
  /// section 11.13.19.2 CS parameter encoding rules, page 707
  enum CsSpecification
  {
    ATM = 99,
    IPV4 = 100,
    IPV6 = 101,
    ETHERNET = 102,
    VLAN = 103,
    IPV4_OVER_ETHERNET = 104,
    IPV6_OVER_ETHERNET = 105,
    IPV4_OVER_VLAN = 106,
    IPV6_OVER_VLAN = 107
  };
  /// Modulation type enumeration, Table 356 and 362
  enum ModulationType
  {
    MODULATION_TYPE_BPSK_12,
    MODULATION_TYPE_QPSK_12,
    MODULATION_TYPE_QPSK_34,
    MODULATION_TYPE_QAM16_12,
    MODULATION_TYPE_QAM16_34,
    MODULATION_TYPE_QAM64_23,
    MODULATION_TYPE_QAM64_34
  };

  /**
   * \brief creates a TLV from this service flow
   * \return the created tlv
   */
  Tlv ToTlv (void) const;
  /**
   * \brief creates a service flow from a TLV
   * \param tlv the tlv from which the service flow will be created
   */
  ServiceFlow (Tlv tlv);
  /**
   * \return true if the passed parameters match the classifier of the service flow, false otherwise
   * \param srcAddress the source ip address
   * \param dstAddress the destination ip address
   * \param srcPort the source port
   * \param dstPort the destination port
   * \param proto the layer 4 protocol
   */
  bool CheckClassifierMatch (Ipv4Address srcAddress,
                             Ipv4Address dstAddress,
                             uint16_t srcPort,
                             uint16_t dstPort,
                             uint8_t proto) const;
  /// default constructor
  ServiceFlow ();
  /**
   * Constructor
   *
   * \param direction the direction
   */
  ServiceFlow (enum Direction direction);
  /**
   * Constructor
   *
   * \param sf service flow
   */
  ServiceFlow (const ServiceFlow & sf);
  /**
   * Constructor
   *
   * \param sfid the SFID
   * \param direction the direction
   * \param connection the connection object
   */
  ServiceFlow (uint32_t sfid,
               enum Direction direction,
               Ptr<WimaxConnection> connection);
  ~ServiceFlow (void);
  /**
   * assignment operator
   * \param o the service flow to assign
   * \returns the service flow
   */
  ServiceFlow & operator = (ServiceFlow const& o);

  /// Initial values
  void InitValues (void);
  /**
   * Set direction
   * \param direction the direction value
   */
  void SetDirection (enum Direction direction);
  /**
   * Get direction
   * \returns the direction
   */
  enum Direction GetDirection (void) const;
  /**
   * Copy parameters from another service flow
   * \param sf the service flow
   */
  void CopyParametersFrom (ServiceFlow sf);

  /**
   * Set type of service flow
   * \param type the type value
   */
  void SetType (enum Type type);
  /**
   * Get type of service flow
   * \returns the type
   */
  enum Type GetType (void) const;
  /**
   * Set connection
   * \param connection the connection
   */
  void SetConnection (Ptr<WimaxConnection> connection);
  /**
   * Can return a null connection is this service flow has not
   * been associated yet to a connection.
   * \returns pointer to the WimaxConnection
   */
  Ptr<WimaxConnection> GetConnection (void) const;

  /**
   * Set is enabled flag
   * \param isEnabled is enabled flag
   */
  void SetIsEnabled (bool isEnabled);
  /**
   * Get is enabled flag
   * \returns is enabled
   */
  bool GetIsEnabled (void) const;

  /**
   * Set service flow record
   * \param record pointer to the service flow record
   */
  void SetRecord (ServiceFlowRecord *record);
  /**
   * Get service flow record
   * \returns pointer to the service flow record
   */
  ServiceFlowRecord* GetRecord (void) const;

  // wrapper functions
  /**
   * Get pointer to queue
   * \returns pointer to the wimax mac queue
   */
  Ptr<WimaxMacQueue> GetQueue (void) const;
  /**
   * Get scheduling type
   * \returns the scheduling type
   */
  enum ServiceFlow::SchedulingType GetSchedulingType (void) const;
  /**
   * Check if packets are present
   * \returns true if there are packets
   */
  bool HasPackets (void) const;
  /**
   * Check if packets of particular type are present
   * \param packetType the packet type to select
   * \returns true if there are packets of the packet type
   */
  bool HasPackets (MacHeaderType::HeaderType packetType) const;

  /// shall be called only by BS
  void CleanUpQueue (void);

  /// Print QOS parameters
  void PrintQoSParameters (void) const;

  /**
   * Get scheduling type string
   * \returns the name of the scheduling type
   */
  char* GetSchedulingTypeStr (void) const;

  /**
   * Get SFID
   * \returns the SFID
   */
  uint32_t GetSfid (void) const;
  /**
   * Get CID
   * \returns the CID
   */
  uint16_t GetCid (void) const;
  /**
   * Get service class name
   * \returns the service class name
   */
  std::string GetServiceClassName () const;
  /**
   * Get QOS parameter set type
   * \returns the QOS parameter set type
   */
  uint8_t GetQosParamSetType (void) const;
  /**
   * Get traffic priority
   * \returns the traffic priority
   */
  uint8_t GetTrafficPriority (void) const;
  /**
   * Get max sustained traffic rate
   * \returns the maximum sustained traffic rate
   */
  uint32_t GetMaxSustainedTrafficRate (void) const;
  /**
   * Get max traffic burst
   * \returns the maximum traffic burst
   */
  uint32_t GetMaxTrafficBurst (void) const;
  /**
   * Get minimum reserved traffic rate
   * \returns the minimum reserved traffic rate
   */
  uint32_t GetMinReservedTrafficRate (void) const;
  /**
   * Get minimum tolerable traffic rate
   * \returns the minimum tolerable traffic rate
   */
  uint32_t GetMinTolerableTrafficRate (void) const;
  /**
   * Get service scheduling type
   * \returns the scheduling type
   */
  enum ServiceFlow::SchedulingType GetServiceSchedulingType (void) const;
  /**
   * Get request transmission policy
   * \returns the request transmission policy
   */
  uint32_t GetRequestTransmissionPolicy (void) const;
  /**
   * Get tolerated jitter
   * \returns the tolerated jitter
   */
  uint32_t GetToleratedJitter (void) const;
  /**
   * Get maximum latency
   * \returns the maximum latency
   */
  uint32_t GetMaximumLatency (void) const;
  /**
   * Get fixed versus varaiable SDU indicator
   * \returns the fixed vs variable SDU indicator
   */
  uint8_t GetFixedversusVariableSduIndicator (void) const;
  /**
   * Get SDU size
   * \returns the SDU size
   */
  uint8_t GetSduSize (void) const;
  /**
   * Get target SAID
   * \returns the target SAID
   */
  uint16_t GetTargetSAID (void) const;
  /**
   * Get ARQ enable
   * \returns the ARQ enable
   */
  uint8_t GetArqEnable (void) const;
  /**
   * Get ARQ retry timeout transmit
   * \returns the ARQ retry timeout
   */
  uint16_t GetArqWindowSize (void) const;
  /**
   * Get ARQ retry timeout transmit
   * \returns the ARQ retry timeout transmit
   */
  uint16_t GetArqRetryTimeoutTx (void) const;
  /**
   * Get ARQ retry timeout receive
   * \returns the ARQ retry timeout receive
   */
  uint16_t GetArqRetryTimeoutRx (void) const;
  /**
   * Get ARQ block lifetime
   * \returns the ARQ block lifetime
   */
  uint16_t GetArqBlockLifeTime (void) const;
  /**
   * Get ARQ sync loss
   * \returns the ARQ sync loss value
   */
  uint16_t GetArqSyncLoss (void) const;
  /**
   * Get ARQ deliver in order
   * \returns the ARQ deliver in order
   */
  uint8_t GetArqDeliverInOrder (void) const;
  /**
   * Get ARQ purge timeout
   * \returns the ARQ purge timeout value
   */
  uint16_t GetArqPurgeTimeout (void) const;
  /**
   * Get ARQ block size
   * \returns the ARQ block size
   */
  uint16_t GetArqBlockSize (void) const;
  /**
   * Get CS specification
   * \returns the CS specification
   */
  enum CsSpecification GetCsSpecification (void) const;
  /**
   * Get convergence sublayer
   * \returns the convergence sublayer
   */
  CsParameters GetConvergenceSublayerParam (void) const;
  /**
   * Get unsolicited grant interval
   * \returns the unsolicited grant interval
   */
  uint16_t GetUnsolicitedGrantInterval (void) const;
  /**
   * Get unsolicited polling interval
   * \returns the unsolicited polling interval
   */
  uint16_t GetUnsolicitedPollingInterval (void) const;
  /**
   * Get is multicast
   * \returns the is multicast flag
   */
  bool GetIsMulticast (void) const;
  /**
   * Get modulation
   * \returns the modulation
   */
  enum WimaxPhy::ModulationType GetModulation (void) const;

  /**
   * Set SFID
   * \param sfid the SFID
   */
  void SetSfid (uint32_t sfid);
  /** Set service class name
   * \param name the service class name
   */
  void SetServiceClassName (std::string name);
  /**
   * Set QOS parameter set type
   * \param type the QOS paraneter set type
   */
  void SetQosParamSetType (uint8_t type);
  /**
   * Set traffic priority
   * \param priority the traffic priority
   */
  void SetTrafficPriority (uint8_t priority);
  /**
   * Set max sustained traffic rate
   * \param maxSustainedRate the maximum sustained traffic rate
   */
  void SetMaxSustainedTrafficRate (uint32_t maxSustainedRate);
  /**
   * Set maximum traffic burst
   * \param maxTrafficBurst the maximum traffic burst
   */
  void SetMaxTrafficBurst (uint32_t maxTrafficBurst);
  /**
   * Set minimum reserved traffic rate
   * \param minResvRate the minimum reserved traffic rate
   */
  void SetMinReservedTrafficRate (uint32_t minResvRate);
  /**
   * Set minimum tolerable traffic rate
   * \param minJitter the minimum tolerable traffic rate
   */
  void SetMinTolerableTrafficRate (uint32_t minJitter);
  /**
   * Set service scheduling type
   * \param schedType the service scheduling type
   */
  void SetServiceSchedulingType (enum ServiceFlow::SchedulingType schedType);
  /**
   * Set request transmission policy
   * \param policy the request transmission policy
   */
  void SetRequestTransmissionPolicy (uint32_t policy);
  /**
   * Set tolerated jitter
   * \param jitter the tolerated jitter
   */
  void SetToleratedJitter (uint32_t jitter);
  /**
   * Set maximum latency
   * \param MaximumLatency the maximjum latency
   */
  void SetMaximumLatency (uint32_t MaximumLatency);
  /**
   * Set fixed versus variable SDU indicator
   * \param sduIndicator fixed vs variable SDU indicator
   */
  void SetFixedversusVariableSduIndicator (uint8_t sduIndicator);
  /**
   * Set SDU size
   * \param sduSize the SDU size
   */
  void SetSduSize (uint8_t sduSize);
  /**
   * Set target SAID
   * \param targetSaid the target SAID value
   */
  void SetTargetSAID (uint16_t targetSaid);
  /**
   * Set ARQ enable
   * \param arqEnable the ARQ enable setting
   */
  void SetArqEnable (uint8_t arqEnable);
  /**
   * Set ARQ retry timeout transmit
   * \param arqWindowSize the ARQ retry timeout transmit
   */
  void SetArqWindowSize (uint16_t arqWindowSize);
  /**
   * Set ARQ retry timeout transmit
   * \param timeout the ARQ retry timeout transmit
   */
  void SetArqRetryTimeoutTx (uint16_t timeout);
  /**
   * Set ARQ retry timeout receive
   * \param timeout the timeout
   */
  void SetArqRetryTimeoutRx (uint16_t timeout);
  /**
   * Set ARQ block lifetime
   * \param lifeTime the ARQ block life time
   */
  void SetArqBlockLifeTime (uint16_t lifeTime);
  /**
   * Set ARQ sync loss
   * \param syncLoss the ARQ sync loss
   */
  void SetArqSyncLoss (uint16_t syncLoss);
  /**
   * Set ARQ deliver in order
   * \param inOrder the deliver in order setting
   */
  void SetArqDeliverInOrder (uint8_t inOrder);
  /**
   * Set ARQ purge timeout
   * \param timeout the timeout value
   */
  void SetArqPurgeTimeout (uint16_t timeout);
  /**
   * Set ARQ block size
   * \param size the size
   */
  void SetArqBlockSize (uint16_t size);
  /**
   * Set CS specification
   * \param spec the CS specification
   */
  void SetCsSpecification (enum CsSpecification spec);
  /**
   * Set convergence sublayer parameters
   * \param csparam the convergence sublayer parameters
   */
  void SetConvergenceSublayerParam (CsParameters csparam);

  /**
   * Set unsolicied grant interval
   * \param unsolicitedGrantInterval the unsolicited grant interval
   */
  void SetUnsolicitedGrantInterval (uint16_t unsolicitedGrantInterval);
  /**
   * Set unsolicited polling interval
   * \param unsolicitedPollingInterval the unsolicited polling interval
   */
  void SetUnsolicitedPollingInterval (uint16_t unsolicitedPollingInterval);
  /**
   * Set is multicast
   * \param isMulticast the is multicast flag
   */
  void SetIsMulticast (bool isMulticast);
  /**
   * Set modulation
   * \param modulationType the modulation type
   */
  void SetModulation (enum WimaxPhy::ModulationType modulationType);



private:
  uint32_t m_sfid; ///< SFID
  std::string m_serviceClassName; ///< service class name
  uint8_t m_qosParamSetType; ///< QOS parameter type
  uint8_t m_trafficPriority; ///< traffic priority
  uint32_t m_maxSustainedTrafficRate; ///< maximum sustained traffic rate
  uint32_t m_maxTrafficBurst; ///< maximum traffic burst
  uint32_t m_minReservedTrafficRate; ///< minimum reserved traffic rate
  uint32_t m_minTolerableTrafficRate; ///< minimum tolerable traffic rate
  enum ServiceFlow::SchedulingType m_schedulingType; ///< scheduling type
  uint32_t m_requestTransmissionPolicy; ///< request transmission policy
  uint32_t m_toleratedJitter; ///< tolerated jitter
  uint32_t m_maximumLatency; ///< maximum latency
  uint8_t m_fixedversusVariableSduIndicator; ///< fixed versus variable SDI indicator
  uint8_t m_sduSize; ///< SDU size
  uint16_t m_targetSAID; ///< traget SAID
  uint8_t m_arqEnable; ///< ARQ enable
  uint16_t m_arqWindowSize; ///< ARQ window size
  uint16_t m_arqRetryTimeoutTx; ///< ARQ retry timeout transmit
  uint16_t m_arqRetryTimeoutRx; ///< ARQ retry timeout receive
  uint16_t m_arqBlockLifeTime; ///< ARQ block life time
  uint16_t m_arqSyncLoss; ///< ARQ sync loss
  uint8_t m_arqDeliverInOrder; ///< ARQ deliver in order 
  uint16_t m_arqPurgeTimeout; ///< ARQ purge timeout
  uint16_t m_arqBlockSize; ///< ARQ block size
  enum CsSpecification m_csSpecification; ///< CS specification
  CsParameters m_convergenceSublayerParam; ///< convergence sublayer parameters
  uint16_t m_unsolicitedGrantInterval; ///< unsolicited grant interval
  uint16_t m_unsolicitedPollingInterval; ///< unsolicited polling interval
  Direction m_direction; ///< direction
  Type m_type; ///< type
  Ptr<WimaxConnection> m_connection; ///< connection
  bool m_isEnabled; ///< is enabled?
  bool m_isMulticast; ///< is multicast?
  enum WimaxPhy::ModulationType m_modulationType; ///< modulation type
  // will be used by the BS
  ServiceFlowRecord *m_record; ///< service flow record
};

} // namespace ns3

#endif /* SERVICE_FLOW_H */
