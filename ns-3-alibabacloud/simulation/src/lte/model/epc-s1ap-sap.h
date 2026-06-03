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
 * Author: Nicola Baldo <nbaldo@cttc.cat>
 */

#ifndef EPC_S1AP_SAP_H
#define EPC_S1AP_SAP_H

#include <ns3/address.h>
#include <ns3/ptr.h>
#include <ns3/object.h>
#include <ns3/eps-bearer.h>
#include <ns3/epc-tft.h>
#include <list>


namespace ns3 {

/**
 * \ingroup lte
 *
 * Base class that defines EPC S1-AP Service Access Point (SAP) interface.
 */
class EpcS1apSap
{
public:
  virtual ~EpcS1apSap ();

};


/**
 * \ingroup lte
 *
 * MME side of the S1-AP Service Access Point (SAP), provides the MME
 * methods to be called when an S1-AP message is received by the MME. 
 */
class EpcS1apSapMme : public EpcS1apSap
{
public:

  /** 
   * 
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param enbUeS1Id in practice, we use the RNTI
   * \param stmsi in practice, the imsi
   * \param ecgi in practice, the cell Id
   * 
   */
  virtual void InitialUeMessage (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, uint64_t stmsi, uint16_t ecgi) = 0;


  /**
   *  E-RAB Release Indication Item IEs, 3GPP TS 36.413 version 9.8.0 section 9.1.3.7
   *
   */
  struct ErabToBeReleasedIndication
  {
    uint8_t erabId; ///< E-RAB ID
  };

  /**
    * \brief As per 3GPP TS 36.413 version 9.8.0 section 8.2.3.2.2, the eNB
    * indicates bearer release by sending an E-RAB RELEASE INDICATION message
    * towards MME
    *
    * \param mmeUeS1Id in practice, we use the IMSI
    * \param enbUeS1Id in practice, we use the RNTI
    * \param erabToBeReleaseIndication List of bearers to be deactivated
    *
    */
  virtual void ErabReleaseIndication (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<ErabToBeReleasedIndication> erabToBeReleaseIndication ) = 0;

  /**
   *  E-RAB Setup Item IEs, see 3GPP TS 36.413 9.1.4.2 
   * 
   */
  struct ErabSetupItem
  {
    uint16_t    erabId; ///< E-RAB ID
    Ipv4Address enbTransportLayerAddress; ///< transport layer address
    uint32_t    enbTeid; ///< TEID
  };

  /** 
   * INITIAL CONTEXT SETUP RESPONSE message,  see 3GPP TS 36.413 9.1.4.2 
   * 
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param enbUeS1Id in practice, we use the RNTI
   * \param erabSetupList
   * 
   */
  virtual void InitialContextSetupResponse (uint64_t mmeUeS1Id,
                                            uint16_t enbUeS1Id,
                                            std::list<ErabSetupItem> erabSetupList) = 0;


  /**
   * E-RABs Switched in Downlink Item IE, see 3GPP TS 36.413 9.1.5.8
   * 
   */
  struct ErabSwitchedInDownlinkItem
  {
    uint16_t    erabId; ///< ERAB ID
    Ipv4Address enbTransportLayerAddress; ///< address 
    uint32_t    enbTeid; ///< TEID   
  };

  /**
   * PATH SWITCH REQUEST message, see 3GPP TS 36.413 9.1.5.8
   * 
   * \param enbUeS1Id in practice, we use the RNTI
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param gci
   * \param erabToBeSwitchedInDownlinkList
   */
  virtual void PathSwitchRequest (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t gci, std::list<ErabSwitchedInDownlinkItem> erabToBeSwitchedInDownlinkList) = 0;
};



/**
 * \ingroup lte
 *
 * eNB side of the S1-AP Service Access Point (SAP), provides the eNB
 * methods to be called when an S1-AP message is received by the eNB. 
 */
class EpcS1apSapEnb : public EpcS1apSap
{
public:


  /// ErabToBeSetupItem structure
  struct ErabToBeSetupItem
  {
    uint8_t    erabId; ///< ERAB iD
    EpsBearer   erabLevelQosParameters; ///< Level QOS parameters
    Ipv4Address transportLayerAddress; ///< transport layer address
    uint32_t    sgwTeid; ///< TEID
  };

  /** 
   * Initial context setup request
   * 
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param enbUeS1Id in practice, we use the RNTI
   * \param erabToBeSetupList
   * 
   */
  virtual void InitialContextSetupRequest (uint64_t mmeUeS1Id,
                                           uint16_t enbUeS1Id,
                                           std::list<ErabToBeSetupItem> erabToBeSetupList) = 0;


  /**
   * E-RABs Switched in Uplink Item IE, see 3GPP TS 36.413 9.1.5.9
   * 
   */
  struct ErabSwitchedInUplinkItem
  {
    uint8_t    erabId; ///< E_RAB ID
    Ipv4Address transportLayerAddress; ///< transport layer address
    uint32_t    enbTeid; ///< TEID
  };

  /**
   * PATH SWITCH REQUEST ACKNOWLEDGE message, see 3GPP TS 36.413 9.1.5.9
   * 
   * \param enbUeS1Id in practice, we use the RNTI
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param cgi
   * \param erabToBeSwitchedInUplinkList
   */
  virtual void PathSwitchRequestAcknowledge (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t cgi, std::list<ErabSwitchedInUplinkItem> erabToBeSwitchedInUplinkList) = 0;


};






/**
 * Template for the implementation of the EpcS1apSapMme as a member
 * of an owner class of type C to which all methods are forwarded
 * 
 */
template <class C>
class MemberEpcS1apSapMme : public EpcS1apSapMme
{
public:
/**
 * Constructor
 *
 * \param owner the owner class
 */
  MemberEpcS1apSapMme (C* owner);

  // inherited from EpcS1apSapMme
  /**
   * Initial UE Message function
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param enbUeS1Id in practice, we use the RNTI
   * \param imsi the IMSI
   * \param ecgi
   */
  virtual void InitialUeMessage (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, uint64_t imsi, uint16_t ecgi);
  /**
   * ERAB Release Indiation function
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param enbUeS1Id in practice, we use the RNTI
   * \param erabToBeReleaseIndication
   */
  virtual void ErabReleaseIndication (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<ErabToBeReleasedIndication> erabToBeReleaseIndication );

  /**
   * Initial context setup response
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param enbUeS1Id in practice, we use the RNTI
   * \param erabSetupList
   */
  virtual void InitialContextSetupResponse (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<ErabSetupItem> erabSetupList);
  /**
   * Path switch request
   * \param enbUeS1Id in practice, we use the RNTI
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param cgi
   * \param erabToBeSwitchedInDownlinkList
   */
  virtual void PathSwitchRequest (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t cgi, std::list<ErabSwitchedInDownlinkItem> erabToBeSwitchedInDownlinkList);

private:
  MemberEpcS1apSapMme ();
  C* m_owner; ///< owner class
};

template <class C>
MemberEpcS1apSapMme<C>::MemberEpcS1apSapMme (C* owner)
  : m_owner (owner)
{
}

template <class C>
MemberEpcS1apSapMme<C>::MemberEpcS1apSapMme ()
{
}

template <class C>
void MemberEpcS1apSapMme<C>::InitialUeMessage (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, uint64_t imsi, uint16_t ecgi)
{
  m_owner->DoInitialUeMessage (mmeUeS1Id, enbUeS1Id, imsi, ecgi);
}

template <class C>
void MemberEpcS1apSapMme<C>::ErabReleaseIndication (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<ErabToBeReleasedIndication> erabToBeReleaseIndication)
{
  m_owner->DoErabReleaseIndication (mmeUeS1Id, enbUeS1Id, erabToBeReleaseIndication);
}

template <class C>
void MemberEpcS1apSapMme<C>::InitialContextSetupResponse (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<ErabSetupItem> erabSetupList)
{
  m_owner->DoInitialContextSetupResponse (mmeUeS1Id, enbUeS1Id, erabSetupList);
}

template <class C>
void MemberEpcS1apSapMme<C>::PathSwitchRequest (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t cgi, std::list<ErabSwitchedInDownlinkItem> erabToBeSwitchedInDownlinkList)
{
  m_owner->DoPathSwitchRequest (enbUeS1Id, mmeUeS1Id, cgi, erabToBeSwitchedInDownlinkList);
}







/**
 * Template for the implementation of the EpcS1apSapEnb as a member
 * of an owner class of type C to which all methods are forwarded
 * 
 */
template <class C>
class MemberEpcS1apSapEnb : public EpcS1apSapEnb
{
public:
/**
 * Constructor
 *
 * \param owner the owner class
 */
  MemberEpcS1apSapEnb (C* owner);

  // inherited from EpcS1apSapEnb
  /**
   * Initial context setup request function
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param enbUeS1Id in practice, we use the RNTI
   * \param erabToBeSetupList
   */
  virtual void InitialContextSetupRequest (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<ErabToBeSetupItem> erabToBeSetupList);
  /**
   * Path switch request acknowledge function
   * \param enbUeS1Id in practice, we use the RNTI
   * \param mmeUeS1Id in practice, we use the IMSI
   * \param cgi
   * \param erabToBeSwitchedInUplinkList
   */
  virtual void PathSwitchRequestAcknowledge (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t cgi, std::list<ErabSwitchedInUplinkItem> erabToBeSwitchedInUplinkList);

private:
  MemberEpcS1apSapEnb ();
  C* m_owner; ///< owner class
};

template <class C>
MemberEpcS1apSapEnb<C>::MemberEpcS1apSapEnb (C* owner)
  : m_owner (owner)
{
}

template <class C>
MemberEpcS1apSapEnb<C>::MemberEpcS1apSapEnb ()
{
}

template <class C>
void MemberEpcS1apSapEnb<C>::InitialContextSetupRequest (uint64_t mmeUeS1Id, uint16_t enbUeS1Id, std::list<ErabToBeSetupItem> erabToBeSetupList)
{
  m_owner->DoInitialContextSetupRequest (mmeUeS1Id, enbUeS1Id, erabToBeSetupList);
}

template <class C>
void MemberEpcS1apSapEnb<C>::PathSwitchRequestAcknowledge (uint64_t enbUeS1Id, uint64_t mmeUeS1Id, uint16_t cgi, std::list<ErabSwitchedInUplinkItem> erabToBeSwitchedInUplinkList)
{
  m_owner->DoPathSwitchRequestAcknowledge (enbUeS1Id, mmeUeS1Id, cgi, erabToBeSwitchedInUplinkList);
}







} //namespace ns3

#endif /* EPC_S1AP_SAP_H */

