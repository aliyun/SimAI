/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo <nbaldo@cttc.es>,
 *         Marco Miozzo <mmiozzo@cttc.es>
 */

#ifndef LTE_ENB_CPHY_SAP_H
#define LTE_ENB_CPHY_SAP_H

#include <stdint.h>
#include <ns3/ptr.h>

#include <ns3/lte-rrc-sap.h>

namespace ns3 {

class LteEnbNetDevice;

/**
 * Service Access Point (SAP) offered by the UE PHY to the UE RRC for control purposes
 *
 * This is the PHY SAP Provider, i.e., the part of the SAP that contains
 * the PHY methods called by the MAC
 */
class LteEnbCphySapProvider
{
public:

  /** 
   * destructor
   */
  virtual ~LteEnbCphySapProvider ();

  /** 
   * 
   * 
   * \param cellId the Cell Identifier
   */
  virtual void SetCellId (uint16_t cellId) = 0;

  /**
   * \param ulBandwidth the UL bandwidth in PRBs
   * \param dlBandwidth the DL bandwidth in PRBs
   */
  virtual void SetBandwidth (uint16_t ulBandwidth, uint16_t dlBandwidth) = 0;

  /**
   * \param ulEarfcn the UL EARFCN
   * \param dlEarfcn the DL EARFCN
   */
  virtual void SetEarfcn (uint32_t ulEarfcn, uint32_t dlEarfcn) = 0;
  
  /** 
   * Add a new UE to the cell
   * 
   * \param rnti the UE id relative to this cell
   */
  virtual void AddUe (uint16_t rnti) = 0;

  /** 
   * Remove an UE from the cell
   * 
   * \param rnti the UE id relative to this cell
   */
  virtual void RemoveUe (uint16_t rnti) = 0;
  
  /**
   * Set the UE transmission power offset P_A
   *
   * \param rnti the UE id relative to this cell
   * \param pa transmission power offset
   */
  virtual void SetPa (uint16_t rnti, double pa) = 0;

  /**
   * \param rnti the RNTI of the user
   * \param txMode the transmissionMode of the user
   */
  virtual void SetTransmissionMode (uint16_t rnti, uint8_t txMode) = 0;

  /**
   * \param rnti the RNTI of the user
   * \param srsCi the SRS Configuration Index of the user
   */
  virtual void SetSrsConfigurationIndex (uint16_t rnti, uint16_t srsCi) = 0;

  /** 
   * 
   * \param mib the Master Information Block to be sent on the BCH
   */
  virtual void SetMasterInformationBlock (LteRrcSap::MasterInformationBlock mib) = 0;

  /**
   *
   * \param sib1 the System Information Block Type 1 to be sent on the BCH
   */
  virtual void SetSystemInformationBlockType1 (LteRrcSap::SystemInformationBlockType1 sib1) = 0;

  /**
   *
   * \return Reference Signal Power for SIB2
   */
  virtual int8_t GetReferenceSignalPower () = 0;
};


/**
 * Service Access Point (SAP) offered by the UE PHY to the UE RRC for control purposes
 *
 * This is the CPHY SAP User, i.e., the part of the SAP that contains the RRC
 * methods called by the PHY
*/
class LteEnbCphySapUser
{
public:
  
  /** 
   * destructor
   */
  virtual ~LteEnbCphySapUser ();

};


/**
 * Template for the implementation of the LteEnbCphySapProvider as a member
 * of an owner class of type C to which all methods are forwarded
 * 
 */
template <class C>
class MemberLteEnbCphySapProvider : public LteEnbCphySapProvider
{
public:
  /**
   * Constructor
   *
   * \param owner the owner class
   */
  MemberLteEnbCphySapProvider (C* owner);

  // inherited from LteEnbCphySapProvider
  virtual void SetCellId (uint16_t cellId);
  virtual void SetBandwidth (uint16_t ulBandwidth, uint16_t dlBandwidth);
  virtual void SetEarfcn (uint32_t ulEarfcn, uint32_t dlEarfcn);
  virtual void AddUe (uint16_t rnti);
  virtual void RemoveUe (uint16_t rnti);
  virtual void SetPa (uint16_t rnti, double pa);
  virtual void SetTransmissionMode (uint16_t  rnti, uint8_t txMode);
  virtual void SetSrsConfigurationIndex (uint16_t  rnti, uint16_t srsCi);
  virtual void SetMasterInformationBlock (LteRrcSap::MasterInformationBlock mib);
  virtual void SetSystemInformationBlockType1 (LteRrcSap::SystemInformationBlockType1 sib1);
  virtual int8_t GetReferenceSignalPower ();
  
private:
  MemberLteEnbCphySapProvider ();
  C* m_owner; ///< the owner class
};

template <class C>
MemberLteEnbCphySapProvider<C>::MemberLteEnbCphySapProvider (C* owner)
  : m_owner (owner)
{
}

template <class C>
MemberLteEnbCphySapProvider<C>::MemberLteEnbCphySapProvider ()
{
}

template <class C>
void 
MemberLteEnbCphySapProvider<C>::SetCellId (uint16_t cellId)
{
  m_owner->DoSetCellId (cellId);
}


template <class C>
void 
MemberLteEnbCphySapProvider<C>::SetBandwidth (uint16_t ulBandwidth, uint16_t dlBandwidth)
{
  m_owner->DoSetBandwidth (ulBandwidth, dlBandwidth);
}

template <class C>
void 
MemberLteEnbCphySapProvider<C>::SetEarfcn (uint32_t ulEarfcn, uint32_t dlEarfcn)
{
  m_owner->DoSetEarfcn (ulEarfcn, dlEarfcn);
}

template <class C>
void 
MemberLteEnbCphySapProvider<C>::AddUe (uint16_t rnti)
{
  m_owner->DoAddUe (rnti);
}

template <class C>
void 
MemberLteEnbCphySapProvider<C>::RemoveUe (uint16_t rnti)
{
  m_owner->DoRemoveUe (rnti);
}

template <class C>
void 
MemberLteEnbCphySapProvider<C>::SetPa (uint16_t rnti, double pa)
{
  m_owner->DoSetPa (rnti, pa);
}

template <class C>
void
MemberLteEnbCphySapProvider<C>::SetTransmissionMode (uint16_t  rnti, uint8_t txMode)
{
  m_owner->DoSetTransmissionMode (rnti, txMode);
}

template <class C>
void 
MemberLteEnbCphySapProvider<C>::SetSrsConfigurationIndex (uint16_t  rnti, uint16_t srsCi)
{
  m_owner->DoSetSrsConfigurationIndex (rnti, srsCi);
}

template <class C> 
void 
MemberLteEnbCphySapProvider<C>::SetMasterInformationBlock (LteRrcSap::MasterInformationBlock mib)
{
  m_owner->DoSetMasterInformationBlock (mib);
}

template <class C>
void
MemberLteEnbCphySapProvider<C>::SetSystemInformationBlockType1 (LteRrcSap::SystemInformationBlockType1 sib1)
{
  m_owner->DoSetSystemInformationBlockType1 (sib1);
}

template <class C>
int8_t
MemberLteEnbCphySapProvider<C>::GetReferenceSignalPower ()
{
  return m_owner->DoGetReferenceSignalPower ();
}

/**
 * Template for the implementation of the LteEnbCphySapUser as a member
 * of an owner class of type C to which all methods are forwarded
 * 
 */
template <class C>
class MemberLteEnbCphySapUser : public LteEnbCphySapUser
{
public:
  /**
   * Constructor
   *
   * \param owner the owner class
   */
  MemberLteEnbCphySapUser (C* owner);

  // methods inherited from LteEnbCphySapUser go here

private:
  MemberLteEnbCphySapUser ();
  C* m_owner; ///< the owner class
};

template <class C>
MemberLteEnbCphySapUser<C>::MemberLteEnbCphySapUser (C* owner)
  : m_owner (owner)
{
}

template <class C>
MemberLteEnbCphySapUser<C>::MemberLteEnbCphySapUser ()
{
}






} // namespace ns3


#endif // LTE_ENB_CPHY_SAP_H
