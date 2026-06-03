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

#ifndef LTE_PDCP_SAP_H
#define LTE_PDCP_SAP_H

#include "ns3/packet.h"

namespace ns3 {

/**
 * Service Access Point (SAP) offered by the PDCP entity to the RRC entity
 * See 3GPP 36.323 Packet Data Convergence Protocol (PDCP) specification
 *
 * This is the PDCP SAP Provider
 * (i.e. the part of the SAP that contains the PDCP methods called by the RRC)
 */
class LtePdcpSapProvider
{
public:
  virtual ~LtePdcpSapProvider ();

  /**
   * Parameters for LtePdcpSapProvider::TransmitPdcpSdu
   */
  struct TransmitPdcpSduParameters
  {
    Ptr<Packet> pdcpSdu;  /**< the RRC PDU */
    uint16_t    rnti; /**< the C-RNTI identifying the UE */
    uint8_t     lcid; /**< the logical channel id corresponding to the sending RLC instance */
  };

  /**
   * Send RRC PDU parameters to the PDCP for transmission
   *
   * This method is to be called when upper RRC entity has a
   * RRC PDU ready to send
   * 
   * \param params 
   */
  virtual void TransmitPdcpSdu (TransmitPdcpSduParameters params) = 0;
};


/**
 * Service Access Point (SAP) offered by the PDCP entity to the RRC entity
 * See 3GPP 36.323 Packet Data Convergence Protocol (PDCP) specification
 *
 * This is the PDCP SAP User
 * (i.e. the part of the SAP that contains the RRC methods called by the PDCP)
 */
class LtePdcpSapUser
{
public:
  virtual ~LtePdcpSapUser ();

  /**
   * Parameters for LtePdcpSapUser::ReceivePdcpSdu
   */
  struct ReceivePdcpSduParameters
  {
    Ptr<Packet> pdcpSdu;  /**< the RRC PDU */
    uint16_t    rnti; /**< the C-RNTI identifying the UE */
    uint8_t     lcid; /**< the logical channel id corresponding to the sending RLC instance */
  };

  /**
  * Called by the PDCP entity to notify the RRC entity of the reception of a new RRC PDU
  *
  * \param params
  */
  virtual void ReceivePdcpSdu (ReceivePdcpSduParameters params) = 0;
};


/// LtePdcpSpecificLtePdcpSapProvider class
template <class C>
class LtePdcpSpecificLtePdcpSapProvider : public LtePdcpSapProvider
{
public:
  /**
   * Constructor
   * 
   * \param pdcp PDCP
   */
  LtePdcpSpecificLtePdcpSapProvider (C* pdcp);

  // Interface implemented from LtePdcpSapProvider
  virtual void TransmitPdcpSdu (TransmitPdcpSduParameters params);

private:
  LtePdcpSpecificLtePdcpSapProvider ();
  C* m_pdcp; ///< the PDCP
};

template <class C>
LtePdcpSpecificLtePdcpSapProvider<C>::LtePdcpSpecificLtePdcpSapProvider (C* pdcp)
  : m_pdcp (pdcp)
{
}

template <class C>
LtePdcpSpecificLtePdcpSapProvider<C>::LtePdcpSpecificLtePdcpSapProvider ()
{
}

template <class C>
void LtePdcpSpecificLtePdcpSapProvider<C>::TransmitPdcpSdu (TransmitPdcpSduParameters params)
{
  m_pdcp->DoTransmitPdcpSdu (params);
}


/// LtePdcpSpecificLtePdcpSapUser class
template <class C>
class LtePdcpSpecificLtePdcpSapUser : public LtePdcpSapUser
{
public:
  /**
   * Constructor
   *
   * \param rrc RRC
   */
  LtePdcpSpecificLtePdcpSapUser (C* rrc);

  // Interface implemented from LtePdcpSapUser
  virtual void ReceivePdcpSdu (ReceivePdcpSduParameters params);

private:
  LtePdcpSpecificLtePdcpSapUser ();
  C* m_rrc; ///< RRC
};

template <class C>
LtePdcpSpecificLtePdcpSapUser<C>::LtePdcpSpecificLtePdcpSapUser (C* rrc)
  : m_rrc (rrc)
{
}

template <class C>
LtePdcpSpecificLtePdcpSapUser<C>::LtePdcpSpecificLtePdcpSapUser ()
{
}

template <class C>
void LtePdcpSpecificLtePdcpSapUser<C>::ReceivePdcpSdu (ReceivePdcpSduParameters params)
{
  m_rrc->DoReceivePdcpSdu (params);
}


} // namespace ns3

#endif // LTE_PDCP_SAP_H
