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
 * Author: Marco Miozzo <mmiozzo@cttc.es>
 */



#ifndef LTE_UE_PHY_SAP_H
#define LTE_UE_PHY_SAP_H

#include <ns3/packet.h>

namespace ns3 {

class LteControlMessage;

 /**
  * Service Access Point (SAP) offered by the UE-PHY to the UE-MAC
  *
  * This is the PHY SAP Provider, i.e., the part of the SAP that contains
  * the PHY methods called by the MAC
  */
class LteUePhySapProvider
{
public:
  virtual ~LteUePhySapProvider ();

  /**
   * \brief Send the MAC PDU to the channel
   *
   * \param p the MAC PDU to send
   */
  virtual void SendMacPdu (Ptr<Packet> p) = 0;

  /**
   * \brief Send SendLteControlMessage (PDCCH map, CQI feedbacks) using the ideal control channel
   *
   * \param msg the Ideal Control Message to send
   */
  virtual void SendLteControlMessage (Ptr<LteControlMessage> msg) = 0;

  /** 
   * \brief Send a preamble on the PRACH
   * 
   * \param prachId the ID of the preamble
   * \param raRnti the RA RNTI
   */
  virtual void SendRachPreamble (uint32_t prachId, uint32_t raRnti) = 0;

  /**
   * \brief Notify PHY about the successful RRC connection
   * establishment.
   */
  virtual void NotifyConnectionSuccessful () = 0;

};


 /**
  * Service Access Point (SAP) offered by the PHY to the MAC
  *
  * This is the PHY SAP User, i.e., the part of the SAP that contains the MAC
  * methods called by the PHY
  */
class LteUePhySapUser
{
public:
  virtual ~LteUePhySapUser ();


  /**
   * \brief Receive Phy Pdu funtion.
   *
   * It is called by the Phy to notify the MAC of the reception of a new PHY-PDU
   *
   * \param p
   */
  virtual void ReceivePhyPdu (Ptr<Packet> p) = 0;

  /**
   * \brief Trigger the start from a new frame (input from Phy layer)
   *
   * \param frameNo frame number
   * \param subframeNo subframe number
   */
  virtual void SubframeIndication (uint32_t frameNo, uint32_t subframeNo) = 0;

  /**
   * \brief Receive SendLteControlMessage (PDCCH map, CQI feedbacks) using the ideal control channel
   *
   * \param msg the Ideal Control Message to receive
   */
  virtual void ReceiveLteControlMessage (Ptr<LteControlMessage> msg) = 0;

};



} // namespace ns3


#endif // LTE_UE_PHY_SAP_H
