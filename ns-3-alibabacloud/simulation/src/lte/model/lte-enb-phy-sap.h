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



#ifndef LTE_ENB_PHY_SAP_H
#define LTE_ENB_PHY_SAP_H

#include <ns3/packet.h>
#include <ns3/ff-mac-common.h>
#include <ns3/ff-mac-sched-sap.h>

namespace ns3 {

class LteControlMessage;

/**
 * Service Access Point (SAP) offered by the eNB-PHY to the eNB-MAC
 *
 * This is the PHY SAP Provider, i.e., the part of the SAP that contains
 * the PHY methods called by the MAC
 */
class LteEnbPhySapProvider
{
public:
  virtual ~LteEnbPhySapProvider ();

  /**
   * \brief Send the MAC PDU to the channel
   * \param p the MAC PDU to send
   */
  virtual void SendMacPdu (Ptr<Packet> p) = 0;

  /** 
   * \brief Send SendLteControlMessage (PDCCH map, CQI feedbacks) using the ideal control channel
   * \param msg the Ideal Control Message to send
   */
  virtual void SendLteControlMessage (Ptr<LteControlMessage> msg) = 0;
  
  /**
  * \brief Get the delay from MAC to Channel expressed in TTIs
  * \returns MAC channel TTI delay
  */
  virtual uint8_t GetMacChTtiDelay () = 0;


};


/**
 * Service Access Point (SAP) offered by the eNB-PHY to the eNB-MAC
 *
 * This is the MAC SAP User, i.e., the part of the SAP that contains the MAC
 * methods called by the PHY
 */
class LteEnbPhySapUser
{
public:
  virtual ~LteEnbPhySapUser ();


  /**
   * Called by the Phy to notify the MAC of the reception of a new PHY-PDU
   *
   * \param p
   */
  virtual void ReceivePhyPdu (Ptr<Packet> p) = 0;

  /**
   * \brief Trigger the start from a new frame (input from Phy layer)
   * \param frameNo frame number
   * \param subframeNo subframe number
   */
  virtual void SubframeIndication (uint32_t frameNo, uint32_t subframeNo) = 0;

  /**
   * \brief Receive SendLteControlMessage (PDCCH map, CQI feedbacks) using the ideal control channel
   * \param msg the Ideal Control Message to receive
   */
  virtual void ReceiveLteControlMessage (Ptr<LteControlMessage> msg) = 0;

  /** 
   * notify the reception of a RACH preamble on the PRACH 
   * 
   * \param prachId the ID of the preamble
   */
  virtual void ReceiveRachPreamble (uint32_t prachId) = 0;

  /**
   * \brief Returns to MAC level the UL-CQI evaluated
   * \param ulcqi the UL-CQI (see FF MAC API 4.3.29)
   */
  virtual void UlCqiReport (FfMacSchedSapProvider::SchedUlCqiInfoReqParameters ulcqi) = 0;

  /**
   * Notify the HARQ on the UL tranmission status
   *
   * \param params
   */
  virtual void UlInfoListElementHarqFeeback (UlInfoListElement_s params) = 0;


  /**
   * Notify the HARQ on the DL tranmission status
   *
   * \param params
   */
  virtual void DlInfoListElementHarqFeeback (DlInfoListElement_s params) = 0;

};



} // namespace ns3


#endif // LTE_ENB_PHY_SAP_H
