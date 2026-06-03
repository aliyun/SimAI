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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 *         Marco Miozzo <mmiozzo@cttc.es>
 */

#ifndef LTE_ENB_CMAC_SAP_H
#define LTE_ENB_CMAC_SAP_H

#include <ns3/packet.h>
#include <ns3/ff-mac-common.h>
#include <ns3/eps-bearer.h>
#include <ns3/lte-common.h>

namespace ns3 {


class LteMacSapUser;

/**
 * Service Access Point (SAP) offered by the eNB MAC to the eNB RRC
 * See Femto Forum MAC Scheduler Interface Specification v 1.11, Figure 1
 *
 * This is the MAC SAP Provider, i.e., the part of the SAP that contains the MAC methods called by the RRC
 */
class LteEnbCmacSapProvider
{
public:
  virtual ~LteEnbCmacSapProvider ();
  /**
   *
   *
   * @param ulBandwidth
   * @param dlBandwidth
   */
  virtual void ConfigureMac (uint16_t ulBandwidth,
                             uint16_t dlBandwidth) = 0;

  /**
   * Add UE function
   *
   * \param rnti 
   */
  virtual void AddUe (uint16_t rnti) = 0;

  /** 
   * remove the UE, e.g., after handover or termination of the RRC connection
   * 
   * \param rnti 
   */
  virtual void RemoveUe (uint16_t rnti) = 0;

  /**
   * Logical Channel information to be passed to CmacSapProvider::ConfigureLc
   *
   */
  struct LcInfo
  {
    uint16_t rnti;   /**< C-RNTI identifying the UE */
    uint8_t  lcId;    /**< logical channel identifier */
    uint8_t  lcGroup; /**< logical channel group */
    uint8_t  qci;     /**< QoS Class Identifier */
    bool     isGbr;   /**< true if the bearer is GBR, false if the bearer is NON-GBR */
    uint64_t mbrUl;   /**< maximum bitrate in uplink */
    uint64_t mbrDl;   /**< maximum bitrate in downlink */
    uint64_t gbrUl;   /**< guaranteed bitrate in uplink */
    uint64_t gbrDl;   /**< guaranteed bitrate in downlink */
  };

  /**
   * Add a new logical channel
   *
   * \param lcinfo
   * \param msu
   */
  virtual void AddLc (LcInfo lcinfo, LteMacSapUser* msu) = 0;


  /**
   * Reconfigure an existing logical channel
   *
   * \param lcinfo
   */
  virtual void ReconfigureLc (LcInfo lcinfo) = 0;


  /**
   * release an existing logical channel
   *
   * \param rnti
   * \param lcid
   */
  virtual void ReleaseLc (uint16_t rnti, uint8_t lcid) = 0;
  
  /**
   * \brief Parameters for [re]configuring the UE 
   */
  struct UeConfig
  {
    /**
     * UE id within this cell
     */
    uint16_t  m_rnti;
    /**
     * Transmission mode [1..7] (i.e., SISO, MIMO, etc.)
     */
    uint8_t   m_transmissionMode;
  };

  /** 
   * update the configuration of the UE
   * 
   * \param params 
   */
  virtual void UeUpdateConfigurationReq (UeConfig params) = 0;


  /**
   * struct defining the RACH configuration of the MAC
   * 
   */
  struct RachConfig
  {
    uint8_t numberOfRaPreambles; ///< number of RA preambles
    uint8_t preambleTransMax; ///< preamble transmit maximum
    uint8_t raResponseWindowSize; ///< RA response window size
    uint8_t connEstFailCount; ///< the counter value for T300 timer expiration
  };

  /** 
   * 
   * \return the current RACH configuration of the MAC
   */
  virtual RachConfig GetRachConfig () = 0;

  /**
   * \brief AllocateNcRaPreambleReturnValue structure
   * 
   */
  struct AllocateNcRaPreambleReturnValue
  {
    bool valid; ///< true if a valid RA config was allocated, false otherwise
    uint8_t raPreambleId; ///< random access preamble id
    uint8_t raPrachMaskIndex; ///< PRACH mask index
  };

  /** 
   * Allocate a random access preamble for non-contention based random access (e.g., for handover).
   * 
   * \param rnti the RNTI of the UE who will perform non-contention based random access
   * 
   * \return  the newly allocated random access preamble 
   */
  virtual AllocateNcRaPreambleReturnValue AllocateNcRaPreamble (uint16_t rnti) = 0;

};



/**
 * Service Access Point (SAP) offered by the MAC to the RRC
 * See Femto Forum MAC Scheduler Interface Specification v 1.11, Figure 1
 *
 * This is the MAC SAP User, i.e., the part of the SAP that contains the RRC methods called by the MAC
 */
class LteEnbCmacSapUser
{
public:
  virtual ~LteEnbCmacSapUser ();

  /** 
   * request the allocation of a Temporary C-RNTI 
   *
   * \return the T-C-RNTI
   */
  virtual uint16_t AllocateTemporaryCellRnti () = 0;

  /**
   * notify the result of the last LC config operation
   *
   * \param rnti the rnti of the user
   * \param lcid the logical channel id
   * \param success true if the operation was successful, false otherwise
   */
  virtual void NotifyLcConfigResult (uint16_t rnti, uint8_t lcid, bool success) = 0;

  /**
   * \brief Parameters for [re]configuring the UE 
   */
  struct UeConfig
  {
    /**
     * UE id within this cell
     */
    uint16_t  m_rnti;
    /**
     * Transmission mode [1..7] (i.e., SISO, MIMO, etc.)
     */
    uint8_t   m_transmissionMode;
  };

  /** 
   * Notify the RRC of a UE config updated requested by the MAC (normally, by the scheduler)
   * 
   * \param params 
   */
  virtual void RrcConfigurationUpdateInd (UeConfig params) = 0;

  /**
   * \brief Is random access completed function
   *
   * This method is executed to decide if the non contention based
   * preamble has to be reused or not upon preamble expiry. If the random access
   * in connected mode is completed, then the preamble can be reused by other UEs.
   * If not, the same UE retains the preamble and other available preambles are
   * assigned to the required UEs.
   *
   * \param rnti the C-RNTI identifying the user
   * \return true if the random access in connected mode is completed
   */
  virtual bool IsRandomAccessCompleted (uint16_t rnti) = 0;
};







} // namespace ns3


#endif // MAC_SAP_H
