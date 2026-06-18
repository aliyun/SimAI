/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef WIFI_ACK_MANAGER_H
#define WIFI_ACK_MANAGER_H

#include "wifi-acknowledgment.h"
#include <memory>
#include "ns3/object.h"


namespace ns3 {

class WifiTxParameters;
class WifiMacQueueItem;
class WifiPsdu;
class WifiMac;

/**
 * \ingroup wifi
 *
 * WifiAckManager is an abstract base class. Each subclass defines a logic
 * to select the acknowledgment method for a given frame.
 */
class WifiAckManager : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual ~WifiAckManager ();

  /**
   * Set the MAC which is using this Acknowledgment Manager
   *
   * \param mac a pointer to the MAC
   */
  void SetWifiMac (Ptr<WifiMac> mac);

  /**
   * Set the QoS Ack policy for the given MPDU, which must be a QoS data frame.
   *
   * \param item the MPDU
   * \param acknowledgment the WifiAcknowledgment object storing the QoS Ack policy to set
   */
  static void SetQosAckPolicy (Ptr<WifiMacQueueItem> item, const WifiAcknowledgment* acknowledgment);

  /**
   * Set the QoS Ack policy for the given PSDU, which must include at least a QoS data frame.
   *
   * \param psdu the PSDU
   * \param acknowledgment the WifiAcknowledgment object storing the QoS Ack policy to set
   */
  static void SetQosAckPolicy (Ptr<WifiPsdu> psdu, const WifiAcknowledgment* acknowledgment);

  /**
   * Determine the acknowledgment method to use if the given MPDU is added to the current
   * frame. Return a null pointer if the acknowledgment method is unchanged or the new
   * acknowledgment method otherwise.
   *
   * \param mpdu the MPDU to be added to the current frame
   * \param txParams the current TX parameters for the current frame
   * \return a null pointer if the acknowledgment method is unchanged or the new
   *         acknowledgment method otherwise
   */
  virtual std::unique_ptr<WifiAcknowledgment> TryAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                                          const WifiTxParameters& txParams) = 0;

  /**
   * Determine the acknowledgment method to use if the given MSDU is aggregated to the current
   * frame. Return a null pointer if the acknowledgment method is unchanged or the new
   * acknowledgment method otherwise.
   *
   * \param msdu the MSDU to be aggregated to the current frame
   * \param txParams the current TX parameters for the current frame
   * \return a null pointer if the acknowledgment method is unchanged or the new
   *         acknowledgment method otherwise
   */
  virtual std::unique_ptr<WifiAcknowledgment> TryAggregateMsdu (Ptr<const WifiMacQueueItem> msdu,
                                                                const WifiTxParameters& txParams) = 0;

protected:
  void DoDispose (void) override;

  Ptr<WifiMac> m_mac; //!< MAC which is using this Acknowledgment Manager
};



} //namespace ns3

#endif /* WIFI_ACK_MANAGER_H */
