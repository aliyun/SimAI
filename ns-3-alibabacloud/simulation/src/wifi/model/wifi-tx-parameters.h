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

#ifndef WIFI_TX_PARAMETERS_H
#define WIFI_TX_PARAMETERS_H

#include "wifi-tx-vector.h"
#include "wifi-mac-header.h"
#include "ns3/nstime.h"
#include <map>
#include <set>
#include <memory>

namespace ns3 {

class WifiMacQueueItem;
struct WifiProtection;
struct WifiAcknowledgment;

/**
 * \ingroup wifi
 *
 * This class stores the TX parameters (TX vector, protection mechanism,
 * acknowledgment mechanism, TX duration, ...) for a frame of different types
 * (MPDU, A-MPDU, multi-TID A-MPDU, MU PPDU, ...).
 */
class WifiTxParameters
{
public:
  WifiTxParameters ();
  /**
   * Copy constructor.
   *
   * \param txParams the WifiTxParameters to copy
   */
  WifiTxParameters (const WifiTxParameters& txParams);

  /**
   * Copy assignment operator.
   * \param txParams the TX parameters to assign to this object
   * \return the reference to this object
   */
  WifiTxParameters& operator= (const WifiTxParameters& txParams);

  WifiTxVector m_txVector;                               //!< TXVECTOR of the frame being prepared
  std::unique_ptr<WifiProtection> m_protection;          //!< protection method
  std::unique_ptr<WifiAcknowledgment> m_acknowledgment;  //!< acknowledgment method
  Time m_txDuration {Time::Min ()};                      //!< TX duration of the frame

  /**
   * Reset the TX parameters.
   */
  void Clear (void);

  /**
   * Record that an MPDU is being added to the current frame. If an MPDU addressed
   * to the same receiver already exists in the frame, A-MPDU aggregation is considered.
   *
   * \param mpdu the MPDU being added
   */
  void AddMpdu (Ptr<const WifiMacQueueItem> mpdu);

  /**
   * Record that an MSDU is being aggregated to the last MPDU added to the frame
   * that hase the same receiver.
   *
   * \param msdu the MSDU being aggregated
   */
  void AggregateMsdu (Ptr<const WifiMacQueueItem> msdu);

  /**
   * Get the size in bytes of the frame in case the given MPDU is added.
   *
   * \param mpdu the given MPDU
   * \return the size in bytes of the frame in case the given MPDU is added
   */
  uint32_t GetSizeIfAddMpdu (Ptr<const WifiMacQueueItem> mpdu) const;

  /**
   * Get the size in bytes of the frame in case the given MSDU is aggregated.
   *
   * \param msdu the given MSDU
   * \return a pair (size in bytes of the current A-MSDU, size in bytes of the frame)
             in case the given MSDU is aggregated
   */
  std::pair<uint32_t, uint32_t> GetSizeIfAggregateMsdu (Ptr<const WifiMacQueueItem> msdu) const;

  /**
   * Get the size in bytes of the (A-)MPDU addressed to the given receiver.
   *
   * \param receiver the MAC address of the given receiver
   * \return the size in bytes of the (A-)MPDU addressed to the given receiver
   */
  uint32_t GetSize (Mac48Address receiver) const;

  /// information about the frame being prepared for a specific receiver
  struct PsduInfo
    {
      WifiMacHeader header;                             //!< MAC header of the last MPDU added
      uint32_t amsduSize;                               //!< the size in bytes of the MSDU or A-MSDU
                                                        //!< included in the last MPDU added
      uint32_t ampduSize;                               //!< the size in bytes of the A-MPDU if multiple
                                                        //!< MPDUs have been added, and zero otherwise
      std::map<uint8_t, std::set<uint16_t>> seqNumbers; //!< set of the sequence numbers of the MPDUs added
                                                        //!< for each TID
    };

  /**
   * Get a pointer to the information about the PSDU addressed to the given
   * receiver, if present, and a null pointer otherwise.
   *
   * \param receiver the MAC address of the receiver
   * \return a pointer to an entry in the PSDU information map or a null pointer
   */
  const PsduInfo* GetPsduInfo (Mac48Address receiver) const;

  /// Map containing information about the PSDUs addressed to every receiver
  typedef std::map<Mac48Address, PsduInfo> PsduInfoMap;

  /**
   * Get a const reference to the map containing information about PSDUs.
   *
   * \return a const reference to the map containing information about PSDUs
   */
  const PsduInfoMap& GetPsduInfoMap (void) const;

  /**
   * \brief Print the object contents.
   * \param os output stream in which the data should be printed.
   */
  void Print (std::ostream &os) const;

private:
  PsduInfoMap m_info;                  //!< information about the frame being prepared. Handles
                                       //!< multi-TID A-MPDUs, MU PPDUs, etc.
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the output stream
 * \param txParams the TX parameters
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const WifiTxParameters* txParams);

} //namespace ns3

#endif /* WIFI_TX_PARAMETERS_H */
