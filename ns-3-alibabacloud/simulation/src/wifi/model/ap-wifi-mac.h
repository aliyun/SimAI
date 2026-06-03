/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006, 2009 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#ifndef AP_WIFI_MAC_H
#define AP_WIFI_MAC_H

#include "wifi-mac.h"
#include <unordered_map>

namespace ns3 {

class SupportedRates;
class CapabilityInformation;
class DsssParameterSet;
class ErpInformation;
class EdcaParameterSet;
class MuEdcaParameterSet;
class HtOperation;
class VhtOperation;
class HeOperation;
class CfParameterSet;
class UniformRandomVariable;

/**
 * \brief Wi-Fi AP state machine
 * \ingroup wifi
 *
 * Handle association, dis-association and authentication,
 * of STAs within an infrastructure BSS.
 */
class ApWifiMac : public WifiMac
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  ApWifiMac ();
  virtual ~ApWifiMac ();

  void SetLinkUpCallback (Callback<void> linkUp) override;
  bool CanForwardPacketsTo (Mac48Address to) const override;
  void Enqueue (Ptr<Packet> packet, Mac48Address to) override;
  void Enqueue (Ptr<Packet> packet, Mac48Address to, Mac48Address from) override;
  bool SupportsSendFrom (void) const override;
  void SetAddress (Mac48Address address) override;
  Ptr<WifiMacQueue> GetTxopQueue (AcIndex ac) const override;

  /**
   * \param interval the interval between two beacon transmissions.
   */
  void SetBeaconInterval (Time interval);
  /**
   * \return the interval between two beacon transmissions.
   */
  Time GetBeaconInterval (void) const;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   *
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

  /**
   * Get a const reference to the map of associated stations. Each station is
   * specified by an (association ID, MAC address) pair. Make sure not to use
   * the returned reference after that this object has been deallocated.
   *
   * \return a const reference to the map of associated stations
   */
  const std::map<uint16_t, Mac48Address>& GetStaList (void) const;
  /**
   * \param addr the address of the associated station
   * \return the Association ID allocated by the AP to the station, SU_STA_ID if unallocated
   */
  uint16_t GetAssociationId (Mac48Address addr) const;

  /**
   * Return the value of the Queue Size subfield of the last QoS Data or QoS Null
   * frame received from the station with the given MAC address and belonging to
   * the given TID.
   *
   * The Queue Size value is the total size, rounded up to the nearest multiple
   * of 256 octets and expressed in units of 256 octets, of all  MSDUs and A-MSDUs
   * buffered at the STA (excluding the MSDU or A-MSDU of the present QoS Data frame).
   * A queue size value of 254 is used for all sizes greater than 64 768 octets.
   * A queue size value of 255 is used to indicate an unspecified or unknown size.
   * See Section 9.2.4.5.6 of 802.11-2016
   *
   * \param tid the given TID
   * \param address the given MAC address
   * \return the value of the Queue Size subfield
   */
  uint8_t GetBufferStatus (uint8_t tid, Mac48Address address) const;
  /**
   * Store the value of the Queue Size subfield of the last QoS Data or QoS Null
   * frame received from the station with the given MAC address and belonging to
   * the given TID.
   *
   * \param tid the given TID
   * \param address the given MAC address
   * \param size the value of the Queue Size subfield
   */
  void SetBufferStatus (uint8_t tid, Mac48Address address, uint8_t size);
  /**
   * Return the maximum among the values of the Queue Size subfield of the last
   * QoS Data or QoS Null frames received from the station with the given MAC address
   * and belonging to any TID.
   *
   * \param address the given MAC address
   * \return the maximum among the values of the Queue Size subfields
   */
  uint8_t GetMaxBufferStatus (Mac48Address address) const;

private:
  void Receive (Ptr<WifiMacQueueItem> mpdu)  override;
  /**
   * The packet we sent was successfully received by the receiver
   * (i.e. we received an Ack from the receiver).  If the packet
   * was an association response to the receiver, we record that
   * the receiver is now associated with us.
   *
   * \param mpdu the MPDU that we successfully sent
   */
  void TxOk (Ptr<const WifiMacQueueItem> mpdu);
  /**
   * The packet we sent was successfully received by the receiver
   * (i.e. we did not receive an Ack from the receiver).  If the packet
   * was an association response to the receiver, we record that
   * the receiver is not associated with us yet.
   *
   * \param timeoutReason the reason why the TX timer was started (\see WifiTxTimer::Reason)
   * \param mpdu the MPDU that we failed to sent
   */
  void TxFailed (WifiMacDropReason timeoutReason, Ptr<const WifiMacQueueItem> mpdu);

  /**
   * This method is called to de-aggregate an A-MSDU and forward the
   * constituent packets up the stack. We override the WifiMac version
   * here because, as an AP, we also need to think about redistributing
   * to other associated STAs.
   *
   * \param mpdu the MPDU containing the A-MSDU.
   */
  void DeaggregateAmsduAndForward (Ptr<WifiMacQueueItem> mpdu) override;
  /**
   * Forward the packet down to DCF/EDCAF (enqueue the packet). This method
   * is a wrapper for ForwardDown with traffic id.
   *
   * \param packet the packet we are forwarding to DCF/EDCAF
   * \param from the address to be used for Address 3 field in the header
   * \param to the address to be used for Address 1 field in the header
   */
  void ForwardDown (Ptr<Packet> packet, Mac48Address from, Mac48Address to);
  /**
   * Forward the packet down to DCF/EDCAF (enqueue the packet).
   *
   * \param packet the packet we are forwarding to DCF/EDCAF
   * \param from the address to be used for Address 3 field in the header
   * \param to the address to be used for Address 1 field in the header
   * \param tid the traffic id for the packet
   */
  void ForwardDown (Ptr<Packet> packet, Mac48Address from, Mac48Address to, uint8_t tid);
  /**
   * Forward a probe response packet to the DCF. The standard is not clear on the correct
   * queue for management frames if QoS is supported. We always use the DCF.
   *
   * \param to the address of the STA we are sending a probe response to
   */
  void SendProbeResp (Mac48Address to);
  /**
   * Forward an association or a reassociation response packet to the DCF.
   * The standard is not clear on the correct queue for management frames if QoS is supported.
   * We always use the DCF.
   *
   * \param to the address of the STA we are sending an association response to
   * \param success indicates whether the association was successful or not
   * \param isReassoc indicates whether it is a reassociation response
   */
  void SendAssocResp (Mac48Address to, bool success, bool isReassoc);
  /**
   * Forward a beacon packet to the beacon special DCF.
   */
  void SendOneBeacon (void);

  /**
   * Return the Capability information of the current AP.
   *
   * \return the Capability information that we support
   */
  CapabilityInformation GetCapabilities (void) const;
  /**
   * Return the ERP information of the current AP.
   *
   * \return the ERP information that we support
   */
  ErpInformation GetErpInformation (void) const;
  /**
   * Return the EDCA Parameter Set of the current AP.
   *
   * \return the EDCA Parameter Set that we support
   */
  EdcaParameterSet GetEdcaParameterSet (void) const;
  /**
   * Return the MU EDCA Parameter Set of the current AP.
   *
   * \return the MU EDCA Parameter Set that we support
   */
  MuEdcaParameterSet GetMuEdcaParameterSet (void) const;
  /**
   * Return the HT operation of the current AP.
   *
   * \return the HT operation that we support
   */
  HtOperation GetHtOperation (void) const;
  /**
   * Return the VHT operation of the current AP.
   *
   * \return the VHT operation that we support
   */
  VhtOperation GetVhtOperation (void) const;
  /**
   * Return the HE operation of the current AP.
   *
   * \return the HE operation that we support
   */
  HeOperation GetHeOperation (void) const;
  /**
   * Return an instance of SupportedRates that contains all rates that we support
   * including HT rates.
   *
   * \return SupportedRates all rates that we support
   */
  SupportedRates GetSupportedRates (void) const;
  /**
   * Return the DSSS Parameter Set that we support.
   *
   * \return the DSSS Parameter Set that we support
   */
  DsssParameterSet GetDsssParameterSet (void) const;
  /**
   * Enable or disable beacon generation of the AP.
   *
   * \param enable enable or disable beacon generation
   */
  void SetBeaconGeneration (bool enable);

  /**
   * Update whether short slot time should be enabled or not in the BSS.
   * Typically, short slot time is enabled only when there is no non-ERP station
   * associated  to the AP, and that short slot time is supported by the AP and by all
   * other ERP stations that are associated to the AP. Otherwise, it is disabled.
   */
  void UpdateShortSlotTimeEnabled (void);
  /**
   * Update whether short preamble should be enabled or not in the BSS.
   * Typically, short preamble is enabled only when the AP and all associated
   * stations support short PHY preamble. Otherwise, it is disabled.
   */
  void UpdateShortPreambleEnabled (void);

  /**
   * Return whether protection for non-ERP stations is used in the BSS.
   *
   * \return true if protection for non-ERP stations is used in the BSS,
   *         false otherwise
   */
  bool GetUseNonErpProtection (void) const;

  void DoDispose (void) override;
  void DoInitialize (void) override;

  /**
   * \return the next Association ID to be allocated by the AP
   */
  uint16_t GetNextAssociationId (void);

  Ptr<Txop> m_beaconTxop;                    //!< Dedicated Txop for beacons
  bool m_enableBeaconGeneration;             //!< Flag whether beacons are being generated
  Time m_beaconInterval;                     //!< Beacon interval
  EventId m_beaconEvent;                     //!< Event to generate one beacon
  Ptr<UniformRandomVariable> m_beaconJitter; //!< UniformRandomVariable used to randomize the time of the first beacon
  bool m_enableBeaconJitter;                 //!< Flag whether the first beacon should be generated at random time
  std::map<uint16_t, Mac48Address> m_staList; //!< Map of all stations currently associated to the AP with their association ID
  uint16_t m_numNonErpStations;              //!< Number of non-ERP stations currently associated to the AP
  uint16_t m_numNonHtStations;               //!< Number of non-HT stations currently associated to the AP
  bool m_shortSlotTimeEnabled;               //!< Flag whether short slot time is enabled within the BSS
  bool m_shortPreambleEnabled;               //!< Flag whether short preamble is enabled in the BSS
  bool m_enableNonErpProtection;             //!< Flag whether protection mechanism is used or not when non-ERP STAs are present within the BSS
  Time m_bsrLifetime;                        //!< Lifetime of Buffer Status Reports
  /// store value and timestamp for each Buffer Status Report
  typedef struct
  {
    uint8_t value;  //!< value of BSR
    Time timestamp; //!< timestamp of BSR
  } bsrType;
  /// Per (MAC address, TID) buffer status reports
  std::unordered_map<WifiAddressTidPair, bsrType, WifiAddressTidHash> m_bufferStatus;

  /**
   * TracedCallback signature for association/deassociation events.
   *
   * \param aid the AID of the station
   * \param address the MAC address of the station
   */
  typedef void (* AssociationCallback)(uint16_t aid, Mac48Address address);

  TracedCallback<uint16_t /* AID */, Mac48Address> m_assocLogger;   ///< association logger
  TracedCallback<uint16_t /* AID */, Mac48Address> m_deAssocLogger; ///< deassociation logger
};

} //namespace ns3

#endif /* AP_WIFI_MAC_H */
