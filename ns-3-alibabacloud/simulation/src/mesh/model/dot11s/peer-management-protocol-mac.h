/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef PEER_MANAGEMENT_PROTOCOL_MAC_H
#define PEER_MANAGEMENT_PROTOCOL_MAC_H

#include "ns3/mesh-wifi-interface-mac-plugin.h"

namespace ns3 {
class MeshWifiInterfaceMac;
class WifiMacQueueItem;
enum WifiMacDropReason : uint8_t;  // opaque enum declaration
namespace dot11s {
class PeerManagementProtocol;
class IeConfiguration;
class IePeerManagement;
class PeerManagementProtocol;
/**
 * \ingroup dot11s
 *
 * \brief This is plugin to Mesh WiFi MAC, which implements the
 * interface to dot11s peer management protocol: it takes proper
 * frames from MAC-layer, extracts peer link management information
 * element and mesh configuration element and passes it to main part
 * of protocol
 */
class PeerManagementProtocolMac : public MeshWifiInterfaceMacPlugin
{
public:
  /**
   * Constructor
   *
   * \param interface interface index
   * \param protocol peer management protocol
   */
  PeerManagementProtocolMac (uint32_t interface, Ptr<PeerManagementProtocol> protocol);
  ~PeerManagementProtocolMac ();
  
  // Inherited from plugin abstract class
  /**
   * Set pointer to parent
   * \param parent Ptr<MeshWifiInterfaceMac>
   */ 
  void SetParent (Ptr<MeshWifiInterfaceMac> parent);
  /**
   * Receive and process a packet
   * \param packet the packet received
   * \param header the header
   * \returns true if received
   */
  bool Receive (Ptr<Packet> packet, const WifiMacHeader & header);
  /**
   * This method appears to test a few conditions.  If an action frame,
   * it returns true if SELF_PROTECTED.  It then checks if it is either
   * a broadcast or sent on an active link, and returns true if so.
   * Otherwise, it returns false (indicating it is to be dropped)
   * \param packet the packet
   * \param header the header
   * \param from the MAC address of the sender
   * \param to the MAC address of the receiver
   * \returns true if successful, false if to be dropped
   */
  bool UpdateOutcomingFrame (Ptr<Packet> packet, WifiMacHeader & header, Mac48Address from, Mac48Address to);
  /**
   * Add beacon timing and mesh ID information elements, and notify beacon sent
   * \param beacon the beacon
   */
  void UpdateBeacon (MeshWifiBeacon & beacon) const;
  /**
   * Assign the streams
   * \param stream the stream to assign
   * \return the assigned stream
   */
  int64_t AssignStreams (int64_t stream);

  /// \name Statistics
  ///@{
  /**
   * \brief Report statistics
   * \param os the output stream
   */
  void Report (std::ostream &) const;
  /// Reset stats
  void ResetStats ();
  /**
   * Get the link metric
   * \param peerAddress the peer address
   * \return The metric
   */
  uint32_t GetLinkMetric (Mac48Address peerAddress);
  ///@}

private:
  /**
   * assignment operator
   *
   * \param peer the object to assign
   * \returns the assigned value
   */
  PeerManagementProtocolMac& operator= (const PeerManagementProtocolMac & peer);
  /// type conversion operator
  PeerManagementProtocolMac (const PeerManagementProtocolMac &);

  /// allow PeerManagementProtocol class access friend access
  friend class PeerManagementProtocol;
  /// allow PeerLink class access friend access
  friend class PeerLink;
  ///\name Create peer link management frames
  // \{
  /**
   * \brief This structure keeps all fields in peer link management frame,
   * which are not subclasses of WifiInformationElement
   */
  struct PlinkFrameStart
  {
    uint8_t subtype; ///< subtype
    uint16_t aid; ///< aid
    SupportedRates rates; ///< rates
    uint16_t qos; ///< QOS
  };
  /**
   * Create peer link open frame function
   * \returns the packet
   */
  Ptr<Packet> CreatePeerLinkOpenFrame ();
  /**
   * Create peer link confirm frame function
   * \returns the packet
   */
  Ptr<Packet> CreatePeerLinkConfirmFrame ();
  /**
   * Create peer link clode frame function
   * \returns the packet
   */
  Ptr<Packet> CreatePeerLinkCloseFrame ();
  /**
   * Parses the start of the frame, where no WifiInformationElements exist
   * \param packet the packet
   * \returns PlinkFrameStart
   */
  PlinkFrameStart ParsePlinkFrame (Ptr<const Packet> packet);
  // \}
  /**
   *  Closes link when a proper number of successive transmissions have failed
   * \param reason the reason why the MPDU was dropped
   * \param mpdu the dropped MPDU
   */
  void TxError (WifiMacDropReason reason, Ptr<const WifiMacQueueItem> mpdu);
  /**
   * Transmit OK function
   * \param mpdu the MPDU
   */
  void TxOk (Ptr <const WifiMacQueueItem> mpdu);
  // BCA functionality
  /**
   * Set beacon shift function
   * \param shift the beacon time shift
   */
  void SetBeaconShift (Time shift);
  /**
   * Set peer manager protocol function
   * \param protocol the peer manager protocol
   */
  void SetPeerManagerProtcol (Ptr<PeerManagementProtocol> protocol);
  /**
   * Send peer link management frame function
   * \param peerAddress the peer MAC address
   * \param peerMpAddress the peer MP address
   * \param aid the AID
   * \param peerElement IePeerManagement
   * \param meshConfig IeConfiguration
   */
  void SendPeerLinkManagementFrame (
    Mac48Address peerAddress,
    Mac48Address peerMpAddress,
    uint16_t aid,
    IePeerManagement peerElement,
    IeConfiguration meshConfig
    );
  /**
   * \brief debug only, used to print established links
   * \returns the MAC address
   */
  Mac48Address GetAddress () const;
  /// Statistics structure
  struct Statistics
  {
    uint16_t txOpen; ///< transmit open
    uint16_t txConfirm; ///< transmit confirm
    uint16_t txClose; ///< transmit close
    uint16_t rxOpen; ///< receive open
    uint16_t rxConfirm; ///< receive confirm
    uint16_t rxClose; ///< receive close
    uint16_t dropped; ///< dropped
    uint16_t brokenMgt; ///< broken management
    uint16_t txMgt; ///< transmit management
    uint32_t txMgtBytes; ///< transmit management bytes
    uint16_t rxMgt; ///< receive management
    uint32_t rxMgtBytes; ///< receive management bytes
    uint16_t beaconShift; ///< beacon shift

    /// constructor
    Statistics ();
    /**
     * Print function
     * \param os the output stream
     */
    void Print (std::ostream & os) const;
  };

private:
  struct Statistics m_stats; ///< statistics
  ///\name Information about MAC and protocol:
  Ptr<MeshWifiInterfaceMac> m_parent; ///< parent
  uint32_t m_ifIndex; ///< IF index
  Ptr<PeerManagementProtocol> m_protocol; ///< protocol
};

} // namespace dot11s
} // namespace ns3

#endif /* PEER_MANAGEMENT_PROTOCOL_MAC_H */

