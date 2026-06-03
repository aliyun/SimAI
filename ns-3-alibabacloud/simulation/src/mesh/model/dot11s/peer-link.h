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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 *          Aleksey Kovalenko <kovalenko@iitp.ru>
 */

#ifndef PEER_LINK_H
#define PEER_LINK_H

#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/callback.h"
#include "ns3/mac48-address.h"
#include "ns3/event-id.h"
#include "ns3/ie-dot11s-beacon-timing.h"
#include "ns3/ie-dot11s-peer-management.h"
#include "ns3/ie-dot11s-configuration.h"

namespace ns3 {
namespace dot11s {

class PeerManagementProtocolMac;
/**
 * \ingroup dot11s
 *
 * \brief Peer link model for 802.11s Peer Management protocol
 */
class PeerLink : public Object
{
public:
  /// allow PeerManagementProtocol class friend access
  friend class PeerManagementProtocol;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  /// C-tor create empty link
  PeerLink ();
  ~PeerLink ();
  void DoDispose ();
  /// Peer Link state:
  enum  PeerState {
    IDLE,
    OPN_SNT,
    CNF_RCVD,
    OPN_RCVD,
    ESTAB,
    HOLDING,
  };
  /**
   * \brief Literal names of Mesh Peer Management states for use in log messages
   */
  static const char* const PeerStateNames[6];
  /**
   * Process beacon received from peer
   * 
   * \param lastBeacon the last beacon
   * \param BeaconInterval the beacon interval
   */
  void SetBeaconInformation (Time lastBeacon, Time BeaconInterval);
  /**
   * \brief Method used to detect peer link changes
   *
   * \param cb is a callback, which notifies, that on interface (uint32_t), peer link
   * with address (Mac48Address) was opened (bool is true) or closed (bool is false) 
   */
  void  SetLinkStatusCallback (Callback<void, uint32_t, Mac48Address, bool> cb);
  /// \name Peer link getters/setters
  ///@{
  /**
   * Set the peer address
   * \param macaddr the peer mac address
   */
  void SetPeerAddress (Mac48Address macaddr);
  /**
   * Set the peer mesh point address
   * \param macaddr the peer mesh point address
   */
  void SetPeerMeshPointAddress (Mac48Address macaddr);
  /**
   * Set the interface
   * \param interface the interface
   */
  void SetInterface (uint32_t interface);
  /**
   * Set the local link ID
   * \param id the local link ID
   */
  void SetLocalLinkId (uint16_t id);
  /**
   * Set the local association ID
   * \param aid the local association ID
   */
  void SetLocalAid (uint16_t aid);
  /**
   * Set the peer association ID
   * \return The peer association ID
   */
  uint16_t GetPeerAid () const;
  /**
   * Set the beacon timing element
   * \param beaconTiming the beacon timing element
   */
  void SetBeaconTimingElement (IeBeaconTiming beaconTiming);
  /**
   * Get the peer address
   * \return The peer address
   */
  Mac48Address GetPeerAddress () const;
  /**
   * Get the local association ID
   * \return The local association ID
   */
  uint16_t GetLocalAid () const;
  /**
   * Get the time of the last received beacon
   * \return The time of the last received beacon
   */
  Time GetLastBeacon () const;
  /**
   * Get the beacon interval
   * \return The beacon interval
   */
  Time GetBeaconInterval () const;
  /**
   * Get the beacon timing element
   * \return The beacon timing element
   */
  IeBeaconTiming GetBeaconTimingElement () const;
  //IePeerManagement GetPeerLinkDescriptorElement ()const;
  ///@}

  /// \name MLME
  ///@{
  /**
   * MLME-CancelPeerLink.request
   * \param reason the reason for the request
   */
  void MLMECancelPeerLink (PmpReasonCode reason);
  /// MLME-ActivePeerLinkOpen.request
  void MLMEActivePeerLinkOpen ();
  /// MLME-PeeringRequestReject
  void MLMEPeeringRequestReject ();
  /// Callback type for MLME-SignalPeerLinkStatus event
  typedef Callback<void, uint32_t, Mac48Address, Mac48Address, PeerLink::PeerState, PeerLink::PeerState> SignalStatusCallback;
  /**
   * Set callback
   * \param cb the callback function
   */
  void MLMESetSignalStatusCallback (SignalStatusCallback cb);
  /// Reports about transmission success/failure
  void TransmissionSuccess ();
  void TransmissionFailure ();
  ///@}

  /**
   * \brief Statistics
   * \param os the output stream
   */
  void Report (std::ostream & os) const;
private:
  /// Peer link events, see 802.11s draft 11B.3.3.2
  enum  PeerEvent
  {
    CNCL,       ///< Cancel peer link
    ACTOPN,     ///< Active peer link open
    CLS_ACPT,   ///< PeerLinkClose_Accept
    OPN_ACPT,   ///< PeerLinkOpen_Accept
    OPN_RJCT,   ///< PeerLinkOpen_Reject
    REQ_RJCT,   ///< PeerLinkOpenReject by internal reason
    CNF_ACPT,   ///< PeerLinkConfirm_Accept
    CNF_RJCT,   ///< PeerLinkConfirm_Reject
    TOR1,       ///< Timeout of retry timer
    TOR2,       ///< also timeout of retry timer
    TOC,        ///< Timeout of confirm timer
    TOH         ///< Timeout of holding (graceful closing) timer
  };
  /**
   * State transition
   *
   * \param event the event to update the state machine
   * \param reasoncode the reason for the state transition
   */
  void StateMachine (PeerEvent event, PmpReasonCode = REASON11S_RESERVED);
  /**
   * \name Link response to received management frames
   *
   * \attention In all this methods {local/peer}LinkID correspond to _peer_ station, as written in
   * received frame, e.g. I am peerLinkID and peer link is localLinkID .
   *
   */
  ///@{
  /**
   * Close link
   *
   * \param localLinkID the local link ID
   * \param peerLinkID the peer link ID
   * \param reason the reason to close
   */
  void Close (uint16_t localLinkID, uint16_t peerLinkID, PmpReasonCode reason);
  /**
   * Accept open link
   *
   * \param localLinkId the local link ID
   * \param conf the IE configuration
   * \param peerMp the peer MP
   */
  void OpenAccept (uint16_t localLinkId, IeConfiguration conf, Mac48Address peerMp);
  /**
   * Reject open link
   *
   * \param localLinkId the local link ID
   * \param conf the IE configuration
   * \param peerMp the peer MP
   * \param reason the reason to close
   */
  void OpenReject (uint16_t localLinkId, IeConfiguration conf, Mac48Address peerMp, PmpReasonCode reason);
  /**
   * Confirm accept
   *
   * \param localLinkId the local link ID
   * \param peerLinkId the peer link ID
   * \param peerAid the peer AID
   * \param conf the IE configuration
   * \param peerMp the peer MP
   */
  void ConfirmAccept (
    uint16_t localLinkId,
    uint16_t peerLinkId,
    uint16_t peerAid,
    IeConfiguration conf,
    Mac48Address peerMp
    );
  /**
   * Confirm reject
   *
   * \param localLinkId the local link ID
   * \param peerLinkId the peer link ID
   * \param conf the IE configuration
   * \param peerMp the peer MP
   * \param reason the reason to close
   */
  void  ConfirmReject (
    uint16_t localLinkId,
    uint16_t peerLinkId,
    IeConfiguration  conf,
    Mac48Address peerMp,
    PmpReasonCode reason
    );
  ///@}

  /**
   * \returns True if link is established
   */
  bool  LinkIsEstab () const;
  /**
   * \returns True if link is idle. Link can be deleted in this state
   */
  bool  LinkIsIdle () const;
  /**
   * Set pointer to MAC-plugin, which is responsible for sending peer
   * link management frames
   * \param plugin the peer management protocol MAC
   */
  void SetMacPlugin (Ptr<PeerManagementProtocolMac> plugin);
  /**
   * \name Event handlers
   */
  ///@{
  /// Clear the retry timer
  void ClearRetryTimer ();
  /// Clear the confirm timer
  void ClearConfirmTimer ();
  /// Clear the holding timer
  void ClearHoldingTimer ();
  /// Set the holding timer
  void SetHoldingTimer ();
  /// Set the retry timer
  void SetRetryTimer ();
  /// Set the confirm timer
  void SetConfirmTimer ();
  ///@}

  /**
   * \name Work with management frames
   */
  ///@{
  /**
   * Send a peer link close
   * \param reasoncode reason for closing the line
   */
  void SendPeerLinkClose (PmpReasonCode reasoncode);
  /// Send a peer link open
  void SendPeerLinkOpen ();
  /// Send a peer link confirm
  void SendPeerLinkConfirm ();
  ///@}

  /**
   * \name Timeout handlers
   */
  ///@{
  /// Holding Timeout event handler
  void HoldingTimeout ();
  /// Retry Timeout event handler
  void RetryTimeout ();
  /// Confirm Timeout event handler
  void ConfirmTimeout ();
  ///@}

  /// Several successive beacons were lost, close link
  void BeaconLoss ();
private:

  /**
   * assignment operator
   * \param link the peer link
   * \returns the peer link assigned
   */
  PeerLink& operator= (const PeerLink & link);
  /**
   * Copy constructor. Intentionally unimplemented
   * \param o object to copy
   */
  PeerLink (const PeerLink & o);

  /// The number of interface I am associated with
  uint32_t m_interface;
  /// pointer to MAC plugin, which is responsible for peer management
  Ptr<PeerManagementProtocolMac> m_macPlugin;
  /// Peer address
  Mac48Address m_peerAddress;
  /// Mesh point address, equal to peer address in case of single
  /// interface mesh point
  Mac48Address m_peerMeshPointAddress;
  /// My ID of this link
  uint16_t m_localLinkId;
  /// Peer ID of this link
  uint16_t m_peerLinkId;
  /// My association ID
  uint16_t m_assocId;
  /// Assoc Id assigned to me by peer
  uint16_t m_peerAssocId;

  /// When last beacon was received
  Time  m_lastBeacon;
  /// Current beacon interval on corresponding interface
  Time  m_beaconInterval;
  /// How many successive packets were failed to transmit
  uint16_t m_packetFail;

  /// Current state
  PeerState m_state;
  /**
   * \brief Mesh interface configuration
   * \attention Is not used now, nothing to configure :)
   */
  IeConfiguration m_configuration;
  /// Beacon timing element received from the peer. Needed by BCA
  IeBeaconTiming m_beaconTiming;

  /**
   * \name Timers & counters used for internal state transitions
   */
  ///@{
  uint16_t m_dot11MeshMaxRetries;     //!< Maximum number of retries
  Time     m_dot11MeshRetryTimeout;   //!< Retry timeout
  Time     m_dot11MeshHoldingTimeout; //!< Holding timeout
  Time     m_dot11MeshConfirmTimeout; //!< Confirm timeout

  EventId  m_retryTimer;            //!< Retry timer
  EventId  m_holdingTimer;          //!< Holding timer
  EventId  m_confirmTimer;          //!< Confirm timer
  uint16_t m_retryCounter;          //!< Retry counter
  EventId  m_beaconLossTimer;       //!< Beacon loss timer
  uint16_t m_maxBeaconLoss;         //!< Maximum number of lost beacons before link will be closed
  uint16_t m_maxPacketFail;         //!< Maximum number of failed packets before link will be closed
  ///@}

  /// How to report my status change
  SignalStatusCallback m_linkStatusCallback;
};

} // namespace dot11s
} // namespace ns3

#endif /* PEER_LINK_H */
