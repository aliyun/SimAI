/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#ifndef UAN_MAC_RC_H
#define UAN_MAC_RC_H

#include "uan-mac.h"
#include "ns3/mac8-address.h"

#include "ns3/nstime.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/traced-callback.h"
#include "ns3/event-id.h"
#include "ns3/random-variable-stream.h"

#include <list>
#include <utility>
#include <vector>


namespace ns3 {

class Address;
class UanTxMode;
class UanHeaderRcRts;
class UanHeaderRcCts;
class UanHeaderRcCtsGlobal;
class UanPhy;


/**
 * Stores reservation info for use in scheduling data channel
 * by reservation channel MAC.
 */
class Reservation
{
public:
  /** Default constructor. */
  Reservation ();
  /**
   * Create Reservation object with given packet list,
   * frame number and max packets.
   * 
   * \param list List of packets for assigned to reservation.
   * \param frameNo Frame number of reservation transmission.
   * \param maxPkts Maximum number of packets to assign to reservation
   *   from packet list (0 = no maximum).
   */
  Reservation (std::list<std::pair <Ptr<Packet>, Mac8Address > > &list, uint8_t frameNo, uint32_t maxPkts = 0);
  /** Destructor */
  ~Reservation ();
  /**
   * Get the number of frames in this Reservation.
   *
   * \return Number of frames.
   */
  uint32_t GetNoFrames () const;
  /**
   * Get the total length of the Reservation.
   *
   * This is the sum of packets with headers.
   *
   * \return Total length, in bytes.
   */
  uint32_t GetLength () const;
  /**
   * Get the list of packets.
   *
   * \return The list of packets.
   */
  const std::list<std::pair <Ptr<Packet>, Mac8Address > > &GetPktList (void) const;
  /**
   * Get the frame number.
   * 
   * \return The frame number.
   */
  uint8_t GetFrameNo () const;
  /**
   * Get the retry number.
   *
   * \return The retry number.
   */
  uint8_t GetRetryNo () const;
  /**
   * Get the timestamp for the n'th RTS.
   *
   * \param n Which retry number.
   * \return N'th timestamp.
   */
  Time GetTimestamp (uint8_t n) const;

  /** \return True if reservation packets have been transmitted. */
  bool IsTransmitted () const;
  /**
   * Set the frame number.
   *
   * \param fn The frame number.
   */
  void SetFrameNo (uint8_t fn);
  /**
   * Set the time of the latest RTS sent.
   *
   * \param t RTS timestamp.
   */
  void AddTimestamp (Time t);
  /** Increment the retry count. */
  void IncrementRetry ();
  /**
   * Set the reservation transmitted state.
   * 
   * \param t True if resevation has been transmitted.
   */
  void SetTransmitted (bool t = true);

private:
  /** Queued packets for each address. */
  std::list<std::pair <Ptr<Packet>, Mac8Address > > m_pktList;
  /** Total length of queued packets. */
  uint32_t m_length;
  /** Frame number. */
  uint8_t m_frameNo;
  /** Timestamps for each retry. */
  std::vector<Time> m_timestamp;
  /** Number of retries. */
  uint8_t m_retryNo;
  /** Has this reservation been transmitted. */
  bool m_transmitted;

};  // class Reservation


/**
 * \ingroup uan
 *
 * Non-gateway node MAC for reservation channel MAC protocol.
 *
 * This MAC protocol assumes a network topology where all traffic
 * is destined for a set of GW nodes which are connected via
 * some out of band (RF?) means.  This particular implementation
 * assumes that there is only a single gateway.
 *
 * For more information on class operation email
 * lentracy@u.washington.edu
 * (This work is, as of yet, unpublished)
 */
class UanMacRc : public UanMac
{
public:
  /** Packet types. */
  enum {
    TYPE_DATA,    //!< Data.
    TYPE_GWPING,  //!< Gateway ping.
    TYPE_RTS,     //!< RTS.
    TYPE_CTS,     //!< CTS.
    TYPE_ACK      //!< ACK.
  };
  /** Default constructor */
  UanMacRc ();
  /** Dummy destructor, DoDispose. */
  virtual ~UanMacRc ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  // Inherited methods
  virtual bool Enqueue (Ptr<Packet> pkt, uint16_t protocolNumber, const Address &dest);
  virtual void SetForwardUpCb (Callback<void, Ptr<Packet>, uint16_t, const Mac8Address&> cb);
  virtual void AttachPhy (Ptr<UanPhy> phy);
  virtual void Clear (void);
  int64_t AssignStreams (int64_t stream);

  /**
   *  TracedCallback signature for dequeue of a packet.
   *
   * \param [in] packet The Packet being received.
   * \param [in] proto The protocol number.
   */
  typedef void (* QueueTracedCallback)
    (Ptr<const Packet> packet, uint32_t proto);
  
private:
  /** MAC state. */
  enum State {
    UNASSOCIATED,  //!< Initial state.
    GWPSENT,       //!< Associated with gateway.
    IDLE,          //!< Finished scheduling packet sends.
    RTSSENT,       //!< RTS just sent.
    DATATX         //!< (Unused).
  };

  State m_state;           //!< MAC state.
  bool m_rtsBlocked;       //!< RTS blocked while processing ACK.

  EventId m_startAgain;    //!< (Unused).
  double m_retryRate;      //!< Number of retry attempts per second (of RTS/GWPING.
  Mac8Address m_assocAddr;  //!< Next hop address.
  Ptr<UanPhy> m_phy;       //!< PHY layer attached to this MAC.
  uint32_t m_numRates;     //!< Number of rates per Phy layer.
  uint32_t m_currentRate;  //!< Rate number corresponding to data rate of current cycle.
  uint32_t m_maxFrames;    //!< Maximum number of frames to include in a single RTS.
  uint32_t m_queueLimit;   //!< Maximum packets to queue at MAC.
  uint8_t m_frameNo;       //!< Current frame number.
  Time m_sifs;             //!< Spacing between frames to account for timing error and processing delay.
  Time m_learnedProp;      //!< Propagation delay to gateway.

  double m_minRetryRate;   //!< Smallest allowed RTS retry rate.
  double m_retryStep;      //!< Retry rate increment.

  uint32_t m_ctsSizeN;     //!< Size of UanHeaderRcCts.
  uint32_t m_ctsSizeG;     //!< Size of UanHeaderCommon and UanHeaderRcCtsGlobal.

  bool m_cleared;          //!< Flag when we've been cleared.

  /** Pending packets. */
  std::list<std::pair <Ptr<Packet>, Mac8Address > > m_pktQueue;
  /** List of scheduled reservations. */
  std::list<Reservation> m_resList;

  /** The callback to forward a packet up to higher layer. */
  Callback<void, Ptr<Packet>, uint16_t, const Mac8Address&> m_forwardUpCb;

  /** A packet was destined for and received at this MAC layer. */
  TracedCallback<Ptr<const Packet>, UanTxMode > m_rxLogger;
  /** A packet arrived at the MAC for transmission. */
  TracedCallback<Ptr<const Packet>, uint32_t > m_enqueueLogger;
  /** A was passed down to the PHY from the MAC. */
  TracedCallback<Ptr<const Packet>, uint32_t > m_dequeueLogger;

  /** The RTS event. */
  EventId m_rtsEvent;
  /**
   * PHY receive ok Callback.
   *
   * \param pkt The received packet.
   * \param sinr (Unused).
   * \param mode Modulation mode.
   */
  void ReceiveOkFromPhy (Ptr<Packet> pkt, double sinr, UanTxMode mode);
  /** Associate with a gateway by sending the first GWPING. */
  void Associate (void);
  /** Periodically retry association. */
  void AssociateTimeout (void);
  /** Send RTS packet. */
  void SendRts (void);
  /** Retry RTS. */
  void RtsTimeout (void);
  /**
   * Create the RTS header from a Reservation.
   *
   * \param res The Reservation.
   * \return A RTS header.
   */
  UanHeaderRcRts CreateRtsHeader (const Reservation &res);
  /**
   * Schedule Packet sends.
   *
   * \param ctsh The CTS header identifying the frame number and delay.
   * \param ctsg The CTS global header giving the transmit time stamp base.
   * \param ctsBytes Number of bytes total in CTS packet.
   */
  void ScheduleData (const UanHeaderRcCts &ctsh, const UanHeaderRcCtsGlobal &ctsg, uint32_t ctsBytes);
  /**
   * Process a received ACK.
   *
   * \param ack The ACK packet.
   */
  void ProcessAck (Ptr<Packet> ack);
  /**
   * Send on packet on the PHY.
   *
   * \param pkt The packet.
   * \param rate The transmission rate.
   */
  void SendPacket (Ptr<Packet> pkt, uint32_t rate);
  /**
   * Check that PHY is ok:
   *   not CTS or ACK
   *   not to my address
   * \return True if PHY is ok.
   */
  bool IsPhy1Ok (void);
  /** Callback to block RST. */
  void BlockRtsing (void);

  /**
   * Global count of calls to Associate, AssociateTimeout,
   * SendRts, and RtsTimeout.
   */
  static uint32_t m_cntrlSends;

  /** Provides exponential random variables. */
  Ptr<ExponentialRandomVariable> m_ev;

protected:
  void DoDispose ();

};  // class UanMacRc

} // namespace ns3

#endif /* UAN_MAC_RC_H */
