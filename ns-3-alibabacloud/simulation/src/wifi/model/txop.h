/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef TXOP_H
#define TXOP_H

#include "ns3/traced-value.h"
#include "wifi-mac-header.h"

namespace ns3 {

class Packet;
class ChannelAccessManager;
class MacTxMiddle;
class WifiMode;
class WifiMacQueue;
class WifiMacQueueItem;
class UniformRandomVariable;
class CtrlBAckResponseHeader;
class WifiMac;
enum WifiMacDropReason : uint8_t;  // opaque enum declaration

/**
 * \brief Handle packet fragmentation and retransmissions
 * for data and management frames.
 * \ingroup wifi
 *
 * This class implements the packet fragmentation and
 * retransmission policy for data and management frames.
 * It uses the ns3::ChannelAccessManager helper
 * class to decide when to send a packet.
 * Packets are stored in a ns3::WifiMacQueue
 * until they can be sent.
 *
 * The policy currently implemented uses a simple fragmentation
 * threshold: any packet bigger than this threshold is fragmented
 * in fragments whose size is smaller than the threshold.
 *
 * The retransmission policy is also very simple: every packet is
 * retransmitted until it is either successfully transmitted or
 * it has been retransmitted up until the SSRC or SLRC thresholds.
 *
 * The RTS/CTS policy is similar to the fragmentation policy: when
 * a packet is bigger than a threshold, the RTS/CTS protocol is used.
 */

class Txop : public Object
{
public:
  Txop ();

  /**
   * Constructor
   *
   * \param queue the wifi MAC queue
   */
  Txop (Ptr<WifiMacQueue> queue);

  virtual ~Txop ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * typedef for a callback to invoke when an MPDU is dropped.
   */
  typedef Callback <void, WifiMacDropReason, Ptr<const WifiMacQueueItem> > DroppedMpdu;

  /**
   * Enumeration for channel access status
   */
  enum ChannelAccessStatus
  {
    NOT_REQUESTED = 0,
    REQUESTED,
    GRANTED
  };

  /**
   * Check for QoS TXOP.
   *
   * \returns true if QoS TXOP.
   */
  virtual bool IsQosTxop () const;

  /**
   * Set ChannelAccessManager this Txop is associated to.
   *
   * \param manager ChannelAccessManager to associate.
   */
  void SetChannelAccessManager (const Ptr<ChannelAccessManager> manager);
  /**
   * Set the wifi MAC this Txop is associated to.
   *
   * \param mac associated wifi MAC
   */
  virtual void SetWifiMac (const Ptr<WifiMac> mac);
  /**
   * Set MacTxMiddle this Txop is associated to.
   *
   * \param txMiddle MacTxMiddle to associate.
   */
  void SetTxMiddle (const Ptr<MacTxMiddle> txMiddle);

  /**
   * \param callback the callback to invoke when an MPDU is dropped
   */
  virtual void SetDroppedMpduCallback (DroppedMpdu callback);

  /**
   * Return the packet queue associated with this Txop.
   *
   * \return the associated WifiMacQueue
   */
  Ptr<WifiMacQueue > GetWifiMacQueue () const;

  /**
   * Set the minimum contention window size.
   *
   * \param minCw the minimum contention window size.
   */
  void SetMinCw (uint32_t minCw);
  /**
   * Set the maximum contention window size.
   *
   * \param maxCw the maximum contention window size.
   */
  void SetMaxCw (uint32_t maxCw);
  /**
   * Set the number of slots that make up an AIFS.
   *
   * \param aifsn the number of slots that make up an AIFS.
   */
  void SetAifsn (uint8_t aifsn);
  /**
   * Set the TXOP limit.
   *
   * \param txopLimit the TXOP limit.
   *        Value zero corresponds to default Txop.
   */
  void SetTxopLimit (Time txopLimit);
  /**
   * Return the minimum contention window size.
   *
   * \return the minimum contention window size.
   */
  virtual uint32_t GetMinCw (void) const;
  /**
   * Return the maximum contention window size.
   *
   * \return the maximum contention window size.
   */
  virtual uint32_t GetMaxCw (void) const;
  /**
   * Return the number of slots that make up an AIFS.
   *
   * \return the number of slots that make up an AIFS.
   */
  virtual uint8_t GetAifsn (void) const;
  /**
   * Return the TXOP limit.
   *
   * \return the TXOP limit.
   */
  Time GetTxopLimit (void) const;
  /**
   * Update the value of the CW variable to take into account
   * a transmission success or a transmission abort (stop transmission
   * of a packet after the maximum number of retransmissions has been
   * reached). By default, this resets the CW variable to minCW.
   */
  void ResetCw (void);
  /**
   * Update the value of the CW variable to take into account
   * a transmission failure. By default, this triggers a doubling
   * of CW (capped by maxCW).
   */
  void UpdateFailedCw (void);

  /**
   * When sleep operation occurs, if there is a pending packet transmission,
   * it will be reinserted to the front of the queue.
   */
  virtual void NotifySleep (void);
  /**
   * When off operation occurs, the queue gets cleaned up.
   */
  virtual void NotifyOff (void);
  /**
   * When wake up operation occurs, channel access will be restarted.
   */
  virtual void NotifyWakeUp (void);
  /**
   * When on operation occurs, channel access will be started.
   */
  virtual void NotifyOn (void);

  /* Event handlers */
  /**
   * \param packet packet to send.
   * \param hdr header of packet to send.
   *
   * Store the packet in the internal queue until it
   * can be sent safely.
   */
  virtual void Queue (Ptr<Packet> packet, const WifiMacHeader &hdr);

  /**
   * Called by the FrameExchangeManager to notify that channel access has
   * been granted for the given amount of time.
   *
   * \param txopDuration the duration of the TXOP gained (zero for DCF)
   */
  virtual void NotifyChannelAccessed (Time txopDuration = Seconds (0));
  /**
   * Called by the FrameExchangeManager to notify the completion of the transmissions.
   * This method generates a new backoff and restarts access if needed.
   */
  virtual void NotifyChannelReleased (void);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model. Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use.
   *
   * \return the number of stream indices assigned by this model.
   */
  int64_t AssignStreams (int64_t stream);

  /**
   * \return the current channel access status.
   */
  virtual ChannelAccessStatus GetAccessStatus (void) const;

  /**
   * \param nSlots the number of slots of the backoff.
   *
   * Start a backoff by initializing the backoff counter to the number of
   * slots specified.
   */
  void StartBackoffNow (uint32_t nSlots);

protected:
  ///< ChannelAccessManager associated class
  friend class ChannelAccessManager;

  void DoDispose (void) override;
  void DoInitialize (void) override;

  /* Txop notifications forwarded here */
  /**
   * Notify that access request has been received.
   */
  virtual void NotifyAccessRequested (void);

  /**
   * Check if the Txop has frames to transmit.
   * \return true if the Txop has frames to transmit.
   */
  virtual bool HasFramesToTransmit (void);
  /**
   * Generate a new backoff now.
   */
  virtual void GenerateBackoff (void);
  /**
   * Request access from Txop if needed.
   */
  virtual void StartAccessIfNeeded (void);
  /**
   * Request access to the ChannelAccessManager
   */
  void RequestAccess (void);

  /**
   * \returns the current value of the CW variable. The initial value is
   *          minCW.
   */
  uint32_t GetCw (void) const;
  /**
   * Return the current number of backoff slots.
   *
   * \return the current number of backoff slots
   */
  uint32_t GetBackoffSlots (void) const;
  /**
   * Return the time when the backoff procedure started.
   *
   * \return the time when the backoff procedure started
   */
  Time GetBackoffStart (void) const;
  /**
   * Update backoff slots that nSlots has passed.
   *
   * \param nSlots the number of slots to decrement
   * \param backoffUpdateBound the time at which backoff should start
   */
  void UpdateBackoffSlotsNow (uint32_t nSlots, Time backoffUpdateBound);

  Ptr<ChannelAccessManager> m_channelAccessManager; //!< the channel access manager
  DroppedMpdu m_droppedMpduCallback;                //!< the dropped MPDU callback
  Ptr<WifiMacQueue> m_queue;                        //!< the wifi MAC queue
  Ptr<MacTxMiddle> m_txMiddle;                      //!< the MacTxMiddle
  Ptr<WifiMac> m_mac;                               //!< the wifi MAC
  Ptr<UniformRandomVariable> m_rng;                 //!< the random stream

  uint32_t m_cwMin;              //!< the minimum contention window
  uint32_t m_cwMax;              //!< the maximum contention window
  uint32_t m_cw;                 //!< the current contention window
  uint32_t m_backoff;            //!< the current backoff
  ChannelAccessStatus m_access;  //!< channel access status
  uint32_t m_backoffSlots;       //!< the number of backoff slots
  /**
   * the backoffStart variable is used to keep track of the
   * time at which a backoff was started or the time at which
   * the backoff counter was last updated.
   */
  Time m_backoffStart;

  uint8_t m_aifsn;        //!< the AIFSN
  Time m_txopLimit;       //!< the TXOP limit time

  TracedCallback<uint32_t> m_backoffTrace;      //!< backoff trace value
  TracedValue<uint32_t> m_cwTrace;              //!< CW trace value
};

} //namespace ns3

#endif /* TXOP_H */
