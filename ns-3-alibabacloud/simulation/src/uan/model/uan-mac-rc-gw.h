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

#ifndef UAN_MAC_RC_GW_H
#define UAN_MAC_RC_GW_H

#include "uan-mac.h"
#include "ns3/mac8-address.h"

#include "ns3/nstime.h"
#include "ns3/traced-callback.h"




#include <set>
#include <map>

namespace ns3 {

class UanTxMode;

/**
 * \ingroup uan
 *
 * Gateway side of RC-MAC.
 *
 * This MAC protocol assumes a network topology where all traffic
 * is destined for a set of GW nodes which are connected via
 * some out of band (RF?) means.  UanMacRcGw is the protocol
 * which runs on the gateway nodes.
 *
 * This particular implementation
 * assumes that there is only a single gateway.
 *
 * For more information on class operation email
 * lentracy@u.washington.edu
 * (This work is, as of yet, unpublished)
 */
class UanMacRcGw : public UanMac
{
public:
  UanMacRcGw ();           //!< Constructor
  virtual ~UanMacRcGw ();  //!< Dummy destructor, see DoDispose.

  /**
   * Register this type.
   * \return The type ID.
   */
  static TypeId GetTypeId (void);

  // Inherited methods
  virtual bool Enqueue (Ptr<Packet> pkt, uint16_t protocolNumber, const Address &dest);
  virtual void SetForwardUpCb (Callback<void, Ptr<Packet>, uint16_t, const Mac8Address&> cb);
  virtual void AttachPhy (Ptr<UanPhy> phy);
  virtual void Clear (void);
  int64_t AssignStreams (int64_t stream);

  /**
   * TracedCallback signature for
   *
   * \param [in] now, The current simulation time.
   * \param [in] delay Minimum path delay of first RTS.
   * \param [in] numRts Number of pending RTS.
   * \param [in] totalBytes Length of RTS header.
   * \param [in] secs Effective window size for RcCtsGlobal message, or
   *             time to next cycle, both in seconds.
   * \param [in] ctlRate CTL rate in Bps.
   * \param [in] actualX Current retry rate.
   */
  typedef void (* CycleCallback)
    (Time now, Time delay, uint32_t numRts, uint32_t totalBytes,
     double secs, uint32_t ctlRate, double actualX);

  
private:
  /** Gateway state. */
  enum State {
    IDLE,     //!< Initial idle state.
    INCYCLE,  //!< Cycling through nodes.
    CTSING    //!< Sending CTS.
  };
  State m_state;  //!< Gateway processing state.

  /**
   * \ingroup uan
   * Reservation request.
   */
  struct Request
  {
    uint8_t numFrames;  //!< Number of frames.
    uint8_t frameNo;    //!< Current frame number.
    uint8_t retryNo;    //!< Retry number.
    uint16_t length;    //!< Request header length.
    Time rxTime;        //!< Time request received.
  };

  /**
   * \ingroup uan
   * Packet ACK data.
   */
  struct AckData
  {
    uint8_t frameNo;    //!< Frame number being ACK'ed.
    std::set<uint8_t> rxFrames;  //!< Received frames.
    uint8_t expFrames;  //!< Expected number of frames.
  };
  /** Forwarding up callback. */
  Callback<void, Ptr<Packet>, uint16_t, const Mac8Address&> m_forwardUpCb;
  
  Ptr<UanPhy> m_phy;            //!< PHY layer attached to this MAC.
  Time m_maxDelta;              //!< Maximum propagation delay between gateway and non-gateway nodes .
  Time m_sifs;                  //!< Spacing between frames to account for timing error and processing delay.
  uint32_t m_maxRes;            //!< Maximum number of reservations to accept per cycle.
  uint32_t m_numRates;          //!< Number of rates per Phy layer.
  uint32_t m_rtsSize;           //!< Size of UanHeaderCommon and UanHeaderRcRts.
  uint32_t m_ctsSizeN;          //!< Size of UanHeaderRcCts.
  uint32_t m_ctsSizeG;          //!< Size of UanHeaderCommon and UanHeaderRcCtsGlobal.
  uint32_t m_ackSize;           //!< Size of UanHeaderCommon and UanHeaderRcAck.
  uint16_t m_currentRetryRate;  //!< Retry rate number for current cycle.
  uint32_t m_currentRateNum;    //!< Rate number corresponding to data rate of current cycle.
  uint32_t m_numNodes;          //!< Number of non-gateway nodes in this gateway's neighborhood.
  uint32_t m_totalRate;         //!< Total available channel rate in bps (for a single channel, without splitting reservation channel).
  uint32_t m_rateStep;          //!< Increments available for rate assignment in bps.
  uint32_t m_frameSize;         //!< Size of data frames in bytes.

  double m_minRetryRate;        //!< Smallest allowed RTS retry rate.
  double m_retryStep;           //!< Retry rate increment.

  /** Propagation delay to each node. */
  std::map<Mac8Address, Time> m_propDelay;

  /** AckData for each node. */
  std::map<Mac8Address, AckData> m_ackData;

  /** Request for each node. */
  std::map<Mac8Address, Request> m_requests;
  /** Queued request times. */
  std::set<std::pair<Time, Mac8Address> > m_sortedRes;

  /** Flag when we've been cleared. */
  bool m_cleared;

  /** A packet was destined for and received at this MAC layer. */
  TracedCallback<Ptr<const Packet>, UanTxMode > m_rxLogger;

  /**
   * A packet was destined for and received at this MAC layer.
   *
   * \pname{Start time}
   * \pname{min propagation delay}
   * \pname{reservations}
   * \pname{frames}
   * \pname{bytes}
   * \pname{window size}
   * \pname{CTS rate}
   * \pname{retry rate}
   */
  TracedCallback<Time, Time, uint32_t, uint32_t, double, uint32_t, double> m_cycleLogger;

  /**
   * PHY receive ok callback.
   *
   * \param pkt The Packet to receive
   * \param sinr The SINR on the channel
   * \param mode The transmission mode
   */
  void ReceivePacket (Ptr<Packet> pkt, double sinr, UanTxMode mode);

  /** Cycle through pending requests. */
  void StartCycle (void);
  /** End cycle by scheduling pending ACKs. */
  void EndCycle (void);
  /**
   * Send packet on PHY.
   *
   * \param pkt The Packet.
   * \param rate The UanTxMode number, m_currentRateNum.
   */
  void SendPacket (Ptr<Packet> pkt, uint32_t rate);
  /** Set state to INCYCLE. */
  void CycleStarted (void);
  /**
   * PHY receive error callback.
   *
   * \param pkt The failed packet.
   * \param sinr The SINR value on the channel.
   */
  void ReceiveError (Ptr<Packet> pkt, double sinr);

  // Stuff for computing exp throughput
  /**
   * Compute alpha parameter.
   *
   * \param totalFrames Total number of frames in m_requests.
   * \param totalBytes Total number of bytes in m_requests.
   * \param n Number of nodes.
   * \param a m_maxRes, or optimal A value.
   * \param deltaK Propagation delay.
   * \return Alpha parameter.
   */
   double ComputeAlpha (uint32_t totalFrames, uint32_t totalBytes, uint32_t n, uint32_t a, double deltaK);
  /**
   * Get the expected propagation delay to each node.
   *
   * \return Vector of expected propagation delays.
   */
  std::vector<double>  GetExpPdk (void);
  /**
   * Throughput for \pname{a} reservations with framesize \pname{ld},
   * given expected delays exppdk.
   *
   * \param a Number of reservations.
   * \param ld Frame size.
   * \param exppdk Expected delays, given by GetExpPdk.
   * \return Expected throughput.
   */
  double ComputeExpS (uint32_t a, uint32_t ld, std::vector<double> exppdk);
  /**
   * Throughput for \pname{a} reservations with framesize \pname{ld}.
   *
   * \param a Number of reservations.
   * \param ld Frame size.
   * \return Expected throughput.
   */
  double ComputeExpS (uint32_t a, uint32_t ld);
  /**
   * Index to the k'th expected delay among n nodes.
   *
   * \param n Number of nodes.
   * \param k Target index.
   * \return The expected index.
   */
  uint32_t CompExpMinIndex (uint32_t n, uint32_t k);
  /**
   * Numeric function.
   *
   * \param a Number of reservations.
   * \param n number of nodes.
   * \param k K'th node.
   * \return Value.
   */
  double ComputePiK (uint32_t a, uint32_t n, uint32_t k);
  /**
   * Numeric function.
   *
   * \param n Number of nodes.
   * \param a Number of reservations.
   * \param ldlh Sum of common header length and frame size.
   * \param deltaK Result of GetExpPdk
   * \return value.
   */
  double ComputeExpBOverA (uint32_t n, uint32_t a, uint32_t ldlh, std::vector<double> deltaK);
  /**
   * Binomial coefficient.
   *
   * \param n Pool size.
   * \param k Selection size.
   * \return Binomial coefficient n choose k.
   */
  uint64_t NchooseK (uint32_t n, uint32_t k);
  /**
   * Compute the optimum maximum number of reservations to accept per cycle.
   *
   * \return Optimum number.
   */
  uint32_t FindOptA (void);
protected:
  virtual void DoDispose ();

};  // class UanMacRcGw

} // namespace ns3

#endif /* UAN_MAC_RC_GW_H */
