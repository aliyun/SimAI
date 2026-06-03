/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Natale Patriciello <natale.patriciello@gmail.com>
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
 */
#ifndef TCP_RATE_OPS_H
#define TCP_RATE_OPS_H

#include "ns3/object.h"
#include "ns3/tcp-tx-item.h"
#include "ns3/traced-callback.h"
#include "ns3/data-rate.h"
#include "ns3/traced-value.h"

namespace ns3 {

/**
 * \brief Interface for all operations that involve a Rate monitoring for TCP.
 *
 * The interface is meant to take information to generate rate information
 * valid for congestion avoidance algorithm such as BBR.
 *
 * Please take a look to the TcpRateLinux class for an example.
 */
class TcpRateOps : public Object
{
public:
  struct TcpRateSample;
  struct TcpRateConnection;

  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * \brief Put the rate information inside the sent skb
   *
   * Snapshot the current delivery information in the skb, to generate
   * a rate sample later when the skb is (s)acked in SkbDelivered ().
   *
   * \param skb The SKB sent
   * \param isStartOfTransmission true if this is a start of transmission
   * (i.e., in_flight == 0)
   */
  virtual void SkbSent (TcpTxItem *skb, bool isStartOfTransmission) = 0;

  /**
   * \brief Update the Rate information after an item is received
   *
   * When an skb is sacked or acked, we fill in the rate sample with the (prior)
   * delivery information when the skb was last transmitted.
   *
   * If an ACK (s)acks multiple skbs (e.g., stretched-acks), this function is
   * called multiple times. We favor the information from the most recently
   * sent skb, i.e., the skb with the highest prior_delivered count.
   *
   * \param skb The SKB delivered ((s)ACKed)
   */
  virtual void SkbDelivered (TcpTxItem * skb) = 0;

  /**
   * \brief If a gap is detected between sends, it means we are app-limited.
   * TODO What the Linux kernel is setting in tp->app_limited?
   * https://elixir.bootlin.com/linux/latest/source/net/ipv4/tcp_rate.c#L177
   *
   * \param cWnd Congestion Window
   * \param in_flight In Flight size (in bytes)
   * \param segmentSize Segment size
   * \param tailSeq Tail Sequence
   * \param nextTx NextTx
   * \param lostOut Number of lost bytes
   * \param retransOut Number of retransmitted bytes
   */
  virtual void CalculateAppLimited (uint32_t cWnd, uint32_t in_flight,
                                    uint32_t segmentSize, const SequenceNumber32 &tailSeq,
                                    const SequenceNumber32 &nextTx, const uint32_t lostOut,
                                    const uint32_t retransOut) = 0;

  /**
   *
   * \brief Generate a TcpRateSample to feed a congestion avoidance algorithm.
   *
   * This function will be called after an ACK (or a SACK) is received. The
   * (S)ACK carries some implicit information, such as how many segments have been
   * lost or delivered. These values will be this function input.
   *
   * \param delivered number of delivered segments (e.g., receiving a cumulative
   * ACK means having more than 1 segment delivered) relative to the most recent
   * (S)ACK received
   * \param lost number of segments that we detected as lost after the reception
   * of the most recent (S)ACK
   * \param priorInFlight number of segments previously considered in flight
   * \param is_sack_reneg Is SACK reneged?
   * \param minRtt Minimum RTT so far
   * \return The TcpRateSample that will be used for CA
   */
  virtual const TcpRateSample & GenerateSample (uint32_t delivered, uint32_t lost,
                                                bool is_sack_reneg, uint32_t priorInFlight,
                                                const Time &minRtt) = 0;

  /**
   * \return The information about the rate connection
   *
   */
  virtual const TcpRateConnection & GetConnectionRate () = 0;

  /**
   * \brief Rate Sample structure
   *
   * A rate sample measures the number of (original/retransmitted) data
   * packets delivered "delivered" over an interval of time "interval_us".
   * The tcp_rate code fills in the rate sample, and congestion
   * control modules that define a cong_control function to run at the end
   * of ACK processing can optionally chose to consult this sample when
   * setting cwnd and pacing rate.
   * A sample is invalid if "delivered" or "interval_us" is negative.
   */
  struct TcpRateSample
  {
    DataRate      m_deliveryRate   {DataRate ("0bps")};//!< The delivery rate sample
    bool          m_isAppLimited   {false};            //!< Indicates whether the rate sample is application-limited
    Time          m_interval       {Seconds (0.0)};    //!< The length of the sampling interval
    int32_t       m_delivered      {0};                //!< The amount of data marked as delivered over the sampling interval
    uint32_t      m_priorDelivered {0};                //!< The delivered count of the most recent packet delivered
    Time          m_priorTime      {Seconds (0.0)};    //!< The delivered time of the most recent packet delivered
    Time          m_sendElapsed    {Seconds (0.0)};    //!< Send time interval calculated from the most recent packet delivered
    Time          m_ackElapsed     {Seconds (0.0)};    //!< ACK time interval calculated from the most recent packet delivered
    uint32_t      m_bytesLoss      {0};                //!< The amount of data marked as lost from the most recent ack received
    uint32_t      m_priorInFlight  {0};                //!< The value if bytes in flight prior to last received ack
    uint32_t      m_ackedSacked    {0};                //!< The amount of data acked and sacked in the last received ack
    /**
     * \brief Is the sample valid?
     * \return true if the sample is valid, false otherwise.
     */
    bool IsValid () const
    {
      return (m_priorTime != Seconds (0.0) || m_interval != Seconds (0.0));
    }
  };

  /**
   * \brief Information about the connection rate
   *
   * In this struct, the values are for the entire connection, and not just
   * for an interval of time
   */
  struct TcpRateConnection
  {
    uint64_t  m_delivered       {0};           //!< The total amount of data in bytes delivered so far
    Time      m_deliveredTime   {Seconds (0)}; //!< Simulator time when m_delivered was last updated
    Time      m_firstSentTime   {Seconds (0)}; //!< The send time of the packet that was most recently marked as delivered
    uint32_t  m_appLimited      {0};           //!< The index of the last transmitted packet marked as application-limited
    uint32_t  m_txItemDelivered {0};           //!< The value of delivered when the acked item was sent
    int32_t   m_rateDelivered   {0};           //!< The amount of data delivered considered to calculate delivery rate.
    Time      m_rateInterval    {Seconds (0)}; //!< The value of interval considered to calculate delivery rate.
    bool      m_rateAppLimited  {false};       //!< Was sample was taken when data is app limited?
  };
};

/**
 * \brief Linux management and generation of Rate information for TCP
 *
 * This class is inspired by what Linux is performing in tcp_rate.c
 */
class TcpRateLinux : public TcpRateOps
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual ~TcpRateLinux () override
  {
  }

  virtual void SkbSent (TcpTxItem *skb, bool isStartOfTransmission) override;
  virtual void SkbDelivered (TcpTxItem * skb) override;
  virtual void CalculateAppLimited (uint32_t cWnd, uint32_t in_flight,
                                    uint32_t segmentSize, const SequenceNumber32 &tailSeq,
                                    const SequenceNumber32 &nextTx, const uint32_t lostOut,
                                    const uint32_t retransOut) override;
  virtual const TcpRateSample & GenerateSample (uint32_t delivered, uint32_t lost,
                                                bool is_sack_reneg, uint32_t priorInFlight,
                                                const Time &minRtt) override;
  virtual const TcpRateConnection & GetConnectionRate () override
  {
    return m_rate;
  }

  /**
   * TracedCallback signature for tcp rate update events.
   *
   * The callback will be fired each time the rate is updated.
   *
   * \param [in] rate The rate information.
   */
  typedef void (* TcpRateUpdated)(const TcpRateConnection &rate);

  /**
   * TracedCallback signature for tcp rate sample update events.
   *
   * The callback will be fired each time the rate sample is updated.
   *
   * \param [in] sample The rate sample that will be passed to congestion control
   * algorithms.
   */
  typedef void (* TcpRateSampleUpdated)(const TcpRateSample &sample);

private:
  // Rate sample related variables
  TcpRateConnection m_rate;         //!< Rate information
  TcpRateSample     m_rateSample;   //!< Rate sample (continuosly updated)

  TracedCallback<const TcpRateConnection &> m_rateTrace;       //!< Rate trace
  TracedCallback<const TcpRateSample &>     m_rateSampleTrace; //!< Rate Sample trace
};

/**
 * \brief Output operator.
 * \param os The output stream.
 * \param sample the TcpRateLinux::TcpRateSample to print.
 * \returns The output stream.
 */
std::ostream & operator<< (std::ostream & os, TcpRateLinux::TcpRateSample const & sample);

/**
 * \brief Output operator.
 * \param os The output stream.
 * \param rate the TcpRateLinux::TcpRateConnection to print.
 * \returns The output stream.
 */
std::ostream & operator<< (std::ostream & os, TcpRateLinux::TcpRateConnection const & rate);

/**
 * Comparison operator
 * \param lhs left operand
 * \param rhs right operand
 * \return true if the operands are equal
 */
bool operator== (TcpRateLinux::TcpRateSample const & lhs, TcpRateLinux::TcpRateSample const & rhs);

/**
 * Comparison operator
 * \param lhs left operand
 * \param rhs right operand
 * \return true if the operands are equal
 */
bool operator== (TcpRateLinux::TcpRateConnection const & lhs, TcpRateLinux::TcpRateConnection const & rhs);

} //namespace ns3

#endif /* TCP_RATE_OPS_H */
