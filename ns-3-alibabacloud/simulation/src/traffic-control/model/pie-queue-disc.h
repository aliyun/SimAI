/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 NITK Surathkal
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
 * Authors: Shravya Ks <shravya.ks0@gmail.com>
 *          Smriti Murali <m.smriti.95@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

/*
 * PORT NOTE: This code was ported from ns-2.36rc1 (queue/pie.h).
 * Most of the comments are also ported from the same.
 */

#ifndef PIE_QUEUE_DISC_H
#define PIE_QUEUE_DISC_H

#include "ns3/queue-disc.h"
#include "ns3/nstime.h"
#include "ns3/boolean.h"
#include "ns3/data-rate.h"
#include "ns3/timer.h"
#include "ns3/event-id.h"
#include "ns3/random-variable-stream.h"

#define BURST_RESET_TIMEOUT 1.5

class PieQueueDiscTestCase;  // Forward declaration for unit test
namespace ns3 {

class TraceContainer;
class UniformRandomVariable;

/**
 * \ingroup traffic-control
 *
 * \brief Implements PIE Active Queue Management discipline
 */
class PieQueueDisc : public QueueDisc
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief PieQueueDisc Constructor
   */
  PieQueueDisc ();

  /**
   * \brief PieQueueDisc Destructor
   */
  virtual ~PieQueueDisc ();

  /**
   * \brief Burst types
   */
  enum BurstStateT
  {
    NO_BURST,
    IN_BURST,
    IN_BURST_PROTECTING,
  };

  /**
   * \brief Get queue delay.
   *
   * \returns The current queue delay.
   */
  Time GetQueueDelay (void);
  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

  // Reasons for dropping packets
  static constexpr const char* UNFORCED_DROP = "Unforced drop";  //!< Early probability drops: proactive
  static constexpr const char* FORCED_DROP = "Forced drop";      //!< Drops due to queue limit: reactive
  static constexpr const char* UNFORCED_MARK = "Unforced mark";  //!< Early probability marks: proactive
  static constexpr const char* CE_THRESHOLD_EXCEEDED_MARK = "CE threshold exceeded mark";  //!< Early probability marks: proactive

protected:
  /**
   * \brief Dispose of the object
   */
  virtual void DoDispose (void);

private:
  friend class::PieQueueDiscTestCase;         // Test code
  virtual bool DoEnqueue (Ptr<QueueDiscItem> item);
  virtual Ptr<QueueDiscItem> DoDequeue (void);
  virtual bool CheckConfig (void);

  /**
   * \brief Initialize the queue parameters.
   */
  virtual void InitializeParams (void);

  /**
   * \brief Check if a packet needs to be dropped due to probability drop
   * \param item queue item
   * \param qSize queue size
   * \returns 0 for no drop, 1 for drop
   */
  bool DropEarly (Ptr<QueueDiscItem> item, uint32_t qSize);

  /**
   * Periodically update the drop probability based on the delay samples:
   * not only the current delay sample but also the trend where the delay
   * is going, up or down
   */
  void CalculateP ();


  static const uint64_t DQCOUNT_INVALID = std::numeric_limits<uint64_t>::max();  //!< Invalid dqCount value

  // ** Variables supplied by user
  Time m_sUpdate;                               //!< Start time of the update timer
  Time m_tUpdate;                               //!< Time period after which CalculateP () is called
  Time m_qDelayRef;                             //!< Desired queue delay
  uint32_t m_meanPktSize;                       //!< Average packet size in bytes
  Time m_maxBurst;                              //!< Maximum burst allowed before random early dropping kicks in
  double m_a;                                   //!< Parameter to pie controller
  double m_b;                                   //!< Parameter to pie controller
  uint32_t m_dqThreshold;                       //!< Minimum queue size in bytes before dequeue rate is measured
  bool m_useDqRateEstimator;                    //!< Enable/Disable usage of dequeue rate estimator for queue delay calculation
  bool  m_isCapDropAdjustment;                  //!< Enable/Disable Cap Drop Adjustment feature mentioned in RFC 8033
  bool m_useEcn;                                //!< Enable ECN Marking functionality
  bool m_useDerandomization;                    //!< Enable Derandomization feature mentioned in RFC 8033
  double m_markEcnTh;                           //!< ECN marking threshold (default 10% as suggested in RFC 8033)
  Time m_activeThreshold;                       //!< Threshold for activating PIE (disabled by default)
  Time m_ceThreshold;                           //!< Threshold above which to CE mark
  bool m_useL4s;                                //!< True if L4S is used (ECT1 packets are marked at CE threshold)

  // ** Variables maintained by PIE
  double m_dropProb;                            //!< Variable used in calculation of drop probability
  Time m_qDelayOld;                             //!< Old value of queue delay
  Time m_qDelay;                                //!< Current value of queue delay
  Time m_burstAllowance;                        //!< Current max burst value in seconds that is allowed before random drops kick in
  uint32_t m_burstReset;                        //!< Used to reset value of burst allowance
  BurstStateT m_burstState;                     //!< Used to determine the current state of burst
  bool m_inMeasurement;                         //!< Indicates whether we are in a measurement cycle
  double m_avgDqRate;                           //!< Time averaged dequeue rate
  Time m_dqStart;                               //!< Start timestamp of current measurement cycle
  uint64_t m_dqCount;                           //!< Number of bytes departed since current measurement cycle starts
  EventId m_rtrsEvent;                          //!< Event used to decide the decision of interval of drop probability calculation
  Ptr<UniformRandomVariable> m_uv;              //!< Rng stream
  double m_accuProb;                            //!< Accumulated drop probability
  bool m_active;                                //!< Indicates whether PIE is in active state or not
};

};   // namespace ns3

#endif

