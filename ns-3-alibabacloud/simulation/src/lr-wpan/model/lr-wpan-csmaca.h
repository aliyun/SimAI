/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Author:
 *  kwong yin <kwong-sang.yin@boeing.com>
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 *  Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */

#ifndef LR_WPAN_CSMACA_H
#define LR_WPAN_CSMACA_H

#include <ns3/object.h>
#include <ns3/event-id.h>
#include <ns3/lr-wpan-mac.h>

namespace ns3 {

class UniformRandomVariable;

/**
 * \ingroup lr-wpan
 *
 * This method informs the MAC whether the channel is idle or busy.
 */
typedef Callback<void, LrWpanMacState> LrWpanMacStateCallback;
/**
 * \ingroup lr-wpan
 *
 * This method informs the transaction cost in a slotted CSMA-CA data transmission.
 * i.e. Reports number of symbols (time) it would take slotted CSMA-CA to process the current transaction.
 * 1 Transaction = 2 CCA + frame transmission (PPDU) + turnaroudtime or Ack time (optional) + IFS
 * See IEEE 802.15.4-2011 (Sections 5.1.1.1 and 5.1.1.4)
 */
typedef Callback<void, uint32_t> LrWpanMacTransCostCallback;
/**
 * \ingroup lr-wpan
 *
 * This class is a helper for the LrWpanMac to manage the Csma/CA
 * state machine according to IEEE 802.15.4-2006, section 7.5.1.4.
 */
class LrWpanCsmaCa : public Object
{

public:
  /**
   * Get the type ID.
   *
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * Default constructor.
   */
  LrWpanCsmaCa (void);
  virtual ~LrWpanCsmaCa (void);
  /**
   * Set the MAC to which this CSMA/CA implementation is attached to.
   *
   * \param mac the used MAC
   */
  void SetMac (Ptr<LrWpanMac> mac);
  /**
   * Get the MAC to which this CSMA/CA implementation is attached to.
   *
   * \return the used MAC
   */
  Ptr<LrWpanMac> GetMac (void) const;

  /**
   * Configure for the use of the slotted CSMA/CA version.
   */
  void SetSlottedCsmaCa (void);
  /**
   * Configure for the use of the unslotted CSMA/CA version.
   */
  void SetUnSlottedCsmaCa (void);
  /**
   * Check if the slotted CSMA/CA version is being used.
   *
   * \return true, if slotted CSMA/CA is used, false otherwise.
   */
  bool IsSlottedCsmaCa (void) const;
  /**
   * Check if the unslotted CSMA/CA version is being used.
   *
   * \return true, if unslotted CSMA/CA is used, false otherwise.
   */
  bool IsUnSlottedCsmaCa (void) const;
  /**
   * Set the minimum backoff exponent value.
   * See IEEE 802.15.4-2006, section 7.4.2, Table 86.
   *
   * \param macMinBE the minimum backoff exponent value
   */
  void SetMacMinBE (uint8_t macMinBE);
  /**
   * Get the minimum backoff exponent value.
   * See IEEE 802.15.4-2006, section 7.4.2, Table 86.
   *
   * \return the minimum backoff exponent value
   */
  uint8_t GetMacMinBE (void) const;
  /**
   * Set the maximum backoff exponent value.
   * See IEEE 802.15.4-2006, section 7.4.2, Table 86.
   *
   * \param macMaxBE the maximum backoff exponent value
   */
  void SetMacMaxBE (uint8_t macMaxBE);
  /**
   * Get the maximum backoff exponent value.
   * See IEEE 802.15.4-2006, section 7.4.2, Table 86.
   *
   * \return the maximum backoff exponent value
   */
  uint8_t GetMacMaxBE (void) const;
  /**
   * Set the maximum number of backoffs.
   * See IEEE 802.15.4-2006, section 7.4.2, Table 86.
   *
   * \param macMaxCSMABackoffs the maximum number of backoffs
   */
  void SetMacMaxCSMABackoffs (uint8_t macMaxCSMABackoffs);

  /**
   * Get the maximum number of backoffs.
   * See IEEE 802.15.4-2006, section 7.4.2, Table 86.
   *
   * \return the maximum number of backoffs
   */
  uint8_t GetMacMaxCSMABackoffs (void) const;
  /**
   * Set the number of symbols forming the basic time period used by the
   * CSMA-CA algorithm.
   * See IEEE 802.15.4-2006, section 7.4.1, Table 85.
   *
   * \param unitBackoffPeriod the period length in symbols
   */
  void SetUnitBackoffPeriod (uint64_t unitBackoffPeriod);
  /**
   * Get the number of symbols forming the basic time period used by the
   * CSMA-CA algorithm.
   * See IEEE 802.15.4-2006, section 7.4.1, Table 85.
   *
   * \return the period length in symbols
   */
  uint64_t GetUnitBackoffPeriod (void) const;
  /**
   * Locates the time to the next backoff period boundary in the SUPERFRAME
   * and returns the amount of time left to this moment.
   *
   * \return time offset to the next slot
   */
  Time GetTimeToNextSlot (void) const;
  /**
   * Start CSMA-CA algorithm (step 1), initialize NB, BE for both slotted and unslotted
   * CSMA-CA. For slotted CSMA-CA initializes CW and starts the backoff slot count.
   */
  void Start (void);
  /**
   * Cancel CSMA-CA algorithm.
   */
  void Cancel (void);
  /**
   * In step 2 of the CSMA-CA, perform a random backoff in the range of 0 to 2^BE -1
   */
  void RandomBackoffDelay (void);
  /**
   * In the slotted CSMA-CA, after random backoff, determine if the remaining
   * CSMA-CA operation can proceed, i.e. can the entire transactions can be
   * transmitted before the end of the CAP. This step is performed between step
   * 2 and 3. This step is NOT performed for the unslotted CSMA-CA. If it can
   * proceed function RequestCCA() is called.
   */
  void CanProceed (void);
  /**
   * Request the Phy to perform CCA (Step 3)
   */
  void RequestCCA (void);
  /**
   * The CSMA algorithm call this function at the end of the CAP to return the MAC state
   * back to to IDLE after a transmission was deferred due to the lack of time in the CAP.
   */
  void DeferCsmaTimeout (void);
  /**
   *  IEEE 802.15.4-2006 section 6.2.2.2
   *  PLME-CCA.confirm status
   *  @param status TRX_OFF, BUSY or IDLE
   *
   * When Phy has completed CCA, it calls back here which in turn execute the final steps
   * of the CSMA-CA algorithm.
   * It checks to see if the Channel is idle, if so check the Contention window  before
   * permitting transmission (step 5). If channel is busy, either backoff and perform CCA again or
   * treat as channel access failure (step 4).
   */
  void PlmeCcaConfirm (LrWpanPhyEnumeration status);
  /**
   * Set the callback function to report a transaction cost in slotted CSMA-CA. The callback is
   * triggered in CanProceed() after calculating the transaction cost (2 CCA checks,transmission cost, turnAroundTime, ifs)
   * in the boundary of an Active Period.
   *
   * \param trans the transaction cost callback
   */
  void SetLrWpanMacTransCostCallback (LrWpanMacTransCostCallback trans);
  /**
   * Set the callback function to the MAC. Used at the end of a Channel Assessment, as part of the
   * interconnections between the CSMA-CA and the MAC. The callback
   * lets MAC know a channel is either idle or busy.
   *
   * \param macState the mac state callback
   */
  void SetLrWpanMacStateCallback (LrWpanMacStateCallback macState);
  /**
   * Set the value of the Battery Life Extension
   *
   * \param batteryLifeExtension the Battery Life Extension value active or inactive
   */
  void SetBatteryLifeExtension (bool batteryLifeExtension);
  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams that have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);
  /**
   * Get the number of CSMA retries
   *
   * \returns the number of CSMA retries
   */
  uint8_t GetNB (void);
  /**
   * Get the value of the Battery Life Extension
   *
   * \returns  true or false to Battery Life Extension support
   */
  bool GetBatteryLifeExtension (void);

private:
  // Disable implicit copy constructors
  /**
   * \brief Copy constructor - defined and not implemented.
   */
  LrWpanCsmaCa (LrWpanCsmaCa const &);
  /**
   * \brief Copy constructor - defined and not implemented.
   * \returns
   */
  LrWpanCsmaCa& operator= (LrWpanCsmaCa const &);

  virtual void DoDispose (void);
  /**
   *  \brief Get the time left in the CAP portion of the Outgoing or Incoming superframe.
   *  \return the time left in the CAP
   */
  Time GetTimeLeftInCap ();
  /**
   * The callback to inform the cost of a transaction in slotted CSMA-CA.
   */
  LrWpanMacTransCostCallback m_lrWpanMacTransCostCallback;
  /**
   * The callback to inform the configured MAC of the CSMA/CA result.
   */
  LrWpanMacStateCallback m_lrWpanMacStateCallback;
  /**
   * Beacon-enabled slotted or nonbeacon-enabled unslotted CSMA-CA.
   */
  bool m_isSlotted;
  /**
   * The MAC instance for which this CSMA/CA implemenation is configured.
   */
  Ptr<LrWpanMac> m_mac;
  /**
   * Number of backoffs for the current transmission.
   */
  uint8_t m_NB;
  /**
   * Contention window length (used in slotted ver only).
   */
  uint8_t m_CW;
  /**
   * Backoff exponent.
   */
  uint8_t m_BE;
  /**
   * Battery Life Extension.
   */
  bool m_macBattLifeExt;
  /**
   * Minimum backoff exponent. 0 - macMaxBE, default 3
   */
  uint8_t m_macMinBE;                   //
  /**
   * Maximum backoff exponent. 3 - 8, default 5
   */
  uint8_t m_macMaxBE;
  /**
   * Maximum number of backoffs. 0 - 5, default 4
   */
  uint8_t m_macMaxCSMABackoffs;
  /**
   * Number of symbols per CSMA/CA time unit, default 20 symbols.
   */
  uint64_t m_aUnitBackoffPeriod;
  /**
   * Count the number of remaining random backoff periods left to delay.
   */
  uint64_t m_randomBackoffPeriodsLeft;
  /**
   * Uniform random variable stream.
   */
  Ptr<UniformRandomVariable> m_random;
  /**
   * Scheduler event for the start of the next random backoff/slot.
   */
  EventId m_randomBackoffEvent;
  /**
   * Scheduler event for the end of the current CAP
   */
  EventId m_endCapEvent;
  /**
   * Scheduler event when to start the CCA after a random backoff.
   */
  EventId m_requestCcaEvent;
  /**
   * Scheduler event for checking if we can complete the transmission before the
   * end of the CAP.
   */
  EventId m_canProceedEvent;
  /**
   * Flag indicating that the PHY is currently running a CCA. Used to prevent
   * reporting the channel status to the MAC while canceling the CSMA algorithm.
   */
  bool m_ccaRequestRunning;
  /**
   * Indicates whether the CSMA procedure is targeted for a message to be sent to the coordinator.
   * Used to run slotted CSMA/CA on the incoming or outgoing superframe
   * according to the target.
   */
  bool m_coorDest;

};

}

// namespace ns-3

#endif /* LR_WPAN_CSMACA_H */
