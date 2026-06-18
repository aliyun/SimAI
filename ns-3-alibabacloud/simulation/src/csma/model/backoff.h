/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Emmanuelle Laprise
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
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca
 * Derived from the p2p net device file
Transmi */

#ifndef BACKOFF_H
#define BACKOFF_H

#include <stdint.h>
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"

namespace ns3 {

/**
 * \ingroup csma
 * \brief The backoff class is used for calculating backoff times
 * when many net devices can write to the same channel
 */

class Backoff {
public:
  /**
   * Minimum number of backoff slots (when multiplied by m_slotTime, determines minimum backoff time)
   */
  uint32_t m_minSlots; 

  /**
   * Maximum number of backoff slots (when multiplied by m_slotTime, determines maximum backoff time)
   */
  uint32_t m_maxSlots; 

  /**
   * Caps the exponential function when the number of retries reaches m_ceiling.
   */
  uint32_t m_ceiling;

  /**
   * Maximum number of transmission retries before the packet is dropped.
   */
  uint32_t m_maxRetries;

  /**
   * Length of one slot. A slot time, it usually the packet transmission time, if the packet size is fixed.
   */
  Time m_slotTime;

  Backoff (void);
  /**
   * \brief Constructor
   * \param slotTime Length of one slot
   * \param minSlots Minimum number of backoff slots
   * \param maxSlots Maximum number of backoff slots
   * \param ceiling Cap to the exponential function
   * \param maxRetries Maximum number of transmission retries
   */
  Backoff (Time slotTime, uint32_t minSlots, uint32_t maxSlots, uint32_t ceiling, uint32_t maxRetries);

  /**
   * \return The amount of time that the net device should wait before
   * trying to retransmit the packet
   */
  Time GetBackoffTime ();

  /**
   * Indicates to the backoff object that the last packet was
   * successfully transmitted and that the number of retries should be
   * reset to 0.
   */
  void ResetBackoffTime (void);

  /**
   * \return True if the maximum number of retries has been reached
   */
  bool MaxRetriesReached (void);

  /**
   * Increments the number of retries by 1.
   */
  void IncrNumRetries (void);

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  int64_t AssignStreams (int64_t stream);

private:

  /**
   * Number of times that the transmitter has tried to unsuccessfully transmit the current packet.
   */
  uint32_t m_numBackoffRetries;

  /**
   * Random number generator
   */
  Ptr<UniformRandomVariable> m_rng;
};

} // namespace ns3

#endif /* BACKOFF_H */
