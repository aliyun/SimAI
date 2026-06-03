/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 INRIA/LRC - Computer Networks Laboratory
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Flavio Kobuta <flaviokubota@gmail.com>
 *
 */

#ifndef UPLINK_SCHEDULER_MBQOS_H
#define UPLINK_SCHEDULER_MBQOS_H

#include <stdint.h>
#include "ul-mac-messages.h"
#include "ns3/nstime.h"
#include "wimax-phy.h"
#include "ul-job.h"
#include "service-flow-record.h"
#include "ns3/object.h"
#include "bs-uplink-scheduler.h"
#include "service-flow.h"

namespace ns3 {

class BaseStationNetDevice;
class SSRecord;
class ServiceFlow;
class ServiceFlowRecord;
class UlJob;

/**
 * \ingroup wimax
 * \brief This class implements a Migration-based Quality of Service uplink scheduler(MBQoS).
 *
 * This uplink scheduler uses three queues, the low priority
 * queue, the intermediate queue and the high priority queue.
 * The scheduler serves the requests in strict priority order
 * from the high priority queue to the low priority queue. The
 * low priority queue stores the bandwidth requests of the BE
 * service flow. The intermediate queue holds bandwidth requests
 * sent by rtPS and by nrtPS connections. rtPS and nrtPS requests
 * can migrate to the high priority queue to guarantee that
 * their QoS requirements are met. Besides the requests migrated
 * from the intermediate queue, the high priority queue stores
 * periodic grants and unicast request opportunities that must be
 * scheduled in the following frame. To guarantee the maximum delay
 * requirement, the BS assigns a deadline to each rtPS bandwidth
 * request in the intermediate queue. The minimum bandwidth
 * requirement of both rtPS and nrtPS connections is guaranteed
 * over a window of duration T .
 * Implementation of uplink scheduler:
 * Freitag, J.; da Fonseca, N.L.S., "Uplink Scheduling with Quality of Service in IEEE 802.16 Networks,"
 * Global Telecommunications Conference, 2007. GLOBECOM '07. IEEE , vol., no., pp.2503-2508, 26-30 Nov. 2007
 * URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4411386&isnumber=4410910
 */
class UplinkSchedulerMBQoS : public UplinkScheduler
{
public:
  UplinkSchedulerMBQoS ();
  /**
   * Constructor
   *
   * \param time the time
   */
  UplinkSchedulerMBQoS (Time time);
  ~UplinkSchedulerMBQoS (void);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Get uplink allocations
   * \returns std::list<OfdmUlMapIe>
   */
  std::list<OfdmUlMapIe> GetUplinkAllocations (void) const;

  /**
   * Determines if channel descriptors sent in the current frame are
   * required to be updated
   * \param updateDcd update DCD if true
   * \param updateUcd update UCD if true
   * \param sendDcd send DCD if true
   * \param sendUcd send UCD if true
   */
  void GetChannelDescriptorsToUpdate (bool &updateDcd, bool &updateUcd, bool &sendDcd, bool &sendUcd);
  /**
   * Calculate allocation start time
   * \returns the allocation start time
   */
  uint32_t CalculateAllocationStartTime (void);
  /**
   * Add uplink allocation
   * \param ulMapIe the UL map IE
   * \param allocationSize the allocation size
   * \param symbolsToAllocation the symbols to allocation
   * \param availableSymbols the available symbols
   */
  void AddUplinkAllocation (OfdmUlMapIe &ulMapIe,
                            const uint32_t &allocationSize,
                            uint32_t &symbolsToAllocation,
                            uint32_t &availableSymbols);

  /**
   * Schedule function
   */
  void Schedule (void);
  /**
   * Service unsolicited grants
   * \param ssRecord the SS record
   * \param schedulingType the scheduling type
   * \param ulMapIe the UL map IE
   * \param modulationType the modulation type
   * \param symbolsToAllocation the symbols to allocation
   * \param availableSymbols the available symbols
   */
  void ServiceUnsolicitedGrants (const SSRecord *ssRecord,
                                 enum ServiceFlow::SchedulingType schedulingType,
                                 OfdmUlMapIe &ulMapIe,
                                 const WimaxPhy::ModulationType modulationType,
                                 uint32_t &symbolsToAllocation,
                                 uint32_t &availableSymbols);
  /**
   * Service bandwidth requests
   * \param ssRecord the SS record
   * \param schedulingType the scheduling type
   * \param ulMapIe the UL map IE
   * \param modulationType the modulation type
   * \param symbolsToAllocation the symbols to allocation
   * \param availableSymbols the available symbols
   */
  void ServiceBandwidthRequests (const SSRecord *ssRecord,
                                 enum ServiceFlow::SchedulingType schedulingType,
                                 OfdmUlMapIe &ulMapIe,
                                 const WimaxPhy::ModulationType modulationType,
                                 uint32_t &symbolsToAllocation,
                                 uint32_t &availableSymbols);
  /**
   * Service bandwidth requests
   * \param serviceFlow the service flow
   * \param schedulingType the scheduling type
   * \param ulMapIe the UL map IE
   * \param modulationType the modulation type
   * \param symbolsToAllocation the symbols to allocation
   * \param availableSymbols the available symbols
   * \returns true if successful 
   */
  bool ServiceBandwidthRequests (ServiceFlow *serviceFlow,
                                 enum ServiceFlow::SchedulingType schedulingType,
                                 OfdmUlMapIe &ulMapIe,
                                 const WimaxPhy::ModulationType modulationType,
                                 uint32_t &symbolsToAllocation,
                                 uint32_t &availableSymbols);
  /**
   * Allocate initial ranging interval
   * \param symbolsToAllocation the symbols to allocation
   * \param availableSymbols the available symbols
   */
  void AllocateInitialRangingInterval (uint32_t &symbolsToAllocation, uint32_t &availableSymbols);
  /**
   * Setup service flow
   * \param ssRecord the SS record
   * \param serviceFlow the service flow
   */
  void SetupServiceFlow (SSRecord *ssRecord, ServiceFlow *serviceFlow);

  /**
   * \param availableSymbols available symbols in the uplink frame
   * \brief Check deadline from jobs. Migrate requests if necessary.
   *
   * This method verifies for each rtPS request whether it should be
   * migrated to the high priority queue or not. The conditions for
   * migration are: request deadline expires in the frame following
   * the next one, and the amount of bandwidth requested is less than
   * or equal to the amount of available bytes in the next uplink frame.
   */
  void CheckDeadline (uint32_t &availableSymbols);

  /**
   * \param availableSymbols available symbols in the uplink frame.
   * \brief Check if Minimum bandwidth is guarantee. Migrate requests if necessary.
   *
   * This method first calculate a priority value for each request
   * in the intermediate queue. Then, sorts the intermediate queue
   * according to the priority values. Finally, while there is available
   * bandwidth, the scheduler migrate the requests to the high priority queue.
   */
  void CheckMinimumBandwidth (uint32_t &availableSymbols);

  /**
   * \brief Reset the current window.
   * According to a configured time, reset the window.
   */
  void UplinkSchedWindowTimer (void);

  /**
   * \param priority Priority of queue
   * \param job job information
   *
   * \brief Enqueue a job in a priority queue.
   */
  void EnqueueJob (UlJob::JobPriority priority, Ptr<UlJob> job);

  /**
   * \param priority Priority of queue
   * \return Ptr<UlJob>
   *
   * \brief Dequeue a job from a priority queue.
   */
  Ptr<UlJob> DequeueJob (UlJob::JobPriority priority);

  void ProcessBandwidthRequest (const BandwidthRequestHeader &bwRequestHdr);

  /**
   * \param serviceFlow Service flow of connection
   * \return Time
   *
   * \brief Calculates deadline of a request.
   */
  Time DetermineDeadline (ServiceFlow *serviceFlow);

  /**
   * This method is called once to initialize window.
   */
  void InitOnce (void);

  /**
   * \param jobs List of jobs
   * \returns the symbols count
   *
   * Sum the amount of symbols of each job of a queue
   */
  uint32_t CountSymbolsQueue (std::list<Ptr<UlJob> > jobs);

  /**
   * \param job job
   * \returns the symbols count
   *
   * Count the amount of symbols of a job.
   */
  uint32_t CountSymbolsJobs (Ptr<UlJob> job);

  /**
   * Set requested bandwidth
   * \param sfr the service flow record
   */
  void OnSetRequestedBandwidth (ServiceFlowRecord *sfr);

  /**
   * \param ssRecord Subscriber station record
   * \param schedType Service flow type
   * \param reqType Type of packet
   * \return Ptr<UlJob>
   *
   * Create and fill information of a job.
   */
  Ptr<UlJob>
  CreateUlJob (SSRecord *ssRecord, enum ServiceFlow::SchedulingType schedType, ReqType reqType);

  /**
   * \param serviceFlow ServiceFlow 
   * \return Ptr<UlJob>
   *
   * Get pending size.
   */
  uint32_t
  GetPendingSize (ServiceFlow* serviceFlow);

  /**
   * Service bandwidth requests bytes.
   * \param serviceFlow the service flow
   * \param schedulingType the scheduling type
   * \param ulMapIe the UL map IE
   * \param modulationType the modulation type
   * \param symbolsToAllocation the symbols to allocation
   * \param availableSymbols the available symbols
   * \param allocationSizeBytes the allocation size in bytes
   * \returns true if successful
   */
  bool
  ServiceBandwidthRequestsBytes (ServiceFlow *serviceFlow,
                                 enum ServiceFlow::SchedulingType schedulingType, OfdmUlMapIe &ulMapIe,
                                 const WimaxPhy::ModulationType modulationType,
                                 uint32_t &symbolsToAllocation, uint32_t &availableSymbols, uint32_t allocationSizeBytes);

private:
  std::list<OfdmUlMapIe> m_uplinkAllocations; ///< uplink allocations

  // queues for scheduler
  std::list<Ptr<UlJob> > m_uplinkJobs_high; ///< uplink jobs high priority
  std::list<Ptr<UlJob> > m_uplinkJobs_inter; ///< uplink jobs intermedite priority
  std::list<Ptr<UlJob> > m_uplinkJobs_low; ///< uplink jobs low priority

  // interval to reset window
  Time m_windowInterval; ///< windows interval
};

} // namespace ns3

#endif /* UPLINK_SCHEDULER_MBQOS_H */
