/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>
 * Modification: Dizhi Zhou <dizhi.zhou@gmail.com>    // modify codes related to downlink scheduler
 */

#ifndef TDBET_FF_MAC_SCHEDULER_H
#define TDBET_FF_MAC_SCHEDULER_H

#include <ns3/lte-common.h>
#include <ns3/ff-mac-csched-sap.h>
#include <ns3/ff-mac-sched-sap.h>
#include <ns3/ff-mac-scheduler.h>
#include <vector>
#include <map>
#include <ns3/nstime.h>
#include <ns3/lte-amc.h>
#include <ns3/lte-ffr-sap.h>

// value for SINR outside the range defined by FF-API, used to indicate that there
// is no CQI for this element
#define NO_SINR -5000


#define HARQ_PROC_NUM 8
#define HARQ_DL_TIMEOUT 11

namespace ns3 {


typedef std::vector < uint8_t > DlHarqProcessesStatus_t;
typedef std::vector < uint8_t > DlHarqProcessesTimer_t;
typedef std::vector < DlDciListElement_s > DlHarqProcessesDciBuffer_t;
typedef std::vector < std::vector <struct RlcPduListElement_s> > RlcPduList_t; // vector of the LCs and layers per UE
typedef std::vector < RlcPduList_t > DlHarqRlcPduListBuffer_t; // vector of the 8 HARQ processes per UE

typedef std::vector < UlDciListElement_s > UlHarqProcessesDciBuffer_t;
typedef std::vector < uint8_t > UlHarqProcessesStatus_t;


/// tdbetsFlowPerf_t structure
struct tdbetsFlowPerf_t
{
  Time flowStart; ///< flow start time
  unsigned long totalBytesTransmitted; ///< total bytes transmitted
  unsigned int lastTtiBytesTrasmitted; ///< last total bytes transmitted
  double lastAveragedThroughput; ///< last average throughput
};


/**
 * \ingroup ff-api
 * \brief Implements the SCHED SAP and CSCHED SAP for a Time Domain Blind Equal Throughput scheduler
 *
 * This class implements the interface defined by the FfMacScheduler abstract class
 */

class TdBetFfMacScheduler : public FfMacScheduler
{
public:
  /**
   * \brief Constructor
   *
   * Creates the MAC Scheduler interface implementation
   */
  TdBetFfMacScheduler ();

  /**
   * Destructor
   */
  virtual ~TdBetFfMacScheduler ();

  // inherited from Object
  virtual void DoDispose (void);
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited from FfMacScheduler
  virtual void SetFfMacCschedSapUser (FfMacCschedSapUser* s);
  virtual void SetFfMacSchedSapUser (FfMacSchedSapUser* s);
  virtual FfMacCschedSapProvider* GetFfMacCschedSapProvider ();
  virtual FfMacSchedSapProvider* GetFfMacSchedSapProvider ();

  // FFR SAPs
  virtual void SetLteFfrSapProvider (LteFfrSapProvider* s);
  virtual LteFfrSapUser* GetLteFfrSapUser ();

  /// allow MemberCschedSapProvider<TdBetFfMacScheduler> class friend access
  friend class MemberCschedSapProvider<TdBetFfMacScheduler>;
  /// allow MemberSchedSapProvider<TdBetFfMacScheduler> class friend access
  friend class MemberSchedSapProvider<TdBetFfMacScheduler>;

  /**
   * \brief Transmission mode configuration update function
   * \param rnti the RNTI
   * \param txMode the transmission mode
   */
  void TransmissionModeConfigurationUpdate (uint16_t rnti, uint8_t txMode);

private:
  //
  // Implementation of the CSCHED API primitives
  // (See 4.1 for description of the primitives)
  //

  /**
   * \brief CSched cell config request
   * \param params FfMacCschedSapProvider::CschedCellConfigReqParameters
   */
  void DoCschedCellConfigReq (const struct FfMacCschedSapProvider::CschedCellConfigReqParameters& params);

  /**
   * \brief CSched UE config request
   * \param params FfMacCschedSapProvider::CschedUeConfigReqParameters
   */
  void DoCschedUeConfigReq (const struct FfMacCschedSapProvider::CschedUeConfigReqParameters& params);

  /**
   * \brief CSched LC config request
   * \param params FfMacCschedSapProvider::CschedLcConfigReqParameters
   */
  void DoCschedLcConfigReq (const struct FfMacCschedSapProvider::CschedLcConfigReqParameters& params);

  /**
   * \brief CSched LC release request
   * \param params FfMacCschedSapProvider::CschedLcReleaseReqParameters
   */
  void DoCschedLcReleaseReq (const struct FfMacCschedSapProvider::CschedLcReleaseReqParameters& params);

  /**
   * \brief CSched UE release request
   * \param params FfMacCschedSapProvider::CschedUeReleaseReqParameters
   */
  void DoCschedUeReleaseReq (const struct FfMacCschedSapProvider::CschedUeReleaseReqParameters& params);

  //
  // Implementation of the SCHED API primitives
  // (See 4.2 for description of the primitives)
  //

  /**
   * \brief Sched DL RLC buffer request
   * \param params FfMacSchedSapProvider::SchedDlRlcBufferReqParameters
   */
  void DoSchedDlRlcBufferReq (const struct FfMacSchedSapProvider::SchedDlRlcBufferReqParameters& params);

  /**
   * \brief Sched DL paging buffer request
   * \param params FfMacSchedSapProvider::SchedDlPagingBufferReqParameters
   */
  void DoSchedDlPagingBufferReq (const struct FfMacSchedSapProvider::SchedDlPagingBufferReqParameters& params);

  /**
   * \brief Sched DL MAC buffer request
   * \param params FfMacSchedSapProvider::SchedDlMacBufferReqParameters
   */
  void DoSchedDlMacBufferReq (const struct FfMacSchedSapProvider::SchedDlMacBufferReqParameters& params);

  /**
   * \brief Sched DL trigger request
   * \param params FfMacSchedSapProvider::SchedDlTriggerReqParameters
   */
  void DoSchedDlTriggerReq (const struct FfMacSchedSapProvider::SchedDlTriggerReqParameters& params);

  /**
   * \brief Sched DL RACH info request
   * \param params FfMacSchedSapProvider::SchedDlRachInfoReqParameters
   */
  void DoSchedDlRachInfoReq (const struct FfMacSchedSapProvider::SchedDlRachInfoReqParameters& params);

  /**
   * \brief Sched DL CQI info request
   * \param params FfMacSchedSapProvider::SchedDlCqiInfoReqParameters
   */
  void DoSchedDlCqiInfoReq (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params);

  /**
   * \brief Sched UL trigger request
   * \param params FfMacSchedSapProvider::SchedUlTriggerReqParameters
   */
  void DoSchedUlTriggerReq (const struct FfMacSchedSapProvider::SchedUlTriggerReqParameters& params);

  /**
   * \brief Sched UL noise interference request
   * \param params FfMacSchedSapProvider::SchedUlNoiseInterferenceReqParameters
   */
  void DoSchedUlNoiseInterferenceReq (const struct FfMacSchedSapProvider::SchedUlNoiseInterferenceReqParameters& params);

  /**
   * \brief Sched UL SR info request
   * \param params FfMacSchedSapProvider::SchedUlSrInfoReqParameters
   */
  void DoSchedUlSrInfoReq (const struct FfMacSchedSapProvider::SchedUlSrInfoReqParameters& params);

  /**
   * \brief Sched UL MAC control info request
   * \param params FfMacSchedSapProvider::SchedUlMacCtrlInfoReqParameters
   */
  void DoSchedUlMacCtrlInfoReq (const struct FfMacSchedSapProvider::SchedUlMacCtrlInfoReqParameters& params);

  /**
   * \brief Sched UL CQI info request
   * \param params FfMacSchedSapProvider::SchedUlCqiInfoReqParameters
   */
  void DoSchedUlCqiInfoReq (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params);

  /**
   * \brief Get RBG size function
   * \param dlbandwidth DL bandwidth
   * \returns RBG size
   */
  int GetRbgSize (int dlbandwidth);

  /**
   * \brief LC active flow function
   * \param rnti the RNTI
   * \returns LC active flow
   */
  unsigned int LcActivePerFlow (uint16_t rnti);

  /**
   * \brief Estimate UL SINR function
   * \param rnti the RNTI
   * \param rb the RB
   * \returns SINR
   */
  double EstimateUlSinr (uint16_t rnti, uint16_t rb);

  /// Refresh DL CQI maps function
  void RefreshDlCqiMaps (void);
  /// Refresh UL CQI maps function
  void RefreshUlCqiMaps (void);

  /**
   * \brief Update DL RLC buffer info function
   * \param rnti the RNTI
   * \param lcid the LCID
   * \param size the size
   */
  void UpdateDlRlcBufferInfo (uint16_t rnti, uint8_t lcid, uint16_t size);
  /**
   * \brief Update UL RLC buffer info function
   * \param rnti the RNTI
   * \param size the size
   */
  void UpdateUlRlcBufferInfo (uint16_t rnti, uint16_t size);

  /**
  * \brief Update and return a new process Id for the RNTI specified
  *
  * \param rnti the RNTI of the UE to be updated
  * \return the process id  value
  */
  uint8_t UpdateHarqProcessId (uint16_t rnti);

  /**
  * \brief Return the availability of free process for the RNTI specified
  *
  * \param rnti the RNTI of the UE to be updated
  * \return the process id  value
  */
  uint8_t HarqProcessAvailability (uint16_t rnti);

  /**
  * \brief Refresh HARQ processes according to the timers
  *
  */
  void RefreshHarqProcesses ();

  Ptr<LteAmc> m_amc; ///< AMC

  /**
   * Vectors of UE's LC info
  */
  std::map <LteFlowId_t, FfMacSchedSapProvider::SchedDlRlcBufferReqParameters> m_rlcBufferReq;


  /**
  * Map of UE statistics (per RNTI basis) in downlink
  */
  std::map <uint16_t, tdbetsFlowPerf_t> m_flowStatsDl;

  /**
  * Map of UE statistics (per RNTI basis)
  */
  std::map <uint16_t, tdbetsFlowPerf_t> m_flowStatsUl;


  /**
  * Map of UE's DL CQI P01 received
  */
  std::map <uint16_t,uint8_t> m_p10CqiRxed;
  /**
  * Map of UE's timers on DL CQI P01 received
  */
  std::map <uint16_t,uint32_t> m_p10CqiTimers;

  /**
  * Map of UE's DL CQI A30 received
  */
  std::map <uint16_t,SbMeasResult_s> m_a30CqiRxed;
  /**
  * Map of UE's timers on DL CQI A30 received
  */
  std::map <uint16_t,uint32_t> m_a30CqiTimers;

  /**
  * Map of previous allocated UE per RBG
  * (used to retrieve info from UL-CQI)
  */
  std::map <uint16_t, std::vector <uint16_t> > m_allocationMaps;

  /**
  * Map of UEs' UL-CQI per RBG
  */
  std::map <uint16_t, std::vector <double> > m_ueCqi;
  /**
  * Map of UEs' timers on UL-CQI per RBG
  */
  std::map <uint16_t, uint32_t> m_ueCqiTimers;

  /**
  * Map of UE's buffer status reports received
  */
  std::map <uint16_t,uint32_t> m_ceBsrRxed;

  // MAC SAPs
  FfMacCschedSapUser* m_cschedSapUser; ///< CSched SAP user
  FfMacSchedSapUser* m_schedSapUser; ///< Sched SAP user
  FfMacCschedSapProvider* m_cschedSapProvider; ///< CSched SAP provider
  FfMacSchedSapProvider* m_schedSapProvider; ///< Sched SAP provider

  // FFR SAPs
  LteFfrSapUser* m_ffrSapUser; ///< FFR SAP user
  LteFfrSapProvider* m_ffrSapProvider; ///< FFR SAP provider

  // Internal parameters
  FfMacCschedSapProvider::CschedCellConfigReqParameters m_cschedCellConfig; ///< CSched cell config


  double m_timeWindow; ///< time window

  uint16_t m_nextRntiUl; ///< RNTI of the next user to be served next scheduling in UL

  uint32_t m_cqiTimersThreshold; ///< # of TTIs for which a CQI can be considered valid

  std::map <uint16_t,uint8_t> m_uesTxMode; ///< txMode of the UEs

  // HARQ attributes
  /**
  * m_harqOn when false inhibit the HARQ mechanisms (by default active)
  */
  bool m_harqOn;
  std::map <uint16_t, uint8_t> m_dlHarqCurrentProcessId; ///< DL HARQ process ID
  //HARQ status
  // 0: process Id available
  // x>0: process Id equal to `x` transmission count
  std::map <uint16_t, DlHarqProcessesStatus_t> m_dlHarqProcessesStatus; ///< DL HARQ process status
  std::map <uint16_t, DlHarqProcessesTimer_t> m_dlHarqProcessesTimer; ///< DL HARQ process timer
  std::map <uint16_t, DlHarqProcessesDciBuffer_t> m_dlHarqProcessesDciBuffer; ///< DL HARQ process DCI buffer
  std::map <uint16_t, DlHarqRlcPduListBuffer_t> m_dlHarqProcessesRlcPduListBuffer; ///< DL HARQ process RLC PDU list buffer
  std::vector <DlInfoListElement_s> m_dlInfoListBuffered; ///< HARQ retx buffered

  std::map <uint16_t, uint8_t> m_ulHarqCurrentProcessId; ///< UL HARQ current process ID
  //HARQ status
  // 0: process Id available
  // x>0: process Id equal to `x` transmission count
  std::map <uint16_t, UlHarqProcessesStatus_t> m_ulHarqProcessesStatus; ///< UL HARQ process status
  std::map <uint16_t, UlHarqProcessesDciBuffer_t> m_ulHarqProcessesDciBuffer; ///< UL HARQ process DCI buffer


  // RACH attributes
  std::vector <struct RachListElement_s> m_rachList; ///< RACH list
  std::vector <uint16_t> m_rachAllocationMap; ///< RACH allocation map
  uint8_t m_ulGrantMcs; ///< MCS for UL grant (default 0)

};

} // namespace ns3

#endif /* TDBET_FF_MAC_SCHEDULER_H */
