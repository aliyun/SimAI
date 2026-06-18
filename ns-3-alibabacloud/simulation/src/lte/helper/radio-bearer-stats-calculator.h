/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
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
 * Author: Jaume Nin <jnin@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef RADIO_BEARER_STATS_CALCULATOR_H_
#define RADIO_BEARER_STATS_CALCULATOR_H_

#include "ns3/lte-stats-calculator.h"
#include "ns3/lte-common.h"
#include "ns3/uinteger.h"
#include "ns3/object.h"
#include "ns3/basic-data-calculators.h"
#include "ns3/lte-common.h"
#include <string>
#include <map>
#include <fstream>

namespace ns3
{
/// Container: (IMSI, LCID) pair, uint32_t
typedef std::map<ImsiLcidPair_t, uint32_t> Uint32Map;
/// Container: (IMSI, LCID) pair, uint64_t
typedef std::map<ImsiLcidPair_t, uint64_t> Uint64Map;
/// Container: (IMSI, LCID) pair, uint32_t calculator
typedef std::map<ImsiLcidPair_t, Ptr<MinMaxAvgTotalCalculator<uint32_t> > > Uint32StatsMap;
/// Container: (IMSI, LCID) pair, uint64_t calculator
typedef std::map<ImsiLcidPair_t, Ptr<MinMaxAvgTotalCalculator<uint64_t> > > Uint64StatsMap;
/// Container: (IMSI, LCID) pair, double
typedef std::map<ImsiLcidPair_t, double> DoubleMap;
/// Container: (IMSI, LCID) pair, LteFlowId_t
typedef std::map<ImsiLcidPair_t, LteFlowId_t> FlowIdMap;

/**
 * \ingroup lte
 *
 * This class is an ns-3 trace sink that performs the calculation of
 * PDU statistics for uplink and downlink. Statistics are generated
 * on a per radio bearer basis. This class can be used for 
 * RLC PDU stats or PDCP PDU stats by connecting to the appropriate
 * trace sources at the RLC or PDCP layer.
 * 
 * The statistics are calculated at consecutive time windows and
 * periodically written to a file. The calculated statistics are:
 *
 *   - Number of transmitted PDUs
 *   - Number of received PDUs
 *   - Number of transmitted bytes
 *   - Number of received bytes
 *   - Average, min, max and standard deviation of PDU delay (delay is
 *     calculated from the generation of the PDU to its reception)
 *   - Average, min, max and standard deviation of PDU size
 */
class RadioBearerStatsCalculator : public LteStatsCalculator
{
public:
  /**
   * Class constructor
   */
  RadioBearerStatsCalculator ();

  /**
    * Class constructor
    * \param protocolType the name of the protocol type
    */
  RadioBearerStatsCalculator (std::string protocolType);

  /**
   * Class destructor
   */
  virtual
  ~RadioBearerStatsCalculator ();

  // Inherited from ns3::Object
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  void DoDispose ();

  /**
   * Get the name of the file where the uplink statistics will be stored.
   * @return the name of the file where the uplink statistics will be stored
   */
  std::string GetUlOutputFilename (void);

  /**
   * Get the name of the file where the downlink statistics will be stored.
   * @return the name of the file where the downlink statistics will be stored
   */
  std::string GetDlOutputFilename (void);

  /**
   * Set the name of the file where the uplink PDCP statistics will be stored.
   *
   * @param outputFilename string with the name of the file
   */
  void SetUlPdcpOutputFilename (std::string outputFilename);

  /**
   * Get the name of the file where the uplink PDCP statistics will be stored.
   * @return the name of the file where the uplink PDCP statistics will be stored
   */
  std::string GetUlPdcpOutputFilename (void);

  /**
   * Set the name of the file where the downlink PDCP statistics will be stored.
   *
   * @param outputFilename string with the name of the file
   */
  void SetDlPdcpOutputFilename (std::string outputFilename);

  /**
   * Get the name of the file where the downlink PDCP statistics will be stored.
   * @return the name of the file where the downlink PDCP statistics will be stored
   */
  std::string GetDlPdcpOutputFilename (void);


  /** 
   * 
   * \param t the value of the StartTime attribute
   */
  void SetStartTime (Time t);

  /** 
   * 
   * \return the value of the StartTime attribute
   */
  Time GetStartTime () const;

  /** 
   * 
   * \param e the epoch duration
   */
  void SetEpoch (Time e);

  /** 
   * 
   * \return the epoch duration
   */
  Time GetEpoch () const;

  /**
   * Notifies the stats calculator that an uplink transmission has occurred.
   * @param cellId CellId of the attached Enb
   * @param imsi IMSI of the UE who transmitted the PDU
   * @param rnti C-RNTI of the UE who transmitted the PDU
   * @param lcid LCID through which the PDU has been transmitted
   * @param packetSize size of the PDU in bytes
   */
  void
  UlTxPdu (uint16_t cellId, uint64_t imsi, uint16_t rnti, uint8_t lcid, uint32_t packetSize);

  /**
   * Notifies the stats calculator that an uplink reception has occurred.
   * @param cellId CellId of the attached Enb
   * @param imsi IMSI of the UE who received the PDU
   * @param rnti C-RNTI of the UE who received the PDU
   * @param lcid LCID through which the PDU has been received
   * @param packetSize size of the PDU in bytes
   * @param delay RLC to RLC delay in nanoseconds
   */
  void
  UlRxPdu (uint16_t cellId, uint64_t imsi, uint16_t rnti, uint8_t lcid, uint32_t packetSize, uint64_t delay);

  /**
   * Notifies the stats calculator that an downlink transmission has occurred.
   * @param cellId CellId of the attached Enb
   * @param imsi IMSI of the UE who is receiving the PDU
   * @param rnti C-RNTI of the UE who is receiving the PDU
   * @param lcid LCID through which the PDU has been transmitted
   * @param packetSize size of the PDU in bytes
   */
  void
  DlTxPdu (uint16_t cellId, uint64_t imsi, uint16_t rnti, uint8_t lcid, uint32_t packetSize);

  /**
   * Notifies the stats calculator that an downlink reception has occurred.
   * @param cellId CellId of the attached Enb
   * @param imsi IMSI of the UE who received the PDU
   * @param rnti C-RNTI of the UE who received the PDU
   * @param lcid LCID through which the PDU has been transmitted
   * @param packetSize size of the PDU in bytes
   * @param delay RLC to RLC delay in nanoseconds
   */
  void
  DlRxPdu (uint16_t cellId, uint64_t imsi, uint16_t rnti, uint8_t lcid, uint32_t packetSize, uint64_t delay);

  /**
   * Gets the number of transmitted uplink packets.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of transmitted uplink packets
   */
  uint32_t
  GetUlTxPackets (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the number of received uplink packets.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of received uplink packets
   */
  uint32_t
  GetUlRxPackets (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the number of transmitted uplink data bytes.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of transmitted data bytes
   */
  uint64_t
  GetUlTxData (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the number of received uplink data bytes.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of received data bytes
   */
  uint64_t
  GetUlRxData (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the attached Enb cellId.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return Enb cellId
   */
  uint32_t
  GetUlCellId (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the uplink RLC to RLC delay
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return RLC to RLC delay in seconds
   */
  double
  GetUlDelay (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the uplink RLC to RLC statistics: average, min, max and standard deviation.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return RLC to RLC delay statistics average, min, max and standard deviation in seconds
   */
  std::vector<double>
  GetUlDelayStats (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the uplink PDU size statistics: average, min, max and standard deviation.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return PDU size statistics average, min, max and standard deviation in seconds
   */
  std::vector<double>
  GetUlPduSizeStats (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the number of transmitted downlink data bytes.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of transmitted data bytes
   */
  uint32_t
  GetDlTxPackets (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the number of received downlink data bytes.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of received data bytes
   */
  uint32_t
  GetDlRxPackets (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the number of transmitted downlink data bytes.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of transmitted data bytes
   */
  uint64_t
  GetDlTxData (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the number of received downlink data bytes.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return number of received data bytes
   */
  uint64_t
  GetDlRxData (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the attached Enb cellId.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return Enb cellId
   */
  uint32_t
  GetDlCellId (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the downlink RLC to RLC delay
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return RLC to RLC delay in seconds
   */
  double
  GetDlDelay (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the downlink RLC to RLC statistics: average, min, max and standard deviation.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return RLC to RLC delay statistics average, min, max and standard deviation in seconds
   */
  std::vector<double>
  GetDlDelayStats (uint64_t imsi, uint8_t lcid);

  /**
   * Gets the downlink PDU size statistics: average, min, max and standard deviation.
   * @param imsi IMSI of the UE
   * @param lcid LCID
   * @return PDU size statistics average, min, max and standard deviation in seconds
   */
  std::vector<double>
  GetDlPduSizeStats (uint64_t imsi, uint8_t lcid);

private:
  /**
   * Called after each epoch to write collected
   * statistics to output files. During first call
   * it opens output files and write columns descriptions.
   * During next calls it opens output files in append mode.
   */
  void
  ShowResults (void);

  /**
   * Writes collected statistics to UL output file and
   * closes UL output file.
   * @param outFile ofstream for UL statistics
   */
  void
  WriteUlResults (std::ofstream& outFile);

  /**
   * Writes collected statistics to DL output file and
   * closes DL output file.
   * @param outFile ofstream for DL statistics
   */
  void
  WriteDlResults (std::ofstream& outFile);

  /**
   * Erases collected statistics
   */
  void
  ResetResults (void);

  /**
   * Reschedules EndEpoch event. Usually used after
   * execution of SetStartTime() or SetEpoch()
   */
  void RescheduleEndEpoch ();

  /**
   * Function called in every endEpochEvent. It calls
   * ShowResults() to write statistics to output files
   * and ResetResults() to clear collected statistics.
   */
  void EndEpoch (void);

  EventId m_endEpochEvent; //!< Event id for next end epoch event

  FlowIdMap m_flowId; //!< List of FlowIds, ie. (RNTI, LCID) by (IMSI, LCID) pair

  Uint32Map m_dlCellId; //!< List of DL CellIds by (IMSI, LCID) pair
  Uint32Map m_dlTxPackets; //!< Number of DL TX Packets by (IMSI, LCID) pair
  Uint32Map m_dlRxPackets; //!< Number of DL RX Packets by (IMSI, LCID) pair
  Uint64Map m_dlTxData; //!< Amount of DL TX Data by (IMSI, LCID) pair
  Uint64Map m_dlRxData; //!< Amount of DL RX Data by (IMSI, LCID) pair
  Uint64StatsMap m_dlDelay; //!< DL delay by (IMSI, LCID) pair
  Uint32StatsMap m_dlPduSize; //!< DL PDU Size by (IMSI, LCID) pair

  Uint32Map m_ulCellId; //!< List of UL CellIds by (IMSI, LCID) pair
  Uint32Map m_ulTxPackets; //!< Number of UL TX Packets by (IMSI, LCID) pair
  Uint32Map m_ulRxPackets; //!< Number of UL RX Packets by (IMSI, LCID) pair
  Uint64Map m_ulTxData; //!< Amount of UL TX Data by (IMSI, LCID) pair
  Uint64Map m_ulRxData; //!< Amount of UL RX Data by (IMSI, LCID) pair
  Uint64StatsMap m_ulDelay; //!< UL delay by (IMSI, LCID) pair
  Uint32StatsMap m_ulPduSize; //!< UL PDU Size by (IMSI, LCID) pair

  /**
   * Start time of the on going epoch
   */
  Time m_startTime;

  /**
   * Epoch duration
   */
  Time m_epochDuration;

  /**
   * true if output files have not been opened yet
   */
  bool m_firstWrite;

  /**
   * true if any output is pending
   */
  bool m_pendingOutput;

  /**
   * Protocol type, by default RLC
   */
  std::string m_protocolType;

  /**
   * Name of the file where the downlink PDCP statistics will be saved
   */
  std::string m_dlPdcpOutputFilename;

  /**
   * Name of the file where the uplink PDCP statistics will be saved
   */
  std::string m_ulPdcpOutputFilename;

};

} // namespace ns3

#endif /* RADIO_BEARER_STATS_CALCULATOR_H_ */
