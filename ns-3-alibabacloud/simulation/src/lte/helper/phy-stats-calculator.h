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
 */

#ifndef PHY_STATS_CALCULATOR_H_
#define PHY_STATS_CALCULATOR_H_

#include "ns3/lte-stats-calculator.h"
#include "ns3/nstime.h"
#include "ns3/uinteger.h"
#include "ns3/spectrum-value.h"
#include <string>
#include <fstream>

namespace ns3 {

/**
 * \ingroup lte
 *
 * Takes care of storing the information generated at PHY layer. Metrics saved are:
 * - RSRP and average SINR for DL
 *   - Timestamp (in seconds)
 *   - IMSI
 *   - C-RNTI
 *   - RSRP
 *   - SINR
 *   - componentCarrierId
 * - UE SINR
 *   - Timestamp (in seconds)
 *   - Cell ID of the reported Enb
 *   - IMSI
 *   - C-RNTI
 *   - measured and reported SINR value in linear
 *   - componentCarrierId
 * - Interference for UL
 *   - Cell ID of the reported Enb
 *   - IMSI of the scheduled UE
 *   - C-RNTI scheduled
 *   - Measured interference for each RB
 */
class PhyStatsCalculator : public LteStatsCalculator
{
public:
  /**
   * Constructor
   */
  PhyStatsCalculator ();

  /**
   * Destructor
   */
  virtual ~PhyStatsCalculator ();

  // Inherited from ns3::Object
  /**
   *  Register this type.
   *  @return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Set the name of the file where the RSRP/SINR statistics will be stored.
   *
   * @param filename string with the name of the file
   */
  void SetCurrentCellRsrpSinrFilename (std::string filename);

  /**
   * Get the name of the file where the RSRP/SINR statistics will be stored.
   * @return the name of the file where the RSRP/SINR statistics will be stored
   */
  std::string GetCurrentCellRsrpSinrFilename  (void);

  /**
   * Set the name of the file where the UE SINR statistics will be stored.
   *
   * @param filename string with the name of the file
   */
  void SetUeSinrFilename (std::string filename);

  /**
   * Get the name of the file where the UE SINR statistics will be stored.
   * @return the name of the file where the UE SINR statistics will be stored
   */
  std::string GetUeSinrFilename (void);

  /**
   * Set the name of the file where the interference statistics will be stored.
   *
   * @param filename string with the name of the file
   */
  void SetInterferenceFilename (std::string filename);

  /**
   * Get the name of the file where the interference statistics will be stored.
   * @return the name of the file where the interference statistics will be stored
   */
  std::string GetInterferenceFilename (void);

  /**
   * Notifies the stats calculator that an RSRP and SINR report has occurred.
   * @param cellId CellId for which stats are generated
   * @param imsi IMSI of the scheduled UE
   * @param rnti C-RNTI scheduled
   * @param rsrp Reference Signal Received Power
   * @param sinr SINR averaged among RBs
   * @param componentCarrierId
   */
  void ReportCurrentCellRsrpSinr (uint16_t cellId, uint64_t imsi, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId);

  /**
   * Notifies the stats calculator that an UE SINR report has occurred.
   * @param cellId Cell ID of the reported Enb
   * @param imsi IMSI of the scheduled UE
   * @param rnti C-RNTI scheduled
   * @param sinrLinear measured and reported SINR value in linear
   * @param componentCarrierId
   */
  void ReportUeSinr (uint16_t cellId, uint64_t imsi, uint16_t rnti, double sinrLinear, uint8_t componentCarrierId);

  /**
   * Notifies the stats calculator that an interference report has occurred.
   * @param cellId Cell ID of the reported Enb
   * @param interference Measured interference for each RB
   */
  void ReportInterference (uint16_t cellId, Ptr<SpectrumValue> interference);

  /** 
   * trace sink
   * 
   * \param phyStats 
   * \param path 
   * \param cellId 
   * \param rnti 
   * \param rsrp 
   * \param sinr 
   * \param componentCarrierId
   */
  static void ReportCurrentCellRsrpSinrCallback (Ptr<PhyStatsCalculator> phyStats,
                                          std::string path, uint16_t cellId, uint16_t rnti,
                                          double rsrp, double sinr, uint8_t componentCarrierId);
  
  /** 
   * trace sink
   * 
   * \param phyStats 
   * \param path 
   * \param cellId 
   * \param rnti 
   * \param sinrLinear
   * \param componentCarrierId
   */
  static void ReportUeSinr (Ptr<PhyStatsCalculator> phyStats, std::string path,
                     uint16_t cellId, uint16_t rnti, double sinrLinear, uint8_t componentCarrierId);

  /** 
   * trace sink
   * 
   * \param phyStats 
   * \param path 
   * \param cellId 
   * \param interference 
   */
  static void ReportInterference (Ptr<PhyStatsCalculator> phyStats, std::string path,
                           uint16_t cellId, Ptr<SpectrumValue> interference);


private:
  /**
   * When writing RSRP SINR statistics first time to file,
   * columns description is added. Then next lines are
   * appended to file. This value is true if output
   * files have not been opened yet
   */
  bool m_RsrpSinrFirstWrite;

  /**
   * When writing UE SINR statistics first time to file,
   * columns description is added. Then next lines are
   * appended to file. This value is true if output
   * files have not been opened yet
   */
  bool m_UeSinrFirstWrite;

  /**
   * When writing interference statistics first time to file,
   * columns description is added. Then next lines are
   * appended to file. This value is true if output
   * files have not been opened yet
   */
  bool m_InterferenceFirstWrite;

  /**
   * Name of the file where the RSRP/SINR statistics will be saved
   */
  std::string m_RsrpSinrFilename;

  /**
   * Name of the file where the UE SINR statistics will be saved
   */
  std::string m_ueSinrFilename;

  /**
   * Name of the file where the interference statistics will be saved
   */
  std::string m_interferenceFilename;

  /**
   * RSRP statistics output trace file
   */
  std::ofstream m_rsrpOutFile;

  /**
   * UE SINR statistics output trace file
   */
  std::ofstream m_ueSinrOutFile;

  /**
   * Interference statistics output trace file
   */
  std::ofstream m_interferenceOutFile;
};

} // namespace ns3

#endif /* PHY_STATS_CALCULATOR_H_ */
