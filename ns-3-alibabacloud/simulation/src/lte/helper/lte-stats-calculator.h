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

#ifndef LTE_STATS_CALCULATOR_H_
#define LTE_STATS_CALCULATOR_H_

#include "ns3/object.h"
#include "ns3/string.h"
#include <map>

namespace ns3 {

/**
 * \ingroup lte
 *
 * Base class for ***StatsCalculator classes. Provides
 * basic functionality to parse and store IMSI and CellId.
 * Also stores names of output files.
 */

class LteStatsCalculator : public Object
{
public:
  /**
   * Constructor
   */
  LteStatsCalculator ();

  /**
   * Destructor
   */
  virtual ~LteStatsCalculator ();

  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Set the name of the file where the uplink statistics will be stored.
   *
   * \param outputFilename string with the name of the file
   */
  void SetUlOutputFilename (std::string outputFilename);

  /**
   * Get the name of the file where the uplink statistics will be stored.
   * @return the name of the file where the uplink statistics will be stored
   */
  std::string GetUlOutputFilename (void);

  /**
   * Set the name of the file where the downlink statistics will be stored.
   *
   * @param outputFilename string with the name of the file
   */
  void SetDlOutputFilename (std::string outputFilename);

  /**
   * Get the name of the file where the downlink statistics will be stored.
   * @return the name of the file where the downlink statistics will be stored
   */
  std::string GetDlOutputFilename (void);

  /**
   * Checks if there is an already stored IMSI for the given path
   * @param path Path in the attribute system to check
   * @return true if the path exists, false otherwise
   */
  bool ExistsImsiPath (std::string path);

  /**
   * Stores the (path, imsi) pairs in a map
   * @param path Path in the attribute system to store
   * @param imsi IMSI value to store
   */
  void SetImsiPath (std::string path, uint64_t imsi);

  /**
   * Retrieves the imsi information for the given path
   * @param path Path in the attribute system to get
   * @return the IMSI associated with the given path
   */
  uint64_t GetImsiPath (std::string path);

  /**
   * Checks if there is an already stored cell id for the given path
   * @param path Path in the attribute system to check
   * @return true if the path exists, false otherwise
   */
  bool ExistsCellIdPath (std::string path);

  /**
   * Stores the (path, cellId) pairs in a map
   * @param path Path in the attribute system to store
   * @param cellId cell id value to store
   */
  void SetCellIdPath (std::string path, uint16_t cellId);

  /**
   * Retrieves the cell id information for the given path
   * @param path Path in the attribute system to get
   * @return the cell ID associated with the given path
   */
  uint16_t GetCellIdPath (std::string path);

protected:

  /**
   * Retrieves IMSI from Enb RLC path in the attribute system
   * @param path Path in the attribute system to get
   * @return the IMSI associated with the given path
   */
  static uint64_t FindImsiFromEnbRlcPath (std::string path);

  /**
   * Retrieves IMSI from Ue PHY path in the attribute system
   * @param path Path in the attribute system to get
   * @return the IMSI associated with the given path
   */
  static uint64_t FindImsiFromUePhy (std::string path);

  /**
   * Retrieves IMSI from LteNetDevice path in the attribute system
   * @param path Path in the attribute system to get
   * @return the IMSI associated with the given path
   */
  static uint64_t FindImsiFromLteNetDevice (std::string path);

  /**
   * Retrieves CellId from Enb RLC path in the attribute system
   * @param path Path in the attribute system to get
   * @return the CellId associated with the given path
   */
  static uint16_t FindCellIdFromEnbRlcPath (std::string path);

  /**
   * Retrieves IMSI from Enb MAC path in the attribute system
   * @param path Path in the attribute system to get
   * @param rnti RNTI of UE for which IMSI is needed
   * @return the IMSI associated with the given path and RNTI
   */
  static uint64_t FindImsiFromEnbMac (std::string path, uint16_t rnti);

  /**
   * Retrieves CellId from Enb MAC path in the attribute system
   * @param path Path in the attribute system to get
   * @param rnti RNTI of UE for which CellId is needed
   * @return the CellId associated with the given path and RNTI
   */
  static uint16_t FindCellIdFromEnbMac (std::string path, uint16_t rnti);

  /**
   * Retrieves IMSI from path for Enb in the attribute system
   * @param path Path in the attribute system to get
   * @param rnti RNTI of UE for which IMSI is needed
   * @return the IMSI associated with the given path and RNTI
   */
  static uint64_t FindImsiForEnb (std::string path, uint16_t rnti);

  /**
   * Retrieves IMSI from path for Ue in the attribute system
   * @param path Path in the attribute system to get
   * @param rnti RNTI of UE for which IMSI is needed
   * @return the IMSI associated with the given path and RNTI
   */
  static uint64_t FindImsiForUe (std::string path, uint16_t rnti);

private:
  /**
   * List of IMSI by path in the attribute system
   */
  std::map<std::string, uint64_t> m_pathImsiMap;

  /**
   * List of CellId by path in the attribute system
   */
  std::map<std::string, uint16_t> m_pathCellIdMap;

  /**
   * Name of the file where the downlink results will be saved
   */
  std::string m_dlOutputFilename;

  /**
   * Name of the file where the uplink results will be saved
   */
  std::string m_ulOutputFilename;
};

} // namespace ns3

#endif /* LTE_STATS_CALCULATOR_H_ */
