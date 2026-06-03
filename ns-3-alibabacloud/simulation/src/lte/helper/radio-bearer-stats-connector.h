/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012-2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Authors: Nicola Baldo <nbaldo@cttc.es>
 *          Manuel Requena <manuel.requena@cttc.es>
 */

#ifndef RADIO_BEARER_STATS_CONNECTOR_H
#define RADIO_BEARER_STATS_CONNECTOR_H

#include "ns3/ptr.h"
#include "ns3/simple-ref-count.h"
#include "ns3/config.h"
#include "ns3/traced-callback.h"

#include <map>

namespace ns3 {

class RadioBearerStatsCalculator;

/**
 * \ingroup lte
 *
 * This class is very useful when user needs to collect
 * statistics from PDCP and RLC. It automatically connects
 * RadioBearerStatsCalculator to appropriate trace sinks.
 * Usually user does not use this class. All he/she needs
 * to do is to call: LteHelper::EnablePdcpTraces() and/or
 * LteHelper::EnableRlcTraces().
 */

class RadioBearerStatsConnector
{
public:
  /// Constructor
  RadioBearerStatsConnector ();

  /**
   * Enables trace sinks for RLC layer. Usually, this function
   * is called by LteHelper::EnableRlcTraces().
   * \param rlcStats statistics calculator for RLC layer
   */
  void EnableRlcStats (Ptr<RadioBearerStatsCalculator> rlcStats);

  /**
   * Enables trace sinks for PDCP layer. Usually, this function
   * is called by LteHelper::EnablePdcpTraces().
   * \param pdcpStats statistics calculator for PDCP layer
   */
  void EnablePdcpStats (Ptr<RadioBearerStatsCalculator> pdcpStats);

  /**
   * Connects trace sinks to appropriate trace sources
   */
  void EnsureConnected ();

  // trace sinks, to be used with MakeBoundCallback

  /**
   * Function hooked to NewUeContext trace source at eNB RRC,
   * which is fired upon creation of a new UE context.
   * It stores the UE manager path and connects the callback that will be called
   * when the DRB is created in the eNB.
   * \param c
   * \param context
   * \param cellid
   * \param rnti
   */
  static void NotifyNewUeContextEnb (RadioBearerStatsConnector* c, std::string context, uint16_t cellid, uint16_t rnti);

  /**
   * Function hooked to RandomAccessSuccessful trace source at UE RRC,
   * which is fired upon successful completion of the random access procedure.
   * It connects the callbacks for the SRB0 at the eNB and the UE.
   * \param c
   * \param context
   * \param imsi
   * \param cellid
   * \param rnti
   */
  static void NotifyRandomAccessSuccessfulUe (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti);

  /**
   * Function hooked to Srb1Created trace source at UE RRC,
   * which is fired when SRB1 is created, i.e. RLC and PDCP are created for one LC = 1.
   * It connects the callbacks for the DRB at the eNB.
   * \param c
   * \param context
   * \param imsi
   * \param cellid
   * \param rnti
   */
  static void CreatedSrb1Ue (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti);

  /**
   * Function hooked to DrbCreated trace source at UE manager in eNB RRC,
   * which is fired when DRB is created, i.e. RLC and PDCP are created for LC = lcid.
   * It connects the callbacks for the DRB at the eNB.
   * \param c
   * \param context
   * \param imsi
   * \param cellid
   * \param rnti
   * \param lcid
   */
  static void CreatedDrbEnb (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti, uint8_t lcid);

  /**
   * Function hooked to DrbCreated trace source at UE RRC,
   * which is fired when DRB is created, i.e. RLC and PDCP are created for LC = lcid.
   * It connects the callbacks for the DRB at the UE.
   * \param c
   * \param context
   * \param imsi
   * \param cellid
   * \param rnti
   * \param lcid
   */
  static void CreatedDrbUe (RadioBearerStatsConnector* c, std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti, uint8_t lcid);

  /**
   * Disconnects all trace sources at eNB to RLC and PDCP calculators.
   * Function is not implemented.
   * \param context
   * \param imsi
   * \param cellid
   * \param rnti
   */
  void DisconnectTracesEnb (std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti);

  /**
   * Disconnects all trace sources at UE to RLC and PDCP calculators.
   * Function is not implemented.
   * \param context
   * \param imsi
   * \param cellid
   * \param rnti
   */
  void DisconnectTracesUe (std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti);


private:
  /**
   * Creates UE Manager path and stores it in m_ueManagerPathByCellIdRnti
   * \param ueManagerPath
   * \param cellId
   * \param rnti
   */
  void StoreUeManagerPath (std::string ueManagerPath, uint16_t cellId, uint16_t rnti);

  /**
   * Connects SRB0 trace sources at UE and eNB to RLC and PDCP calculators
   * \param context
   * \param imsi
   * \param cellId
   * \param rnti
   */
  void ConnectTracesSrb0 (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti);

  /**
   * Connects SRB1 trace sources at UE and eNB to RLC and PDCP calculators
   * \param context
   * \param imsi
   * \param cellId
   * \param rnti
   */
  void ConnectTracesSrb1 (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti);

  /**
   * Connects DRB trace sources at eNB to RLC and PDCP calculators
   * \param context
   * \param imsi
   * \param cellId
   * \param rnti
   * \param lcid
   */
  void ConnectTracesDrbEnb (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid);

  /**
   * Connects DRB trace sources at UE to RLC and PDCP calculators
   * \param context
   * \param imsi
   * \param cellId
   * \param rnti
   * \param lcid
   */
  void ConnectTracesDrbUe (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t lcid);

  Ptr<RadioBearerStatsCalculator> m_rlcStats; //!< Calculator for RLC Statistics
  Ptr<RadioBearerStatsCalculator> m_pdcpStats; //!< Calculator for PDCP Statistics

  bool m_connected; //!< true if traces are connected to sinks, initially set to false

  /**
   * Struct used as key in m_ueManagerPathByCellIdRnti map
   */
  struct CellIdRnti
  {
    uint16_t cellId; //!< Cell Id
    uint16_t rnti; //!< RNTI
  };

  /**
   * Less than operator for CellIdRnti, because it is used as key in map
   *
   * \param a the lhs operand
   * \param b the rhs operand
   * \returns true if less than
   */
  friend bool operator < (const CellIdRnti &a, const CellIdRnti &b);

  /**
   * List UE Manager Paths by CellIdRnti
   */
  std::map<CellIdRnti, std::string> m_ueManagerPathByCellIdRnti;

};

} // namespace ns3

#endif // RADIO_BEARER_STATS_CONNECTOR_H
