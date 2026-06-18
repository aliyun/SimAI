/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Author: Federico Maguolo <maguolof@dei.unipd.it>
 */

#ifndef RRAA_WIFI_MANAGER_H
#define RRAA_WIFI_MANAGER_H

#include "ns3/nstime.h"
#include "ns3/traced-value.h"
#include "ns3/wifi-remote-station-manager.h"

namespace ns3 {

struct RraaWifiRemoteStation;

/// WifiRraaThresholds structure
struct WifiRraaThresholds
  {
    double m_ori; ///< Opportunistic Rate Increase threshold
    double m_mtl; ///< Maximum Tolerable Loss threshold
    uint32_t m_ewnd; ///< Evaluation Window
  };

/**
 * List of thresholds for each mode.
 */
typedef std::vector<std::pair<WifiRraaThresholds, WifiMode> > RraaThresholdsTable;

/**
 * \brief Robust Rate Adaptation Algorithm
 * \ingroup wifi
 *
 * This is an implementation of RRAA as described in
 * "Robust rate adaptation for 802.11 wireless networks"
 * by "Starsky H. Y. Wong", "Hao Yang", "Songwu Lu", and,
 * "Vaduvur Bharghavan" published in Mobicom 06.
 *
 * This RAA does not support HT modes and will error
 * exit if the user tries to configure this RAA with a Wi-Fi MAC
 * that supports 802.11n or higher.
 */
class RraaWifiManager : public WifiRemoteStationManager
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  RraaWifiManager ();
  virtual ~RraaWifiManager ();

  void SetupPhy (const Ptr<WifiPhy> phy) override;
  void SetupMac (const Ptr<WifiMac> mac) override;


private:
  void DoInitialize (void) override;
  WifiRemoteStation * DoCreateStation (void) const override;
  void DoReportRxOk (WifiRemoteStation *station,
                     double rxSnr, WifiMode txMode) override;
  void DoReportRtsFailed (WifiRemoteStation *station) override;
  void DoReportDataFailed (WifiRemoteStation *station) override;
  void DoReportRtsOk (WifiRemoteStation *station,
                      double ctsSnr, WifiMode ctsMode, double rtsSnr) override;
  void DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode,
                       double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportFinalRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalDataFailed (WifiRemoteStation *station) override;
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station) override;
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station) override;
  bool DoNeedRts (WifiRemoteStation *st,
                  uint32_t size, bool normally) override;

  /**
   * Check for initializations.
   * \param station The remote station.
   */
  void CheckInit (RraaWifiRemoteStation *station);
  /**
   * Return the index for the maximum transmission rate for
   * the given station.
   *
   * \param station the remote station
   *
   * \return the index for the maximum transmission rate
   */
  uint8_t GetMaxRate (RraaWifiRemoteStation *station) const;
  /**
   * Check if the counter should be reset.
   *
   * \param station the remote station
   */
  void CheckTimeout (RraaWifiRemoteStation *station);
  /**
   * Find an appropriate rate for the given station, using
   * a basic algorithm.
   *
   * \param station the remote station
   */
  void RunBasicAlgorithm (RraaWifiRemoteStation *station);
  /**
   * Activate the use of RTS for the given station if the conditions are met.
   *
   * \param station the remote station
   */
  void ARts (RraaWifiRemoteStation *station);
  /**
   * Reset the counters of the given station.
   *
   * \param station the remote station
   */
  void ResetCountersBasic (RraaWifiRemoteStation *station);
  /**
   * Initialize the thresholds internal list for the given station.
   *
   * \param station the remote station
   */
  void InitThresholds (RraaWifiRemoteStation *station);
  /**
   * Get the thresholds for the given station and mode.
   *
   * \param station the remote station
   * \param mode the WifiMode
   *
   * \return the RRAA thresholds
   */
  WifiRraaThresholds GetThresholds (RraaWifiRemoteStation *station, WifiMode mode) const;
  /**
   * Get the thresholds for the given station and mode index.
   *
   * \param station the remote station
   * \param index the mode index in the supported rates
   *
   * \return the RRAA thresholds
   */
  WifiRraaThresholds GetThresholds (RraaWifiRemoteStation *station, uint8_t index) const;
  /**
   * Get the estimated TxTime of a packet with a given mode.
   *
   * \param mode the WifiMode
   *
   * \return the estimated TX time
   */
  Time GetCalcTxTime (WifiMode mode) const;
  /**
   * Add transmission time for the given mode to an internal list.
   *
   * \param mode the WifiMode
   * \param t transmission time
   */
  void AddCalcTxTime (WifiMode mode, Time t);
  /**
   * typedef for a vector of a pair of Time, WifiMode.
   * Essentially a list for WifiMode and its corresponding transmission time
   * to transmit a reference packet.
   */
  typedef std::vector<std::pair<Time,WifiMode> > TxTime;

  TxTime m_calcTxTime;     //!< To hold all the calculated TxTime for all modes.
  Time m_sifs;             //!< Value of SIFS configured in the device.
  Time m_difs;             //!< Value of DIFS configured in the device.

  uint32_t m_frameLength;  //!< Data frame length used for calculate mode TxTime.
  uint32_t m_ackLength;    //!< Ack frame length used for calculate mode TxTime.

  bool m_basic;    ///< basic
  Time m_timeout;  ///< timeout
  double m_alpha;  //!< Alpha value for RRAA (value for calculating MTL threshold)
  double m_beta;   //!< Beta value for RRAA (value for calculating ORI threshold).
  double m_tau;    //!< Tau value for RRAA (value for calculating EWND size).

  TracedValue<uint64_t> m_currentRate; //!< Trace rate changes
};

} //namespace ns3

#endif /* RRAA_WIFI_MANAGER_H */
