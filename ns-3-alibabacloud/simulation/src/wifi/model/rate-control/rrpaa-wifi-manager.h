/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universidad de la República - Uruguay
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
 * Author: Matías Richart <mrichart@fing.edu.uy>
 */

#ifndef RRPAA_WIFI_MANAGER_H
#define RRPAA_WIFI_MANAGER_H

#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"
#include "ns3/wifi-remote-station-manager.h"

namespace ns3 {

struct RrpaaWifiRemoteStation;

/**
 * \ingroup wifi
 * Robust Rate and Power Adaptation Algorithm
 *
 * This class implements the RRPAA algorithm as described in <i>Rate, Power and Carrier-Sense
 * Threshold Coordinated Management for High-Density IEEE 802.11 Networks</i>
 * by Matías Richart; Jorge Visca and Javier Baliosian in Integrated Network Management (IM),
 * 2015 IFIP/IEEE International Symposium on (pp. 139-146). IEEE.
 * http://dl.ifip.org/db/conf/im/im2015/137430.pdf
 *
 * RRPAA adds power control to the RRAA mechanism. RRAA is described in
 * <i>Robust rate adaptation for 802.11 wireless networks</i> by Starsky H. Y. Wong;
 * Hao Yang; Songwu Lu and Vaduvur Bharghavan in Proceedings of the 12th annual
 * international conference on Mobile computing and networking (pp. 146-157). ACM.
 * http://ocw.cs.pub.ro/courses/_media/isrm/articole/rrate_adapt_mobicom06.pdf
 *
 * This RAA does not support HT modes and will error
 * exit if the user tries to configure this RAA with a Wi-Fi MAC
 * that supports 802.11n or higher.
 */

/**
 * For each rate there is a Opportunistic Rate Increase threshold,
 * a Maximum Tolerable Loss threshold and an Evaluation Window.
 */
struct WifiRrpaaThresholds
{
  double m_ori; //!< The Opportunistic Rate Increase threshold.
  double m_mtl; //!< The Maximum Tolerable Loss threshold.
  uint32_t m_ewnd; //!< The Estimation Window size.
};

/**
 * List of thresholds for each mode.
 */
typedef std::vector<std::pair<WifiRrpaaThresholds,WifiMode> > RrpaaThresholdsTable;

/**
 * List of probabilities.
 */
typedef std::vector<std::vector<double> > RrpaaProbabilitiesTable;

class RrpaaWifiManager : public WifiRemoteStationManager
{
public:
  /**
   * Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  RrpaaWifiManager ();
  virtual ~RrpaaWifiManager ();

  void SetupPhy (const Ptr<WifiPhy> phy) override;
  void SetupMac (const Ptr<WifiMac> mac) override;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   *
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream) override;

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
   * \param station the remote station.
   */
  void CheckInit (RrpaaWifiRemoteStation *station);

  /**
   * Check if the counter should be reset.
   *
   * \param station the remote station
   */
  void CheckTimeout (RrpaaWifiRemoteStation *station);
  /**
   * Find an appropriate rate and power for the given station, using
   * a basic algorithm.
   *
   * \param station the remote station
   */
  void RunBasicAlgorithm (RrpaaWifiRemoteStation *station);
  /**
   * Run an enhanced algorithm which activates the use of RTS
   * for the given station if the conditions are met.
   *
   * \param station the remote station
   */
  void RunAdaptiveRtsAlgorithm (RrpaaWifiRemoteStation *station);
  /**
   * Reset the counters of the given station.
   *
   * \param station the remote station
   */
  void ResetCountersBasic (RrpaaWifiRemoteStation *station);

  /**
   * Initialize the thresholds internal list for the given station.
   *
   * \param station the remote station
   */
  void InitThresholds (RrpaaWifiRemoteStation *station);

  /**
   * Get the thresholds for the given station and mode.
   *
   * \param station the remote station
   * \param mode the WifiMode
   *
   * \return the RRPAA thresholds
   */
  WifiRrpaaThresholds GetThresholds (RrpaaWifiRemoteStation *station, WifiMode mode) const;

  /**
   * Get the thresholds for the given station and mode index.
   *
   * \param station the remote station
   * \param index the mode index in the supported rates
   *
   * \return the RRPAA thresholds
   */
  WifiRrpaaThresholds GetThresholds (RrpaaWifiRemoteStation *station, uint8_t index) const;

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

  uint32_t m_frameLength;  //!< Data frame length used for calculate mode TxTime (in bytes).
  uint32_t m_ackLength;    //!< Ack frame length used for calculate mode TxTime (in bytes).

  bool m_basic;            //!< If using the basic algorithm (without RTS/CTS).
  Time m_timeout;          //!< Timeout for the RRAA BASIC loss estimation block.
  double m_alpha;          //!< Alpha value for RRPAA (value for calculating MTL threshold)
  double m_beta;           //!< Beta value for RRPAA (value for calculating ORI threshold).
  double m_tau;            //!< Tau value for RRPAA (value for calculating EWND size).
  double m_gamma;          //!< Gamma value for RRPAA (value for pdTable decrements).
  double m_delta;          //!< Delta value for RRPAA (value for pdTable increments).

  /**
   * Differently form rate, power levels do not depend on the remote station.
   * The levels depend only on the physical layer of the device.
   */
  uint8_t m_minPowerLevel;     //!< Minimal power level.
  uint8_t m_maxPowerLevel;     //!< Maximal power level.
  uint8_t m_nPowerLevels;      //!< Number of power levels.

  /**
   * The trace source fired when the transmission power change
   */
  TracedCallback<double, double, Mac48Address> m_powerChange;
  /**
   * The trace source fired when the transmission rate change
   */
  TracedCallback<DataRate, DataRate, Mac48Address> m_rateChange;

  Ptr<UniformRandomVariable> m_uniformRandomVariable; //!< Provides uniform random variables for probabilistic changes.
};

} //namespace ns3

#endif /* RRPAA__WIFI_MANAGER_H */
