/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 IITP RAS
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 */

#ifndef THOMPSON_SAMPLING_WIFI_MANAGER_H
#define THOMPSON_SAMPLING_WIFI_MANAGER_H

#include "ns3/random-variable-stream.h"

#include "ns3/wifi-remote-station-manager.h"

namespace ns3 {

/**
 * \brief Thompson Sampling rate control algorithm
 * \ingroup wifi
 *
 * This class implements Thompson Sampling rate control algorithm.
 *
 * It was implemented for use as a baseline in
 * https://doi.org/10.1109/ACCESS.2020.3023552
 */
class ThompsonSamplingWifiManager : public WifiRemoteStationManager
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  ThompsonSamplingWifiManager ();
  virtual ~ThompsonSamplingWifiManager ();

  int64_t AssignStreams (int64_t stream) override;

private:
  WifiRemoteStation *DoCreateStation () const override;
  void DoReportRxOk (WifiRemoteStation *station,
                     double rxSnr, WifiMode txMode) override;
  void DoReportRtsFailed (WifiRemoteStation *station) override;
  void DoReportDataFailed (WifiRemoteStation *station) override;
  void DoReportRtsOk (WifiRemoteStation *station,
                      double ctsSnr, WifiMode ctsMode, double rtsSnr) override;
  void DoReportDataOk (WifiRemoteStation *station,
                       double ackSnr, WifiMode ackMode, double dataSnr,
                       uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportAmpduTxStatus (WifiRemoteStation *station,
                              uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus,
                              double rxSnr, double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportFinalRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalDataFailed (WifiRemoteStation *station) override;
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station) override;
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station) override;

  /**
   * Initializes station rate tables. If station is already initialized,
   * nothing is done.
   *
   * \param station Station which should be initialized.
   */
  void InitializeStation (WifiRemoteStation *station) const;

  /**
   * Draws a new MCS and related parameters to try next time for this
   * station.
   *
   * This method should only be called between TXOPs to avoid sending
   * multiple frames using different modes. Otherwise it is impossible
   * to tell which mode was used for succeeded/failed frame when
   * feedback is received.
   *
   * \param station Station for which a new mode should be drawn.
   */
  void UpdateNextMode (WifiRemoteStation *station) const;

  /**
   * Applies exponential decay to MCS statistics.
   *
   * \param st Remote STA for which MCS statistics is to be updated.
   * \param i MCS index.
   */
  void Decay (WifiRemoteStation *st, size_t i) const;

  /**
   * Returns guard interval in nanoseconds for the given mode.
   *
   * \param st Remote STA.
   * \param mode The WifiMode.
   * \return the guard interval in nanoseconds
   */
  uint16_t GetModeGuardInterval (WifiRemoteStation *st, WifiMode mode) const;

  /**
   * Sample beta random variable with given parameters
   * \param alpha first parameter of beta distribution
   * \param beta second parameter of beta distribution
   * \return beta random variable sample
   */
  double SampleBetaVariable (uint64_t alpha, uint64_t beta) const;

  Ptr<GammaRandomVariable> m_gammaRandomVariable; //!< Variable used to sample beta-distributed random variables

  double m_decay; //!< Exponential decay coefficient, Hz

  TracedValue<uint64_t> m_currentRate; //!< Trace rate changes
};

} //namespace ns3

#endif /* THOMPSON_SAMPLING_WIFI_MANAGER_H */
