/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universidad de la República - Uruguay
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
 * Author: Matias Richart <mrichart@fing.edu.uy>
 */

#ifndef APARF_WIFI_MANAGER_H
#define APARF_WIFI_MANAGER_H

#include "ns3/wifi-remote-station-manager.h"

namespace ns3 {

struct AparfWifiRemoteStation;

/**
 * \ingroup wifi
 * APARF Power and rate control algorithm
 *
 * This class implements the High Performance power and rate control algorithm
 * described in <i>Dynamic data rate and transmit power adjustment
 * in IEEE 802.11 wireless LANs</i> by Chevillat, P.; Jelitto, J.
 * and Truong, H. L. in International Journal of Wireless Information
 * Networks, Springer, 2005, 12, 123-145.
 * http://www.cs.mun.ca/~yzchen/papers/papers/rate_adaptation/80211_dynamic_rate_power_adjustment_chevillat_j2005.pdf
 *
 * This RAA does not support HT modes and will error
 * exit if the user tries to configure this RAA with a Wi-Fi MAC
 * that supports 802.11n or higher.
 */
class AparfWifiManager : public WifiRemoteStationManager
{
public:
  /**
   * Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  AparfWifiManager ();
  virtual ~AparfWifiManager ();

  void SetupPhy (const Ptr<WifiPhy> phy) override;

  /**
   * Enumeration of the possible states of the channel.
   */
  enum State
  {
    High,
    Low,
    Spread
  };


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

  /** Check for initializations.
   *
   * \param station The remote station.
   */
  void CheckInit (AparfWifiRemoteStation *station);

  uint32_t m_succesMax1; //!< The minimum number of successful transmissions in \"High\" state to try a new power or rate.
  uint32_t m_succesMax2; //!< The minimum number of successful transmissions in \"Low\" state to try a new power or rate.
  uint32_t m_failMax;    //!< The minimum number of failed transmissions to try a new power or rate.
  uint32_t m_powerMax;   //!< The maximum number of power changes.
  uint8_t m_powerInc;    //!< Step size for increment the power.
  uint8_t m_powerDec;    //!< Step size for decrement the power.
  uint8_t m_rateInc;     //!< Step size for increment the rate.
  uint8_t m_rateDec;     //!< Step size for decrement the rate.

  /**
   * Minimal power level.
   * Differently form rate, power levels do not depend on the remote station.
   * The levels depend only on the physical layer of the device.
   */
  uint8_t m_minPower;

  /**
   * Maximal power level.
   */
  uint8_t m_maxPower;

  /**
   * The trace source fired when the transmission power changes.
   */
  TracedCallback<double, double, Mac48Address> m_powerChange;
  /**
   * The trace source fired when the transmission rate changes.
   */
  TracedCallback<DataRate, DataRate, Mac48Address> m_rateChange;
};

} //namespace ns3

#endif /* APARF_WIFI_MANAGER_H */
