/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef IDEAL_WIFI_MANAGER_H
#define IDEAL_WIFI_MANAGER_H

#include "ns3/traced-value.h"
#include "ns3/wifi-remote-station-manager.h"

namespace ns3 {

struct IdealWifiRemoteStation;

/**
 * \brief Ideal rate control algorithm
 * \ingroup wifi
 *
 * This class implements an 'ideal' rate control algorithm
 * similar to RBAR in spirit (see <i>A rate-adaptive MAC
 * protocol for multihop wireless networks</i> by G. Holland,
 * N. Vaidya, and P. Bahl.): every station keeps track of the
 * SNR of every packet received and sends back this SNR to the
 * original transmitter by an out-of-band mechanism. Each
 * transmitter keeps track of the last SNR sent back by a receiver
 * and uses it to pick a transmission mode based on a set
 * of SNR thresholds built from a target BER and transmission
 * mode-specific SNR/BER curves.
 */
class IdealWifiManager : public WifiRemoteStationManager
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  IdealWifiManager ();
  virtual ~IdealWifiManager ();

  void SetupPhy (const Ptr<WifiPhy> phy) override;


private:
  void DoInitialize (void) override;
  WifiRemoteStation* DoCreateStation (void) const override;
  void DoReportRxOk (WifiRemoteStation *station,
                     double rxSnr, WifiMode txMode) override;
  void DoReportRtsFailed (WifiRemoteStation *station) override;
  void DoReportDataFailed (WifiRemoteStation *station) override;
  void DoReportRtsOk (WifiRemoteStation *station,
                      double ctsSnr, WifiMode ctsMode, double rtsSnr) override;
  void DoReportDataOk (WifiRemoteStation *station, double ackSnr, WifiMode ackMode,
                       double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportAmpduTxStatus (WifiRemoteStation *station, uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus,
                              double rxSnr, double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss) override;
  void DoReportFinalRtsFailed (WifiRemoteStation *station) override;
  void DoReportFinalDataFailed (WifiRemoteStation *station) override;
  WifiTxVector DoGetDataTxVector (WifiRemoteStation *station) override;
  WifiTxVector DoGetRtsTxVector (WifiRemoteStation *station) override;

  /**
   * Reset the station, invoked if the maximum amount of retries has failed.
   *
   * \param station the station for which statistics should be reset
   */
  void Reset (WifiRemoteStation *station) const;

  /**
   * Construct the vector of minimum SNRs needed to successfully transmit for
   * all possible combinations (rate, channel width, nss) based on PHY capabilities.
   * This is called at initialization and if PHY capabilities changed.
   */
  void BuildSnrThresholds (void);

  /**
   * Return the minimum SNR needed to successfully transmit
   * data with this WifiTxVector at the specified BER.
   *
   * \param txVector WifiTxVector (containing valid mode, width, and Nss)
   *
   * \return the minimum SNR for the given WifiTxVector in linear scale
   */
  double GetSnrThreshold (WifiTxVector txVector);
  /**
   * Adds a pair of WifiTxVector and the minimum SNR for that given vector
   * to the list.
   *
   * \param txVector the WifiTxVector storing mode, channel width, and Nss
   * \param snr the minimum SNR for the given txVector in linear scale
   */
  void AddSnrThreshold (WifiTxVector txVector, double snr);

  /**
   * Convenience function for selecting a channel width for non-HT mode
   * \param mode non-HT WifiMode
   * \return the channel width (MHz) for the selected mode
   */
  uint16_t GetChannelWidthForNonHtMode (WifiMode mode) const;

  /**
   * Convenience function to get the last observed SNR from a given station for a given channel width and a given NSS.
   * Since the previously received SNR information might be related to a different channel width than the requested one,
   * and/or a different NSS,  the function does some computations to get the corresponding SNR.
   *
   * \param station the station being queried
   * \param channelWidth the channel width (in MHz)
   * \param nss the number of spatial streams
   * \return the SNR in linear scale
   */
  double GetLastObservedSnr (IdealWifiRemoteStation *station, uint16_t channelWidth, uint8_t nss) const;

  /**
   * A vector of <snr, WifiTxVector> pair holding the minimum SNR for the
   * WifiTxVector
   */
  typedef std::vector<std::pair<double, WifiTxVector> > Thresholds;

  double m_ber;             //!< The maximum Bit Error Rate acceptable at any transmission mode
  Thresholds m_thresholds;  //!< List of WifiTxVector and the minimum SNR pair

  TracedValue<uint64_t> m_currentRate; //!< Trace rate changes
};

} //namespace ns3

#endif /* IDEAL_WIFI_MANAGER_H */
