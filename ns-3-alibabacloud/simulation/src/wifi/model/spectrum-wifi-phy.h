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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Ghada Badawy <gbadawy@gmail.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 *
 * Ported from yans-wifi-phy.h by several contributors starting
 * with Nicola Baldo and Dean Armstrong
 */

#ifndef SPECTRUM_WIFI_PHY_H
#define SPECTRUM_WIFI_PHY_H

#include "ns3/antenna-model.h"
#include "ns3/spectrum-channel.h"
#include "ns3/spectrum-model.h"
#include "wifi-phy.h"
#include <map>

class SpectrumWifiPhyFilterTest;

namespace ns3 {

class WifiSpectrumPhyInterface;
struct WifiSpectrumSignalParameters;

/**
 * \brief 802.11 PHY layer model
 * \ingroup wifi
 *
 * This PHY implements a spectrum-aware enhancement of the 802.11 SpectrumWifiPhy
 * model.
 *
 * This PHY model depends on a channel loss and delay
 * model as provided by the ns3::SpectrumPropagationLossModel
 * and ns3::PropagationDelayModel classes.
 *
 */
class SpectrumWifiPhy : public WifiPhy
{
public:
  /// allow SpectrumWifiPhyFilterTest class access
  friend class ::SpectrumWifiPhyFilterTest;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  SpectrumWifiPhy ();
  virtual ~SpectrumWifiPhy ();

  // Implementation of pure virtual method.
  void StartTx (Ptr<WifiPpdu> ppdu) override;
  Ptr<Channel> GetChannel (void) const override;
  uint16_t GetGuardBandwidth (uint16_t currentChannelWidth) const override;
  std::tuple<double, double, double> GetTxMaskRejectionParams (void) const override;

  /**
   * Set the SpectrumChannel this SpectrumWifiPhy is to be connected to.
   *
   * \param channel the SpectrumChannel this SpectrumWifiPhy is to be connected to
   */
  void SetChannel (const Ptr<SpectrumChannel> channel);

  /**
   * Input method for delivering a signal from the spectrum channel
   * and low-level PHY interface to this SpectrumWifiPhy instance.
   *
   * \param rxParams Input signal parameters
   */
  void StartRx (Ptr<SpectrumSignalParameters> rxParams);

  /**
   * Method to encapsulate the creation of the WifiSpectrumPhyInterface
   * object (used to bind the WifiSpectrumPhy to a SpectrumChannel) and
   * to link it to this SpectrumWifiPhy instance
   *
   * \param device pointer to the NetDevice object including this new object
   */
  void CreateWifiSpectrumPhyInterface (Ptr<NetDevice> device);
  /**
   * \param antenna an AntennaModel to include in the transmitted
   *                SpectrumSignalParameters (in case any objects downstream of the
   *                SpectrumWifiPhy wish to adjust signal properties based on the
   *                transmitted antenna model.  This antenna is also used when
   *                the underlying WifiSpectrumPhyInterface::GetAntenna() method
   *                is called.
   *
   * Note:  this method may be split into separate SetTx and SetRx
   * methods in the future if the modeling need for this arises
   */
  void SetAntenna (const Ptr<AntennaModel> antenna);
  /**
   * Get the antenna model used for reception
   *
   * \return the AntennaModel used for reception
   */
  Ptr<Object> GetAntenna (void) const;
  /**
   * \return the SpectrumModel that this SpectrumPhy expects to be used
   *         for all SpectrumValues that are passed to StartRx. If 0 is
   *         returned, it means that any model will be accepted.
   */
  Ptr<const SpectrumModel> GetRxSpectrumModel ();

  /**
   * \return the width of each band (Hz)
   */
  uint32_t GetBandBandwidth (void) const;

  /**
   * Callback invoked when the PHY model starts to process a signal
   *
   * \param signalType Whether signal is WiFi (true) or foreign (false)
   * \param senderNodeId Node Id of the sender of the signal
   * \param rxPower received signal power (dBm)
   * \param duration Signal duration
   */
  typedef void (* SignalArrivalCallback) (bool signalType, uint32_t senderNodeId, double rxPower, Time duration);

  // The following method calls the base WifiPhy class method
  // but also generates a new SpectrumModel if called during runtime
  void DoChannelSwitch (void) override;

  /**
   * This function is sending the signal to the Spectrum channel
   * after finishing the configuration of the transmit parameters.
   *
   * \param txParams the parameters to be provided to the Spectrum channel
   */
  void Transmit (Ptr<WifiSpectrumSignalParameters> txParams);

protected:
  void DoDispose (void) override;
  void DoInitialize (void) override;

  /**
   * Get the start band index and the stop band index for a given band
   *
   * \param bandWidth the width of the band to be returned (MHz)
   * \param bandIndex the index of the band to be returned
   *
   * \return a pair of start and stop indexes that defines the band
   */
  WifiSpectrumBand GetBand (uint16_t bandWidth, uint8_t bandIndex = 0) override;

private:
  /**
   * \param bandWidth the width (MHz) of the band used for the OFDMA transmission. Must be
   *                  a multiple of 20 MHz
   * \param guardBandwidth width of the guard band (MHz)
   * \param range the subcarrier range of the HE RU
   * \param bandIndex the index (starting at 0) of the band within the operating channel
   * \return the converted subcarriers
   *
   * This is a helper function to convert HE RU subcarriers, which are relative to the center frequency subcarrier, to the indexes used by the Spectrum model.
   */
  WifiSpectrumBand ConvertHeRuSubcarriers (uint16_t bandWidth, uint16_t guardBandwidth,
                                           HeRu::SubcarrierRange range, uint8_t bandIndex = 0) const override;

  /**
   * Perform run-time spectrum model change
   */
  void ResetSpectrumModel (void);
  /**
   * This function is called to update the bands handled by the InterferenceHelper.
   */
  void UpdateInterferenceHelperBands (void);

  Ptr<SpectrumChannel> m_channel; //!< SpectrumChannel that this SpectrumWifiPhy is connected to

  Ptr<WifiSpectrumPhyInterface> m_wifiSpectrumPhyInterface; //!< Spectrum PHY interface
  Ptr<AntennaModel> m_antenna;                              //!< antenna model
  mutable Ptr<const SpectrumModel> m_rxSpectrumModel;       //!< receive spectrum model

  /// Map a spectrum band associated with an RU to the RU specification
  typedef std::map<WifiSpectrumBand, HeRu::RuSpec> RuBand;

  std::map<uint16_t, RuBand> m_ruBands;  /**< For each channel width, store all the distinct spectrum
                                              bands associated with every RU in a channel of that width */
  bool m_disableWifiReception;                              //!< forces this PHY to fail to sync on any signal
  TracedCallback<bool, uint32_t, double, Time> m_signalCb;  //!< Signal callback

  double m_txMaskInnerBandMinimumRejection; //!< The minimum rejection (in dBr) for the inner band of the transmit spectrum mask
  double m_txMaskOuterBandMinimumRejection; //!< The minimum rejection (in dBr) for the outer band of the transmit spectrum mask
  double m_txMaskOuterBandMaximumRejection; //!< The maximum rejection (in dBr) for the outer band of the transmit spectrum mask
};

} //namespace ns3

#endif /* SPECTRUM_WIFI_PHY_H */
