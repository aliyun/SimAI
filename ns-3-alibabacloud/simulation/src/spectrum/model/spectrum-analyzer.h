/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef SPECTRUM_ANALYZER_H
#define SPECTRUM_ANALYZER_H


#include <ns3/spectrum-value.h>
#include <ns3/mobility-model.h>
#include <ns3/packet.h>
#include <ns3/nstime.h>
#include <ns3/net-device.h>
#include <ns3/spectrum-phy.h>
#include <ns3/spectrum-channel.h>
#include <string>
#include <fstream>

namespace ns3 {


/**
 * \ingroup spectrum
 *
 * Simple SpectrumPhy implementation that averages the spectrum power
 * density of incoming transmissions to produce a spectrogram.
 *
 *
 * This PHY model supports a single antenna model instance which is
 * used for reception (this PHY model never transmits). 
 */
class SpectrumAnalyzer : public SpectrumPhy
{

public:
  SpectrumAnalyzer ();
  virtual ~SpectrumAnalyzer ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited from SpectrumPhy
  void SetChannel (Ptr<SpectrumChannel> c);
  void SetMobility (Ptr<MobilityModel> m);
  void SetDevice (Ptr<NetDevice> d);
  Ptr<MobilityModel> GetMobility () const;
  Ptr<NetDevice> GetDevice () const;
  Ptr<const SpectrumModel> GetRxSpectrumModel () const;
  Ptr<Object> GetAntenna () const;
  void StartRx (Ptr<SpectrumSignalParameters> params);


  /**
   * Set the spectrum model used by the SpectrumAnalyzer to represent incoming signals
   *
   * @param m the Rx Spectrum model
   */
  void SetRxSpectrumModel (Ptr<SpectrumModel> m);

  /** 
   * Set the AntennaModel to be used
   * 
   * \param a the Antenna Model
   */
  void SetAntenna (Ptr<AntennaModel> a);

  /**
   * Start the spectrum analyzer
   *
   */
  virtual void Start ();

  /**
   * Stop the spectrum analyzer
   *
   */
  virtual void Stop ();


protected:
  void DoDispose ();

private:
  Ptr<MobilityModel> m_mobility;    //!< Pointer to the mobility model
  Ptr<AntennaModel> m_antenna;      //!< Pointer to the Antenna model
  Ptr<NetDevice> m_netDevice;       //!< Pointer to the NetDevice using this object
  Ptr<SpectrumChannel> m_channel;   //!< Pointer to the channel to be analyzed

  /**
   * Generates a report of the data collected so far.
   *
   * This function is called periodically.
   */
  virtual void GenerateReport ();

  /**
   * Adds a signal to the data collected.
   * \param psd signal to add
   */
  void AddSignal (Ptr<const SpectrumValue> psd);
  /**
   * Removes a signal to the data collected.
   * \param psd signal to subtract
   */
  void SubtractSignal  (Ptr<const SpectrumValue> psd);
  /**
   * Updates the data about the received Energy
   */
  void UpdateEnergyReceivedSoFar ();

  Ptr<SpectrumModel> m_spectrumModel;             //!< Spectrum model
  Ptr<SpectrumValue> m_sumPowerSpectralDensity;   //!< Sum of the received PSD
  Ptr<SpectrumValue> m_energySpectralDensity;     //!< Energy spectral density
  double m_noisePowerSpectralDensity;             //!< Noise power spectral density
  Time m_resolution;                              //!< Time resolution
  Time m_lastChangeTime;                          //!< When the last update happened
  bool m_active;                                  //!< True if the analyzer is active

  /// TracedCallback - average power spectral density report.
  TracedCallback<Ptr<const SpectrumValue> > m_averagePowerSpectralDensityReportTrace;

};






}





#endif /* SPECTRUM_ANALYZER_H */
