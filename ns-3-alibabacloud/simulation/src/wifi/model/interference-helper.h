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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef INTERFERENCE_HELPER_H
#define INTERFERENCE_HELPER_H

#include "phy-entity.h"

namespace ns3 {

class WifiPpdu;
class WifiPsdu;
class ErrorRateModel;

/**
 * \ingroup wifi
 * \brief handles interference calculations
 * \brief signal event for a PPDU.
 */
class Event : public SimpleRefCount<Event>
{
public:
  /**
   * Create an Event with the given parameters. Note that <i>rxPower</i> will
   * be moved into this object.
   *
   * \param ppdu the PPDU
   * \param txVector the TXVECTOR
   * \param duration duration of the PPDU
   * \param rxPower the received power per band (W)
   */
  Event (Ptr<const WifiPpdu> ppdu, const WifiTxVector& txVector, Time duration,  RxPowerWattPerChannelBand&& rxPower);
  ~Event ();

  /**
   * Return the PPDU.
   *
   * \return the PPDU
   */
  Ptr<const WifiPpdu> GetPpdu (void) const;
  /**
   * Return the start time of the signal.
   *
   * \return the start time of the signal
   */
  Time GetStartTime (void) const;
  /**
   * Return the end time of the signal.
   *
   * \return the end time of the signal
   */
  Time GetEndTime (void) const;
  /**
   * Return the duration of the signal.
   *
   * \return the duration of the signal
   */
  Time GetDuration (void) const;
  /**
   * Return the total received power (W).
   *
   * \return the total received power (W)
   */
  double GetRxPowerW (void) const;
  /**
   * Return the received power (W) for a given band.
   *
   * \param band the band for which the power should be returned
   * \return the received power (W) for a given band
   */
  double GetRxPowerW (WifiSpectrumBand band) const;
  /**
   * Return the received power (W) for all bands.
   *
   * \return the received power (W) for all bands.
   */
  const RxPowerWattPerChannelBand& GetRxPowerWPerBand (void) const;
  /**
   * Return the TXVECTOR of the PPDU.
   *
   * \return the TXVECTOR of the PPDU
   */
  const WifiTxVector& GetTxVector (void) const;
  /**
   * Update the received power (W) for all bands, i.e. add up the received power
   * to the current received power, for each band.
   *
   * \param rxPower the received power (W) for all bands.
   */
  void UpdateRxPowerW (const RxPowerWattPerChannelBand& rxPower);


private:
  Ptr<const WifiPpdu> m_ppdu;           //!< PPDU
  WifiTxVector m_txVector;              //!< TXVECTOR
  Time m_startTime;                     //!< start time
  Time m_endTime;                       //!< end time
  RxPowerWattPerChannelBand m_rxPowerW; //!< received power in watts per band
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param event the event
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const Event &event);


/**
 * \ingroup wifi
 * \brief handles interference calculations
 */
class InterferenceHelper : public Object
{
public:
  InterferenceHelper ();
  virtual ~InterferenceHelper ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Add a frequency band.
   *
   * \param band the band to be created
   */
  void AddBand (WifiSpectrumBand band);

  /**
   * Remove the frequency bands.
   */
  void RemoveBands (void);

  /**
   * Set the noise figure.
   *
   * \param value noise figure in linear scale
   */
  void SetNoiseFigure (double value);
  /**
   * Set the error rate model for this interference helper.
   *
   * \param rate Error rate model
   */
  void SetErrorRateModel (const Ptr<ErrorRateModel> rate);

  /**
   * Return the error rate model.
   *
   * \return Error rate model
   */
  Ptr<ErrorRateModel> GetErrorRateModel (void) const;
  /**
   * Set the number of RX antennas in the receiver corresponding to this
   * interference helper.
   *
   * \param rx the number of RX antennas
   */
  void SetNumberOfReceiveAntennas (uint8_t rx);

  /**
   * \param energyW the minimum energy (W) requested
   * \param band identify the requested band
   *
   * \returns the expected amount of time the observed
   *          energy on the medium for a given band will
   *          be higher than the requested threshold.
   */
  Time GetEnergyDuration (double energyW, WifiSpectrumBand band);

  /**
   * Add the PPDU-related signal to interference helper.
   *
   * \param ppdu the PPDU
   * \param txVector the TXVECTOR
   * \param duration the PPDU duration
   * \param rxPower received power per band (W)
   * \param isStartOfdmaRxing flag whether the event corresponds to the start of the OFDMA payload reception (only used for UL-OFDMA) //TODO simplify this once WifiPpdu is subclassed by adding an attribute
   *
   * \return Event
   */
  Ptr<Event> Add (Ptr<const WifiPpdu> ppdu, const WifiTxVector& txVector, Time duration, RxPowerWattPerChannelBand& rxPower, bool isStartOfdmaRxing = false);

  /**
   * Add a non-Wifi signal to interference helper.
   * \param duration the duration of the signal
   * \param rxPower received power per band (W)
   */
  void AddForeignSignal (Time duration, RxPowerWattPerChannelBand& rxPower);
  /**
   * Calculate the SNIR at the start of the payload and accumulate
   * all SNIR changes in the SNIR vector for each MPDU of an A-MPDU.
   * This workaround is required in order to provide one PER per MPDU, for
   * reception success/failure evaluation, while hiding aggregation details from
   * this class.
   *
   * \param event the event corresponding to the first time the corresponding PPDU arrives
   * \param channelWidth the channel width used to transmit the PSDU (in MHz)
   * \param band identify the band used by the PSDU
   * \param staId the station ID of the PSDU (only used for MU)
   * \param relativeMpduStartStop the time window (pair of start and end times) of PHY payload to focus on
   *
   * \return struct of SNR and PER (with PER being evaluated over the provided time window)
   */
  struct PhyEntity::SnrPer CalculatePayloadSnrPer (Ptr<Event> event, uint16_t channelWidth, WifiSpectrumBand band,
                                                            uint16_t staId, std::pair<Time, Time> relativeMpduStartStop) const;
  /**
   * Calculate the SNIR for the event (starting from now until the event end).
   *
   * \param event the event corresponding to the first time the corresponding PPDU arrives
   * \param channelWidth the channel width (in MHz)
   * \param nss the number of spatial streams
   * \param band identify the band used by the PSDU
   *
   * \return the SNR for the PPDU in linear scale
   */
  double CalculateSnr (Ptr<Event> event, uint16_t channelWidth, uint8_t nss, WifiSpectrumBand band) const;
  /**
   * Calculate the SNIR at the start of the PHY header and accumulate
   * all SNIR changes in the SNIR vector.
   *
   * \param event the event corresponding to the first time the corresponding PPDU arrives
   * \param channelWidth the channel width (in MHz) for header measurement
   * \param band identify the band used by the PSDU
   * \param header the PHY header to consider
   *
   * \return struct of SNR and PER
   */
  struct PhyEntity::SnrPer CalculatePhyHeaderSnrPer (Ptr<Event> event, uint16_t channelWidth, WifiSpectrumBand band,
                                                              WifiPpduField header) const;

  /**
   * Notify that RX has started.
   */
  void NotifyRxStart ();
  /**
   * Notify that RX has ended.
   * 
   * \param endTime the end time of the signal
   */
  void NotifyRxEnd (Time endTime);
  /**
   * Erase all events.
   */
  void EraseEvents (void);

  /**
   * Update event to scale its received power (W) per band.
   *
   * \param event the event to be updated
   * \param rxPower the received power (W) per band to be added to the current event
   */
  void UpdateEvent (Ptr<Event> event, const RxPowerWattPerChannelBand& rxPower);

protected:
  void DoDispose (void) override;

  /**
   * Calculate SNR (linear ratio) from the given signal power and noise+interference power.
   *
   * \param signal signal power, W
   * \param noiseInterference noise and interference power, W
   * \param channelWidth signal width (MHz)
   * \param nss the number of spatial streams
   *
   * \return SNR in linear scale
   */
  double CalculateSnr (double signal, double noiseInterference, uint16_t channelWidth, uint8_t nss) const;
  /**
   * Calculate the success rate of the chunk given the SINR, duration, and TXVECTOR.
   * The duration and TXVECTOR are used to calculate how many bits are present in the chunk.
   *
   * \param snir the SINR
   * \param duration the duration of the chunk
   * \param mode the WifiMode
   * \param txVector the TXVECTOR
   * \param field the PPDU field to which the chunk belongs to
   *
   * \return the success rate
   */
  double CalculateChunkSuccessRate (double snir, Time duration, WifiMode mode, const WifiTxVector& txVector, WifiPpduField field) const;
  /**
   * Calculate the success rate of the payload chunk given the SINR, duration, and TXVECTOR.
   * The duration and TXVECTOR are used to calculate how many bits are present in the payload chunk.
   *
   * \param snir the SINR
   * \param duration the duration of the chunk
   * \param txVector the TXVECTOR
   * \param staId the station ID of the PSDU (only used for MU)
   *
   * \return the success rate
   */
  double CalculatePayloadChunkSuccessRate (double snir, Time duration, const WifiTxVector& txVector, uint16_t staId = SU_STA_ID) const;

private:
  /**
   * Noise and Interference (thus Ni) event.
   */
  class NiChange
  {
  public:
    /**
     * Create a NiChange at the given time and the amount of NI change.
     *
     * \param power the power in watts
     * \param event causes this NI change
     */
    NiChange (double power, Ptr<Event> event);
    ~NiChange ();
    /**
     * Return the power
     *
     * \return the power in watts
     */
    double GetPower (void) const;
    /**
     * Add a given amount of power.
     *
     * \param power the power to be added to the existing value in watts
     */
    void AddPower (double power);
    /**
     * Return the event causes the corresponding NI change
     *
     * \return the event
     */
    Ptr<Event> GetEvent (void) const;

  private:
    double m_power; ///< power in watts
    Ptr<Event> m_event; ///< event
  };

  /**
   * typedef for a multimap of NiChange
   */
  typedef std::multimap<Time, NiChange> NiChanges;

  /**
   * Map of NiChanges per band
   */
  typedef std::map <WifiSpectrumBand, NiChanges> NiChangesPerBand;

  /**
   * Append the given Event.
   *
   * \param event the event to be appended
   * \param isStartOfdmaRxing flag whether event corresponds to the start of the OFDMA payload reception (only used for UL-OFDMA)
   */
  void AppendEvent (Ptr<Event> event, bool isStartOfdmaRxing);

  /**
   * Calculate noise and interference power in W.
   *
   * \param event the event
   * \param nis the NiChanges
   * \param band the band
   *
   * \return noise and interference power
   */
  double CalculateNoiseInterferenceW (Ptr<Event> event, NiChangesPerBand *nis, WifiSpectrumBand band) const;
  /**
   * Calculate the error rate of the given PHY payload only in the provided time
   * window (thus enabling per MPDU PER information). The PHY payload can be divided into
   * multiple chunks (e.g. due to interference from other transmissions).
   *
   * \param event the event
   * \param channelWidth the channel width used to transmit the PSDU (in MHz)
   * \param nis the NiChanges
   * \param band identify the band used by the PSDU
   * \param staId the station ID of the PSDU (only used for MU)
   * \param window time window (pair of start and end times) of PHY payload to focus on
   *
   * \return the error rate of the payload
   */
  double CalculatePayloadPer (Ptr<const Event> event, uint16_t channelWidth, NiChangesPerBand *nis, WifiSpectrumBand band,
                              uint16_t staId, std::pair<Time, Time> window) const;
  /**
   * Calculate the error rate of the PHY header. The PHY header
   * can be divided into multiple chunks (e.g. due to interference from other transmissions).
   *
   * \param event the event
   * \param nis the NiChanges
   * \param channelWidth the channel width (in MHz) for header measurement
   * \param band the band
   * \param header the PHY header to consider
   *
   * \return the error rate of the HT PHY header
   */
  double CalculatePhyHeaderPer (Ptr<const Event> event, NiChangesPerBand *nis,
                                uint16_t channelWidth, WifiSpectrumBand band,
                                WifiPpduField header) const;
  /**
   * Calculate the success rate of the PHY header sections for the provided event.
   *
   * \param event the event
   * \param nis the NiChanges
   * \param channelWidth the channel width (in MHz) for header measurement
   * \param band the band
   * \param phyHeaderSections the map of PHY header sections (\see PhyEntity::PhyHeaderSections)
   *
   * \return the success rate of the PHY header sections
   */
  double CalculatePhyHeaderSectionPsr (Ptr<const Event> event, NiChangesPerBand *nis,
                                       uint16_t channelWidth, WifiSpectrumBand band,
                                       PhyEntity::PhyHeaderSections phyHeaderSections) const;

  double m_noiseFigure;                                    //!< noise figure (linear)
  Ptr<ErrorRateModel> m_errorRateModel;                    //!< error rate model
  uint8_t m_numRxAntennas;                                 //!< the number of RX antennas in the corresponding receiver
  NiChangesPerBand m_niChangesPerBand;                     //!< NI Changes for each band
  std::map <WifiSpectrumBand, double> m_firstPowerPerBand; //!< first power of each band in watts
  bool m_rxing;                                            //!< flag whether it is in receiving state

  /**
   * Returns an iterator to the first NiChange that is later than moment
   *
   * \param moment time to check from
   * \param niIt iterator of the band to check
   * \returns an iterator to the list of NiChanges
   */
  NiChanges::iterator GetNextPosition (Time moment, NiChangesPerBand::iterator niIt);
  /**
   * Returns an iterator to the last NiChange that is before than moment
   *
   * \param moment time to check from
   * \param niIt iterator of the band to check
   * \returns an iterator to the list of NiChanges
   */
  NiChanges::iterator GetPreviousPosition (Time moment, NiChangesPerBand::iterator niIt);

  /**
   * Add NiChange to the list at the appropriate position and
   * return the iterator of the new event.
   *
   * \param moment time to check from
   * \param change the NiChange to add
   * \param niIt iterator of the band to check
   * \returns the iterator of the new event
   */
  NiChanges::iterator AddNiChangeEvent (Time moment, NiChange change, NiChangesPerBand::iterator niIt);
};

} //namespace ns3

#endif /* INTERFERENCE_HELPER_H */
