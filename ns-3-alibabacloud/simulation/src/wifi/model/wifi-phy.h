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
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef WIFI_PHY_H
#define WIFI_PHY_H

#include "ns3/error-model.h"
#include "wifi-standards.h"
#include "wifi-phy-state-helper.h"
#include "phy-entity.h"
#include "wifi-phy-operating-channel.h"

namespace ns3 {

class Channel;
class WifiNetDevice;
class MobilityModel;
class WifiPhyStateHelper;
class FrameCaptureModel;
class PreambleDetectionModel;
class WifiRadioEnergyModel;
class UniformRandomVariable;
class InterferenceHelper;
class ErrorRateModel;

/**
 * \brief 802.11 PHY layer model
 * \ingroup wifi
 *
 */
class WifiPhy : public Object
{
public:
  friend class PhyEntity;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  WifiPhy ();
  virtual ~WifiPhy ();

  /**
   * Return the WifiPhyStateHelper of this PHY
   *
   * \return the WifiPhyStateHelper of this PHY
   */
  Ptr<WifiPhyStateHelper> GetState (void) const;

  /**
   * \param callback the callback to invoke
   *        upon successful packet reception.
   */
  void SetReceiveOkCallback (RxOkCallback callback);
  /**
   * \param callback the callback to invoke
   *        upon erroneous packet reception.
   */
  void SetReceiveErrorCallback (RxErrorCallback callback);

  /**
   * \param listener the new listener
   *
   * Add the input listener to the list of objects to be notified of
   * PHY-level events.
   */
  void RegisterListener (WifiPhyListener *listener);
  /**
   * \param listener the listener to be unregistered
   *
   * Remove the input listener from the list of objects to be notified of
   * PHY-level events.
   */
  void UnregisterListener (WifiPhyListener *listener);

  /**
   * \param callback the callback to invoke when PHY capabilities have changed.
   */
  void SetCapabilitiesChangedCallback (Callback<void> callback);

  /**
   * Start receiving the PHY preamble of a PPDU (i.e. the first bit of the preamble has arrived).
   *
   * \param ppdu the arriving PPDU
   * \param rxPowersW the receive power in W per band
   * \param rxDuration the duration of the PPDU
   */
  void StartReceivePreamble (Ptr<WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW, Time rxDuration);

  /**
   * Reset PHY at the end of the packet under reception after it has failed the PHY header.
   *
   * \param event the corresponding event of the first time the packet arrives (also storing packet and TxVector information)
   */
  void ResetReceive (Ptr<Event> event);

  /**
   * For HE receptions only, check and possibly modify the transmit power restriction state at
   * the end of PPDU reception.
   */
  void EndReceiveInterBss (void);

  /**
   * Get a WifiConstPsduMap from a PSDU and the TXVECTOR to use to send the PSDU.
   * The STA-ID value is properly determined based on whether the given PSDU has
   * to be transmitted as a DL or UL frame.
   *
   * \param psdu the given PSDU
   * \param txVector the TXVECTOR to use to send the PSDU
   * \return a WifiConstPsduMap built from the given PSDU and the given TXVECTOR
   */
  static WifiConstPsduMap GetWifiConstPsduMap (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector);

  /**
   * This function is a wrapper for the Send variant that accepts a WifiConstPsduMap
   * as first argument. This function inserts the given PSDU in a WifiConstPsduMap
   * along with a STA-ID value that is determined based on whether the given PSDU has
   * to be transmitted as a DL or UL frame.
   *
   * \param psdu the PSDU to send (in a SU PPDU)
   * \param txVector the TXVECTOR that has TX parameters such as mode, the transmission mode to use to send
   *        this PSDU, and txPowerLevel, a power level to use to send the whole PPDU. The real transmission
   *        power is calculated as txPowerMin + txPowerLevel * (txPowerMax - txPowerMin) / nTxLevels
   */
  void Send (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector);
  /**
   * \param psdus the PSDUs to send
   * \param txVector the TXVECTOR that has tx parameters such as mode, the transmission mode to use to send
   *        this PSDU, and txPowerLevel, a power level to use to send the whole PPDU. The real transmission
   *        power is calculated as txPowerMin + txPowerLevel * (txPowerMax - txPowerMin) / nTxLevels
   */
  void Send (WifiConstPsduMap psdus, const WifiTxVector& txVector);

  /**
   * \param ppdu the PPDU to send
   */
  virtual void StartTx (Ptr<WifiPpdu> ppdu) = 0;

  /**
   * Put in sleep mode.
   */
  void SetSleepMode (void);
  /**
   * Resume from sleep mode.
   */
  void ResumeFromSleep (void);
  /**
   * Put in off mode.
   */
  void SetOffMode (void);
  /**
   * Resume from off mode.
   */
  void ResumeFromOff (void);

  /**
   * \return true of the current state of the PHY layer is WifiPhy::IDLE, false otherwise.
   */
  bool IsStateIdle (void) const;
  /**
   * \return true of the current state of the PHY layer is WifiPhy::CCA_BUSY, false otherwise.
   */
  bool IsStateCcaBusy (void) const;
  /**
   * \return true of the current state of the PHY layer is WifiPhy::RX, false otherwise.
   */
  bool IsStateRx (void) const;
  /**
   * \return true of the current state of the PHY layer is WifiPhy::TX, false otherwise.
   */
  bool IsStateTx (void) const;
  /**
   * \return true of the current state of the PHY layer is WifiPhy::SWITCHING, false otherwise.
   */
  bool IsStateSwitching (void) const;
  /**
   * \return true if the current state of the PHY layer is WifiPhy::SLEEP, false otherwise.
   */
  bool IsStateSleep (void) const;
  /**
   * \return true if the current state of the PHY layer is WifiPhy::OFF, false otherwise.
   */
  bool IsStateOff (void) const;

  /**
   * \return the predicted delay until this PHY can become WifiPhy::IDLE.
   *
   * The PHY will never become WifiPhy::IDLE _before_ the delay returned by
   * this method but it could become really idle later.
   */
  Time GetDelayUntilIdle (void);

  /**
   * Return the start time of the last received packet.
   *
   * \return the start time of the last received packet
   */
  Time GetLastRxStartTime (void) const;
  /**
   * Return the end time of the last received packet.
   *
   * \return the end time of the last received packet
   */
  Time GetLastRxEndTime (void) const;

  /**
   * \param size the number of bytes in the packet to send
   * \param txVector the TXVECTOR used for the transmission of this packet
   * \param band the frequency band being used
   * \param staId the STA-ID of the recipient (only used for MU)
   *
   * \return the total amount of time this PHY will stay busy for the transmission of these bytes.
   */
  static Time CalculateTxDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band,
                                   uint16_t staId = SU_STA_ID);
  /**
   * This function is a wrapper for the CalculateTxDuration variant that accepts a
   * WifiConstPsduMap as first argument. This function inserts the given PSDU in a
   * WifiConstPsduMap along with a STA-ID value that is determined based on whether
   * the given PSDU has to be transmitted as a DL or UL frame, thus allowing to
   * properly calculate the TX duration in case the PSDU has to be transmitted as
   * an UL frame.
   *
   * \param psdu the PSDU to transmit
   * \param txVector the TXVECTOR used for the transmission of the PSDU
   * \param band the frequency band
   *
   * \return the total amount of time this PHY will stay busy for the transmission of the PPDU
   */
  static Time CalculateTxDuration (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, WifiPhyBand band);
  /**
   * \param psduMap the PSDU(s) to transmit indexed by STA-ID
   * \param txVector the TXVECTOR used for the transmission of the PPDU
   * \param band the frequency band being used
   *
   * \return the total amount of time this PHY will stay busy for the transmission of the PPDU
   */
  static Time CalculateTxDuration (WifiConstPsduMap psduMap, const WifiTxVector& txVector, WifiPhyBand band);

  /**
   * \param txVector the transmission parameters used for this packet
   *
   * \return the total amount of time this PHY will stay busy for the transmission of the PHY preamble and PHY header.
   */
  static Time CalculatePhyPreambleAndHeaderDuration (const WifiTxVector& txVector);
  /**
   * \return the preamble detection duration, which is the time correlation needs to detect the start of an incoming frame.
   */
  static Time GetPreambleDetectionDuration (void);
  /**
   * \param size the number of bytes in the packet to send
   * \param txVector the TXVECTOR used for the transmission of this packet
   * \param band the frequency band
   * \param mpdutype the type of the MPDU as defined in WifiPhy::MpduType.
   * \param staId the STA-ID of the PSDU (only used for MU PPDUs)
   *
   * \return the duration of the PSDU
   */
  static Time GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype = NORMAL_MPDU,
                                  uint16_t staId = SU_STA_ID);
  /**
   * \param size the number of bytes in the packet to send
   * \param txVector the TXVECTOR used for the transmission of this packet
   * \param band the frequency band
   * \param mpdutype the type of the MPDU as defined in WifiPhy::MpduType.
   * \param incFlag this flag is used to indicate that the variables need to be update or not
   * This function is called a couple of times for the same packet so variables should not be increased each time.
   * \param totalAmpduSize the total size of the previously transmitted MPDUs for the concerned A-MPDU.
   * If incFlag is set, this parameter will be updated.
   * \param totalAmpduNumSymbols the number of symbols previously transmitted for the MPDUs in the concerned A-MPDU,
   * used for the computation of the number of symbols needed for the last MPDU.
   * If incFlag is set, this parameter will be updated.
   * \param staId the STA-ID of the PSDU (only used for MU PPDUs)
   *
   * \return the duration of the PSDU
   */
  static Time GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype,
                                  bool incFlag, uint32_t &totalAmpduSize, double &totalAmpduNumSymbols,
                                  uint16_t staId);
  /**
   * \param txVector the transmission parameters used for this packet
   *
   * \return the duration until the start of the packet
   */
  static Time GetStartOfPacketDuration (const WifiTxVector& txVector);

  /**
   * The WifiPhy::GetModeList() method is used
   * (e.g., by a WifiRemoteStationManager) to determine the set of
   * transmission/reception (non-MCS) modes that this WifiPhy(-derived class)
   * can support - a set of modes which is stored by each non-HT PHY.
   *
   * It is important to note that this list is a superset (not
   * necessarily proper) of the OperationalRateSet (which is
   * logically, if not actually, a property of the associated
   * WifiRemoteStationManager), which itself is a superset (again, not
   * necessarily proper) of the BSSBasicRateSet.
   *
   * \return the list of supported (non-MCS) modes.
   */
  std::list<WifiMode> GetModeList (void) const;
  /**
   * Get the list of supported (non-MCS) modes for the given modulation class (i.e.
   * corresponding to a given PHY entity).
   *
   * \param modulation the modulation class
   * \return the list of supported (non-MCS) modes for the given modulation class.
   *
   * \see GetModeList (void)
   */
  std::list<WifiMode> GetModeList (WifiModulationClass modulation) const;
  /**
   * Check if the given WifiMode is supported by the PHY.
   *
   * \param mode the wifi mode to check
   *
   * \return true if the given mode is supported,
   *         false otherwise
   */
  bool IsModeSupported (WifiMode mode) const;
  /**
   * Get the default WifiMode supported by the PHY.
   * This is the first mode to be added (i.e. the lowest one
   * over all supported PHY entities).
   *
   * \return the default WifiMode
   */
  WifiMode GetDefaultMode (void) const;
  /**
   * Check if the given MCS of the given modulation class is supported by the PHY.
   *
   * \param modulation the modulation class
   * \param mcs the MCS value
   *
   * \return true if the given mode is supported,
   *         false otherwise
   */
  bool IsMcsSupported (WifiModulationClass modulation, uint8_t mcs) const;

  /**
   * \param txVector the transmission vector
   * \param ber the probability of bit error rate
   *
   * \return the minimum SNR which is required to achieve
   *          the requested BER for the specified transmission vector. (W/W)
   */
  double CalculateSnr (const WifiTxVector& txVector, double ber) const;

  /**
   * Set the Short Interframe Space (SIFS) for this PHY.
   *
   * \param sifs the SIFS duration
   */
  void SetSifs (Time sifs);
  /**
   * Return the Short Interframe Space (SIFS) for this PHY.
   *
   * \return the SIFS duration
   */
  Time GetSifs (void) const;
  /**
   * Set the slot duration for this PHY.
   *
   * \param slot the slot duration
   */
  void SetSlot (Time slot);
  /**
   * Return the slot duration for this PHY.
   *
   * \return the slot duration
   */
  Time GetSlot (void) const;
  /**
   * Set the PCF Interframe Space (PIFS) for this PHY.
   *
   * \param pifs the PIFS duration
   */
  void SetPifs (Time pifs);
  /**
   * Return the PCF Interframe Space (PIFS) for this PHY.
   *
   * \return the PIFS duration
   */
  Time GetPifs (void) const;
  /**
   * Return the estimated Ack TX time for this PHY.
   *
   * \return the estimated Ack TX time
   */
  Time GetAckTxTime (void) const;
  /**
   * Return the estimated BlockAck TX time for this PHY.
   *
   * \return the estimated BlockAck TX time
   */
  Time GetBlockAckTxTime (void) const;

  /**
   * Get the maximum PSDU size in bytes for the given modulation class.
   *
   * \param modulation the modulation class
   * \return the maximum PSDU size in bytes for the given modulation class
   */
  static uint32_t GetMaxPsduSize (WifiModulationClass modulation);

  /**
  * The WifiPhy::BssMembershipSelector() method is used
  * (e.g., by a WifiRemoteStationManager) to determine the set of
  * transmission/reception modes that this WifiPhy(-derived class)
  * can support - a set of WifiMode objects which we call the
  * BssMembershipSelectorSet, and which is stored inside HT PHY (and above)
  * instances.
  *
  * \return the list of BSS membership selectors.
  */
  std::list<uint8_t> GetBssMembershipSelectorList (void) const;
  /**
   * \return the number of supported MCSs.
   *
   * \see GetMcsList (void)
   */
  uint16_t GetNMcs (void) const;
  /**
   * The WifiPhy::GetMcsList() method is used
   * (e.g., by a WifiRemoteStationManager) to determine the set of
   * transmission/reception MCS indices that this WifiPhy(-derived class)
   * can support - a set of MCS indices which is stored by each HT PHY (and above).
   *
   * \return the list of supported MCSs.
   */
  std::list<WifiMode> GetMcsList (void) const;
  /**
   * Get the list of supported MCSs for the given modulation class (i.e.
   * corresponding to a given PHY entity).
   *
   * \param modulation the modulation class
   * \return the list of supported MCSs for the given modulation class.
   *
   * \see GetMcsList (void)
   */
  std::list<WifiMode> GetMcsList (WifiModulationClass modulation) const;
  /**
   * Get the WifiMode object corresponding to the given MCS of the given
   * modulation class.
   *
   * \param modulation the modulation class
   * \param mcs the MCS value
   *
   * \return the WifiMode object corresponding to the given MCS of the given
   *         modulation class
   */
  WifiMode GetMcs (WifiModulationClass modulation, uint8_t mcs) const;

  /**
   * Return current channel number.
   *
   * \return the current channel number
   */
  uint8_t GetChannelNumber (void) const;
  /**
   * \return the required time for channel switch operation of this WifiPhy
   */
  Time GetChannelSwitchDelay (void) const;

  /**
   * Configure the PHY-level parameters for different Wi-Fi standard.
   *
   * \param standard the Wi-Fi standard
   */
  virtual void ConfigureStandard (WifiStandard standard);

  /**
   * Get the configured Wi-Fi standard
   *
   * \return the Wi-Fi standard that has been configured
   */
  WifiStandard GetStandard (void) const;

  /**
   * Get the configured Wi-Fi band
   *
   * \return the Wi-Fi band that has been configured
   */
  WifiPhyBand GetPhyBand (void) const;

  /**
   * Get a const reference to the operating channel
   *
   * \return a const reference to the operating channel
   */
  const WifiPhyOperatingChannel& GetOperatingChannel (void) const;

  /**
   * Return the Channel this WifiPhy is connected to.
   *
   * \return the Channel this WifiPhy is connected to
   */
  virtual Ptr<Channel> GetChannel (void) const = 0;

  /**
   * Public method used to fire a PhyTxBegin trace.
   * Implemented for encapsulation purposes.
   *
   * \param psdus the PSDUs being transmitted (only one unless DL MU transmission)
   * \param txPowerW the transmit power in Watts
   */
  void NotifyTxBegin (WifiConstPsduMap psdus, double txPowerW);
  /**
   * Public method used to fire a PhyTxEnd trace.
   * Implemented for encapsulation purposes.
   *
   * \param psdus the PSDUs being transmitted (only one unless DL MU transmission)
   */
  void NotifyTxEnd (WifiConstPsduMap psdus);
  /**
   * Public method used to fire a PhyTxDrop trace.
   * Implemented for encapsulation purposes.
   *
   * \param psdu the PSDU being transmitted
   */
  void NotifyTxDrop (Ptr<const WifiPsdu> psdu);
  /**
   * Public method used to fire a PhyRxBegin trace.
   * Implemented for encapsulation purposes.
   *
   * \param psdu the PSDU being transmitted
   * \param rxPowersW the receive power per channel band in Watts
   */
  void NotifyRxBegin (Ptr<const WifiPsdu> psdu, const RxPowerWattPerChannelBand& rxPowersW);
  /**
   * Public method used to fire a PhyRxEnd trace.
   * Implemented for encapsulation purposes.
   *
   * \param psdu the PSDU being transmitted
   */
  void NotifyRxEnd (Ptr<const WifiPsdu> psdu);
  /**
   * Public method used to fire a PhyRxDrop trace.
   * Implemented for encapsulation purposes.
   *
   * \param psdu the PSDU being transmitted
   * \param reason the reason the packet was dropped
   */
  void NotifyRxDrop (Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason);

  /**
   * Public method used to fire a MonitorSniffer trace for a wifi PSDU being received.
   * Implemented for encapsulation purposes.
   * This method will extract all MPDUs if packet is an A-MPDU and will fire tracedCallback.
   * The A-MPDU reference number (RX side) is set within the method. It must be a different value
   * for each A-MPDU but the same for each subframe within one A-MPDU.
   *
   * \param psdu the PSDU being received
   * \param channelFreqMhz the frequency in MHz at which the packet is
   *        received. Note that in real devices this is normally the
   *        frequency to which  the receiver is tuned, and this can be
   *        different than the frequency at which the packet was originally
   *        transmitted. This is because it is possible to have the receiver
   *        tuned on a given channel and still to be able to receive packets
   *        on a nearby channel.
   * \param txVector the TXVECTOR that holds RX parameters
   * \param signalNoise signal power and noise power in dBm (noise power includes the noise figure)
   * \param statusPerMpdu reception status per MPDU
   * \param staId the STA-ID
   */
  void NotifyMonitorSniffRx (Ptr<const WifiPsdu> psdu,
                             uint16_t channelFreqMhz,
                             WifiTxVector txVector,
                             SignalNoiseDbm signalNoise,
                             std::vector<bool> statusPerMpdu,
                             uint16_t staId = SU_STA_ID);

  /**
   * TracedCallback signature for monitor mode receive events.
   *
   *
   * \param packet the packet being received
   * \param channelFreqMhz the frequency in MHz at which the packet is
   *        received. Note that in real devices this is normally the
   *        frequency to which  the receiver is tuned, and this can be
   *        different than the frequency at which the packet was originally
   *        transmitted. This is because it is possible to have the receiver
   *        tuned on a given channel and still to be able to receive packets
   *        on a nearby channel.
   * \param txVector the TXVECTOR that holds RX parameters
   * \param aMpdu the type of the packet (0 is not A-MPDU, 1 is a MPDU that is part of an A-MPDU and 2 is the last MPDU in an A-MPDU)
   *        and the A-MPDU reference number (must be a different value for each A-MPDU but the same for each subframe within one A-MPDU)
   * \param signalNoise signal power and noise power in dBm
   * \param staId the STA-ID
   * \todo WifiTxVector should be passed by const reference because
   * of its size.
   */
  typedef void (* MonitorSnifferRxCallback)(Ptr<const Packet> packet,
                                            uint16_t channelFreqMhz,
                                            WifiTxVector txVector,
                                            MpduInfo aMpdu,
                                            SignalNoiseDbm signalNoise,
                                            uint16_t staId);

  /**
   * Public method used to fire a MonitorSniffer trace for a wifi PSDU being transmitted.
   * Implemented for encapsulation purposes.
   * This method will extract all MPDUs if packet is an A-MPDU and will fire tracedCallback.
   * The A-MPDU reference number (RX side) is set within the method. It must be a different value
   * for each A-MPDU but the same for each subframe within one A-MPDU.
   *
   * \param psdu the PSDU being received
   * \param channelFreqMhz the frequency in MHz at which the packet is
   *        transmitted.
   * \param txVector the TXVECTOR that holds TX parameters
   * \param staId the STA-ID
   */
  void NotifyMonitorSniffTx (Ptr<const WifiPsdu> psdu,
                             uint16_t channelFreqMhz,
                             WifiTxVector txVector,
                             uint16_t staId = SU_STA_ID);

  /**
   * TracedCallback signature for monitor mode transmit events.
   *
   * \param packet the packet being transmitted
   * \param channelFreqMhz the frequency in MHz at which the packet is
   *        transmitted.
   * \param txVector the TXVECTOR that holds TX parameters
   * \param aMpdu the type of the packet (0 is not A-MPDU, 1 is a MPDU that is part of an A-MPDU and 2 is the last MPDU in an A-MPDU)
   *        and the A-MPDU reference number (must be a different value for each A-MPDU but the same for each subframe within one A-MPDU)
   * \param staId the STA-ID
   * \todo WifiTxVector should be passed by const reference because
   * of its size.
   */
  typedef void (* MonitorSnifferTxCallback)(const Ptr<const Packet> packet,
                                            uint16_t channelFreqMhz,
                                            WifiTxVector txVector,
                                            MpduInfo aMpdu,
                                            uint16_t staId);

  /**
   * TracedCallback signature for Phy transmit events.
   *
   * \param packet the packet being transmitted
   * \param txPowerW the transmit power in Watts
   */
  typedef void (* PhyTxBeginTracedCallback)(Ptr<const Packet> packet, double txPowerW);

  /**
   * TracedCallback signature for PSDU transmit events.
   *
   * \param psduMap the PSDU map being transmitted
   * \param txVector the TXVECTOR holding the TX parameters
   * \param txPowerW the transmit power in Watts
   */
  typedef void (* PsduTxBeginCallback)(WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);

  /**
   * TracedCallback signature for PhyRxBegin trace source.
   *
   * \param packet the packet being received
   * \param rxPowersW the receive power per channel band in Watts
   */
  typedef void (* PhyRxBeginTracedCallback) (Ptr<const Packet> packet, RxPowerWattPerChannelBand rxPowersW);

  /**
   * TracedCallback signature for start of PSDU reception events.
   *
   * \param txVector the TXVECTOR decoded from the PHY header
   * \param psduDuration the duration of the PSDU
   */
  typedef void (* PhyRxPayloadBeginTracedCallback)(WifiTxVector txVector, Time psduDuration);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model. Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  virtual int64_t AssignStreams (int64_t stream);

  /**
   * Sets the receive sensitivity threshold (dBm).
   * The energy of a received signal should be higher than
   * this threshold to allow the PHY layer to detect the signal.
   *
   * \param threshold the receive sensitivity threshold in dBm
   */
  void SetRxSensitivity (double threshold);
  /**
   * Return the receive sensitivity threshold (dBm).
   *
   * \return the receive sensitivity threshold in dBm
   */
  double GetRxSensitivity (void) const;
  /**
   * Sets the CCA threshold (dBm). The energy of a received signal
   * should be higher than this threshold to allow the PHY
   * layer to declare CCA BUSY state.
   *
   * \param threshold the CCA threshold in dBm
   */
  void SetCcaEdThreshold (double threshold);
  /**
   * Return the CCA threshold (dBm).
   *
   * \return the CCA threshold in dBm
   */
  double GetCcaEdThreshold (void) const;
  /**
   * Sets the RX loss (dB) in the Signal-to-Noise-Ratio due to non-idealities in the receiver.
   *
   * \param noiseFigureDb noise figure in dB
   */
  void SetRxNoiseFigure (double noiseFigureDb);
  /**
   * Sets the minimum available transmission power level (dBm).
   *
   * \param start the minimum transmission power level (dBm)
   */
  void SetTxPowerStart (double start);
  /**
   * Return the minimum available transmission power level (dBm).
   *
   * \return the minimum available transmission power level (dBm)
   */
  double GetTxPowerStart (void) const;
  /**
   * Sets the maximum available transmission power level (dBm).
   *
   * \param end the maximum transmission power level (dBm)
   */
  void SetTxPowerEnd (double end);
  /**
   * Return the maximum available transmission power level (dBm).
   *
   * \return the maximum available transmission power level (dBm)
   */
  double GetTxPowerEnd (void) const;
  /**
   * Sets the number of transmission power levels available between the
   * minimum level and the maximum level. Transmission power levels are
   * equally separated (in dBm) with the minimum and the maximum included.
   *
   * \param n the number of available levels
   */
  void SetNTxPower (uint8_t n);
  /**
   * Return the number of available transmission power levels.
   *
   * \return the number of available transmission power levels
   */
  uint8_t GetNTxPower (void) const;
  /**
   * Sets the transmission gain (dB).
   *
   * \param gain the transmission gain in dB
   */
  void SetTxGain (double gain);
  /**
   * Return the transmission gain (dB).
   *
   * \return the transmission gain in dB
   */
  double GetTxGain (void) const;
  /**
   * Sets the reception gain (dB).
   *
   * \param gain the reception gain in dB
   */
  void SetRxGain (double gain);
  /**
   * Return the reception gain (dB).
   *
   * \return the reception gain in dB
   */
  double GetRxGain (void) const;

  /**
   * Sets the device this PHY is associated with.
   *
   * \param device the device this PHY is associated with
   */
  void SetDevice (const Ptr<WifiNetDevice> device);
  /**
   * Return the device this PHY is associated with
   *
   * \return the device this PHY is associated with
   */
  Ptr<WifiNetDevice> GetDevice (void) const;
  /**
   * \brief assign a mobility model to this device
   *
   * This method allows a user to specify a mobility model that should be
   * associated with this physical layer.  Calling this method is optional
   * and only necessary if the user wants to override the mobility model
   * that is aggregated to the node.
   *
   * \param mobility the mobility model this PHY is associated with
   */
  void SetMobility (const Ptr<MobilityModel> mobility);
  /**
   * Return the mobility model this PHY is associated with.
   * This method will return either the mobility model that has been
   * explicitly set by a call to YansWifiPhy::SetMobility(), or else
   * will return the mobility model (if any) that has been aggregated
   * to the node.
   *
   * \return the mobility model this PHY is associated with
   */
  Ptr<MobilityModel> GetMobility (void) const;

  using ChannelTuple = std::tuple<uint8_t /* channel number */,
                                  uint16_t /* channel width */,
                                  int /* WifiPhyBand */,
                                  uint8_t /* primary20 index*/>;  //!< Tuple identifying an operating channel

  /**
   * If the standard for this object has not been set yet, store the given channel
   * settings. Otherwise, check if a channel switch can be performed now. If not,
   * schedule another call to this method when channel switch can be performed.
   * Otherwise, set the operating channel based on the given channel settings and
   * call ConfigureStandard if the PHY band has changed.
   *
   * \param channelTuple the given channel settings
   */
  void SetOperatingChannel (const ChannelTuple& channelTuple);
  /**
   * \return the operating center frequency (MHz)
   */
  uint16_t GetFrequency (void) const;
  /**
   * \return the index of the primary 20 MHz channel
   */
  uint8_t GetPrimary20Index (void) const;
  /**
   * \param antennas the number of antennas on this node.
   */
  void SetNumberOfAntennas (uint8_t antennas);
  /**
   * \return the number of antennas on this device
   */
  uint8_t GetNumberOfAntennas (void) const;
  /**
   * \param streams the maximum number of supported TX spatial streams.
   */
  void SetMaxSupportedTxSpatialStreams (uint8_t streams);
  /**
   * \return the maximum number of supported TX spatial streams
   */
  uint8_t GetMaxSupportedTxSpatialStreams (void) const;
  /**
   * \param streams the maximum number of supported RX spatial streams.
   */
  void SetMaxSupportedRxSpatialStreams (uint8_t streams);
  /**
   * \return the maximum number of supported RX spatial streams
   */
  uint8_t GetMaxSupportedRxSpatialStreams (void) const;
  /**
   * Enable or disable short PHY preamble.
   *
   * \param preamble sets whether short PHY preamble is supported or not
   */
  void SetShortPhyPreambleSupported (bool preamble);
  /**
   * Return whether short PHY preamble is supported.
   *
   * \returns if short PHY preamble is supported or not
   */
  bool GetShortPhyPreambleSupported (void) const;

  /**
   * Sets the interference helper.
   *
   * \param helper the interference helper
   */
  virtual void SetInterferenceHelper (const Ptr<InterferenceHelper> helper);

  /**
   * Sets the error rate model.
   *
   * \param model the error rate model
   */
  void SetErrorRateModel (const Ptr<ErrorRateModel> model);
  /**
   * Attach a receive ErrorModel to the WifiPhy.
   *
   * The WifiPhy may optionally include an ErrorModel in
   * the packet receive chain. The error model is additive
   * to any modulation-based error model based on SNR, and
   * is typically used to force specific packet losses or
   * for testing purposes.
   *
   * \param em Pointer to the ErrorModel.
   */
  void SetPostReceptionErrorModel (const Ptr<ErrorModel> em);
  /**
   * Sets the frame capture model.
   *
   * \param frameCaptureModel the frame capture model
   */
  void SetFrameCaptureModel (const Ptr<FrameCaptureModel> frameCaptureModel);
  /**
   * Sets the preamble detection model.
   *
   * \param preambleDetectionModel the preamble detection model
   */
  void SetPreambleDetectionModel (const Ptr<PreambleDetectionModel> preambleDetectionModel);
  /**
   * Sets the wifi radio energy model.
   *
   * \param wifiRadioEnergyModel the wifi radio energy model
   */
  void SetWifiRadioEnergyModel (const Ptr<WifiRadioEnergyModel> wifiRadioEnergyModel);

  /**
   * \return the channel width in MHz
   */
  uint16_t GetChannelWidth (void) const;
  /**
   * \param width the channel width (in MHz) to support
   */
  void AddSupportedChannelWidth (uint16_t width);
  /**
   * \return a vector containing the supported channel widths, values in MHz
   */
  std::vector<uint16_t> GetSupportedChannelWidthSet (void) const;

  /**
   * Get the power of the given power level in dBm.
   * In SpectrumWifiPhy implementation, the power levels are equally spaced (in dBm).
   *
   * \param power the power level
   *
   * \return the transmission power in dBm at the given power level
   */
  double GetPowerDbm (uint8_t power) const;

  /**
   * Reset PHY to IDLE, with some potential TX power restrictions for the next transmission.
   *
   * \param powerRestricted flag whether the transmit power is restricted for the next transmission
   * \param txPowerMaxSiso the SISO transmit power restriction for the next transmission in dBm
   * \param txPowerMaxMimo the MIMO transmit power restriction for the next transmission in dBm
   */
  void ResetCca (bool powerRestricted, double txPowerMaxSiso = 0, double txPowerMaxMimo = 0);
  /**
   * Compute the transmit power for the next transmission.
   * The returned power will satisfy the power density constraints
   * after addition of antenna gain.
   *
   * \param ppdu the PPDU to transmit
   * \return the transmit power in dBm for the next transmission
   */
  double GetTxPowerForTransmission (Ptr<const WifiPpdu> ppdu) const;
  /**
   * Notify the PHY that an access to the channel was requested.
   * This is typically called by the channel access manager to
   * to notify the PHY about an ongoing transmission.
   * The PHY will use this information to determine whether
   * it should use power restriction as imposed by OBSS_PD SR.
   */
  void NotifyChannelAccessRequested (void);


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
  virtual WifiSpectrumBand ConvertHeRuSubcarriers (uint16_t bandWidth, uint16_t guardBandwidth,
                                                   HeRu::SubcarrierRange range, uint8_t bandIndex = 0) const;

  /**
   * Add the PHY entity to the map of __implemented__ PHY entities for the
   * given modulation class.
   * Through this method, child classes can add their own PHY entities in
   * a static manner.
   *
   * \param modulation the modulation class
   * \param phyEntity the PHY entity
   */
  static void AddStaticPhyEntity (WifiModulationClass modulation, Ptr<PhyEntity> phyEntity);

  /**
   * Get the __implemented__ PHY entity corresponding to the modulation class.
   * This is used to compute the different amendment-specific parameters within
   * calling static methods.
   *
   * \param modulation the modulation class
   * \return the pointer to the static implemented PHY entity
   */
  static const Ptr<const PhyEntity> GetStaticPhyEntity (WifiModulationClass modulation);

  /**
   * Get the supported PHY entity corresponding to the modulation class, for
   * the WifiPhy instance.
   *
   * \param modulation the modulation class
   * \return the pointer to the supported PHY entity
   */
  Ptr<PhyEntity> GetPhyEntity (WifiModulationClass modulation) const;

  /**
   * \return the UID of the previously received PPDU (reset to UINT64_MAX upon transmission)
   */
  uint64_t GetPreviouslyRxPpduUid (void) const;

  /**
   * \param currentChannelWidth channel width of the current transmission (MHz)
   * \return the width of the guard band (MHz)
   *
   * Note: in order to properly model out of band transmissions for OFDM, the guard
   * band has been configured so as to expand the modeled spectrum up to the
   * outermost referenced point in "Transmit spectrum mask" sections' PSDs of
   * each PHY specification of 802.11-2016 standard. It thus ultimately corresponds
   * to the current channel bandwidth (which can be different from devices max
   * channel width).
   *
   * This method is only relevant for SpectrumWifiPhy.
   */
  virtual uint16_t GetGuardBandwidth (uint16_t currentChannelWidth) const = 0;
  /**
   * \return a tuple containing the minimum rejection (in dBr) for the inner band,
   *                            the minimum rejection (in dBr) for the outer band, and
   *                            the maximum rejection (in dBr) for the outer band
   *                            for the transmit spectrum mask.
   *
   * This method is only relevant for SpectrumWifiPhy.
   */
  virtual std::tuple<double, double, double> GetTxMaskRejectionParams (void) const = 0;

  /**
   * Get channel number of the primary channel
   * \param primaryChannelWidth the width of the primary channel (MHz)
   *
   * \return channel number of the primary channel
   */
  uint8_t GetPrimaryChannelNumber (uint16_t primaryChannelWidth) const;

protected:
  virtual void DoDispose (void);

  /**
   * Reset data upon end of TX or RX
   */
  void Reset (void);

  /**
   * Perform any actions necessary when user changes operating channel after
   * initialization.
   *
   * \return zero if the PHY can immediately switch channel, a positive value
   *         indicating the amount of time to wait until the channel switch can
   *         be performed or a negative value indicating that channel switch is
   *         currently not possible (i.e., the radio is in sleep mode)
   */
  Time GetDelayUntilChannelSwitch (void);
  /**
   * Actually switch channel based on the stored channel settings.
   */
  virtual void DoChannelSwitch (void);

  /**
   * Check if PHY state should move to CCA busy state based on current
   * state of interference tracker.  In this model, CCA becomes busy when
   * the aggregation of all signals as tracked by the InterferenceHelper
   * class is higher than the CcaEdThreshold
   *
   * \param channelWidth the channel width in MHz used for RSSI measurement
   */
  void SwitchMaybeToCcaBusy (uint16_t channelWidth);

  /**
   * Return the channel width used to measure the RSSI.
   * This corresponds to the primary channel unless it corresponds to the
   * HE TB PPDU solicited by the AP.
   *
   * \param ppdu the PPDU that is being received
   * \return the channel width (in MHz) used for RSSI measurement
   */
  uint16_t GetMeasurementChannelWidth (const Ptr<const WifiPpdu> ppdu) const;

  /**
   * Get the start band index and the stop band index for a given band
   *
   * \param bandWidth the width of the band to be returned (MHz)
   * \param bandIndex the index of the band to be returned
   *
   * \return a pair of start and stop indexes that defines the band
   */
  virtual WifiSpectrumBand GetBand (uint16_t bandWidth, uint8_t bandIndex = 0);

  /**
   * If the operating channel width is a multiple of 20 MHz, return the start
   * band index and the stop band index for the primary channel of the given
   * bandwidth (which must be a multiple of 20 MHz and not exceed the operating
   * channel width). Otherwise, this call is equivalent to GetBand with
   * <i>bandIndex</i> equal to zero.
   *
   * \param bandWidth the width of the band to be returned (MHz)
   *
   * \return a pair of start and stop indexes that defines the band
   */
  WifiSpectrumBand GetPrimaryBand (uint16_t bandWidth);

  /**
   * Add the PHY entity to the map of supported PHY entities for the
   * given modulation class for the WifiPhy instance.
   * This is a wrapper method used to check that the PHY entity is
   * in the static map of implemented PHY entities (\see GetStaticPhyEntities).
   * In addition, child classes can add their own PHY entities.
   *
   * \param modulation the modulation class
   * \param phyEntity the PHY entity
   */
  void AddPhyEntity (WifiModulationClass modulation, Ptr<PhyEntity> phyEntity);

  Ptr<InterferenceHelper> m_interference; //!< Pointer to a helper responsible for interference computations

  Ptr<UniformRandomVariable> m_random; //!< Provides uniform random variables.
  Ptr<WifiPhyStateHelper> m_state;     //!< Pointer to WifiPhyStateHelper

  uint32_t m_txMpduReferenceNumber;    //!< A-MPDU reference number to identify all transmitted subframes belonging to the same received A-MPDU
  uint32_t m_rxMpduReferenceNumber;    //!< A-MPDU reference number to identify all received subframes belonging to the same received A-MPDU

  EventId m_endPhyRxEvent;             //!< the end of PHY receive event
  EventId m_endTxEvent;                //!< the end of transmit event

  Ptr<Event> m_currentEvent; //!< Hold the current event
  std::map <std::pair<uint64_t /* UID*/, WifiPreamble>, Ptr<Event> > m_currentPreambleEvents; //!< store event associated to a PPDU (that has a unique ID and preamble combination) whose preamble is being received

  uint64_t m_previouslyRxPpduUid;  //!< UID of the previously received PPDU, reset to UINT64_MAX upon transmission

  /**
   * This map holds the supported PHY entities.
   *
   * The set of parameters (e.g. mode) that this WifiPhy(-derived class) can
   * support can be obtained through it.
   *
   * When it comes to modes, in conversation we call this set
   * the DeviceRateSet (not a term you'll find in the standard), and
   * it is a superset of standard-defined parameters such as the
   * OperationalRateSet, and the BSSBasicRateSet (which, themselves,
   * have a superset/subset relationship).
   *
   * Mandatory rates relevant to this WifiPhy can be found by
   * iterating over the elements of this map, for each modulation class,
   * looking for WifiMode objects for which
   * WifiMode::IsMandatory() is true.
   */
  std::map<WifiModulationClass, Ptr<PhyEntity> > m_phyEntities;


private:
  /**
   * Configure WifiPhy with appropriate channel frequency and
   * supported rates for 802.11a standard.
   */
  void Configure80211a (void);
  /**
   * Configure WifiPhy with appropriate channel frequency and
   * supported rates for 802.11b standard.
   */
  void Configure80211b (void);
  /**
   * Configure WifiPhy with appropriate channel frequency and
   * supported rates for 802.11g standard.
   */
  void Configure80211g (void);
  /**
   * Configure WifiPhy with appropriate channel frequency and
   * supported rates for 802.11p standard.
   */
  void Configure80211p (void);
  /**
   * Configure WifiPhy with appropriate channel frequency and
   * supported rates for 802.11n standard.
   */
  void Configure80211n (void);
  /**
   * Configure WifiPhy with appropriate channel frequency and
   * supported rates for 802.11ac standard.
   */
  void Configure80211ac (void);
  /**
   * Configure WifiPhy with appropriate channel frequency and
   * supported rates for 802.11ax standard.
   */
  void Configure80211ax (void);
  /**
   * Configure the device MCS set with the appropriate HtMcs modes for
   * the number of available transmit spatial streams
   */
  void ConfigureHtDeviceMcsSet (void);
  /**
   * Add the given MCS to the device MCS set.
   *
   * \param mode the MCS to add to the device MCS set
   */
  void PushMcs (WifiMode mode);
  /**
   * Rebuild the mapping of MCS values to indices in the device MCS set.
   */
  void RebuildMcsMap (void);

  /**
   * Due to newly arrived signal, the current reception cannot be continued and has to be aborted
   * \param reason the reason the reception is aborted
   *
   */
  void AbortCurrentReception (WifiPhyRxfailureReason reason);

  /**
   * Get the PSDU addressed to that PHY in a PPDU (useful for MU PPDU).
   *
   * \param ppdu the PPDU to extract the PSDU from
   * \return the PSDU addressed to that PHY
   */
  Ptr<const WifiPsdu> GetAddressedPsduInPpdu (Ptr<const WifiPpdu> ppdu) const;

  /**
   * The trace source fired when a packet begins the transmission process on
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet>, double > m_phyTxBeginTrace;
  /**
   * The trace source fired when a PSDU map begins the transmission process on
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<WifiConstPsduMap, WifiTxVector, double /* TX power (W) */> m_phyTxPsduBeginTrace;

  /**
   * The trace source fired when a packet ends the transmission process on
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyTxEndTrace;

  /**
   * The trace source fired when the PHY layer drops a packet as it tries
   * to transmit it.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyTxDropTrace;

  /**
   * The trace source fired when a packet begins the reception process from
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet>, RxPowerWattPerChannelBand > m_phyRxBeginTrace;

  /**
   * The trace source fired when the reception of the PHY payload (PSDU) begins.
   *
   * This traced callback models the behavior of the PHY-RXSTART
   * primitive which is launched upon correct decoding of
   * the PHY header and support of modes within.
   * We thus assume that it is sent just before starting
   * the decoding of the payload, since it's there that
   * support of the header's content is checked. In addition,
   * it's also at that point that the correct decoding of
   * HT-SIG, VHT-SIGs, and HE-SIGs are checked.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<WifiTxVector, Time> m_phyRxPayloadBeginTrace;

  /**
   * The trace source fired when a packet ends the reception process from
   * the medium.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet> > m_phyRxEndTrace;

  /**
   * The trace source fired when the PHY layer drops a packet it has received.
   *
   * \see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet>, WifiPhyRxfailureReason > m_phyRxDropTrace;

  /**
   * A trace source that emulates a Wi-Fi device in monitor mode
   * sniffing a packet being received.
   *
   * As a reference with the real world, firing this trace
   * corresponds in the madwifi driver to calling the function
   * ieee80211_input_monitor()
   *
   * \see class CallBackTraceSource
   * \todo WifiTxVector and signalNoiseDbm should be passed as
   *       const references because of their sizes.
   */
  TracedCallback<Ptr<const Packet>, uint16_t /* frequency (MHz) */, WifiTxVector, MpduInfo, SignalNoiseDbm, uint16_t /* STA-ID*/> m_phyMonitorSniffRxTrace;

  /**
   * A trace source that emulates a Wi-Fi device in monitor mode
   * sniffing a packet being transmitted.
   *
   * As a reference with the real world, firing this trace
   * corresponds in the madwifi driver to calling the function
   * ieee80211_input_monitor()
   *
   * \see class CallBackTraceSource
   * \todo WifiTxVector should be passed by const reference because
   * of its size.
   */
  TracedCallback<Ptr<const Packet>, uint16_t /* frequency (MHz) */, WifiTxVector, MpduInfo, uint16_t /* STA-ID*/> m_phyMonitorSniffTxTrace;

  /**
   * \return the map of __implemented__ PHY entities.
   * This is used to compute the different
   * amendment-specific parameters in a static manner.
   * For PHY entities supported by a given WifiPhy instance,
   * \see m_phyEntities.
   */
  static std::map<WifiModulationClass, Ptr<PhyEntity> > & GetStaticPhyEntities (void);

  WifiStandard m_standard;                  //!< WifiStandard
  WifiPhyBand m_band;                       //!< WifiPhyBand
  ChannelTuple m_channelSettings;           //!< Store operating channel settings until initialization
  WifiPhyOperatingChannel m_operatingChannel;       //!< Operating channel
  std::vector<uint16_t> m_supportedChannelWidthSet; //!< Supported channel width set (MHz)

  Time m_sifs;                              //!< Short Interframe Space (SIFS) duration
  Time m_slot;                              //!< Slot duration
  Time m_pifs;                              //!< PCF Interframe Space (PIFS) duration
  Time m_ackTxTime;                         //!< estimated Ack TX time
  Time m_blockAckTxTime;                    //!< estimated BlockAck TX time

  double   m_rxSensitivityW;      //!< Receive sensitivity threshold in watts
  double   m_ccaEdThresholdW;     //!< Clear channel assessment (CCA) threshold in watts
  double   m_txGainDb;            //!< Transmission gain (dB)
  double   m_rxGainDb;            //!< Reception gain (dB)
  double   m_txPowerBaseDbm;      //!< Minimum transmission power (dBm)
  double   m_txPowerEndDbm;       //!< Maximum transmission power (dBm)
  uint8_t  m_nTxPower;            //!< Number of available transmission power levels
  double m_powerDensityLimit;     //!< the power density limit (dBm/MHz)

  bool m_powerRestricted;        //!< Flag whether transmit power is restricted by OBSS PD SR
  double m_txPowerMaxSiso;       //!< SISO maximum transmit power due to OBSS PD SR power restriction (dBm)
  double m_txPowerMaxMimo;       //!< MIMO maximum transmit power due to OBSS PD SR power restriction (dBm)
  bool m_channelAccessRequested; //!< Flag if channels access has been requested (used for OBSS_PD SR)

  bool m_shortPreamble;        //!< Flag if short PHY preamble is supported
  uint8_t m_numberOfAntennas;  //!< Number of transmitters
  uint8_t m_txSpatialStreams;  //!< Number of supported TX spatial streams
  uint8_t m_rxSpatialStreams;  //!< Number of supported RX spatial streams

  double m_noiseFigureDb; //!< The noise figure in dB

  Time m_channelSwitchDelay;     //!< Time required to switch between channel

  Ptr<WifiNetDevice> m_device;   //!< Pointer to the device
  Ptr<MobilityModel> m_mobility; //!< Pointer to the mobility model

  Ptr<FrameCaptureModel> m_frameCaptureModel;           //!< Frame capture model
  Ptr<PreambleDetectionModel> m_preambleDetectionModel; //!< Preamble detection model
  Ptr<WifiRadioEnergyModel> m_wifiRadioEnergyModel;     //!< Wifi radio energy model
  Ptr<ErrorModel> m_postReceptionErrorModel;            //!< Error model for receive packet events
  Time m_timeLastPreambleDetected;                      //!< Record the time the last preamble was detected

  Callback<void> m_capabilitiesChangedCallback; //!< Callback when PHY capabilities changed
};

/**
 * \param os           output stream
 * \param rxSignalInfo received signal info to stringify
 * \return output stream
 */
std::ostream& operator<< (std::ostream& os, RxSignalInfo rxSignalInfo);

} //namespace ns3

#endif /* WIFI_PHY_H */
