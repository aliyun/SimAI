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

#ifndef WIFI_PHY_STATE_HELPER_H
#define WIFI_PHY_STATE_HELPER_H

#include "ns3/object.h"
#include "ns3/callback.h"
#include "ns3/traced-callback.h"
#include "ns3/nstime.h"
#include "wifi-phy-state.h"
#include "wifi-phy-common.h"
#include "wifi-ppdu.h"

namespace ns3 {

class WifiPhyListener;
class WifiTxVector;
class WifiMode;
class Packet;
class WifiPsdu;
struct RxSignalInfo;

/**
 * Callback if PSDU successfully received (i.e. if aggregate,
 * it means that at least one MPDU of the A-MPDU was received,
 * considering that the per-MPDU reception status is also provided).
 *
 * arg1: PSDU received successfully
 * arg2: info on the received signal (\see RxSignalInfo)
 * arg3: TXVECTOR of PSDU
 * arg4: vector of per-MPDU status of reception.
 */
typedef Callback<void, Ptr<WifiPsdu>, RxSignalInfo,
                       WifiTxVector, std::vector<bool>> RxOkCallback;
/**
 * Callback if PSDU unsuccessfully received
 *
 * arg1: PSDU received unsuccessfully
 */
typedef Callback<void, Ptr<WifiPsdu>> RxErrorCallback;

/**
 * \ingroup wifi
 *
 * This objects implements the PHY state machine of the Wifi device.
 */
class WifiPhyStateHelper : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  WifiPhyStateHelper ();

  /**
   * Set a callback for a successful reception.
   *
   * \param callback the RxOkCallback to set
   */
  void SetReceiveOkCallback (RxOkCallback callback);
  /**
   * Set a callback for a failed reception.
   *
   * \param callback the RxErrorCallback to set
   */
  void SetReceiveErrorCallback (RxErrorCallback callback);
  /**
   * Register WifiPhyListener to this WifiPhyStateHelper.
   *
   * \param listener the WifiPhyListener to register
   */
  void RegisterListener (WifiPhyListener *listener);
  /**
   * Remove WifiPhyListener from this WifiPhyStateHelper.
   *
   * \param listener the WifiPhyListener to unregister
   */
  void UnregisterListener (WifiPhyListener *listener);
  /**
   * Return the current state of WifiPhy.
   *
   * \return the current state of WifiPhy
   */
  WifiPhyState GetState (void) const;
  /**
   * Check whether the current state is CCA busy.
   *
   * \return true if the current state is CCA busy, false otherwise
   */
  bool IsStateCcaBusy (void) const;
  /**
   * Check whether the current state is IDLE.
   *
   * \return true if the current state is IDLE, false otherwise
   */
  bool IsStateIdle (void) const;
  /**
   * Check whether the current state is RX.
   *
   * \return true if the current state is RX, false otherwise
   */
  bool IsStateRx (void) const;
  /**
   * Check whether the current state is TX.
   *
   * \return true if the current state is TX, false otherwise
   */
  bool IsStateTx (void) const;
  /**
   * Check whether the current state is SWITCHING.
   *
   * \return true if the current state is SWITCHING, false otherwise
   */
  bool IsStateSwitching (void) const;
  /**
   * Check whether the current state is SLEEP.
   *
   * \return true if the current state is SLEEP, false otherwise
   */
  bool IsStateSleep (void) const;
  /**
   * Check whether the current state is OFF.
   *
   * \return true if the current state is OFF, false otherwise
   */
  bool IsStateOff (void) const;
  /**
   * Return the time before the state is back to IDLE.
   *
   * \return the delay before the state is back to IDLE
   */
  Time GetDelayUntilIdle (void) const;
  /**
   * Return the time the last RX start.
   *
   * \return the time the last RX start.
   */
  Time GetLastRxStartTime (void) const;
  /**
   * Return the time the last RX end.
   *
   * \return the time the last RX end.
   */
  Time GetLastRxEndTime (void) const;

  /**
   * Switch state to TX for the given duration.
   *
   * \param txDuration the duration of the PPDU to transmit
   * \param psdus the PSDUs in the transmitted PPDU (only one unless it is a MU PPDU)
   * \param txPowerDbm the nominal TX power in dBm
   * \param txVector the TX vector for the transmission
   */
  void SwitchToTx (Time txDuration, WifiConstPsduMap psdus, double txPowerDbm, WifiTxVector txVector);
  /**
   * Switch state to RX for the given duration.
   *
   * \param rxDuration the duration of the RX
   */
  void SwitchToRx (Time rxDuration);
  /**
   * Switch state to channel switching for the given duration.
   *
   * \param switchingDuration the duration of required to switch the channel
   */
  void SwitchToChannelSwitching (Time switchingDuration);
  /**
   * Continue RX after the reception of an MPDU in an A-MPDU was successful.
   *
   * \param psdu the successfully received PSDU
   * \param rxSignalInfo the info on the received signal (\see RxSignalInfo)
   * \param txVector TXVECTOR of the PSDU
   */
  void ContinueRxNextMpdu (Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo, WifiTxVector txVector);
  /**
   * Switch from RX after the reception was successful.
   *
   * \param psdu the successfully received PSDU
   * \param rxSignalInfo the info on the received signal (\see RxSignalInfo)
   * \param txVector TXVECTOR of the PSDU
   * \param staId the station ID of the PSDU (only used for MU)
   * \param statusPerMpdu reception status per MPDU
   */
  void SwitchFromRxEndOk (Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo, WifiTxVector txVector,
                          uint16_t staId, std::vector<bool> statusPerMpdu);
  /**
   * Switch from RX after the reception failed.
   *
   * \param psdu the PSDU that we failed to received
   * \param snr the SNR of the received PSDU in linear scale
   */
  void SwitchFromRxEndError (Ptr<WifiPsdu> psdu, double snr);
  /**
   * Switch to CCA busy.
   *
   * \param duration the duration of CCA busy state
   */
  void SwitchMaybeToCcaBusy (Time duration);
  /**
   * Switch to sleep mode.
   */
  void SwitchToSleep (void);
  /**
   * Switch from sleep mode.
   */
  void SwitchFromSleep (void);
  /**
   * Abort current reception
   */
  void SwitchFromRxAbort (void);
  /**
   * Switch to off mode.
   */
  void SwitchToOff (void);
  /**
   * Switch from off mode.
   */
  void SwitchFromOff (void);

  /**
   * TracedCallback signature for state changes.
   *
   * \param [in] start Time when the \pname{state} started.
   * \param [in] duration Amount of time we've been in (or will be in)
   *             the \pname{state}.
   * \param [in] state The state.
   */
  typedef void (* StateTracedCallback)(Time start, Time duration, WifiPhyState state);

  /**
   * TracedCallback signature for receive end OK event.
   *
   * \param [in] packet The received packet.
   * \param [in] snr    The SNR of the received packet in linear scale.
   * \param [in] mode   The transmission mode of the packet.
   * \param [in] preamble The preamble of the packet.
   */
  typedef void (* RxOkTracedCallback)(Ptr<const Packet> packet, double snr, WifiMode mode, WifiPreamble preamble);

  /**
   * TracedCallback signature for receive end error event.
   *
   * \param [in] packet       The received packet.
   * \param [in] snr          The SNR of the received packet in linear scale.
   */
  typedef void (* RxEndErrorTracedCallback)(Ptr<const Packet> packet, double snr);

  /**
   * TracedCallback signature for transmit event.
   *
   * \param [in] packet The received packet.
   * \param [in] mode   The transmission mode of the packet.
   * \param [in] preamble The preamble of the packet.
   * \param [in] power  The transmit power level.
   */
  typedef void (* TxTracedCallback)(Ptr<const Packet> packet, WifiMode mode,
                                    WifiPreamble preamble, uint8_t power);


private:
  /**
   * typedef for a list of WifiPhyListeners
   */
  typedef std::vector<WifiPhyListener *> Listeners;

  /**
   * Log the idle and CCA busy states.
   */
  void LogPreviousIdleAndCcaBusyStates (void);

  /**
   * Notify all WifiPhyListener that the transmission has started for the given duration.
   *
   * \param duration the duration of the transmission
   * \param txPowerDbm the nominal TX power in dBm
   */
  void NotifyTxStart (Time duration, double txPowerDbm);
  /**
   * Notify all WifiPhyListener that the reception has started for the given duration.
   *
   * \param duration the duration of the reception
   */
  void NotifyRxStart (Time duration);
  /**
   * Notify all WifiPhyListener that the reception was successful.
   */
  void NotifyRxEndOk (void);
  /**
   * Notify all WifiPhyListener that the reception was not successful.
   */
  void NotifyRxEndError (void);
  /**
   * Notify all WifiPhyListener that the CCA has started for the given duration.
   *
   * \param duration the duration of the CCA state
   */
  void NotifyMaybeCcaBusyStart (Time duration);
  /**
   * Notify all WifiPhyListener that we are switching channel with the given channel
   * switching delay.
   *
   * \param duration the delay to switch the channel
   */
  void NotifySwitchingStart (Time duration);
  /**
   * Notify all WifiPhyListener that we are going to sleep
   */
  void NotifySleep (void);
  /**
   * Notify all WifiPhyListener that we are going to switch off
   */
  void NotifyOff (void);
  /**
   * Notify all WifiPhyListener that we woke up
   */
  void NotifyWakeup (void);
  /**
   * Switch the state from RX.
   */
  void DoSwitchFromRx (void);
  /**
   * Notify all WifiPhyListener that we are going to switch on
   */
  void NotifyOn (void);

  /**
   * The trace source fired when state is changed.
   */
  TracedCallback<Time, Time, WifiPhyState> m_stateLogger;

  bool m_sleeping; ///< sleeping
  bool m_isOff; ///< switched off
  Time m_endTx; ///< end transmit
  Time m_endRx; ///< end receive
  Time m_endCcaBusy; ///< end CCA busy
  Time m_endSwitching; ///< end switching
  Time m_startTx; ///< start transmit
  Time m_startRx; ///< start receive
  Time m_startCcaBusy; ///< start CCA busy
  Time m_startSwitching; ///< start switching
  Time m_startSleep; ///< start sleep
  Time m_previousStateChangeTime; ///< previous state change time

  Listeners m_listeners; ///< listeners
  TracedCallback<Ptr<const Packet>, double, WifiMode, WifiPreamble> m_rxOkTrace; ///< receive OK trace callback
  TracedCallback<Ptr<const Packet>, double> m_rxErrorTrace; ///< receive error trace callback
  TracedCallback<Ptr<const Packet>, WifiMode, WifiPreamble, uint8_t> m_txTrace; ///< transmit trace callback
  RxOkCallback m_rxOkCallback; ///< receive OK callback
  RxErrorCallback m_rxErrorCallback; ///< receive error callback
};

} //namespace ns3

#endif /* WIFI_PHY_STATE_HELPER_H */
