/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Andrea Sacco
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
 * Author: Andrea Sacco <andrea.sacco85@gmail.com>
 */

#ifndef ACOUSTIC_MODEM_ENERGY_MODEL_H
#define ACOUSTIC_MODEM_ENERGY_MODEL_H

#include "ns3/device-energy-model.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/traced-value.h"

namespace ns3 {

/**
 * \ingroup uan
 *
 * WHOI micro-modem energy model.
 *
 * Basing on the Device Energy Model interface, has been implemented a specific
 * energy model for the WHOI micro modem. The class follows pretty closely the
 * RadioEnergyModel class as the transducer behaviour is pretty close to the one
 * of a wifi radio, with identical states (rx, tx, idle, sleep).
 *
 * The power consumption values implemented into the model are as follows [1]:
 *
 * Modem State   Power Consumption
 * TX            50 W
 * RX            158 mW
 * Idle          158 mW
 * Sleep         5.8 mW
 *
 * References:
 * [1] Freitag et al., The whoi micro-modem: an acoustic communications
 *     and navigation system for multiple platforms,
 *     in In Proc. IEEE OCEANS05 Conf, 2005.
 *     URL: http://ieeexplore.ieee.org/iel5/10918/34367/01639901.pdf
 */
class AcousticModemEnergyModel : public DeviceEnergyModel
{
public:
  /** Callback type for energy depletion handling. */
  typedef Callback<void> AcousticModemEnergyDepletionCallback;
  
  /** Callback type for energy recharge handling. */
  typedef Callback<void> AcousticModemEnergyRechargeCallback;

public:  
  /**
   * Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  /** Constructor. */
  AcousticModemEnergyModel ();
  /** Dummy destructor, see DoDispose */
  virtual ~AcousticModemEnergyModel ();

  /**
   * Sets pointer to node.
   *
   * \param node Pointer to node.
   */
  virtual void SetNode (Ptr<Node> node);

  /**
   * Gets pointer to node.
   *
   * \return Pointer to node.
   */
  virtual Ptr<Node> GetNode (void) const;

  // Inherited methods.
  virtual void SetEnergySource (Ptr<EnergySource> source);
  virtual double GetTotalEnergyConsumption (void) const;

  /**
   * Get the transmission power of the modem.
   *
   * \return The transmission power in Watts.
   */
  double GetTxPowerW (void) const;

  /**
   * Set the transmission power of the modem.
   *
   * \param txPowerW Transmission power in watts.
   */
  void SetTxPowerW (double txPowerW);

  /**
   * Get the receiving power.
   *
   * \return The receiving power in Watts
   */
  double GetRxPowerW (void) const;

  /**
   * Set the receiving power of the modem.
   *
   * \param rxPowerW Receiving power in watts
   */
  void SetRxPowerW (double rxPowerW);

  /**
   *Get the idle power of the modem.
   *
   * \return The idle power in Watts
   */
  double GetIdlePowerW (void) const;

  /**
   * Set the idle state power of the modem.
   *
   * \param idlePowerW Idle power of the modem in watts.
   */
  void SetIdlePowerW (double idlePowerW);

  /**
   * Get the sleep state power of the modem.
   *
   * \return Sleep power of the modem in Watts
   */
  double GetSleepPowerW (void) const;

  /**
   * Set the sleep power of the modem.
   *
   * \param sleepPowerW Sleep power of the modem in watts.
   */
  void SetSleepPowerW (double sleepPowerW);

  /**
   * Get the current state of the modem.
   *
   * \return Current state.
   */
  int GetCurrentState (void) const;

  /**
   * \param callback Callback function.
   *
   * Sets callback for energy depletion handling.
   */
  void SetEnergyDepletionCallback (AcousticModemEnergyDepletionCallback callback);
  
  /**
   * \param callback Callback function.
   *
   * Sets callback for energy recharge handling.
   */
  void SetEnergyRechargeCallback (AcousticModemEnergyRechargeCallback callback);

  /**
   * Changes state of the AcousticModemEnergyModel.
   *
   * \param newState New state the modem is in.
   */
  virtual void ChangeState (int newState);

  /**
   * \brief Handles energy depletion.
   */
  virtual void HandleEnergyDepletion (void);

  /**
   * \brief Handles energy recharged.
   */
  virtual void HandleEnergyRecharged (void);
  
  /**
   * \brief Handles energy changed.
   *
   * Not implemented
   */
  virtual void HandleEnergyChanged (void);


private:
  void DoDispose (void);

  /**
   * \return Current draw of device, at current state.
   */
  virtual double DoGetCurrentA (void) const;

  /**
   * \param destState Modem state to switch to.
   * \return True if the transition is allowed.
   *
   * This function checks if a given modem state transition is allowed.
   */
  bool IsStateTransitionValid (const int destState);

  /**
   * \param state New state the modem is currently in.
   *
   * Sets current state. This function is private so that only the energy model
   * can change its own state.
   */
  void SetMicroModemState (const int state);

private:
  Ptr<Node> m_node;            //!< The node hosting this transducer.
  Ptr<EnergySource> m_source;  //!< The energy source.

  // Member variables for power consumption in different modem states.
  double m_txPowerW;           //!< The transmitter power, in watts.
  double m_rxPowerW;           //!< The receiver power, in watts.
  double m_idlePowerW;         //!< The idle power, in watts.
  double m_sleepPowerW;        //!< The sleep power, in watts.

  /** The total energy consumed by this model. */
  TracedValue<double> m_totalEnergyConsumption;

  // State variables.
  int m_currentState;          //!< Current modem state.
  Time m_lastUpdateTime;       //!< Time stamp of previous energy update.

  /** Energy depletion callback. */
  AcousticModemEnergyDepletionCallback m_energyDepletionCallback;
  
  /** Energy recharge callback. */
  AcousticModemEnergyRechargeCallback m_energyRechargeCallback;

};  // class AcousticModemEnergyModel

} // namespace ns3

#endif /* ACOUSTIC_MODEM_ENERGY_MODEL_H */
