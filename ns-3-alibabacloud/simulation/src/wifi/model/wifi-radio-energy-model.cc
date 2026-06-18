/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Network Security Lab, University of Washington, Seattle.
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
 * Author: Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/pointer.h"
#include "ns3/energy-source.h"
#include "wifi-radio-energy-model.h"
#include "wifi-tx-current-model.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiRadioEnergyModel");

NS_OBJECT_ENSURE_REGISTERED (WifiRadioEnergyModel);

TypeId
WifiRadioEnergyModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiRadioEnergyModel")
    .SetParent<DeviceEnergyModel> ()
    .SetGroupName ("Energy")
    .AddConstructor<WifiRadioEnergyModel> ()
    .AddAttribute ("IdleCurrentA",
                   "The default radio Idle current in Ampere.",
                   DoubleValue (0.273),  // idle mode = 273mA
                   MakeDoubleAccessor (&WifiRadioEnergyModel::SetIdleCurrentA,
                                       &WifiRadioEnergyModel::GetIdleCurrentA),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("CcaBusyCurrentA",
                   "The default radio CCA Busy State current in Ampere.",
                   DoubleValue (0.273),  // default to be the same as idle mode
                   MakeDoubleAccessor (&WifiRadioEnergyModel::SetCcaBusyCurrentA,
                                       &WifiRadioEnergyModel::GetCcaBusyCurrentA),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxCurrentA",
                   "The radio TX current in Ampere.",
                   DoubleValue (0.380),    // transmit at 0dBm = 380mA
                   MakeDoubleAccessor (&WifiRadioEnergyModel::SetTxCurrentA,
                                       &WifiRadioEnergyModel::GetTxCurrentA),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("RxCurrentA",
                   "The radio RX current in Ampere.",
                   DoubleValue (0.313),    // receive mode = 313mA
                   MakeDoubleAccessor (&WifiRadioEnergyModel::SetRxCurrentA,
                                       &WifiRadioEnergyModel::GetRxCurrentA),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("SwitchingCurrentA",
                   "The default radio Channel Switch current in Ampere.",
                   DoubleValue (0.273),  // default to be the same as idle mode
                   MakeDoubleAccessor (&WifiRadioEnergyModel::SetSwitchingCurrentA,
                                       &WifiRadioEnergyModel::GetSwitchingCurrentA),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("SleepCurrentA",
                   "The radio Sleep current in Ampere.",
                   DoubleValue (0.033),  // sleep mode = 33mA
                   MakeDoubleAccessor (&WifiRadioEnergyModel::SetSleepCurrentA,
                                       &WifiRadioEnergyModel::GetSleepCurrentA),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxCurrentModel", "A pointer to the attached TX current model.",
                   PointerValue (),
                   MakePointerAccessor (&WifiRadioEnergyModel::m_txCurrentModel),
                   MakePointerChecker<WifiTxCurrentModel> ())
    .AddTraceSource ("TotalEnergyConsumption",
                     "Total energy consumption of the radio device.",
                     MakeTraceSourceAccessor (&WifiRadioEnergyModel::m_totalEnergyConsumption),
                     "ns3::TracedValueCallback::Double")
  ;
  return tid;
}

WifiRadioEnergyModel::WifiRadioEnergyModel ()
  : m_source (0),
    m_currentState (WifiPhyState::IDLE),
    m_lastUpdateTime (Seconds (0.0)),
    m_nPendingChangeState (0)
{
  NS_LOG_FUNCTION (this);
  m_energyDepletionCallback.Nullify ();
  // set callback for WifiPhy listener
  m_listener = new WifiRadioEnergyModelPhyListener;
  m_listener->SetChangeStateCallback (MakeCallback (&DeviceEnergyModel::ChangeState, this));
  // set callback for updating the TX current
  m_listener->SetUpdateTxCurrentCallback (MakeCallback (&WifiRadioEnergyModel::SetTxCurrentFromModel, this));
}

WifiRadioEnergyModel::~WifiRadioEnergyModel ()
{
  NS_LOG_FUNCTION (this);
  m_txCurrentModel = 0;
  delete m_listener;
}

void
WifiRadioEnergyModel::SetEnergySource (const Ptr<EnergySource> source)
{
  NS_LOG_FUNCTION (this << source);
  NS_ASSERT (source != NULL);
  m_source = source;
  m_switchToOffEvent.Cancel ();
  Time durationToOff = GetMaximumTimeInState (m_currentState);
  m_switchToOffEvent = Simulator::Schedule (durationToOff, &WifiRadioEnergyModel::ChangeState, this, WifiPhyState::OFF);
}

double
WifiRadioEnergyModel::GetTotalEnergyConsumption (void) const
{
  NS_LOG_FUNCTION (this);

  Time duration = Simulator::Now () - m_lastUpdateTime;
  NS_ASSERT (duration.IsPositive ()); // check if duration is valid

  // energy to decrease = current * voltage * time
  double supplyVoltage = m_source->GetSupplyVoltage ();
  double energyToDecrease = duration.GetSeconds () * GetStateA (m_currentState) * supplyVoltage;

  // notify energy source
  m_source->UpdateEnergySource ();

  return m_totalEnergyConsumption + energyToDecrease;
}

double
WifiRadioEnergyModel::GetIdleCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_idleCurrentA;
}

void
WifiRadioEnergyModel::SetIdleCurrentA (double idleCurrentA)
{
  NS_LOG_FUNCTION (this << idleCurrentA);
  m_idleCurrentA = idleCurrentA;
}

double
WifiRadioEnergyModel::GetCcaBusyCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ccaBusyCurrentA;
}

void
WifiRadioEnergyModel::SetCcaBusyCurrentA (double CcaBusyCurrentA)
{
  NS_LOG_FUNCTION (this << CcaBusyCurrentA);
  m_ccaBusyCurrentA = CcaBusyCurrentA;
}

double
WifiRadioEnergyModel::GetTxCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_txCurrentA;
}

void
WifiRadioEnergyModel::SetTxCurrentA (double txCurrentA)
{
  NS_LOG_FUNCTION (this << txCurrentA);
  m_txCurrentA = txCurrentA;
}

double
WifiRadioEnergyModel::GetRxCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_rxCurrentA;
}

void
WifiRadioEnergyModel::SetRxCurrentA (double rxCurrentA)
{
  NS_LOG_FUNCTION (this << rxCurrentA);
  m_rxCurrentA = rxCurrentA;
}

double
WifiRadioEnergyModel::GetSwitchingCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_switchingCurrentA;
}

void
WifiRadioEnergyModel::SetSwitchingCurrentA (double switchingCurrentA)
{
  NS_LOG_FUNCTION (this << switchingCurrentA);
  m_switchingCurrentA = switchingCurrentA;
}

double
WifiRadioEnergyModel::GetSleepCurrentA (void) const
{
  NS_LOG_FUNCTION (this);
  return m_sleepCurrentA;
}

void
WifiRadioEnergyModel::SetSleepCurrentA (double sleepCurrentA)
{
  NS_LOG_FUNCTION (this << sleepCurrentA);
  m_sleepCurrentA = sleepCurrentA;
}

WifiPhyState
WifiRadioEnergyModel::GetCurrentState (void) const
{
  NS_LOG_FUNCTION (this);
  return m_currentState;
}

void
WifiRadioEnergyModel::SetEnergyDepletionCallback (
  WifiRadioEnergyDepletionCallback callback)
{
  NS_LOG_FUNCTION (this);
  if (callback.IsNull ())
    {
      NS_LOG_DEBUG ("WifiRadioEnergyModel:Setting NULL energy depletion callback!");
    }
  m_energyDepletionCallback = callback;
}

void
WifiRadioEnergyModel::SetEnergyRechargedCallback (
  WifiRadioEnergyRechargedCallback callback)
{
  NS_LOG_FUNCTION (this);
  if (callback.IsNull ())
    {
      NS_LOG_DEBUG ("WifiRadioEnergyModel:Setting NULL energy recharged callback!");
    }
  m_energyRechargedCallback = callback;
}

void
WifiRadioEnergyModel::SetTxCurrentModel (const Ptr<WifiTxCurrentModel> model)
{
  m_txCurrentModel = model;
}

void
WifiRadioEnergyModel::SetTxCurrentFromModel (double txPowerDbm)
{
  if (m_txCurrentModel)
    {
      m_txCurrentA = m_txCurrentModel->CalcTxCurrent (txPowerDbm);
    }
}

Time
WifiRadioEnergyModel::GetMaximumTimeInState (int state) const
{
  if (state == WifiPhyState::OFF)
    {
      NS_FATAL_ERROR ("Requested maximum remaining time for OFF state");
    }
  double remainingEnergy = m_source->GetRemainingEnergy ();
  double supplyVoltage = m_source->GetSupplyVoltage ();
  double current = GetStateA (state);
  return Seconds (remainingEnergy / (current * supplyVoltage));
}

void
WifiRadioEnergyModel::ChangeState (int newState)
{
  NS_LOG_FUNCTION (this << newState);

  m_nPendingChangeState++;

  if (m_nPendingChangeState > 1 && newState == WifiPhyState::OFF)
    {
      SetWifiRadioState ((WifiPhyState) newState);
      m_nPendingChangeState--;
      return;
    }

  if (newState != WifiPhyState::OFF)
    {
      m_switchToOffEvent.Cancel ();
      Time durationToOff = GetMaximumTimeInState (newState);
      m_switchToOffEvent = Simulator::Schedule (durationToOff, &WifiRadioEnergyModel::ChangeState, this, WifiPhyState::OFF);
    }

  Time duration = Simulator::Now () - m_lastUpdateTime;
  NS_ASSERT (duration.IsPositive ()); // check if duration is valid

  // energy to decrease = current * voltage * time
  double supplyVoltage = m_source->GetSupplyVoltage ();
  double energyToDecrease = duration.GetSeconds () * GetStateA (m_currentState) * supplyVoltage;

  // update total energy consumption
  m_totalEnergyConsumption += energyToDecrease;
  NS_ASSERT (m_totalEnergyConsumption <= m_source->GetInitialEnergy ());

  // update last update time stamp
  m_lastUpdateTime = Simulator::Now ();

  // notify energy source
  m_source->UpdateEnergySource ();

  // in case the energy source is found to be depleted during the last update, a callback might be
  // invoked that might cause a change in the Wifi PHY state (e.g., the PHY is put into SLEEP mode).
  // This in turn causes a new call to this member function, with the consequence that the previous
  // instance is resumed after the termination of the new instance. In particular, the state set
  // by the previous instance is erroneously the final state stored in m_currentState. The check below
  // ensures that previous instances do not change m_currentState.

  if (m_nPendingChangeState <= 1 && m_currentState != WifiPhyState::OFF)
    {
      // update current state & last update time stamp
      SetWifiRadioState ((WifiPhyState) newState);

      // some debug message
      NS_LOG_DEBUG ("WifiRadioEnergyModel:Total energy consumption is " <<
                    m_totalEnergyConsumption << "J");
    }

  m_nPendingChangeState--;
}

void
WifiRadioEnergyModel::HandleEnergyDepletion (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("WifiRadioEnergyModel:Energy is depleted!");
  // invoke energy depletion callback, if set.
  if (!m_energyDepletionCallback.IsNull ())
    {
      m_energyDepletionCallback ();
    }
}

void
WifiRadioEnergyModel::HandleEnergyRecharged (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("WifiRadioEnergyModel:Energy is recharged!");
  // invoke energy recharged callback, if set.
  if (!m_energyRechargedCallback.IsNull ())
    {
      m_energyRechargedCallback ();
    }
}

void
WifiRadioEnergyModel::HandleEnergyChanged (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("WifiRadioEnergyModel:Energy is changed!");
  if (m_currentState != WifiPhyState::OFF)
    {
      m_switchToOffEvent.Cancel ();
      Time durationToOff = GetMaximumTimeInState (m_currentState);
      m_switchToOffEvent = Simulator::Schedule (durationToOff, &WifiRadioEnergyModel::ChangeState, this, WifiPhyState::OFF);
    }
}

WifiRadioEnergyModelPhyListener *
WifiRadioEnergyModel::GetPhyListener (void)
{
  NS_LOG_FUNCTION (this);
  return m_listener;
}

/*
 * Private functions start here.
 */

void
WifiRadioEnergyModel::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_source = NULL;
  m_energyDepletionCallback.Nullify ();
}

double
WifiRadioEnergyModel::GetStateA (int state) const
{
  switch (state)
    {
    case WifiPhyState::IDLE:
      return m_idleCurrentA;
    case WifiPhyState::CCA_BUSY:
      return m_ccaBusyCurrentA;
    case WifiPhyState::TX:
      return m_txCurrentA;
    case WifiPhyState::RX:
      return m_rxCurrentA;
    case WifiPhyState::SWITCHING:
      return m_switchingCurrentA;
    case WifiPhyState::SLEEP:
      return m_sleepCurrentA;
    case WifiPhyState::OFF:
      return 0.0;
    }
  NS_FATAL_ERROR ("WifiRadioEnergyModel: undefined radio state " << state);
}

double
WifiRadioEnergyModel::DoGetCurrentA (void) const
{
  return GetStateA (m_currentState);
}

void
WifiRadioEnergyModel::SetWifiRadioState (const WifiPhyState state)
{
  NS_LOG_FUNCTION (this << state);
  m_currentState = state;
  std::string stateName;
  switch (state)
    {
    case WifiPhyState::IDLE:
      stateName = "IDLE";
      break;
    case WifiPhyState::CCA_BUSY:
      stateName = "CCA_BUSY";
      break;
    case WifiPhyState::TX:
      stateName = "TX";
      break;
    case WifiPhyState::RX:
      stateName = "RX";
      break;
    case WifiPhyState::SWITCHING:
      stateName = "SWITCHING";
      break;
    case WifiPhyState::SLEEP:
      stateName = "SLEEP";
      break;
    case WifiPhyState::OFF:
      stateName = "OFF";
      break;
    }
  NS_LOG_DEBUG ("WifiRadioEnergyModel:Switching to state: " << stateName <<
                " at time = " << Simulator::Now ());
}

// -------------------------------------------------------------------------- //

WifiRadioEnergyModelPhyListener::WifiRadioEnergyModelPhyListener ()
{
  NS_LOG_FUNCTION (this);
  m_changeStateCallback.Nullify ();
  m_updateTxCurrentCallback.Nullify ();
}

WifiRadioEnergyModelPhyListener::~WifiRadioEnergyModelPhyListener ()
{
  NS_LOG_FUNCTION (this);
}

void
WifiRadioEnergyModelPhyListener::SetChangeStateCallback (DeviceEnergyModel::ChangeStateCallback callback)
{
  NS_LOG_FUNCTION (this << &callback);
  NS_ASSERT (!callback.IsNull ());
  m_changeStateCallback = callback;
}

void
WifiRadioEnergyModelPhyListener::SetUpdateTxCurrentCallback (UpdateTxCurrentCallback callback)
{
  NS_LOG_FUNCTION (this << &callback);
  NS_ASSERT (!callback.IsNull ());
  m_updateTxCurrentCallback = callback;
}

void
WifiRadioEnergyModelPhyListener::NotifyRxStart (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::RX);
  m_switchToIdleEvent.Cancel ();
}

void
WifiRadioEnergyModelPhyListener::NotifyRxEndOk (void)
{
  NS_LOG_FUNCTION (this);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::IDLE);
}

void
WifiRadioEnergyModelPhyListener::NotifyRxEndError (void)
{
  NS_LOG_FUNCTION (this);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::IDLE);
}

void
WifiRadioEnergyModelPhyListener::NotifyTxStart (Time duration, double txPowerDbm)
{
  NS_LOG_FUNCTION (this << duration << txPowerDbm);
  if (m_updateTxCurrentCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Update tx current callback not set!");
    }
  m_updateTxCurrentCallback (txPowerDbm);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::TX);
  // schedule changing state back to IDLE after TX duration
  m_switchToIdleEvent.Cancel ();
  m_switchToIdleEvent = Simulator::Schedule (duration, &WifiRadioEnergyModelPhyListener::SwitchToIdle, this);
}

void
WifiRadioEnergyModelPhyListener::NotifyMaybeCcaBusyStart (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::CCA_BUSY);
  // schedule changing state back to IDLE after CCA_BUSY duration
  m_switchToIdleEvent.Cancel ();
  m_switchToIdleEvent = Simulator::Schedule (duration, &WifiRadioEnergyModelPhyListener::SwitchToIdle, this);
}

void
WifiRadioEnergyModelPhyListener::NotifySwitchingStart (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::SWITCHING);
  // schedule changing state back to IDLE after CCA_BUSY duration
  m_switchToIdleEvent.Cancel ();
  m_switchToIdleEvent = Simulator::Schedule (duration, &WifiRadioEnergyModelPhyListener::SwitchToIdle, this);
}

void
WifiRadioEnergyModelPhyListener::NotifySleep (void)
{
  NS_LOG_FUNCTION (this);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::SLEEP);
  m_switchToIdleEvent.Cancel ();
}

void
WifiRadioEnergyModelPhyListener::NotifyWakeup (void)
{
  NS_LOG_FUNCTION (this);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::IDLE);
}

void
WifiRadioEnergyModelPhyListener::NotifyOff (void)
{
  NS_LOG_FUNCTION (this);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::OFF);
  m_switchToIdleEvent.Cancel ();
}

void
WifiRadioEnergyModelPhyListener::NotifyOn (void)
{
  NS_LOG_FUNCTION (this);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::IDLE);
}

void
WifiRadioEnergyModelPhyListener::SwitchToIdle (void)
{
  NS_LOG_FUNCTION (this);
  if (m_changeStateCallback.IsNull ())
    {
      NS_FATAL_ERROR ("WifiRadioEnergyModelPhyListener:Change state callback not set!");
    }
  m_changeStateCallback (WifiPhyState::IDLE);
}

} // namespace ns3
