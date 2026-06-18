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

#include "li-ion-energy-source.h"
#include "ns3/log.h"
#include "ns3/assert.h"
#include "ns3/double.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/simulator.h"

#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LiIonEnergySource");

NS_OBJECT_ENSURE_REGISTERED (LiIonEnergySource);

TypeId
LiIonEnergySource::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LiIonEnergySource")
    .SetParent<EnergySource> ()
    .SetGroupName ("Energy")
    .AddConstructor<LiIonEnergySource> ()
    .AddAttribute ("LiIonEnergySourceInitialEnergyJ",
                   "Initial energy stored in basic energy source.",
                   DoubleValue (31752.0),  // in Joules
                   MakeDoubleAccessor (&LiIonEnergySource::SetInitialEnergy,
                                       &LiIonEnergySource::GetInitialEnergy),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("LiIonEnergyLowBatteryThreshold",
                   "Low battery threshold for LiIon energy source.",
                   DoubleValue (0.10), // as a fraction of the initial energy
                   MakeDoubleAccessor (&LiIonEnergySource::m_lowBatteryTh),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("InitialCellVoltage",
                   "Initial (maximum) voltage of the cell (fully charged).",
                   DoubleValue (4.05), // in Volts
                   MakeDoubleAccessor (&LiIonEnergySource::SetInitialSupplyVoltage,
                                       &LiIonEnergySource::GetSupplyVoltage),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("NominalCellVoltage",
                   "Nominal voltage of the cell.",
                   DoubleValue (3.6),  // in Volts
                   MakeDoubleAccessor (&LiIonEnergySource::m_eNom),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ExpCellVoltage",
                   "Cell voltage at the end of the exponential zone.",
                   DoubleValue (3.6),  // in Volts
                   MakeDoubleAccessor (&LiIonEnergySource::m_eExp),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("RatedCapacity",
                   "Rated capacity of the cell.",
                   DoubleValue (2.45),   // in Ah
                   MakeDoubleAccessor (&LiIonEnergySource::m_qRated),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("NomCapacity",
                   "Cell capacity at the end of the nominal zone.",
                   DoubleValue (1.1),  // in Ah
                   MakeDoubleAccessor (&LiIonEnergySource::m_qNom),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ExpCapacity",
                   "Cell Capacity at the end of the exponential zone.",
                   DoubleValue (1.2),  // in Ah
                   MakeDoubleAccessor (&LiIonEnergySource::m_qExp),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("InternalResistance",
                   "Internal resistance of the cell",
                   DoubleValue (0.083),  // in Ohms
                   MakeDoubleAccessor (&LiIonEnergySource::m_internalResistance),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TypCurrent",
                   "Typical discharge current used to fit the curves",
                   DoubleValue (2.33), // in A
                   MakeDoubleAccessor (&LiIonEnergySource::m_typCurrent),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ThresholdVoltage",
                   "Minimum threshold voltage to consider the battery depleted.",
                   DoubleValue (3.3), // in Volts
                   MakeDoubleAccessor (&LiIonEnergySource::m_minVoltTh),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("PeriodicEnergyUpdateInterval",
                   "Time between two consecutive periodic energy updates.",
                   TimeValue (Seconds (1.0)),
                   MakeTimeAccessor (&LiIonEnergySource::SetEnergyUpdateInterval,
                                     &LiIonEnergySource::GetEnergyUpdateInterval),
                   MakeTimeChecker ())
    .AddTraceSource ("RemainingEnergy",
                     "Remaining energy at BasicEnergySource.",
                     MakeTraceSourceAccessor (&LiIonEnergySource::m_remainingEnergyJ),
                     "ns3::TracedValueCallback::Double")
  ;
  return tid;
}

LiIonEnergySource::LiIonEnergySource ()
  : m_drainedCapacity (0.0),
    m_lastUpdateTime (Seconds (0.0))
{
  NS_LOG_FUNCTION (this);
}

LiIonEnergySource::~LiIonEnergySource ()
{
  NS_LOG_FUNCTION (this);
}

void
LiIonEnergySource::SetInitialEnergy (double initialEnergyJ)
{
  NS_LOG_FUNCTION (this << initialEnergyJ);
  NS_ASSERT (initialEnergyJ >= 0);
  m_initialEnergyJ = initialEnergyJ;
  // set remaining energy to be initial energy
  m_remainingEnergyJ = m_initialEnergyJ;
}

double
LiIonEnergySource::GetInitialEnergy (void) const
{
  NS_LOG_FUNCTION (this);
  return m_initialEnergyJ;
}

void
LiIonEnergySource::SetInitialSupplyVoltage (double supplyVoltageV)
{
  NS_LOG_FUNCTION (this << supplyVoltageV);
  m_eFull = supplyVoltageV;
  m_supplyVoltageV = supplyVoltageV;
}

double
LiIonEnergySource::GetSupplyVoltage (void) const
{
  NS_LOG_FUNCTION (this);
  return m_supplyVoltageV;
}

void
LiIonEnergySource::SetEnergyUpdateInterval (Time interval)
{
  NS_LOG_FUNCTION (this << interval);
  m_energyUpdateInterval = interval;
}

Time
LiIonEnergySource::GetEnergyUpdateInterval (void) const
{
  NS_LOG_FUNCTION (this);
  return m_energyUpdateInterval;
}

double
LiIonEnergySource::GetRemainingEnergy (void)
{
  NS_LOG_FUNCTION (this);
  // update energy source to get the latest remaining energy.
  UpdateEnergySource ();
  return m_remainingEnergyJ;
}

double
LiIonEnergySource::GetEnergyFraction (void)
{
  NS_LOG_FUNCTION (this);
  // update energy source to get the latest remaining energy.
  UpdateEnergySource ();
  return m_remainingEnergyJ / m_initialEnergyJ;
}

void
LiIonEnergySource::DecreaseRemainingEnergy (double energyJ)
{
  NS_LOG_FUNCTION (this << energyJ);
  NS_ASSERT (energyJ >= 0);
  m_remainingEnergyJ -= energyJ;

  // check if remaining energy is 0
  if (m_supplyVoltageV <= m_minVoltTh)
    {
      HandleEnergyDrainedEvent ();
    }
}

void
LiIonEnergySource::IncreaseRemainingEnergy (double energyJ)
{
  NS_LOG_FUNCTION (this << energyJ);
  NS_ASSERT (energyJ >= 0);
  m_remainingEnergyJ += energyJ;
}

void
LiIonEnergySource::UpdateEnergySource (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("LiIonEnergySource:Updating remaining energy at node #" <<
                GetNode ()->GetId ());

  // do not update if simulation has finished
  if (Simulator::IsFinished ())
    {
      return;
    }

  m_energyUpdateEvent.Cancel ();

  CalculateRemainingEnergy ();

  m_lastUpdateTime = Simulator::Now ();

  if (m_remainingEnergyJ <= m_lowBatteryTh * m_initialEnergyJ)
    {
      HandleEnergyDrainedEvent ();
      return; // stop periodic update
    }

  m_energyUpdateEvent = Simulator::Schedule (m_energyUpdateInterval,
                                             &LiIonEnergySource::UpdateEnergySource,
                                             this);
}

/*
 * Private functions start here.
 */
void
LiIonEnergySource::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  UpdateEnergySource ();  // start periodic update
}

void
LiIonEnergySource::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  BreakDeviceEnergyModelRefCycle ();  // break reference cycle
}


void
LiIonEnergySource::HandleEnergyDrainedEvent (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("LiIonEnergySource:Energy depleted at node #" <<
                GetNode ()->GetId ());
  NotifyEnergyDrained (); // notify DeviceEnergyModel objects
}


void
LiIonEnergySource::CalculateRemainingEnergy (void)
{
  NS_LOG_FUNCTION (this);
  double totalCurrentA = CalculateTotalCurrent ();
  Time duration = Simulator::Now () - m_lastUpdateTime;
  NS_ASSERT (duration.GetSeconds () >= 0);
  // energy = current * voltage * time
  double energyToDecreaseJ = totalCurrentA * m_supplyVoltageV * duration.GetSeconds ();

  if (m_remainingEnergyJ < energyToDecreaseJ)
    {
      m_remainingEnergyJ = 0; // energy never goes below 0
    } 
  else 
    {
      m_remainingEnergyJ -= energyToDecreaseJ;
    }  

  m_drainedCapacity += (totalCurrentA * duration).GetHours ();
  // update the supply voltage
  m_supplyVoltageV = GetVoltage (totalCurrentA);
  NS_LOG_DEBUG ("LiIonEnergySource:Remaining energy = " << m_remainingEnergyJ);
}

double
LiIonEnergySource::GetVoltage (double i) const
{
  NS_LOG_FUNCTION (this << i);

  // integral of i in dt, drained capacity in Ah
  double it = m_drainedCapacity;

  // empirical factors
  double A = m_eFull - m_eExp;
  double B = 3 / m_qExp;

  // slope of the polarization curve
  double K = std::abs ( (m_eFull - m_eNom + A * (std::exp (-B * m_qNom) - 1)) * (m_qRated - m_qNom) / m_qNom);

  // constant voltage
  double E0 = m_eFull + K + m_internalResistance * m_typCurrent - A;

  double E = E0 - K * m_qRated / (m_qRated - it) + A * std::exp (-B * it);

  // cell voltage
  double V = E - m_internalResistance * i;

  NS_LOG_DEBUG ("Voltage: " << V << " with E: " << E);

  return V;
}

} // namespace ns3
