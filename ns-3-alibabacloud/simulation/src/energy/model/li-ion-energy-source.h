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

#ifndef LI_ION_ENERGY_SOURCE_H
#define LI_ION_ENERGY_SOURCE_H

#include "ns3/traced-value.h"
#include "ns3/energy-source.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"

namespace ns3 {

/**
 * \ingroup energy
 * \brief Model a generic Lithium Ion Battery basing on [1][2].
 *
 * The model can be fitted to any type of Li-Ion Battery, simply changing the
 * model parameters.
 * The default values are fitted for the Panasonic CGR18650DA Li-Ion Battery [3].
 *
 * The energy is drained as defined from the EnergySource interface but, this class
 * consider the non-linear behaviour of Li-Ion cell. Each time energy is drained from
 * the cell, the class evaluates the discharge curve to get the actual cell's voltage,
 * accordingly to State of Charge (SOC) and current's drain.
 *
 * If the actual voltage of the cell goes below the minimum threshold voltage, the
 * cell is considered depleted and the energy drained event fired up.
 *
 *
 * The model requires several parameters to approximates the discharge curves:
 * - IntialCellVoltage, maximum voltage of the fully charged cell
 * - NominalCellVoltage, nominal cell's voltage, is used to determine the end of the
 *   nominal zone.
 * - ExpCellVoltage, cell's voltage at the end of the exponential zone
 * - RatedCapacity, rated capacity of the cell, in Ah
 * - NomCapacity, cell's capacity at the end of the nominal zone, in Ah
 * - ExpCapacity, cell's capacity at the end of the exponential zone, in Ah
 * - InternalResistance, internal resistance of the cell, in Ohms
 * - TypCurrent, typical discharge current value, used during the fitting process, in Ah
 * - ThresholdVoltage, minimum threshold voltage below which the cell is considered
 *   depleted
 *
 * For a complete reference of the energy source model and model's fitting please refer
 * to <a href="http://www.nsnam.org/wiki/GSOC2010UANFramework">UAN Framework</a>
 * page and <a href="http://www.nsnam.org/wiki/Li-Ion_model_fitting">Li-Ion model
 * fitting</a> page.
 *
 * References:
 * [1] C. M. Shepherd, "Design of Primary and Secondary Cells - Part 3. Battery discharge equation," U.S. Naval Research Laboratory, 1963
 * [2] Tremblay, O.; Dessaint, L.-A.; Dekkiche, A.-I., "A Generic Battery Model for the Dynamic Simulation of Hybrid Electric Vehicles," Ecole de Technologie Superieure, Universite du Quebec, 2007
 * [3] http://www.panasonic.com/industrial/includes/pdf/Panasonic_LiIon_CGR18650DA.pdf
 *
 */
class LiIonEnergySource : public EnergySource
{
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  LiIonEnergySource ();
  virtual ~LiIonEnergySource ();

  /**
   * \return Initial energy stored in energy source, in Joules.
   *
   * Implements GetInitialEnergy.
   */
  virtual double GetInitialEnergy (void) const;

  /**
   * \param initialEnergyJ Initial energy, in Joules
   *
   * Implements SetInitialEnergy. Note that initial energy is assumed to be set
   * before simulation starts and is set only once per simulation.
   */
  void SetInitialEnergy (double initialEnergyJ);

  /**
   * \returns Supply voltage at the energy source.
   *
   * Implements GetSupplyVoltage.
   */
  virtual double GetSupplyVoltage (void) const;

  /**
   * \param supplyVoltageV Initial Supply voltage at the energy source, in Volts.
   *
   * Sets the initial supply voltage of the energy source.
   * To be called only once.
   */
  void SetInitialSupplyVoltage (double supplyVoltageV);

  /**
   * \return Remaining energy in energy source, in Joules
   *
   * Implements GetRemainingEnergy.
   */
  virtual double GetRemainingEnergy (void);

  /**
   * \returns Energy fraction.
   *
   * Implements GetEnergyFraction.
   */
  virtual double GetEnergyFraction (void);

  /**
   * \param energyJ Amount of energy (in Joules) to decrease from energy source.
   *
   * Implements DecreaseRemainingEnergy.
   */
  virtual void DecreaseRemainingEnergy (double energyJ);

  /**
   * \param energyJ Amount of energy (in Joules) to increase from energy source.
   *
   * Implements IncreaseRemainingEnergy.
   */
  virtual void IncreaseRemainingEnergy (double energyJ);

  /**
   * Implements UpdateEnergySource.
   */
  virtual void UpdateEnergySource (void);

  /**
   * \param interval Energy update interval.
   *
   * This function sets the interval between each energy update.
   */
  void SetEnergyUpdateInterval (Time interval);

  /**
   * \returns The interval between each energy update.
   */
  Time GetEnergyUpdateInterval (void) const;
private:
  void DoInitialize (void);
  void DoDispose (void);

  /**
   * Handles the remaining energy going to zero event. This function notifies
   * all the energy models aggregated to the node about the energy being
   * depleted. Each energy model is then responsible for its own handler.
   */
  void HandleEnergyDrainedEvent (void);

  /**
   * Calculates remaining energy. This function uses the total current from all
   * device models to calculate the amount of energy to decrease. The energy to
   * decrease is given by:
   *    energy to decrease = total current * supply voltage * time duration
   * This function subtracts the calculated energy to decrease from remaining
   * energy.
   */
  void CalculateRemainingEnergy (void);

  /**
   *  \param current the actual discharge current value.
   *  \return the cell voltage 
   *
   *  Get the cell voltage in function of the discharge current.
   *  It consider different discharge curves for different discharge currents
   *  and the remaining energy of the cell.
   */
  double GetVoltage (double current) const;

private:
  double m_initialEnergyJ;                //!< initial energy, in Joules
  TracedValue<double> m_remainingEnergyJ; //!< remaining energy, in Joules
  double m_drainedCapacity;               //!< capacity drained from the cell, in Ah
  double m_supplyVoltageV;                //!< actual voltage of the cell
  double m_lowBatteryTh;                  //!< low battery threshold, as a fraction of the initial energy
  EventId m_energyUpdateEvent;            //!< energy update event
  Time m_lastUpdateTime;                  //!< last update time
  Time m_energyUpdateInterval;            //!< energy update interval
  double m_eFull;                         //!< initial voltage of the cell, in Volts
  double m_eNom;                          //!< nominal voltage of the cell, in Volts
  double m_eExp;                          //!< cell voltage at the end of the exponential zone, in Volts
  double m_internalResistance;            //!< internal resistance of the cell, in Ohms
  double m_qRated;                        //!< rated capacity of the cell, in Ah
  double m_qNom;                          //!< cell capacity at the end of the nominal zone, in Ah
  double m_qExp;                          //!< capacity value at the end of the exponential zone, in Ah
  double m_typCurrent;                    //!< typical discharge current used to fit the curves
  double m_minVoltTh;                     //!< minimum threshold voltage to consider the battery depleted
};

} // namespace ns3

#endif /* LI_ION_ENERGY_SOURCE_H */
