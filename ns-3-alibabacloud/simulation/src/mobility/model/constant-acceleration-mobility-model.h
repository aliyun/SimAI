/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Gustavo Carneiro  <gjc@inescporto.pt>
 */
#ifndef CONSTANT_ACCELERATION_MOBILITY_MODEL_H
#define CONSTANT_ACCELERATION_MOBILITY_MODEL_H

#include "mobility-model.h"
#include "ns3/nstime.h"

namespace ns3 {

/**
 * \ingroup mobility
 *
 * \brief Mobility model for which the current acceleration does not change once it has been set and until it is set again explicitly to a new value.
 */
class ConstantAccelerationMobilityModel : public MobilityModel 
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * Create position located at coordinates (0,0,0) with
   * speed (0,0,0).
   */
  ConstantAccelerationMobilityModel ();
  virtual ~ConstantAccelerationMobilityModel ();
  /**
   * Set the model's velocity and acceleration
   * \param velocity the velocity (m/s)
   * \param acceleration the acceleration (m/s^2)
   */
  void SetVelocityAndAcceleration (const Vector &velocity, const Vector &acceleration);

private:
  virtual Vector DoGetPosition (void) const;
  virtual void DoSetPosition (const Vector &position);
  virtual Vector DoGetVelocity (void) const;

  Time m_baseTime;  //!< the base time
  Vector m_basePosition; //!< the base position
  Vector m_baseVelocity; //!< the base velocity
  Vector m_acceleration;  //!< the acceleration
};

} // namespace ns3

#endif /* CONSTANT_ACCELERATION_MOBILITY_MODEL_H */
