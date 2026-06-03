/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
#ifndef RANDOM_DIRECTION_MOBILITY_MODEL_H
#define RANDOM_DIRECTION_MOBILITY_MODEL_H

#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/rectangle.h"
#include "ns3/random-variable-stream.h"
#include "mobility-model.h"
#include "constant-velocity-helper.h"

namespace ns3 {

/**
 * \ingroup mobility
 * \brief Random direction mobility model.
 *
 * The movement of objects is based on random directions: each object
 * pauses for a specific delay, chooses a random direction and speed and 
 * then travels in the specific direction until it reaches one of
 * the boundaries of the model. When it reaches the boundary, it pauses,
 * selects a new direction and speed, aso.
 */
class RandomDirection2dMobilityModel : public MobilityModel
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  RandomDirection2dMobilityModel ();

private:
  /**
   * Set a new direction and speed
   */
  void ResetDirectionAndSpeed (void);
  /**
   * Pause, cancel currently scheduled event, schedule end of pause event
   */
  void BeginPause (void);
  /**
   * Set new velocity and direction, and schedule next pause event  
   * \param direction (radians)
   */
  void SetDirectionAndSpeed (double direction);
  /**
   * Sets a new random direction and calls SetDirectionAndSpeed
   */
  void DoInitializePrivate (void);
  virtual void DoDispose (void);
  virtual void DoInitialize (void);
  virtual Vector DoGetPosition (void) const;
  virtual void DoSetPosition (const Vector &position);
  virtual Vector DoGetVelocity (void) const;
  virtual int64_t DoAssignStreams (int64_t);

  Ptr<UniformRandomVariable> m_direction; //!< rv to control direction
  Rectangle m_bounds; //!< the 2D bounding area
  Ptr<RandomVariableStream> m_speed; //!< a random variable to control speed
  Ptr<RandomVariableStream> m_pause; //!< a random variable to control pause 
  EventId m_event; //!< event ID of next scheduled event
  ConstantVelocityHelper m_helper; //!< helper for velocity computations
};

} // namespace ns3

#endif /* RANDOM_DIRECTION_MOBILITY_MODEL_H */
