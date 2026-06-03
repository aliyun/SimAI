/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006,2007 INRIA
 * Copyright (c) 2019 University of Padova
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
 * Author: Michele Polese <michele.polese@gmail.com>
 */

#ifndef RANDOM_WALK_2D_OUTDOOR_MOBILITY_MODEL_H
#define RANDOM_WALK_2D_OUTDOOR_MOBILITY_MODEL_H

#include "ns3/object.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/rectangle.h"
#include "ns3/random-variable-stream.h"
#include "ns3/mobility-model.h"
#include "ns3/constant-velocity-helper.h"
#include "ns3/building.h"

namespace ns3 {


/**
 * \ingroup mobility
 * \brief 2D random walk mobility model which avoids buildings.
 *
 * This class reuses most of the code of RandomWalk2dMobilityModel,
 * but adds the awareness of buildings objects which are avoided
 * by moving users.
 * Each instance moves with a speed and direction chosen at random
 * with the user-provided random variables until
 * either a fixed distance has been walked or until a fixed amount
 * of time. If we hit one of the boundaries (specified by a rectangle)
 * of the model, we rebound on the boundary with a reflexive angle
 * and speed. If we hit one of the buildings, we rebound with a random
 * direction which makes sure that the next step does not enter the building.
 *
 * The default values for the random variable that describes the speed is
 * taken from Figure 1 in the paper:
 * Henderson, L.F., 1971. The statistics of crowd fluids. nature, 229(5284), p.381.
 */
class RandomWalk2dOutdoorMobilityModel : public MobilityModel
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /** An enum representing the different working modes of this module. */
  enum Mode
  {
    MODE_DISTANCE,
    MODE_TIME
  };

private:
  /**
   * \brief Performs the rebound of the node if it reaches a boundary
   * \param timeLeft The remaining time of the walk
   */
  void Rebound (Time timeLeft);
  /**
   * \brief Avoid a building
   * \param delayLeft The remaining time of the walk
   * \param intersectPosition The position at which the building is intersected
   */
  void AvoidBuilding (Time delayLeft, Vector intersectPosition);
  /**
   * Walk according to position and velocity, until distance is reached,
   * time is reached, or intersection with the bounding box, or building
   * \param delayLeft The remaining time of the walk
   */
  void DoWalk (Time delayLeft);
  /**
   * Perform initialization of the object before MobilityModel::DoInitialize ()
   */
  void DoInitializePrivate (void);
  /**
   * Check if there is a building between two positions (or if the nextPosition is inside a building).
   * The code is taken from MmWave3gppBuildingsPropagationLossModel from the NYU/UNIPD ns-3 mmWave module
   * \param currentPosition The current position of the node
   * \param nextPosition The position to check
   * \return a pair with a boolean (true if the line between the two position does not intersect building),
   * and a pointer which is 0 if the boolean is true, or it points to the building which is intersected
   */
  std::pair<bool, Ptr<Building> > IsLineClearOfBuildings (Vector currentPosition, Vector nextPosition ) const;
  /**
   * Compute the intersecting point of the box represented by boundaries and the line between current and next
   * Notice that we only consider a 2d plane
   * \param current The current position
   * \param next The next position
   * \param boundaries The boundaries of the building we will intersect
   * \return a vector with the position of the intersection
   */
  Vector CalculateIntersectionFromOutside (const Vector &current, const Vector &next, const Box boundaries) const;

  virtual void DoDispose (void);
  virtual void DoInitialize (void);
  virtual Vector DoGetPosition (void) const;
  virtual void DoSetPosition (const Vector &position);
  virtual Vector DoGetVelocity (void) const;
  virtual int64_t DoAssignStreams (int64_t);

  ConstantVelocityHelper m_helper; //!< helper for this object
  EventId m_event; //!< stored event ID
  enum Mode m_mode; //!< whether in time or distance mode
  double m_modeDistance; //!< Change direction and speed after this distance
  Time m_modeTime; //!< Change current direction and speed after this delay
  Ptr<RandomVariableStream> m_speed; //!< rv for picking speed
  Ptr<RandomVariableStream> m_direction; //!< rv for picking direction
  Rectangle m_bounds; //!< Bounds of the area to cruise
  double m_epsilon; //!< Tolerance for the intersection point with buildings
  uint32_t m_maxIter; //!< Maximum number of tries to find the next position
  Vector m_prevPosition; //!< Store the previous position in case a step back is needed
};


} // namespace ns3

#endif /* RANDOM_WALK_2D_OUTDOOR_MOBILITY_MODEL_H */