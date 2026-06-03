/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006,2007 INRIA
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
#ifndef MOBILITY_MODEL_H
#define MOBILITY_MODEL_H

#include "ns3/vector.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"

namespace ns3 {

/**
 * \ingroup mobility
 * \brief Keep track of the current position and velocity of an object.
 *
 * All space coordinates in this class and its subclasses are
 * understood to be meters or meters/s. i.e., they are all
 * metric international units.
 *
 * This is a base class for all specific mobility models.
 */
class MobilityModel : public Object
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  MobilityModel ();
  virtual ~MobilityModel () = 0;

  /**
   * \return the current position
   */
  Vector GetPosition (void) const;
  /**
   * This method may be used if the position returned may depend on some
   * reference position provided.  For example, in a hierarchical mobility
   * model that is buildings-aware, the child mobility model may not be able
   * to determine if it is inside or outside of a building unless it knows
   * the parent position.
   *
   * \param referencePosition reference position to consider
   * \return the current position based on the provided referencePosition
   * \sa ns3::MobilityModel::DoGetPositionWithReference
   */
  Vector GetPositionWithReference (const Vector &referencePosition) const;
  /**
   * \param position the position to set.
   */
  void SetPosition (const Vector &position);
  /**
   * \return the current velocity.
   */
  Vector GetVelocity (void) const;
  /**
   * \param position a reference to another mobility model
   * \return the distance between the two objects. Unit is meters.
   */
  double GetDistanceFrom (Ptr<const MobilityModel> position) const;
  /**
   * \param other reference to another object's mobility model
   * \return the relative speed between the two objects. Unit is meters/s.
   */
  double GetRelativeSpeed (Ptr<const MobilityModel> other) const;
  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model. Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

  /**
   *  TracedCallback signature.
   *
   * \param [in] model Value of the MobilityModel.
   */
  typedef void (* TracedCallback)(Ptr<const MobilityModel> model);
  
protected:
  /**
   * Must be invoked by subclasses when the course of the
   * position changes to notify course change listeners.
   */
  void NotifyCourseChange (void) const;
private:
  /**
   * \return the current position.
   *
   * Concrete subclasses of this base class must 
   * implement this method.
   */
  virtual Vector DoGetPosition (void) const = 0;
  /**
   * \param referencePosition the reference position to consider
   * \return the current position.
   *
   * Unless subclasses override, this method will disregard the reference
   * position and return "DoGetPosition (void)".
   */
  virtual Vector DoGetPositionWithReference (const Vector& referencePosition) const;
  /**
   * \param position the position to set.
   *
   * Concrete subclasses of this base class must 
   * implement this method.
   */
  virtual void DoSetPosition (const Vector &position) = 0;
  /**
   * \return the current velocity.
   *
   * Concrete subclasses of this base class must 
   * implement this method.
   */
  virtual Vector DoGetVelocity (void) const = 0;
  /**
   * The default implementation does nothing but return the passed-in
   * parameter.  Subclasses using random variables are expected to
   * override this.
   * \param start  starting stream index
   * \return the number of streams used
   */
  virtual int64_t DoAssignStreams (int64_t start);

  /**
   * Used to alert subscribers that a change in direction, velocity,
   * or position has occurred.
   */
  ns3::TracedCallback<Ptr<const MobilityModel> > m_courseChangeTrace;

};

} // namespace ns3

#endif /* MOBILITY_MODEL_H */
