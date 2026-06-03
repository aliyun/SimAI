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
#ifndef HIERARCHICAL_MOBILITY_MODEL_H
#define HIERARCHICAL_MOBILITY_MODEL_H

#include "mobility-model.h"

namespace ns3 {

/**
 * \ingroup mobility
 * \brief Hierarchical mobility model.
 * 
 * This model allows you to specify the position of a child object
 * relative to a parent object.
 * 
 * Basically this is a mobility model that combines two other mobility
 * models: a "parent" model and a "child" model.  The position of the
 * hierarchical model is always the vector sum of the parent + child
 * positions, so that if the parent model "moves", then this model
 * will report an equal relative movement.  Useful, for instance, if
 * you want to simulate a node inside another node that moves, such as
 * a vehicle.
 *
 * Setting the position on this model is always done using world
 * absolute coordinates, and it changes only the child mobility model
 * position, never the parent.  The child mobility model always uses a
 * coordinate system relative to the parent model position.
 *
 * \note: as a special case, the parent model may be NULL, which is
 * semantically equivalent to having a ConstantPositionMobilityModel
 * as parent positioned at origin (0,0,0).  In other words, setting
 * the parent model to NULL makes the child model and the hierarchical
 * model start using world absolute coordinates.
 *
 * \warning: changing the parent/child mobility models in the middle
 * of a simulation will probably not play very well with the
 * ConfigStore APIs, so do this only if you know what you are doing.
 */
class HierarchicalMobilityModel : public MobilityModel
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  HierarchicalMobilityModel ();

  /**
   * \return the child mobility model.
   *
   * Calling GetPosition() on the model returned by this method allows
   * one to access the position of the child relative to its parent.
   */
  Ptr<MobilityModel> GetChild (void) const;
  /**
   * \return the parent mobility model.
   *
   * Calling GetPosition() on the model returned by this method allows
   * one to access the position of the parent alone, which is used
   * as the reference position to which the child position is added.
   */
  Ptr<MobilityModel> GetParent (void) const;
  /**
   * Sets the child mobility model to a new one, possibly replacing
   * an existing one.  If the child model is being replaced,
   * then the new child mobility model's current position is also set to
   * the previous position to ensure that the composite
   * position is preserved by this operation.
   * \param model new mobility model child
   */
  void SetChild (Ptr<MobilityModel> model);
  /**
   * Sets the parent mobility model to a new one, possibly replacing
   * an existing one.  If the parent model is being replaced,
   * then the new position is set to the position that was set before
   * replacement, to ensure that the composite position is preserved
   * across changes to the parent model.
   * \param model new mobility model parent
   */
  void SetParent (Ptr<MobilityModel> model);

private:
  virtual Vector DoGetPosition (void) const;
  virtual void DoSetPosition (const Vector &position);
  virtual Vector DoGetVelocity (void) const;
  virtual void DoInitialize (void);
  virtual int64_t DoAssignStreams (int64_t);

  /**
   * Callback for when parent mobility model course change occurs
   * \param model mobility mode (unused)
   */
  void ParentChanged (Ptr<const MobilityModel> model);
  /**
   * Callback for when child mobility model course change occurs
   * \param model mobility mode (unused)
   */
  void ChildChanged (Ptr<const MobilityModel> model);

  Ptr<MobilityModel> m_child; //!< pointer to child mobility model
  Ptr<MobilityModel> m_parent; //!< pointer to parent mobility model
};


} // namespace ns3

#endif /* HIERARCHICAL_MOBILITY_MODEL_H */
