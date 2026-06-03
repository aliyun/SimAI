/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006,2007 INRIA
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
#ifndef PROPAGATION_DELAY_MODEL_H
#define PROPAGATION_DELAY_MODEL_H

#include "ns3/ptr.h"
#include "ns3/object.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"

namespace ns3 {

class MobilityModel;

/**
 * \ingroup propagation
 *
 * \brief calculate a propagation delay.
 */
class PropagationDelayModel : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual ~PropagationDelayModel ();
  /**
   * \param a the source
   * \param b the destination
   * \returns the calculated propagation delay
   *
   * Calculate the propagation delay between the specified
   * source and destination.
   */
  virtual Time GetDelay (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const = 0;
  /**
   * If this delay model uses objects of type RandomVariableStream,
   * set the stream numbers to the integers starting with the offset
   * 'stream'.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);
protected:
  /**
   * Assign a fixed random variable stream number to the random variables used by this model.
   * 
   * Subclasses must implement this; those not using random variables
   * can return zero.
   * 
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  virtual int64_t DoAssignStreams (int64_t stream) = 0;
};

/**
 * \ingroup propagation
 *
 * \brief the propagation delay is random
 */
class RandomPropagationDelayModel : public PropagationDelayModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Use the default parameters from PropagationDelayRandomDistribution.
   */
  RandomPropagationDelayModel ();
  virtual ~RandomPropagationDelayModel ();
  Time GetDelay (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override;
private:
  int64_t DoAssignStreams (int64_t stream) override;
  Ptr<RandomVariableStream> m_variable; //!< random generator
};

/**
 * \ingroup propagation
 *
 * \brief the propagation speed is constant
 */
class ConstantSpeedPropagationDelayModel : public PropagationDelayModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Use the default parameters from PropagationDelayConstantSpeed.
   */
  ConstantSpeedPropagationDelayModel ();
  Time GetDelay (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override;
  /**
   * \param speed the new speed (m/s)
   */
  void SetSpeed (double speed);
  /**
   * \returns the current propagation speed (m/s).
   */
  double GetSpeed (void) const;
private:
  int64_t DoAssignStreams (int64_t stream) override;
  double m_speed; //!< speed
};

} // namespace ns3

#endif /* PROPAGATION_DELAY_MODEL_H */
