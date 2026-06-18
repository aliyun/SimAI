/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef ISOTROPIC_ANTENNA_MODEL_H
#define ISOTROPIC_ANTENNA_MODEL_H


#include <ns3/object.h>
#include <ns3/antenna-model.h>

namespace ns3 {

/**
 * \ingroup antenna
 * 
 * \brief Isotropic antenna model
 * 
 * This is the simplest antenna model. The gain of this antenna is the same in all directions.
 */
class IsotropicAntennaModel : public AntennaModel
{
public:

  IsotropicAntennaModel ();
  
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId ();

  // inherited from AntennaModel
  virtual double GetGainDb (Angles a);

protected:

  /**
   * gain of the antenna in dB, in all directions
   * 
   */
  double m_gainDb;
};



} // namespace ns3


#endif // ISOTROPIC_ANTENNA_MODEL_H
