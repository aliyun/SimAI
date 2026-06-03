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

#ifndef ANTENNA_MODEL_H
#define ANTENNA_MODEL_H


#include <ns3/object.h>
#include <ns3/angles.h>

namespace ns3 {


/**
 * \defgroup antenna Antenna module
 * \brief This module contains several antenna
 * radiation pattern models.
 */

/**
 * \ingroup antenna
 * \ingroup tests
 * \defgroup antenna-tests Antenna module tests
 */

/** 
 * \ingroup antenna
 * 
 * \brief interface for antenna radiation pattern models
 * 
 * This class provides an interface for the definition of antenna
 * radiation pattern models. This interface is based on the use of
 * spherical coordinates, in particular of the azimuth and inclination
 * angles. This choice is the one proposed "Antenna Theory - Analysis
 * and Design", C.A. Balanis, Wiley, 2nd Ed., see in particular
 * section 2.2 "Radiation pattern".
 */
class AntennaModel : public Object
{
public:


  AntennaModel ();
  virtual ~AntennaModel ();

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId ();


  /**
   * this method is expected to be re-implemented by each antenna model 
   * 
   * \param a the spherical angles at which the radiation pattern should
   * be evaluated
   * 
   * \return the power gain in dBi of the antenna radiation pattern at
   * the specified angles; dBi means dB with respect to the gain of an
   * isotropic radiator. Since a power gain is used, the efficiency of
   * the antenna is expected to be included in the gain value. 
   */
  virtual double GetGainDb (Angles a) = 0;

};



} // namespace ns3

#endif // ANTENNA_MODEL_H
