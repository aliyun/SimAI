/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 University of Padova, Dep. of Information Engineering, SIGNET lab.
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
 */

#ifndef THREE_GPP_ANTENNA_MODEL_H
#define THREE_GPP_ANTENNA_MODEL_H


#include <ns3/object.h>
#include <ns3/antenna-model.h>

namespace ns3 {

/**
 *
 * \brief  Antenna model based on a parabolic approximation of the main lobe radiation pattern.
 *
 * This class implements the parabolic model as described in 3GPP TR 38.901 v15.0.0
 *
 */
class ThreeGppAntennaModel : public AntennaModel
{
public:
  ThreeGppAntennaModel (void);
  virtual ~ThreeGppAntennaModel (void) override;

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId ();

  // inherited from AntennaModel
  virtual double GetGainDb (Angles a) override;

  /**
   * Get the vertical beamwidth of the antenna element.
   * \return the vertical beamwidth in degrees
   */
  double GetVerticalBeamwidth () const;

  /**
   * Get the horizontal beamwidth of the antenna element.
   * \return the horizontal beamwidth in degrees
   */
  double GetHorizontalBeamwidth () const;

  /**
   * Get the side-lobe attenuation in the vertical direction of the antenna element.
   * \return side-lobe attenuation in the vertical direction in dB
   */
  double GetSlaV () const;

  /**
   * Get the naximum attenuation of the antenna element.
   * \return the naximum attenuation in dB
   */
  double GetMaxAttenuation () const;

  /**
   * Get the maximum directional gain of the antenna element.
   * \return the maximum directional gain in dBi
   */
  double GetAntennaElementGain () const;

private:
  double m_verticalBeamwidthDegrees;    //!< beamwidth in the vertical direction \f$(\theta_{3dB})\f$ [deg]
  double m_horizontalBeamwidthDegrees;  //!< beamwidth in the horizontal direction \f$(\phi_{3dB})\f$ [deg]
  double m_aMax;                        //!< maximum attenuation (A_{max}) [dB]
  double m_slaV;                        //!< side-lobe attenuation in the vertical direction (SLA_V) [dB]
  double m_geMax;                       //!< maximum directional gain of the antenna element (G_{E,max}) [dBi]
};



} // namespace ns3


#endif // THREE_GPP_ANTENNA_MODEL_H
