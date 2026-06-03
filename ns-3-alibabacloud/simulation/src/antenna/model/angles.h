/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 CTTC
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

#ifndef ANGLES_H
#define ANGLES_H


#include <ns3/vector.h>
#include <vector>

namespace ns3 {


/** 
 * \brief converts degrees to radians
 * 
 * \param degrees the angle in degrees
 * \return the angle in radians
 */
double DegreesToRadians (double degrees);

/** 
 * \brief converts degrees to radians
 * 
 * \param degrees the angles in degrees
 * \return the angles in radians
 */
std::vector<double> DegreesToRadians (const std::vector<double> &degrees);

/** 
 * \brief converts radians to degrees
 * 
 * \param radians the angle in radians
 * \return the angle in degrees
 */
double RadiansToDegrees (double radians);

/** 
 * \brief converts radians to degrees
 * 
 * \param radians the angles in radians
 * \return the angles in degrees
 */
std::vector<double> RadiansToDegrees (const std::vector<double> &radians);

/** 
 * \brief Wrap angle in [0, 360)
 * 
 * \param a the angle in degrees
 * \return the wrapped angle in degrees
 */
double WrapTo360 (double a);

/** 
 * \brief Wrap angle in [-180, 180)
 * 
 * \param a the angle in degrees
 * \return the wrapped angle in degrees
 */
double WrapTo180 (double a);

/** 
 * \brief Wrap angle in [0, 2*M_PI)
 * 
 * \param a the angle in radians
 * \return the wrapped angle in radians
 */
double WrapTo2Pi (double a);

/** 
 * \brief Wrap angle in [-M_PI, M_PI)
 * 
 * \param a the angle in radians
 * \return the wrapped angle in radians
 */
double WrapToPi (double a);

/** 
 * 
 * Class holding the azimuth and inclination angles of spherical coordinates.
 * The notation is the one used in  "Antenna Theory - Analysis
 * and Design", C.A. Balanis, Wiley, 2nd Ed., section 2.2 "Radiation pattern".
 * This notation corresponds to the standard spherical coordinates, with azimuth
 * measured counterclockwise in the x-y plane off the x-axis, and
 * inclination measured off the z-axis.
 * Azimuth is consistently normalized to be in [-M_PI, M_PI).
 * 
 *          ^
 *        z | 
 *          |_ inclination
 *          | \
 *          | /|
 *          |/ |   y
 *          +-------->
 *         /  \|
 *        /___/
 *     x /  azimuth
 *      |/
 *
 */
class Angles
{
public:
  /** 
   * This constructor allows to specify azimuth and inclination.
   * Inclination must be in [0, M_PI], while azimuth is
   * automatically notmalized in [-M_PI, M_PI)
   * 
   * \param azimuth the azimuth angle in radians
   * \param inclination the inclination angle in radians
   */
  Angles (double azimuth, double inclination);

  /** 
   * This constructor will initialize azimuth and inclination by converting the
   * given 3D vector from cartesian coordinates to spherical coordinates
   * Note: azimuth and inclination angles for a zero-length vector are not defined
   * and are thus initialized to NAN
   * 
   * \param v the 3D vector in cartesian coordinates
   * 
   */
  Angles (Vector v);

  /** 
   * This constructor initializes an Angles instance with the angles
   * of the spherical coordinates of point v respect to point o 
   * 
   * \param v the point (in cartesian coordinates) for which the angles are determined
   * \param o the origin (in cartesian coordinates) of the spherical coordinate system
   * 
   */
  Angles (Vector v, Vector o);

  /**
   * Setter for azimuth angle
   *
   * \param azimuth angle in radians
   */
  void SetAzimuth (double azimuth);

  /**
   * Setter for inclination angle
   *
   * \param inclination angle in radians. Must be in [0, M_PI]
   */
  void SetInclination (double inclination);

  /**
   * Getter for azimuth angle
   *
   * \return azimuth angle in radians
   */
  double GetAzimuth (void) const;

  /**
   * Getter for inclination angle
   *
   * \return inclination angle in radians
   */
  double GetInclination (void) const;

  // friend methods
  /**
   * \brief Stream insertion operator.
   *
   * \param [in] os The reference to the output stream.
   * \param [in] a The angle.
   * \returns The reference to the output stream.
   */
  friend std::ostream& operator<< (std::ostream& os, const Angles& a);
  /**
   * \brief Stream extraction operator.
   *
   * \param [in] is The reference to the input stream.
   * \param [out] a The angle.
   * \returns The reference to the input stream.
   */
  friend std::istream& operator>> (std::istream& is, Angles& a);

  static bool m_printDeg; //!< flag for printing in radians or degrees units

private:
  /** 
   * Default constructor is disabled
   */
  Angles ();

  /**
    * Normalize the angle azimuth angle range between in [-M_PI, M_PI)
    * while checking if the angle is valid, i.e., finite and within
    * the bounds.
    * 
    * Note: while an arbitrary value for the azimuth angle is valid
    * and can be wrapped in [-M_PI, M_PI), an inclination angle outside
    * the [0, M_PI] range can be ambiguos and is thus not valid.
    */
  void NormalizeAngles (void);

  /**
    * Check if Angle is valid or not
    * Warns the user if the Angle is undefined (non-finite azimuth or inclination),
    * throws an assert if the inclination angle is invalid (not in [0, M_PI])
    */
  void CheckIfValid (void) const;


  double m_azimuth; //!< the azimuth angle in radians
  double m_inclination; //!< the inclination angle in radians
};


} // namespace ns3

#endif // ANGLES_H
