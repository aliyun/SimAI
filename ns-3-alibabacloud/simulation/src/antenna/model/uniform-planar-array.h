/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
*   Copyright (c) 2020 University of Padova, Dep. of Information Engineering, SIGNET lab.
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License version 2 as
*   published by the Free Software Foundation;
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with this program; if not, write to the Free Software
*   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef UNIFORM_PLANAR_ARRAY_H
#define UNIFORM_PLANAR_ARRAY_H


#include <ns3/object.h>
#include <ns3/phased-array-model.h>


namespace ns3 {


/**
 * \ingroup antenna
 *
 * \brief Class implementing Uniform Planar Array (UPA) model.
 *
 * \note the current implementation supports the modeling of antenna arrays
 * composed of a single panel and with single (configured) polarization.
 */
class UniformPlanarArray : public PhasedArrayModel
{
public:
  /**
   * Constructor
   */
  UniformPlanarArray (void);


  /**
   * Destructor
   */
  virtual ~UniformPlanarArray (void);

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);


  /**
   * Returns the horizontal and vertical components of the antenna element field
   * pattern at the specified direction. Single polarization is considered.
   * \param a the angle indicating the interested direction
   * \return a pair in which the first element is the horizontal component
   *         of the field pattern and the second element is the vertical
   *         component of the field pattern
   */
  std::pair<double, double> GetElementFieldPattern (Angles a) const override;


  /**
  * Returns the location of the antenna element with the specified
  * index assuming the left bottom corner is (0,0,0), normalized
  * with respect to the wavelength.
  * Antenna elements are scanned row by row, left to right and bottom to top.
  * For example, an antenna with 2 rows and 3 columns will be ordered as follows:
  * ^ z
  * |  3 4 5
  * |  0 1 2
  * ----------> y
  *
  * \param index index of the antenna element
  * \return the 3D vector that represents the position of the element
  */
  Vector GetElementLocation (uint64_t index) const override;

  /**
   * Returns the number of antenna elements
   * \return the number of antenna elements
   */
  uint64_t GetNumberOfElements (void) const override;

private:
  /**
   * Set the number of columns of the phased array
   * This method resets the stored beamforming vector to a ComplexVector
   * of the correct size, but zero-filled
   * \param n the number of columns
   */
  void SetNumColumns (uint32_t n);


  /**
   * Get the number of columns of the phased array
   * \return the number of columns
   */
  uint32_t GetNumColumns (void) const;


  /**
   * Set the number of rows of the phased array
   * This method resets the stored beamforming vector to a ComplexVector
   * of the correct size, but zero-filled
   * \param n the number of rows
   */
  void SetNumRows (uint32_t n);


  /**
   * Get the number of rows of the phased array
   * \return the number of rows
   */
  uint32_t GetNumRows (void) const;

  /**
   * \brief Set the bearing angle
   * This method sets the bearing angle and
   * computes its cosine and sine
   * \param alpha the bearing angle in radians
   */
  void SetAlpha (double alpha);

  /**
   * \brief Set the downtilt angle
   * This method sets the downtilt angle and
   * computes its cosine and sine
   * \param beta the downtilt angle in radians
   */
  void SetBeta (double beta);

  /**
   * \brief Set the polarization slant angle
   * This method sets the polarization slant angle and
   * computes its cosine and sine
   * \param polSlant the polarization slant angle in radians
   */
  void SetPolSlant (double polSlant);

  /**
   * Set the horizontal spacing for the antenna elements of the phased array
   * This method resets the stored beamforming vector to a ComplexVector
   * of the correct size, but zero-filled
   * \param s the horizontal spacing in multiples of wavelength
   */
  void SetAntennaHorizontalSpacing (double s);


  /**
   * Get the horizontal spacing for the antenna elements of the phased array
   * \return the horizontal spacing in multiples of wavelength
   */
  double GetAntennaHorizontalSpacing (void) const;


  /**
   * Set the vertical spacing for the antenna elements of the phased array
   * This method resets the stored beamforming vector to a ComplexVector
   * of the correct size, but zero-filled
   * \param s the vertical spacing in multiples of wavelength
   */
  void SetAntennaVerticalSpacing (double s);


  /**
   * Get the vertical spacing for the antenna elements of the phased array
   * \return the vertical spacing in multiples of wavelength
   */
  double GetAntennaVerticalSpacing (void) const;

  uint32_t m_numColumns {1}; //!< number of columns
  uint32_t m_numRows {1}; //!< number of rows
  double m_disV {0.5}; //!< antenna spacing in the vertical direction in multiples of wave length
  double m_disH {0.5}; //!< antenna spacing in the horizontal direction in multiples of wave length
  double m_alpha {0.0}; //!< the bearing angle in radians
  double m_cosAlpha {1.0}; //!< the cosine of alpha
  double m_sinAlpha {0.0}; //!< the sine of alpha
  double m_beta {0.0}; //!< the downtilt angle in radians
  double m_cosBeta {1.0}; //!< the cosine of Beta
  double m_sinBeta {0.0}; //!< the sine of Beta
  double m_polSlant {0.0}; //!< the polarization slant angle in radians
  double m_cosPolSlant {1.0}; //!< the cosine of polarization slant angle
  double m_sinPolSlant {0.0}; //!< the sine polarization slant angle
};

} /* namespace ns3 */

#endif /* UNIFORM_PLANAR_ARRAY_H */
