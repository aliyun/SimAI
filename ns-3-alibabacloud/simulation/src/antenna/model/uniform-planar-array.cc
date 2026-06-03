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


#include "uniform-planar-array.h"
#include <ns3/log.h>
#include <ns3/double.h>
#include <ns3/uinteger.h>
#include <ns3/boolean.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("UniformPlanarArray");

NS_OBJECT_ENSURE_REGISTERED (UniformPlanarArray);



UniformPlanarArray::UniformPlanarArray ()
  : PhasedArrayModel ()
{}

UniformPlanarArray::~UniformPlanarArray ()
{}

TypeId
UniformPlanarArray::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UniformPlanarArray")
    .SetParent<PhasedArrayModel> ()
    .AddConstructor<UniformPlanarArray> ()
    .SetGroupName ("Antenna")
    .AddAttribute ("AntennaHorizontalSpacing",
                   "Horizontal spacing between antenna elements, in multiples of wave length",
                   DoubleValue (0.5),
                   MakeDoubleAccessor (&UniformPlanarArray::SetAntennaHorizontalSpacing,
                                       &UniformPlanarArray::GetAntennaHorizontalSpacing),
                   MakeDoubleChecker<double> (0.0))
    .AddAttribute ("AntennaVerticalSpacing",
                   "Vertical spacing between antenna elements, in multiples of wave length",
                   DoubleValue (0.5),
                   MakeDoubleAccessor (&UniformPlanarArray::SetAntennaVerticalSpacing,
                                       &UniformPlanarArray::GetAntennaVerticalSpacing),
                   MakeDoubleChecker<double> (0.0))
    .AddAttribute ("NumColumns",
                   "Horizontal size of the array",
                   UintegerValue (4),
                   MakeUintegerAccessor (&UniformPlanarArray::SetNumColumns,
                                         &UniformPlanarArray::GetNumColumns),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("NumRows",
                   "Vertical size of the array",
                   UintegerValue (4),
                   MakeUintegerAccessor (&UniformPlanarArray::SetNumRows,
                                         &UniformPlanarArray::GetNumRows),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("BearingAngle",
                   "The bearing angle in radians",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&UniformPlanarArray::SetAlpha),
                   MakeDoubleChecker<double> (-M_PI, M_PI))
    .AddAttribute ("DowntiltAngle",
                   "The downtilt angle in radians",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&UniformPlanarArray::SetBeta),
                   MakeDoubleChecker<double> (-M_PI, M_PI))
    .AddAttribute ("PolSlantAngle",
                   "The polarization slant angle in radians",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&UniformPlanarArray::SetPolSlant),
                   MakeDoubleChecker<double> (-M_PI, M_PI))
  ;
  return tid;
}


void
UniformPlanarArray::SetNumColumns (uint32_t n)
{
  NS_LOG_FUNCTION (this << n);
  if (n != m_numColumns)
    {
      m_isBfVectorValid = false;
    }
  m_numColumns = n;
}


uint32_t
UniformPlanarArray::GetNumColumns (void) const
{
  return m_numColumns;
}


void
UniformPlanarArray::SetNumRows (uint32_t n)
{
  NS_LOG_FUNCTION (this << n);
  if (n != m_numRows)
    {
      m_isBfVectorValid = false;
    }
  m_numRows = n;
}


uint32_t
UniformPlanarArray::GetNumRows (void) const
{
  return m_numRows;
}

void
UniformPlanarArray::SetAlpha (double alpha)
{
  m_alpha = alpha;
  m_cosAlpha = cos (m_alpha);
  m_sinAlpha = sin (m_alpha);
}

void
UniformPlanarArray::SetBeta (double beta)
{
  m_beta = beta;
  m_cosBeta = cos (m_beta);
  m_sinBeta = sin (m_beta);
}

void
UniformPlanarArray::SetPolSlant (double polSlant)
{
  m_polSlant = polSlant;
  m_cosPolSlant = cos (m_polSlant);
  m_sinPolSlant = sin (m_polSlant);
}

void
UniformPlanarArray::SetAntennaHorizontalSpacing (double s)
{
  NS_LOG_FUNCTION (this << s);
  NS_ABORT_MSG_IF (s <= 0, "Trying to set an invalid spacing: " << s);

  if (s != m_disH)
    {
      m_isBfVectorValid = false;
    }
  m_disH = s;
}


double
UniformPlanarArray::GetAntennaHorizontalSpacing (void) const
{
  return m_disH;
}


void
UniformPlanarArray::SetAntennaVerticalSpacing (double s)
{
  NS_LOG_FUNCTION (this << s);
  NS_ABORT_MSG_IF (s <= 0, "Trying to set an invalid spacing: " << s);

  if (s != m_disV)
    {
      m_isBfVectorValid = false;
    }
  m_disV = s;
}


double
UniformPlanarArray::GetAntennaVerticalSpacing (void) const
{
  return m_disV;
}


std::pair<double, double>
UniformPlanarArray::GetElementFieldPattern (Angles a) const
{
  NS_LOG_FUNCTION (this << a);

  // convert the theta and phi angles from GCS to LCS using eq. 7.1-7 and 7.1-8 in 3GPP TR 38.901
  // NOTE we assume a fixed slant angle of 0 degrees
  double cosIncl = cos (a.GetInclination ());
  double sinIncl = sin (a.GetInclination ());
  double cosAzim = cos (a.GetAzimuth () - m_alpha);
  double sinAzim = sin (a.GetAzimuth () - m_alpha);
  double thetaPrime = std::acos (m_cosBeta * cosIncl + m_sinBeta * cosAzim * sinIncl);
  double phiPrime = std::arg (std::complex<double> (m_cosBeta * sinIncl * cosAzim - m_sinBeta * cosIncl, sinAzim * sinIncl));
  Angles aPrime (phiPrime, thetaPrime);
  NS_LOG_DEBUG (a << " -> " << aPrime);

  // compute the antenna element field patterns using eq. 7.3-4 and 7.3-5 in 3GPP TR 38.901,
  // using the configured polarization slant angle (m_polSlant)
  // NOTE: the slant angle (assumed to be 0) differs from the polarization slant angle
  // (m_polSlant, given by the attribute), in 3GPP TR 38.901
  double aPrimeDb = m_antennaElement->GetGainDb (aPrime);
  double fieldThetaPrime = pow (10, aPrimeDb / 20) * m_cosPolSlant; // convert to linear magnitude
  double fieldPhiPrime = pow (10, aPrimeDb / 20) * m_sinPolSlant; // convert to linear magnitude

  // compute psi using eq. 7.1-15 in 3GPP TR 38.901, assuming that the slant
  // angle (gamma) is 0
  double psi = std::arg (std::complex<double> (m_cosBeta * sinIncl - m_sinBeta * cosIncl * cosAzim, m_sinBeta * sinAzim));
  NS_LOG_DEBUG ("psi " << psi);

  // convert the antenna element field pattern to GCS using eq. 7.1-11
  // in 3GPP TR 38.901
  double fieldTheta = cos (psi) * fieldThetaPrime - sin (psi) * fieldPhiPrime;
  double fieldPhi = sin (psi) * fieldThetaPrime + cos (psi) * fieldPhiPrime;
  NS_LOG_DEBUG (RadiansToDegrees (a.GetAzimuth ()) << " " << RadiansToDegrees (a.GetInclination ()) << " " << fieldTheta * fieldTheta + fieldPhi * fieldPhi);

  return std::make_pair (fieldPhi, fieldTheta);
}


Vector
UniformPlanarArray::GetElementLocation (uint64_t index) const
{
  NS_LOG_FUNCTION (this << index);

  // compute the element coordinates in the LCS
  // assume the left bottom corner is (0,0,0), and the rectangular antenna array is on the y-z plane.
  double xPrime = 0;
  double yPrime = m_disH * (index % m_numColumns);
  double zPrime = m_disV * floor (index / m_numColumns);

  // convert the coordinates to the GCS using the rotation matrix 7.1-4 in 3GPP
  // TR 38.901
  Vector loc;
  loc.x = m_cosAlpha * m_cosBeta * xPrime - m_sinAlpha * yPrime + m_cosAlpha * m_sinBeta * zPrime;
  loc.y = m_sinAlpha * m_cosBeta * xPrime + m_cosAlpha * yPrime + m_sinAlpha * m_sinBeta * zPrime;
  loc.z = -m_sinBeta * xPrime + m_cosBeta * zPrime;
  return loc;
}

uint64_t
UniformPlanarArray::GetNumberOfElements () const
{
  return m_numRows * m_numColumns;
}

} /* namespace ns3 */
