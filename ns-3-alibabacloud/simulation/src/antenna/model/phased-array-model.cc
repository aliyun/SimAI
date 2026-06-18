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

#include "phased-array-model.h"
#include <ns3/isotropic-antenna-model.h>
#include <ns3/log.h>
#include <ns3/double.h>
#include <ns3/uinteger.h>
#include <ns3/boolean.h>
#include <ns3/pointer.h>

namespace ns3 {

uint32_t PhasedArrayModel::m_idCounter = 0 ;

NS_LOG_COMPONENT_DEFINE ("PhasedArrayModel");

NS_OBJECT_ENSURE_REGISTERED (PhasedArrayModel);

std::ostream&
operator<< (std::ostream& os, const PhasedArrayModel::ComplexVector& cv)
{
  size_t N = cv.size ();

  // empty
  if (N == 0)
    {
      os << "[]";
      return os;
    }

  // non-empty
  os << "[";
  for (std::size_t i = 0; i < N - 1; ++i)
    {
      os << cv[i] << ", ";
    }
  os << cv[N - 1] << "]";
  return os;
}

PhasedArrayModel::PhasedArrayModel ()
  : m_isBfVectorValid {false}
{
  m_id = m_idCounter++;
}


PhasedArrayModel::~PhasedArrayModel ()
{
  m_beamformingVector.clear ();
}


TypeId
PhasedArrayModel::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::PhasedArrayModel")
    .SetParent<Object> ()
    .SetGroupName ("Antenna")
    .AddAttribute ("AntennaElement",
                   "A pointer to the antenna element used by the phased array",
                   PointerValue (CreateObject<IsotropicAntennaModel> ()),
                   MakePointerAccessor (&PhasedArrayModel::m_antennaElement),
                   MakePointerChecker<AntennaModel> ())
  ;
  return tid;
}


void
PhasedArrayModel::SetBeamformingVector (const ComplexVector &beamformingVector)
{
  NS_LOG_FUNCTION (this << beamformingVector);
  NS_ASSERT_MSG (beamformingVector.size () == GetNumberOfElements (),
                 beamformingVector.size () << " != " << GetNumberOfElements ());
  m_beamformingVector = beamformingVector;
  m_isBfVectorValid = true;
}


PhasedArrayModel::ComplexVector
PhasedArrayModel::GetBeamformingVector () const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (m_isBfVectorValid, "The beamforming vector should be Set before it's Get, and should refer to the current array configuration");
  return m_beamformingVector;
}


double
PhasedArrayModel::ComputeNorm (const ComplexVector &vector)
{
  double norm = 0;

  for (uint64_t i = 0; i < vector.size (); i++)
    {
      norm += std::norm (vector[i]);
    }

  return std::sqrt (norm);

}


PhasedArrayModel::ComplexVector
PhasedArrayModel::GetBeamformingVector (Angles a) const
{
  NS_LOG_FUNCTION (this << a);

  ComplexVector beamformingVector = GetSteeringVector (a);
  double norm = ComputeNorm (beamformingVector);

  for (uint64_t i = 0; i < beamformingVector.size (); i++)
    {
      beamformingVector[i] = std::conj (beamformingVector[i]) / norm;
    }

  return beamformingVector;
}



PhasedArrayModel::ComplexVector
PhasedArrayModel::GetSteeringVector (Angles a) const
{
  ComplexVector steeringVector;
  steeringVector.resize (GetNumberOfElements ());
  for (uint64_t i = 0; i < GetNumberOfElements (); i++)
    {
      Vector loc = GetElementLocation (i);
      double phase = -2 * M_PI * (sin (a.GetInclination ()) * cos (a.GetAzimuth ()) * loc.x +
                                  sin (a.GetInclination ()) * sin (a.GetAzimuth ()) * loc.y +
                                  cos (a.GetInclination ()) * loc.z);
      steeringVector[i] = std::polar<double> (1.0, phase);
    }
  return steeringVector;
}


void
PhasedArrayModel::SetAntennaElement (Ptr<AntennaModel> antennaElement)
{
  NS_LOG_FUNCTION (this);
  m_antennaElement = antennaElement;
}


Ptr<const AntennaModel>
PhasedArrayModel::GetAntennaElement () const
{
  NS_LOG_FUNCTION (this);
  return m_antennaElement;
}

uint32_t
PhasedArrayModel::GetId () const
{
  return m_id;
}

} /* namespace ns3 */

