/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Telum (www.telum.ru)
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
 * Author: Kirill Andreev <andreev@telum.ru>
 */

#include "jakes-propagation-loss-model.h"
#include "ns3/double.h"
#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("Jakes");
  
NS_OBJECT_ENSURE_REGISTERED (JakesPropagationLossModel);


JakesPropagationLossModel::JakesPropagationLossModel()
{
  m_uniformVariable = CreateObject<UniformRandomVariable> ();
  m_uniformVariable->SetAttribute ("Min", DoubleValue (-1.0 * M_PI));
  m_uniformVariable->SetAttribute ("Max", DoubleValue (M_PI));
}

JakesPropagationLossModel::~JakesPropagationLossModel()
{}

TypeId
JakesPropagationLossModel::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::JakesPropagationLossModel")
    .SetParent<PropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<JakesPropagationLossModel> ()
  ;
  return tid;
}

double
JakesPropagationLossModel::DoCalcRxPower (double txPowerDbm,
                                          Ptr<MobilityModel> a,
                                          Ptr<MobilityModel> b) const
{
  Ptr<JakesProcess> pathData = m_propagationCache.GetPathData (a, b, 0 /**Spectrum model uid is not used in PropagationLossModel*/);
  if (pathData == 0)
    {
      pathData = CreateObject<JakesProcess> ();
      pathData->SetPropagationLossModel (this);
      m_propagationCache.AddPathData (pathData, a, b, 0 /**Spectrum model uid is not used in PropagationLossModel*/);
    }
  return txPowerDbm + pathData->GetChannelGainDb ();
}

Ptr<UniformRandomVariable>
JakesPropagationLossModel::GetUniformRandomVariable () const
{
  return m_uniformVariable;
}

int64_t
JakesPropagationLossModel::DoAssignStreams (int64_t stream)
{
  m_uniformVariable->SetStream (stream);
  return 1;
}

} // namespace ns3

