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


#include <ns3/log.h>
#include <ns3/double.h>

#include "antenna-model.h"
#include "isotropic-antenna-model.h"



namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("IsotropicAntennaModel");

NS_OBJECT_ENSURE_REGISTERED (IsotropicAntennaModel);


TypeId 
IsotropicAntennaModel::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::IsotropicAntennaModel")
    .SetParent<AntennaModel> ()
    .SetGroupName("Antenna")
    .AddConstructor<IsotropicAntennaModel> ()
    .AddAttribute ("Gain",
                   "The gain of the antenna in dB",
                   DoubleValue (0),
                   MakeDoubleAccessor (&IsotropicAntennaModel::m_gainDb),
                   MakeDoubleChecker<double> ())
  ;
  return tid;
}

IsotropicAntennaModel::IsotropicAntennaModel ()
{
  NS_LOG_FUNCTION (this);
}

double 
IsotropicAntennaModel::GetGainDb (Angles a)
{
  NS_LOG_FUNCTION (this << a);
  return m_gainDb;
}

}

