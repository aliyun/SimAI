/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#include "uan-prop-model-ideal.h"
#include "uan-tx-mode.h"
#include "ns3/mobility-model.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (UanPropModelIdeal);

UanPropModelIdeal::UanPropModelIdeal ()
{
}

UanPropModelIdeal::~UanPropModelIdeal ()
{
}

TypeId
UanPropModelIdeal::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanPropModelIdeal")
    .SetParent<UanPropModel> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanPropModelIdeal> ()
  ;
  return tid;
}


double
UanPropModelIdeal::GetPathLossDb (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode)
{

  return 0;
}
UanPdp
UanPropModelIdeal::GetPdp (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode)
{
  return UanPdp::CreateImpulsePdp ();
}

Time
UanPropModelIdeal::GetDelay (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode)
{
  return Seconds (a->GetDistanceFrom (b) / 1500.0);
}


} // namespace ns3
