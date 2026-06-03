/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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

#include "spectrum-phy.h"
#include <ns3/net-device.h>
#include <ns3/mobility-model.h>
#include <ns3/spectrum-value.h>
#include <ns3/spectrum-channel.h>
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SpectrumPhy");

NS_OBJECT_ENSURE_REGISTERED (SpectrumPhy);


TypeId
SpectrumPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SpectrumPhy")
    .SetParent<Object> ()
    .SetGroupName ("Spectrum")
  ;
  return tid;
}

SpectrumPhy::SpectrumPhy ()
{
  NS_LOG_FUNCTION (this);
}

SpectrumPhy::~SpectrumPhy ()
{
  NS_LOG_FUNCTION (this);
}
} // namespace
