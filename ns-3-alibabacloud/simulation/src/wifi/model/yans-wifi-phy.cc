/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Ghada Badawy <gbadawy@gmail.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/log.h"
#include "yans-wifi-phy.h"
#include "yans-wifi-channel.h"
#include "interference-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("YansWifiPhy");

NS_OBJECT_ENSURE_REGISTERED (YansWifiPhy);

TypeId
YansWifiPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::YansWifiPhy")
    .SetParent<WifiPhy> ()
    .SetGroupName ("Wifi")
    .AddConstructor<YansWifiPhy> ()
  ;
  return tid;
}

YansWifiPhy::YansWifiPhy ()
{
  NS_LOG_FUNCTION (this);
}

void
YansWifiPhy::SetInterferenceHelper (const Ptr<InterferenceHelper> helper)
{
  WifiPhy::SetInterferenceHelper (helper);
  //add dummy band for Yans
  WifiSpectrumBand band;
  band.first = 0;
  band.second = 0;
  m_interference->AddBand (band);
}

YansWifiPhy::~YansWifiPhy ()
{
  NS_LOG_FUNCTION (this);
}

void
YansWifiPhy::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_channel = 0;
  WifiPhy::DoDispose ();
}

Ptr<Channel>
YansWifiPhy::GetChannel (void) const
{
  return m_channel;
}

void
YansWifiPhy::SetChannel (const Ptr<YansWifiChannel> channel)
{
  NS_LOG_FUNCTION (this << channel);
  m_channel = channel;
  m_channel->Add (this);
}

void
YansWifiPhy::StartTx (Ptr<WifiPpdu> ppdu)
{
  NS_LOG_FUNCTION (this << ppdu);
  NS_LOG_DEBUG ("Start transmission: signal power before antenna gain=" << GetPowerDbm (ppdu->GetTxVector ().GetTxPowerLevel ()) << "dBm");
  m_channel->Send (this, ppdu, GetTxPowerForTransmission (ppdu) + GetTxGain ());
}

uint16_t
YansWifiPhy::GetGuardBandwidth (uint16_t currentChannelWidth) const
{
  NS_ABORT_MSG ("Guard bandwidth not relevant for Yans");
  return 0;
}

std::tuple<double, double, double>
YansWifiPhy::GetTxMaskRejectionParams (void) const
{
  NS_ABORT_MSG ("Tx mask rejection params not relevant for Yans");
  return std::make_tuple (0.0, 0.0, 0.0);
}

} //namespace ns3
