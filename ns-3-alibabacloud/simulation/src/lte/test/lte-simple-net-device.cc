/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/simulator.h"
#include "ns3/pointer.h"
#include "ns3/log.h"
#include "ns3/queue.h"
#include "lte-simple-net-device.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteSimpleNetDevice");

NS_OBJECT_ENSURE_REGISTERED (LteSimpleNetDevice);


TypeId LteSimpleNetDevice::GetTypeId (void)
{
  static TypeId
    tid =
    TypeId ("ns3::LteSimpleNetDevice")
    .SetParent<SimpleNetDevice> ()
    .AddConstructor<LteSimpleNetDevice> ()
  ;

  return tid;
}


LteSimpleNetDevice::LteSimpleNetDevice (void)
{
  NS_LOG_FUNCTION (this);
}


LteSimpleNetDevice::LteSimpleNetDevice (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);
  SetNode (node);
}

LteSimpleNetDevice::~LteSimpleNetDevice (void)
{
  NS_LOG_FUNCTION (this);
}

void
LteSimpleNetDevice::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  SimpleNetDevice::DoDispose ();
}


void
LteSimpleNetDevice::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
}

bool
LteSimpleNetDevice::Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << dest << protocolNumber);
  return SimpleNetDevice::Send (packet, dest, protocolNumber);
}


} // namespace ns3
