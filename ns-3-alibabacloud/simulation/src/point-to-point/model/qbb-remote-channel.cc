/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007, 2008 University of Washington
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
 * Author: George Riley <riley@ece.gatech.edu>
 */

#include <iostream>

#include "qbb-remote-channel.h"
#include "qbb-net-device.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#endif

using namespace std;

NS_LOG_COMPONENT_DEFINE ("QbbRemoteChannel");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (QbbRemoteChannel);

TypeId
QbbRemoteChannel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::QbbRemoteChannel")
    .SetParent<QbbChannel> ()
    .AddConstructor<QbbRemoteChannel> ()
  ;
  return tid;
}

QbbRemoteChannel::QbbRemoteChannel ()
{
}

QbbRemoteChannel::~QbbRemoteChannel ()
{
}

bool
QbbRemoteChannel::TransmitStart (
  Ptr<Packet> p,
  Ptr<QbbNetDevice> src,
  Time txTime)
{
  NS_LOG_FUNCTION (this << p << src);
  NS_LOG_LOGIC ("UID is " << p->GetUid () << ")");

  IsInitialized ();

  uint32_t wire = src == GetSource (0) ? 0 : 1;
  Ptr<QbbNetDevice> dst = GetDestination (wire);

#ifdef NS3_MPI
  // Calculate the rxTime (absolute)
  Time rxTime = Simulator::Now () + txTime + GetDelay ();
  MpiInterface::SendPacket (p, rxTime, dst->GetNode ()->GetId (), dst->GetIfIndex ());
#else
  NS_FATAL_ERROR ("Can't use distributed simulator without MPI compiled in");
#endif
  return true;
}

} // namespace ns3
