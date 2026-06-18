/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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

/**
 * \file
 * \ingroup mpi
 * ns3::MpiReciver implementation,
 * provides an interface to aggregate to MPI-compatible NetDevices.
 */

#include "mpi-receiver.h"

namespace ns3 {

TypeId
MpiReceiver::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MpiReceiver")
    .SetParent<Object> ()
    .SetGroupName ("Mpi")
    .AddConstructor <MpiReceiver> ();
  return tid;
}

MpiReceiver::~MpiReceiver ()
{
}

void
MpiReceiver::SetReceiveCallback (Callback<void, Ptr<Packet> > callback)
{
  m_rxCallback = callback;
}

void
MpiReceiver::Receive (Ptr<Packet> p)
{
  NS_ASSERT (!m_rxCallback.IsNull ());
  m_rxCallback (p);
}

void
MpiReceiver::DoDispose(void)
{
  m_rxCallback = MakeNullCallback<void, Ptr<Packet> >();
}

} // namespace ns3
