/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Georgia Tech Research Corporation
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
 * Author: Raj Bhattacharjea <raj.b@gatech.edu>
 */
#include "tcp-socket-factory-impl.h"
#include "tcp-l4-protocol.h"
#include "ns3/socket.h"
#include "ns3/assert.h"

namespace ns3 {

TcpSocketFactoryImpl::TcpSocketFactoryImpl ()
  : m_tcp (0)
{
}
TcpSocketFactoryImpl::~TcpSocketFactoryImpl ()
{
  NS_ASSERT (m_tcp == 0);
}

void
TcpSocketFactoryImpl::SetTcp (Ptr<TcpL4Protocol> tcp)
{
  m_tcp = tcp;
}

Ptr<Socket>
TcpSocketFactoryImpl::CreateSocket (void)
{
  return m_tcp->CreateSocket ();
}

void 
TcpSocketFactoryImpl::DoDispose (void)
{
  m_tcp = 0;
  TcpSocketFactory::DoDispose ();
}

} // namespace ns3
