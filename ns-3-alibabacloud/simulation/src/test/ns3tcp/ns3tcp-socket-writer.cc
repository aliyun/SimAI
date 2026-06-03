/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 University of Washington
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
 */

#include "ns3tcp-socket-writer.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/packet.h"

namespace ns3
{

SocketWriter::SocketWriter () : m_node (0), m_socket (0), m_isSetup (false), m_isConnected (false)
{
}

SocketWriter::~SocketWriter ()
{
  m_socket = 0;
  m_node = 0;
}

/* static */
TypeId
SocketWriter::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SocketWriter")
    .SetParent<Application> ()
    .SetGroupName ("Stats")
    .AddConstructor<SocketWriter> ()
    ;
  return tid;
}
  
void
SocketWriter::StartApplication ()
{
  m_socket = Socket::CreateSocket (m_node, TcpSocketFactory::GetTypeId ());
  m_socket->Bind ();
}

void
SocketWriter::StopApplication ()
{
}

void
SocketWriter::Setup (Ptr<Node> node, Address peer)
{
  m_peer = peer;
  m_node = node;
  m_isSetup = true;
}

void
SocketWriter::Connect ()
{
  if (!m_isSetup)
    {
      NS_FATAL_ERROR ("Forgot to call Setup() first");
    }
  m_socket->Connect (m_peer);
  m_isConnected = true;
}

void
SocketWriter::Write (uint32_t numBytes)
{
  if (!m_isConnected)
    {
      Connect ();
    }
  Ptr<Packet> packet = Create<Packet> (numBytes);
  m_socket->Send (packet);
}

void
SocketWriter::Close ()
{
  m_socket->Close ();
}
}
