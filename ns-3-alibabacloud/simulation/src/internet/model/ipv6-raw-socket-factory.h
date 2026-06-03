/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef IPV6_RAW_SOCKET_FACTORY_H
#define IPV6_RAW_SOCKET_FACTORY_H

#include "ns3/socket-factory.h"

namespace ns3 
{

class Socket;

/**
 * \ingroup ipv6
 * \ingroup socket
 *
 * \brief API to create IPv6 RAW socket instances 
 *
 * This abstract class defines the API for IPv6 RAW socket factory.
 * 
 * A RAW Socket typically is used to access specific IP layers not usually
 * available through L4 sockets, e.g., ICMP. The implementer should take
 * particular care to define the Ipv6RawSocketImpl Attributes, and in
 * particular the Protocol attribute.
 * Not setting it will result in a zero protocol at IP level (corresponding
 * to the HopByHop IPv6 Extension header, i.e., Ipv6ExtensionHopByHopHeader)
 * when sending data through the socket, which is probably not the intended
 * behavior.
 *
 * A correct example is (from src/applications/model/radvd.cc):
 * \code
   if (!m_socket)
     {
       TypeId tid = TypeId::LookupByName ("ns3::Ipv6RawSocketFactory");
       m_socket = Socket::CreateSocket (GetNode (), tid);

       NS_ASSERT (m_socket);

       m_socket->SetAttribute ("Protocol", UintegerValue(Ipv6Header::IPV6_ICMPV6));
       m_socket->SetRecvCallback (MakeCallback (&Radvd::HandleRead, this));
     }
 * \endcode
 *
 */
class Ipv6RawSocketFactory : public SocketFactory
{
public:
  /** 
   * \brief Get the type ID of this class.
   * \return type ID
   */
  static TypeId GetTypeId (void);
};

} // namespace ns3

#endif /* IPV6_RAW_SOCKET_FACTORY_H */

