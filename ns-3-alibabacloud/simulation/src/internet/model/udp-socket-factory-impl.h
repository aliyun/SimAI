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
#ifndef UDP_SOCKET_FACTORY_IMPL_H
#define UDP_SOCKET_FACTORY_IMPL_H

#include "ns3/udp-socket-factory.h"
#include "ns3/ptr.h"

namespace ns3 {

class UdpL4Protocol;

/**
 * \ingroup socket
 * \ingroup udp
 *
 * \brief Object to create UDP socket instances 
 *
 * This class implements the API for creating UDP sockets.
 * It is a socket factory (deriving from class SocketFactory).
 */
class UdpSocketFactoryImpl : public UdpSocketFactory
{
public:
  UdpSocketFactoryImpl ();
  virtual ~UdpSocketFactoryImpl ();

  /**
   * \brief Set the associated UDP L4 protocol.
   * \param udp the UDP L4 protocol
   */
  void SetUdp (Ptr<UdpL4Protocol> udp);

  /**
   * \brief Implements a method to create a Udp-based socket and return
   * a base class smart pointer to the socket.
   *
   * \return smart pointer to Socket
   */
  virtual Ptr<Socket> CreateSocket (void);

protected:
  virtual void DoDispose (void);
private:
  Ptr<UdpL4Protocol> m_udp; //!< the associated UDP L4 protocol
};

} // namespace ns3

#endif /* UDP_SOCKET_FACTORY_IMPL_H */
