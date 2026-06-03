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
#ifndef SOCKET_FACTORY_H
#define SOCKET_FACTORY_H

#include "ns3/object.h"
#include "ns3/ptr.h"

namespace ns3 {

class Socket;

/**
 * \ingroup socket
 *
 * \brief Object to create transport layer instances that provide a 
 * socket API to applications.
 *
 * This base class defines the API for creating sockets.
 * The socket factory also can hold the global variables used to
 * initialize newly created sockets, such as values that are
 * set through the sysctl or proc interfaces in Linux.

 * If you want to write a new transport protocol accessible through
 * sockets, you need to subclass this factory class, implement CreateSocket, 
 * instantiate the object, and aggregate it to the node.
 * 
 * \see Udp
 * \see UdpImpl
 */
class SocketFactory : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  SocketFactory ();

  /**
   * \return smart pointer to Socket
   * 
   * Base class method for creating socket instances.
   */
  virtual Ptr<Socket> CreateSocket (void) = 0;
};

} // namespace ns3


#endif /* SOCKET_FACTORY_H */
