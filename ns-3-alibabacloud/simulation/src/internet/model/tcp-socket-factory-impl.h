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
#ifndef TCP_SOCKET_FACTORY_IMPL_H
#define TCP_SOCKET_FACTORY_IMPL_H

#include "ns3/tcp-socket-factory.h"
#include "ns3/ptr.h"

namespace ns3 {

class TcpL4Protocol;

/**
 * \ingroup socket
 * \ingroup tcp
 *
 * \brief socket factory implementation for native ns-3 TCP
 *
 *
 * This class serves to create sockets of the TcpSocketBase type.
 */
class TcpSocketFactoryImpl : public TcpSocketFactory
{
public:
  TcpSocketFactoryImpl ();
  virtual ~TcpSocketFactoryImpl ();

  /**
   * \brief Set the associated TCP L4 protocol.
   * \param tcp the TCP L4 protocol
   */
  void SetTcp (Ptr<TcpL4Protocol> tcp);

  virtual Ptr<Socket> CreateSocket (void);

protected:
  virtual void DoDispose (void);
private:
  Ptr<TcpL4Protocol> m_tcp; //!< the associated TCP L4 protocol
};

} // namespace ns3

#endif /* TCP_SOCKET_FACTORY_IMPL_H */
