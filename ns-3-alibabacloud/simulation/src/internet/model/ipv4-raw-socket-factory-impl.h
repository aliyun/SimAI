/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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

#ifndef IPV4_RAW_SOCKET_FACTORY_IMPL_H
#define IPV4_RAW_SOCKET_FACTORY_IMPL_H

#include "ns3/ipv4-raw-socket-factory.h"

namespace ns3 {

/**
 * \ingroup socket
 * \ingroup ipv4
 *
 * \brief Implementation of IPv4 raw socket factory.
 */
class Ipv4RawSocketFactoryImpl : public Ipv4RawSocketFactory
{
public:
  virtual Ptr<Socket> CreateSocket (void);
};


} // namespace ns3

#endif /* IPV4_RAW_SOCKET_FACTORY_IMPL_H */
