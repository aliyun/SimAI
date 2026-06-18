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

#include "ns3/application.h"
#include "ns3/node.h"
#include "ns3/ptr.h"
#include "ns3/socket.h"
#include "ns3/address.h"

namespace ns3
{

/**
 * \ingroup system-tests-tcp
 * 
 * \brief Simple class to write data to sockets.
 */
class SocketWriter : public Application
{
public:
  SocketWriter ();
  virtual ~SocketWriter ();
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * Setup the socket.
   * \param node The node owning the socket.
   * \param peer The destinaiton address.
   */
  void Setup (Ptr<Node> node, Address peer);
  /**
   * Connect the socket.
   */
  void Connect ();
  /**
   * Write to the socket.
   * \param numBytes The number of bytes to write.
   */
  void Write (uint32_t numBytes);
  /**
   * Close the socket.
   */
  void Close ();

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);
  Address m_peer;         //!< Peer's address.
  Ptr<Node> m_node;       //!< Node pointer
  Ptr<Socket> m_socket;   //!< Socket.
  bool m_isSetup;         //!< True if the socket is connected.
  bool m_isConnected;     //!< True if the socket setup has been done.
};
}
