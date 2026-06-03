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
 */

#ifndef TUTORIAL_APP_H
#define TUTORIAL_APP_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"

namespace ns3 {

class Application;

/**
 * Tutorial - a simple Application sending packets.
 */
class TutorialApp : public Application
{
public:
  TutorialApp ();
  virtual ~TutorialApp ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Setup the socket.
   * \param socket The socket.
   * \param address The destination address.
   * \param packetSize The packet size to transmit.
   * \param nPackets The number of packets to transmit.
   * \param dataRate the datarate to use.
   */
  void Setup (Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  /// Schedule a new transmission.
  void ScheduleTx (void);
  /// Send a packet.
  void SendPacket (void);

  Ptr<Socket>     m_socket;       //!< The tranmission socket.
  Address         m_peer;         //!< The destination address.
  uint32_t        m_packetSize;   //!< The packet size.
  uint32_t        m_nPackets;     //!< The number of pacts to send.
  DataRate        m_dataRate;     //!< The datarate to use.
  EventId         m_sendEvent;    //!< Send event.
  bool            m_running;      //!< True if the application is running.
  uint32_t        m_packetsSent;  //!< The number of pacts sent.
};

} // namespace ns3

#endif /* TUTORIAL_APP_H */
