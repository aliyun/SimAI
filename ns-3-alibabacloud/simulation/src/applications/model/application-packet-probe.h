/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Authors: L. Felipe Perrone (perrone@bucknell.edu)
 *          Tiago G. Rodrigues (tgr002@bucknell.edu)
 *
 * Modified by: Mitch Watrous (watrous@u.washington.edu)
 */

#ifndef APPLICATION_PACKET_PROBE_H
#define APPLICATION_PACKET_PROBE_H

#include "ns3/object.h"
#include "ns3/callback.h"
#include "ns3/boolean.h"
#include "ns3/nstime.h"
#include "ns3/packet.h"
#include "ns3/application.h"
#include "ns3/traced-value.h"
#include "ns3/simulator.h"
#include "ns3/probe.h"

namespace ns3 {

/**
 * \brief Probe to translate from a TraceSource to two more easily parsed TraceSources.
 *
 * This class is designed to probe an underlying ns3 TraceSource
 * exporting a packet and a socket address.  This probe exports a
 * trace source "Output" with arguments of type Ptr<const Packet> and
 * const Address&.  This probe exports another trace source
 * "OutputBytes" with arguments of type uint32_t, which is the number
 * of bytes in the packet.  The trace sources emit values when either
 * the probed trace source emits a new value, or when SetValue () is
 * called.
 */
class ApplicationPacketProbe : public Probe
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  ApplicationPacketProbe ();
  virtual ~ApplicationPacketProbe ();

  /**
   * \brief Set a probe value
   *
   * \param packet set the traced packet equal to this
   * \param address set the socket address for the traced packet equal to this
   */
  void SetValue (Ptr<const Packet> packet, const Address& address);

  /**
   * \brief Set a probe value by its name in the Config system
   *
   * \param path config path to access the probe
   * \param packet set the traced packet equal to this
   * \param address set the socket address for the traced packet equal to this
   */
  static void SetValueByPath (std::string path, Ptr<const Packet> packet, const Address& address);

  /**
   * \brief connect to a trace source attribute provided by a given object
   *
   * \param traceSource the name of the attribute TraceSource to connect to
   * \param obj ns3::Object to connect to
   * \return true if the trace source was successfully connected
   */
  virtual bool ConnectByObject (std::string traceSource, Ptr<Object> obj);

  /**
   * \brief connect to a trace source provided by a config path
   *
   * \param path Config path to bind to
   *
   * Note, if an invalid path is provided, the probe will not be connected
   * to anything.
   */
  virtual void ConnectByPath (std::string path);

private:
  /**
   * \brief Method to connect to an underlying ns3::TraceSource with
   * arguments of type Ptr<const Packet> and const Address&
   *
   * \param packet the traced packet
   * \param address the socket address for the traced packet
   *
   */
  void TraceSink (Ptr<const Packet> packet, const Address& address);

  /// Output trace, packet and source address
  TracedCallback<Ptr<const Packet>, const Address&> m_output;
  /// Output trace, previous packet size and current packet size
  TracedCallback<uint32_t, uint32_t> m_outputBytes;

  /// The traced packet.
  Ptr<const Packet> m_packet;

  /// The socket address for the traced packet.
  Address m_address;

  /// The size of the traced packet.
  uint32_t m_packetSizeOld;
};


} // namespace ns3

#endif // APPLICATION_PACKET_PROBE_H
