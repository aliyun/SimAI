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

#ifndef PACKET_PROBE_H
#define PACKET_PROBE_H

#include "ns3/object.h"
#include "ns3/callback.h"
#include "ns3/boolean.h"
#include "ns3/nstime.h"
#include "ns3/packet.h"
#include "ns3/traced-value.h"
#include "ns3/simulator.h"
#include "ns3/probe.h"

namespace ns3 {

/**
 * This class is designed to probe an underlying ns3 TraceSource
 * exporting a packet.  This probe exports a trace source "Output"
 * with argument of type Ptr<const Packet>.  This probe exports
 * another trace source "OutputBytes" with arguments of type uint32_t,
 * which is the number of bytes in the packet.  The trace sources emit
 * values when either the probed trace source emits a new value, or
 * when SetValue () is called.
 */
class PacketProbe : public Probe
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  PacketProbe ();
  virtual ~PacketProbe ();

  /**
   * \brief Set a probe value
   *
   * \param packet set the traced packet equal to this
   */
  void SetValue (Ptr<const Packet> packet);

  /**
   * \brief Set a probe value by its name in the Config system
   *
   * \param path config path to access the probe
   * \param packet set the traced packet equal to this
   */
  static void SetValueByPath (std::string path, Ptr<const Packet> packet);

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
   * arguments of type Ptr<const Packet>
   *
   * \param packet the traced packet
   */
  void TraceSink (Ptr<const Packet> packet);

  /// Traced callback: packet received
  TracedCallback<Ptr<const Packet> > m_output;
  /// Traced callback: size of previous packet receive, size of actual packet received
  TracedCallback<uint32_t, uint32_t> m_outputBytes;

  /// The traced packet.
  Ptr<const Packet> m_packet;

  /// The size of the traced packet.
  uint32_t m_packetSizeOld;
};


} // namespace ns3

#endif // PACKET_PROBE_H
