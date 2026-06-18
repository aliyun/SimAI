/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* 
 * Copyright (c) 2009 IITP RAS
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
 * Author: Pavel Boyko <boyko@iitp.ru>
 */

#ifndef MESH_WIFI_INTERFACE_MAC_PLUGIN_H
#define MESH_WIFI_INTERFACE_MAC_PLUGIN_H

#include "ns3/packet.h"
#include "ns3/mac48-address.h"
#include "ns3/mesh-wifi-beacon.h"
#include "ns3/simple-ref-count.h"

namespace ns3 {

class MeshWifiInterfaceMac;

/**
 * \ingroup mesh
 * 
 * \brief Common interface for mesh point interface MAC plugins
 * 
 * \todo plugins description
 */
class MeshWifiInterfaceMacPlugin : public SimpleRefCount<MeshWifiInterfaceMacPlugin>
{
public:
  /// This is for subclasses
  virtual ~MeshWifiInterfaceMacPlugin (){};
  /**
   * Each plugin must be installed on an interface to work
   *
   * \param parent the parent object
   */
  virtual void SetParent (Ptr<MeshWifiInterfaceMac> parent) = 0; 
  /** 
   * \brief Process received frame
   * \param packet
   * \param header
   * 
   * \return false if (and only if) frame should be dropped
   * \todo define when MAC call this
   */
  virtual bool Receive (Ptr<Packet> packet, const WifiMacHeader & header) = 0;
  /**
   * \brief Update frame before it will be forwarded down
   * \param packet
   * \param header
   * \param from
   * \param to
   * \return false if (and only if) frame should be dropped
   * \todo define when MAC call this, preconditions & postconditions
   */
  virtual bool UpdateOutcomingFrame (Ptr<Packet> packet, WifiMacHeader & header, Mac48Address from, Mac48Address to) = 0;
  /**
   * \brief Update beacon before it will be formed and sent
   * \param beacon
   *
   * \todo define when MAC call this
   */
  virtual void UpdateBeacon (MeshWifiBeacon & beacon) const = 0;
  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  virtual int64_t AssignStreams (int64_t stream) = 0;

};

} // namespace ns3

#endif /* MESH_WIFI_INTERFACE_MAC_PLUGIN_H */
