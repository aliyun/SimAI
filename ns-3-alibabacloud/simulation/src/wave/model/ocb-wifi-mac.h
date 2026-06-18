/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2013 Dalian University of Technology
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
 *         Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef OCB_WIFI_MAC_H
#define OCB_WIFI_MAC_H

#include "ns3/object-factory.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mac-queue.h"
#include "vendor-specific-action.h"
#include "wave-net-device.h"

namespace ns3 {

class OrganizationIdentifier;
class WaveNetDevice;

/**
 * \brief STAs communicate with each directly outside the context of a BSS
 * \ingroup wave
 *
 * In OCB mac mode,synchronization, association, dis-association
 * and authentication of normal wifi are not used for wireless access in
 * vehicular environments.
 *
 * Although Timing Advertisement frame is a specific management frame defined
 * in 802.11p. It is mainly used by IEEE Std 1609.4 for channel switch synchronization.
 * However in simulation nodes are supposed to have GPS synchronization ability,
 * so we will not implement this feature.
 */
class OcbWifiMac : public WifiMac
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  OcbWifiMac (void);
  virtual ~OcbWifiMac (void);
  /**
   * \param vsc management packet to send.
   * \param peer the address to which the packet should be sent.
   * \param oi Organization Identifier field
   * see 7.3.1.31 Organization Identifier field and 10.3.29 Vendor-specific action
   *
   * management information can be transmitted over vender specific action frame.
   * This will be mainly called by IEEE Std 1609.4 to send WSA from IEEE Std 1609.3
   */
  void SendVsc (Ptr<Packet> vsc, Mac48Address peer, OrganizationIdentifier oi);
  /**
   * \param oi Organization Identifier
   * \param cb callback to invoke whenever a vender specific action frame has been received and must
   *        be forwarded to the higher layers.
   * every node shall register first if it wants to receive specific vendor specific content.
   */
  void AddReceiveVscCallback (OrganizationIdentifier oi, VscCallback cb);
  /**
   * \param oi Organization Identifier
   */
  void RemoveReceiveVscCallback (OrganizationIdentifier oi);

  /**
   * \returns the ssid which this MAC layer is going to try to stay in.
   *
   * This method shall not be used in WAVE environment and
   * here it will overloaded to log warn message
   */
  virtual Ssid GetSsid (void) const;
  /**
   * \param ssid the current ssid of this MAC layer.
   *
   * This method shall not be used in WAVE environment and
   * here it will overloaded to log warn message
   */
  virtual void SetSsid (Ssid ssid);
  /**
   * \param bssid the BSSID of the network that this device belongs to.
   * This method shall not be used in WAVE environment and
   * here it will overloaded to log warn message
   */
  virtual void SetBssid (Mac48Address bssid);
  /**
   * This method shall not be used in WAVE environment and
   * here it will overloaded to log warn message
   * \return An invalid BSSID.
   */
  virtual Mac48Address GetBssid (void) const;
  /**
   * SetLinkUpCallback and SetLinkDownCallback will be overloaded
   * In OCB mode, stations can send packets directly whenever they want
   * so the link is always up and never down even during channel switch
   */
  /**
   * \param linkUp the callback to invoke when the link becomes up.
   */
  virtual void SetLinkUpCallback (Callback<void> linkUp);
  /**
   * \param linkDown the callback to invoke when the link becomes down.
   */
  virtual void SetLinkDownCallback (Callback<void> linkDown);
  /**
   * \param packet the packet to send.
   * \param to the address to which the packet should be sent.
   *
   * The packet should be enqueued in a tx queue, and should be
   * dequeued as soon as the channel access function determines that
   * access is granted to this MAC.
   */
  virtual void Enqueue (Ptr<Packet> packet, Mac48Address to);
  virtual bool CanForwardPacketsTo (Mac48Address to) const;
  /**
    * \param cwmin the min contention window
    * \param cwmax the max contention window
    * \param aifsn the arbitration inter-frame space
    * \param ac    the access category index
    *
    * configure EDCA queue parameters
    */
  void ConfigureEdca (uint32_t cwmin, uint32_t cwmax, uint32_t aifsn, enum AcIndex ac);

  // below six public methods are used for MAC extension defined in IEEE 1609.4
  /**
   * \param device make current MAC entity associated with WaveNetDevice
   *
   * To support MAC extension for multiple channel operation,
   *  WaveMacLow object will be used to replace original MacLow object.
   */
  void EnableForWave (Ptr<WaveNetDevice> device);
  /**
   * To support MAC extension for multiple channel operation,
   * Suspend the activity in current MAC entity
   */
  void Suspend (void);
  /**
   * To support MAC extension for multiple channel operation,
   * Resume the activity of suspended MAC entity
   */
  void Resume (void);
  /**
   * \param duration the virtual busy time for MAC entity
   *
   * To support MAC extension for multiple channel operation,
   * Notify MAC entity busy for some time to prevent transmission
   */
  void MakeVirtualBusy (Time duration);
  /**
   * \param ac the specified access category
   *
   * To support MAC extension for multiple channel operation,
   * Cancel transmit operation for internal queue associated with a specified Access Category.
   */
  void CancleTx (enum AcIndex ac);
  /**
   * To support MAC extension for multiple channel operation,
   * Reset current MAC entity and flush its internal queues.
   */
  void Reset (void);

  // Inherited from base class
  virtual void ConfigureStandard (enum WifiStandard standard);
protected:
  virtual void DoDispose (void);
private:
  virtual void Receive (Ptr<WifiMacQueueItem> mpdu);

  VendorSpecificContentManager m_vscManager; ///< VSC manager
};

}
#endif /* OCB_WIFI_MAC_H */
