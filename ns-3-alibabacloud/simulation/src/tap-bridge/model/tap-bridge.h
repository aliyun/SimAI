/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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

#ifndef TAP_BRIDGE_H
#define TAP_BRIDGE_H

#include <cstring>
#include "ns3/address.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/traced-callback.h"
#include "ns3/event-id.h"
#include "ns3/nstime.h"
#include "ns3/data-rate.h"
#include "ns3/ptr.h"
#include "ns3/mac48-address.h"
#include "ns3/unix-fd-reader.h"

namespace ns3 {

/**
 * \ingroup tap-bridge
 * Class to perform the actual reading from a socket
 */
class TapBridgeFdReader : public FdReader
{
private:
  FdReader::Data DoRead (void);
};

class Node;

/**
 * \ingroup tap-bridge
 *
 * \brief A bridge to make it appear that a real host process is connected to
 * an ns-3 net device.
 *
 * The Tap Bridge lives in a kind of a gray world somewhere between a
 * Linux host and an ns-3 bridge device.  From the Linux perspective,
 * this code appears as the user mode handler for a Tap net device.  That
 * is, when the Linux host writes to a /dev/tap device (that is either
 * manually or automatically created depending on basic operating mode
 * -- more on this later), the write is redirected into the TapBridge that
 * lives in the ns-3 world; and from this perspective, becomes a read.
 * In other words, a Linux process writes a packet to a tap device and
 * this packet is redirected to an ns-3 process where it is received by
 * the TapBridge as a result of a read operation there.  The TapBridge
 * then sends the packet to the ns-3 net device to which it is bridged.
 * In the other direction, a packet received by an ns-3 net device is
 * bridged to the TapBridge (it appears via a callback from that net
 * device.  The TapBridge then takes that packet and writes it back to
 * the host using the Linux TAP mechanism.  This write to the device will
 * then appear to the Linux host as if a packet has arrived on its
 * device.
 *
 * The upshot is that the Tap Bridge appears to bridge a tap device on a
 * Linux host in the "real world" to an ns-3 net device in the simulation
 * and make is appear that a ns-3 net device is actually installed in the
 * Linux host.  In order to do this on the ns-3 side, we need a "ghost
 * node" in the simulation to hold the bridged ns-3 net device and the
 * TapBridge.  This node should not actually do anything else in the
 * simulation since its job is simply to make the net device appear in
 * Linux.  This is not just arbitrary policy, it is because:
 *
 * - Bits sent to the Tap Bridge from higher layers in the ghost node (using
 *   the TapBridge Send() method) are completely ignored.  The Tap Bridge is
 *   not, itself, connected to any network, neither in Linux nor in ns-3;
 * - The bridged ns-3 net device is has had its receive callback disconnected
 *   from the ns-3 node and reconnected to the Tap Bridge.  All data received
 *   by a bridged device will be sent to the Linux host and will not be
 *   received by the node.  From the perspective of the ghost node, you can
 *   send over this device but you cannot ever receive.
 *
 * Of course, if you understand all of the issues you can take control of
 * your own destiny and do whatever you want -- we do not actively
 * prevent you from using the ghost node for anything you decide.  You
 * will be able to perform typical ns-3 operations on the ghost node if
 * you so desire.  The internet stack, for example, must be there and
 * functional on that node in order to participate in IP address
 * assignment and global routing.  However, as mentioned above,
 * interfaces talking any Tap Bridge or associated bridged net devices
 * will not work completely.  If you understand exactly what you are
 * doing, you can set up other interfaces and devices on the ghost node
 * and use them; or take advantage of the operational send side of the
 * bridged devices to create traffic generators.  We generally recommend
 * that you treat this node as a ghost of the Linux host and leave it to
 * itself, though.
 */
class TapBridge : public NetDevice
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Enumeration of the operating modes supported in the class.
   *
   */
  enum Mode {
    ILLEGAL,         /**< mode not set */
    CONFIGURE_LOCAL, /**< ns-3 creates and configures tap device */
    USE_LOCAL,       /**< ns-3 uses a pre-created tap, without configuring it */
    USE_BRIDGE, /**< ns-3 uses a pre-created tap, and bridges to a bridging net device */
  };

  TapBridge ();
  virtual ~TapBridge ();

  /**
   * \brief Get the bridged net device.
   *
   * The bridged net device is the ns-3 device to which this bridge is connected,
   *
   * \returns the bridged net device.
   */
  Ptr<NetDevice> GetBridgedNetDevice (void);

  /**
   * \brief Set the ns-3 net device to bridge.
   *
   * This method tells the bridge which ns-3 net device it should use to connect
   * the simulation side of the bridge.
   *
   * \param bridgedDevice device to set
   *
   * \attention The ns-3 net device that is being set as the device must have an
   * an IP address assigned to it before the simulation is run.  This address
   * will be used to set the hardware address of the host Linux device.
   */
  void SetBridgedNetDevice (Ptr<NetDevice> bridgedDevice);

  /**
   * \brief Set a start time for the device.
   *
   * The tap bridge consumes a non-trivial amount of time to start.  It starts
   * up in the context of a scheduled event to ensure that all configuration
   * has been completed before extracting the configuration (IP addresses, etc.)
   * In order to allow a more reasonable start-up sequence than a thundering
   * herd of devices, the time at which each device starts is also configurable
   * bot via the Attribute system and via this call.
   *
   * \param tStart the start time
   */
  void Start (Time tStart);

  /**
   * Set a stop time for the device.
   *
   * @param tStop the stop time
   *
   * \see TapBridge::Start
   */
  void Stop (Time tStop);

  /**
   * Set the operating mode of this device.
   *
   * \param mode The operating mode of this device.
   */
  void SetMode (TapBridge::Mode mode);

  /**
   * Get the operating mode of this device.
   *
   * \returns The operating mode of this device.
   */
  TapBridge::Mode  GetMode (void);

  //
  // The following methods are inherited from NetDevice base class and are
  // documented there.
  //
  virtual void SetIfIndex (const uint32_t index);
  virtual uint32_t GetIfIndex (void) const;
  virtual Ptr<Channel> GetChannel (void) const;
  virtual void SetAddress (Address address);
  virtual Address GetAddress (void) const;
  virtual bool SetMtu (const uint16_t mtu);
  virtual uint16_t GetMtu (void) const;
  virtual bool IsLinkUp (void) const;
  virtual void AddLinkChangeCallback (Callback<void> callback);
  virtual bool IsBroadcast (void) const;
  virtual Address GetBroadcast (void) const;
  virtual bool IsMulticast (void) const;
  virtual Address GetMulticast (Ipv4Address multicastGroup) const;
  virtual bool IsPointToPoint (void) const;
  virtual bool IsBridge (void) const;
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual Ptr<Node> GetNode (void) const;
  virtual void SetNode (Ptr<Node> node);
  virtual bool NeedsArp (void) const;
  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual void SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb);
  virtual bool SupportsSendFrom () const;
  virtual Address GetMulticast (Ipv6Address addr) const;

protected:
  /**
   * Call out to a separate process running as suid root in order to get our
   * tap device created.  We do this to avoid having the entire simulation
   * running as root.  If this method returns, we'll have a socket waiting
   * for us in m_sock that we can use to talk to the tap device.
   */
  virtual void DoDispose (void);

  /**
   * Receives a packet from a bridged Device
   * \param device the originating port
   * \param packet the received packet
   * \param protocol the packet protocol (e.g., Ethertype)
   * \param src the packet source
   * \param dst the packet destination
   * \param packetType the packet type (e.g., host, broadcast, etc.)
   * \returns true on success
   */
  bool ReceiveFromBridgedDevice (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol,
                                 Address const &src, Address const &dst, PacketType packetType);

  /**
   * Receives a packet from a bridged Device
   * \param device the originating port
   * \param packet the received packet
   * \param protocol the packet protocol (e.g., Ethertype)
   * \param src the packet source
   * \returns true on success
   */
  bool DiscardFromBridgedDevice (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol,
                                 Address const &src);

private:

  /**
   * Call out to a separate process running as suid root in order to get our
   * tap device created.  We do this to avoid having the entire simulation
   * running as root.  If this method returns, we'll have a socket waiting
   * for us in m_sock that we can use to talk to the tap device.
   */
  void CreateTap (void);

  /**
   * Spin up the device
   */
  void StartTapDevice (void);

  /**
   * Tear down the device
   */
  void StopTapDevice (void);

  /**
   * Callback to process packets that are read
   * \param buf input buffer
   * \param len input buffer length
   */
  void ReadCallback (uint8_t *buf, ssize_t len);

  /**
   * Forward a packet received from the tap device to the bridged ns-3
   * device
   *
   * \param buf A character buffer containing the actual packet bits that were
   *            received from the host.
   * \param len The length of the buffer.
   */
  void ForwardToBridgedDevice (uint8_t *buf, ssize_t len);

  /**
   * The host we are bridged to is in the evil real world.  Do some sanity
   * checking on a received packet to make sure it isn't too evil for our
   * poor naive virginal simulator to handle.
   *
   * \param packet The packet we received from the host, and which we need
   *               to check.
   * \param src    A pointer to the data structure that will get the source
   *               MAC address of the packet (extracted from the packet Ethernet
   *               header).
   * \param dst    A pointer to the data structure that will get the destination
   *               MAC address of the packet (extracted from the packet Ethernet
   *               header).
   * \param type   A pointer to the variable that will get the packet type from
   *               either the Ethernet header in the case of type interpretation
   *               (DIX framing) or from the 802.2 LLC header in the case of
   *               length interpretation (802.3 framing).
   * \returns the packet, or null if the packet has been filtered.
   */
  Ptr<Packet> Filter (Ptr<Packet> packet, Address *src, Address *dst, uint16_t *type);

  /**
   * Notifies that the link is up and ready.
   */
  void NotifyLinkUp (void);

  /**
   * Callback used to hook the standard packet receive callback of the TapBridge
   * ns-3 net device.  This is never called, and therefore no packets will ever
   * be received forwarded up the IP stack on the ghost node through this device.
   */
  NetDevice::ReceiveCallback m_rxCallback;

  /**
   * Callback used to hook the promiscuous packet receive callback of the TapBridge
   * ns-3 net device.  This is never called, and therefore no packets will ever
   * be received forwarded up the IP stack on the ghost node through this device.
   *
   * Note that we intercept the similar callback in the bridged device in order to
   * do the actual bridging between the bridged ns-3 net device and the Tap device
   * on the host.
   */
  NetDevice::PromiscReceiveCallback m_promiscRxCallback;

  /**
   * Pointer to the (ghost) Node to which we are connected.
   */
  Ptr<Node> m_node;


  /**
   * The ns-3 interface index of this TapBridge net device.
   */
  uint32_t m_ifIndex;

  /**
   * The common mtu to use for the net devices
   */
  uint16_t m_mtu;

  /**
   * The socket (actually interpreted as fd) to use to talk to the Tap device on
   * the real internet host.
   */
  int m_sock;

  /**
   * The ID of the ns-3 event used to schedule the start up of the underlying
   * host Tap device and ns-3 read thread.
   */
  EventId m_startEvent;

  /**
   * The ID of the ns-3 event used to schedule the tear down of the underlying
   * host Tap device and ns-3 read thread.
   */
  EventId m_stopEvent;

  /**
   * Includes the ns-3 read thread used to do blocking reads on the fd
   * corresponding to the host device.
   */
  Ptr<TapBridgeFdReader> m_fdReader;

  /**
   * The operating mode of the bridge.  Tells basically who creates and
   * configures the underlying network tap.
   */
  Mode m_mode;

  /**
   * The (unused) MAC address of the TapBridge net device.  Since the TapBridge
   * is implemented as a ns-3 net device, it is required to implement certain
   * functionality.  In this case, the TapBridge is automatically assigned a
   * MAC address, but it is not used.
   */
  Mac48Address m_address;

  /**
   * Time to start spinning up the device
   */
  Time m_tStart;

  /**
   * Time to start tearing down the device
   */
  Time m_tStop;

  /**
   * The name of the device to create on the host.  If the device name is the
   * empty string, we allow the host kernel to choose a name.
   */
  std::string m_tapDeviceName;

  /**
   * The IP address to use as the device default gateway on the host.
   */
  Ipv4Address m_tapGateway;

  /**
   * The IP address to use as the device IP on the host.
   */
  Ipv4Address m_tapIp;
  /**
   * The MAC address to use as the hardware address on the host; only used
   * in UseLocal mode.  This value comes from the MAC
   * address assigned to the bridged ns-3 net device and matches the MAC
   * address of the underlying network TAP which we configured to have the
   * same value.
   */
  Mac48Address m_tapMac;

  /**
   * The network mask to assign to the device created on the host.
   */
  Ipv4Mask m_tapNetmask;

  /**
   * The ns-3 net device to which we are bridging.
   */
  Ptr<NetDevice> m_bridgedDevice;
  /**
   * Whether the MAC address of the underlying ns-3 device has already been
   * rewritten is stored in this variable (for UseLocal/UseBridge mode only).
   */
  bool m_ns3AddressRewritten;

  /**
   * A 64K buffer to hold packet data while it is being sent.
   */
  uint8_t *m_packetBuffer;

  /**
   * a copy of the node id so the read thread doesn't have to GetNode() in
   * in order to find the node ID.  Thread unsafe reference counting in
   * multithreaded apps is not a good thing.
   */
  uint32_t m_nodeId;

  /**
   * Flag indicating whether or not the link is up.  In this case,
   * whether or not ns-3 is connected to the underlying TAP device
   * with a file descriptor.
   */
  bool m_linkUp;

  /**
   * Flag indicating whether or not the link is up.  In this case,
   * whether or not ns-3 is connected to the underlying TAP device
   * with a file descriptor.
   */
  bool m_verbose;

  /**
   * Callbacks to fire if the link changes state (up or down).
   */
  TracedCallback<> m_linkChangeCallbacks;
};

} // namespace ns3

#endif /* TAP_BRIDGE_H */
