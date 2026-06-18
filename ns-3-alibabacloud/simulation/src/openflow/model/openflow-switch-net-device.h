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
 *
 * Author: Blake Hurd  <naimorai@gmail.com>
 */

/**
 * \defgroup openflow OpenFlow Switch Device
 * This section documents the API of the ns-3 OpenFlow module. For a generic functional description, please refer to the ns-3 manual.
 */

#ifndef OPENFLOW_SWITCH_NET_DEVICE_H
#define OPENFLOW_SWITCH_NET_DEVICE_H

#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/mac48-address.h"

#include "ns3/ethernet-header.h"
#include "ns3/arp-header.h"
#include "ns3/tcp-header.h"
#include "ns3/udp-header.h"

#include "ns3/ipv4-l3-protocol.h"
#include "ns3/arp-l3-protocol.h"

#include "ns3/bridge-channel.h"
#include "ns3/node.h"
#include "ns3/enum.h"
#include "ns3/string.h"
#include "ns3/integer.h"
#include "ns3/uinteger.h"

#include <map>
#include <set>

#include "openflow-interface.h"

namespace ns3 {

/**
 * \ingroup openflow
 * \brief A net device that switches multiple LAN segments via an OpenFlow-compatible flow table
 *
 * The OpenFlowSwitchNetDevice object aggregates multiple netdevices as ports
 * and acts like a switch. It implements OpenFlow-compatibility,
 * according to the OpenFlow Switch Specification v0.8.9
 * <www.openflowswitch.org/documents/openflow-spec-v0.8.9.pdf>.
 * It implements a flow table that all received packets are run through.
 * It implements a connection to a controller via a subclass of the Controller class,
 * which can send messages to manipulate the flow table, thereby manipulating
 * how the OpenFlow switch behaves.
 *
 * There are two controllers available in the original package. DropController
 * builds a flow for each received packet to drop all packets it matches (this
 * demonstrates the flow table's basic implementation), and the LearningController
 * implements a "learning switch" algorithm (see 802.1D), where incoming unicast
 * frames from one port may occasionally be forwarded throughout all other ports,
 * but usually they are forwarded only to a single correct output port.
 *
 * \attention The Spanning Tree Protocol part of 802.1D is not
 * implemented.  Therefore, you have to be careful not to create
 * bridging loops, or else the network will collapse.
 *
 * \attention Each NetDevice used must only be assigned a Mac Address, adding it
 * to an Ipv4 or Ipv6 layer will cause an error. It also must support a SendFrom
 * call.
 */

/**
 * \ingroup openflow
 * \ingroup tests
 * \defgroup openflow-tests OpenFlow module tests
 */


/**
 * \ingroup openflow 
 * \brief A net device that switches multiple LAN segments via an OpenFlow-compatible flow table
 */
class OpenFlowSwitchNetDevice : public NetDevice
{
public:
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \name Descriptive Data
   * \brief OpenFlowSwitchNetDevice Description Data
   *
   * These four data describe the OpenFlowSwitchNetDevice as if it were
   * a real OpenFlow switch.
   *
   * There is a type of stats request that OpenFlow switches are supposed
   * to handle that returns the description of the OpenFlow switch. Currently
   * manufactured by "The ns-3 team", software description is "Simulated
   * OpenFlow Switch", and the other two are "N/A".
   * @{
   */
  /** \returns The descriptive string. */
  static const char * GetManufacturerDescription ();
  static const char * GetHardwareDescription ();
  static const char * GetSoftwareDescription ();
  static const char * GetSerialNumber ();
  /**@}*/

  OpenFlowSwitchNetDevice ();
  virtual ~OpenFlowSwitchNetDevice ();

  /**
   * \brief Set up the Switch's controller connection.
   *
   * \param c Pointer to a Controller.
   */
  void SetController (Ptr<ofi::Controller> c);

  /**
   * \brief Add a 'port' to a switch device
   *
   * This method adds a new switch port to a OpenFlowSwitchNetDevice, so that
   * the new switch port NetDevice becomes part of the switch and L2
   * frames start being forwarded to/from this NetDevice.
   *
   * \note The netdevice that is being added as switch port must
   * _not_ have an IP address.  In order to add IP connectivity to a
   * bridging node you must enable IP on the OpenFlowSwitchNetDevice itself,
   * never on its port netdevices.
   *
   * \param switchPort The port to add.
   * \return 0 if everything's ok, otherwise an error number.
   * Possible error numbers: EXFULL
   */
  int AddSwitchPort (Ptr<NetDevice> switchPort);

  /**
   * \brief Add a virtual port to a switch device
   *
   * The Ericsson OFSID has the concept of virtual ports and virtual
   * port tables. These are implemented in the OpenFlowSwitchNetDevice, but
   * don't have an understood use [perhaps it may have to do with
   * MPLS integration].
   *
   * Possible error numbers: EINVAL
   *
   * \param ovpm The data for adding a virtual port.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int AddVPort (const ofp_vport_mod *ovpm);

  /**
   * \brief Stats callback is ready for a dump.
   *
   * Controllers have a callback system for status requests which calls this function.
   *
   * \param cb_ The callback data.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int StatsDump (ofi::StatsDumpCallback *cb_);

  /**
   * \brief Stats callback is done.
   *
   * Controllers have a callback system for status requests which calls this function.
   *
   * \param cb_ The callback data.
   */
  void StatsDone (ofi::StatsDumpCallback *cb_);

  /**
   * \brief Called from the OpenFlow Interface to output the Packet on either a Port or the Controller
   *
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param in_port The index of the port the Packet was initially received on.
   * \param max_len The maximum number of bytes the caller wants to be sent; a value of 0 indicates the entire packet should be sent. Used when outputting to controller.
   * \param out_port The port we want to output on.
   * \param ignore_no_fwd If true, Ports that are set to not forward are forced to forward.
   */
  void DoOutput (uint32_t packet_uid, int in_port, size_t max_len, int out_port, bool ignore_no_fwd);

  /**
   * \brief The registered controller calls this method when sending a message to the switch.
   *
   * \param msg The message received from the controller.
   * \param length Length of the message.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int ForwardControlInput (const void *msg, size_t length);

  /**
   * \return The flow table chain.
   */
  sw_chain* GetChain ();

  /**
   * \return Number of switch ports attached to this switch.
   */
  uint32_t GetNSwitchPorts (void) const;

  /**
   * \param p The Port to get the index of.
   * \return The index of the provided Port.
   */
  int GetSwitchPortIndex (ofi::Port p);

  /**
   * \param n index of the Port.
   * \return The Port.
   */
  ofi::Port GetSwitchPort (uint32_t n) const;

  /**
   * \return The virtual port table.
   */
  vport_table_t GetVPortTable ();

  // From NetDevice
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
  virtual void DoDispose (void);

  /**
   * Called when a packet is received on one of the switch's ports.
   *
   * \param netdev The port the packet was received on.
   * \param packet The Packet itself.
   * \param protocol The protocol defining the Packet.
   * \param src The source address of the Packet.
   * \param dst The destination address of the Packet.
   * \param packetType Type of the packet.
   */
  void ReceiveFromDevice (Ptr<NetDevice> netdev, Ptr<const Packet> packet, uint16_t protocol, const Address& src, const Address& dst, PacketType packetType);

  /**
   * Takes a packet and generates an OpenFlow buffer from it, loading the packet data as well as its headers.
   *
   * \param packet The packet.
   * \param src The source address.
   * \param dst The destination address.
   * \param mtu The Maximum Transmission Unit.
   * \param protocol The protocol defining the packet.
   * \return The OpenFlow Buffer created from the packet.
   */
  ofpbuf * BufferFromPacket (Ptr<const Packet> packet, Address src, Address dst, int mtu, uint16_t protocol);

private:
  /**
   * Add a flow.
   *
   * Possible error numbers: ENOMEM, ENOBUFS, ESRCH
   *
   * \param ofm The flow data to add.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int AddFlow (const ofp_flow_mod *ofm);

  /**
   * Modify a flow.
   *
   * \param ofm The flow data to modify.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int ModFlow (const ofp_flow_mod *ofm);

  /**
   * Send packets out all the ports except the originating one
   *
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param in_port The index of the port the Packet was initially received on. This port doesn't forward when flooding.
   * \param flood If true, don't send out on the ports with flooding disabled.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int OutputAll (uint32_t packet_uid, int in_port, bool flood);

  /**
   * Sends a copy of the Packet over the provided output port
   *
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param out_port Output port.
   */
  void OutputPacket (uint32_t packet_uid, int out_port);

  /**
   * Seeks to send out a Packet over the provided output port. This is called generically
   * when we may or may not know the specific port we're outputting on. There are many
   * pre-set types of port options besides a Port that's hooked to our OpenFlowSwitchNetDevice.
   * For example, it could be outputting as a flood, or seeking to output to the controller.
   *
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param in_port The index of the port the Packet was initially received on.
   * \param out_port The port we want to output on.
   * \param ignore_no_fwd If true, Ports that are set to not forward are forced to forward.
   */
  void OutputPort (uint32_t packet_uid, int in_port, int out_port, bool ignore_no_fwd);

  /**
   * Sends a copy of the Packet to the controller. If the packet can be saved
   * in an OpenFlow buffer, then only the first 'max_len' bytes of the packet
   * are sent; otherwise, all of the packet is sent.
   *
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param in_port The index of the port the Packet was initially received on.
   * \param max_len The maximum number of bytes that the caller wants to be sent; a value of 0 indicates the entire packet should be sent.
   * \param reason Why the packet is being sent.
   */
  void OutputControl (uint32_t packet_uid, int in_port, size_t max_len, int reason);

  /**
   * If an error message happened during the controller's request, send it to the controller.
   *
   * \param type The type of error.
   * \param code The error code.
   * \param data The faulty data that lead to the error.
   * \param len The length of the faulty data.
   */
  void SendErrorMsg (uint16_t type, uint16_t code, const void *data, size_t len);

  /**
   * Send a reply about this OpenFlow switch's features to the controller.
   *
   * List of capabilities and actions to support are found in the specification
   * <www.openflowswitch.org/documents/openflow-spec-v0.8.9.pdf>.
   *
   * Supported capabilities and actions are defined in the openflow interface.
   * To recap, flow status, flow table status, port status, virtual port table
   * status can all be requested. It can also transmit over multiple physical
   * interfaces.
   *
   * It supports every action: outputting over a port, and all of the flow table
   * manipulation actions: setting the 802.1q VLAN ID, the 802.1q priority,
   * stripping the 802.1 header, setting the Ethernet source address and destination,
   * setting the IP source address and destination, setting the TCP/UDP source address
   * and destination, and setting the MPLS label and EXP bits.
   *
   * \attention Capabilities STP (Spanning Tree Protocol) and IP packet
   * reassembly are not currently supported.
   *
   */
  void SendFeaturesReply ();

  /**
   * Send a reply to the controller that a specific flow has expired.
   *
   * \param flow The flow that expired.
   * \param reason The reason for sending this expiration notification.
   */
  void SendFlowExpired (sw_flow *flow, enum ofp_flow_expired_reason reason);

  /**
   * Send a reply about a Port's status to the controller.
   *
   * \param p The port to get status from.
   * \param status The reason for sending this reply.
   */
  void SendPortStatus (ofi::Port p, uint8_t status);

  /**
   * Send a reply about this OpenFlow switch's virtual port table features to the controller.
   */
  void SendVPortTableFeatures ();

  /**
   * Send a message to the controller. This method is the key
   * to communicating with the controller, it does the actual
   * sending. The other Send methods call this one when they
   * are ready to send a message.
   *
   * \param buffer Buffer of the message to send out.
   * \return 0 if successful, otherwise an error number.
   */
  int SendOpenflowBuffer (ofpbuf *buffer);

  /**
   * Run the packet through the flow table. Looks up in the flow table for a match.
   * If it doesn't match, it forwards the packet to the registered controller, if the flag is set.
   *
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param port The port this packet was received over.
   * \param send_to_controller If set, sends to the controller if the packet isn't matched.
   */
  void RunThroughFlowTable (uint32_t packet_uid, int port, bool send_to_controller = true);

  /**
   * Run the packet through the vport table. As with AddVPort,
   * this doesn't have an understood use yet.
   *
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param port The port this packet was received over.
   * \param vport The virtual port this packet identifies itself by.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int RunThroughVPortTable (uint32_t packet_uid, int port, uint32_t vport);

  /**
   * Called by RunThroughFlowTable on a scheduled delay
   * to account for the flow table lookup overhead.
   *
   * \param key Matching key to look up in the flow table.
   * \param buffer Buffer of the packet received.
   * \param packet_uid Packet UID; used to fetch the packet and its metadata.
   * \param port The port the packet was received over.
   * \param send_to_controller 
   */
  void FlowTableLookup (sw_flow_key key, ofpbuf* buffer, uint32_t packet_uid, int port, bool send_to_controller);

  /**
   * Update the port status field of the switch port.
   * A non-zero return value indicates some field has changed.
   *
   * \param p A reference to a Port; used to change its config and flag fields.
   * \return true if the status of the Port is changed, false if unchanged (was already the right status).
   */
  int UpdatePortStatus (ofi::Port& p);

  /**
   * Fill out a description of the switch port.
   *
   * \param p The port to get the description from.
   * \param desc A pointer to the description message; used to fill the description message with the data from the port.
   */
  void FillPortDesc (ofi::Port p, ofp_phy_port *desc);

  /**
   * Generates an OpenFlow reply message based on the type.
   *
   * \param openflow_len Length of the reply to make.
   * \param type Type of reply message to make.
   * \param bufferp Message buffer; used to make the reply.
   * \return The OpenFlow reply message.
   */
  void* MakeOpenflowReply (size_t openflow_len, uint8_t type, ofpbuf **bufferp);

  /**
   * \name Receive Methods
   *
   * Actions to do when a specific OpenFlow message/packet is received
   *
   * @{
   */
  /**
   * \param msg The OpenFlow message received.
   * \return 0 if everything's ok, otherwise an error number.
   */
  int ReceiveFeaturesRequest (const void *msg);
  int ReceiveGetConfigRequest (const void *msg);
  int ReceiveSetConfig (const void *msg);
  int ReceivePacketOut (const void *msg);
  int ReceiveFlow (const void *msg);
  int ReceivePortMod (const void *msg);
  int ReceiveStatsRequest (const void *msg);
  int ReceiveEchoRequest (const void *msg);
  int ReceiveEchoReply (const void *msg);
  int ReceiveVPortMod (const void *msg);
  int ReceiveVPortTableFeaturesRequest (const void *msg);
  /**@}*/

  /// Rx Callback
  NetDevice::ReceiveCallback m_rxCallback;
  /// Promiscuopus Rx Callback
  NetDevice::PromiscReceiveCallback m_promiscRxCallback;

  Mac48Address m_address;               ///< Address of this device.
  Ptr<Node> m_node;                     ///< Node this device is installed on.
  Ptr<BridgeChannel> m_channel;         ///< Collection of port channels into the Switch Channel.
  uint32_t m_ifIndex;                   ///< Interface Index
  uint16_t m_mtu;                       ///< Maximum Transmission Unit

  /// PacketData type
  typedef std::map<uint32_t,ofi::SwitchPacketMetadata> PacketData_t;
  PacketData_t m_packetData;            ///< Packet data

  /// Switch's port type
  typedef std::vector<ofi::Port> Ports_t;
  Ports_t m_ports;                      ///< Switch's ports

  Ptr<ofi::Controller> m_controller;    ///< Connection to controller.

  uint64_t m_id;                        ///< Unique identifier for this switch, needed for OpenFlow
  Time m_lookupDelay;                   ///< Flow Table Lookup Delay [overhead].

  Time m_lastExecute;                   ///< Last time the periodic execution occurred.
  uint16_t m_flags;                     ///< Flags; configurable by the controller.
  uint16_t m_missSendLen;               ///< Flow Table Miss Send Length; configurable by the controller.

  sw_chain *m_chain;             ///< Flow Table; forwarding rules.
  vport_table_t m_vportTable;    ///< Virtual Port Table
};

} // namespace ns3

#endif /* OPENFLOW_SWITCH_NET_DEVICE_H */
