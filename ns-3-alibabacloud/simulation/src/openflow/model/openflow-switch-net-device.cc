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
#ifdef NS3_OPENFLOW

#include "openflow-switch-net-device.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/tcp-l4-protocol.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OpenFlowSwitchNetDevice");

NS_OBJECT_ENSURE_REGISTERED (OpenFlowSwitchNetDevice);

const char *
OpenFlowSwitchNetDevice::GetManufacturerDescription ()
{
  return "The ns-3 team";
}

const char *
OpenFlowSwitchNetDevice::GetHardwareDescription ()
{
  return "N/A";
}

const char *
OpenFlowSwitchNetDevice::GetSoftwareDescription ()
{
  return "Simulated OpenFlow Switch";
}

const char *
OpenFlowSwitchNetDevice::GetSerialNumber ()
{
  return "N/A";
}

static uint64_t
GenerateId ()
{
  uint8_t ea[ETH_ADDR_LEN];
  eth_addr_random (ea);
  return eth_addr_to_uint64 (ea);
}

TypeId
OpenFlowSwitchNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OpenFlowSwitchNetDevice")
    .SetParent<NetDevice> ()
    .SetGroupName ("Openflow")
    .AddConstructor<OpenFlowSwitchNetDevice> ()
    .AddAttribute ("ID",
                   "The identification of the OpenFlowSwitchNetDevice/Datapath, needed for OpenFlow compatibility.",
                   UintegerValue (GenerateId ()),
                   MakeUintegerAccessor (&OpenFlowSwitchNetDevice::m_id),
                   MakeUintegerChecker<uint64_t> ())
    .AddAttribute ("FlowTableLookupDelay",
                   "A real switch will have an overhead for looking up in the flow table. For the default, we simulate a standard TCAM on an FPGA.",
                   TimeValue (NanoSeconds (30)),
                   MakeTimeAccessor (&OpenFlowSwitchNetDevice::m_lookupDelay),
                   MakeTimeChecker ())
    .AddAttribute ("Flags", // Note: The Controller can configure this value, overriding the user's setting.
                   "Flags to turn different functionality on/off, such as whether to inform the controller when a flow expires, or how to handle fragments.",
                   UintegerValue (0), // Look at the ofp_config_flags enum in openflow/include/openflow.h for options.
                   MakeUintegerAccessor (&OpenFlowSwitchNetDevice::m_flags),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("FlowTableMissSendLength", // Note: The Controller can configure this value, overriding the user's setting.
                   "When forwarding a packet the switch didn't match up to the controller, it can be more efficient to forward only the first x bytes.",
                   UintegerValue (OFP_DEFAULT_MISS_SEND_LEN), // 128 bytes
                   MakeUintegerAccessor (&OpenFlowSwitchNetDevice::m_missSendLen),
                   MakeUintegerChecker<uint16_t> ())
  ;
  return tid;
}

OpenFlowSwitchNetDevice::OpenFlowSwitchNetDevice ()
  : m_node (0),
    m_ifIndex (0),
    m_mtu (0xffff)
{
  NS_LOG_FUNCTION_NOARGS ();

  m_channel = CreateObject<BridgeChannel> ();

  time_init (); // OFSI's clock; needed to use the buffer storage system.
  // m_lastTimeout = time_now ();

  m_controller = 0;
  // m_listenPVConn = 0;

  m_chain = chain_create ();
  if (m_chain == 0)
    {
      NS_LOG_ERROR ("Not enough memory to create the flow table.");
    }

  m_ports.reserve (DP_MAX_PORTS);
  vport_table_init (&m_vportTable);
}

OpenFlowSwitchNetDevice::~OpenFlowSwitchNetDevice ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
OpenFlowSwitchNetDevice::DoDispose ()
{
  NS_LOG_FUNCTION_NOARGS ();

  for (Ports_t::iterator b = m_ports.begin (), e = m_ports.end (); b != e; b++)
    {
      SendPortStatus (*b, OFPPR_DELETE);
      b->netdev = 0;
    }
  m_ports.clear ();

  m_controller = 0;

  chain_destroy (m_chain);
  RBTreeDestroy (m_vportTable.table);
  m_channel = 0;
  m_node = 0;
  NetDevice::DoDispose ();
}

void
OpenFlowSwitchNetDevice::SetController (Ptr<ofi::Controller> c)
{
  if (m_controller != 0)
    {
      NS_LOG_ERROR ("Controller already set.");
      return;
    }

  m_controller = c;
  m_controller->AddSwitch (this);
}

int
OpenFlowSwitchNetDevice::AddSwitchPort (Ptr<NetDevice> switchPort)
{
  NS_LOG_FUNCTION_NOARGS ();
  NS_ASSERT (switchPort != this);
  if (!Mac48Address::IsMatchingType (switchPort->GetAddress ()))
    {
      NS_FATAL_ERROR ("Device does not support eui 48 addresses: cannot be added to switch.");
    }
  if (!switchPort->SupportsSendFrom ())
    {
      NS_FATAL_ERROR ("Device does not support SendFrom: cannot be added to switch.");
    }
  if (m_address == Mac48Address ())
    {
      m_address = Mac48Address::ConvertFrom (switchPort->GetAddress ());
    }

  if (m_ports.size () < DP_MAX_PORTS)
    {
      ofi::Port p;
      p.config = 0;
      p.netdev = switchPort;
      m_ports.push_back (p);

      // Notify the controller that this port has been added
      SendPortStatus (p, OFPPR_ADD);

      NS_LOG_DEBUG ("RegisterProtocolHandler for " << switchPort->GetInstanceTypeId ().GetName ());
      m_node->RegisterProtocolHandler (MakeCallback (&OpenFlowSwitchNetDevice::ReceiveFromDevice, this),
                                       0, switchPort, true);
      m_channel->AddChannel (switchPort->GetChannel ());
    }
  else
    {
      return EXFULL;
    }

  return 0;
}

void
OpenFlowSwitchNetDevice::SetIfIndex (const uint32_t index)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_ifIndex = index;
}

uint32_t
OpenFlowSwitchNetDevice::GetIfIndex (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_ifIndex;
}

Ptr<Channel>
OpenFlowSwitchNetDevice::GetChannel (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_channel;
}

void
OpenFlowSwitchNetDevice::SetAddress (Address address)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_address = Mac48Address::ConvertFrom (address);
}

Address
OpenFlowSwitchNetDevice::GetAddress (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_address;
}

bool
OpenFlowSwitchNetDevice::SetMtu (const uint16_t mtu)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_mtu = mtu;
  return true;
}

uint16_t
OpenFlowSwitchNetDevice::GetMtu (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_mtu;
}


bool
OpenFlowSwitchNetDevice::IsLinkUp (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}


void
OpenFlowSwitchNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
}

bool
OpenFlowSwitchNetDevice::IsBroadcast (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

Address
OpenFlowSwitchNetDevice::GetBroadcast (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return Mac48Address ("ff:ff:ff:ff:ff:ff");
}

bool
OpenFlowSwitchNetDevice::IsMulticast (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

Address
OpenFlowSwitchNetDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  NS_LOG_FUNCTION (this << multicastGroup);
  Mac48Address multicast = Mac48Address::GetMulticast (multicastGroup);
  return multicast;
}


bool
OpenFlowSwitchNetDevice::IsPointToPoint (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return false;
}

bool
OpenFlowSwitchNetDevice::IsBridge (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

void
OpenFlowSwitchNetDevice::DoOutput (uint32_t packet_uid, int in_port, size_t max_len, int out_port, bool ignore_no_fwd)
{
  if (out_port != OFPP_CONTROLLER)
    {
      OutputPort (packet_uid, in_port, out_port, ignore_no_fwd);
    }
  else
    {
      OutputControl (packet_uid, in_port, max_len, OFPR_ACTION);
    }
}

bool
OpenFlowSwitchNetDevice::Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION_NOARGS ();
  return SendFrom (packet, m_address, dest, protocolNumber);
}

bool
OpenFlowSwitchNetDevice::SendFrom (Ptr<Packet> packet, const Address& src, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION_NOARGS ();

  ofpbuf *buffer = BufferFromPacket (packet,src,dest,GetMtu (),protocolNumber);

  uint32_t packet_uid = save_buffer (buffer);
  ofi::SwitchPacketMetadata data;
  data.packet = packet;
  data.buffer = buffer;
  data.protocolNumber = protocolNumber;
  data.src = Address (src);
  data.dst = Address (dest);
  m_packetData.insert (std::make_pair (packet_uid, data));

  RunThroughFlowTable (packet_uid, -1);

  return true;
}


Ptr<Node>
OpenFlowSwitchNetDevice::GetNode (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_node;
}

void
OpenFlowSwitchNetDevice::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_node = node;
}

bool
OpenFlowSwitchNetDevice::NeedsArp (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

void
OpenFlowSwitchNetDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_rxCallback = cb;
}

void
OpenFlowSwitchNetDevice::SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_promiscRxCallback = cb;
}

bool
OpenFlowSwitchNetDevice::SupportsSendFrom () const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

Address
OpenFlowSwitchNetDevice::GetMulticast (Ipv6Address addr) const
{
  NS_LOG_FUNCTION (this << addr);
  return Mac48Address::GetMulticast (addr);
}

// Add a virtual port table entry.
int
OpenFlowSwitchNetDevice::AddVPort (const ofp_vport_mod *ovpm)
{
  size_t actions_len = ntohs (ovpm->header.length) - sizeof *ovpm;
  unsigned int vport = ntohl (ovpm->vport);
  unsigned int parent_port = ntohl (ovpm->parent_port);

  // check whether port table entry exists for specified port number
  vport_table_entry *vpe = vport_table_lookup (&m_vportTable, vport);
  if (vpe != 0)
    {
      NS_LOG_ERROR ("vport " << vport << " already exists!");
      SendErrorMsg (OFPET_BAD_ACTION, OFPET_VPORT_MOD_FAILED, ovpm, ntohs (ovpm->header.length));
      return EINVAL;
    }

  // check whether actions are valid
  uint16_t v_code = ofi::ValidateVPortActions (ovpm->actions, actions_len);
  if (v_code != ACT_VALIDATION_OK)
    {
      SendErrorMsg (OFPET_BAD_ACTION, v_code, ovpm, ntohs (ovpm->header.length));
      return EINVAL;
    }

  vpe = vport_table_entry_alloc (actions_len);

  vpe->vport = vport;
  vpe->parent_port = parent_port;
  if (vport < OFPP_VP_START || vport > OFPP_VP_END)
    {
      NS_LOG_ERROR ("port " << vport << " is not in the virtual port range (" << OFPP_VP_START << "-" << OFPP_VP_END << ")");
      SendErrorMsg (OFPET_BAD_ACTION, OFPET_VPORT_MOD_FAILED, ovpm, ntohs (ovpm->header.length));
      free_vport_table_entry (vpe); // free allocated entry
      return EINVAL;
    }

  vpe->port_acts->actions_len = actions_len;
  memcpy (vpe->port_acts->actions, ovpm->actions, actions_len);

  int error = insert_vport_table_entry (&m_vportTable, vpe);
  if (error)
    {
      NS_LOG_ERROR ("could not insert port table entry for port " << vport);
    }

  return error;
}

ofpbuf *
OpenFlowSwitchNetDevice::BufferFromPacket (Ptr<const Packet> constPacket, Address src, Address dst, int mtu, uint16_t protocol)
{
  NS_LOG_INFO ("Creating Openflow buffer from packet.");

  Ptr<Packet> packet = constPacket->Copy ();
  /*
   * Allocate buffer with some headroom to add headers in forwarding
   * to the controller or adding a vlan tag, plus an extra 2 bytes to
   * allow IP headers to be aligned on a 4-byte boundary.
   */
  const int headroom = 128 + 2;
  const int hard_header = VLAN_ETH_HEADER_LEN;
  ofpbuf *buffer = ofpbuf_new (headroom + hard_header + mtu);
  buffer->data = (char*)buffer->data + headroom + hard_header;

  int l2_length = 0, l3_length = 0, l4_length = 0;

  //Parse Ethernet header
  buffer->l2 = new eth_header;
  eth_header* eth_h = (eth_header*)buffer->l2;
  dst.CopyTo (eth_h->eth_dst);              // Destination Mac Address
  src.CopyTo (eth_h->eth_src);              // Source Mac Address
  if (protocol == ArpL3Protocol::PROT_NUMBER)
    {
      eth_h->eth_type = htons (ETH_TYPE_ARP);    // Ether Type
    }
  else if (protocol == Ipv4L3Protocol::PROT_NUMBER)
    {
      eth_h->eth_type = htons (ETH_TYPE_IP);    // Ether Type
    }
  else
    {
      NS_LOG_WARN ("Protocol unsupported: " << protocol);
    }
  NS_LOG_INFO ("Parsed EthernetHeader");

  l2_length = ETH_HEADER_LEN;

  // We have to wrap this because PeekHeader has an assert fail if we check for an Ipv4Header that isn't there.
  if (protocol == Ipv4L3Protocol::PROT_NUMBER)
    {
      Ipv4Header ip_hd;
      if (packet->PeekHeader (ip_hd))
        {
          buffer->l3 = new ip_header;
          ip_header* ip_h = (ip_header*)buffer->l3;
          ip_h->ip_ihl_ver  = IP_IHL_VER (5, IP_VERSION);       // Version
          ip_h->ip_tos      = ip_hd.GetTos ();                  // Type of Service/Differentiated Services
          ip_h->ip_tot_len  = packet->GetSize ();               // Total Length
          ip_h->ip_id       = ip_hd.GetIdentification ();       // Identification
          ip_h->ip_frag_off = ip_hd.GetFragmentOffset ();       // Fragment Offset
          ip_h->ip_ttl      = ip_hd.GetTtl ();                  // Time to Live
          ip_h->ip_proto    = ip_hd.GetProtocol ();             // Protocol
          ip_h->ip_src      = htonl (ip_hd.GetSource ().Get ()); // Source Address
          ip_h->ip_dst      = htonl (ip_hd.GetDestination ().Get ()); // Destination Address
          ip_h->ip_csum     = csum (&ip_h, sizeof ip_h);        // Header Checksum
          NS_LOG_INFO ("Parsed Ipv4Header");
          packet->RemoveHeader (ip_hd);

          l3_length = IP_HEADER_LEN;
        }
    }
  else
    {
      // ARP Packet; the underlying OpenFlow header isn't used to match, so this is probably superfluous.
      ArpHeader arp_hd;
      if (packet->PeekHeader (arp_hd))
        {
          buffer->l3 = new arp_eth_header;
          arp_eth_header* arp_h = (arp_eth_header*)buffer->l3;
          arp_h->ar_hrd = ARP_HRD_ETHERNET;                             // Hardware type.
          arp_h->ar_pro = ARP_PRO_IP;                                   // Protocol type.
          arp_h->ar_op = arp_hd.m_type;                                 // Opcode.
          arp_hd.GetDestinationHardwareAddress ().CopyTo (arp_h->ar_tha); // Target hardware address.
          arp_hd.GetSourceHardwareAddress ().CopyTo (arp_h->ar_sha);    // Sender hardware address.
          arp_h->ar_tpa = arp_hd.GetDestinationIpv4Address ().Get ();   // Target protocol address.
          arp_h->ar_spa = arp_hd.GetSourceIpv4Address ().Get ();        // Sender protocol address.
          arp_h->ar_hln = sizeof arp_h->ar_tha;                         // Hardware address length.
          arp_h->ar_pln = sizeof arp_h->ar_tpa;                         // Protocol address length.
          NS_LOG_INFO ("Parsed ArpHeader");
          packet->RemoveHeader (arp_hd);

          l3_length = ARP_ETH_HEADER_LEN;
        }
    }

  if (protocol == Ipv4L3Protocol::PROT_NUMBER)
    {
      ip_header* ip_h = (ip_header*)buffer->l3;
      if (ip_h->ip_proto == TcpL4Protocol::PROT_NUMBER)
        {
          TcpHeader tcp_hd;
          if (packet->PeekHeader (tcp_hd))
            {
              buffer->l4 = new tcp_header;
              tcp_header* tcp_h = (tcp_header*)buffer->l4;
              tcp_h->tcp_src = htons (tcp_hd.GetSourcePort ());         // Source Port
              tcp_h->tcp_dst = htons (tcp_hd.GetDestinationPort ());    // Destination Port
              tcp_h->tcp_seq = tcp_hd.GetSequenceNumber ().GetValue (); // Sequence Number
              tcp_h->tcp_ack = tcp_hd.GetAckNumber ().GetValue ();      // ACK Number
              tcp_h->tcp_ctl = TCP_FLAGS (tcp_hd.GetFlags ());  // Data Offset + Reserved + Flags
              tcp_h->tcp_winsz = tcp_hd.GetWindowSize ();       // Window Size
              tcp_h->tcp_urg = tcp_hd.GetUrgentPointer ();      // Urgent Pointer
              tcp_h->tcp_csum = csum (&tcp_h, sizeof tcp_h);    // Header Checksum
              NS_LOG_INFO ("Parsed TcpHeader");
              packet->RemoveHeader (tcp_hd);

              l4_length = TCP_HEADER_LEN;
            }
        }
      else if (ip_h->ip_proto == UdpL4Protocol::PROT_NUMBER)
        {
          UdpHeader udp_hd;
          if (packet->PeekHeader (udp_hd))
            {
              buffer->l4 = new udp_header;
              udp_header* udp_h = (udp_header*)buffer->l4;
              udp_h->udp_src = htons (udp_hd.GetSourcePort ());     // Source Port
              udp_h->udp_dst = htons (udp_hd.GetDestinationPort ()); // Destination Port
              udp_h->udp_len = htons (UDP_HEADER_LEN + packet->GetSize ());

              ip_header* ip_h = (ip_header*)buffer->l3;
              uint32_t udp_csum = csum_add32 (0, ip_h->ip_src);
              udp_csum = csum_add32 (udp_csum, ip_h->ip_dst);
              udp_csum = csum_add16 (udp_csum, IP_TYPE_UDP << 8);
              udp_csum = csum_add16 (udp_csum, udp_h->udp_len);
              udp_csum = csum_continue (udp_csum, udp_h, sizeof udp_h);
              udp_h->udp_csum = csum_finish (csum_continue (udp_csum, buffer->data, buffer->size)); // Header Checksum
              NS_LOG_INFO ("Parsed UdpHeader");
              packet->RemoveHeader (udp_hd);

              l4_length = UDP_HEADER_LEN;
            }
        }
    }

  // Load any remaining packet data into buffer data
  packet->CopyData ((uint8_t*)buffer->data, packet->GetSize ());

  if (buffer->l4)
    {
      ofpbuf_push (buffer, buffer->l4, l4_length);
      delete (tcp_header*)buffer->l4;
    }
  if (buffer->l3)
    {
      ofpbuf_push (buffer, buffer->l3, l3_length);
      delete (ip_header*)buffer->l3;
    }
  if (buffer->l2)
    {
      ofpbuf_push (buffer, buffer->l2, l2_length);
      delete (eth_header*)buffer->l2;
    }

  return buffer;
}

void
OpenFlowSwitchNetDevice::ReceiveFromDevice (Ptr<NetDevice> netdev, Ptr<const Packet> packet, uint16_t protocol,
                                            const Address& src, const Address& dst, PacketType packetType)
{
  NS_LOG_FUNCTION_NOARGS ();
  NS_LOG_INFO ("--------------------------------------------");
  NS_LOG_DEBUG ("UID is " << packet->GetUid ());

  if (!m_promiscRxCallback.IsNull ())
    {
      m_promiscRxCallback (this, packet, protocol, src, dst, packetType);
    }

  Mac48Address dst48 = Mac48Address::ConvertFrom (dst);
  NS_LOG_INFO ("Received packet from " << Mac48Address::ConvertFrom (src) << " looking for " << dst48);

  for (size_t i = 0; i < m_ports.size (); i++)
    {
      if (m_ports[i].netdev == netdev)
        {
          if (packetType == PACKET_HOST && dst48 == m_address)
            {
              m_rxCallback (this, packet, protocol, src);
            }
          else if (packetType == PACKET_BROADCAST || packetType == PACKET_MULTICAST || packetType == PACKET_OTHERHOST)
            {
              if (packetType == PACKET_OTHERHOST && dst48 == m_address)
                {
                  m_rxCallback (this, packet, protocol, src);
                }
              else
                {
                  if (packetType != PACKET_OTHERHOST)
                    {
                      m_rxCallback (this, packet, protocol, src);
                    }

                  ofi::SwitchPacketMetadata data;
                  data.packet = packet->Copy ();

                  ofpbuf *buffer = BufferFromPacket (data.packet,src,dst,netdev->GetMtu (),protocol);
                  m_ports[i].rx_packets++;
                  m_ports[i].rx_bytes += buffer->size;
                  data.buffer = buffer;
                  uint32_t packet_uid = save_buffer (buffer);

                  data.protocolNumber = protocol;
                  data.src = Address (src);
                  data.dst = Address (dst);
                  m_packetData.insert (std::make_pair (packet_uid, data));

                  RunThroughFlowTable (packet_uid, i);
                }
            }

          break;
        }
    }

  // Run periodic execution.
  Time now = Simulator::Now ();
  if (now >= Seconds (m_lastExecute.GetSeconds () + 1)) // If a second or more has passed from the simulation time, execute.
    {
      // If port status is modified in any way, notify the controller.
      for (size_t i = 0; i < m_ports.size (); i++)
        {
          if (UpdatePortStatus (m_ports[i]))
            {
              SendPortStatus (m_ports[i], OFPPR_MODIFY);
            }
        }

      // If any flows have expired, delete them and notify the controller.
      List deleted = LIST_INITIALIZER (&deleted);
      sw_flow *f, *n;
      chain_timeout (m_chain, &deleted);
      LIST_FOR_EACH_SAFE (f, n, sw_flow, node, &deleted)
      {
        std::ostringstream str;
        str << "Flow [";
        for (int i = 0; i < 6; i++)
          str << (i!=0 ? ":" : "") << std::hex << f->key.flow.dl_src[i]/16 << f->key.flow.dl_src[i]%16;
        str << " -> ";
        for (int i = 0; i < 6; i++)
          str << (i!=0 ? ":" : "") << std::hex << f->key.flow.dl_dst[i]/16 << f->key.flow.dl_dst[i]%16;
        str <<  "] expired.";
	
        NS_LOG_INFO (str.str ());
        SendFlowExpired (f, (ofp_flow_expired_reason)f->reason);
        list_remove (&f->node);
        flow_free (f);
      }

      m_lastExecute = now;
    }
}

int
OpenFlowSwitchNetDevice::OutputAll (uint32_t packet_uid, int in_port, bool flood)
{
  NS_LOG_FUNCTION_NOARGS ();
  NS_LOG_INFO ("Flooding over ports.");

  int prev_port = -1;
  for (size_t i = 0; i < m_ports.size (); i++)
    {
      if (i == (unsigned)in_port) // Originating port
        {
          continue;
        }
      if (flood && m_ports[i].config & OFPPC_NO_FLOOD) // Port configured to not allow flooding
        {
          continue;
        }
      if (prev_port != -1)
        {
          OutputPort (packet_uid, in_port, prev_port, false);
        }
      prev_port = i;
    }
  if (prev_port != -1)
    {
      OutputPort (packet_uid, in_port, prev_port, false);
    }

  return 0;
}

void
OpenFlowSwitchNetDevice::OutputPacket (uint32_t packet_uid, int out_port)
{
  if (out_port >= 0 && out_port < DP_MAX_PORTS)
    {
      ofi::Port& p = m_ports[out_port];
      if (p.netdev != 0 && !(p.config & OFPPC_PORT_DOWN))
        {
          ofi::SwitchPacketMetadata data = m_packetData.find (packet_uid)->second;
          size_t bufsize = data.buffer->size;
          NS_LOG_INFO ("Sending packet " << data.packet->GetUid () << " over port " << out_port);
          if (p.netdev->SendFrom (data.packet->Copy (), data.src, data.dst, data.protocolNumber))
            {
              p.tx_packets++;
              p.tx_bytes += bufsize;
            }
          else
            {
              p.tx_dropped++;
            }
          return;
        }
    }

  NS_LOG_DEBUG ("can't forward to bad port " << out_port);
}

void
OpenFlowSwitchNetDevice::OutputPort (uint32_t packet_uid, int in_port, int out_port, bool ignore_no_fwd)
{
  NS_LOG_FUNCTION_NOARGS ();

  if (out_port == OFPP_FLOOD)
    {
      OutputAll (packet_uid, in_port, true);
    }
  else if (out_port == OFPP_ALL)
    {
      OutputAll (packet_uid, in_port, false);
    }
  else if (out_port == OFPP_CONTROLLER)
    {
      OutputControl (packet_uid, in_port, 0, OFPR_ACTION);
    }
  else if (out_port == OFPP_IN_PORT)
    {
      OutputPacket (packet_uid, in_port);
    }
  else if (out_port == OFPP_TABLE)
    {
      RunThroughFlowTable (packet_uid, in_port < DP_MAX_PORTS ? in_port : -1, false);
    }
  else if (out_port >= OFPP_VP_START && out_port <= OFPP_VP_END)
    {
      // port is a virtual port
      NS_LOG_INFO ("packet sent to virtual port " << out_port);
      if (in_port < DP_MAX_PORTS)
        {
          RunThroughVPortTable (packet_uid, in_port, out_port);
        }
      else
        {
          RunThroughVPortTable (packet_uid, -1, out_port);
        }
    }
  else if (in_port == out_port)
    {
      NS_LOG_DEBUG ("can't directly forward to input port");
    }
  else
    {
      OutputPacket (packet_uid, out_port);
    }
}

void*
OpenFlowSwitchNetDevice::MakeOpenflowReply (size_t openflow_len, uint8_t type, ofpbuf **bufferp)
{
  return make_openflow_xid (openflow_len, type, 0, bufferp);
}

int
OpenFlowSwitchNetDevice::SendOpenflowBuffer (ofpbuf *buffer)
{
  if (m_controller != 0)
    {
      update_openflow_length (buffer);
      m_controller->ReceiveFromSwitch (this, buffer);
    }

  return 0;
}

void
OpenFlowSwitchNetDevice::OutputControl (uint32_t packet_uid, int in_port, size_t max_len, int reason)
{
  NS_LOG_INFO ("Sending packet to controller");

  ofpbuf* buffer = m_packetData.find (packet_uid)->second.buffer;
  size_t total_len = buffer->size;
  if (packet_uid != std::numeric_limits<uint32_t>::max () && max_len != 0 && buffer->size > max_len)
    {
      buffer->size = max_len;
    }

  ofp_packet_in *opi = (ofp_packet_in*)ofpbuf_push_uninit (buffer, offsetof (ofp_packet_in, data));
  opi->header.version = OFP_VERSION;
  opi->header.type    = OFPT_PACKET_IN;
  opi->header.length  = htons (buffer->size);
  opi->header.xid     = htonl (0);
  opi->buffer_id      = htonl (packet_uid);
  opi->total_len      = htons (total_len);
  opi->in_port        = htons (in_port);
  opi->reason         = reason;
  opi->pad            = 0;
  SendOpenflowBuffer (buffer);
}

void
OpenFlowSwitchNetDevice::FillPortDesc (ofi::Port p, ofp_phy_port *desc)
{
  desc->port_no = htons (GetSwitchPortIndex (p));

  std::ostringstream nm;
  nm << "eth" << GetSwitchPortIndex (p);
  strncpy ((char *)desc->name, nm.str ().c_str (), sizeof desc->name);

  p.netdev->GetAddress ().CopyTo (desc->hw_addr);
  desc->config = htonl (p.config);
  desc->state = htonl (p.state);

  /// \todo This should probably be fixed eventually to specify different available features.
  desc->curr = 0; // htonl(netdev_get_features(p->netdev, NETDEV_FEAT_CURRENT));
  desc->supported = 0; // htonl(netdev_get_features(p->netdev, NETDEV_FEAT_SUPPORTED));
  desc->advertised = 0; // htonl(netdev_get_features(p->netdev, NETDEV_FEAT_ADVERTISED));
  desc->peer = 0; // htonl(netdev_get_features(p->netdev, NETDEV_FEAT_PEER));
}

void
OpenFlowSwitchNetDevice::SendFeaturesReply ()
{
  ofpbuf *buffer;
  ofp_switch_features *ofr = (ofp_switch_features*)MakeOpenflowReply (sizeof *ofr, OFPT_FEATURES_REPLY, &buffer);
  ofr->datapath_id  = htonll (m_id);
  ofr->n_tables     = m_chain->n_tables;
  ofr->n_buffers    = htonl (N_PKT_BUFFERS);
  ofr->capabilities = htonl (OFP_SUPPORTED_CAPABILITIES);
  ofr->actions      = htonl (OFP_SUPPORTED_ACTIONS);

  for (size_t i = 0; i < m_ports.size (); i++)
    {
      ofp_phy_port* opp = (ofp_phy_port*)ofpbuf_put_zeros (buffer, sizeof *opp);
      FillPortDesc (m_ports[i], opp);
    }

  SendOpenflowBuffer (buffer);
}

void
OpenFlowSwitchNetDevice::SendVPortTableFeatures ()
{
  ofpbuf *buffer;
  ofp_vport_table_features *ovtfr = (ofp_vport_table_features*)MakeOpenflowReply (sizeof *ovtfr, OFPT_VPORT_TABLE_FEATURES_REPLY, &buffer);
  ovtfr->actions = htonl (OFP_SUPPORTED_VPORT_TABLE_ACTIONS);
  ovtfr->max_vports = htonl (m_vportTable.max_vports);
  ovtfr->max_chain_depth = htons (-1); // support a chain depth of 2^16
  ovtfr->mixed_chaining = true;
  SendOpenflowBuffer (buffer);
}

int
OpenFlowSwitchNetDevice::UpdatePortStatus (ofi::Port& p)
{
  uint32_t orig_config = p.config;
  uint32_t orig_state = p.state;

  // Port is always enabled because the Net Device is always enabled.
  p.config &= ~OFPPC_PORT_DOWN;

  if (p.netdev->IsLinkUp ())
    {
      p.state &= ~OFPPS_LINK_DOWN;
    }
  else
    {
      p.state |= OFPPS_LINK_DOWN;
    }

  return ((orig_config != p.config) || (orig_state != p.state));
}

void
OpenFlowSwitchNetDevice::SendPortStatus (ofi::Port p, uint8_t status)
{
  ofpbuf *buffer;
  ofp_port_status *ops = (ofp_port_status*)MakeOpenflowReply (sizeof *ops, OFPT_PORT_STATUS, &buffer);
  ops->reason = status;
  memset (ops->pad, 0, sizeof ops->pad);
  FillPortDesc (p, &ops->desc);

  SendOpenflowBuffer (buffer);
  ofpbuf_delete (buffer);
}

void
OpenFlowSwitchNetDevice::SendFlowExpired (sw_flow *flow, enum ofp_flow_expired_reason reason)
{
  ofpbuf *buffer;
  ofp_flow_expired *ofe = (ofp_flow_expired*)MakeOpenflowReply (sizeof *ofe, OFPT_FLOW_EXPIRED, &buffer);
  flow_fill_match (&ofe->match, &flow->key);

  ofe->priority = htons (flow->priority);
  ofe->reason = reason;
  memset (ofe->pad, 0, sizeof ofe->pad);

  ofe->duration     = htonl (time_now () - flow->created);
  memset (ofe->pad2, 0, sizeof ofe->pad2);
  ofe->packet_count = htonll (flow->packet_count);
  ofe->byte_count   = htonll (flow->byte_count);
  SendOpenflowBuffer (buffer);
}

void
OpenFlowSwitchNetDevice::SendErrorMsg (uint16_t type, uint16_t code, const void *data, size_t len)
{
  ofpbuf *buffer;
  ofp_error_msg *oem = (ofp_error_msg*)MakeOpenflowReply (sizeof(*oem) + len, OFPT_ERROR, &buffer);
  oem->type = htons (type);
  oem->code = htons (code);
  memcpy (oem->data, data, len);
  SendOpenflowBuffer (buffer);
}

void
OpenFlowSwitchNetDevice::FlowTableLookup (sw_flow_key key, ofpbuf* buffer, uint32_t packet_uid, int port, bool send_to_controller)
{
  sw_flow *flow = chain_lookup (m_chain, &key);
  if (flow != 0)
    {
      NS_LOG_INFO ("Flow matched");
      flow_used (flow, buffer);
      ofi::ExecuteActions (this, packet_uid, buffer, &key, flow->sf_acts->actions, flow->sf_acts->actions_len, false);
    }
  else
    {
      NS_LOG_INFO ("Flow not matched.");

      if (send_to_controller)
        {
          OutputControl (packet_uid, port, m_missSendLen, OFPR_NO_MATCH);
        }
    }

  // Clean up; at this point we're done with the packet.
  m_packetData.erase (packet_uid);
  discard_buffer (packet_uid);
  ofpbuf_delete (buffer);
}

void
OpenFlowSwitchNetDevice::RunThroughFlowTable (uint32_t packet_uid, int port, bool send_to_controller)
{
  ofi::SwitchPacketMetadata data = m_packetData.find (packet_uid)->second;
  ofpbuf* buffer = data.buffer;

  sw_flow_key key;
  key.wildcards = 0; // Lookup cannot take wildcards.
  // Extract the matching key's flow data from the packet's headers; if the policy is to drop fragments and the message is a fragment, drop it.
  if (flow_extract (buffer, port != -1 ? port : OFPP_NONE, &key.flow) && (m_flags & OFPC_FRAG_MASK) == OFPC_FRAG_DROP)
    {
      ofpbuf_delete (buffer);
      return;
    }

  // drop MPLS packets with TTL 1
  if (buffer->l2_5)
    {
      mpls_header mpls_h;
      mpls_h.value = ntohl (*((uint32_t*)buffer->l2_5));
      if (mpls_h.ttl == 1)
        {
          // increment mpls drop counter
          if (port != -1)
            {
              m_ports[port].mpls_ttl0_dropped++;
            }
          return;
        }
    }

  // If we received the packet on a port, and opted not to receive any messages from it...
  if (port != -1)
    {
      uint32_t config = m_ports[port].config;
      if (config & (OFPPC_NO_RECV | OFPPC_NO_RECV_STP)
          && config & (!eth_addr_equals (key.flow.dl_dst, stp_eth_addr) ? OFPPC_NO_RECV : OFPPC_NO_RECV_STP))
        {
          return;
        }
    }

  NS_LOG_INFO ("Matching against the flow table.");
  Simulator::Schedule (m_lookupDelay, &OpenFlowSwitchNetDevice::FlowTableLookup, this, key, buffer, packet_uid, port, send_to_controller);
}

int
OpenFlowSwitchNetDevice::RunThroughVPortTable (uint32_t packet_uid, int port, uint32_t vport)
{
  ofpbuf* buffer = m_packetData.find (packet_uid)->second.buffer;

  // extract the flow again since we need it
  // and the layer pointers may changed
  sw_flow_key key;
  key.wildcards = 0;
  if (flow_extract (buffer, port != -1 ? port : OFPP_NONE, &key.flow)
      && (m_flags & OFPC_FRAG_MASK) == OFPC_FRAG_DROP)
    {
      return 0;
    }

  // run through the chain of port table entries
  vport_table_entry *vpe = vport_table_lookup (&m_vportTable, vport);
  m_vportTable.lookup_count++;
  if (vpe)
    {
      m_vportTable.port_match_count++;
    }
  while (vpe != 0)
    {
      ofi::ExecuteVPortActions (this, packet_uid, m_packetData.find (packet_uid)->second.buffer, &key, vpe->port_acts->actions, vpe->port_acts->actions_len);
      vport_used (vpe, buffer); // update counters for virtual port
      if (vpe->parent_port_ptr == 0)
        {
          // if a port table's parent_port_ptr is 0 then
          // the parent_port should be a physical port
          if (vpe->parent_port <= OFPP_VP_START) // done traversing port chain, send packet to output port
            {
              OutputPort (packet_uid, port != -1 ? port : OFPP_NONE, vpe->parent_port, false);
            }
          else
            {
              NS_LOG_ERROR ("virtual port points to parent port\n");
            }
        }
      else // increment the number of port entries accessed by chaining
        {
          m_vportTable.chain_match_count++;
        }
      // move to the parent port entry
      vpe = vpe->parent_port_ptr;
    }

  return 0;
}

int
OpenFlowSwitchNetDevice::ReceiveFeaturesRequest (const void *msg)
{
  SendFeaturesReply ();
  return 0;
}

int
OpenFlowSwitchNetDevice::ReceiveVPortTableFeaturesRequest (const void *msg)
{
  SendVPortTableFeatures ();
  return 0;
}

int
OpenFlowSwitchNetDevice::ReceiveGetConfigRequest (const void *msg)
{
  ofpbuf *buffer;
  ofp_switch_config *osc = (ofp_switch_config*)MakeOpenflowReply (sizeof *osc, OFPT_GET_CONFIG_REPLY, &buffer);
  osc->flags = htons (m_flags);
  osc->miss_send_len = htons (m_missSendLen);

  return SendOpenflowBuffer (buffer);
}

int
OpenFlowSwitchNetDevice::ReceiveSetConfig (const void *msg)
{
  const ofp_switch_config *osc = (ofp_switch_config*)msg;

  int n_flags = ntohs (osc->flags) & (OFPC_SEND_FLOW_EXP | OFPC_FRAG_MASK);
  if ((n_flags & OFPC_FRAG_MASK) != OFPC_FRAG_NORMAL && (n_flags & OFPC_FRAG_MASK) != OFPC_FRAG_DROP)
    {
      n_flags = (n_flags & ~OFPC_FRAG_MASK) | OFPC_FRAG_DROP;
    }

  m_flags = n_flags;
  m_missSendLen = ntohs (osc->miss_send_len);
  return 0;
}

int
OpenFlowSwitchNetDevice::ReceivePacketOut (const void *msg)
{
  const ofp_packet_out *opo = (ofp_packet_out*)msg;
  ofpbuf *buffer;
  size_t actions_len = ntohs (opo->actions_len);

  if (actions_len > (ntohs (opo->header.length) - sizeof *opo))
    {
      NS_LOG_DEBUG ("message too short for number of actions");
      return -EINVAL;
    }

  if (ntohl (opo->buffer_id) == (uint32_t) -1)
    {
      // FIXME: can we avoid copying data here?
      int data_len = ntohs (opo->header.length) - sizeof *opo - actions_len;
      buffer = ofpbuf_new (data_len);
      ofpbuf_put (buffer, (uint8_t *)opo->actions + actions_len, data_len);
    }
  else
    {
      buffer = retrieve_buffer (ntohl (opo->buffer_id));
      if (buffer == 0)
        {
          return -ESRCH;
        }
    }

  sw_flow_key key;
  flow_extract (buffer, ntohs(opo->in_port), &key.flow); // ntohs(opo->in_port)

  uint16_t v_code = ofi::ValidateActions (&key, opo->actions, actions_len);
  if (v_code != ACT_VALIDATION_OK)
    {
      SendErrorMsg (OFPET_BAD_ACTION, v_code, msg, ntohs (opo->header.length));
      ofpbuf_delete (buffer);
      return -EINVAL;
    }

  ofi::ExecuteActions (this, opo->buffer_id, buffer, &key, opo->actions, actions_len, true);
  return 0;
}

int
OpenFlowSwitchNetDevice::ReceivePortMod (const void *msg)
{
  ofp_port_mod* opm = (ofp_port_mod*)msg;

  int port = opm->port_no; // ntohs(opm->port_no);
  if (port < DP_MAX_PORTS)
    {
      ofi::Port& p = m_ports[port];

      // Make sure the port id hasn't changed since this was sent
      Mac48Address hw_addr = Mac48Address ();
      hw_addr.CopyFrom (opm->hw_addr);
      if (p.netdev->GetAddress () != hw_addr)
        {
          return 0;
        }

      if (opm->mask)
        {
          uint32_t config_mask = ntohl (opm->mask);
          p.config &= ~config_mask;
          p.config |= ntohl (opm->config) & config_mask;
        }

      if (opm->mask & htonl (OFPPC_PORT_DOWN))
        {
          if ((opm->config & htonl (OFPPC_PORT_DOWN)) && (p.config & OFPPC_PORT_DOWN) == 0)
            {
              p.config |= OFPPC_PORT_DOWN;
              /// \todo Possibly disable the Port's Net Device via the appropriate interface.
            }
          else if ((opm->config & htonl (OFPPC_PORT_DOWN)) == 0 && (p.config & OFPPC_PORT_DOWN))
            {
              p.config &= ~OFPPC_PORT_DOWN;
              /// \todo Possibly enable the Port's Net Device via the appropriate interface.
            }
        }
    }

  return 0;
}

// add or remove a virtual port table entry
int
OpenFlowSwitchNetDevice::ReceiveVPortMod (const void *msg)
{
  const ofp_vport_mod *ovpm = (ofp_vport_mod*)msg;

  uint16_t command = ntohs (ovpm->command);
  if (command == OFPVP_ADD)
    {
      return AddVPort (ovpm);
    }
  else if (command == OFPVP_DELETE)
    {
      if (remove_vport_table_entry (&m_vportTable, ntohl (ovpm->vport)))
        {
          SendErrorMsg (OFPET_BAD_ACTION, OFPET_VPORT_MOD_FAILED, ovpm, ntohs (ovpm->header.length));
        }
    }

  return 0;
}

int
OpenFlowSwitchNetDevice::AddFlow (const ofp_flow_mod *ofm)
{
  size_t actions_len = ntohs (ofm->header.length) - sizeof *ofm;

  // Allocate memory.
  sw_flow *flow = flow_alloc (actions_len);
  if (flow == 0)
    {
      if (ntohl (ofm->buffer_id) != (uint32_t) -1)
        {
          discard_buffer (ntohl (ofm->buffer_id));
        }
      return -ENOMEM;
    }

  flow_extract_match (&flow->key, &ofm->match);

  uint16_t v_code = ofi::ValidateActions (&flow->key, ofm->actions, actions_len);
  if (v_code != ACT_VALIDATION_OK)
    {
      SendErrorMsg (OFPET_BAD_ACTION, v_code, ofm, ntohs (ofm->header.length));
      flow_free (flow);
      if (ntohl (ofm->buffer_id) != (uint32_t) -1)
        {
          discard_buffer (ntohl (ofm->buffer_id));
        }
      return -ENOMEM;
    }

  // Fill out flow.
  flow->priority = flow->key.wildcards ? ntohs (ofm->priority) : -1;
  flow->idle_timeout = ntohs (ofm->idle_timeout);
  flow->hard_timeout = ntohs (ofm->hard_timeout);
  flow->used = flow->created = time_now ();
  flow->sf_acts->actions_len = actions_len;
  flow->byte_count = 0;
  flow->packet_count = 0;
  memcpy (flow->sf_acts->actions, ofm->actions, actions_len);

  // Act.
  int error = chain_insert (m_chain, flow);
  if (error)
    {
      if (error == -ENOBUFS)
        {
          SendErrorMsg (OFPET_FLOW_MOD_FAILED, OFPFMFC_ALL_TABLES_FULL, ofm, ntohs (ofm->header.length));
        }
      flow_free (flow);
      if (ntohl (ofm->buffer_id) != (uint32_t) -1)
        {
          discard_buffer (ntohl (ofm->buffer_id));
        }
      return error;
    }

  NS_LOG_INFO ("Added new flow.");
  if (ntohl (ofm->buffer_id) != std::numeric_limits<uint32_t>::max ())
    {
      ofpbuf *buffer = retrieve_buffer (ofm->buffer_id); // ntohl(ofm->buffer_id)
      if (buffer)
        {
          sw_flow_key key;
          flow_used (flow, buffer);
          flow_extract (buffer, ntohs(ofm->match.in_port), &key.flow); // ntohs(ofm->match.in_port);
          ofi::ExecuteActions (this, ofm->buffer_id, buffer, &key, ofm->actions, actions_len, false);
          ofpbuf_delete (buffer);
        }
      else
        {
          return -ESRCH;
        }
    }
  return 0;
}

int
OpenFlowSwitchNetDevice::ModFlow (const ofp_flow_mod *ofm)
{
  sw_flow_key key;
  flow_extract_match (&key, &ofm->match);

  size_t actions_len = ntohs (ofm->header.length) - sizeof *ofm;

  uint16_t v_code = ofi::ValidateActions (&key, ofm->actions, actions_len);
  if (v_code != ACT_VALIDATION_OK)
    {
      SendErrorMsg ((ofp_error_type)OFPET_BAD_ACTION, v_code, ofm, ntohs (ofm->header.length));
      if (ntohl (ofm->buffer_id) != (uint32_t) -1)
        {
          discard_buffer (ntohl (ofm->buffer_id));
        }
      return -ENOMEM;
    }

  uint16_t priority = key.wildcards ? ntohs (ofm->priority) : -1;
  int strict = (ofm->command == htons (OFPFC_MODIFY_STRICT)) ? 1 : 0;
  chain_modify (m_chain, &key, priority, strict, ofm->actions, actions_len);

  if (ntohl (ofm->buffer_id) != std::numeric_limits<uint32_t>::max ())
    {
      ofpbuf *buffer = retrieve_buffer (ofm->buffer_id); // ntohl (ofm->buffer_id)
      if (buffer)
        {
          sw_flow_key skb_key;
          flow_extract (buffer, ntohs(ofm->match.in_port), &skb_key.flow); // ntohs(ofm->match.in_port);
          ofi::ExecuteActions (this, ofm->buffer_id, buffer, &skb_key, ofm->actions, actions_len, false);
          ofpbuf_delete (buffer);
        }
      else
        {
          return -ESRCH;
        }
    }
  return 0;
}

int
OpenFlowSwitchNetDevice::ReceiveFlow (const void *msg)
{
  NS_LOG_FUNCTION_NOARGS ();
  const ofp_flow_mod *ofm = (ofp_flow_mod*)msg;
  uint16_t command = ntohs (ofm->command);

  if (command == OFPFC_ADD)
    {
      return AddFlow (ofm);
    }
  else if ((command == OFPFC_MODIFY) || (command == OFPFC_MODIFY_STRICT))
    {
      return ModFlow (ofm);
    }
  else if (command == OFPFC_DELETE)
    {
      sw_flow_key key;
      flow_extract_match (&key, &ofm->match);
      return chain_delete (m_chain, &key, ofm->out_port, 0, 0) ? 0 : -ESRCH;
    }
  else if (command == OFPFC_DELETE_STRICT)
    {
      sw_flow_key key;
      uint16_t priority;
      flow_extract_match (&key, &ofm->match);
      priority = key.wildcards ? ntohs (ofm->priority) : -1;
      return chain_delete (m_chain, &key, ofm->out_port, priority, 1) ? 0 : -ESRCH;
    }
  else
    {
      return -ENODEV;
    }
}

int
OpenFlowSwitchNetDevice::StatsDump (ofi::StatsDumpCallback *cb)
{
  ofp_stats_reply *osr;
  ofpbuf *buffer;
  int err;

  if (cb->done)
    {
      return 0;
    }

  osr = (ofp_stats_reply*)MakeOpenflowReply (sizeof *osr, OFPT_STATS_REPLY, &buffer);
  osr->type = htons (cb->s->type);
  osr->flags = 0;

  err = cb->s->DoDump (this, cb->state, buffer);
  if (err >= 0)
    {
      if (err == 0)
        {
          cb->done = true;
        }
      else
        {
          // Buffer might have been reallocated, so find our data again.
          osr = (ofp_stats_reply*)ofpbuf_at_assert (buffer, 0, sizeof *osr);
          osr->flags = ntohs (OFPSF_REPLY_MORE);
        }

      int err2 = SendOpenflowBuffer (buffer);
      if (err2)
        {
          err = err2;
        }
    }

  return err;
}

void
OpenFlowSwitchNetDevice::StatsDone (ofi::StatsDumpCallback *cb)
{
  if (cb)
    {
      cb->s->DoCleanup (cb->state);
      free (cb->s);
      free (cb);
    }
}

int
OpenFlowSwitchNetDevice::ReceiveStatsRequest (const void *oh)
{
  const ofp_stats_request *rq = (ofp_stats_request*)oh;
  size_t rq_len = ntohs (rq->header.length);
  int type = ntohs (rq->type);
  int body_len = rq_len - offsetof (ofp_stats_request, body);
  ofi::Stats* st = new ofi::Stats ((ofp_stats_types)type, (unsigned)body_len);

  if (st == 0)
    {
      return -EINVAL;
    }

  ofi::StatsDumpCallback cb;
  cb.done = false;
  cb.rq = (ofp_stats_request*)xmemdup (rq, rq_len);
  cb.s = st;
  cb.state = 0;
  cb.swtch = this;

  if (cb.s)
    {
      int err = cb.s->DoInit (rq->body, body_len, &cb.state);
      if (err)
        {
          NS_LOG_WARN ("failed initialization of stats request type " << type << ": " << strerror (-err));
          free (cb.rq);
          return err;
        }
    }

  if (m_controller != 0)
    {
      m_controller->StartDump (&cb);
    }
  else
    {
      NS_LOG_ERROR ("Switch needs to be registered to a controller in order to start the stats reply.");
    }

  return 0;
}

int
OpenFlowSwitchNetDevice::ReceiveEchoRequest (const void *oh)
{
  return SendOpenflowBuffer (make_echo_reply ((ofp_header*)oh));
}

int
OpenFlowSwitchNetDevice::ReceiveEchoReply (const void *oh)
{
  return 0;
}

int
OpenFlowSwitchNetDevice::ForwardControlInput (const void *msg, size_t length)
{
  // Check encapsulated length.
  ofp_header *oh = (ofp_header*) msg;
  if (ntohs (oh->length) > length)
    {
      return -EINVAL;
    }
  assert (oh->version == OFP_VERSION);

  int error = 0;

  // Figure out how to handle it.
  switch (oh->type)
    {
    case OFPT_FEATURES_REQUEST:
      error = length < sizeof(ofp_header) ? -EFAULT : ReceiveFeaturesRequest (msg);
      break;
    case OFPT_GET_CONFIG_REQUEST:
      error = length < sizeof(ofp_header) ? -EFAULT : ReceiveGetConfigRequest (msg);
      break;
    case OFPT_SET_CONFIG:
      error = length < sizeof(ofp_switch_config) ? -EFAULT : ReceiveSetConfig (msg);
      break;
    case OFPT_PACKET_OUT:
      error = length < sizeof(ofp_packet_out) ? -EFAULT : ReceivePacketOut (msg);
      break;
    case OFPT_FLOW_MOD:
      error = length < sizeof(ofp_flow_mod) ? -EFAULT : ReceiveFlow (msg);
      break;
    case OFPT_PORT_MOD:
      error = length < sizeof(ofp_port_mod) ? -EFAULT : ReceivePortMod (msg);
      break;
    case OFPT_STATS_REQUEST:
      error = length < sizeof(ofp_stats_request) ? -EFAULT : ReceiveStatsRequest (msg);
      break;
    case OFPT_ECHO_REQUEST:
      error = length < sizeof(ofp_header) ? -EFAULT : ReceiveEchoRequest (msg);
      break;
    case OFPT_ECHO_REPLY:
      error = length < sizeof(ofp_header) ? -EFAULT : ReceiveEchoReply (msg);
      break;
    case OFPT_VPORT_MOD:
      error = length < sizeof(ofp_vport_mod) ? -EFAULT : ReceiveVPortMod (msg);
      break;
    case OFPT_VPORT_TABLE_FEATURES_REQUEST:
      error = length < sizeof(ofp_header) ? -EFAULT : ReceiveVPortTableFeaturesRequest (msg);
      break;
    default:
      SendErrorMsg ((ofp_error_type)OFPET_BAD_REQUEST, (ofp_bad_request_code)OFPBRC_BAD_TYPE, msg, length);
      error = -EINVAL;
    }

  if (msg != 0)
    {
      free ((ofpbuf*)msg);
    }
  return error;
}

sw_chain*
OpenFlowSwitchNetDevice::GetChain ()
{
  return m_chain;
}

uint32_t
OpenFlowSwitchNetDevice::GetNSwitchPorts (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_ports.size ();
}

ofi::Port
OpenFlowSwitchNetDevice::GetSwitchPort (uint32_t n) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_ports[n];
}

int
OpenFlowSwitchNetDevice::GetSwitchPortIndex (ofi::Port p)
{
  for (size_t i = 0; i < m_ports.size (); i++)
    {
      if (m_ports[i].netdev == p.netdev)
        {
          return i;
        }
    }
  return -1;
}

vport_table_t
OpenFlowSwitchNetDevice::GetVPortTable ()
{
  return m_vportTable;
}

} // namespace ns3

#endif // NS3_OPENFLOW
