/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Emmanuelle Laprise
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
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#include "ns3/log.h"
#include "ns3/queue.h"
#include "ns3/simulator.h"
#include "ns3/ethernet-header.h"
#include "ns3/ethernet-trailer.h"
#include "ns3/llc-snap-header.h"
#include "ns3/error-model.h"
#include "ns3/enum.h"
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ns3/pointer.h"
#include "ns3/trace-source-accessor.h"
#include "csma-net-device.h"
#include "csma-channel.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("CsmaNetDevice");

NS_OBJECT_ENSURE_REGISTERED (CsmaNetDevice);

TypeId
CsmaNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::CsmaNetDevice")
    .SetParent<NetDevice> ()
    .SetGroupName ("Csma")
    .AddConstructor<CsmaNetDevice> ()
    .AddAttribute ("Address", 
                   "The MAC address of this device.",
                   Mac48AddressValue (Mac48Address ("ff:ff:ff:ff:ff:ff")),
                   MakeMac48AddressAccessor (&CsmaNetDevice::m_address),
                   MakeMac48AddressChecker ())
    .AddAttribute ("Mtu", "The MAC-level Maximum Transmission Unit",
                   UintegerValue (DEFAULT_MTU),
                   MakeUintegerAccessor (&CsmaNetDevice::SetMtu,
                                         &CsmaNetDevice::GetMtu),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("EncapsulationMode", 
                   "The link-layer encapsulation type to use.",
                   EnumValue (DIX),
                   MakeEnumAccessor (&CsmaNetDevice::SetEncapsulationMode),
                   MakeEnumChecker (DIX, "Dix",
                                    LLC, "Llc"))
    .AddAttribute ("SendEnable", 
                   "Enable or disable the transmitter section of the device.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&CsmaNetDevice::m_sendEnable),
                   MakeBooleanChecker ())
    .AddAttribute ("ReceiveEnable",
                   "Enable or disable the receiver section of the device.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&CsmaNetDevice::m_receiveEnable),
                   MakeBooleanChecker ())
    .AddAttribute ("ReceiveErrorModel", 
                   "The receiver error model used to simulate packet loss",
                   PointerValue (),
                   MakePointerAccessor (&CsmaNetDevice::m_receiveErrorModel),
                   MakePointerChecker<ErrorModel> ())

    //
    // Transmit queueing discipline for the device which includes its own set
    // of trace hooks.
    //
    .AddAttribute ("TxQueue", 
                   "A queue to use as the transmit queue in the device.",
                   PointerValue (),
                   MakePointerAccessor (&CsmaNetDevice::m_queue),
                   MakePointerChecker<Queue<Packet> > ())

    //
    // Trace sources at the "top" of the net device, where packets transition
    // to/from higher layers.
    //
    .AddTraceSource ("MacTx", 
                     "Trace source indicating a packet has "
                     "arrived for transmission by this device",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_macTxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacTxDrop", 
                     "Trace source indicating a packet has been "
                     "dropped by the device before transmission",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_macTxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacPromiscRx", 
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a promiscuous trace,",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_macPromiscRxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacRx", 
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a non-promiscuous trace,",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_macRxTrace),
                     "ns3::Packet::TracedCallback")
#if 0
    // Not currently implemented in this device
    .AddTraceSource ("MacRxDrop", 
                     "Trace source indicating a packet was received, "
                     "but dropped before being forwarded up the stack",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_macRxDropTrace),
                     "ns3::Packet::TracedCallback")
#endif
    .AddTraceSource ("MacTxBackoff", 
                     "Trace source indicating a packet has been "
                     "delayed by the CSMA backoff process",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_macTxBackoffTrace),
                     "ns3::Packet::TracedCallback")
    //
    // Trace sources at the "bottom" of the net device, where packets transition
    // to/from the channel.
    //
    .AddTraceSource ("PhyTxBegin", 
                     "Trace source indicating a packet has "
                     "begun transmitting over the channel",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_phyTxBeginTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyTxEnd", 
                     "Trace source indicating a packet has been "
                     "completely transmitted over the channel",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_phyTxEndTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyTxDrop", 
                     "Trace source indicating a packet has been "
                     "dropped by the device during transmission",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_phyTxDropTrace),
                     "ns3::Packet::TracedCallback")
#if 0
    // Not currently implemented in this device
    .AddTraceSource ("PhyRxBegin", 
                     "Trace source indicating a packet has "
                     "begun being received by the device",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_phyRxBeginTrace),
                     "ns3::Packet::TracedCallback")
#endif
    .AddTraceSource ("PhyRxEnd", 
                     "Trace source indicating a packet has been "
                     "completely received by the device",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_phyRxEndTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PhyRxDrop", 
                     "Trace source indicating a packet has been "
                     "dropped by the device during reception",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_phyRxDropTrace),
                     "ns3::Packet::TracedCallback")
    //
    // Trace sources designed to simulate a packet sniffer facility (tcpdump). 
    //
    .AddTraceSource ("Sniffer", 
                     "Trace source simulating a non-promiscuous "
                     "packet sniffer attached to the device",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_snifferTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PromiscSniffer", 
                     "Trace source simulating a promiscuous "
                     "packet sniffer attached to the device",
                     MakeTraceSourceAccessor (&CsmaNetDevice::m_promiscSnifferTrace),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}

CsmaNetDevice::CsmaNetDevice ()
  : m_linkUp (false)
{
  NS_LOG_FUNCTION (this);
  m_txMachineState = READY;
  m_tInterframeGap = Seconds (0);
  m_channel = 0;

  // 
  // We would like to let the attribute system take care of initializing the 
  // packet encapsulation stuff, but we also don't want to get caught up in
  // initialization order changes.  So we'll get the three problem variables
  // into a consistent state here before the attribute calls, and then depend
  // on the semantics of the setters to preserve a consistent state.  This 
  // really doesn't have to be the same set of values as the initial values 
  // set by the attributes, but it does have to be a consistent set.  That is,
  // you can just change the default encapsulation mode above without having 
  // to change it here.
  //
  m_encapMode = DIX;
}

CsmaNetDevice::~CsmaNetDevice()
{
  NS_LOG_FUNCTION_NOARGS ();
  m_queue = 0;
}

void
CsmaNetDevice::DoDispose ()
{
  NS_LOG_FUNCTION_NOARGS ();
  m_channel = 0;
  m_node = 0;
  m_queue = 0;
  NetDevice::DoDispose ();
}

void
CsmaNetDevice::SetEncapsulationMode (enum EncapsulationMode mode)
{
  NS_LOG_FUNCTION (mode);

  m_encapMode = mode;

  NS_LOG_LOGIC ("m_encapMode = " << m_encapMode);
  NS_LOG_LOGIC ("m_mtu = " << m_mtu);
}

CsmaNetDevice::EncapsulationMode
CsmaNetDevice::GetEncapsulationMode (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_encapMode;
}

bool
CsmaNetDevice::SetMtu (uint16_t mtu)
{
  NS_LOG_FUNCTION (this << mtu);
  m_mtu = mtu;

  NS_LOG_LOGIC ("m_encapMode = " << m_encapMode);
  NS_LOG_LOGIC ("m_mtu = " << m_mtu);

  return true;
}

uint16_t
CsmaNetDevice::GetMtu (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_mtu;
}


void
CsmaNetDevice::SetSendEnable (bool sendEnable)
{
  NS_LOG_FUNCTION (sendEnable);
  m_sendEnable = sendEnable;
}

void
CsmaNetDevice::SetReceiveEnable (bool receiveEnable)
{
  NS_LOG_FUNCTION (receiveEnable);
  m_receiveEnable = receiveEnable;
}

bool
CsmaNetDevice::IsSendEnabled (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_sendEnable;
}

bool
CsmaNetDevice::IsReceiveEnabled (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_receiveEnable;
}

void
CsmaNetDevice::SetInterframeGap (Time t)
{
  NS_LOG_FUNCTION (t);
  m_tInterframeGap = t;
}

void
CsmaNetDevice::SetBackoffParams (Time slotTime, uint32_t minSlots, uint32_t maxSlots, uint32_t ceiling, uint32_t maxRetries)
{
  NS_LOG_FUNCTION (slotTime << minSlots << maxSlots << ceiling << maxRetries);
  m_backoff.m_slotTime = slotTime;
  m_backoff.m_minSlots = minSlots;
  m_backoff.m_maxSlots = maxSlots;
  m_backoff.m_ceiling = ceiling;
  m_backoff.m_maxRetries = maxRetries;
}

void
CsmaNetDevice::AddHeader (Ptr<Packet> p,   Mac48Address source,  Mac48Address dest,  uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (p << source << dest << protocolNumber);

  EthernetHeader header (false);
  header.SetSource (source);
  header.SetDestination (dest);

  EthernetTrailer trailer;

  NS_LOG_LOGIC ("p->GetSize () = " << p->GetSize ());
  NS_LOG_LOGIC ("m_encapMode = " << m_encapMode);
  NS_LOG_LOGIC ("m_mtu = " << m_mtu);

  uint16_t lengthType = 0;
  switch (m_encapMode) 
    {
    case DIX:
      NS_LOG_LOGIC ("Encapsulating packet as DIX (type interpretation)");
      //
      // This corresponds to the type interpretation of the lengthType field as
      // in the old Ethernet Blue Book.
      //
      lengthType = protocolNumber;

      //
      // All Ethernet frames must carry a minimum payload of 46 bytes.  We need
      // to pad out if we don't have enough bytes.  These must be real bytes 
      // since they will be written to pcap files and compared in regression 
      // trace files.
      //
      if (p->GetSize () < 46)
        {
          uint8_t buffer[46];
          memset (buffer, 0, 46);
          Ptr<Packet> padd = Create<Packet> (buffer, 46 - p->GetSize ());
          p->AddAtEnd (padd);
        }
      break;
    case LLC: 
      {
        NS_LOG_LOGIC ("Encapsulating packet as LLC (length interpretation)");

        LlcSnapHeader llc;
        llc.SetType (protocolNumber);
        p->AddHeader (llc);

        //
        // This corresponds to the length interpretation of the lengthType 
        // field but with an LLC/SNAP header added to the payload as in 
        // IEEE 802.2
        //
        lengthType = p->GetSize ();

        //
        // All Ethernet frames must carry a minimum payload of 46 bytes.  The 
        // LLC SNAP header counts as part of this payload.  We need to padd out
        // if we don't have enough bytes.  These must be real bytes since they 
        // will be written to pcap files and compared in regression trace files.
        //
        if (p->GetSize () < 46)
          {
            uint8_t buffer[46];
            memset (buffer, 0, 46);
            Ptr<Packet> padd = Create<Packet> (buffer, 46 - p->GetSize ());
            p->AddAtEnd (padd);
          }

        NS_ASSERT_MSG (p->GetSize () <= GetMtu (),
                       "CsmaNetDevice::AddHeader(): 802.3 Length/Type field with LLC/SNAP: "
                       "length interpretation must not exceed device frame size minus overhead");
      }
      break;
    case ILLEGAL:
    default:
      NS_FATAL_ERROR ("CsmaNetDevice::AddHeader(): Unknown packet encapsulation mode");
      break;
    }

  NS_LOG_LOGIC ("header.SetLengthType (" << lengthType << ")");
  header.SetLengthType (lengthType);
  p->AddHeader (header);

  if (Node::ChecksumEnabled ())
    {
      trailer.EnableFcs (true);
    }
  trailer.CalcFcs (p);
  p->AddTrailer (trailer);
}

#if 0
bool
CsmaNetDevice::ProcessHeader (Ptr<Packet> p, uint16_t & param)
{
  NS_LOG_FUNCTION (p << param);

  EthernetTrailer trailer;
  p->RemoveTrailer (trailer);

  EthernetHeader header (false);
  p->RemoveHeader (header);

  if ((header.GetDestination () != GetBroadcast ()) &&
      (header.GetDestination () != GetAddress ()))
    {
      return false;
    }

  switch (m_encapMode)
    {
    case DIX:
      param = header.GetLengthType ();
      break;
    case LLC: 
      {
        LlcSnapHeader llc;
        p->RemoveHeader (llc);
        param = llc.GetType ();
      } 
      break;
    case ILLEGAL:
    default:
      NS_FATAL_ERROR ("CsmaNetDevice::ProcessHeader(): Unknown packet encapsulation mode");
      break;
    }
  return true;
}
#endif

void
CsmaNetDevice::TransmitStart (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  //
  // This function is called to start the process of transmitting a packet.  We 
  // expect that the packet to transmit will be found in m_currentPkt.
  //
  NS_ASSERT_MSG (m_currentPkt != 0, "CsmaNetDevice::TransmitStart(): m_currentPkt not set");

  NS_LOG_LOGIC ("m_currentPkt = " << m_currentPkt);
  NS_LOG_LOGIC ("UID = " << m_currentPkt->GetUid ());

  //
  // Only transmit if the send side of net device is enabled
  //
  if (IsSendEnabled () == false)
    {
      m_phyTxDropTrace (m_currentPkt);
      m_currentPkt = 0;
      return;
    }

  //
  // Somebody has called here telling us to start transmitting a packet.  They 
  // can only do this if the state machine is in the READY or BACKOFF state.
  // Specifically, if we are ready to start transmitting, we cannot already
  // be transmitting (i.e., BUSY)
  //
  NS_ASSERT_MSG ((m_txMachineState == READY) || (m_txMachineState == BACKOFF), 
                 "Must be READY to transmit. Tx state is: " << m_txMachineState);

  //
  // Now we have to sense the state of the medium and either start transmitting
  // if it is idle, or backoff our transmission if someone else is on the wire.
  //
  if (m_channel->GetState () != IDLE)
    {
      //
      // The channel is busy -- backoff and rechedule TransmitStart() unless
      // we have exhausted all of our retries.
      //
      m_txMachineState = BACKOFF;

      if (m_backoff.MaxRetriesReached ())
        { 
          //
          // Too many retries, abort transmission of packet
          //
          TransmitAbort ();
        } 
      else 
        {
          m_macTxBackoffTrace (m_currentPkt);

          m_backoff.IncrNumRetries ();
          Time backoffTime = m_backoff.GetBackoffTime ();

          NS_LOG_LOGIC ("Channel busy, backing off for " << backoffTime.As (Time::S));

          Simulator::Schedule (backoffTime, &CsmaNetDevice::TransmitStart, this);
        }
    } 
  else 
    {
      //
      // The channel is free, transmit the packet
      //
      m_phyTxBeginTrace (m_currentPkt);
      if (m_channel->TransmitStart (m_currentPkt, m_deviceId) == false)
        {
          NS_LOG_WARN ("Channel TransmitStart returns an error");
          m_phyTxDropTrace (m_currentPkt);
          m_currentPkt = 0;
          m_txMachineState = READY;
        } 
      else 
        {
          //
          // Transmission succeeded, reset the backoff time parameters and
          // schedule a transmit complete event.
          //
          m_backoff.ResetBackoffTime ();
          m_txMachineState = BUSY;

          Time tEvent = m_bps.CalculateBytesTxTime (m_currentPkt->GetSize ());
          NS_LOG_LOGIC ("Schedule TransmitCompleteEvent in " << tEvent.As (Time::S));
          Simulator::Schedule (tEvent, &CsmaNetDevice::TransmitCompleteEvent, this);
        }
    }
}

void
CsmaNetDevice::TransmitAbort (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  //
  // When we started the process of transmitting the current packet, it was 
  // placed in m_currentPkt.  So we had better find one there.
  //
  NS_ASSERT_MSG (m_currentPkt != 0, "CsmaNetDevice::TransmitAbort(): m_currentPkt zero");
  NS_LOG_LOGIC ("m_currentPkt=" << m_currentPkt);
  NS_LOG_LOGIC ("Pkt UID is " << m_currentPkt->GetUid () << ")");

  m_phyTxDropTrace (m_currentPkt);
  m_currentPkt = 0;

  NS_ASSERT_MSG (m_txMachineState == BACKOFF, "Must be in BACKOFF state to abort.  Tx state is: " << m_txMachineState);

  // 
  // We're done with that one, so reset the backoff algorithm and ready the
  // transmit state machine.
  //
  m_backoff.ResetBackoffTime ();
  m_txMachineState = READY;

  //
  // If there is another packet on the input queue, we need to start trying to 
  // get that out.  If the queue is empty we just wait until someone puts one
  // in.
  //
  if (m_queue->IsEmpty ())
    {
      return;
    }
  else
    {
      Ptr<Packet> packet = m_queue->Dequeue ();
      NS_ASSERT_MSG (packet != 0, "CsmaNetDevice::TransmitAbort(): IsEmpty false but no Packet on queue?");
      m_currentPkt = packet;
      m_snifferTrace (m_currentPkt);
      m_promiscSnifferTrace (m_currentPkt);
      TransmitStart ();
    }
}

void
CsmaNetDevice::TransmitCompleteEvent (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  //
  // This function is called to finish the  process of transmitting a packet.
  // We need to tell the channel that we've stopped wiggling the wire and
  // schedule an event that will be executed when it's time to re-enable
  // the transmitter after the interframe gap.
  //
  NS_ASSERT_MSG (m_txMachineState == BUSY, "CsmaNetDevice::transmitCompleteEvent(): Must be BUSY if transmitting");
  NS_ASSERT (m_channel->GetState () == TRANSMITTING);
  m_txMachineState = GAP;

  //
  // When we started transmitting the current packet, it was placed in 
  // m_currentPkt.  So we had better find one there.
  //
  NS_ASSERT_MSG (m_currentPkt != 0, "CsmaNetDevice::TransmitCompleteEvent(): m_currentPkt zero");
  NS_LOG_LOGIC ("m_currentPkt=" << m_currentPkt);
  NS_LOG_LOGIC ("Pkt UID is " << m_currentPkt->GetUid () << ")");

  m_channel->TransmitEnd (); 
  m_phyTxEndTrace (m_currentPkt);
  m_currentPkt = 0;

  NS_LOG_LOGIC ("Schedule TransmitReadyEvent in " << m_tInterframeGap.As (Time::S));

  Simulator::Schedule (m_tInterframeGap, &CsmaNetDevice::TransmitReadyEvent, this);
}

void
CsmaNetDevice::TransmitReadyEvent (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  //
  // This function is called to enable the transmitter after the interframe
  // gap has passed.  If there are pending transmissions, we use this opportunity
  // to start the next transmit.
  //
  NS_ASSERT_MSG (m_txMachineState == GAP, "CsmaNetDevice::TransmitReadyEvent(): Must be in interframe gap");
  m_txMachineState = READY;

  //
  // We expect that the packet we had been transmitting was cleared when the 
  // TransmitCompleteEvent() was executed.
  //
  NS_ASSERT_MSG (m_currentPkt == 0, "CsmaNetDevice::TransmitReadyEvent(): m_currentPkt nonzero");

  //
  // Get the next packet from the queue for transmitting
  //
  if (m_queue->IsEmpty ())
    {
      return;
    }
  else
    {
      Ptr<Packet> packet = m_queue->Dequeue ();
      NS_ASSERT_MSG (packet != 0, "CsmaNetDevice::TransmitReadyEvent(): IsEmpty false but no Packet on queue?");
      m_currentPkt = packet;
      m_snifferTrace (m_currentPkt);
      m_promiscSnifferTrace (m_currentPkt);
      TransmitStart ();
    }
}

bool
CsmaNetDevice::Attach (Ptr<CsmaChannel> ch)
{
  NS_LOG_FUNCTION (this << &ch);

  m_channel = ch;

  m_deviceId = m_channel->Attach (this);

  //
  // The channel provides us with the transmitter data rate.
  //
  m_bps = m_channel->GetDataRate ();

  //
  // We use the Ethernet interframe gap of 96 bit times.
  //
  m_tInterframeGap = m_bps.CalculateBytesTxTime (96/8);

  //
  // This device is up whenever a channel is attached to it.
  //
  NotifyLinkUp ();
  return true;
}

void
CsmaNetDevice::SetQueue (Ptr<Queue<Packet> > q)
{
  NS_LOG_FUNCTION (q);
  m_queue = q;
}

void
CsmaNetDevice::SetReceiveErrorModel (Ptr<ErrorModel> em)
{
  NS_LOG_FUNCTION (em);
  m_receiveErrorModel = em; 
}

void
CsmaNetDevice::Receive (Ptr<Packet> packet, Ptr<CsmaNetDevice> senderDevice)
{
  NS_LOG_FUNCTION (packet << senderDevice);
  NS_LOG_LOGIC ("UID is " << packet->GetUid ());

  //
  // We never forward up packets that we sent.  Real devices don't do this since
  // their receivers are disabled during send, so we don't.
  // 
  if (senderDevice == this)
    {
      return;
    }

  //
  // Hit the trace hook.  This trace will fire on all packets received from the
  // channel except those originated by this device.
  //
  m_phyRxEndTrace (packet);

  // 
  // Only receive if the send side of net device is enabled
  //
  if (IsReceiveEnabled () == false)
    {
      m_phyRxDropTrace (packet);
      return;
    }

  if (m_receiveErrorModel && m_receiveErrorModel->IsCorrupt (packet) )
    {
      NS_LOG_LOGIC ("Dropping pkt due to error model ");
      m_phyRxDropTrace (packet);
      return;
    }

  //
  // Trace sinks will expect complete packets, not packets without some of the
  // headers.
  //
  Ptr<Packet> originalPacket = packet->Copy ();

  EthernetTrailer trailer;
  packet->RemoveTrailer (trailer);
  if (Node::ChecksumEnabled ())
    {
      trailer.EnableFcs (true);
    }

  bool crcGood = trailer.CheckFcs (packet);
  if (!crcGood)
    {
      NS_LOG_INFO ("CRC error on Packet " << packet);
      m_phyRxDropTrace (packet);
      return;
    }

  EthernetHeader header (false);
  packet->RemoveHeader (header);

  NS_LOG_LOGIC ("Pkt source is " << header.GetSource ());
  NS_LOG_LOGIC ("Pkt destination is " << header.GetDestination ());

  uint16_t protocol;
  //
  // If the length/type is less than 1500, it corresponds to a length 
  // interpretation packet.  In this case, it is an 802.3 packet and 
  // will also have an 802.2 LLC header.  If greater than 1500, we
  // find the protocol number (Ethernet type) directly.
  //
  if (header.GetLengthType () <= 1500)
    {
      NS_ASSERT (packet->GetSize () >= header.GetLengthType ());
      uint32_t padlen = packet->GetSize () - header.GetLengthType ();
      NS_ASSERT (padlen <= 46);
      if (padlen > 0)
        {
          packet->RemoveAtEnd (padlen);
        }

      LlcSnapHeader llc;
      packet->RemoveHeader (llc);
      protocol = llc.GetType ();
    }
  else
    {
      protocol = header.GetLengthType ();
    }

  //
  // Classify the packet based on its destination.
  //
  PacketType packetType;

  if (header.GetDestination ().IsBroadcast ())
    {
      packetType = PACKET_BROADCAST;
    }
  else if (header.GetDestination ().IsGroup ())
    {
      packetType = PACKET_MULTICAST;
    }
  else if (header.GetDestination () == m_address)
    {
      packetType = PACKET_HOST;
    }
  else
    {
      packetType = PACKET_OTHERHOST;
    }

  // 
  // For all kinds of packetType we receive, we hit the promiscuous sniffer
  // hook and pass a copy up to the promiscuous callback.  Pass a copy to 
  // make sure that nobody messes with our packet.
  //
  m_promiscSnifferTrace (originalPacket);
  if (!m_promiscRxCallback.IsNull ())
    {
      m_macPromiscRxTrace (originalPacket);
      m_promiscRxCallback (this, packet, protocol, header.GetSource (), header.GetDestination (), packetType);
    }

  //
  // If this packet is not destined for some other host, it must be for us
  // as either a broadcast, multicast or unicast.  We need to hit the mac
  // packet received trace hook and forward the packet up the stack.
  //
  if (packetType != PACKET_OTHERHOST)
    {
      m_snifferTrace (originalPacket);
      m_macRxTrace (originalPacket);
      m_rxCallback (this, packet, protocol, header.GetSource ());
    }
}

Ptr<Queue<Packet> >
CsmaNetDevice::GetQueue (void) const 
{ 
  NS_LOG_FUNCTION_NOARGS ();
  return m_queue;
}

void
CsmaNetDevice::NotifyLinkUp (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_linkUp = true;
  m_linkChangeCallbacks ();
}

void
CsmaNetDevice::SetIfIndex (const uint32_t index)
{
  NS_LOG_FUNCTION (index);
  m_ifIndex = index;
}

uint32_t
CsmaNetDevice::GetIfIndex (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_ifIndex;
}

Ptr<Channel>
CsmaNetDevice::GetChannel (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_channel;
}

void
CsmaNetDevice::SetAddress (Address address)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_address = Mac48Address::ConvertFrom (address);
}

Address
CsmaNetDevice::GetAddress (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_address;
}

bool
CsmaNetDevice::IsLinkUp (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_linkUp;
}

void
CsmaNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
  NS_LOG_FUNCTION (&callback);
  m_linkChangeCallbacks.ConnectWithoutContext (callback);
}

bool
CsmaNetDevice::IsBroadcast (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

Address
CsmaNetDevice::GetBroadcast (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return Mac48Address ("ff:ff:ff:ff:ff:ff");
}

bool
CsmaNetDevice::IsMulticast (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

Address
CsmaNetDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  NS_LOG_FUNCTION (multicastGroup);

  Mac48Address ad = Mac48Address::GetMulticast (multicastGroup);

  //
  // Implicit conversion (operator Address ()) is defined for Mac48Address, so
  // use it by just returning the EUI-48 address which is automagically converted
  // to an Address.
  //
  NS_LOG_LOGIC ("multicast address is " << ad);

  return ad;
}

bool
CsmaNetDevice::IsPointToPoint (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return false;
}

bool
CsmaNetDevice::IsBridge (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return false;
}

bool
CsmaNetDevice::Send (Ptr<Packet> packet,const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (packet << dest << protocolNumber);
  return SendFrom (packet, m_address, dest, protocolNumber);
}

bool
CsmaNetDevice::SendFrom (Ptr<Packet> packet, const Address& src, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (packet << src << dest << protocolNumber);
  NS_LOG_LOGIC ("packet =" << packet);
  NS_LOG_LOGIC ("UID is " << packet->GetUid () << ")");

  NS_ASSERT (IsLinkUp ());

  //
  // Only transmit if send side of net device is enabled
  //
  if (IsSendEnabled () == false)
    {
      m_macTxDropTrace (packet);
      return false;
    }

  Mac48Address destination = Mac48Address::ConvertFrom (dest);
  Mac48Address source = Mac48Address::ConvertFrom (src);
  AddHeader (packet, source, destination, protocolNumber);

  m_macTxTrace (packet);

  //
  // Place the packet to be sent on the send queue.  Note that the 
  // queue may fire a drop trace, but we will too.
  //
  if (m_queue->Enqueue (packet) == false)
    {
      m_macTxDropTrace (packet);
      return false;
    }

  //
  // If the device is idle, we need to start a transmission. Otherwise,
  // the transmission will be started when the current packet finished
  // transmission (see TransmitCompleteEvent)
  //
  if (m_txMachineState == READY) 
    {
      if (m_queue->IsEmpty () == false)
        {
          Ptr<Packet> packet = m_queue->Dequeue ();
          NS_ASSERT_MSG (packet != 0, "CsmaNetDevice::SendFrom(): IsEmpty false but no Packet on queue?");
          m_currentPkt = packet;
          m_promiscSnifferTrace (m_currentPkt);
          m_snifferTrace (m_currentPkt);
          TransmitStart ();
        }
    }
  return true;
}

Ptr<Node>
CsmaNetDevice::GetNode (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_node;
}

void
CsmaNetDevice::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (node);

  m_node = node;
}

bool
CsmaNetDevice::NeedsArp (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

void
CsmaNetDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  NS_LOG_FUNCTION (&cb);
  m_rxCallback = cb;
}

Address CsmaNetDevice::GetMulticast (Ipv6Address addr) const
{
  Mac48Address ad = Mac48Address::GetMulticast (addr);

  NS_LOG_LOGIC ("MAC IPv6 multicast address is " << ad);
  return ad;
}

void
CsmaNetDevice::SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb)
{
  NS_LOG_FUNCTION (&cb);
  m_promiscRxCallback = cb;
}

bool
CsmaNetDevice::SupportsSendFrom () const
{
  NS_LOG_FUNCTION_NOARGS ();
  return true;
}

int64_t
CsmaNetDevice::AssignStreams (int64_t stream)
{
  return m_backoff.AssignStreams (stream);
}

} // namespace ns3
