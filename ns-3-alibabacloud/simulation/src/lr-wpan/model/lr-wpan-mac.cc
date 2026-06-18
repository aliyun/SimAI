/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Authors:
 *  Gary Pei <guangyu.pei@boeing.com>
 *  kwong yin <kwong-sang.yin@boeing.com>
 *  Tom Henderson <thomas.r.henderson@boeing.com>
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 *  Erwan Livolant <erwan.livolant@inria.fr>
 *  Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */
#include "lr-wpan-mac.h"
#include "lr-wpan-csmaca.h"
#include "lr-wpan-mac-header.h"
#include "lr-wpan-mac-pl-headers.h"
#include "lr-wpan-mac-trailer.h"
#include <ns3/simulator.h>
#include <ns3/log.h>
#include <ns3/uinteger.h>
#include <ns3/node.h>
#include <ns3/packet.h>
#include <ns3/random-variable-stream.h>
#include <ns3/double.h>

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT                                   \
  std::clog << "[address " << m_shortAddress << "] ";

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LrWpanMac");

NS_OBJECT_ENSURE_REGISTERED (LrWpanMac);

//IEEE 802.15.4-2011 Table 51
const uint32_t LrWpanMac::aMinMPDUOverhead = 9;
const uint32_t LrWpanMac::aBaseSlotDuration = 60;
const uint32_t LrWpanMac::aNumSuperframeSlots = 16;
const uint32_t LrWpanMac::aBaseSuperframeDuration = aBaseSlotDuration * aNumSuperframeSlots;
const uint32_t LrWpanMac::aMaxLostBeacons = 4;
const uint32_t LrWpanMac::aMaxSIFSFrameSize = 18;

TypeId
LrWpanMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LrWpanMac")
    .SetParent<Object> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<LrWpanMac> ()
    .AddAttribute ("PanId", "16-bit identifier of the associated PAN",
                   UintegerValue (),
                   MakeUintegerAccessor (&LrWpanMac::m_macPanId),
                   MakeUintegerChecker<uint16_t> ())
    .AddTraceSource ("MacTxEnqueue",
                     "Trace source indicating a packet has been "
                     "enqueued in the transaction queue",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macTxEnqueueTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacTxDequeue",
                     "Trace source indicating a packet has was "
                     "dequeued from the transaction queue",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macTxDequeueTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacTx",
                     "Trace source indicating a packet has "
                     "arrived for transmission by this device",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macTxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacTxOk",
                     "Trace source indicating a packet has been "
                     "successfully sent",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macTxOkTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacTxDrop",
                     "Trace source indicating a packet has been "
                     "dropped during transmission",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macTxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacPromiscRx",
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a promiscuous trace,",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macPromiscRxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacRx",
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a non-promiscuous trace,",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macRxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacRxDrop",
                     "Trace source indicating a packet was received, "
                     "but dropped before being forwarded up the stack",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macRxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("Sniffer",
                     "Trace source simulating a non-promiscuous "
                     "packet sniffer attached to the device",
                     MakeTraceSourceAccessor (&LrWpanMac::m_snifferTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PromiscSniffer",
                     "Trace source simulating a promiscuous "
                     "packet sniffer attached to the device",
                     MakeTraceSourceAccessor (&LrWpanMac::m_promiscSnifferTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacStateValue",
                     "The state of LrWpan Mac",
                     MakeTraceSourceAccessor (&LrWpanMac::m_lrWpanMacState),
                     "ns3::TracedValueCallback::LrWpanMacState")
    .AddTraceSource ("MacIncSuperframeStatus",
                     "The period status of the incoming superframe",
                     MakeTraceSourceAccessor (&LrWpanMac::m_incSuperframeStatus),
                     "ns3::TracedValueCallback::SuperframeState")
    .AddTraceSource ("MacOutSuperframeStatus",
                     "The period status of the outgoing superframe",
                     MakeTraceSourceAccessor (&LrWpanMac::m_outSuperframeStatus),
                     "ns3::TracedValueCallback::SuperframeState")
    .AddTraceSource ("MacState",
                     "The state of LrWpan Mac",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macStateLogger),
                     "ns3::LrWpanMac::StateTracedCallback")
    .AddTraceSource ("MacSentPkt",
                     "Trace source reporting some information about "
                     "the sent packet",
                     MakeTraceSourceAccessor (&LrWpanMac::m_sentPktTrace),
                     "ns3::LrWpanMac::SentTracedCallback")
    .AddTraceSource ("IfsEnd",
                     "Trace source reporting the end of an "
                     "Interframe space (IFS) ",
                     MakeTraceSourceAccessor (&LrWpanMac::m_macIfsEndTrace),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}

LrWpanMac::LrWpanMac ()
{

  // First set the state to a known value, call ChangeMacState to fire trace source.
  m_lrWpanMacState = MAC_IDLE;

  ChangeMacState (MAC_IDLE);

  m_incSuperframeStatus = INACTIVE;
  m_outSuperframeStatus = INACTIVE;

  m_macRxOnWhenIdle = true;
  m_macPanId = 0xffff;
  m_deviceCapability = DeviceType::FFD;
  m_associationStatus = ASSOCIATED;
  m_selfExt = Mac64Address::Allocate ();
  m_macPromiscuousMode = false;
  m_macMaxFrameRetries = 3;
  m_retransmission = 0;
  m_numCsmacaRetry = 0;
  m_txPkt = 0;
  m_ifs = 0;

  m_macLIFSPeriod = 40;
  m_macSIFSPeriod = 12;

  m_macBeaconOrder = 15;
  m_macSuperframeOrder = 15;
  m_macTransactionPersistanceTime = 500; //0x01F5
  m_macAutoRequest = true;

  m_incomingBeaconOrder = 15;
  m_incomingSuperframeOrder = 15;
  m_beaconTrackingOn = false;
  m_numLostBeacons = 0;


  Ptr<UniformRandomVariable> uniformVar = CreateObject<UniformRandomVariable> ();
  uniformVar->SetAttribute ("Min", DoubleValue (0.0));
  uniformVar->SetAttribute ("Max", DoubleValue (255.0));
  m_macDsn = SequenceNumber8 (uniformVar->GetValue ());
  m_macBsn = SequenceNumber8 (uniformVar->GetValue ());
  m_shortAddress = Mac16Address ("00:00");
}

LrWpanMac::~LrWpanMac ()
{}

void
LrWpanMac::DoInitialize ()
{
  if (m_macRxOnWhenIdle)
    {
      m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_RX_ON);
    }
  else
    {
      m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_TRX_OFF);
    }

  Object::DoInitialize ();
}

void
LrWpanMac::DoDispose ()
{
  if (m_csmaCa != 0)
    {
      m_csmaCa->Dispose ();
      m_csmaCa = 0;
    }
  m_txPkt = 0;
  for (uint32_t i = 0; i < m_txQueue.size (); i++)
    {
      m_txQueue[i]->txQPkt = 0;
      delete m_txQueue[i];
    }
  m_txQueue.clear ();
  m_phy = 0;
  m_mcpsDataIndicationCallback = MakeNullCallback< void, McpsDataIndicationParams, Ptr<Packet> > ();
  m_mcpsDataConfirmCallback = MakeNullCallback< void, McpsDataConfirmParams > ();

  m_beaconEvent.Cancel ();

  Object::DoDispose ();
}

bool
LrWpanMac::GetRxOnWhenIdle ()
{
  return m_macRxOnWhenIdle;
}

void
LrWpanMac::SetRxOnWhenIdle (bool rxOnWhenIdle)
{
  NS_LOG_FUNCTION (this << rxOnWhenIdle);
  m_macRxOnWhenIdle = rxOnWhenIdle;

  if (m_lrWpanMacState == MAC_IDLE)
    {
      if (m_macRxOnWhenIdle)
        {
          m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_RX_ON);
        }
      else
        {
          m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_TRX_OFF);
        }
    }
}

void
LrWpanMac::SetShortAddress (Mac16Address address)
{
  //NS_LOG_FUNCTION (this << address);
  m_shortAddress = address;
}

void
LrWpanMac::SetExtendedAddress (Mac64Address address)
{
  //NS_LOG_FUNCTION (this << address);
  m_selfExt = address;
}


Mac16Address
LrWpanMac::GetShortAddress () const
{
  NS_LOG_FUNCTION (this);
  return m_shortAddress;
}

Mac64Address
LrWpanMac::GetExtendedAddress () const
{
  NS_LOG_FUNCTION (this);
  return m_selfExt;
}
void
LrWpanMac::McpsDataRequest (McpsDataRequestParams params, Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this << p);

  McpsDataConfirmParams confirmParams;
  confirmParams.m_msduHandle = params.m_msduHandle;

  // TODO: We need a drop trace for the case that the packet is too large or the request parameters are maleformed.
  //       The current tx drop trace is not suitable, because packets dropped using this trace carry the mac header
  //       and footer, while packets being dropped here do not have them.

  LrWpanMacHeader macHdr (LrWpanMacHeader::LRWPAN_MAC_DATA, m_macDsn.GetValue ());
  m_macDsn++;

  if (p->GetSize () > LrWpanPhy::aMaxPhyPacketSize - aMinMPDUOverhead)
    {
      // Note, this is just testing maximum theoretical frame size per the spec
      // The frame could still be too large once headers are put on
      // in which case the phy will reject it instead
      NS_LOG_ERROR (this << " packet too big: " << p->GetSize ());
      confirmParams.m_status = IEEE_802_15_4_FRAME_TOO_LONG;
      if (!m_mcpsDataConfirmCallback.IsNull ())
        {
          m_mcpsDataConfirmCallback (confirmParams);
        }
      return;
    }

  if ((params.m_srcAddrMode == NO_PANID_ADDR)
      && (params.m_dstAddrMode == NO_PANID_ADDR))
    {
      NS_LOG_ERROR (this << " Can not send packet with no Address field" );
      confirmParams.m_status = IEEE_802_15_4_INVALID_ADDRESS;
      if (!m_mcpsDataConfirmCallback.IsNull ())
        {
          m_mcpsDataConfirmCallback (confirmParams);
        }
      return;
    }
  switch (params.m_srcAddrMode)
    {
      case NO_PANID_ADDR:
        macHdr.SetSrcAddrMode (params.m_srcAddrMode);
        macHdr.SetNoPanIdComp ();
        break;
      case ADDR_MODE_RESERVED:
        NS_ABORT_MSG ("Can not set source address type to ADDR_MODE_RESERVED. Aborting.");
        break;
      case SHORT_ADDR:
        macHdr.SetSrcAddrMode (params.m_srcAddrMode);
        macHdr.SetSrcAddrFields (GetPanId (), GetShortAddress ());
        break;
      case EXT_ADDR:
        macHdr.SetSrcAddrMode (params.m_srcAddrMode);
        macHdr.SetSrcAddrFields (GetPanId (), GetExtendedAddress ());
        break;
      default:
        NS_LOG_ERROR (this << " Can not send packet with incorrect Source Address mode = " << params.m_srcAddrMode);
        confirmParams.m_status = IEEE_802_15_4_INVALID_ADDRESS;
        if (!m_mcpsDataConfirmCallback.IsNull ())
          {
            m_mcpsDataConfirmCallback (confirmParams);
          }
        return;
    }
  switch (params.m_dstAddrMode)
    {
      case NO_PANID_ADDR:
        macHdr.SetDstAddrMode (params.m_dstAddrMode);
        macHdr.SetNoPanIdComp ();
        break;
      case ADDR_MODE_RESERVED:
        NS_ABORT_MSG ("Can not set destination address type to ADDR_MODE_RESERVED. Aborting.");
        break;
      case SHORT_ADDR:
        macHdr.SetDstAddrMode (params.m_dstAddrMode);
        macHdr.SetDstAddrFields (params.m_dstPanId, params.m_dstAddr);
        break;
      case EXT_ADDR:
        macHdr.SetDstAddrMode (params.m_dstAddrMode);
        macHdr.SetDstAddrFields (params.m_dstPanId, params.m_dstExtAddr);
        break;
      default:
        NS_LOG_ERROR (this << " Can not send packet with incorrect Destination Address mode = " << params.m_dstAddrMode);
        confirmParams.m_status = IEEE_802_15_4_INVALID_ADDRESS;
        if (!m_mcpsDataConfirmCallback.IsNull ())
          {
            m_mcpsDataConfirmCallback (confirmParams);
          }
        return;
    }

  macHdr.SetSecDisable ();
  //extract the first 3 bits in TxOptions
  int b0 = params.m_txOptions & TX_OPTION_ACK;
  int b1 = params.m_txOptions & TX_OPTION_GTS;
  int b2 = params.m_txOptions & TX_OPTION_INDIRECT;

  if (b0 == TX_OPTION_ACK)
    {
      // Set AckReq bit only if the destination is not the broadcast address.
      if (macHdr.GetDstAddrMode () == SHORT_ADDR)
        {
          // short address and ACK requested.
          Mac16Address shortAddr = macHdr.GetShortDstAddr ();
          if (shortAddr.IsBroadcast () || shortAddr.IsMulticast ())
            {
              NS_LOG_LOGIC ("LrWpanMac::McpsDataRequest: requested an ACK on broadcast or multicast destination (" << shortAddr << ") - forcefully removing it.");
              macHdr.SetNoAckReq ();
              params.m_txOptions &= ~uint8_t (TX_OPTION_ACK);
            }
          else
            {
              macHdr.SetAckReq ();
            }
        }
      else
        {
          // other address (not short) and ACK requested
          macHdr.SetAckReq ();
        }
    }
  else
    {
      macHdr.SetNoAckReq ();
    }


  if (b1 == TX_OPTION_GTS)
    {
      //TODO:GTS Transmission
    }
  else if (b2 == TX_OPTION_INDIRECT)
    {
      // Indirect Tx
      // A COORDINATOR will save the packet in the pending queue and await for data
      // requests from its associated devices. The devices are aware of pending data,
      // from the pending bit information extracted from the received beacon.
      // A DEVICE must be tracking beacons (MLME-SYNC.request is running) before attempting
      // request data from the coordinator.


      //TODO: Check if the current device is coordinator or PAN coordinator
      p->AddHeader (macHdr);

      LrWpanMacTrailer macTrailer;
      // Calculate FCS if the global attribute ChecksumEnable is set.
      if (Node::ChecksumEnabled ())
        {
          macTrailer.EnableFcs (true);
          macTrailer.SetFcs (p);
        }
      p->AddTrailer (macTrailer);

      if (m_txQueue.size () == m_txQueue.max_size ())
        {
          confirmParams.m_status = IEEE_802_15_4_TRANSACTION_OVERFLOW;
          if (!m_mcpsDataConfirmCallback.IsNull ())
            {
              m_mcpsDataConfirmCallback (confirmParams);
            }
        }
      else
        {
          IndTxQueueElement *indTxQElement = new IndTxQueueElement;
          uint64_t unitPeriodSymbols;
          Time   expireTime;

          if (m_macBeaconOrder == 15)
            {
              unitPeriodSymbols = aBaseSuperframeDuration;
            }
          else
            {
              unitPeriodSymbols = ((uint64_t) 1 << m_macBeaconOrder) * aBaseSuperframeDuration;
            }

          //TODO:  check possible incorrect expire time here.

          expireTime = Simulator::Now () + m_macTransactionPersistanceTime
            * MicroSeconds (unitPeriodSymbols * 1000 * 1000 / m_phy->GetDataOrSymbolRate (false));
          indTxQElement->expireTime = expireTime;
          indTxQElement->txQMsduHandle = params.m_msduHandle;
          indTxQElement->txQPkt = p;

          m_indTxQueue.push_back (indTxQElement);

          std::cout << "Indirect Transmission Pushed | Elements in the queue: " << m_indTxQueue.size ()
                    << " " << "Element to expire in: " << expireTime.GetSeconds () << "secs\n";
        }
    }
  else
    {
      // Direct Tx
      // From this point the packet will be pushed to a Tx queue and immediately
      // use a slotted (beacon-enabled) or unslotted (nonbeacon-enabled) version of CSMA/CA
      // before sending the packet, depending on whether it has previously
      // received a valid beacon or not.

      p->AddHeader (macHdr);

      LrWpanMacTrailer macTrailer;
      // Calculate FCS if the global attribute ChecksumEnable is set.
      if (Node::ChecksumEnabled ())
        {
          macTrailer.EnableFcs (true);
          macTrailer.SetFcs (p);
        }
      p->AddTrailer (macTrailer);

      m_macTxEnqueueTrace (p);

      TxQueueElement *txQElement = new TxQueueElement;
      txQElement->txQMsduHandle = params.m_msduHandle;
      txQElement->txQPkt = p;
      m_txQueue.push_back (txQElement);
      CheckQueue ();
    }

}


void
LrWpanMac::MlmeStartRequest (MlmeStartRequestParams params)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_deviceCapability == DeviceType::FFD);

  MlmeStartConfirmParams confirmParams;


  if (GetShortAddress () == Mac16Address ("ff:ff"))
    {
      NS_LOG_ERROR (this << " Invalid MAC short address" );
      confirmParams.m_status = MLMESTART_NO_SHORT_ADDRESS;
      if (!m_mlmeStartConfirmCallback.IsNull ())
        {
          m_mlmeStartConfirmCallback (confirmParams);
        }
      return;
    }


  if ( (params.m_logCh > 26)
       || (params.m_bcnOrd > 15)
       || (params.m_sfrmOrd > params.m_bcnOrd))
    {
      NS_LOG_ERROR (this << " One or more parameters are invalid" );
      confirmParams.m_status = MLMESTART_INVALID_PARAMETER;
      if (!m_mlmeStartConfirmCallback.IsNull ())
        {
          m_mlmeStartConfirmCallback (confirmParams);
        }
      return;
    }


  if (params.m_coorRealgn)  //Coordinator Realignment
    {
      // TODO:: Send realignment request command frame in CSMA/CA
      return;
    }
  else
    {
      if (params.m_panCoor)
        {
          m_panCoor = true;
          m_macPanId = params.m_PanId;

          // Setting Channel and Page in the LrWpanPhy
          LrWpanPhyPibAttributes pibAttr;
          pibAttr.phyCurrentChannel = params.m_logCh;
          pibAttr.phyCurrentPage = params.m_logChPage;
          m_phy->PlmeSetAttributeRequest (LrWpanPibAttributeIdentifier::phyCurrentChannel,&pibAttr);
          m_phy->PlmeSetAttributeRequest (LrWpanPibAttributeIdentifier::phyCurrentPage,&pibAttr);
        }

      NS_ASSERT (params.m_PanId != 0xffff);


      m_macBeaconOrder = params.m_bcnOrd;
      if (m_macBeaconOrder == 15)
        {
          //Non-beacon enabled PAN
          m_macSuperframeOrder = 15;
          m_beaconInterval = 0;
          m_csmaCa->SetUnSlottedCsmaCa ();

          confirmParams.m_status = MLMESTART_SUCCESS;
          if (!m_mlmeStartConfirmCallback.IsNull ())
            {
              m_mlmeStartConfirmCallback (confirmParams);
            }
        }
      else
        {
          m_macSuperframeOrder = params.m_sfrmOrd;
          m_csmaCa->SetBatteryLifeExtension (params.m_battLifeExt);

          m_csmaCa->SetSlottedCsmaCa ();

          //TODO: Calculate the real Final CAP slot (requires GTS implementation)
          // FinalCapSlot = Superframe duration slots - CFP slots.
          // In the current implementation the value of the final cap slot is equal to
          // the total number of possible slots in the superframe (15).
          m_fnlCapSlot = 15;

          m_beaconInterval = ((uint32_t) 1 << m_macBeaconOrder) * aBaseSuperframeDuration;
          m_superframeDuration = ((uint32_t) 1 << m_macSuperframeOrder) * aBaseSuperframeDuration;

          //TODO: change the beacon sending according to the startTime parameter (if not PAN coordinator)

          m_beaconEvent = Simulator::ScheduleNow (&LrWpanMac::SendOneBeacon, this);
        }
    }
}


void
LrWpanMac::MlmeSyncRequest (MlmeSyncRequestParams params)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (params.m_logCh <= 26 && m_macPanId != 0xffff);

  uint64_t symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false); //symbols per second
  //change phy current logical channel
  LrWpanPhyPibAttributes pibAttr;
  pibAttr.phyCurrentChannel = params.m_logCh;
  m_phy->PlmeSetAttributeRequest (LrWpanPibAttributeIdentifier::phyCurrentChannel,&pibAttr);

  //Enable Phy receiver
  m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_RX_ON);

  uint64_t searchSymbols;
  Time searchBeaconTime;

  if (m_trackingEvent.IsRunning ())
    {
      m_trackingEvent.Cancel ();
    }

  if (params.m_trackBcn)
    {
      m_numLostBeacons = 0;
      //search for a beacon for a time = incomingSuperframe symbols + 960 symbols
      searchSymbols = ((uint64_t) 1 << m_incomingBeaconOrder) + 1 * aBaseSuperframeDuration;
      searchBeaconTime = Seconds ((double) searchSymbols / symbolRate);
      m_beaconTrackingOn = true;
      m_trackingEvent = Simulator::Schedule (searchBeaconTime, &LrWpanMac::BeaconSearchTimeout, this);
    }
  else
    {
      m_beaconTrackingOn = false;
    }

}


void
LrWpanMac::MlmePollRequest (MlmePollRequestParams params)
{
  NS_LOG_FUNCTION (this);

  LrWpanMacHeader macHdr (LrWpanMacHeader::LRWPAN_MAC_COMMAND, m_macBsn.GetValue ());
  m_macBsn++;

  CommandPayloadHeader macPayload (CommandPayloadHeader::DATA_REQ);

  Ptr<Packet> beaconPacket = Create <Packet> ();
}


void
LrWpanMac::SendOneBeacon ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_lrWpanMacState == MAC_IDLE);

  LrWpanMacHeader macHdr (LrWpanMacHeader::LRWPAN_MAC_BEACON, m_macBsn.GetValue ());
  m_macBsn++;
  BeaconPayloadHeader macPayload;
  Ptr<Packet> beaconPacket = Create <Packet> ();
  LrWpanMacTrailer macTrailer;

  macHdr.SetDstAddrMode (LrWpanMacHeader::SHORTADDR);
  macHdr.SetDstAddrFields (GetPanId (),Mac16Address ("ff:ff"));

  //see IEEE 802.15.4-2011 Section 5.1.2.4
  if (GetShortAddress () == Mac16Address ("ff:fe"))
    {
      macHdr.SetSrcAddrMode (LrWpanMacHeader::EXTADDR);
      macHdr.SetSrcAddrFields (GetPanId (),GetExtendedAddress ());
    }
  else
    {
      macHdr.SetSrcAddrMode (LrWpanMacHeader::SHORTADDR);
      macHdr.SetSrcAddrFields (GetPanId (), GetShortAddress ());
    }

  macHdr.SetSecDisable ();
  macHdr.SetNoAckReq ();

  macPayload.SetSuperframeSpecField (GetSuperframeField ());
  macPayload.SetGtsFields (GetGtsFields ());
  macPayload.SetPndAddrFields (GetPendingAddrFields ());

  beaconPacket->AddHeader (macPayload);
  beaconPacket->AddHeader (macHdr);

  // Calculate FCS if the global attribute ChecksumEnable is set.
  if (Node::ChecksumEnabled ())
    {
      macTrailer.EnableFcs (true);
      macTrailer.SetFcs (beaconPacket);
    }

  beaconPacket->AddTrailer (macTrailer);

  //Set the Beacon packet to be transmitted
  m_txPkt = beaconPacket;

  m_outSuperframeStatus = BEACON;

  NS_LOG_DEBUG ("Outgoing superframe Active Portion (Beacon + CAP + CFP): " << m_superframeDuration << " symbols");

  ChangeMacState (MAC_SENDING);
  m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_TX_ON);
}


void
LrWpanMac::StartCAP (SuperframeType superframeType)
{
  uint32_t activeSlot;
  uint64_t capDuration;
  Time endCapTime;
  uint64_t symbolRate;

  symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false); //symbols per second

  if (superframeType == OUTGOING)
    {
      m_outSuperframeStatus = CAP;
      activeSlot = m_superframeDuration / 16;
      capDuration = activeSlot * (m_fnlCapSlot + 1);
      endCapTime = Seconds ((double) capDuration  / symbolRate);
      // Obtain the end of the CAP by adjust the time it took to send the beacon
      endCapTime -= (Simulator::Now () - m_macBeaconTxTime);

      NS_LOG_DEBUG ("Outgoing superframe CAP duration " << (endCapTime.GetSeconds () * symbolRate) << " symbols (" << endCapTime.As (Time::S) << ")");
      NS_LOG_DEBUG ("Active Slots duration " << activeSlot << " symbols");

      m_capEvent =  Simulator::Schedule (endCapTime,
                                         &LrWpanMac::StartCFP, this, SuperframeType::OUTGOING);

    }
  else
    {
      m_incSuperframeStatus = CAP;
      activeSlot = m_incomingSuperframeDuration / 16;
      capDuration = activeSlot * (m_incomingFnlCapSlot + 1);
      endCapTime = Seconds ((double) capDuration / symbolRate);
      // Obtain the end of the CAP by adjust the time it took to receive the beacon
      endCapTime -= (Simulator::Now () - m_macBeaconRxTime);

      NS_LOG_DEBUG ("Incoming superframe CAP duration " << (endCapTime.GetSeconds () * symbolRate) << " symbols (" << endCapTime.As (Time::S) << ")");
      NS_LOG_DEBUG ("Active Slots duration " << activeSlot << " symbols");

      m_capEvent =  Simulator::Schedule (endCapTime,
                                         &LrWpanMac::StartCFP, this, SuperframeType::INCOMING);
    }

  CheckQueue ();

}


void
LrWpanMac::StartCFP (SuperframeType superframeType)
{
  uint32_t activeSlot;
  uint64_t cfpDuration;
  Time endCfpTime;
  uint64_t symbolRate;

  symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false); //symbols per second

  if (superframeType == INCOMING)
    {
      activeSlot = m_incomingSuperframeDuration / 16;
      cfpDuration = activeSlot * (15 - m_incomingFnlCapSlot);
      endCfpTime = Seconds ((double) cfpDuration / symbolRate);
      if (cfpDuration > 0)
        {
          m_incSuperframeStatus = CFP;
        }

      NS_LOG_DEBUG ("Incoming superframe CFP duration " << cfpDuration << " symbols (" << endCfpTime.As (Time::S) << ")");

      m_incCfpEvent =  Simulator::Schedule (endCfpTime,
                                            &LrWpanMac::StartInactivePeriod, this, SuperframeType::INCOMING);
    }
  else
    {
      activeSlot = m_superframeDuration / 16;
      cfpDuration = activeSlot * (15 - m_fnlCapSlot);
      endCfpTime = Seconds ((double) cfpDuration / symbolRate);

      if (cfpDuration > 0)
        {
          m_outSuperframeStatus = CFP;
        }

      NS_LOG_DEBUG ("Outgoing superframe CFP duration " << cfpDuration << " symbols (" << endCfpTime.As (Time::S) << ")");

      m_cfpEvent =  Simulator::Schedule (endCfpTime,
                                         &LrWpanMac::StartInactivePeriod, this, SuperframeType::OUTGOING);

    }
  //TODO: Start transmit or receive  GTS here.
}


void
LrWpanMac::StartInactivePeriod (SuperframeType superframeType)
{
  uint64_t inactiveDuration;
  Time endInactiveTime;
  uint64_t symbolRate;

  symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false); //symbols per second

  if (superframeType == INCOMING)
    {
      inactiveDuration = m_incomingBeaconInterval - m_incomingSuperframeDuration;
      endInactiveTime = Seconds ((double) inactiveDuration / symbolRate);

      if (inactiveDuration > 0)
        {
          m_incSuperframeStatus = INACTIVE;
        }

      NS_LOG_DEBUG ("Incoming superframe Inactive Portion duration " << inactiveDuration << " symbols (" << endInactiveTime.As (Time::S) << ")");
      m_beaconEvent = Simulator::Schedule (endInactiveTime, &LrWpanMac::AwaitBeacon, this);
    }
  else
    {
      inactiveDuration = m_beaconInterval - m_superframeDuration;
      endInactiveTime = Seconds ((double) inactiveDuration / symbolRate);

      if (inactiveDuration > 0)
        {
          m_outSuperframeStatus = INACTIVE;
        }

      NS_LOG_DEBUG ("Outgoing superframe Inactive Portion duration " << inactiveDuration << " symbols (" << endInactiveTime.As (Time::S) << ")");
      m_beaconEvent =  Simulator::Schedule (endInactiveTime, &LrWpanMac::SendOneBeacon, this);
    }
}

void
LrWpanMac::AwaitBeacon (void)
{
  m_incSuperframeStatus = BEACON;

  //TODO: If the device waits more than the expected time to receive the beacon (wait = 46 symbols for default beacon size)
  //      it should continue with the start of the incoming CAP even if it did not receive the beacon.
  //      At the moment, the start of the incoming CAP is only triggered if the beacon is received.
  //      See MLME-SyncLoss for details.


}

void
LrWpanMac::BeaconSearchTimeout (void)
{
  uint64_t symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false); //symbols per second

  if (m_numLostBeacons > aMaxLostBeacons)
    {
      MlmeSyncLossIndicationParams syncLossParams;
      //syncLossParams.m_logCh =
      syncLossParams.m_lossReason = MLMESYNCLOSS_BEACON_LOST;
      syncLossParams.m_panId = m_macPanId;
      m_mlmeSyncLossIndicationCallback (syncLossParams);

      m_beaconTrackingOn = false;
      m_numLostBeacons = 0;
    }
  else
    {
      m_numLostBeacons++;

      //Search for one more beacon
      uint64_t searchSymbols;
      Time searchBeaconTime;
      searchSymbols = ((uint64_t) 1 << m_incomingBeaconOrder) + 1 * aBaseSuperframeDuration;
      searchBeaconTime = Seconds ((double) searchSymbols / symbolRate);
      m_trackingEvent = Simulator::Schedule (searchBeaconTime, &LrWpanMac::BeaconSearchTimeout, this);

    }
}


void
LrWpanMac::CheckQueue ()
{
  NS_LOG_FUNCTION (this);
  // Pull a packet from the queue and start sending if we are not already sending.
  if (m_lrWpanMacState == MAC_IDLE && !m_txQueue.empty () && !m_setMacState.IsRunning ())
    {
      //TODO: this should check if the node is a coordinator and using the outcoming superframe not just the PAN coordinator
      if (m_csmaCa->IsUnSlottedCsmaCa () || (m_outSuperframeStatus == CAP && m_panCoor) || m_incSuperframeStatus == CAP)
        {
          // check MAC is not in a IFS
          if (!m_ifsEvent.IsRunning ())
            {
              TxQueueElement *txQElement = m_txQueue.front ();
              m_txPkt = txQElement->txQPkt;
              m_setMacState = Simulator::ScheduleNow (&LrWpanMac::SetLrWpanMacState, this, MAC_CSMA);
            }
        }
    }
}


SuperframeField
LrWpanMac::GetSuperframeField ()
{
  SuperframeField sfrmSpec;

  sfrmSpec.SetBeaconOrder (m_macBeaconOrder);
  sfrmSpec.SetSuperframeOrder (m_macSuperframeOrder);
  sfrmSpec.SetFinalCapSlot (m_fnlCapSlot);

  if (m_csmaCa->GetBatteryLifeExtension ())
    {
      sfrmSpec.SetBattLifeExt (true);
    }

  if (m_panCoor)
    {
      sfrmSpec.SetPanCoor (true);
    }
  //TODO: It is possible to do association using Beacons,
  //      however, the current implementation only support manual association.
  //      The association permit should be set here.

  return sfrmSpec;
}

GtsFields
LrWpanMac::GetGtsFields ()
{
  GtsFields gtsFields;

  // TODO: Logic to populate the GTS Fields from local information here

  return gtsFields;
}

PendingAddrFields
LrWpanMac::GetPendingAddrFields ()
{
  PendingAddrFields pndAddrFields;

  // TODO: Logic to populate the Pending Address Fields from local information here
  return pndAddrFields;
}


void
LrWpanMac::SetCsmaCa (Ptr<LrWpanCsmaCa> csmaCa)
{
  m_csmaCa = csmaCa;
}

void
LrWpanMac::SetPhy (Ptr<LrWpanPhy> phy)
{
  m_phy = phy;
}

Ptr<LrWpanPhy>
LrWpanMac::GetPhy (void)
{
  return m_phy;
}

void
LrWpanMac::SetMcpsDataIndicationCallback (McpsDataIndicationCallback c)
{
  m_mcpsDataIndicationCallback = c;
}

void
LrWpanMac::SetMcpsDataConfirmCallback (McpsDataConfirmCallback c)
{
  m_mcpsDataConfirmCallback = c;
}

void
LrWpanMac::SetMlmeStartConfirmCallback (MlmeStartConfirmCallback c)
{
  m_mlmeStartConfirmCallback = c;
}

void
LrWpanMac::SetMlmeBeaconNotifyIndicationCallback (MlmeBeaconNotifyIndicationCallback c)
{
  m_mlmeBeaconNotifyIndicationCallback = c;
}

void
LrWpanMac::SetMlmeSyncLossIndicationCallback (MlmeSyncLossIndicationCallback c)
{
  m_mlmeSyncLossIndicationCallback = c;
}

void
LrWpanMac::SetMlmePollConfirmCallback (MlmePollConfirmCallback c)
{
  m_mlmePollConfirmCallback = c;
}

void
LrWpanMac::PdDataIndication (uint32_t psduLength, Ptr<Packet> p, uint8_t lqi)
{
  NS_ASSERT (m_lrWpanMacState == MAC_IDLE || m_lrWpanMacState == MAC_ACK_PENDING || m_lrWpanMacState == MAC_CSMA);
  NS_LOG_FUNCTION (this << psduLength << p << (uint16_t)lqi);

  bool acceptFrame;

  // from sec 7.5.6.2 Reception and rejection, Std802.15.4-2006
  // level 1 filtering, test FCS field and reject if frame fails
  // level 2 filtering if promiscuous mode pass frame to higher layer otherwise perform level 3 filtering
  // level 3 filtering accept frame
  // if Frame type and version is not reserved, and
  // if there is a dstPanId then dstPanId=m_macPanId or broadcastPanI, and
  // if there is a shortDstAddr then shortDstAddr =shortMacAddr or broadcastAddr, and
  // if beacon frame then srcPanId = m_macPanId
  // if only srcAddr field in Data or Command frame,accept frame if srcPanId=m_macPanId

  Ptr<Packet> originalPkt = p->Copy (); // because we will strip headers
  uint64_t symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false); //symbols per second
  m_promiscSnifferTrace (originalPkt);

  m_macPromiscRxTrace (originalPkt);
  // XXX no rejection tracing (to macRxDropTrace) being performed below

  LrWpanMacTrailer receivedMacTrailer;
  p->RemoveTrailer (receivedMacTrailer);
  if (Node::ChecksumEnabled ())
    {
      receivedMacTrailer.EnableFcs (true);
    }

  // level 1 filtering
  if (!receivedMacTrailer.CheckFcs (p))
    {
      m_macRxDropTrace (originalPkt);
    }
  else
    {
      LrWpanMacHeader receivedMacHdr;
      p->RemoveHeader (receivedMacHdr);

      McpsDataIndicationParams params;
      params.m_dsn = receivedMacHdr.GetSeqNum ();
      params.m_mpduLinkQuality = lqi;
      params.m_srcPanId = receivedMacHdr.GetSrcPanId ();
      params.m_srcAddrMode = receivedMacHdr.GetSrcAddrMode ();
      switch (params.m_srcAddrMode)
        {
          case SHORT_ADDR:
            params.m_srcAddr = receivedMacHdr.GetShortSrcAddr ();
            break;
          case EXT_ADDR:
            params.m_srcExtAddr = receivedMacHdr.GetExtSrcAddr ();
            break;
          default:
            break;
        }
      params.m_dstPanId = receivedMacHdr.GetDstPanId ();
      params.m_dstAddrMode = receivedMacHdr.GetDstAddrMode ();
      switch (params.m_dstAddrMode)
        {
          case SHORT_ADDR:
            params.m_dstAddr = receivedMacHdr.GetShortDstAddr ();
            break;
          case EXT_ADDR:
            params.m_dstExtAddr = receivedMacHdr.GetExtDstAddr ();
            break;
          default:
            break;
        }

      if (m_macPromiscuousMode)
        {
          //level 2 filtering
          if (receivedMacHdr.GetDstAddrMode () == SHORT_ADDR)
            {
              NS_LOG_DEBUG ("Packet from " << params.m_srcAddr);
              NS_LOG_DEBUG ("Packet to " << params.m_dstAddr);
            }
          else if (receivedMacHdr.GetDstAddrMode () == EXT_ADDR)
            {
              NS_LOG_DEBUG ("Packet from " << params.m_srcExtAddr);
              NS_LOG_DEBUG ("Packet to " << params.m_dstExtAddr);
            }

          //TODO: Fix here, this should trigger different Indication Callbacks
          //depending the type of frame received (data,command, beacon)
          if (!m_mcpsDataIndicationCallback.IsNull ())
            {
              NS_LOG_DEBUG ("promiscuous mode, forwarding up");
              m_mcpsDataIndicationCallback (params, p);
            }
          else
            {
              NS_LOG_ERROR (this << " Data Indication Callback not initialized");
            }
        }
      else
        {
          //level 3 frame filtering
          acceptFrame = (receivedMacHdr.GetType () != LrWpanMacHeader::LRWPAN_MAC_RESERVED);

          if (acceptFrame)
            {
              acceptFrame = (receivedMacHdr.GetFrameVer () <= 1);
            }

          if (acceptFrame
              && (receivedMacHdr.GetDstAddrMode () > 1))
            {
              acceptFrame = receivedMacHdr.GetDstPanId () == m_macPanId
                || receivedMacHdr.GetDstPanId () == 0xffff;
            }

          if (acceptFrame
              && (receivedMacHdr.GetDstAddrMode () == SHORT_ADDR))
            {
              if (receivedMacHdr.GetShortDstAddr () == m_shortAddress)
                {
                  // unicast, for me
                  acceptFrame = true;
                }
              else if (receivedMacHdr.GetShortDstAddr ().IsBroadcast () || receivedMacHdr.GetShortDstAddr ().IsMulticast ())
                {
                  // broadcast or multicast
                  if (receivedMacHdr.IsAckReq ())
                    {
                      // discard broadcast/multicast with the ACK bit set
                      acceptFrame = false;
                    }
                  else
                    {
                      acceptFrame = true;
                    }
                }
              else
                {
                  acceptFrame = false;
                }
            }

          if (acceptFrame
              && (receivedMacHdr.GetDstAddrMode () == EXT_ADDR))
            {
              acceptFrame = (receivedMacHdr.GetExtDstAddr () == m_selfExt);
            }

          if (acceptFrame
              && (receivedMacHdr.GetType () == LrWpanMacHeader::LRWPAN_MAC_BEACON))
            {
              if (m_macPanId == 0xffff)
                {
                  // TODO: Accept only if the frame version field is valid
                  acceptFrame = true;
                }
              else
                {
                  acceptFrame = receivedMacHdr.GetSrcPanId () == m_macPanId;
                }
            }

          if (acceptFrame
              && ((receivedMacHdr.GetType () == LrWpanMacHeader::LRWPAN_MAC_DATA)
                  || (receivedMacHdr.GetType () == LrWpanMacHeader::LRWPAN_MAC_COMMAND))
              && (receivedMacHdr.GetSrcAddrMode () > 1))
            {
              acceptFrame = receivedMacHdr.GetSrcPanId () == m_macPanId; // TODO: need to check if PAN coord
            }

          if (acceptFrame)
            {
              m_macRxTrace (originalPkt);
              // \todo: What should we do if we receive a frame while waiting for an ACK?
              //        Especially if this frame has the ACK request bit set, should we reply with an ACK, possibly missing the pending ACK?

              // If the received frame is a frame with the ACK request bit set, we immediately send back an ACK.
              // If we are currently waiting for a pending ACK, we assume the ACK was lost and trigger a retransmission after sending the ACK.
              if ((receivedMacHdr.IsData () || receivedMacHdr.IsCommand ()) && receivedMacHdr.IsAckReq ()
                  && !(receivedMacHdr.GetDstAddrMode () == SHORT_ADDR && (receivedMacHdr.GetShortDstAddr ().IsBroadcast () || receivedMacHdr.GetShortDstAddr ().IsMulticast ())))
                {
                  // If this is a data or mac command frame, which is not a broadcast or multicast,
                  // with ack req set, generate and send an ack frame.
                  // If there is a CSMA medium access in progress we cancel the medium access
                  // for sending the ACK frame. A new transmission attempt will be started
                  // after the ACK was send.
                  if (m_lrWpanMacState == MAC_ACK_PENDING)
                    {
                      m_ackWaitTimeout.Cancel ();
                      PrepareRetransmission ();
                    }
                  else if (m_lrWpanMacState == MAC_CSMA)
                    {
                      // \todo: If we receive a packet while doing CSMA/CA, should  we drop the packet because of channel busy,
                      //        or should we restart CSMA/CA for the packet after sending the ACK?
                      // Currently we simply restart CSMA/CA after sending the ACK.
                      m_csmaCa->Cancel ();
                    }
                  // Cancel any pending MAC state change, ACKs have higher priority.
                  m_setMacState.Cancel ();
                  ChangeMacState (MAC_IDLE);

                  m_setMacState = Simulator::ScheduleNow (&LrWpanMac::SendAck, this, receivedMacHdr.GetSeqNum ());
                }


              if (receivedMacHdr.GetDstAddrMode () == SHORT_ADDR)
                {
                  NS_LOG_DEBUG ("Packet from " << params.m_srcAddr);
                  NS_LOG_DEBUG ("Packet to " << params.m_dstAddr);
                }
              else if (receivedMacHdr.GetDstAddrMode () == EXT_ADDR)
                {
                  NS_LOG_DEBUG ("Packet from " << params.m_srcExtAddr);
                  NS_LOG_DEBUG ("Packet to " << params.m_dstExtAddr);
                }


              if (receivedMacHdr.IsBeacon ())
                {


                  // The received beacon size in symbols
                  // Beacon = 5 bytes Sync Header (SHR) +  1 byte PHY header (PHR) + PSDU (default 17 bytes)
                  m_rxBeaconSymbols = m_phy->GetPhySHRDuration () + 1 * m_phy->GetPhySymbolsPerOctet () +
                    (originalPkt->GetSize () * m_phy->GetPhySymbolsPerOctet ());

                  // The start of Rx beacon time and start of the Incoming superframe Active Period
                  m_macBeaconRxTime = Simulator::Now () - Seconds (double(m_rxBeaconSymbols) / symbolRate);

                  NS_LOG_DEBUG ("Beacon Received; forwarding up (m_macBeaconRxTime: " << m_macBeaconRxTime.As (Time::S) << ")");


                  //TODO: Handle mlme-scan.request here

                  // Device not associated.
                  if (m_macPanId == 0xffff)
                    {
                      //TODO: mlme-associate.request here.
                      NS_LOG_ERROR (this << "The current device is not associated to any coordinator");
                      return;
                    }


                  if (m_macPanId != receivedMacHdr.GetDstPanId ())
                    {
                      return;
                    }

                  BeaconPayloadHeader receivedMacPayload;
                  p->RemoveHeader (receivedMacPayload);

                  SuperframeField incomingSuperframe;
                  incomingSuperframe = receivedMacPayload.GetSuperframeSpecField ();

                  m_incomingBeaconOrder = incomingSuperframe.GetBeaconOrder ();
                  m_incomingSuperframeOrder = incomingSuperframe.GetFrameOrder ();
                  m_incomingFnlCapSlot = incomingSuperframe.GetFinalCapSlot ();

                  m_incomingBeaconInterval = ((uint32_t) 1 << m_incomingBeaconOrder) * aBaseSuperframeDuration;
                  m_incomingSuperframeDuration = aBaseSuperframeDuration * ((uint32_t) 1 << m_incomingSuperframeOrder);

                  if (incomingSuperframe.IsBattLifeExt ())
                    {
                      m_csmaCa->SetBatteryLifeExtension (true);
                    }
                  else
                    {
                      m_csmaCa->SetBatteryLifeExtension (false);
                    }

                  if (m_incomingBeaconOrder  < 15  && !m_csmaCa->IsSlottedCsmaCa ())
                    {
                      m_csmaCa->SetSlottedCsmaCa ();
                    }

                  //TODO: get Incoming frame GTS Fields here

                  NS_LOG_DEBUG ("Incoming superframe Active Portion (Beacon + CAP + CFP): " << m_incomingSuperframeDuration << " symbols");

                  //Begin CAP on the current device using info from the Incoming superframe
                  m_incCapEvent =  Simulator::ScheduleNow (&LrWpanMac::StartCAP, this,SuperframeType::INCOMING);


                  // Send a Beacon notification only if we are not
                  // automatically sending data command requests or the beacon contains
                  // a beacon payload in its MAC payload.
                  // see IEEE 802.15.4-2011 Section 6.2.4.1
                  if ((m_macAutoRequest == false) || p->GetSize () > 0)
                    {
                      //TODO: Add the rest of the MlmeBeaconNotifyIndication params
                      if (!m_mlmeBeaconNotifyIndicationCallback.IsNull ())
                        {
                          MlmeBeaconNotifyIndicationParams beaconParams;
                          beaconParams.m_bsn = receivedMacHdr.GetSeqNum ();
                          m_mlmeBeaconNotifyIndicationCallback (beaconParams,originalPkt);
                        }
                    }

                  if (m_macAutoRequest)
                    {
                      // check if MLME-SYNC.request was previously issued and running
                      //  Sync. is necessary to handle pending messages (indirect transmissions)
                      if (m_trackingEvent.IsRunning ())
                        {
                          m_trackingEvent.Cancel ();
                          m_numLostBeacons = 0;

                          if (m_beaconTrackingOn)
                            {
                              //if tracking option is on keep tracking the next beacon
                              uint64_t searchSymbols;
                              Time searchBeaconTime;

                              searchSymbols = ((uint64_t) 1 << m_incomingBeaconOrder) + 1 * aBaseSuperframeDuration;
                              searchBeaconTime = Seconds ((double) searchSymbols / symbolRate);
                              m_trackingEvent = Simulator::Schedule (searchBeaconTime, &LrWpanMac::BeaconSearchTimeout, this);
                            }

                          PendingAddrFields pndAddrFields;
                          pndAddrFields = receivedMacPayload.GetPndAddrFields ();

                          //TODO: Ignore pending request if the address is in the GTS list.
                          //      If the address is not in the GTS list, then  check if the address
                          //      is in the short address pending list or in the extended address
                          //      pending list.

                        }
                    }
                }
              else if (receivedMacHdr.IsData () && !m_mcpsDataIndicationCallback.IsNull ())
                {
                  // If it is a data frame, push it up the stack.
                  NS_LOG_DEBUG ("Data Packet is for me; forwarding up");
                  m_mcpsDataIndicationCallback (params, p);
                }
              else if (receivedMacHdr.IsAcknowledgment () && m_txPkt && m_lrWpanMacState == MAC_ACK_PENDING)
                {
                  LrWpanMacHeader macHdr;
                  Time ifsWaitTime = Seconds ((double) GetIfsSize () / symbolRate);
                  m_txPkt->PeekHeader (macHdr);
                  if (receivedMacHdr.GetSeqNum () == macHdr.GetSeqNum ())
                    {
                      m_macTxOkTrace (m_txPkt);
                      // If it is an ACK with the expected sequence number, finish the transmission
                      // and notify the upper layer.
                      m_ackWaitTimeout.Cancel ();
                      if (!m_mcpsDataConfirmCallback.IsNull ())
                        {
                          TxQueueElement *txQElement = m_txQueue.front ();
                          McpsDataConfirmParams confirmParams;
                          confirmParams.m_msduHandle = txQElement->txQMsduHandle;
                          confirmParams.m_status = IEEE_802_15_4_SUCCESS;
                          m_mcpsDataConfirmCallback (confirmParams);
                        }
                      RemoveFirstTxQElement ();
                      m_setMacState.Cancel ();
                      m_setMacState = Simulator::ScheduleNow (&LrWpanMac::SetLrWpanMacState, this, MAC_IDLE);
                      // Ack was succesfully received, wait for the Interframe Space (IFS) and then proceed
                      m_ifsEvent = Simulator::Schedule (ifsWaitTime, &LrWpanMac::IfsWaitTimeout, this, ifsWaitTime);
                    }
                  else
                    {
                      // If it is an ACK with an unexpected sequence number, mark the current transmission as failed and start a retransmit. (cf 7.5.6.4.3)
                      m_ackWaitTimeout.Cancel ();
                      if (!PrepareRetransmission ())
                        {
                          m_setMacState.Cancel ();
                          m_setMacState = Simulator::ScheduleNow (&LrWpanMac::SetLrWpanMacState, this, MAC_IDLE);
                        }
                      else
                        {
                          m_setMacState.Cancel ();
                          m_setMacState = Simulator::ScheduleNow (&LrWpanMac::SetLrWpanMacState, this, MAC_CSMA);
                        }
                    }
                }
            }
          else
            {
              m_macRxDropTrace (originalPkt);
            }
        }
    }
}



void
LrWpanMac::SendAck (uint8_t seqno)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (seqno));

  NS_ASSERT (m_lrWpanMacState == MAC_IDLE);

  // Generate a corresponding ACK Frame.
  LrWpanMacHeader macHdr (LrWpanMacHeader::LRWPAN_MAC_ACKNOWLEDGMENT, seqno);
  LrWpanMacTrailer macTrailer;
  Ptr<Packet> ackPacket = Create<Packet> (0);
  ackPacket->AddHeader (macHdr);
  // Calculate FCS if the global attribute ChecksumEnable is set.
  if (Node::ChecksumEnabled ())
    {
      macTrailer.EnableFcs (true);
      macTrailer.SetFcs (ackPacket);
    }
  ackPacket->AddTrailer (macTrailer);

  // Enqueue the ACK packet for further processing
  // when the transmitter is activated.
  m_txPkt = ackPacket;

  // Switch transceiver to TX mode. Proceed sending the Ack on confirm.
  ChangeMacState (MAC_SENDING);
  m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_TX_ON);

}

void
LrWpanMac::RemoveFirstTxQElement ()
{
  TxQueueElement *txQElement = m_txQueue.front ();
  Ptr<const Packet> p = txQElement->txQPkt;
  m_numCsmacaRetry += m_csmaCa->GetNB () + 1;

  Ptr<Packet> pkt = p->Copy ();
  LrWpanMacHeader hdr;
  pkt->RemoveHeader (hdr);
  if (!hdr.GetShortDstAddr ().IsBroadcast () && !hdr.GetShortDstAddr ().IsMulticast ())
    {
      m_sentPktTrace (p, m_retransmission + 1, m_numCsmacaRetry);
    }

  txQElement->txQPkt = 0;
  delete txQElement;
  m_txQueue.pop_front ();
  m_txPkt = 0;
  m_retransmission = 0;
  m_numCsmacaRetry = 0;
  m_macTxDequeueTrace (p);
}

void
LrWpanMac::AckWaitTimeout (void)
{
  NS_LOG_FUNCTION (this);

  // TODO: If we are a PAN coordinator and this was an indirect transmission,
  //       we will not initiate a retransmission. Instead we wait for the data
  //       being extracted after a new data request command.
  if (!PrepareRetransmission ())
    {
      SetLrWpanMacState (MAC_IDLE);
    }
  else
    {
      SetLrWpanMacState (MAC_CSMA);
    }
}

void
LrWpanMac::IfsWaitTimeout (Time ifsTime)
{
  uint64_t symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false);
  Time lifsTime = Seconds ((double) m_macLIFSPeriod / symbolRate);
  Time sifsTime = Seconds ((double) m_macSIFSPeriod / symbolRate);

  if (ifsTime == lifsTime)
    {
      NS_LOG_DEBUG ("LIFS of " << m_macLIFSPeriod << " symbols (" << ifsTime.As (Time::S) << ") completed ");
    }
  else if (ifsTime == sifsTime)
    {
      NS_LOG_DEBUG ("SIFS of " << m_macSIFSPeriod << " symbols (" << ifsTime.As (Time::S) << ") completed ");
    }
  else
    {
      NS_LOG_DEBUG ("Unknown IFS size (" << ifsTime.As (Time::S) << ") completed ");
    }

  m_macIfsEndTrace (ifsTime);
  CheckQueue ();
}


bool
LrWpanMac::PrepareRetransmission (void)
{
  NS_LOG_FUNCTION (this);

  if (m_retransmission >= m_macMaxFrameRetries)
    {
      // Maximum number of retransmissions has been reached.
      // remove the copy of the packet that was just sent
      TxQueueElement *txQElement = m_txQueue.front ();
      m_macTxDropTrace (txQElement->txQPkt);
      if (!m_mcpsDataConfirmCallback.IsNull ())
        {
          McpsDataConfirmParams confirmParams;
          confirmParams.m_msduHandle = txQElement->txQMsduHandle;
          confirmParams.m_status = IEEE_802_15_4_NO_ACK;
          m_mcpsDataConfirmCallback (confirmParams);
        }
      RemoveFirstTxQElement ();
      return false;
    }
  else
    {
      m_retransmission++;
      m_numCsmacaRetry += m_csmaCa->GetNB () + 1;
      // Start next CCA process for this packet.
      return true;
    }
}

void
LrWpanMac::PdDataConfirm (LrWpanPhyEnumeration status)
{
  NS_ASSERT (m_lrWpanMacState == MAC_SENDING);
  NS_LOG_FUNCTION (this << status << m_txQueue.size ());

  LrWpanMacHeader macHdr;
  Time ifsWaitTime;
  uint64_t symbolRate;

  symbolRate = (uint64_t) m_phy->GetDataOrSymbolRate (false); //symbols per second

  m_txPkt->PeekHeader (macHdr);

  if (status == IEEE_802_15_4_PHY_SUCCESS)
    {
      if (!macHdr.IsAcknowledgment ())
        {
          if (macHdr.IsBeacon ())
            {
              ifsWaitTime = Seconds ((double) GetIfsSize () / symbolRate);

              // The Tx Beacon in symbols
              // Beacon = 5 bytes Sync Header (SHR) +  1 byte PHY header (PHR) + PSDU (default 17 bytes)
              uint64_t beaconSymbols = m_phy->GetPhySHRDuration () + 1 * m_phy->GetPhySymbolsPerOctet () +
                (m_txPkt->GetSize () * m_phy->GetPhySymbolsPerOctet ());

              // The beacon Tx time and start of the Outgoing superframe Active Period
              m_macBeaconTxTime = Simulator::Now () - Seconds (double(beaconSymbols) / symbolRate);


              m_txPkt = 0;
              m_capEvent = Simulator::ScheduleNow (&LrWpanMac::StartCAP, this, SuperframeType::OUTGOING);
              NS_LOG_DEBUG ("Beacon Sent (m_macBeaconTxTime: " << m_macBeaconTxTime.As (Time::S) << ")");

              MlmeStartConfirmParams  mlmeConfirmParams;
              mlmeConfirmParams.m_status = MLMESTART_SUCCESS;
              if (!m_mlmeStartConfirmCallback.IsNull ())
                {
                  m_mlmeStartConfirmCallback (mlmeConfirmParams);
                }
            }
          else if (macHdr.IsAckReq ()) // We have sent a regular data packet, check if we have to wait  for an ACK.
            {
              // wait for the ack or the next retransmission timeout
              // start retransmission timer
              Time waitTime = Seconds ((double) GetMacAckWaitDuration () / symbolRate);
              NS_ASSERT (m_ackWaitTimeout.IsExpired ());
              m_ackWaitTimeout = Simulator::Schedule (waitTime, &LrWpanMac::AckWaitTimeout, this);
              m_setMacState.Cancel ();
              m_setMacState = Simulator::ScheduleNow (&LrWpanMac::SetLrWpanMacState, this, MAC_ACK_PENDING);
              return;
            }
          else
            {
              m_macTxOkTrace (m_txPkt);
              // remove the copy of the packet that was just sent
              if (!m_mcpsDataConfirmCallback.IsNull ())
                {
                  McpsDataConfirmParams confirmParams;
                  NS_ASSERT_MSG (m_txQueue.size () > 0, "TxQsize = 0");
                  TxQueueElement *txQElement = m_txQueue.front ();
                  confirmParams.m_msduHandle = txQElement->txQMsduHandle;
                  confirmParams.m_status = IEEE_802_15_4_SUCCESS;
                  m_mcpsDataConfirmCallback (confirmParams);
                }
              ifsWaitTime = Seconds ((double) GetIfsSize () / symbolRate);
              RemoveFirstTxQElement ();
            }
        }
      else
        {
          // We have send an ACK. Clear the packet buffer.
          m_txPkt = 0;
        }
    }
  else if (status == IEEE_802_15_4_PHY_UNSPECIFIED)
    {

      if (!macHdr.IsAcknowledgment ())
        {
          NS_ASSERT_MSG (m_txQueue.size () > 0, "TxQsize = 0");
          TxQueueElement *txQElement = m_txQueue.front ();
          m_macTxDropTrace (txQElement->txQPkt);
          if (!m_mcpsDataConfirmCallback.IsNull ())
            {
              McpsDataConfirmParams confirmParams;
              confirmParams.m_msduHandle = txQElement->txQMsduHandle;
              confirmParams.m_status = IEEE_802_15_4_FRAME_TOO_LONG;
              m_mcpsDataConfirmCallback (confirmParams);
            }
          RemoveFirstTxQElement ();
        }
      else
        {
          NS_LOG_ERROR ("Unable to send ACK");
        }
    }
  else
    {
      // Something went really wrong. The PHY is not in the correct state for
      // data transmission.
      NS_FATAL_ERROR ("Transmission attempt failed with PHY status " << status);
    }


  if (!ifsWaitTime.IsZero ())
    {
      m_ifsEvent = Simulator::Schedule (ifsWaitTime, &LrWpanMac::IfsWaitTimeout, this, ifsWaitTime);
    }

  m_setMacState.Cancel ();
  m_setMacState = Simulator::ScheduleNow (&LrWpanMac::SetLrWpanMacState, this, MAC_IDLE);


}

void
LrWpanMac::PlmeCcaConfirm (LrWpanPhyEnumeration status)
{
  NS_LOG_FUNCTION (this << status);
  // Direct this call through the csmaCa object
  m_csmaCa->PlmeCcaConfirm (status);
}

void
LrWpanMac::PlmeEdConfirm (LrWpanPhyEnumeration status, uint8_t energyLevel)
{
  NS_LOG_FUNCTION (this << status << energyLevel);

}

void
LrWpanMac::PlmeGetAttributeConfirm (LrWpanPhyEnumeration status,
                                    LrWpanPibAttributeIdentifier id,
                                    LrWpanPhyPibAttributes* attribute)
{
  NS_LOG_FUNCTION (this << status << id << attribute);
}

void
LrWpanMac::PlmeSetTRXStateConfirm (LrWpanPhyEnumeration status)
{
  NS_LOG_FUNCTION (this << status);

  if (m_lrWpanMacState == MAC_SENDING && (status == IEEE_802_15_4_PHY_TX_ON || status == IEEE_802_15_4_PHY_SUCCESS))
    {
      NS_ASSERT (m_txPkt);

      // Start sending if we are in state SENDING and the PHY transmitter was enabled.
      m_promiscSnifferTrace (m_txPkt);
      m_snifferTrace (m_txPkt);
      m_macTxTrace (m_txPkt);
      m_phy->PdDataRequest (m_txPkt->GetSize (), m_txPkt);
    }
  else if (m_lrWpanMacState == MAC_CSMA && (status == IEEE_802_15_4_PHY_RX_ON || status == IEEE_802_15_4_PHY_SUCCESS))
    {
      // Start the CSMA algorithm as soon as the receiver is enabled.
      m_csmaCa->Start ();
    }
  else if (m_lrWpanMacState == MAC_IDLE)
    {
      NS_ASSERT (status == IEEE_802_15_4_PHY_RX_ON || status == IEEE_802_15_4_PHY_SUCCESS || status == IEEE_802_15_4_PHY_TRX_OFF);

      if (status == IEEE_802_15_4_PHY_RX_ON || status == IEEE_802_15_4_PHY_SUCCESS)
        {
          // Check if there is not messages to transmit when going idle
          CheckQueue ();
        }

    }
  else if (m_lrWpanMacState == MAC_ACK_PENDING)
    {
      NS_ASSERT (status == IEEE_802_15_4_PHY_RX_ON || status == IEEE_802_15_4_PHY_SUCCESS);
    }
  else
    {
      // TODO: What to do when we receive an error?
      // If we want to transmit a packet, but switching the transceiver on results
      // in an error, we have to recover somehow (and start sending again).
      NS_FATAL_ERROR ("Error changing transceiver state");
    }
}

void
LrWpanMac::PlmeSetAttributeConfirm (LrWpanPhyEnumeration status,
                                    LrWpanPibAttributeIdentifier id)
{
  NS_LOG_FUNCTION (this << status << id);
}

void
LrWpanMac::SetLrWpanMacState (LrWpanMacState macState)
{
  NS_LOG_FUNCTION (this << "mac state = " << macState);

  McpsDataConfirmParams confirmParams;

  if (macState == MAC_IDLE)
    {
      ChangeMacState (MAC_IDLE);

      if (m_macRxOnWhenIdle)
        {
          m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_RX_ON);
        }
      else
        {
          m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_TRX_OFF);
        }


    }
  else if (macState == MAC_ACK_PENDING)
    {
      ChangeMacState (MAC_ACK_PENDING);
      m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_RX_ON);
    }
  else if (macState == MAC_CSMA)
    {
      NS_ASSERT (m_lrWpanMacState == MAC_IDLE || m_lrWpanMacState == MAC_ACK_PENDING);

      ChangeMacState (MAC_CSMA);
      m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_RX_ON);
    }
  else if (m_lrWpanMacState == MAC_CSMA && macState == CHANNEL_IDLE)
    {
      // Channel is idle, set transmitter to TX_ON
      ChangeMacState (MAC_SENDING);
      m_phy->PlmeSetTRXStateRequest (IEEE_802_15_4_PHY_TX_ON);
    }
  else if (m_lrWpanMacState == MAC_CSMA && macState == CHANNEL_ACCESS_FAILURE)
    {
      NS_ASSERT (m_txPkt);

      // cannot find a clear channel, drop the current packet.
      NS_LOG_DEBUG ( this << " cannot find clear channel");
      confirmParams.m_msduHandle = m_txQueue.front ()->txQMsduHandle;
      confirmParams.m_status = IEEE_802_15_4_CHANNEL_ACCESS_FAILURE;
      m_macTxDropTrace (m_txPkt);
      if (!m_mcpsDataConfirmCallback.IsNull ())
        {
          m_mcpsDataConfirmCallback (confirmParams);
        }
      // remove the copy of the packet that was just sent
      RemoveFirstTxQElement ();
      ChangeMacState (MAC_IDLE);
    }
  else if (m_lrWpanMacState == MAC_CSMA && macState == MAC_CSMA_DEFERRED)
    {
      ChangeMacState (MAC_IDLE);
      m_txPkt = 0;
      NS_LOG_DEBUG ("****** PACKET DEFERRED to the next superframe *****");
    }
}

LrWpanAssociationStatus
LrWpanMac::GetAssociationStatus (void) const
{
  return m_associationStatus;
}

void
LrWpanMac::SetAssociationStatus (LrWpanAssociationStatus status)
{
  m_associationStatus = status;
}

uint16_t
LrWpanMac::GetPanId (void) const
{
  return m_macPanId;
}

void
LrWpanMac::SetPanId (uint16_t panId)
{
  m_macPanId = panId;
}

void
LrWpanMac::ChangeMacState (LrWpanMacState newState)
{
  NS_LOG_LOGIC (this << " change lrwpan mac state from "
                     << m_lrWpanMacState << " to "
                     << newState);
  m_macStateLogger (m_lrWpanMacState, newState);
  m_lrWpanMacState = newState;
}

uint64_t
LrWpanMac::GetMacAckWaitDuration (void) const
{
  return m_csmaCa->GetUnitBackoffPeriod () + m_phy->aTurnaroundTime + m_phy->GetPhySHRDuration ()
         + ceil (6 * m_phy->GetPhySymbolsPerOctet ());
}

uint8_t
LrWpanMac::GetMacMaxFrameRetries (void) const
{
  return m_macMaxFrameRetries;
}

void
LrWpanMac::PrintTransmitQueueSize (void)
{
  NS_LOG_DEBUG ("Transmit Queue Size: " << m_txQueue.size ());
}

void
LrWpanMac::SetMacMaxFrameRetries (uint8_t retries)
{
  m_macMaxFrameRetries = retries;
}

bool
LrWpanMac::isCoordDest (void)
{
  NS_ASSERT (m_txPkt);
  LrWpanMacHeader macHdr;
  m_txPkt->PeekHeader (macHdr);

  if (m_macCoordShortAddress == macHdr.GetShortDstAddr ()
      || m_macCoordExtendedAddress == macHdr.GetExtDstAddr ())
    {
      return true;
    }
  else
    {
      std::cout << "ERROR: Packet not for the coordinator!\n";
      return false;

    }

}

uint32_t
LrWpanMac::GetIfsSize ()
{
  NS_ASSERT (m_txPkt);

  if (m_txPkt->GetSize () <= aMaxSIFSFrameSize)
    {
      return m_macSIFSPeriod;
    }
  else
    {
      return m_macLIFSPeriod;
    }
}

void
LrWpanMac::SetAssociatedCoor (Mac16Address mac)
{
  m_macCoordShortAddress = mac;
}

void
LrWpanMac::SetAssociatedCoor (Mac64Address mac)
{
  m_macCoordExtendedAddress = mac;
}


uint64_t
LrWpanMac::GetTxPacketSymbols (void)
{
  NS_ASSERT (m_txPkt);
  // Sync Header (SHR) +  8 bits PHY header (PHR) + PSDU
  return (m_phy->GetPhySHRDuration () + 1 * m_phy->GetPhySymbolsPerOctet () +
          (m_txPkt->GetSize () * m_phy->GetPhySymbolsPerOctet ()));
}

bool
LrWpanMac::isTxAckReq (void)
{
  NS_ASSERT (m_txPkt);
  LrWpanMacHeader macHdr;
  m_txPkt->PeekHeader (macHdr);

  return macHdr.IsAckReq ();
}

} // namespace ns3

