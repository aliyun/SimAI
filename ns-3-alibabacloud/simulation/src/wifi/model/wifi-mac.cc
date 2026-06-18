/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 */

#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/packet.h"
#include "wifi-mac.h"
#include "qos-txop.h"
#include "ssid.h"
#include "mgt-headers.h"
#include "wifi-net-device.h"
#include "wifi-mac-queue.h"
#include "mac-rx-middle.h"
#include "mac-tx-middle.h"
#include "channel-access-manager.h"
#include "ns3/ht-configuration.h"
#include "ns3/vht-configuration.h"
#include "ns3/he-configuration.h"
#include "ns3/he-frame-exchange-manager.h"
#include "extended-capabilities.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiMac");

NS_OBJECT_ENSURE_REGISTERED (WifiMac);

WifiMac::WifiMac ()
  : m_qosSupported (false),
    m_erpSupported (false),
    m_dsssSupported (false)
{
  NS_LOG_FUNCTION (this);

  m_rxMiddle = Create<MacRxMiddle> ();
  m_rxMiddle->SetForwardCallback (MakeCallback (&WifiMac::Receive, this));

  m_txMiddle = Create<MacTxMiddle> ();

  m_channelAccessManager = CreateObject<ChannelAccessManager> ();
}

WifiMac::~WifiMac ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
WifiMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiMac")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddAttribute ("Ssid", "The ssid we want to belong to.",
                   SsidValue (Ssid ("default")),
                   MakeSsidAccessor (&WifiMac::GetSsid,
                                     &WifiMac::SetSsid),
                   MakeSsidChecker ())
    .AddAttribute ("QosSupported",
                   "This Boolean attribute is set to enable 802.11e/WMM-style QoS support at this STA.",
                   TypeId::ATTR_GET | TypeId::ATTR_CONSTRUCT,  // prevent setting after construction
                   BooleanValue (false),
                   MakeBooleanAccessor (&WifiMac::SetQosSupported,
                                        &WifiMac::GetQosSupported),
                   MakeBooleanChecker ())
    .AddAttribute ("CtsToSelfSupported",
                   "Use CTS to Self when using a rate that is not in the basic rate set.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&WifiMac::SetCtsToSelfSupported),
                   MakeBooleanChecker ())
    .AddAttribute ("ShortSlotTimeSupported",
                   "Whether or not short slot time is supported (only used by ERP APs or STAs).",
                   BooleanValue (true),
                   MakeBooleanAccessor (&WifiMac::SetShortSlotTimeSupported,
                                        &WifiMac::GetShortSlotTimeSupported),
                   MakeBooleanChecker ())
    .AddAttribute ("Txop",
                   "The Txop object.",
                   PointerValue (),
                   MakePointerAccessor (&WifiMac::GetTxop),
                   MakePointerChecker<Txop> ())
    .AddAttribute ("VO_Txop",
                   "Queue that manages packets belonging to AC_VO access class.",
                   PointerValue (),
                   MakePointerAccessor (&WifiMac::GetVOQueue),
                   MakePointerChecker<QosTxop> ())
    .AddAttribute ("VI_Txop",
                   "Queue that manages packets belonging to AC_VI access class.",
                   PointerValue (),
                   MakePointerAccessor (&WifiMac::GetVIQueue),
                   MakePointerChecker<QosTxop> ())
    .AddAttribute ("BE_Txop",
                   "Queue that manages packets belonging to AC_BE access class.",
                   PointerValue (),
                   MakePointerAccessor (&WifiMac::GetBEQueue),
                   MakePointerChecker<QosTxop> ())
    .AddAttribute ("BK_Txop",
                   "Queue that manages packets belonging to AC_BK access class.",
                   PointerValue (),
                   MakePointerAccessor (&WifiMac::GetBKQueue),
                   MakePointerChecker<QosTxop> ())
    .AddAttribute ("VO_MaxAmsduSize",
                   "Maximum length in bytes of an A-MSDU for AC_VO access class "
                   "(capped to 7935 for HT PPDUs and 11398 for VHT/HE PPDUs). "
                   "Value 0 means A-MSDU aggregation is disabled for that AC.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::m_voMaxAmsduSize),
                   MakeUintegerChecker<uint16_t> (0, 11398))
    .AddAttribute ("VI_MaxAmsduSize",
                   "Maximum length in bytes of an A-MSDU for AC_VI access class "
                   "(capped to 7935 for HT PPDUs and 11398 for VHT/HE PPDUs). "
                   "Value 0 means A-MSDU aggregation is disabled for that AC.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::m_viMaxAmsduSize),
                   MakeUintegerChecker<uint16_t> (0, 11398))
    .AddAttribute ("BE_MaxAmsduSize",
                   "Maximum length in bytes of an A-MSDU for AC_BE access class "
                   "(capped to 7935 for HT PPDUs and 11398 for VHT/HE PPDUs). "
                   "Value 0 means A-MSDU aggregation is disabled for that AC.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::m_beMaxAmsduSize),
                   MakeUintegerChecker<uint16_t> (0, 11398))
    .AddAttribute ("BK_MaxAmsduSize",
                   "Maximum length in bytes of an A-MSDU for AC_BK access class "
                   "(capped to 7935 for HT PPDUs and 11398 for VHT/HE PPDUs). "
                   "Value 0 means A-MSDU aggregation is disabled for that AC.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::m_bkMaxAmsduSize),
                   MakeUintegerChecker<uint16_t> (0, 11398))
    .AddAttribute ("VO_MaxAmpduSize",
                   "Maximum length in bytes of an A-MPDU for AC_VO access class "
                   "(capped to 65535 for HT PPDUs, 1048575 for VHT PPDUs, and 6500631 for HE PPDUs). "
                   "Value 0 means A-MPDU aggregation is disabled for that AC.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::m_voMaxAmpduSize),
                   MakeUintegerChecker<uint32_t> (0, 6500631))
    .AddAttribute ("VI_MaxAmpduSize",
                   "Maximum length in bytes of an A-MPDU for AC_VI access class "
                   "(capped to 65535 for HT PPDUs, 1048575 for VHT PPDUs, and 6500631 for HE PPDUs). "
                   "Value 0 means A-MPDU aggregation is disabled for that AC.",
                   UintegerValue (65535),
                   MakeUintegerAccessor (&WifiMac::m_viMaxAmpduSize),
                   MakeUintegerChecker<uint32_t> (0, 6500631))
    .AddAttribute ("BE_MaxAmpduSize",
                   "Maximum length in bytes of an A-MPDU for AC_BE access class "
                   "(capped to 65535 for HT PPDUs, 1048575 for VHT PPDUs, and 6500631 for HE PPDUs). "
                   "Value 0 means A-MPDU aggregation is disabled for that AC.",
                   UintegerValue (65535),
                   MakeUintegerAccessor (&WifiMac::m_beMaxAmpduSize),
                   MakeUintegerChecker<uint32_t> (0, 6500631))
    .AddAttribute ("BK_MaxAmpduSize",
                   "Maximum length in bytes of an A-MPDU for AC_BK access class "
                   "(capped to 65535 for HT PPDUs, 1048575 for VHT PPDUs, and 6500631 for HE PPDUs). "
                   "Value 0 means A-MPDU aggregation is disabled for that AC.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::m_bkMaxAmpduSize),
                   MakeUintegerChecker<uint32_t> (0, 6500631))
    .AddAttribute ("VO_BlockAckThreshold",
                   "If number of packets in VO queue reaches this value, "
                   "block ack mechanism is used. If this value is 0, block ack is never used."
                   "When A-MPDU is enabled, block ack mechanism is used regardless of this value.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetVoBlockAckThreshold),
                   MakeUintegerChecker<uint8_t> (0, 64))
    .AddAttribute ("VI_BlockAckThreshold",
                   "If number of packets in VI queue reaches this value, "
                   "block ack mechanism is used. If this value is 0, block ack is never used."
                   "When A-MPDU is enabled, block ack mechanism is used regardless of this value.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetViBlockAckThreshold),
                   MakeUintegerChecker<uint8_t> (0, 64))
    .AddAttribute ("BE_BlockAckThreshold",
                   "If number of packets in BE queue reaches this value, "
                   "block ack mechanism is used. If this value is 0, block ack is never used."
                   "When A-MPDU is enabled, block ack mechanism is used regardless of this value.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetBeBlockAckThreshold),
                   MakeUintegerChecker<uint8_t> (0, 64))
    .AddAttribute ("BK_BlockAckThreshold",
                   "If number of packets in BK queue reaches this value, "
                   "block ack mechanism is used. If this value is 0, block ack is never used."
                   "When A-MPDU is enabled, block ack mechanism is used regardless of this value.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetBkBlockAckThreshold),
                   MakeUintegerChecker<uint8_t> (0, 64))
    .AddAttribute ("VO_BlockAckInactivityTimeout",
                   "Represents max time (blocks of 1024 microseconds) allowed for block ack"
                   "inactivity for AC_VO. If this value isn't equal to 0 a timer start after that a"
                   "block ack setup is completed and will be reset every time that a block ack"
                   "frame is received. If this value is 0, block ack inactivity timeout won't be used.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetVoBlockAckInactivityTimeout),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("VI_BlockAckInactivityTimeout",
                   "Represents max time (blocks of 1024 microseconds) allowed for block ack"
                   "inactivity for AC_VI. If this value isn't equal to 0 a timer start after that a"
                   "block ack setup is completed and will be reset every time that a block ack"
                   "frame is received. If this value is 0, block ack inactivity timeout won't be used.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetViBlockAckInactivityTimeout),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("BE_BlockAckInactivityTimeout",
                   "Represents max time (blocks of 1024 microseconds) allowed for block ack"
                   "inactivity for AC_BE. If this value isn't equal to 0 a timer start after that a"
                   "block ack setup is completed and will be reset every time that a block ack"
                   "frame is received. If this value is 0, block ack inactivity timeout won't be used.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetBeBlockAckInactivityTimeout),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("BK_BlockAckInactivityTimeout",
                   "Represents max time (blocks of 1024 microseconds) allowed for block ack"
                   "inactivity for AC_BK. If this value isn't equal to 0 a timer start after that a"
                   "block ack setup is completed and will be reset every time that a block ack"
                   "frame is received. If this value is 0, block ack inactivity timeout won't be used.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WifiMac::SetBkBlockAckInactivityTimeout),
                   MakeUintegerChecker<uint16_t> ())
    .AddTraceSource ("MacTx",
                     "A packet has been received from higher layers and is being processed in preparation for "
                     "queueing for transmission.",
                     MakeTraceSourceAccessor (&WifiMac::m_macTxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacTxDrop",
                     "A packet has been dropped in the MAC layer before being queued for transmission. "
                     "This trace source is fired, e.g., when an AP's MAC receives from the upper layer "
                     "a packet destined to a station that is not associated with the AP or a STA's MAC "
                     "receives a packet from the upper layer while it is not associated with any AP.",
                     MakeTraceSourceAccessor (&WifiMac::m_macTxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacPromiscRx",
                     "A packet has been received by this device, has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  This is a promiscuous trace.",
                     MakeTraceSourceAccessor (&WifiMac::m_macPromiscRxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacRx",
                     "A packet has been received by this device, has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack. This is a non-promiscuous trace.",
                     MakeTraceSourceAccessor (&WifiMac::m_macRxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacRxDrop",
                     "A packet has been dropped in the MAC layer after it has been passed up from the physical layer.",
                     MakeTraceSourceAccessor (&WifiMac::m_macRxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("TxOkHeader",
                     "The header of successfully transmitted packet.",
                     MakeTraceSourceAccessor (&WifiMac::m_txOkCallback),
                     "ns3::WifiMacHeader::TracedCallback",
                     TypeId::OBSOLETE,
                     "Use the AckedMpdu trace instead.")
    .AddTraceSource ("TxErrHeader",
                     "The header of unsuccessfully transmitted packet.",
                     MakeTraceSourceAccessor (&WifiMac::m_txErrCallback),
                     "ns3::WifiMacHeader::TracedCallback",
                     TypeId::OBSOLETE,
                     "Depending on the failure type, use the NAckedMpdu trace, the "
                     "DroppedMpdu trace or one of the traces associated with TX timeouts.")
    .AddTraceSource ("AckedMpdu",
                     "An MPDU that was successfully acknowledged, via either a "
                     "Normal Ack or a Block Ack.",
                     MakeTraceSourceAccessor (&WifiMac::m_ackedMpduCallback),
                     "ns3::WifiMacQueueItem::TracedCallback")
    .AddTraceSource ("NAckedMpdu",
                     "An MPDU that was negatively acknowledged via a Block Ack.",
                     MakeTraceSourceAccessor (&WifiMac::m_nackedMpduCallback),
                     "ns3::WifiMacQueueItem::TracedCallback")
    .AddTraceSource ("DroppedMpdu",
                     "An MPDU that was dropped for the given reason (see WifiMacDropReason).",
                     MakeTraceSourceAccessor (&WifiMac::m_droppedMpduCallback),
                     "ns3::WifiMac::DroppedMpduCallback")
    .AddTraceSource ("MpduResponseTimeout",
                     "An MPDU whose response was not received before the timeout, along with "
                     "an identifier of the type of timeout (see WifiTxTimer::Reason) and the "
                     "TXVECTOR used to transmit the MPDU. This trace source is fired when a "
                     "CTS is missing after an RTS or a Normal Ack is missing after an MPDU "
                     "or after a DL MU PPDU acknowledged in SU format.",
                     MakeTraceSourceAccessor (&WifiMac::m_mpduResponseTimeoutCallback),
                     "ns3::WifiMac::MpduResponseTimeoutCallback")
    .AddTraceSource ("PsduResponseTimeout",
                     "A PSDU whose response was not received before the timeout, along with "
                     "an identifier of the type of timeout (see WifiTxTimer::Reason) and the "
                     "TXVECTOR used to transmit the PSDU. This trace source is fired when a "
                     "BlockAck is missing after an A-MPDU, a BlockAckReq (possibly in the "
                     "context of the acknowledgment of a DL MU PPDU in SU format) or a TB PPDU "
                     "(in the latter case the missing BlockAck is a Multi-STA BlockAck).",
                     MakeTraceSourceAccessor (&WifiMac::m_psduResponseTimeoutCallback),
                     "ns3::WifiMac::PsduResponseTimeoutCallback")
    .AddTraceSource ("PsduMapResponseTimeout",
                     "A PSDU map for which not all the responses were received before the timeout, "
                     "along with an identifier of the type of timeout (see WifiTxTimer::Reason), "
                     "the set of MAC addresses of the stations that did not respond and the total "
                     "number of stations that had to respond. This trace source is fired when not "
                     "all the addressed stations responded to an MU-BAR Trigger frame (either sent as "
                     "a SU frame or aggregated to PSDUs in the DL MU PPDU), a Basic Trigger Frame or "
                     "a BSRP Trigger Frame.",
                     MakeTraceSourceAccessor (&WifiMac::m_psduMapResponseTimeoutCallback),
                     "ns3::WifiMac::PsduMapResponseTimeoutCallback")
  ;
  return tid;
}


void
WifiMac::DoInitialize ()
{
  NS_LOG_FUNCTION (this);

  if (m_txop != nullptr)
    {
      m_txop->Initialize ();
    }

  for (auto it = m_edca.begin (); it != m_edca.end (); ++it)
    {
      it->second->Initialize ();
    }
}

void
WifiMac::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  m_rxMiddle = 0;
  m_txMiddle = 0;

  m_channelAccessManager->Dispose ();
  m_channelAccessManager = 0;

  if (m_txop != nullptr)
    {
      m_txop->Dispose ();
    }
  m_txop = 0;

  for (auto it = m_edca.begin (); it != m_edca.end (); ++it)
    {
      it->second->Dispose ();
      it->second = 0;
    }

  if (m_feManager != 0)
    {
      m_feManager->Dispose ();
    }
  m_feManager = 0;

  m_stationManager = 0;
  m_phy = 0;

  m_device = 0;
}

void
WifiMac::SetTypeOfStation (TypeOfStation type)
{
  NS_LOG_FUNCTION (this << type);
  m_typeOfStation = type;
}

TypeOfStation
WifiMac::GetTypeOfStation (void) const
{
  return m_typeOfStation;
}

void
WifiMac::SetDevice (const Ptr<WifiNetDevice> device)
{
  m_device = device;
}

Ptr<WifiNetDevice>
WifiMac::GetDevice (void) const
{
  return m_device;
}

void
WifiMac::SetAddress (Mac48Address address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = address;
}

Mac48Address
WifiMac::GetAddress (void) const
{
  return m_address;
}

void
WifiMac::SetSsid (Ssid ssid)
{
  NS_LOG_FUNCTION (this << ssid);
  m_ssid = ssid;
}

Ssid
WifiMac::GetSsid (void) const
{
  return m_ssid;
}

void
WifiMac::SetBssid (Mac48Address bssid)
{
  NS_LOG_FUNCTION (this << bssid);
  m_bssid = bssid;
  if (m_feManager)
    {
      m_feManager->SetBssid (bssid);
    }
}

Mac48Address
WifiMac::GetBssid (void) const
{
  return m_bssid;
}

void
WifiMac::SetPromisc (void)
{
  NS_ASSERT (m_feManager != 0);
  m_feManager->SetPromisc ();
}

Ptr<Txop>
WifiMac::GetTxop () const
{
  return m_txop;
}

Ptr<QosTxop>
WifiMac::GetQosTxop (AcIndex ac) const
{
  // Use std::find_if() instead of std::map::find() because the latter compares
  // the given AC index with the AC index of an element in the map by using the
  // operator< defined for AcIndex, which aborts if an operand is not a QoS AC
  // (the AC index passed to this method may not be a QoS AC).
  // The performance penalty is limited because std::map::find() performs 3
  // comparisons in the worst case, while std::find_if() performs 4 comparisons
  // in the worst case.
  const auto it = std::find_if (m_edca.cbegin (), m_edca.cend (),
                                [ac](const auto& pair){ return pair.first == ac; });
  return (it == m_edca.cend () ? nullptr : it->second);
}

Ptr<QosTxop>
WifiMac::GetQosTxop (uint8_t tid) const
{
  return GetQosTxop (QosUtilsMapTidToAc (tid));
}

Ptr<QosTxop>
WifiMac::GetVOQueue () const
{
  return (m_qosSupported ? GetQosTxop (AC_VO) : nullptr);
}

Ptr<QosTxop>
WifiMac::GetVIQueue () const
{
  return (m_qosSupported ? GetQosTxop (AC_VI) : nullptr);
}

Ptr<QosTxop>
WifiMac::GetBEQueue () const
{
  return (m_qosSupported ? GetQosTxop (AC_BE) : nullptr);
}

Ptr<QosTxop>
WifiMac::GetBKQueue () const
{
  return (m_qosSupported ? GetQosTxop (AC_BK) : nullptr);
}

Ptr<WifiMacQueue>
WifiMac::GetTxopQueue (AcIndex ac) const
{
  Ptr<Txop> txop = (ac == AC_BE_NQOS ? m_txop : StaticCast<Txop> (GetQosTxop (ac)));
  return (txop != nullptr ? txop->GetWifiMacQueue () : nullptr);
}

void
WifiMac::NotifyChannelSwitching (void)
{
  NS_LOG_FUNCTION (this);

  // we may have changed PHY band, in which case it is necessary to re-configure
  // the PHY dependent parameters. In any case, this makes no harm
  ConfigurePhyDependentParameters ();

  // SetupPhy not only resets the remote station manager, but also sets the
  // default TX mode and MCS, which is required when switching to a channel
  // in a different band
  m_stationManager->SetupPhy (m_phy);
}

void
WifiMac::NotifyTx (Ptr<const Packet> packet)
{
  m_macTxTrace (packet);
}

void
WifiMac::NotifyTxDrop (Ptr<const Packet> packet)
{
  m_macTxDropTrace (packet);
}

void
WifiMac::NotifyRx (Ptr<const Packet> packet)
{
  m_macRxTrace (packet);
}

void
WifiMac::NotifyPromiscRx (Ptr<const Packet> packet)
{
  m_macPromiscRxTrace (packet);
}

void
WifiMac::NotifyRxDrop (Ptr<const Packet> packet)
{
  m_macRxDropTrace (packet);
}

void
WifiMac::SetupEdcaQueue (AcIndex ac)
{
  NS_LOG_FUNCTION (this << ac);

  //Our caller shouldn't be attempting to setup a queue that is
  //already configured.
  NS_ASSERT (m_edca.find (ac) == m_edca.end ());

  Ptr<QosTxop> edca = CreateObject<QosTxop> (ac);
  edca->SetChannelAccessManager (m_channelAccessManager);
  edca->SetWifiMac (this);
  edca->SetTxMiddle (m_txMiddle);
  edca->GetBaManager ()->SetTxOkCallback (MakeCallback (&MpduTracedCallback::operator(),
                                                        &m_ackedMpduCallback));
  edca->GetBaManager ()->SetTxFailedCallback (MakeCallback (&MpduTracedCallback::operator(),
                                                            &m_nackedMpduCallback));
  edca->SetDroppedMpduCallback (MakeCallback (&DroppedMpduTracedCallback::operator(),
                                              &m_droppedMpduCallback));

  m_edca.insert (std::make_pair (ac, edca));
}

void
WifiMac::ConfigureContentionWindow (uint32_t cwMin, uint32_t cwMax)
{
  bool isDsssOnly = GetDsssSupported () && !GetErpSupported ();
  if (m_txop != nullptr)
    {
      //The special value of AC_BE_NQOS which exists in the Access
      //Category enumeration allows us to configure plain old DCF.
      ConfigureDcf (m_txop, cwMin, cwMax, isDsssOnly, AC_BE_NQOS);
    }

  //Now we configure the EDCA functions
  for (auto it = m_edca.begin (); it!= m_edca.end (); ++it)
    {
      ConfigureDcf (it->second, cwMin, cwMax, isDsssOnly, it->first);
    }
}

void
WifiMac::ConfigureDcf (Ptr<Txop> dcf, uint32_t cwmin, uint32_t cwmax, bool isDsss, AcIndex ac)
{
  NS_LOG_FUNCTION (this << dcf << cwmin << cwmax << isDsss << +ac);
  /* see IEEE 802.11 section 7.3.2.29 */
  switch (ac)
    {
    case AC_VO:
      dcf->SetMinCw ((cwmin + 1) / 4 - 1);
      dcf->SetMaxCw ((cwmin + 1) / 2 - 1);
      dcf->SetAifsn (2);
      if (isDsss)
        {
          dcf->SetTxopLimit (MicroSeconds (3264));
        }
      else
        {
          dcf->SetTxopLimit (MicroSeconds (1504));
        }
      break;
    case AC_VI:
      dcf->SetMinCw ((cwmin + 1) / 2 - 1);
      dcf->SetMaxCw (cwmin);
      dcf->SetAifsn (2);
      if (isDsss)
        {
          dcf->SetTxopLimit (MicroSeconds (6016));
        }
      else
        {
          dcf->SetTxopLimit (MicroSeconds (3008));
        }
      break;
    case AC_BE:
      dcf->SetMinCw (cwmin);
      dcf->SetMaxCw (cwmax);
      dcf->SetAifsn (3);
      dcf->SetTxopLimit (MicroSeconds (0));
      break;
    case AC_BK:
      dcf->SetMinCw (cwmin);
      dcf->SetMaxCw (cwmax);
      dcf->SetAifsn (7);
      dcf->SetTxopLimit (MicroSeconds (0));
      break;
    case AC_BE_NQOS:
      dcf->SetMinCw (cwmin);
      dcf->SetMaxCw (cwmax);
      dcf->SetAifsn (2);
      dcf->SetTxopLimit (MicroSeconds (0));
      break;
    case AC_BEACON:
      // done by ApWifiMac
      break;
    case AC_UNDEF:
      NS_FATAL_ERROR ("I don't know what to do with this");
      break;
    }
}

void
WifiMac::ConfigureStandard (WifiStandard standard)
{
  NS_LOG_FUNCTION (this << standard);

  NS_ABORT_IF (standard >= WIFI_STANDARD_80211n && !m_qosSupported);

  SetupFrameExchangeManager (standard);
}

void
WifiMac::ConfigurePhyDependentParameters (void)
{
  WifiPhyBand band = m_phy->GetPhyBand ();
  NS_LOG_FUNCTION (this << band);

  uint32_t cwmin = 0;
  uint32_t cwmax = 0;

  NS_ASSERT (m_phy != 0);
  WifiStandard standard = m_phy->GetStandard ();

  if (standard == WIFI_STANDARD_80211b)
    {
      SetDsssSupported (true);
      cwmin = 31;
      cwmax = 1023;
    }
  else
    {
      if (standard >= WIFI_STANDARD_80211g && band == WIFI_PHY_BAND_2_4GHZ)
        {
          SetErpSupported (true);
        }

      cwmin = 15;
      cwmax = 1023;
    }

  ConfigureContentionWindow (cwmin, cwmax);
}

void
WifiMac::SetupFrameExchangeManager (WifiStandard standard)
{
  NS_LOG_FUNCTION (this << standard);
  NS_ABORT_MSG_IF (standard == WIFI_STANDARD_UNSPECIFIED, "Wifi standard not set");

  if (standard >= WIFI_STANDARD_80211ax)
    {
      m_feManager = CreateObject<HeFrameExchangeManager> ();
    }
  else if (standard >= WIFI_STANDARD_80211ac)
    {
      m_feManager = CreateObject<VhtFrameExchangeManager> ();
    }
  else if (standard >= WIFI_STANDARD_80211n)
    {
      m_feManager = CreateObject<HtFrameExchangeManager> ();
    }
  else if (m_qosSupported)
    {
      m_feManager = CreateObject<QosFrameExchangeManager> ();
    }
  else
    {
      m_feManager = CreateObject<FrameExchangeManager> ();
    }

  m_feManager->SetWifiMac (this);
  m_feManager->SetMacTxMiddle (m_txMiddle);
  m_feManager->SetMacRxMiddle (m_rxMiddle);
  m_feManager->SetAddress (GetAddress ());
  m_feManager->SetBssid (GetBssid ());
  m_feManager->GetWifiTxTimer ().SetMpduResponseTimeoutCallback (MakeCallback (&MpduResponseTimeoutTracedCallback::operator(),
                                                                               &m_mpduResponseTimeoutCallback));
  m_feManager->GetWifiTxTimer ().SetPsduResponseTimeoutCallback (MakeCallback (&PsduResponseTimeoutTracedCallback::operator(),
                                                                               &m_psduResponseTimeoutCallback));
  m_feManager->GetWifiTxTimer ().SetPsduMapResponseTimeoutCallback (MakeCallback (&PsduMapResponseTimeoutTracedCallback::operator(),
                                                                                  &m_psduMapResponseTimeoutCallback));
  m_feManager->SetDroppedMpduCallback (MakeCallback (&DroppedMpduTracedCallback::operator(),
                                                     &m_droppedMpduCallback));
  m_feManager->SetAckedMpduCallback (MakeCallback (&MpduTracedCallback::operator(),
                                                   &m_ackedMpduCallback));
  m_channelAccessManager->SetupFrameExchangeManager (m_feManager);
  if (GetQosSupported ())
    {
      for (const auto& pair : m_edca)
        {
          pair.second->SetQosFrameExchangeManager (DynamicCast<QosFrameExchangeManager> (m_feManager));
        }
    }
}

Ptr<FrameExchangeManager>
WifiMac::GetFrameExchangeManager (void) const
{
  return m_feManager;
}

void
WifiMac::SetWifiRemoteStationManager (const Ptr<WifiRemoteStationManager> stationManager)
{
  NS_LOG_FUNCTION (this << stationManager);
  m_stationManager = stationManager;
}

Ptr<WifiRemoteStationManager>
WifiMac::GetWifiRemoteStationManager () const
{
  return m_stationManager;
}

void
WifiMac::SetWifiPhy (const Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  m_phy = phy;
  NS_ABORT_MSG_IF (!m_phy->GetOperatingChannel ().IsSet (),
                   "PHY operating channel must have been set");
  ConfigurePhyDependentParameters ();
  m_channelAccessManager->SetupPhyListener (phy);
  NS_ASSERT (m_feManager != 0);
  m_feManager->SetWifiPhy (phy);
}

Ptr<WifiPhy>
WifiMac::GetWifiPhy (void) const
{
  NS_LOG_FUNCTION (this);
  return m_phy;
}

void
WifiMac::ResetWifiPhy (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_feManager != 0);
  m_feManager->ResetPhy ();
  m_channelAccessManager->RemovePhyListener (m_phy);
  m_phy = 0;
}

void
WifiMac::SetQosSupported (bool enable)
{
  NS_LOG_FUNCTION (this << enable);
  NS_ABORT_IF (IsInitialized ());
  m_qosSupported = enable;

  if (!m_qosSupported)
    {
      // create a non-QoS TXOP
      m_txop = CreateObject<Txop> ();
      m_txop->SetChannelAccessManager (m_channelAccessManager);
      m_txop->SetWifiMac (this);
      m_txop->SetTxMiddle (m_txMiddle);
      m_txop->SetDroppedMpduCallback (MakeCallback (&DroppedMpduTracedCallback::operator(),
                                                    &m_droppedMpduCallback));
    }
  else
    {
      //Construct the EDCAFs. The ordering is important - highest
      //priority (Table 9-1 UP-to-AC mapping; IEEE 802.11-2012) must be created
      //first.
      SetupEdcaQueue (AC_VO);
      SetupEdcaQueue (AC_VI);
      SetupEdcaQueue (AC_BE);
      SetupEdcaQueue (AC_BK);
    }
}

bool
WifiMac::GetQosSupported () const
{
  return m_qosSupported;
}

bool
WifiMac::GetErpSupported () const
{
  return m_erpSupported;
}

void
WifiMac::SetErpSupported (bool enable)
{
  NS_LOG_FUNCTION (this);
  if (enable)
    {
      SetDsssSupported (true);
    }
  m_erpSupported = enable;
}

void
WifiMac::SetDsssSupported (bool enable)
{
  NS_LOG_FUNCTION (this);
  m_dsssSupported = enable;
}

bool
WifiMac::GetDsssSupported () const
{
  return m_dsssSupported;
}

void
WifiMac::SetCtsToSelfSupported (bool enable)
{
  NS_LOG_FUNCTION (this);
  m_ctsToSelfSupported = enable;
}

void
WifiMac::SetShortSlotTimeSupported (bool enable)
{
  NS_LOG_FUNCTION (this << enable);
  m_shortSlotTimeSupported = enable;
}

bool
WifiMac::GetShortSlotTimeSupported (void) const
{
  return m_shortSlotTimeSupported;
}

bool
WifiMac::SupportsSendFrom (void) const
{
  return false;
}

void
WifiMac::SetForwardUpCallback (ForwardUpCallback upCallback)
{
  NS_LOG_FUNCTION (this);
  m_forwardUp = upCallback;
}

void
WifiMac::SetLinkUpCallback (Callback<void> linkUp)
{
  NS_LOG_FUNCTION (this);
  m_linkUp = linkUp;
}

void
WifiMac::SetLinkDownCallback (Callback<void> linkDown)
{
  NS_LOG_FUNCTION (this);
  m_linkDown = linkDown;
}

void
WifiMac::Enqueue (Ptr<Packet> packet, Mac48Address to, Mac48Address from)
{
  //We expect WifiMac subclasses which do support forwarding (e.g.,
  //AP) to override this method. Therefore, we throw a fatal error if
  //someone tries to invoke this method on a class which has not done
  //this.
  NS_FATAL_ERROR ("This MAC entity (" << this << ", " << GetAddress ()
                                      << ") does not support Enqueue() with from address");
}

void
WifiMac::ForwardUp (Ptr<const Packet> packet, Mac48Address from, Mac48Address to)
{
  NS_LOG_FUNCTION (this << packet << from << to);
  m_forwardUp (packet, from, to);
}

void
WifiMac::Receive (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);

  const WifiMacHeader* hdr = &mpdu->GetHeader ();
  Ptr<Packet> packet = mpdu->GetPacket ()->Copy ();
  Mac48Address to = hdr->GetAddr1 ();
  Mac48Address from = hdr->GetAddr2 ();

  //We don't know how to deal with any frame that is not addressed to
  //us (and odds are there is nothing sensible we could do anyway),
  //so we ignore such frames.
  //
  //The derived class may also do some such filtering, but it doesn't
  //hurt to have it here too as a backstop.
  if (to != GetAddress ())
    {
      return;
    }

  if (hdr->IsMgt () && hdr->IsAction ())
    {
      //There is currently only any reason for Management Action
      //frames to be flying about if we are a QoS STA.
      NS_ASSERT (GetQosSupported ());

      WifiActionHeader actionHdr;
      packet->RemoveHeader (actionHdr);

      switch (actionHdr.GetCategory ())
        {
        case WifiActionHeader::BLOCK_ACK:

          switch (actionHdr.GetAction ().blockAck)
            {
            case WifiActionHeader::BLOCK_ACK_ADDBA_REQUEST:
              {
                MgtAddBaRequestHeader reqHdr;
                packet->RemoveHeader (reqHdr);

                //We've received an ADDBA Request. Our policy here is
                //to automatically accept it, so we get the ADDBA
                //Response on it's way immediately.
                NS_ASSERT (m_feManager != 0);
                Ptr<HtFrameExchangeManager> htFem = DynamicCast<HtFrameExchangeManager> (m_feManager);
                if (htFem != 0)
                  {
                    htFem->SendAddBaResponse (&reqHdr, from);
                  }
                //This frame is now completely dealt with, so we're done.
                return;
              }
            case WifiActionHeader::BLOCK_ACK_ADDBA_RESPONSE:
              {
                MgtAddBaResponseHeader respHdr;
                packet->RemoveHeader (respHdr);

                //We've received an ADDBA Response. We assume that it
                //indicates success after an ADDBA Request we have
                //sent (we could, in principle, check this, but it
                //seems a waste given the level of the current model)
                //and act by locally establishing the agreement on
                //the appropriate queue.
                GetQosTxop (respHdr.GetTid ())->GotAddBaResponse (&respHdr, from);
                auto htFem = DynamicCast<HtFrameExchangeManager> (m_feManager);
                if (htFem != 0)
                  {
                    GetQosTxop (respHdr.GetTid ())->GetBaManager ()->SetBlockAckInactivityCallback (MakeCallback (&HtFrameExchangeManager::SendDelbaFrame, htFem));
                  }
                //This frame is now completely dealt with, so we're done.
                return;
              }
            case WifiActionHeader::BLOCK_ACK_DELBA:
              {
                MgtDelBaHeader delBaHdr;
                packet->RemoveHeader (delBaHdr);

                if (delBaHdr.IsByOriginator ())
                  {
                    //This DELBA frame was sent by the originator, so
                    //this means that an ingoing established
                    //agreement exists in HtFrameExchangeManager and we need to
                    //destroy it.
                    NS_ASSERT (m_feManager != 0);
                    Ptr<HtFrameExchangeManager> htFem = DynamicCast<HtFrameExchangeManager> (m_feManager);
                    if (htFem != 0)
                      {
                        htFem->DestroyBlockAckAgreement (from, delBaHdr.GetTid ());
                      }
                  }
                else
                  {
                    //We must have been the originator. We need to
                    //tell the correct queue that the agreement has
                    //been torn down
                    GetQosTxop (delBaHdr.GetTid ())->GotDelBaFrame (&delBaHdr, from);
                  }
                //This frame is now completely dealt with, so we're done.
                return;
              }
            default:
              NS_FATAL_ERROR ("Unsupported Action field in Block Ack Action frame");
              return;
            }
        default:
          NS_FATAL_ERROR ("Unsupported Action frame received");
          return;
        }
    }
  NS_FATAL_ERROR ("Don't know how to handle frame (type=" << hdr->GetType ());
}

void
WifiMac::DeaggregateAmsduAndForward (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  for (auto& msduPair : *PeekPointer (mpdu))
    {
      ForwardUp (msduPair.first, msduPair.second.GetSourceAddr (),
                 msduPair.second.GetDestinationAddr ());
    }
}

Ptr<HtConfiguration>
WifiMac::GetHtConfiguration (void) const
{
  return GetDevice ()->GetHtConfiguration ();
}

Ptr<VhtConfiguration>
WifiMac::GetVhtConfiguration (void) const
{
  return GetDevice ()->GetVhtConfiguration ();
}

Ptr<HeConfiguration>
WifiMac::GetHeConfiguration (void) const
{
  return GetDevice ()->GetHeConfiguration ();
}

bool
WifiMac::GetHtSupported () const
{
  if (GetHtConfiguration ())
    {
      return true;
    }
  return false;
}

bool
WifiMac::GetVhtSupported () const
{
  if (GetVhtConfiguration ())
    {
      return true;
    }
  return false;
}

bool
WifiMac::GetHeSupported () const
{
  if (GetHeConfiguration ())
    {
      return true;
    }
  return false;
}

void
WifiMac::SetVoBlockAckThreshold (uint8_t threshold)
{
  NS_LOG_FUNCTION (this << +threshold);
  if (m_qosSupported)
    {
      GetVOQueue ()->SetBlockAckThreshold (threshold);
    }
}

void
WifiMac::SetViBlockAckThreshold (uint8_t threshold)
{
  NS_LOG_FUNCTION (this << +threshold);
  if (m_qosSupported)
    {
      GetVIQueue ()->SetBlockAckThreshold (threshold);
    }
}

void
WifiMac::SetBeBlockAckThreshold (uint8_t threshold)
{
  NS_LOG_FUNCTION (this << +threshold);
  if (m_qosSupported)
    {
      GetBEQueue ()->SetBlockAckThreshold (threshold);
    }
}

void
WifiMac::SetBkBlockAckThreshold (uint8_t threshold)
{
  NS_LOG_FUNCTION (this << +threshold);
  if (m_qosSupported)
    {
      GetBKQueue ()->SetBlockAckThreshold (threshold);
    }
}

void
WifiMac::SetVoBlockAckInactivityTimeout (uint16_t timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  if (m_qosSupported)
    {
      GetVOQueue ()->SetBlockAckInactivityTimeout (timeout);
    }
}

void
WifiMac::SetViBlockAckInactivityTimeout (uint16_t timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  if (m_qosSupported)
    {
      GetVIQueue ()->SetBlockAckInactivityTimeout (timeout);
    }
}

void
WifiMac::SetBeBlockAckInactivityTimeout (uint16_t timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  if (m_qosSupported)
    {
      GetBEQueue ()->SetBlockAckInactivityTimeout (timeout);
    }
}

void
WifiMac::SetBkBlockAckInactivityTimeout (uint16_t timeout)
{
  NS_LOG_FUNCTION (this << timeout);
  if (m_qosSupported)
    {
      GetBKQueue ()->SetBlockAckInactivityTimeout (timeout);
    }
}

ExtendedCapabilities
WifiMac::GetExtendedCapabilities (void) const
{
  NS_LOG_FUNCTION (this);
  ExtendedCapabilities capabilities;
  capabilities.SetHtSupported (GetHtSupported ());
  capabilities.SetVhtSupported (GetVhtSupported ());
  //TODO: to be completed
  return capabilities;
}

HtCapabilities
WifiMac::GetHtCapabilities (void) const
{
  NS_LOG_FUNCTION (this);
  HtCapabilities capabilities;
  if (GetHtSupported ())
    {
      Ptr<HtConfiguration> htConfiguration = GetHtConfiguration ();
      bool sgiSupported = htConfiguration->GetShortGuardIntervalSupported ();
      capabilities.SetHtSupported (1);
      capabilities.SetLdpc (htConfiguration->GetLdpcSupported ());
      capabilities.SetSupportedChannelWidth (GetWifiPhy ()->GetChannelWidth () >= 40);
      capabilities.SetShortGuardInterval20 (sgiSupported);
      capabilities.SetShortGuardInterval40 (GetWifiPhy ()->GetChannelWidth () >= 40 && sgiSupported);
      // Set Maximum A-MSDU Length subfield
      uint16_t maxAmsduSize = std::max ({m_voMaxAmsduSize, m_viMaxAmsduSize,
                                         m_beMaxAmsduSize, m_bkMaxAmsduSize});
      if (maxAmsduSize <= 3839)
        {
          capabilities.SetMaxAmsduLength (3839);
        }
      else
        {
          capabilities.SetMaxAmsduLength (7935);
        }
      uint32_t maxAmpduLength = std::max ({m_voMaxAmpduSize, m_viMaxAmpduSize,
                                           m_beMaxAmpduSize, m_bkMaxAmpduSize});
      // round to the next power of two minus one
      maxAmpduLength = (1ul << static_cast<uint32_t> (std::ceil (std::log2 (maxAmpduLength + 1)))) - 1;
      // The maximum A-MPDU length in HT capabilities elements ranges from 2^13-1 to 2^16-1
      capabilities.SetMaxAmpduLength (std::min (std::max (maxAmpduLength, 8191u), 65535u));

      capabilities.SetLSigProtectionSupport (true);
      uint64_t maxSupportedRate = 0; //in bit/s
      for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_HT))
        {
          capabilities.SetRxMcsBitmask (mcs.GetMcsValue ());
          uint8_t nss = (mcs.GetMcsValue () / 8) + 1;
          NS_ASSERT (nss > 0 && nss < 5);
          uint64_t dataRate = mcs.GetDataRate (GetWifiPhy ()->GetChannelWidth (), sgiSupported ? 400 : 800, nss);
          if (dataRate > maxSupportedRate)
            {
              maxSupportedRate = dataRate;
              NS_LOG_DEBUG ("Updating maxSupportedRate to " << maxSupportedRate);
            }
        }
      capabilities.SetRxHighestSupportedDataRate (static_cast<uint16_t> (maxSupportedRate / 1e6)); //in Mbit/s
      capabilities.SetTxMcsSetDefined (GetWifiPhy ()->GetNMcs () > 0);
      capabilities.SetTxMaxNSpatialStreams (GetWifiPhy ()->GetMaxSupportedTxSpatialStreams ());
      //we do not support unequal modulations
      capabilities.SetTxRxMcsSetUnequal (0);
      capabilities.SetTxUnequalModulation (0);
    }
  return capabilities;
}

VhtCapabilities
WifiMac::GetVhtCapabilities (void) const
{
  NS_LOG_FUNCTION (this);
  VhtCapabilities capabilities;
  if (GetVhtSupported ())
    {
      Ptr<HtConfiguration> htConfiguration = GetHtConfiguration ();
      Ptr<VhtConfiguration> vhtConfiguration = GetVhtConfiguration ();
      bool sgiSupported = htConfiguration->GetShortGuardIntervalSupported ();
      capabilities.SetVhtSupported (1);
      if (GetWifiPhy ()->GetChannelWidth () == 160)
        {
          capabilities.SetSupportedChannelWidthSet (1);
        }
      else
        {
          capabilities.SetSupportedChannelWidthSet (0);
        }
      // Set Maximum MPDU Length subfield
      uint16_t maxAmsduSize = std::max ({m_voMaxAmsduSize, m_viMaxAmsduSize,
                                         m_beMaxAmsduSize, m_bkMaxAmsduSize});
      if (maxAmsduSize <= 3839)
        {
          capabilities.SetMaxMpduLength (3895);
        }
      else if (maxAmsduSize <= 7935)
        {
          capabilities.SetMaxMpduLength (7991);
        }
      else
        {
          capabilities.SetMaxMpduLength (11454);
        }
      uint32_t maxAmpduLength = std::max ({m_voMaxAmpduSize, m_viMaxAmpduSize,
                                           m_beMaxAmpduSize, m_bkMaxAmpduSize});
      // round to the next power of two minus one
      maxAmpduLength = (1ul << static_cast<uint32_t> (std::ceil (std::log2 (maxAmpduLength + 1)))) - 1;
      // The maximum A-MPDU length in VHT capabilities elements ranges from 2^13-1 to 2^20-1
      capabilities.SetMaxAmpduLength (std::min (std::max (maxAmpduLength, 8191u), 1048575u));

      capabilities.SetRxLdpc (htConfiguration->GetLdpcSupported ());
      capabilities.SetShortGuardIntervalFor80Mhz ((GetWifiPhy ()->GetChannelWidth () == 80) && sgiSupported);
      capabilities.SetShortGuardIntervalFor160Mhz ((GetWifiPhy ()->GetChannelWidth () == 160) && sgiSupported);
      uint8_t maxMcs = 0;
      for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_VHT))
        {
          if (mcs.GetMcsValue () > maxMcs)
            {
              maxMcs = mcs.GetMcsValue ();
            }
        }
      // Support same MaxMCS for each spatial stream
      for (uint8_t nss = 1; nss <= GetWifiPhy ()->GetMaxSupportedRxSpatialStreams (); nss++)
        {
          capabilities.SetRxMcsMap (maxMcs, nss);
        }
      for (uint8_t nss = 1; nss <= GetWifiPhy ()->GetMaxSupportedTxSpatialStreams (); nss++)
        {
          capabilities.SetTxMcsMap (maxMcs, nss);
        }
      uint64_t maxSupportedRateLGI = 0; //in bit/s
      for (const auto & mcs : GetWifiPhy ()->GetMcsList (WIFI_MOD_CLASS_VHT))
        {
          if (!mcs.IsAllowed (GetWifiPhy ()->GetChannelWidth (), 1))
            {
              continue;
            }
          if (mcs.GetDataRate (GetWifiPhy ()->GetChannelWidth ()) > maxSupportedRateLGI)
            {
              maxSupportedRateLGI = mcs.GetDataRate (GetWifiPhy ()->GetChannelWidth ());
              NS_LOG_DEBUG ("Updating maxSupportedRateLGI to " << maxSupportedRateLGI);
            }
        }
      capabilities.SetRxHighestSupportedLgiDataRate (static_cast<uint16_t> (maxSupportedRateLGI / 1e6)); //in Mbit/s
      capabilities.SetTxHighestSupportedLgiDataRate (static_cast<uint16_t> (maxSupportedRateLGI / 1e6)); //in Mbit/s
      //To be filled in once supported
      capabilities.SetRxStbc (0);
      capabilities.SetTxStbc (0);
    }
  return capabilities;
}

HeCapabilities
WifiMac::GetHeCapabilities (void) const
{
  NS_LOG_FUNCTION (this);
  HeCapabilities capabilities;
  if (GetHeSupported ())
    {
      Ptr<HtConfiguration> htConfiguration = GetHtConfiguration ();
      Ptr<HeConfiguration> heConfiguration = GetHeConfiguration ();
      capabilities.SetHeSupported (1);
      uint8_t channelWidthSet = 0;
      if ((m_phy->GetChannelWidth () >= 40) && (m_phy->GetPhyBand () == WIFI_PHY_BAND_2_4GHZ))
        {
          channelWidthSet |= 0x01;
        }
      if ((m_phy->GetChannelWidth () >= 80) && ((m_phy->GetPhyBand () == WIFI_PHY_BAND_5GHZ) || (m_phy->GetPhyBand () == WIFI_PHY_BAND_6GHZ)))
        {
          channelWidthSet |= 0x02;
        }
      if ((m_phy->GetChannelWidth () >= 160) && ((m_phy->GetPhyBand () == WIFI_PHY_BAND_5GHZ) || (m_phy->GetPhyBand () == WIFI_PHY_BAND_6GHZ)))
        {
          channelWidthSet |= 0x04;
        }
      capabilities.SetChannelWidthSet (channelWidthSet);
      capabilities.SetLdpcCodingInPayload (htConfiguration->GetLdpcSupported ());
      if (heConfiguration->GetGuardInterval () == NanoSeconds (800))
        {
          //todo: We assume for now that if we support 800ns GI then 1600ns GI is supported as well
          //todo: Assuming reception support for both 1x HE LTF and 4x HE LTF 800 ns
          capabilities.SetHeSuPpdu1xHeLtf800nsGi (true);
          capabilities.SetHePpdu4xHeLtf800nsGi (true);
        }

      uint32_t maxAmpduLength = std::max ({m_voMaxAmpduSize, m_viMaxAmpduSize,
                                           m_beMaxAmpduSize, m_bkMaxAmpduSize});
      // round to the next power of two minus one
      maxAmpduLength = (1ul << static_cast<uint32_t> (std::ceil (std::log2 (maxAmpduLength + 1)))) - 1;
      // The maximum A-MPDU length in HE capabilities elements ranges from 2^20-1 to 2^23-1
      capabilities.SetMaxAmpduLength (std::min (std::max (maxAmpduLength, 1048575u), 8388607u));

      uint8_t maxMcs = 0;
      for (const auto & mcs : m_phy->GetMcsList (WIFI_MOD_CLASS_HE))
        {
          if (mcs.GetMcsValue () > maxMcs)
            {
              maxMcs = mcs.GetMcsValue ();
            }
        }
      capabilities.SetHighestMcsSupported (maxMcs);
      capabilities.SetHighestNssSupported (m_phy->GetMaxSupportedTxSpatialStreams ());
    }
  return capabilities;
}

uint32_t
WifiMac::GetMaxAmpduSize (AcIndex ac) const
{
  uint32_t maxSize = 0;
  switch (ac)
    {
      case AC_BE:
        maxSize = m_beMaxAmpduSize;
        break;
      case AC_BK:
        maxSize = m_bkMaxAmpduSize;
        break;
      case AC_VI:
        maxSize = m_viMaxAmpduSize;
        break;
      case AC_VO:
        maxSize = m_voMaxAmpduSize;
        break;
      default:
        NS_ABORT_MSG ("Unknown AC " << ac);
        return 0;
    }
  return maxSize;
}

uint16_t
WifiMac::GetMaxAmsduSize (AcIndex ac) const
{
  uint16_t maxSize = 0;
  switch (ac)
    {
      case AC_BE:
        maxSize = m_beMaxAmsduSize;
        break;
      case AC_BK:
        maxSize = m_bkMaxAmsduSize;
        break;
      case AC_VI:
        maxSize = m_viMaxAmsduSize;
        break;
      case AC_VO:
        maxSize = m_voMaxAmsduSize;
        break;
      default:
        NS_ABORT_MSG ("Unknown AC " << ac);
        return 0;
    }
  return maxSize;
}

} //namespace ns3

