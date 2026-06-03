/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
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
 * Author: Giuseppe Piro  <g.piro@poliba.it>
 * Author: Marco Miozzo <mmiozzo@cttc.es> : Update to FF API Architecture
 * Author: Nicola Baldo <nbaldo@cttc.es>  : Integrated with new RRC and MAC architecture
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it> : Integrated with new architecture - GSoC 2015 - Carrier Aggregation
 */

#include <ns3/llc-snap-header.h>
#include <ns3/simulator.h>
#include <ns3/callback.h>
#include <ns3/node.h>
#include <ns3/packet.h>
#include <ns3/lte-net-device.h>
#include <ns3/packet-burst.h>
#include <ns3/uinteger.h>
#include <ns3/trace-source-accessor.h>
#include <ns3/pointer.h>
#include <ns3/enum.h>
#include <ns3/lte-amc.h>
#include <ns3/lte-enb-mac.h>
#include <ns3/lte-enb-net-device.h>
#include <ns3/lte-enb-rrc.h>
#include <ns3/lte-ue-net-device.h>
#include <ns3/lte-enb-phy.h>
#include <ns3/ff-mac-scheduler.h>
#include <ns3/lte-handover-algorithm.h>
#include <ns3/lte-anr.h>
#include <ns3/lte-ffr-algorithm.h>
#include <ns3/ipv4-l3-protocol.h>
#include <ns3/ipv6-l3-protocol.h>
#include <ns3/abort.h>
#include <ns3/log.h>
#include <ns3/lte-enb-component-carrier-manager.h>
#include <ns3/object-map.h>
#include <ns3/object-factory.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteEnbNetDevice");

NS_OBJECT_ENSURE_REGISTERED ( LteEnbNetDevice);

TypeId LteEnbNetDevice::GetTypeId (void)
{
  static TypeId
    tid =
    TypeId ("ns3::LteEnbNetDevice")
    .SetParent<LteNetDevice> ()
    .AddConstructor<LteEnbNetDevice> ()
    .AddAttribute ("LteEnbRrc",
                   "The RRC associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&LteEnbNetDevice::m_rrc),
                   MakePointerChecker <LteEnbRrc> ())
    .AddAttribute ("LteHandoverAlgorithm",
                   "The handover algorithm associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&LteEnbNetDevice::m_handoverAlgorithm),
                   MakePointerChecker <LteHandoverAlgorithm> ())
    .AddAttribute ("LteAnr",
                   "The automatic neighbour relation function associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&LteEnbNetDevice::m_anr),
                   MakePointerChecker <LteAnr> ())
    .AddAttribute ("LteFfrAlgorithm",
                   "The FFR algorithm associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&LteEnbNetDevice::m_ffrAlgorithm),
                   MakePointerChecker <LteFfrAlgorithm> ())
    .AddAttribute ("LteEnbComponentCarrierManager",
                   "The RRC associated to this EnbNetDevice",
                   PointerValue (),
                   MakePointerAccessor (&LteEnbNetDevice::m_componentCarrierManager),
                   MakePointerChecker <LteEnbComponentCarrierManager> ())
    .AddAttribute ("ComponentCarrierMap", "List of component carriers.",
                   ObjectMapValue (),
                   MakeObjectMapAccessor (&LteEnbNetDevice::m_ccMap),
                   MakeObjectMapChecker<ComponentCarrierEnb> ())
    .AddAttribute ("UlBandwidth",
                   "Uplink Transmission Bandwidth Configuration in number of Resource Blocks",
                   UintegerValue (25),
                   MakeUintegerAccessor (&LteEnbNetDevice::SetUlBandwidth, 
                                         &LteEnbNetDevice::GetUlBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlBandwidth",
                   "Downlink Transmission Bandwidth Configuration in number of Resource Blocks",
                   UintegerValue (25),
                   MakeUintegerAccessor (&LteEnbNetDevice::SetDlBandwidth, 
                                         &LteEnbNetDevice::GetDlBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CellId",
                   "Cell Identifier",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteEnbNetDevice::m_cellId),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("DlEarfcn",
                   "Downlink E-UTRA Absolute Radio Frequency Channel Number (EARFCN) "
                   "as per 3GPP 36.101 Section 5.7.3. ",
                   UintegerValue (100),
                   MakeUintegerAccessor (&LteEnbNetDevice::m_dlEarfcn),
                   MakeUintegerChecker<uint32_t> (0, 262143))
    .AddAttribute ("UlEarfcn",
                   "Uplink E-UTRA Absolute Radio Frequency Channel Number (EARFCN) "
                   "as per 3GPP 36.101 Section 5.7.3. ",
                   UintegerValue (18100),
                   MakeUintegerAccessor (&LteEnbNetDevice::m_ulEarfcn),
                   MakeUintegerChecker<uint32_t> (0, 262143))
    .AddAttribute ("CsgId",
                   "The Closed Subscriber Group (CSG) identity that this eNodeB belongs to",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteEnbNetDevice::SetCsgId,
                                         &LteEnbNetDevice::GetCsgId),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("CsgIndication",
                   "If true, only UEs which are members of the CSG (i.e. same CSG ID) "
                   "can gain access to the eNodeB, therefore enforcing closed access mode. "
                   "Otherwise, the eNodeB operates as a non-CSG cell and implements open access mode.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&LteEnbNetDevice::SetCsgIndication,
                                        &LteEnbNetDevice::GetCsgIndication),
                   MakeBooleanChecker ())
  ;
  return tid;
}

LteEnbNetDevice::LteEnbNetDevice ()
  : m_isConstructed (false),
    m_isConfigured (false),
    m_anr (0),
    m_componentCarrierManager(0)
{
  NS_LOG_FUNCTION (this);
}

LteEnbNetDevice::~LteEnbNetDevice (void)
{
  NS_LOG_FUNCTION (this);
}

void
LteEnbNetDevice::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  m_rrc->Dispose ();
  m_rrc = 0;

  m_handoverAlgorithm->Dispose ();
  m_handoverAlgorithm = 0;

  if (m_anr != 0)
    {
      m_anr->Dispose ();
      m_anr = 0;
    }
  m_componentCarrierManager->Dispose();
  m_componentCarrierManager = 0;
  // ComponentCarrierEnb::DoDispose() will call DoDispose
  // of its PHY, MAC, FFR and scheduler instance
  for (uint32_t i = 0; i < m_ccMap.size (); i++)
    {
      m_ccMap.at (i)->Dispose ();
      m_ccMap.at (i) = 0;
    }
   
  LteNetDevice::DoDispose ();
}



Ptr<LteEnbMac>
LteEnbNetDevice::GetMac () const
{
  return GetMac (0);
}

Ptr<LteEnbPhy>
LteEnbNetDevice::GetPhy () const
{
  return GetPhy (0);
}

Ptr<LteEnbMac>
LteEnbNetDevice::GetMac (uint8_t index) const
{
  return DynamicCast<ComponentCarrierEnb> (m_ccMap.at (index))->GetMac ();
}

Ptr<LteEnbPhy>
LteEnbNetDevice::GetPhy(uint8_t index) const
{
  return DynamicCast<ComponentCarrierEnb> (m_ccMap.at (index))->GetPhy ();
}

Ptr<LteEnbRrc>
LteEnbNetDevice::GetRrc () const
{
  return m_rrc;
}

Ptr<LteEnbComponentCarrierManager>
LteEnbNetDevice::GetComponentCarrierManager () const
{
  return  m_componentCarrierManager;
}

uint16_t
LteEnbNetDevice::GetCellId () const
{
  return m_cellId;
}

std::vector<uint16_t>
LteEnbNetDevice::GetCellIds () const
{
  std::vector<uint16_t> cellIds;

  for (auto &it: m_ccMap)
    {
      cellIds.push_back (it.second->GetCellId ());
    }
  return cellIds;
}

bool
LteEnbNetDevice::HasCellId (uint16_t cellId) const
{
  return m_rrc->HasCellId (cellId);
}

uint16_t
LteEnbNetDevice::GetUlBandwidth () const
{
  return m_ulBandwidth;
}

void 
LteEnbNetDevice::SetUlBandwidth (uint16_t bw)
{ 
  NS_LOG_FUNCTION (this << bw);
  switch (bw)
    { 
    case 6:
    case 15:
    case 25:
    case 50:
    case 75:
    case 100:
      m_ulBandwidth = bw;
      break;

    default:
      NS_FATAL_ERROR ("invalid bandwidth value " << bw);
      break;
    }
}

uint16_t
LteEnbNetDevice::GetDlBandwidth () const
{
  return m_dlBandwidth;
}

void 
LteEnbNetDevice::SetDlBandwidth (uint16_t bw)
{
  NS_LOG_FUNCTION (this << uint16_t (bw));
  switch (bw)
    { 
    case 6:
    case 15:
    case 25:
    case 50:
    case 75:
    case 100:
      m_dlBandwidth = bw;
      break;

    default:
      NS_FATAL_ERROR ("invalid bandwidth value " << bw);
      break;
    }
}

uint32_t 
LteEnbNetDevice::GetDlEarfcn () const
{
  return m_dlEarfcn;
}

void 
LteEnbNetDevice::SetDlEarfcn (uint32_t earfcn)
{ 
  NS_LOG_FUNCTION (this << earfcn);
  m_dlEarfcn = earfcn;
}

uint32_t 
LteEnbNetDevice::GetUlEarfcn () const
{
  return m_ulEarfcn;
}

void 
LteEnbNetDevice::SetUlEarfcn (uint32_t earfcn)
{ 
  NS_LOG_FUNCTION (this << earfcn);
  m_ulEarfcn = earfcn;
}

uint32_t
LteEnbNetDevice::GetCsgId () const
{
  return m_csgId;
}

void
LteEnbNetDevice::SetCsgId (uint32_t csgId)
{
  NS_LOG_FUNCTION (this << csgId);
  m_csgId = csgId;
  UpdateConfig (); // propagate the change to RRC level
}

bool
LteEnbNetDevice::GetCsgIndication () const
{
  return m_csgIndication;
}

void
LteEnbNetDevice::SetCsgIndication (bool csgIndication)
{
  NS_LOG_FUNCTION (this << csgIndication);
  m_csgIndication = csgIndication;
  UpdateConfig (); // propagate the change to RRC level
}

std::map < uint8_t, Ptr<ComponentCarrierBaseStation> >
LteEnbNetDevice::GetCcMap () const
{
  return m_ccMap;
}

void
LteEnbNetDevice::SetCcMap (std::map< uint8_t, Ptr<ComponentCarrierBaseStation> > ccm)
{
  NS_ASSERT_MSG (!m_isConfigured, "attempt to set CC map after configuration");
  m_ccMap = ccm;
}

void 
LteEnbNetDevice::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  m_isConstructed = true;
  UpdateConfig ();
  for (auto it = m_ccMap.begin (); it != m_ccMap.end (); ++it)
    {
       it->second->Initialize ();
    }
  m_rrc->Initialize ();
  m_componentCarrierManager->Initialize();
  m_handoverAlgorithm->Initialize ();

  if (m_anr != 0)
    {
      m_anr->Initialize ();
    }

  m_ffrAlgorithm->Initialize ();
}


bool
LteEnbNetDevice::Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << packet << dest << protocolNumber);
  NS_ABORT_MSG_IF (protocolNumber != Ipv4L3Protocol::PROT_NUMBER
                   && protocolNumber != Ipv6L3Protocol::PROT_NUMBER,
                   "unsupported protocol " << protocolNumber << ", only IPv4 and IPv6 are supported");
  return m_rrc->SendData (packet);
}


void
LteEnbNetDevice::UpdateConfig (void)
{
  NS_LOG_FUNCTION (this);

  if (m_isConstructed)
    {
      if (!m_isConfigured)
        {
          NS_LOG_LOGIC (this << " Configure cell " << m_cellId);
          // we have to make sure that this function is called only once
          NS_ASSERT (!m_ccMap.empty ());
          m_rrc->ConfigureCell (m_ccMap);
          m_isConfigured = true;
        }

      NS_LOG_LOGIC (this << " Updating SIB1 of cell " << m_cellId
                         << " with CSG ID " << m_csgId
                         << " and CSG indication " << m_csgIndication);
      m_rrc->SetCsgId (m_csgId, m_csgIndication);
    }
  else
    {
      /*
       * Lower layers are not ready yet, so do nothing now and expect
       * ``DoInitialize`` to re-invoke this function.
       */
    }
}


} // namespace ns3
