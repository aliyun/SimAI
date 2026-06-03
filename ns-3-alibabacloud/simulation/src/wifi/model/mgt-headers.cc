/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#include "ns3/simulator.h"
#include "ns3/address-utils.h"
#include "mgt-headers.h"

namespace ns3 {

/***********************************************************
 *          Probe Request
 ***********************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtProbeRequestHeader);

MgtProbeRequestHeader::~MgtProbeRequestHeader ()
{
}

void
MgtProbeRequestHeader::SetSsid (Ssid ssid)
{
  m_ssid = ssid;
}

Ssid
MgtProbeRequestHeader::GetSsid (void) const
{
  return m_ssid;
}

void
MgtProbeRequestHeader::SetSupportedRates (SupportedRates rates)
{
  m_rates = rates;
}

void
MgtProbeRequestHeader::SetExtendedCapabilities (ExtendedCapabilities extendedCapabilities)
{
  m_extendedCapability = extendedCapabilities;
}

ExtendedCapabilities
MgtProbeRequestHeader::GetExtendedCapabilities (void) const
{
  return m_extendedCapability;
}

void
MgtProbeRequestHeader::SetHtCapabilities (HtCapabilities htCapabilities)
{
  m_htCapability = htCapabilities;
}

HtCapabilities
MgtProbeRequestHeader::GetHtCapabilities (void) const
{
  return m_htCapability;
}

void
MgtProbeRequestHeader::SetVhtCapabilities (VhtCapabilities vhtCapabilities)
{
  m_vhtCapability = vhtCapabilities;
}

VhtCapabilities
MgtProbeRequestHeader::GetVhtCapabilities (void) const
{
  return m_vhtCapability;
}

void
MgtProbeRequestHeader::SetHeCapabilities (HeCapabilities heCapabilities)
{
  m_heCapability = heCapabilities;
}

HeCapabilities
MgtProbeRequestHeader::GetHeCapabilities (void) const
{
  return m_heCapability;
}

SupportedRates
MgtProbeRequestHeader::GetSupportedRates (void) const
{
  return m_rates;
}

uint32_t
MgtProbeRequestHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += m_ssid.GetSerializedSize ();
  size += m_rates.GetSerializedSize ();
  size += m_rates.extended.GetSerializedSize ();
  size += m_extendedCapability.GetSerializedSize ();
  size += m_htCapability.GetSerializedSize ();
  size += m_vhtCapability.GetSerializedSize ();
  size += m_heCapability.GetSerializedSize ();
  return size;
}

TypeId
MgtProbeRequestHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtProbeRequestHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtProbeRequestHeader> ()
  ;
  return tid;
}

TypeId
MgtProbeRequestHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
MgtProbeRequestHeader::Print (std::ostream &os) const
{
  os << "ssid=" << m_ssid << ", "
     << "rates=" << m_rates << ", "
     << "Extended Capabilities=" << m_extendedCapability << " , "
     << "HT Capabilities=" << m_htCapability << " , "
     << "VHT Capabilities=" << m_vhtCapability << " , "
     << "HE Capabilities=" << m_heCapability;
}

void
MgtProbeRequestHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i = m_ssid.Serialize (i);
  i = m_rates.Serialize (i);
  i = m_rates.extended.Serialize (i);
  i = m_extendedCapability.Serialize (i);
  i = m_htCapability.Serialize (i);
  i = m_vhtCapability.Serialize (i);
  i = m_heCapability.Serialize (i);
}

uint32_t
MgtProbeRequestHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  i = m_ssid.Deserialize (i);
  i = m_rates.Deserialize (i);
  i = m_rates.extended.DeserializeIfPresent (i);
  i = m_extendedCapability.DeserializeIfPresent (i);
  i = m_htCapability.DeserializeIfPresent (i);
  i = m_vhtCapability.DeserializeIfPresent (i);
  i = m_heCapability.DeserializeIfPresent (i);
  return i.GetDistanceFrom (start);
}


/***********************************************************
 *          Probe Response
 ***********************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtProbeResponseHeader);

MgtProbeResponseHeader::MgtProbeResponseHeader ()
{
}

MgtProbeResponseHeader::~MgtProbeResponseHeader ()
{
}

uint64_t
MgtProbeResponseHeader::GetTimestamp ()
{
  return m_timestamp;
}

Ssid
MgtProbeResponseHeader::GetSsid (void) const
{
  return m_ssid;
}

uint64_t
MgtProbeResponseHeader::GetBeaconIntervalUs (void) const
{
  return m_beaconInterval;
}

SupportedRates
MgtProbeResponseHeader::GetSupportedRates (void) const
{
  return m_rates;
}

void
MgtProbeResponseHeader::SetCapabilities (CapabilityInformation capabilities)
{
  m_capability = capabilities;
}

CapabilityInformation
MgtProbeResponseHeader::GetCapabilities (void) const
{
  return m_capability;
}

void
MgtProbeResponseHeader::SetExtendedCapabilities (ExtendedCapabilities extendedCapabilities)
{
  m_extendedCapability = extendedCapabilities;
}

ExtendedCapabilities
MgtProbeResponseHeader::GetExtendedCapabilities (void) const
{
  return m_extendedCapability;
}

void
MgtProbeResponseHeader::SetHtCapabilities (HtCapabilities htCapabilities)
{
  m_htCapability = htCapabilities;
}

HtCapabilities
MgtProbeResponseHeader::GetHtCapabilities (void) const
{
  return m_htCapability;
}

void
MgtProbeResponseHeader::SetHtOperation (HtOperation htOperation)
{
  m_htOperation = htOperation;
}

HtOperation
MgtProbeResponseHeader::GetHtOperation (void) const
{
  return m_htOperation;
}

void
MgtProbeResponseHeader::SetVhtCapabilities (VhtCapabilities vhtCapabilities)
{
  m_vhtCapability = vhtCapabilities;
}

VhtCapabilities
MgtProbeResponseHeader::GetVhtCapabilities (void) const
{
  return m_vhtCapability;
}

void
MgtProbeResponseHeader::SetVhtOperation (VhtOperation vhtOperation)
{
  m_vhtOperation = vhtOperation;
}

VhtOperation
MgtProbeResponseHeader::GetVhtOperation (void) const
{
  return m_vhtOperation;
}

void
MgtProbeResponseHeader::SetHeCapabilities (HeCapabilities heCapabilities)
{
  m_heCapability = heCapabilities;
}

HeCapabilities
MgtProbeResponseHeader::GetHeCapabilities (void) const
{
  return m_heCapability;
}

void
MgtProbeResponseHeader::SetHeOperation (HeOperation heOperation)
{
  m_heOperation = heOperation;
}

HeOperation
MgtProbeResponseHeader::GetHeOperation (void) const
{
  return m_heOperation;
}

void
MgtProbeResponseHeader::SetSsid (Ssid ssid)
{
  m_ssid = ssid;
}

void
MgtProbeResponseHeader::SetBeaconIntervalUs (uint64_t us)
{
  m_beaconInterval = us;
}

void
MgtProbeResponseHeader::SetSupportedRates (SupportedRates rates)
{
  m_rates = rates;
}

void
MgtProbeResponseHeader::SetDsssParameterSet (DsssParameterSet dsssParameterSet)
{
  m_dsssParameterSet = dsssParameterSet;
}

DsssParameterSet
MgtProbeResponseHeader::GetDsssParameterSet (void) const
{
  return m_dsssParameterSet;
}

void
MgtProbeResponseHeader::SetErpInformation (ErpInformation erpInformation)
{
  m_erpInformation = erpInformation;
}

ErpInformation
MgtProbeResponseHeader::GetErpInformation (void) const
{
  return m_erpInformation;
}

void
MgtProbeResponseHeader::SetEdcaParameterSet (EdcaParameterSet edcaParameters)
{
  m_edcaParameterSet = edcaParameters;
}

void
MgtProbeResponseHeader::SetMuEdcaParameterSet (MuEdcaParameterSet muEdcaParameters)
{
  m_muEdcaParameterSet = muEdcaParameters;
}

EdcaParameterSet
MgtProbeResponseHeader::GetEdcaParameterSet (void) const
{
  return m_edcaParameterSet;
}

MuEdcaParameterSet
MgtProbeResponseHeader::GetMuEdcaParameterSet (void) const
{
  return m_muEdcaParameterSet;
}

TypeId
MgtProbeResponseHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtProbeResponseHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtProbeResponseHeader> ()
  ;
  return tid;
}

TypeId
MgtProbeResponseHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
MgtProbeResponseHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += 8; //timestamp
  size += 2; //beacon interval
  size += m_capability.GetSerializedSize ();
  size += m_ssid.GetSerializedSize ();
  size += m_rates.GetSerializedSize ();
  size += m_dsssParameterSet.GetSerializedSize ();
  size += m_erpInformation.GetSerializedSize ();
  size += m_rates.extended.GetSerializedSize ();
  size += m_edcaParameterSet.GetSerializedSize ();
  size += m_extendedCapability.GetSerializedSize ();
  size += m_htCapability.GetSerializedSize ();
  size += m_htOperation.GetSerializedSize ();
  size += m_vhtCapability.GetSerializedSize ();
  size += m_vhtOperation.GetSerializedSize ();
  size += m_heCapability.GetSerializedSize ();
  size += m_heOperation.GetSerializedSize ();
  size += m_muEdcaParameterSet.GetSerializedSize ();
  return size;
}

void
MgtProbeResponseHeader::Print (std::ostream &os) const
{
  os << "ssid=" << m_ssid << ", "
     << "rates=" << m_rates << ", "
     << "ERP information=" << m_erpInformation << ", "
     << "Extended Capabilities=" << m_extendedCapability << " , "
     << "HT Capabilities=" << m_htCapability << " , "
     << "HT Operation=" << m_htOperation << " , "
     << "VHT Capabilities=" << m_vhtCapability << " , "
     << "VHT Operation=" << m_vhtOperation << " , "
     << "HE Capabilities=" << m_heCapability << " , "
     << "HE Operation=" << m_heOperation;
}

void
MgtProbeResponseHeader::Serialize (Buffer::Iterator start) const
{
  //timestamp
  //beacon interval
  //capability information
  //SSID
  //supported rates
  //FH parameter set
  //DS parameter set
  //CF parameter set
  //IBSS parameter set
  Buffer::Iterator i = start;
  i.WriteHtolsbU64 (Simulator::Now ().GetMicroSeconds ());
  i.WriteHtolsbU16 (static_cast<uint16_t> (m_beaconInterval / 1024));
  i = m_capability.Serialize (i);
  i = m_ssid.Serialize (i);
  i = m_rates.Serialize (i);
  i = m_dsssParameterSet.Serialize (i);
  i = m_erpInformation.Serialize (i);
  i = m_rates.extended.Serialize (i);
  i = m_edcaParameterSet.Serialize (i);
  i = m_extendedCapability.Serialize (i);
  i = m_htCapability.Serialize (i);
  i = m_htOperation.Serialize (i);
  i = m_vhtCapability.Serialize (i);
  i = m_vhtOperation.Serialize (i);
  i = m_heCapability.Serialize (i);
  i = m_heOperation.Serialize (i);
  i = m_muEdcaParameterSet.Serialize (i);
}

uint32_t
MgtProbeResponseHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_timestamp = i.ReadLsbtohU64 ();
  m_beaconInterval = i.ReadLsbtohU16 ();
  m_beaconInterval *= 1024;
  i = m_capability.Deserialize (i);
  i = m_ssid.Deserialize (i);
  i = m_rates.Deserialize (i);
  i = m_dsssParameterSet.DeserializeIfPresent (i);
  i = m_erpInformation.DeserializeIfPresent (i);
  i = m_rates.extended.DeserializeIfPresent (i);
  i = m_edcaParameterSet.DeserializeIfPresent (i);
  i = m_extendedCapability.DeserializeIfPresent (i);
  i = m_htCapability.DeserializeIfPresent (i);
  i = m_htOperation.DeserializeIfPresent (i);
  i = m_vhtCapability.DeserializeIfPresent (i);
  i = m_vhtOperation.DeserializeIfPresent (i);
  i = m_heCapability.DeserializeIfPresent (i);
  i = m_heOperation.DeserializeIfPresent (i);
  i = m_muEdcaParameterSet.DeserializeIfPresent (i);
  return i.GetDistanceFrom (start);
}


/***********************************************************
 *          Beacons
 ***********************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtBeaconHeader);

/* static */
TypeId
MgtBeaconHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtBeaconHeader")
    .SetParent<MgtProbeResponseHeader> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtBeaconHeader> ()
  ;
  return tid;
}


/***********************************************************
 *          Assoc Request
 ***********************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtAssocRequestHeader);

MgtAssocRequestHeader::MgtAssocRequestHeader ()
  : m_listenInterval (0)
{
}

MgtAssocRequestHeader::~MgtAssocRequestHeader ()
{
}

void
MgtAssocRequestHeader::SetSsid (Ssid ssid)
{
  m_ssid = ssid;
}

void
MgtAssocRequestHeader::SetSupportedRates (SupportedRates rates)
{
  m_rates = rates;
}

void
MgtAssocRequestHeader::SetListenInterval (uint16_t interval)
{
  m_listenInterval = interval;
}

void
MgtAssocRequestHeader::SetCapabilities (CapabilityInformation capabilities)
{
  m_capability = capabilities;
}

CapabilityInformation
MgtAssocRequestHeader::GetCapabilities (void) const
{
  return m_capability;
}

void
MgtAssocRequestHeader::SetExtendedCapabilities (ExtendedCapabilities extendedCapabilities)
{
  m_extendedCapability = extendedCapabilities;
}

ExtendedCapabilities
MgtAssocRequestHeader::GetExtendedCapabilities (void) const
{
  return m_extendedCapability;
}

void
MgtAssocRequestHeader::SetHtCapabilities (HtCapabilities htCapabilities)
{
  m_htCapability = htCapabilities;
}

HtCapabilities
MgtAssocRequestHeader::GetHtCapabilities (void) const
{
  return m_htCapability;
}

void
MgtAssocRequestHeader::SetVhtCapabilities (VhtCapabilities vhtCapabilities)
{
  m_vhtCapability = vhtCapabilities;
}

VhtCapabilities
MgtAssocRequestHeader::GetVhtCapabilities (void) const
{
  return m_vhtCapability;
}

void
MgtAssocRequestHeader::SetHeCapabilities (HeCapabilities heCapabilities)
{
  m_heCapability = heCapabilities;
}

HeCapabilities
MgtAssocRequestHeader::GetHeCapabilities (void) const
{
  return m_heCapability;
}

Ssid
MgtAssocRequestHeader::GetSsid (void) const
{
  return m_ssid;
}

SupportedRates
MgtAssocRequestHeader::GetSupportedRates (void) const
{
  return m_rates;
}

uint16_t
MgtAssocRequestHeader::GetListenInterval (void) const
{
  return m_listenInterval;
}

TypeId
MgtAssocRequestHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtAssocRequestHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtAssocRequestHeader> ()
  ;
  return tid;
}

TypeId
MgtAssocRequestHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
MgtAssocRequestHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += m_capability.GetSerializedSize ();
  size += 2;
  size += m_ssid.GetSerializedSize ();
  size += m_rates.GetSerializedSize ();
  size += m_rates.extended.GetSerializedSize ();
  size += m_extendedCapability.GetSerializedSize ();
  size += m_htCapability.GetSerializedSize ();
  size += m_vhtCapability.GetSerializedSize ();
  size += m_heCapability.GetSerializedSize ();
  return size;
}

void
MgtAssocRequestHeader::Print (std::ostream &os) const
{
  os << "ssid=" << m_ssid << ", "
     << "rates=" << m_rates << ", "
     << "Extended Capabilities=" << m_extendedCapability << " , "
     << "HT Capabilities=" << m_htCapability << " , "
     << "VHT Capabilities=" << m_vhtCapability << " , "
     << "HE Capabilities=" << m_heCapability;
}

void
MgtAssocRequestHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i = m_capability.Serialize (i);
  i.WriteHtolsbU16 (m_listenInterval);
  i = m_ssid.Serialize (i);
  i = m_rates.Serialize (i);
  i = m_rates.extended.Serialize (i);
  i = m_extendedCapability.Serialize (i);
  i = m_htCapability.Serialize (i);
  i = m_vhtCapability.Serialize (i);
  i = m_heCapability.Serialize (i);
}

uint32_t
MgtAssocRequestHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  i = m_capability.Deserialize (i);
  m_listenInterval = i.ReadLsbtohU16 ();
  i = m_ssid.Deserialize (i);
  i = m_rates.Deserialize (i);
  i = m_rates.extended.DeserializeIfPresent (i);
  i = m_extendedCapability.DeserializeIfPresent (i);
  i = m_htCapability.DeserializeIfPresent (i);
  i = m_vhtCapability.DeserializeIfPresent (i);
  i = m_heCapability.DeserializeIfPresent (i);
  return i.GetDistanceFrom (start);
}


/***********************************************************
 *          Ressoc Request
 ***********************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtReassocRequestHeader);

MgtReassocRequestHeader::MgtReassocRequestHeader ()
  : m_currentApAddr (Mac48Address ())
{
}

MgtReassocRequestHeader::~MgtReassocRequestHeader ()
{
}

void
MgtReassocRequestHeader::SetSsid (Ssid ssid)
{
  m_ssid = ssid;
}

void
MgtReassocRequestHeader::SetSupportedRates (SupportedRates rates)
{
  m_rates = rates;
}

void
MgtReassocRequestHeader::SetListenInterval (uint16_t interval)
{
  m_listenInterval = interval;
}

void
MgtReassocRequestHeader::SetCapabilities (CapabilityInformation capabilities)
{
  m_capability = capabilities;
}

CapabilityInformation
MgtReassocRequestHeader::GetCapabilities (void) const
{
  return m_capability;
}

void
MgtReassocRequestHeader::SetExtendedCapabilities (ExtendedCapabilities extendedCapabilities)
{
  m_extendedCapability = extendedCapabilities;
}

ExtendedCapabilities
MgtReassocRequestHeader::GetExtendedCapabilities (void) const
{
  return m_extendedCapability;
}

void
MgtReassocRequestHeader::SetHtCapabilities (HtCapabilities htCapabilities)
{
  m_htCapability = htCapabilities;
}

HtCapabilities
MgtReassocRequestHeader::GetHtCapabilities (void) const
{
  return m_htCapability;
}

void
MgtReassocRequestHeader::SetVhtCapabilities (VhtCapabilities vhtCapabilities)
{
  m_vhtCapability = vhtCapabilities;
}

VhtCapabilities
MgtReassocRequestHeader::GetVhtCapabilities (void) const
{
  return m_vhtCapability;
}

void
MgtReassocRequestHeader::SetHeCapabilities (HeCapabilities heCapabilities)
{
  m_heCapability = heCapabilities;
}

HeCapabilities
MgtReassocRequestHeader::GetHeCapabilities (void) const
{
  return m_heCapability;
}

Ssid
MgtReassocRequestHeader::GetSsid (void) const
{
  return m_ssid;
}

SupportedRates
MgtReassocRequestHeader::GetSupportedRates (void) const
{
  return m_rates;
}

uint16_t
MgtReassocRequestHeader::GetListenInterval (void) const
{
  return m_listenInterval;
}

void
MgtReassocRequestHeader::SetCurrentApAddress (Mac48Address currentApAddr)
{
  m_currentApAddr = currentApAddr;
}

TypeId
MgtReassocRequestHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtReassocRequestHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtReassocRequestHeader> ()
  ;
  return tid;
}

TypeId
MgtReassocRequestHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
MgtReassocRequestHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += m_capability.GetSerializedSize ();
  size += 2; //listen interval
  size += 6; //current AP address
  size += m_ssid.GetSerializedSize ();
  size += m_rates.GetSerializedSize ();
  size += m_rates.extended.GetSerializedSize ();
  size += m_extendedCapability.GetSerializedSize ();
  size += m_htCapability.GetSerializedSize ();
  size += m_vhtCapability.GetSerializedSize ();
  size += m_heCapability.GetSerializedSize ();
  return size;
}

void
MgtReassocRequestHeader::Print (std::ostream &os) const
{
  os << "current AP address=" << m_currentApAddr << ", "
     << "ssid=" << m_ssid << ", "
     << "rates=" << m_rates << ", "
     << "Extended Capabilities=" << m_extendedCapability << " , "
     << "HT Capabilities=" << m_htCapability << " , "
     << "VHT Capabilities=" << m_vhtCapability << " , "
     << "HE Capabilities=" << m_heCapability;
}

void
MgtReassocRequestHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i = m_capability.Serialize (i);
  i.WriteHtolsbU16 (m_listenInterval);
  WriteTo (i, m_currentApAddr);
  i = m_ssid.Serialize (i);
  i = m_rates.Serialize (i);
  i = m_rates.extended.Serialize (i);
  i = m_extendedCapability.Serialize (i);
  i = m_htCapability.Serialize (i);
  i = m_vhtCapability.Serialize (i);
  i = m_heCapability.Serialize (i);
}

uint32_t
MgtReassocRequestHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  i = m_capability.Deserialize (i);
  m_listenInterval = i.ReadLsbtohU16 ();
  ReadFrom (i, m_currentApAddr);
  i = m_ssid.Deserialize (i);
  i = m_rates.Deserialize (i);
  i = m_rates.extended.DeserializeIfPresent (i);
  i = m_extendedCapability.DeserializeIfPresent (i);
  i = m_htCapability.DeserializeIfPresent (i);
  i = m_vhtCapability.DeserializeIfPresent (i);
  i = m_heCapability.DeserializeIfPresent (i);
  return i.GetDistanceFrom (start);
}


/***********************************************************
 *          Assoc/Reassoc Response
 ***********************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtAssocResponseHeader);

MgtAssocResponseHeader::MgtAssocResponseHeader ()
  : m_aid (0)
{
}

MgtAssocResponseHeader::~MgtAssocResponseHeader ()
{
}

StatusCode
MgtAssocResponseHeader::GetStatusCode (void)
{
  return m_code;
}

SupportedRates
MgtAssocResponseHeader::GetSupportedRates (void)
{
  return m_rates;
}

void
MgtAssocResponseHeader::SetStatusCode (StatusCode code)
{
  m_code = code;
}

void
MgtAssocResponseHeader::SetSupportedRates (SupportedRates rates)
{
  m_rates = rates;
}

void
MgtAssocResponseHeader::SetCapabilities (CapabilityInformation capabilities)
{
  m_capability = capabilities;
}

CapabilityInformation
MgtAssocResponseHeader::GetCapabilities (void) const
{
  return m_capability;
}

void
MgtAssocResponseHeader::SetExtendedCapabilities (ExtendedCapabilities extendedCapabilities)
{
  m_extendedCapability = extendedCapabilities;
}

ExtendedCapabilities
MgtAssocResponseHeader::GetExtendedCapabilities (void) const
{
  return m_extendedCapability;
}

void
MgtAssocResponseHeader::SetHtCapabilities (HtCapabilities htCapabilities)
{
  m_htCapability = htCapabilities;
}

HtCapabilities
MgtAssocResponseHeader::GetHtCapabilities (void) const
{
  return m_htCapability;
}

void
MgtAssocResponseHeader::SetHtOperation (HtOperation htOperation)
{
  m_htOperation = htOperation;
}

HtOperation
MgtAssocResponseHeader::GetHtOperation (void) const
{
  return m_htOperation;
}

void
MgtAssocResponseHeader::SetVhtCapabilities (VhtCapabilities vhtCapabilities)
{
  m_vhtCapability = vhtCapabilities;
}

VhtCapabilities
MgtAssocResponseHeader::GetVhtCapabilities (void) const
{
  return m_vhtCapability;
}

void
MgtAssocResponseHeader::SetVhtOperation (VhtOperation vhtOperation)
{
  m_vhtOperation = vhtOperation;
}

VhtOperation
MgtAssocResponseHeader::GetVhtOperation (void) const
{
  return m_vhtOperation;
}

void
MgtAssocResponseHeader::SetHeCapabilities (HeCapabilities heCapabilities)
{
  m_heCapability = heCapabilities;
}

HeCapabilities
MgtAssocResponseHeader::GetHeCapabilities (void) const
{
  return m_heCapability;
}

void
MgtAssocResponseHeader::SetHeOperation (HeOperation heOperation)
{
  m_heOperation = heOperation;
}

HeOperation
MgtAssocResponseHeader::GetHeOperation (void) const
{
  return m_heOperation;
}

void
MgtAssocResponseHeader::SetAssociationId (uint16_t aid)
{
  m_aid = aid;
}

uint16_t
MgtAssocResponseHeader::GetAssociationId (void) const
{
  return m_aid;
}

void
MgtAssocResponseHeader::SetErpInformation (ErpInformation erpInformation)
{
  m_erpInformation = erpInformation;
}

ErpInformation
MgtAssocResponseHeader::GetErpInformation (void) const
{
  return m_erpInformation;
}

void
MgtAssocResponseHeader::SetEdcaParameterSet (EdcaParameterSet edcaparameters)
{
  m_edcaParameterSet = edcaparameters;
}

void
MgtAssocResponseHeader::SetMuEdcaParameterSet (MuEdcaParameterSet muEdcaParameters)
{
  m_muEdcaParameterSet = muEdcaParameters;
}

EdcaParameterSet
MgtAssocResponseHeader::GetEdcaParameterSet (void) const
{
  return m_edcaParameterSet;
}

MuEdcaParameterSet
MgtAssocResponseHeader::GetMuEdcaParameterSet (void) const
{
  return m_muEdcaParameterSet;
}

TypeId
MgtAssocResponseHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtAssocResponseHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtAssocResponseHeader> ()
  ;
  return tid;
}

TypeId
MgtAssocResponseHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
MgtAssocResponseHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += m_capability.GetSerializedSize ();
  size += m_code.GetSerializedSize ();
  size += 2; //aid
  size += m_rates.GetSerializedSize ();
  size += m_erpInformation.GetSerializedSize ();
  size += m_rates.extended.GetSerializedSize ();
  size += m_edcaParameterSet.GetSerializedSize ();
  size += m_extendedCapability.GetSerializedSize ();
  size += m_htCapability.GetSerializedSize ();
  size += m_htOperation.GetSerializedSize ();
  size += m_vhtCapability.GetSerializedSize ();
  size += m_vhtOperation.GetSerializedSize ();
  size += m_heCapability.GetSerializedSize ();
  size += m_heOperation.GetSerializedSize ();
  size += m_muEdcaParameterSet.GetSerializedSize ();
  return size;
}

void
MgtAssocResponseHeader::Print (std::ostream &os) const
{
  os << "status code=" << m_code << ", "
     << "aid=" << m_aid << ", "
     << "rates=" << m_rates << ", "
     << "ERP information=" << m_erpInformation << ", "
     << "Extended Capabilities=" << m_extendedCapability << " , "
     << "HT Capabilities=" << m_htCapability << " , "
     << "HT Operation=" << m_htOperation << " , "
     << "VHT Capabilities=" << m_vhtCapability << " , "
     << "VHT Operation=" << m_vhtOperation << " , "
     << "HE Capabilities=" << m_heCapability << " , "
     << "HE Operation=" << m_heOperation;
}

void
MgtAssocResponseHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i = m_capability.Serialize (i);
  i = m_code.Serialize (i);
  i.WriteHtolsbU16 (m_aid);
  i = m_rates.Serialize (i);
  i = m_erpInformation.Serialize (i);
  i = m_rates.extended.Serialize (i);
  i = m_edcaParameterSet.Serialize (i);
  i = m_extendedCapability.Serialize (i);
  i = m_htCapability.Serialize (i);
  i = m_htOperation.Serialize (i);
  i = m_vhtCapability.Serialize (i);
  i = m_vhtOperation.Serialize (i);
  i = m_heCapability.Serialize (i);
  i = m_heOperation.Serialize (i);
  i = m_muEdcaParameterSet.Serialize (i);
}

uint32_t
MgtAssocResponseHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  i = m_capability.Deserialize (i);
  i = m_code.Deserialize (i);
  m_aid = i.ReadLsbtohU16 ();
  i = m_rates.Deserialize (i);
  i = m_erpInformation.DeserializeIfPresent (i);
  i = m_rates.extended.DeserializeIfPresent (i);
  i = m_edcaParameterSet.DeserializeIfPresent (i);
  i = m_extendedCapability.DeserializeIfPresent (i);
  i = m_htCapability.DeserializeIfPresent (i);
  i = m_htOperation.DeserializeIfPresent (i);
  i = m_vhtCapability.DeserializeIfPresent (i);
  i = m_vhtOperation.DeserializeIfPresent (i);
  i = m_heCapability.DeserializeIfPresent (i);
  i = m_heOperation.DeserializeIfPresent (i);
  i = m_muEdcaParameterSet.DeserializeIfPresent (i);
  return i.GetDistanceFrom (start);
}


/**********************************************************
 *   ActionFrame
 **********************************************************/
WifiActionHeader::WifiActionHeader ()
{
}

WifiActionHeader::~WifiActionHeader ()
{
}

void
WifiActionHeader::SetAction (WifiActionHeader::CategoryValue type,
                             WifiActionHeader::ActionValue action)
{
  m_category = static_cast<uint8_t> (type);
  switch (type)
    {
    case BLOCK_ACK:
      {
        m_actionValue = static_cast<uint8_t> (action.blockAck);
        break;
      }
    case MESH:
      {
        m_actionValue = static_cast<uint8_t> (action.meshAction);
        break;
      }
    case MULTIHOP:
      {
        m_actionValue = static_cast<uint8_t> (action.multihopAction);
        break;
      }
    case SELF_PROTECTED:
      {
        m_actionValue = static_cast<uint8_t> (action.selfProtectedAction);
        break;
      }
    case VENDOR_SPECIFIC_ACTION:
      {
        break;
      }
    }
}

WifiActionHeader::CategoryValue
WifiActionHeader::GetCategory ()
{
  switch (m_category)
    {
    case BLOCK_ACK:
      return BLOCK_ACK;
    case MESH:
      return MESH;
    case MULTIHOP:
      return MULTIHOP;
    case SELF_PROTECTED:
      return SELF_PROTECTED;
    case VENDOR_SPECIFIC_ACTION:
      return VENDOR_SPECIFIC_ACTION;
    default:
      NS_FATAL_ERROR ("Unknown action value");
      return SELF_PROTECTED;
    }
}

WifiActionHeader::ActionValue
WifiActionHeader::GetAction ()
{
  ActionValue retval;
  retval.selfProtectedAction = PEER_LINK_OPEN; //Needs to be initialized to something to quiet valgrind in default cases
  switch (m_category)
    {
    case BLOCK_ACK:
      switch (m_actionValue)
        {
        case BLOCK_ACK_ADDBA_REQUEST:
          retval.blockAck = BLOCK_ACK_ADDBA_REQUEST;
          break;
        case BLOCK_ACK_ADDBA_RESPONSE:
          retval.blockAck = BLOCK_ACK_ADDBA_RESPONSE;
          break;
        case BLOCK_ACK_DELBA:
          retval.blockAck = BLOCK_ACK_DELBA;
          break;
        }
      break;

    case SELF_PROTECTED:
      switch (m_actionValue)
        {
        case PEER_LINK_OPEN:
          retval.selfProtectedAction = PEER_LINK_OPEN;
          break;
        case PEER_LINK_CONFIRM:
          retval.selfProtectedAction = PEER_LINK_CONFIRM;
          break;
        case PEER_LINK_CLOSE:
          retval.selfProtectedAction = PEER_LINK_CLOSE;
          break;
        case GROUP_KEY_INFORM:
          retval.selfProtectedAction = GROUP_KEY_INFORM;
          break;
        case GROUP_KEY_ACK:
          retval.selfProtectedAction = GROUP_KEY_ACK;
          break;
        default:
          NS_FATAL_ERROR ("Unknown mesh peering management action code");
          retval.selfProtectedAction = PEER_LINK_OPEN; /* quiet compiler */
        }
      break;

    case MESH:
      switch (m_actionValue)
        {
        case LINK_METRIC_REPORT:
          retval.meshAction = LINK_METRIC_REPORT;
          break;
        case PATH_SELECTION:
          retval.meshAction = PATH_SELECTION;
          break;
        case PORTAL_ANNOUNCEMENT:
          retval.meshAction = PORTAL_ANNOUNCEMENT;
          break;
        case CONGESTION_CONTROL_NOTIFICATION:
          retval.meshAction = CONGESTION_CONTROL_NOTIFICATION;
          break;
        case MDA_SETUP_REQUEST:
          retval.meshAction = MDA_SETUP_REQUEST;
          break;
        case MDA_SETUP_REPLY:
          retval.meshAction = MDA_SETUP_REPLY;
          break;
        case MDAOP_ADVERTISMENT_REQUEST:
          retval.meshAction = MDAOP_ADVERTISMENT_REQUEST;
          break;
        case MDAOP_ADVERTISMENTS:
          retval.meshAction = MDAOP_ADVERTISMENTS;
          break;
        case MDAOP_SET_TEARDOWN:
          retval.meshAction = MDAOP_SET_TEARDOWN;
          break;
        case TBTT_ADJUSTMENT_REQUEST:
          retval.meshAction = TBTT_ADJUSTMENT_REQUEST;
          break;
        case TBTT_ADJUSTMENT_RESPONSE:
          retval.meshAction = TBTT_ADJUSTMENT_RESPONSE;
          break;
        default:
          NS_FATAL_ERROR ("Unknown mesh peering management action code");
          retval.selfProtectedAction = PEER_LINK_OPEN; /* quiet compiler */
        }
      break;

    case MULTIHOP: //not yet supported
      switch (m_actionValue)
        {
        case PROXY_UPDATE: //not used so far
          retval.multihopAction = PROXY_UPDATE;
          break;
        case PROXY_UPDATE_CONFIRMATION: //not used so far
          retval.multihopAction = PROXY_UPDATE;
          break;
        default:
          NS_FATAL_ERROR ("Unknown mesh peering management action code");
          retval.selfProtectedAction = PEER_LINK_OPEN; /* quiet compiler */
        }
      break;
    default:
      NS_FATAL_ERROR ("Unsupported mesh action");
      retval.selfProtectedAction = PEER_LINK_OPEN; /* quiet compiler */
    }
  return retval;
}

TypeId
WifiActionHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::WifiActionHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<WifiActionHeader> ()
  ;
  return tid;
}

TypeId
WifiActionHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

std::string
WifiActionHeader::CategoryValueToString (CategoryValue value) const
{
  if (value == BLOCK_ACK)
    {
      return "BlockAck";
    }
  else if (value == MESH)
    {
      return "Mesh";
    }
  else if (value == SELF_PROTECTED)
    {
      return "SelfProtected";
    }
  else if (value == VENDOR_SPECIFIC_ACTION)
    {
      return "VendorSpecificAction";
    }
  else
    {
      std::ostringstream convert;
      convert << value;
      return convert.str ();
    }
}
std::string
WifiActionHeader::SelfProtectedActionValueToString (SelfProtectedActionValue value) const
{
  if (value == PEER_LINK_OPEN)
    {
      return "PeerLinkOpen";
    }
  else if (value == PEER_LINK_CONFIRM)
    {
      return "PeerLinkConfirm";
    }
  else if (value == PEER_LINK_CLOSE)
    {
      return "PeerLinkClose";
    }
  else if (value == GROUP_KEY_INFORM)
    {
      return "GroupKeyInform";
    }
  else if (value == GROUP_KEY_ACK)
    {
      return "GroupKeyAck";
    }
  else
    {
      std::ostringstream convert;
      convert << value;
      return convert.str ();
    }
}

void
WifiActionHeader::Print (std::ostream &os) const
{
  os << "category=" << CategoryValueToString ((CategoryValue) m_category)
     << ", value=" << SelfProtectedActionValueToString ((SelfProtectedActionValue) m_actionValue);
}

uint32_t
WifiActionHeader::GetSerializedSize () const
{
  return 2;
}

void
WifiActionHeader::Serialize (Buffer::Iterator start) const
{
  start.WriteU8 (m_category);
  start.WriteU8 (m_actionValue);
}

uint32_t
WifiActionHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_category = i.ReadU8 ();
  m_actionValue = i.ReadU8 ();
  return i.GetDistanceFrom (start);
}


/***************************************************
*                 ADDBARequest
****************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtAddBaRequestHeader);

MgtAddBaRequestHeader::MgtAddBaRequestHeader ()
  : m_dialogToken (1),
    m_amsduSupport (1),
    m_bufferSize (0)
{
}

TypeId
MgtAddBaRequestHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtAddBaRequestHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtAddBaRequestHeader> ()
  ;
  return tid;
}

TypeId
MgtAddBaRequestHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
MgtAddBaRequestHeader::Print (std::ostream &os) const
{
}

uint32_t
MgtAddBaRequestHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += 1; //Dialog token
  size += 2; //Block ack parameter set
  size += 2; //Block ack timeout value
  size += 2; //Starting sequence control
  return size;
}

void
MgtAddBaRequestHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_dialogToken);
  i.WriteHtolsbU16 (GetParameterSet ());
  i.WriteHtolsbU16 (m_timeoutValue);
  i.WriteHtolsbU16 (GetStartingSequenceControl ());
}

uint32_t
MgtAddBaRequestHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_dialogToken = i.ReadU8 ();
  SetParameterSet (i.ReadLsbtohU16 ());
  m_timeoutValue = i.ReadLsbtohU16 ();
  SetStartingSequenceControl (i.ReadLsbtohU16 ());
  return i.GetDistanceFrom (start);
}

void
MgtAddBaRequestHeader::SetDelayedBlockAck ()
{
  m_policy = 0;
}

void
MgtAddBaRequestHeader::SetImmediateBlockAck ()
{
  m_policy = 1;
}

void
MgtAddBaRequestHeader::SetTid (uint8_t tid)
{
  NS_ASSERT (tid < 16);
  m_tid = tid;
}

void
MgtAddBaRequestHeader::SetTimeout (uint16_t timeout)
{
  m_timeoutValue = timeout;
}

void
MgtAddBaRequestHeader::SetBufferSize (uint16_t size)
{
  m_bufferSize = size;
}

void
MgtAddBaRequestHeader::SetStartingSequence (uint16_t seq)
{
  m_startingSeq = seq;
}

void
MgtAddBaRequestHeader::SetStartingSequenceControl (uint16_t seqControl)
{
  m_startingSeq = (seqControl >> 4) & 0x0fff;
}

void
MgtAddBaRequestHeader::SetAmsduSupport (bool supported)
{
  m_amsduSupport = supported;
}

uint8_t
MgtAddBaRequestHeader::GetTid (void) const
{
  return m_tid;
}

bool
MgtAddBaRequestHeader::IsImmediateBlockAck (void) const
{
  return m_policy == 1;
}

uint16_t
MgtAddBaRequestHeader::GetTimeout (void) const
{
  return m_timeoutValue;
}

uint16_t
MgtAddBaRequestHeader::GetBufferSize (void) const
{
  return m_bufferSize;
}

bool
MgtAddBaRequestHeader::IsAmsduSupported (void) const
{
  return m_amsduSupport == 1;
}

uint16_t
MgtAddBaRequestHeader::GetStartingSequence (void) const
{
  return m_startingSeq;
}

uint16_t
MgtAddBaRequestHeader::GetStartingSequenceControl (void) const
{
  return (m_startingSeq << 4) & 0xfff0;
}

uint16_t
MgtAddBaRequestHeader::GetParameterSet (void) const
{
  uint16_t res = 0;
  res |= m_amsduSupport;
  res |= m_policy << 1;
  res |= m_tid << 2;
  res |= m_bufferSize << 6;
  return res;
}

void
MgtAddBaRequestHeader::SetParameterSet (uint16_t params)
{
  m_amsduSupport = (params) & 0x01;
  m_policy = (params >> 1) & 0x01;
  m_tid = (params >> 2) & 0x0f;
  m_bufferSize = (params >> 6) & 0x03ff;
}


/***************************************************
*                 ADDBAResponse
****************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtAddBaResponseHeader);

MgtAddBaResponseHeader::MgtAddBaResponseHeader ()
  : m_dialogToken (1),
    m_amsduSupport (1),
    m_bufferSize (0)
{
}

TypeId
MgtAddBaResponseHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::MgtAddBaResponseHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtAddBaResponseHeader> ()
  ;
  return tid;
}

TypeId
MgtAddBaResponseHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
MgtAddBaResponseHeader::Print (std::ostream &os) const
{
  os << "status code=" << m_code;
}

uint32_t
MgtAddBaResponseHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += 1; //Dialog token
  size += m_code.GetSerializedSize (); //Status code
  size += 2; //Block ack parameter set
  size += 2; //Block ack timeout value
  return size;
}

void
MgtAddBaResponseHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_dialogToken);
  i = m_code.Serialize (i);
  i.WriteHtolsbU16 (GetParameterSet ());
  i.WriteHtolsbU16 (m_timeoutValue);
}

uint32_t
MgtAddBaResponseHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_dialogToken = i.ReadU8 ();
  i = m_code.Deserialize (i);
  SetParameterSet (i.ReadLsbtohU16 ());
  m_timeoutValue = i.ReadLsbtohU16 ();
  return i.GetDistanceFrom (start);
}

void
MgtAddBaResponseHeader::SetDelayedBlockAck ()
{
  m_policy = 0;
}

void
MgtAddBaResponseHeader::SetImmediateBlockAck ()
{
  m_policy = 1;
}

void
MgtAddBaResponseHeader::SetTid (uint8_t tid)
{
  NS_ASSERT (tid < 16);
  m_tid = tid;
}

void
MgtAddBaResponseHeader::SetTimeout (uint16_t timeout)
{
  m_timeoutValue = timeout;
}

void
MgtAddBaResponseHeader::SetBufferSize (uint16_t size)
{
  m_bufferSize = size;
}

void
MgtAddBaResponseHeader::SetStatusCode (StatusCode code)
{
  m_code = code;
}

void
MgtAddBaResponseHeader::SetAmsduSupport (bool supported)
{
  m_amsduSupport = supported;
}

StatusCode
MgtAddBaResponseHeader::GetStatusCode (void) const
{
  return m_code;
}

uint8_t
MgtAddBaResponseHeader::GetTid (void) const
{
  return m_tid;
}

bool
MgtAddBaResponseHeader::IsImmediateBlockAck (void) const
{
  return m_policy == 1;
}

uint16_t
MgtAddBaResponseHeader::GetTimeout (void) const
{
  return m_timeoutValue;
}

uint16_t
MgtAddBaResponseHeader::GetBufferSize (void) const
{
  return m_bufferSize;
}

bool
MgtAddBaResponseHeader::IsAmsduSupported (void) const
{
  return m_amsduSupport == 1;
}

uint16_t
MgtAddBaResponseHeader::GetParameterSet (void) const
{
  uint16_t res = 0;
  res |= m_amsduSupport;
  res |= m_policy << 1;
  res |= m_tid << 2;
  res |= m_bufferSize << 6;
  return res;
}

void
MgtAddBaResponseHeader::SetParameterSet (uint16_t params)
{
  m_amsduSupport = (params) & 0x01;
  m_policy = (params >> 1) & 0x01;
  m_tid = (params >> 2) & 0x0f;
  m_bufferSize = (params >> 6) & 0x03ff;
}


/***************************************************
*                     DelBa
****************************************************/

NS_OBJECT_ENSURE_REGISTERED (MgtDelBaHeader);

MgtDelBaHeader::MgtDelBaHeader ()
  : m_reasonCode (1)
{
}

TypeId
MgtDelBaHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MgtDelBaHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MgtDelBaHeader> ()
  ;
  return tid;
}

TypeId
MgtDelBaHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
MgtDelBaHeader::Print (std::ostream &os) const
{
}

uint32_t
MgtDelBaHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += 2; //DelBa parameter set
  size += 2; //Reason code
  return size;
}

void
MgtDelBaHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteHtolsbU16 (GetParameterSet ());
  i.WriteHtolsbU16 (m_reasonCode);
}

uint32_t
MgtDelBaHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  SetParameterSet (i.ReadLsbtohU16 ());
  m_reasonCode = i.ReadLsbtohU16 ();
  return i.GetDistanceFrom (start);
}

bool
MgtDelBaHeader::IsByOriginator (void) const
{
  return m_initiator == 1;
}

uint8_t
MgtDelBaHeader::GetTid (void) const
{
  NS_ASSERT (m_tid < 16);
  uint8_t tid = static_cast<uint8_t> (m_tid);
  return tid;
}

void
MgtDelBaHeader::SetByOriginator (void)
{
  m_initiator = 1;
}

void
MgtDelBaHeader::SetByRecipient (void)
{
  m_initiator = 0;
}

void
MgtDelBaHeader::SetTid (uint8_t tid)
{
  NS_ASSERT (tid < 16);
  m_tid = static_cast<uint16_t> (tid);
}

uint16_t
MgtDelBaHeader::GetParameterSet (void) const
{
  uint16_t res = 0;
  res |= m_initiator << 11;
  res |= m_tid << 12;
  return res;
}

void
MgtDelBaHeader::SetParameterSet (uint16_t params)
{
  m_initiator = (params >> 11) & 0x01;
  m_tid = (params >> 12) & 0x0f;
}

} //namespace ns3
