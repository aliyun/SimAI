/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Orange Labs
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
 * Author: Rediet <getachew.redieteab@orange.com>
 *         Muhammad Iqbal Rochman <muhiqbalcr@uchicago.edu>
 *         Sébastien Deronne <sebastien.deronne@gmail.com> (VhtSigHeader)
 */

#include "ns3/wifi-phy.h"
#include "ns3/wifi-psdu.h"
#include "vht-phy.h"
#include "vht-ppdu.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("VhtPpdu");

VhtPpdu::VhtPpdu (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, Time ppduDuration,
                  WifiPhyBand band, uint64_t uid)
  : OfdmPpdu (psdu, txVector, band, uid, false) //don't instantiate LSigHeader of OfdmPpdu
{
  NS_LOG_FUNCTION (this << psdu << txVector << ppduDuration << band << uid);
  uint16_t length = ((ceil ((static_cast<double> (ppduDuration.GetNanoSeconds () - (20 * 1000)) / 1000) / 4.0) * 3) - 3);
  m_lSig.SetLength (length);
  m_vhtSig.SetMuFlag (m_preamble == WIFI_PREAMBLE_VHT_MU);
  m_vhtSig.SetChannelWidth (m_channelWidth);
  m_vhtSig.SetShortGuardInterval (txVector.GetGuardInterval () == 400);
  uint32_t nSymbols = (static_cast<double> ((ppduDuration - WifiPhy::CalculatePhyPreambleAndHeaderDuration (txVector)).GetNanoSeconds ()) / (3200 + txVector.GetGuardInterval ()));
  if (txVector.GetGuardInterval () == 400)
    {
      m_vhtSig.SetShortGuardIntervalDisambiguation ((nSymbols % 10) == 9);
    }
  m_vhtSig.SetSuMcs (txVector.GetMode ().GetMcsValue ());
  m_vhtSig.SetNStreams (txVector.GetNss ());
}

VhtPpdu::~VhtPpdu ()
{
}

WifiTxVector
VhtPpdu::DoGetTxVector (void) const
{
  WifiTxVector txVector;
  txVector.SetPreambleType (m_preamble);
  txVector.SetMode (VhtPhy::GetVhtMcs (m_vhtSig.GetSuMcs ()));
  txVector.SetChannelWidth (m_vhtSig.GetChannelWidth ());
  txVector.SetNss (m_vhtSig.GetNStreams ());
  txVector.SetGuardInterval (m_vhtSig.GetShortGuardInterval () ? 400 : 800);
  txVector.SetAggregation (GetPsdu ()->IsAggregate ());
  return txVector;
}

Time
VhtPpdu::GetTxDuration (void) const
{
  Time ppduDuration = Seconds (0);
  const WifiTxVector& txVector = GetTxVector ();
  Time tSymbol = NanoSeconds (3200 + txVector.GetGuardInterval ());
  Time preambleDuration = WifiPhy::CalculatePhyPreambleAndHeaderDuration (txVector);
  Time calculatedDuration = MicroSeconds (((ceil (static_cast<double> (m_lSig.GetLength () + 3) / 3)) * 4) + 20);
  uint32_t nSymbols = floor (static_cast<double> ((calculatedDuration - preambleDuration).GetNanoSeconds ()) / tSymbol.GetNanoSeconds ());
  if (m_vhtSig.GetShortGuardInterval () && m_vhtSig.GetShortGuardIntervalDisambiguation ())
    {
      nSymbols--;
    }
  ppduDuration = preambleDuration + (nSymbols * tSymbol);
  return ppduDuration;
}

Ptr<WifiPpdu>
VhtPpdu::Copy (void) const
{
  return Create<VhtPpdu> (GetPsdu (), GetTxVector (), GetTxDuration (), m_band, m_uid);
}

WifiPpduType
VhtPpdu::GetType (void) const
{
  return (m_preamble == WIFI_PREAMBLE_VHT_MU) ? WIFI_PPDU_TYPE_DL_MU : WIFI_PPDU_TYPE_SU;
}

VhtPpdu::VhtSigHeader::VhtSigHeader ()
  : m_bw (0),
    m_nsts (0),
    m_sgi (0),
    m_sgi_disambiguation (0),
    m_suMcs (0),
    m_mu (false)
{
}

VhtPpdu::VhtSigHeader::~VhtSigHeader ()
{
}

TypeId
VhtPpdu::VhtSigHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::VhtSigHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<VhtSigHeader> ()
  ;
  return tid;
}

TypeId
VhtPpdu::VhtSigHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
VhtPpdu::VhtSigHeader::Print (std::ostream &os) const
{
  os << "SU_MCS=" << +m_suMcs
     << " CHANNEL_WIDTH=" << GetChannelWidth ()
     << " SGI=" << +m_sgi
     << " NSTS=" << +m_nsts
     << " MU=" << +m_mu;
}

uint32_t
VhtPpdu::VhtSigHeader::GetSerializedSize (void) const
{
  uint32_t size = 0;
  size += 3; //VHT-SIG-A1
  size += 3; //VHT-SIG-A2
  if (m_mu)
    {
      size += 4; //VHT-SIG-B
    }
  return size;
}

void
VhtPpdu::VhtSigHeader::SetMuFlag (bool mu)
{
  m_mu = mu;
}

void
VhtPpdu::VhtSigHeader::SetChannelWidth (uint16_t channelWidth)
{
  if (channelWidth == 160)
    {
      m_bw = 3;
    }
  else if (channelWidth == 80)
    {
      m_bw = 2;
    }
  else if (channelWidth == 40)
    {
      m_bw = 1;
    }
  else
    {
      m_bw = 0;
    }
}

uint16_t
VhtPpdu::VhtSigHeader::GetChannelWidth (void) const
{
  if (m_bw == 3)
    {
      return 160;
    }
  else if (m_bw == 2)
    {
      return 80;
    }
  else if (m_bw == 1)
    {
      return 40;
    }
  else
    {
      return 20;
    }
}

void
VhtPpdu::VhtSigHeader::SetNStreams (uint8_t nStreams)
{
  NS_ASSERT (nStreams <= 8);
  m_nsts = (nStreams - 1);
}

uint8_t
VhtPpdu::VhtSigHeader::GetNStreams (void) const
{
  return (m_nsts + 1);
}

void
VhtPpdu::VhtSigHeader::SetShortGuardInterval (bool sgi)
{
  m_sgi = sgi ? 1 : 0;
}

bool
VhtPpdu::VhtSigHeader::GetShortGuardInterval (void) const
{
  return m_sgi;
}

void
VhtPpdu::VhtSigHeader::SetShortGuardIntervalDisambiguation (bool disambiguation)
{
  m_sgi_disambiguation = disambiguation ? 1 : 0;
}

bool
VhtPpdu::VhtSigHeader::GetShortGuardIntervalDisambiguation (void) const
{
  return m_sgi_disambiguation;
}

void
VhtPpdu::VhtSigHeader::SetSuMcs (uint8_t mcs)
{
  NS_ASSERT (mcs <= 9);
  m_suMcs = mcs;
}

uint8_t
VhtPpdu::VhtSigHeader::GetSuMcs (void) const
{
  return m_suMcs;
}

void
VhtPpdu::VhtSigHeader::Serialize (Buffer::Iterator start) const
{
  //VHT-SIG-A1
  uint8_t byte = m_bw;
  byte |= (0x01 << 2); //Set Reserved bit #2 to 1
  start.WriteU8 (byte);
  uint16_t bytes = (m_nsts & 0x07) << 2;
  bytes |= (0x01 << (23 - 8)); //Set Reserved bit #23 to 1
  start.WriteU16 (bytes);

  //VHT-SIG-A2
  byte = m_sgi & 0x01;
  byte |= ((m_sgi_disambiguation & 0x01) << 1);
  byte |= ((m_suMcs & 0x0f) << 4);
  start.WriteU8 (byte);
  bytes = (0x01 << (9 - 8)); //Set Reserved bit #9 to 1
  start.WriteU16 (bytes);

  if (m_mu)
    {
      //VHT-SIG-B
      start.WriteU32 (0);
    }
}

uint32_t
VhtPpdu::VhtSigHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  //VHT-SIG-A1
  uint8_t byte = i.ReadU8 ();
  m_bw = byte & 0x03;
  uint16_t bytes = i.ReadU16 ();
  m_nsts = ((bytes >> 2) & 0x07);

  //VHT-SIG-A2
  byte = i.ReadU8 ();
  m_sgi = byte & 0x01;
  m_sgi_disambiguation = ((byte >> 1) & 0x01);
  m_suMcs = ((byte >> 4) & 0x0f);
  i.ReadU16 ();

  if (m_mu)
    {
      //VHT-SIG-B
      i.ReadU32 ();
    }

  return i.GetDistanceFrom (start);
}

} //namespace ns3
