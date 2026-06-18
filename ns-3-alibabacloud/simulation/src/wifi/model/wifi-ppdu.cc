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
 */

#include "wifi-ppdu.h"
#include "wifi-psdu.h"
#include "ns3/log.h"
#include "ns3/packet.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiPpdu");

WifiPpdu::WifiPpdu (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector, uint64_t uid /* = UINT64_MAX */)
  : m_preamble (txVector.GetPreambleType ()),
    m_modulation (txVector.IsValid () ? txVector.GetModulationClass () : WIFI_MOD_CLASS_UNKNOWN),
    m_uid (uid),
    m_truncatedTx (false),
    m_txPowerLevel (txVector.GetTxPowerLevel ())
{
  NS_LOG_FUNCTION (this << *psdu << txVector << uid);
  m_psdus.insert (std::make_pair (SU_STA_ID, psdu));
}

WifiPpdu::WifiPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, uint64_t uid)
  : m_preamble (txVector.GetPreambleType ()),
    m_modulation (txVector.IsValid () ? txVector.GetMode (psdus.begin()->first).GetModulationClass () : WIFI_MOD_CLASS_UNKNOWN),
    m_uid (uid),
    m_truncatedTx (false),
    m_txPowerLevel (txVector.GetTxPowerLevel ()),
    m_txAntennas (txVector.GetNTx ())
{
  NS_LOG_FUNCTION (this << psdus << txVector << uid);
  m_psdus = psdus;
}

WifiPpdu::~WifiPpdu ()
{
  for (auto & psdu : m_psdus)
    {
      psdu.second = 0;
    }
  m_psdus.clear ();
}

WifiTxVector
WifiPpdu::GetTxVector (void) const
{
  WifiTxVector txVector = DoGetTxVector ();
  txVector.SetTxPowerLevel (m_txPowerLevel);
  txVector.SetNTx (m_txAntennas);
  return txVector;
}

WifiTxVector
WifiPpdu::DoGetTxVector (void) const
{
  NS_FATAL_ERROR ("This method should not be called for the base WifiPpdu class. Use the overloaded version in the amendment-specific PPDU subclasses instead!");
  return WifiTxVector (); //should be overloaded
}

Ptr<const WifiPsdu>
WifiPpdu::GetPsdu (void) const
{
  return m_psdus.begin ()->second;
}

bool
WifiPpdu::IsTruncatedTx (void) const
{
  return m_truncatedTx;
}

void
WifiPpdu::SetTruncatedTx (void)
{
  NS_LOG_FUNCTION (this);
  m_truncatedTx = true;
}

WifiModulationClass
WifiPpdu::GetModulation (void) const
{
  return m_modulation;
}

uint16_t
WifiPpdu::GetTransmissionChannelWidth (void) const
{
  return GetTxVector ().GetChannelWidth ();
}

bool
WifiPpdu::CanBeReceived (uint16_t txCenterFreq, uint16_t p20MinFreq,
                              uint16_t p20MaxFreq) const
{
  NS_LOG_FUNCTION (this << txCenterFreq << p20MinFreq << p20MaxFreq);

  uint16_t txChannelWidth = GetTxVector ().GetChannelWidth ();
  uint16_t minTxFreq = txCenterFreq - txChannelWidth / 2;
  uint16_t maxTxFreq = txCenterFreq + txChannelWidth / 2;

  if (minTxFreq > p20MinFreq || maxTxFreq < p20MaxFreq)
    {
      NS_LOG_INFO ("Received PPDU does not overlap with the primary20 channel");
      return false;
    }
  return true;
}


uint64_t
WifiPpdu::GetUid (void) const
{
  return m_uid;
}

WifiPreamble
WifiPpdu::GetPreamble (void) const
{
  return m_preamble;
}

WifiPpduType
WifiPpdu::GetType (void) const
{
  return WIFI_PPDU_TYPE_SU;
}

uint16_t
WifiPpdu::GetStaId (void) const
{
  return SU_STA_ID;
}

Time
WifiPpdu::GetTxDuration (void) const
{
  NS_FATAL_ERROR ("This method should not be called for the base WifiPpdu class. Use the overloaded version in the amendment-specific PPDU subclasses instead!");
  return MicroSeconds (0); //should be overloaded
}

void
WifiPpdu::Print (std::ostream& os) const
{
  os << "[ preamble=" << m_preamble
     << ", modulation=" << m_modulation
     << ", truncatedTx=" << (m_truncatedTx ? "Y" : "N")
     << ", UID=" << m_uid
     << ", " << PrintPayload ()
     << "]";
}

std::string
WifiPpdu::PrintPayload (void) const
{
  std::ostringstream ss;
  ss << "PSDU=" << GetPsdu () << " ";
  return ss.str ();
}

Ptr<WifiPpdu>
WifiPpdu::Copy (void) const
{
  NS_FATAL_ERROR ("This method should not be called for the base WifiPpdu class. Use the overloaded version in the amendment-specific PPDU subclasses instead!");
  return Create<WifiPpdu> (GetPsdu (), GetTxVector ());
}

std::ostream & operator << (std::ostream &os, const Ptr<const WifiPpdu> &ppdu)
{
  ppdu->Print (os);
  return os;
}

std::ostream & operator << (std::ostream &os, const WifiConstPsduMap &psdus)
{
  for (auto const& psdu : psdus)
   {
     os << "PSDU for STA_ID=" << psdu.first
        << " (" << *psdu.second << ") ";
   }
   return os;
}

} //namespace ns3
