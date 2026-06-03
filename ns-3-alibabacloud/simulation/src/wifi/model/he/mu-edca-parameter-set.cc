/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "mu-edca-parameter-set.h"
#include <cmath>
#include <algorithm>

namespace ns3 {

MuEdcaParameterSet::MuEdcaParameterSet ()
  : m_qosInfo (0),
    m_records {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}}
{
}

WifiInformationElementId
MuEdcaParameterSet::ElementId () const
{
  return IE_EXTENSION;
}

WifiInformationElementId
MuEdcaParameterSet::ElementIdExt () const
{
  return IE_EXT_MU_EDCA_PARAMETER_SET;
}

bool
MuEdcaParameterSet::IsPresent (void) const
{
  auto timerNotNull = [](const ParameterRecord &r) { return r.muEdcaTimer != 0; };

  bool isPresent = std::all_of (m_records.begin (), m_records.end (), timerNotNull);
  if (isPresent)
    {
      return true;
    }
  NS_ABORT_MSG_IF (std::any_of (m_records.begin (), m_records.end (), timerNotNull),
                   "MU EDCA Timers must be either all zero or all non-zero.");
  return false;
}

void
MuEdcaParameterSet::SetQosInfo (uint8_t qosInfo)
{
  m_qosInfo = qosInfo;
}

void
MuEdcaParameterSet::SetMuAifsn (uint8_t aci, uint8_t aifsn)
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  NS_ABORT_MSG_IF (aifsn == 1 || aifsn > 15, "Invalid AIFSN value: " << +aifsn);

  m_records[aci].aifsnField |= (aifsn & 0x0f);
  m_records[aci].aifsnField |= (aci & 0x03) << 5;
}

void
MuEdcaParameterSet::SetMuCwMin (uint8_t aci, uint16_t cwMin)
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  NS_ABORT_MSG_IF (cwMin > 32767, "CWmin exceeds the maximum value");

  auto eCwMin = std::log2 (cwMin + 1);
  NS_ABORT_MSG_IF (std::trunc (eCwMin) != eCwMin, "CWmin is not a power of 2 minus 1");

  m_records[aci].cwMinMax |= (static_cast<uint8_t> (eCwMin) & 0x0f);
}

void
MuEdcaParameterSet::SetMuCwMax (uint8_t aci, uint16_t cwMax)
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  NS_ABORT_MSG_IF (cwMax > 32767, "CWmin exceeds the maximum value");

  auto eCwMax = std::log2 (cwMax + 1);
  NS_ABORT_MSG_IF (std::trunc (eCwMax) != eCwMax, "CWmax is not a power of 2 minus 1");

  m_records[aci].cwMinMax |= (static_cast<uint8_t> (eCwMax) & 0x0f) << 4;
}

void
MuEdcaParameterSet::SetMuEdcaTimer (uint8_t aci, Time timer)
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  NS_ABORT_MSG_IF (timer.IsStrictlyPositive () && timer < MicroSeconds (8192),
                   "Timer value is below 8.192 ms");
  NS_ABORT_MSG_IF (timer > MicroSeconds (2088960), "Timer value is above 2088.96 ms");

  double value = timer.GetMicroSeconds () / 8192.;
  NS_ABORT_MSG_IF (std::trunc (value) != value, "Timer value is not a multiple of 8 TUs (8192 us)");

  m_records[aci].muEdcaTimer = static_cast<uint8_t> (value);
}

uint8_t
MuEdcaParameterSet::GetQosInfo (void) const
{
  return m_qosInfo;
}

uint8_t
MuEdcaParameterSet::GetMuAifsn (uint8_t aci) const
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  return (m_records[aci].aifsnField & 0x0f);
}

uint16_t
MuEdcaParameterSet::GetMuCwMin (uint8_t aci) const
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  uint8_t eCwMin = (m_records[aci].cwMinMax & 0x0f);
  return static_cast<uint16_t> (std::exp2 (eCwMin) - 1);
}

uint16_t
MuEdcaParameterSet::GetMuCwMax (uint8_t aci) const
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  uint8_t eCwMax = ((m_records[aci].cwMinMax >> 4) & 0x0f);
  return static_cast<uint16_t> (std::exp2 (eCwMax) - 1);
}

Time
MuEdcaParameterSet::GetMuEdcaTimer (uint8_t aci) const
{
  NS_ABORT_MSG_IF (aci > 3, "Invalid AC Index value: " << +aci);
  return MicroSeconds (m_records[aci].muEdcaTimer * 8192);
}

uint8_t
MuEdcaParameterSet::GetInformationFieldSize () const
{
  NS_ASSERT (IsPresent ());
  // ElementIdExt (1) + QoS Info (1) + MU Parameter Records (4 * 3)
  return 14;
}

Buffer::Iterator
MuEdcaParameterSet::Serialize (Buffer::Iterator i) const
{
  if (!IsPresent ())
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
MuEdcaParameterSet::GetSerializedSize () const
{
  if (!IsPresent ())
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

void
MuEdcaParameterSet::SerializeInformationField (Buffer::Iterator start) const
{
  if (IsPresent ())
    {
      start.WriteU8 (GetQosInfo ());
      for (const auto& record : m_records)
        {
          start.WriteU8 (record.aifsnField);
          start.WriteU8 (record.cwMinMax);
          start.WriteU8 (record.muEdcaTimer);
        }
    }
}

uint8_t
MuEdcaParameterSet::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  m_qosInfo = i.ReadU8 ();
  for (auto& record : m_records)
    {
      record.aifsnField = i.ReadU8 ();
      record.cwMinMax = i.ReadU8 ();
      record.muEdcaTimer = i.ReadU8 ();
    }
  return 13;
}

} //namespace ns3
