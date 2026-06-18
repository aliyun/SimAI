/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 University of Washington
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
 */

#include "ns3/log.h"
#include "ns3/nstime.h"
#include "ns3/uinteger.h"
#include "he-configuration.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HeConfiguration");
NS_OBJECT_ENSURE_REGISTERED (HeConfiguration);

HeConfiguration::HeConfiguration ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
HeConfiguration::GetTypeId (void)
{
  static ns3::TypeId tid = ns3::TypeId ("ns3::HeConfiguration")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<HeConfiguration> ()
    .AddAttribute ("GuardInterval",
                   "Specify the shortest guard interval duration that can be used for HE transmissions."
                   "Possible values are 800ns, 1600ns or 3200ns.",
                   TimeValue (NanoSeconds (3200)),
                   MakeTimeAccessor (&HeConfiguration::GetGuardInterval,
                                     &HeConfiguration::SetGuardInterval),
                   MakeTimeChecker (NanoSeconds (800), NanoSeconds (3200)))
    .AddAttribute ("BssColor",
                   "The BSS color",
                   UintegerValue (0),
                   MakeUintegerAccessor (&HeConfiguration::GetBssColor,
                                         &HeConfiguration::SetBssColor),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("MpduBufferSize",
                   "The MPDU buffer size for receiving A-MPDUs",
                   UintegerValue (64),
                   MakeUintegerAccessor (&HeConfiguration::GetMpduBufferSize,
                                         &HeConfiguration::SetMpduBufferSize),
                   MakeUintegerChecker<uint16_t> (64, 256))
    .AddAttribute ("MuBeAifsn",
                   "AIFSN used by BE EDCA when the MU EDCA Timer is running. "
                   "It must be either zero (EDCA disabled) or a value from 2 to 15.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&HeConfiguration::m_muBeAifsn),
                   MakeUintegerChecker<uint8_t> (0, 15))
    .AddAttribute ("MuBkAifsn",
                   "AIFSN used by BK EDCA when the MU EDCA Timer is running. "
                   "It must be either zero (EDCA disabled) or a value from 2 to 15.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&HeConfiguration::m_muBkAifsn),
                   MakeUintegerChecker<uint8_t> (0, 15))
    .AddAttribute ("MuViAifsn",
                   "AIFSN used by VI EDCA when the MU EDCA Timer is running. "
                   "It must be either zero (EDCA disabled) or a value from 2 to 15.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&HeConfiguration::m_muViAifsn),
                   MakeUintegerChecker<uint8_t> (0, 15))
    .AddAttribute ("MuVoAifsn",
                   "AIFSN used by VO EDCA when the MU EDCA Timer is running. "
                   "It must be either zero (EDCA disabled) or a value from 2 to 15.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&HeConfiguration::m_muVoAifsn),
                   MakeUintegerChecker<uint8_t> (0, 15))
    .AddAttribute ("MuBeCwMin",
                   "CWmin used by BE EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&HeConfiguration::m_muBeCwMin),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("MuBkCwMin",
                   "CWmin used by BK EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&HeConfiguration::m_muBkCwMin),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("MuViCwMin",
                   "CWmin used by VI EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&HeConfiguration::m_muViCwMin),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("MuVoCwMin",
                   "CWmin used by VO EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&HeConfiguration::m_muVoCwMin),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("MuBeCwMax",
                   "CWmax used by BE EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (1023),
                   MakeUintegerAccessor (&HeConfiguration::m_muBeCwMax),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("MuBkCwMax",
                   "CWmax used by BK EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (1023),
                   MakeUintegerAccessor (&HeConfiguration::m_muBkCwMax),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("MuViCwMax",
                   "CWmax used by VI EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (1023),
                   MakeUintegerAccessor (&HeConfiguration::m_muViCwMax),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("MuVoCwMax",
                   "CWmax used by VO EDCA when the MU EDCA Timer is running. "
                   "It must be a power of 2 minus 1 in the range from 0 to 32767.",
                   UintegerValue (1023),
                   MakeUintegerAccessor (&HeConfiguration::m_muVoCwMax),
                   MakeUintegerChecker<uint16_t> (0, 32767))
    .AddAttribute ("BeMuEdcaTimer",
                   "The MU EDCA Timer used by BE EDCA. It must be a multiple of "
                   "8192 us and must be in the range from 8.192 ms to 2088.96 ms. "
                   "0 is a reserved value, but we allow to use this value to indicate "
                   "that an MU EDCA Parameter Set element must not be sent. Therefore, "
                   "0 can only be used if the MU EDCA Timer for all ACs is set to 0.",
                   TimeValue (MicroSeconds (0)),
                   MakeTimeAccessor (&HeConfiguration::m_beMuEdcaTimer),
                   MakeTimeChecker (MicroSeconds (0), MicroSeconds (2088960)))
    .AddAttribute ("BkMuEdcaTimer",
                   "The MU EDCA Timer used by BK EDCA. It must be a multiple of "
                   "8192 us and must be in the range from 8.192 ms to 2088.96 ms."
                   "0 is a reserved value, but we allow to use this value to indicate "
                   "that an MU EDCA Parameter Set element must not be sent. Therefore, "
                   "0 can only be used if the MU EDCA Timer for all ACs is set to 0.",
                   TimeValue (MicroSeconds (0)),
                   MakeTimeAccessor (&HeConfiguration::m_bkMuEdcaTimer),
                   MakeTimeChecker (MicroSeconds (0), MicroSeconds (2088960)))
    .AddAttribute ("ViMuEdcaTimer",
                   "The MU EDCA Timer used by VI EDCA. It must be a multiple of "
                   "8192 us and must be in the range from 8.192 ms to 2088.96 ms."
                   "0 is a reserved value, but we allow to use this value to indicate "
                   "that an MU EDCA Parameter Set element must not be sent. Therefore, "
                   "0 can only be used if the MU EDCA Timer for all ACs is set to 0.",
                   TimeValue (MicroSeconds (0)),
                   MakeTimeAccessor (&HeConfiguration::m_viMuEdcaTimer),
                   MakeTimeChecker (MicroSeconds (0), MicroSeconds (2088960)))
    .AddAttribute ("VoMuEdcaTimer",
                   "The MU EDCA Timer used by VO EDCA. It must be a multiple of "
                   "8192 us and must be in the range from 8.192 ms to 2088.96 ms."
                   "0 is a reserved value, but we allow to use this value to indicate "
                   "that an MU EDCA Parameter Set element must not be sent. Therefore, "
                   "0 can only be used if the MU EDCA Timer for all ACs is set to 0.",
                   TimeValue (MicroSeconds (0)),
                   MakeTimeAccessor (&HeConfiguration::m_voMuEdcaTimer),
                   MakeTimeChecker (MicroSeconds (0), MicroSeconds (2088960)))
    ;
    return tid;
}

void
HeConfiguration::SetGuardInterval (Time guardInterval)
{
  NS_LOG_FUNCTION (this << guardInterval);
  NS_ASSERT (guardInterval == NanoSeconds (800) || guardInterval == NanoSeconds (1600) || guardInterval == NanoSeconds (3200));
  m_guardInterval = guardInterval;
}

Time
HeConfiguration::GetGuardInterval (void) const
{
  return m_guardInterval;
}

void
HeConfiguration::SetBssColor (uint8_t bssColor)
{
  NS_LOG_FUNCTION (this << +bssColor);
  m_bssColor = bssColor;
}

uint8_t
HeConfiguration::GetBssColor (void) const
{
  return m_bssColor;
}

void
HeConfiguration::SetMpduBufferSize (uint16_t size)
{
  NS_LOG_FUNCTION (this << size);
  m_mpduBufferSize = size;
}

uint16_t
HeConfiguration::GetMpduBufferSize (void) const
{
  return m_mpduBufferSize;
}

} //namespace ns3
