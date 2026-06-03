/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Foundation, Include., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Nicola Baldo <nbaldo@cttc.es>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include <iomanip>
#include <cmath>
#include "ns3/log.h"
#include "radiotap-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RadiotapHeader");

NS_OBJECT_ENSURE_REGISTERED (RadiotapHeader);

RadiotapHeader::RadiotapHeader ()
  : m_length (8),
    m_present (0),
    m_tsft (0),
    m_flags (FRAME_FLAG_NONE),
    m_rate (0),
    m_channelFreq (0),
    m_channelFlags (CHANNEL_FLAG_NONE),
    m_antennaSignal (0),
    m_antennaNoise (0),
    m_ampduStatusRef (0),
    m_ampduStatusFlags (0),
    m_ampduStatusCRC (0),
    m_vhtPad (0),
    m_vhtKnown (0),
    m_vhtFlags (0),
    m_vhtBandwidth (0),
    m_vhtCoding (0),
    m_vhtGroupId (0),
    m_vhtPartialAid (0),
    m_hePad (0),
    m_heData1 (0),
    m_heData2 (0),
    m_heData3 (0),
    m_heData4 (0),
    m_heData5 (0),
    m_heData6 (0),
    m_heMuPad (0),
    m_heMuFlags1 (0),
    m_heMuFlags2 (0),
    m_heMuOtherUserPad (0),
    m_heMuPerUser1 (0),
    m_heMuPerUser2 (0),
    m_heMuPerUserPosition (0),
    m_heMuPerUserKnown (0)
{
  NS_LOG_FUNCTION (this);
}

TypeId RadiotapHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RadiotapHeader")
    .SetParent<Header> ()
    .SetGroupName ("Network")

    .AddConstructor<RadiotapHeader> ()
  ;
  return tid;
}

TypeId
RadiotapHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
RadiotapHeader::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_length;
}

void
RadiotapHeader::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);

  start.WriteU8 (0); // major version of radiotap header
  start.WriteU8 (0); // pad field
  start.WriteU16 (m_length); // entire length of radiotap data + header
  start.WriteU32 (m_present); // bits describing which fields follow header

  //
  // Time Synchronization Function Timer (when the first bit of the MPDU
  // arrived at the MAC)
  // Reference: https://www.radiotap.org/fields/TSFT.html
  //
  if (m_present & RADIOTAP_TSFT) // bit 0
    {
      start.WriteU64 (m_tsft);
    }

  //
  // Properties of transmitted and received frames.
  // Reference: https://www.radiotap.org/fields/Flags.html
  //
  if (m_present & RADIOTAP_FLAGS) // bit 1
    {
      start.WriteU8 (m_flags);
    }

  //
  // TX/RX data rate in units of 500 kbps
  // Reference: https://www.radiotap.org/fields/Rate.html
  //
  if (m_present & RADIOTAP_RATE) // bit 2
    {
      start.WriteU8 (m_rate);
    }

  //
  // Tx/Rx frequency in MHz, followed by flags.
  // Reference: https://www.radiotap.org/fields/Channel.html
  //
  if (m_present & RADIOTAP_CHANNEL) // bit 3
    {
      start.WriteU8 (0, m_channelPad);
      start.WriteU16 (m_channelFreq);
      start.WriteU16 (m_channelFlags);
    }

  //
  // The hop set and pattern for frequency-hopping radios.  We don't need it but
  // still need to account for it.
  // Reference: https://www.radiotap.org/fields/FHSS.html
  //
  if (m_present & RADIOTAP_FHSS) // bit 4
    {
      start.WriteU8 (0); //not yet implemented
    }

  //
  // RF signal power at the antenna, decibel difference from an arbitrary, fixed
  // reference.
  // Reference: https://www.radiotap.org/fields/Antenna%20signal.html
  //
  if (m_present & RADIOTAP_DBM_ANTSIGNAL) // bit 5
    {
      start.WriteU8 (m_antennaSignal);
    }

  //
  // RF noise power at the antenna, decibel difference from an arbitrary, fixed
  // reference.
  // Reference: https://www.radiotap.org/fields/Antenna%20noise.html
  //
  if (m_present & RADIOTAP_DBM_ANTNOISE) // bit 6
    {
      start.WriteU8 (m_antennaNoise);
    }

  //
  // Quality of Barker code lock.
  // Reference: https://www.radiotap.org/fields/Lock%20quality.html
  //
  if (m_present & RADIOTAP_LOCK_QUALITY) // bit 7
    {
      start.WriteU16 (0); //not yet implemented
    }

  //
  // Transmit power expressed as unitless distance from max power
  // set at factory calibration (0 is max power).
  // Reference: https://www.radiotap.org/fields/TX%20attenuation.html
  //
  if (m_present & RADIOTAP_TX_ATTENUATION) // bit 8
    {
      start.WriteU16 (0); //not yet implemented
    }

  //
  // Transmit power expressed as decibel distance from max power
  // set at factory calibration (0 is max power).
  // Reference: https://www.radiotap.org/fields/dB%20TX%20attenuation.html
  //
  if (m_present & RADIOTAP_DB_TX_ATTENUATION) // bit 9
    {
      start.WriteU16 (0); //not yet implemented
    }

  //
  // Transmit power expressed as dBm (decibels from a 1 milliwatt reference).
  // This is the absolute power level measured at the antenna port.
  // Reference: https://www.radiotap.org/fields/dBm%20TX%20power.html
  //
  if (m_present & RADIOTAP_DBM_TX_POWER) // bit 10
    {
      start.WriteU8 (0); //not yet implemented
    }

  //
  // Unitless indication of the Rx/Tx antenna for this packet.
  // The first antenna is antenna 0.
  // Reference: https://www.radiotap.org/fields/Antenna.html
  //
  if (m_present & RADIOTAP_ANTENNA) // bit 11
    {
      start.WriteU8 (0); //not yet implemented
    }

  //
  // RF signal power at the antenna (decibel difference from an arbitrary fixed reference).
  // Reference: https://www.radiotap.org/fields/dB%20antenna%20signal.html
  //
  if (m_present & RADIOTAP_DB_ANTSIGNAL) // bit 12
    {
      start.WriteU8 (0); //not yet implemented
    }

  //
  // RF noise power at the antenna (decibel difference from an arbitrary fixed reference).
  // Reference: https://www.radiotap.org/fields/dB%20antenna%20noise.html
  //
  if (m_present & RADIOTAP_DB_ANTNOISE) // bit 13
    {
      start.WriteU8 (0); //not yet implemented
    }

  //
  // Properties of received frames.
  // Reference: https://www.radiotap.org/fields/RX%20flags.html
  //
  if (m_present & RADIOTAP_RX_FLAGS) // bit 14
    {
      start.WriteU16 (0); //not yet implemented
    }

  //
  // MCS field.
  // Reference: https://www.radiotap.org/fields/MCS.html
  //
  if (m_present & RADIOTAP_MCS) // bit 19
    {
      start.WriteU8 (m_mcsKnown);
      start.WriteU8 (m_mcsFlags);
      start.WriteU8 (m_mcsRate);
    }

  //
  // A-MPDU Status, information about the received or transmitted A-MPDU.
  // Reference: https://www.radiotap.org/fields/A-MPDU%20status.html
  //
  if (m_present & RADIOTAP_AMPDU_STATUS) // bit 20
    {
      start.WriteU8 (0, m_ampduStatusPad);
      start.WriteU32 (m_ampduStatusRef);
      start.WriteU16 (m_ampduStatusFlags);
      start.WriteU8 (m_ampduStatusCRC);
      start.WriteU8 (0);
    }

  //
  // Information about the received or transmitted VHT frame.
  // Reference: https://www.radiotap.org/fields/VHT.html
  //
  if (m_present & RADIOTAP_VHT) // bit 21
    {
      start.WriteU8 (0, m_vhtPad);
      start.WriteU16 (m_vhtKnown);
      start.WriteU8 (m_vhtFlags);
      start.WriteU8 (m_vhtBandwidth);
      for (uint8_t i = 0; i < 4; i++)
        {
          start.WriteU8 (m_vhtMcsNss[i]);
        }
      start.WriteU8 (m_vhtCoding);
      start.WriteU8 (m_vhtGroupId);
      start.WriteU16 (m_vhtPartialAid);
    }

  //
  // HE field.
  // Reference: https://www.radiotap.org/fields/HE.html
  //
  if (m_present & RADIOTAP_HE) // bit 23
    {
      start.WriteU8 (0, m_hePad);
      start.WriteU16 (m_heData1);
      start.WriteU16 (m_heData2);
      start.WriteU16 (m_heData3);
      start.WriteU16 (m_heData4);
      start.WriteU16 (m_heData5);
      start.WriteU16 (m_heData6);
    }

  //
  // HE MU field.
  // Reference: https://www.radiotap.org/fields/HE-MU.html
  //
  if (m_present & RADIOTAP_HE_MU) // bit 24
    {
      start.WriteU8 (0, m_heMuPad);
      start.WriteU16 (m_heMuFlags1);
      start.WriteU16 (m_heMuFlags2);
      start.WriteU8 (0);
      start.WriteU8 (0);
      start.WriteU8 (0);
      start.WriteU8 (0);
      start.WriteU8 (0);
      start.WriteU8 (0);
      start.WriteU8 (0);
      start.WriteU8 (0);
    }

  //
  // HE MU other user field.
  // Reference: https://www.radiotap.org/fields/HE-MU-other-user.html
  //
  if (m_present & RADIOTAP_HE_MU_OTHER_USER) // bit 25
    {
      start.WriteU8 (0, m_heMuOtherUserPad);
      start.WriteU16 (m_heMuPerUser1);
      start.WriteU16 (m_heMuPerUser2);
      start.WriteU8 (m_heMuPerUserPosition);
      start.WriteU8 (m_heMuPerUserKnown);
    }
}

uint32_t
RadiotapHeader::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);

  uint8_t tmp = start.ReadU8 (); // major version of radiotap header
  NS_ASSERT_MSG (tmp == 0x00, "RadiotapHeader::Deserialize(): Unexpected major version");
  start.ReadU8 (); // pad field

  m_length = start.ReadU16 (); // entire length of radiotap data + header
  m_present = start.ReadU32 (); // bits describing which fields follow header

  uint32_t bytesRead = 8;

  //
  // Time Synchronization Function Timer (when the first bit of the MPDU arrived at the MAC)
  // Reference: https://www.radiotap.org/fields/TSFT.html
  //
  if (m_present & RADIOTAP_TSFT) // bit 0
    {
      m_tsft = start.ReadU64 ();
      bytesRead += 8;
    }

  //
  // Properties of transmitted and received frames.
  // Reference: https://www.radiotap.org/fields/Flags.html
  //
  if (m_present & RADIOTAP_FLAGS) // bit 1
    {
      m_flags = start.ReadU8 ();
      ++bytesRead;
    }

  //
  // TX/RX data rate in units of 500 kbps
  // Reference: https://www.radiotap.org/fields/Rate.html
  //
  if (m_present & RADIOTAP_RATE) // bit 2
    {
      m_rate = start.ReadU8 ();
      ++bytesRead;
    }

  //
  // Tx/Rx frequency in MHz, followed by flags.
  // Reference: https://www.radiotap.org/fields/Channel.html
  //
  if (m_present & RADIOTAP_CHANNEL) // bit 3
    {
      m_channelPad = ((2 - bytesRead % 2) % 2);
      start.Next (m_channelPad);
      m_channelFreq = start.ReadU16 ();
      m_channelFlags = start.ReadU16 ();
      bytesRead += (4 + m_channelPad);
    }

  //
  // The hop set and pattern for frequency-hopping radios.  We don't need it but
  // still need to account for it.
  // Reference: https://www.radiotap.org/fields/FHSS.html
  //
  if (m_present & RADIOTAP_FHSS) // bit 4
    {
      //not yet implemented
      start.ReadU8 ();
      ++bytesRead;
    }

  //
  // RF signal power at the antenna, decibel difference from an arbitrary, fixed
  // reference.
  // Reference: https://www.radiotap.org/fields/Antenna%20signal.html
  //
  if (m_present & RADIOTAP_DBM_ANTSIGNAL) // bit 5
    {
      m_antennaSignal = start.ReadU8 ();
      ++bytesRead;
    }

  //
  // RF noise power at the antenna, decibel difference from an arbitrary, fixed
  // reference.
  // Reference: https://www.radiotap.org/fields/Antenna%20noise.html
  //
  if (m_present & RADIOTAP_DBM_ANTNOISE) // bit 6
    {
      m_antennaNoise = start.ReadU8 ();
      ++bytesRead;
    }

  //
  // Quality of Barker code lock.
  // Reference: https://www.radiotap.org/fields/Lock%20quality.html
  //
  if (m_present & RADIOTAP_LOCK_QUALITY) // bit 7
    {
      //not yet implemented
      start.ReadU16 ();
      bytesRead += 2;
    }

  //
  // Transmit power expressed as unitless distance from max power
  // set at factory calibration (0 is max power).
  // Reference: https://www.radiotap.org/fields/TX%20attenuation.html
  //
  if (m_present & RADIOTAP_TX_ATTENUATION) // bit 8
    {
      //not yet implemented
      start.ReadU16 ();
      bytesRead += 2;
    }

  //
  // Transmit power expressed as decibel distance from max power
  // set at factory calibration (0 is max power).
  // Reference: https://www.radiotap.org/fields/dB%20TX%20attenuation.html
  //
  if (m_present & RADIOTAP_DB_TX_ATTENUATION) // bit 9
    {
      //not yet implemented
      start.ReadU16 ();
      bytesRead += 2;
    }

  //
  // Transmit power expressed as dBm (decibels from a 1 milliwatt reference).
  // This is the absolute power level measured at the antenna port.
  // Reference: https://www.radiotap.org/fields/dBm%20TX%20power.html
  //
  if (m_present & RADIOTAP_DBM_TX_POWER) // bit 10
    {
      //not yet implemented
      start.ReadU8 ();
      ++bytesRead;
    }

  //
  // Unitless indication of the Rx/Tx antenna for this packet.
  // The first antenna is antenna 0.
  // Reference: https://www.radiotap.org/fields/Antenna.html
  //
  if (m_present & RADIOTAP_ANTENNA) // bit 11
    {
      //not yet implemented
      start.ReadU8 ();
      ++bytesRead;
    }

  //
  // RF signal power at the antenna (decibel difference from an arbitrary fixed reference).
  // Reference: https://www.radiotap.org/fields/dB%20antenna%20signal.html
  //
  if (m_present & RADIOTAP_DB_ANTSIGNAL) // bit 12
    {
      //not yet implemented
      start.ReadU8 ();
      ++bytesRead;
    }

  //
  // RF noise power at the antenna (decibel difference from an arbitrary fixed reference).
  // Reference: https://www.radiotap.org/fields/dB%20antenna%20noise.html
  //
  if (m_present & RADIOTAP_DB_ANTNOISE) // bit 13
    {
      //not yet implemented
      start.ReadU8 ();
      ++bytesRead;
    }

  //
  // Properties of received frames.
  // Reference: https://www.radiotap.org/fields/RX%20flags.html
  //
  if (m_present & RADIOTAP_RX_FLAGS) // bit 14
    {
      //not yet implemented
      start.ReadU16 ();
      bytesRead += 2;
    }

  //
  // MCS field.
  // Reference: https://www.radiotap.org/fields/MCS.html
  //
  if (m_present & RADIOTAP_MCS) // bit 19
    {
      m_mcsKnown = start.ReadU8 ();
      m_mcsFlags = start.ReadU8 ();
      m_mcsRate = start.ReadU8 ();
      bytesRead += 3;
    }

  //
  // A-MPDU Status, information about the received or transmitted A-MPDU.
  // Reference: https://www.radiotap.org/fields/A-MPDU%20status.html
  //
  if (m_present & RADIOTAP_AMPDU_STATUS) // bit 20
    {
      m_ampduStatusPad = ((4 - bytesRead % 4) % 4);
      start.Next (m_ampduStatusPad);
      m_ampduStatusRef = start.ReadU32 ();
      m_ampduStatusFlags = start.ReadU16 ();
      m_ampduStatusCRC = start.ReadU8 ();
      start.ReadU8 ();
      bytesRead += (8 + m_ampduStatusPad);
    }

  //
  // Information about the received or transmitted VHT frame.
  // Reference: https://www.radiotap.org/fields/VHT.html
  //
  if (m_present & RADIOTAP_VHT) // bit 21
    {
      m_vhtPad = ((2 - bytesRead % 2) % 2);
      start.Next (m_vhtPad);
      m_vhtKnown = start.ReadU16 ();
      m_vhtFlags = start.ReadU8 ();
      m_vhtBandwidth = start.ReadU8 ();
      for (uint8_t i = 0; i < 4; i++)
        {
          m_vhtMcsNss[i] = start.ReadU8 ();
        }
      m_vhtCoding = start.ReadU8 ();
      m_vhtGroupId = start.ReadU8 ();
      m_vhtPartialAid = start.ReadU16 ();
      bytesRead += (12 + m_vhtPad);
    }

  //
  // HE field.
  // Reference: https://www.radiotap.org/fields/HE.html
  //
  if (m_present & RADIOTAP_HE) // bit 23
    {
      m_hePad = ((2 - bytesRead % 2) % 2);
      start.Next (m_hePad);
      m_heData1 = start.ReadU16 ();
      m_heData2 = start.ReadU16 ();
      m_heData3 = start.ReadU16 ();
      m_heData4 = start.ReadU16 ();
      m_heData5 = start.ReadU16 ();
      m_heData6 = start.ReadU16 ();
      bytesRead += (12 + m_hePad);
    }

  //
  // HE MU field.
  // Reference: https://www.radiotap.org/fields/HE-MU.html
  //
  if (m_present & RADIOTAP_HE_MU) // bit 24
    {
      m_heMuPad = ((2 - bytesRead % 2) % 2);
      m_heMuFlags1 = start.ReadU16 ();
      m_heMuFlags2 = start.ReadU16 ();
      start.ReadU8 ();
      start.ReadU8 ();
      start.ReadU8 ();
      start.ReadU8 ();
      start.ReadU8 ();
      start.ReadU8 ();
      start.ReadU8 ();
      start.ReadU8 ();
      bytesRead += (12 + m_heMuPad);
    }

  //
  // HE MU other user field.
  // Reference: https://www.radiotap.org/fields/HE-MU-other-user.html
  //
  if (m_present & RADIOTAP_HE_MU_OTHER_USER) // bit 25
    {
      m_heMuOtherUserPad = ((2 - bytesRead % 2) % 2);
      m_heMuPerUser1 = start.ReadU16 ();
      m_heMuPerUser2 = start.ReadU16 ();
      m_heMuPerUserPosition = start.ReadU8 ();
      m_heMuPerUserKnown = start.ReadU8 ();
      bytesRead += (6 + m_heMuOtherUserPad);
    }

  NS_ASSERT_MSG (m_length == bytesRead, "RadiotapHeader::Deserialize(): expected and actual lengths inconsistent");
  return bytesRead;
}

void
RadiotapHeader::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << " tsft=" << m_tsft
     << " flags=" << std::hex << m_flags << std::dec
     << " rate=" << +m_rate
     << " freq=" << m_channelFreq
     << " chflags=" << std::hex << +m_channelFlags << std::dec
     << " signal=" << +m_antennaSignal
     << " noise=" << +m_antennaNoise
     << " mcsKnown=" << m_mcsKnown
     << " mcsFlags=" << m_mcsFlags
     << " mcsRate=" << m_mcsRate
     << " ampduStatusFlags=" << +m_ampduStatusFlags
     << " vhtKnown=" << m_vhtKnown
     << " vhtFlags=" << m_vhtFlags
     << " vhtBandwidth=" << m_vhtBandwidth
     << " vhtMcsNss for user 1=" << m_vhtMcsNss[0]
     << " vhtMcsNss for user 2=" << m_vhtMcsNss[1]
     << " vhtMcsNss for user 3=" << m_vhtMcsNss[2]
     << " vhtMcsNss for user 4=" << m_vhtMcsNss[3]
     << " vhtCoding=" << m_vhtCoding
     << " vhtGroupId=" << m_vhtGroupId
     << " vhtPartialAid=" << m_vhtPartialAid
     << " heData1=" << m_heData1
     << " heData2=" << m_heData2
     << " heData3=" << m_heData3
     << " heData4=" << m_heData4
     << " heData5=" << m_heData5
     << " heData6=" << m_heData6
     << " heMuFlags1=" << m_heMuFlags1
     << " heMuFlags2=" << m_heMuFlags2
     << " heMuPerUser1=" << m_heMuPerUser1
     << " heMuPerUser2=" << m_heMuPerUser2
     << " heMuPerUserPosition=" << +m_heMuPerUserPosition
     << " heMuPerUserKnown=" << +m_heMuPerUserKnown;
}

void
RadiotapHeader::SetTsft (uint64_t value)
{
  NS_LOG_FUNCTION (this << value);
  m_tsft = value;

  if (!(m_present & RADIOTAP_TSFT))
    {
      m_present |= RADIOTAP_TSFT;
      m_length += 8;
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetFrameFlags (uint8_t flags)
{
  NS_LOG_FUNCTION (this << +flags);
  m_flags = flags;

  if (!(m_present & RADIOTAP_FLAGS))
    {
      m_present |= RADIOTAP_FLAGS;
      m_length += 1;
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetRate (uint8_t rate)
{
  NS_LOG_FUNCTION (this << +rate);
  m_rate = rate;

  if (!(m_present & RADIOTAP_RATE))
    {
      m_present |= RADIOTAP_RATE;
      m_length += 1;
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetChannelFrequencyAndFlags (uint16_t frequency, uint16_t flags)
{
  NS_LOG_FUNCTION (this << frequency << flags);
  m_channelFreq = frequency;
  m_channelFlags = flags;

  if (!(m_present & RADIOTAP_CHANNEL))
    {
      m_channelPad = ((2 - m_length % 2) % 2);
      m_present |= RADIOTAP_CHANNEL;
      m_length += (4 + m_channelPad);
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetAntennaSignalPower (double signal)
{
  NS_LOG_FUNCTION (this << signal);

  if (!(m_present & RADIOTAP_DBM_ANTSIGNAL))
    {
      m_present |= RADIOTAP_DBM_ANTSIGNAL;
      m_length += 1;
    }
  if (signal > 127)
    {
      m_antennaSignal = 127;
    }
  else if (signal < -128)
    {
      m_antennaSignal = -128;
    }
  else
    {
      m_antennaSignal = static_cast<int8_t> (floor (signal + 0.5));
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetAntennaNoisePower (double noise)
{
  NS_LOG_FUNCTION (this << noise);

  if (!(m_present & RADIOTAP_DBM_ANTNOISE))
    {
      m_present |= RADIOTAP_DBM_ANTNOISE;
      m_length += 1;
    }
  if (noise > 127.0)
    {
      m_antennaNoise = 127;
    }
  else if (noise < -128.0)
    {
      m_antennaNoise = -128;
    }
  else
    {
      m_antennaNoise = static_cast<int8_t> (floor (noise + 0.5));
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetMcsFields (uint8_t known, uint8_t flags, uint8_t mcs)
{
  NS_LOG_FUNCTION (this << known << +flags << +mcs);
  m_mcsKnown = known;
  m_mcsFlags = flags;
  m_mcsRate = mcs;
  if (!(m_present & RADIOTAP_MCS))
    {
      m_present |= RADIOTAP_MCS;
      m_length += 3;
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetAmpduStatus (uint32_t referenceNumber, uint16_t flags, uint8_t crc)
{
  NS_LOG_FUNCTION (this << referenceNumber << flags);
  m_ampduStatusRef = referenceNumber;
  m_ampduStatusFlags = flags;
  m_ampduStatusCRC = crc;
  if (!(m_present & RADIOTAP_AMPDU_STATUS))
    {
      m_ampduStatusPad = ((4 - m_length % 4) % 4);
      m_present |= RADIOTAP_AMPDU_STATUS;
      m_length += (8 + m_ampduStatusPad);
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetVhtFields (uint16_t known, uint8_t flags, uint8_t bandwidth, uint8_t mcs_nss[4], uint8_t coding, uint8_t group_id, uint16_t partial_aid)
{
  NS_LOG_FUNCTION (this << known << flags << +mcs_nss[0] << +mcs_nss[1] << +mcs_nss[2] << +mcs_nss[3] << +coding << +group_id << +partial_aid);
  m_vhtKnown = known;
  m_vhtFlags = flags;
  m_vhtBandwidth = bandwidth;
  for (uint8_t i = 0; i < 4; i++)
    {
      m_vhtMcsNss[i] = mcs_nss[i];
    }
  m_vhtCoding = coding;
  m_vhtGroupId = group_id;
  m_vhtPartialAid = partial_aid;
  if (!(m_present & RADIOTAP_VHT))
    {
      m_vhtPad = ((2 - m_length % 2) % 2);
      m_present |= RADIOTAP_VHT;
      m_length += (12 + m_vhtPad);
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetHeFields (uint16_t data1, uint16_t data2, uint16_t data3, uint16_t data4, uint16_t data5, uint16_t data6)
{
  NS_LOG_FUNCTION (this << data1 << data2 << data3 << data4 << data5 << data6);
  m_heData1 = data1;
  m_heData2 = data2;
  m_heData3 = data3;
  m_heData4 = data4;
  m_heData5 = data5;
  m_heData6 = data6;
  if (!(m_present & RADIOTAP_HE))
    {
      m_hePad = ((2 - m_length % 2) % 2);
      m_present |= RADIOTAP_HE;
      m_length += (12 + m_hePad);
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetHeMuFields (uint16_t flags1, uint16_t flags2, const std::array<uint8_t, 4> &/*ruChannel1*/, const std::array<uint8_t, 4> &/*ruChannel2*/)
{
  NS_LOG_FUNCTION (this << flags1 << flags2);
  m_heMuFlags1 = flags1;
  m_heMuFlags2 = flags2;
  if (!(m_present & RADIOTAP_HE_MU))
    {
      m_heMuPad = ((2 - m_length % 2) % 2);
      m_present |= RADIOTAP_HE_MU;
      m_length += (12 + m_heMuPad);
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

void
RadiotapHeader::SetHeMuPerUserFields (uint16_t perUser1, uint16_t perUser2, uint8_t perUserPosition, uint8_t perUserKnown)
{
  NS_LOG_FUNCTION (this << perUser1 << perUser2 << +perUserPosition << +perUserKnown);
  m_heMuPerUser1 = perUser1;
  m_heMuPerUser2 = perUser2;
  m_heMuPerUserPosition = perUserPosition;
  m_heMuPerUserKnown = perUserKnown;
  if (!(m_present & RADIOTAP_HE_MU_OTHER_USER))
    {
      m_heMuOtherUserPad = ((2 - m_length % 2) % 2);
      m_present |= RADIOTAP_HE_MU_OTHER_USER;
      m_length += (6 + m_heMuOtherUserPad);
    }

  NS_LOG_LOGIC (this << " m_length=" << m_length << " m_present=0x" << std::hex << m_present << std::dec);
}

} // namespace ns3
