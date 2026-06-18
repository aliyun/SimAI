/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 *         Michele Muccio <michelemuccio@virgilio.it>
 */

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/ipv6-header.h"
#include "ns3/mac64-address.h"
#include "ns3/mac16-address.h"
#include "ns3/ipv6-header.h"
#include "sixlowpan-header.h"


namespace ns3 {

/*
 * SixLowPanDispatch
 */


SixLowPanDispatch::SixLowPanDispatch ()
{
}


SixLowPanDispatch::Dispatch_e
SixLowPanDispatch::GetDispatchType (uint8_t dispatch)
{
  if (dispatch <= LOWPAN_NALP_N)
    {
      return LOWPAN_NALP;
    }
  else if (dispatch == LOWPAN_IPv6)
    {
      return LOWPAN_IPv6;
    }
  else if (dispatch == LOWPAN_HC1)
    {
      return LOWPAN_HC1;
    }
  else if (dispatch == LOWPAN_BC0)
    {
      return LOWPAN_BC0;
    }
  else if ((dispatch >= LOWPAN_IPHC) && (dispatch <= LOWPAN_IPHC_N))
    {
      return LOWPAN_IPHC;
    }
  else if ((dispatch >= LOWPAN_MESH) && (dispatch <= LOWPAN_MESH_N))
    {
      return LOWPAN_MESH;
    }
  else if ((dispatch >= LOWPAN_FRAG1) && (dispatch <= LOWPAN_FRAG1_N))
    {
      return LOWPAN_FRAG1;
    }
  else if ((dispatch >= LOWPAN_FRAGN) && (dispatch <= LOWPAN_FRAGN_N))
    {
      return LOWPAN_FRAGN;
    }
  return LOWPAN_UNSUPPORTED;
}

SixLowPanDispatch::NhcDispatch_e
SixLowPanDispatch::GetNhcDispatchType (uint8_t dispatch)
{
  if ((dispatch >= LOWPAN_NHC) && (dispatch <= LOWPAN_NHC_N))
    {
      return LOWPAN_NHC;
    }
  else if ((dispatch >= LOWPAN_UDPNHC) && (dispatch <= LOWPAN_UDPNHC_N))
    {
      return LOWPAN_UDPNHC;
    }
  return LOWPAN_NHCUNSUPPORTED;
}


/*
 * SixLowPanHc1
 */
NS_OBJECT_ENSURE_REGISTERED (SixLowPanHc1);

SixLowPanHc1::SixLowPanHc1 ()
  : m_hopLimit (0)
{
}

TypeId SixLowPanHc1::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanHc1")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanHc1> ();
  return tid;
}

TypeId SixLowPanHc1::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanHc1::Print (std::ostream & os) const
{
  uint8_t encoding;
  encoding = m_srcCompression;
  encoding <<= 2;
  encoding |= m_dstCompression;
  encoding <<= 1;
  encoding |= m_tcflCompression;
  encoding <<= 2;
  encoding |= m_nextHeaderCompression;
  encoding <<= 1;
  encoding |= m_hc2HeaderPresent;

  os << "encoding " << +encoding << ", hopLimit " << +m_hopLimit;
}

uint32_t SixLowPanHc1::GetSerializedSize () const
{
  uint32_t serializedSize = 3;

  switch (m_srcCompression)
    {
    case HC1_PIII:
      serializedSize += 16;
      break;
    case HC1_PIIC:
      serializedSize += 8;
      break;
    case HC1_PCII:
      serializedSize += 8;
      break;
    case HC1_PCIC:
      break;
    }
  switch (m_dstCompression)
    {
    case HC1_PIII:
      serializedSize += 16;
      break;
    case HC1_PIIC:
      serializedSize += 8;
      break;
    case HC1_PCII:
      serializedSize += 8;
      break;
    case HC1_PCIC:
      break;
    }

  if (m_tcflCompression == false )
    {
      serializedSize += 4;
    }

  if (m_nextHeaderCompression == HC1_NC )
    {
      serializedSize++;
    }

  return serializedSize;
}

void SixLowPanHc1::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  uint8_t encoding;
  encoding = m_srcCompression;
  encoding <<= 2;
  encoding |= m_dstCompression;
  encoding <<= 1;
  encoding |= m_tcflCompression;
  encoding <<= 2;
  encoding |= m_nextHeaderCompression;
  encoding <<= 1;
  encoding |= m_hc2HeaderPresent;

  i.WriteU8 (SixLowPanDispatch::LOWPAN_HC1);
  i.WriteU8 (encoding);
  i.WriteU8 (m_hopLimit);
  switch (m_srcCompression)
    {
    case HC1_PIII:
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_srcPrefix[j]);
        }
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_srcInterface[j]);
        }
      break;
    case HC1_PIIC:
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_srcPrefix[j]);
        }
      break;
    case HC1_PCII:
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_srcInterface[j]);
        }
      break;
    case HC1_PCIC:
      break;
    }
  switch (m_dstCompression)
    {
    case HC1_PIII:
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_dstPrefix[j]);
        }
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_dstInterface[j]);
        }
      break;
    case HC1_PIIC:
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_dstPrefix[j]);
        }
      break;
    case HC1_PCII:
      for ( int j = 0; j < 8; j++ )
        {
          i.WriteU8 (m_dstInterface[j]);
        }
      break;
    case HC1_PCIC:
      break;
    }

  if ( m_tcflCompression == false )
    {
      i.WriteU8 (m_trafficClass);
      uint8_t temp[3];
      temp[0] = uint8_t (m_flowLabel & 0xff);
      temp[1] = uint8_t ((m_flowLabel >> 8) & 0xff);
      temp[2] = uint8_t ((m_flowLabel >> 16) & 0xff);
      i.Write (temp, 3);
    }

  if (m_nextHeaderCompression == HC1_NC )
    {
      i.WriteU8 (m_nextHeader);
    }

  // TODO: HC2 is not yet supported. Should be.
  NS_ASSERT_MSG ( m_hc2HeaderPresent != true, "Can not compress HC2, exiting. Very sorry." );
}

uint32_t SixLowPanHc1::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint32_t serializedSize = 3;

  uint8_t dispatch = i.ReadU8 ();
  if (dispatch != SixLowPanDispatch::LOWPAN_HC1)
    {
      return 0;
    }

  uint8_t encoding = i.ReadU8 ();
  m_hopLimit = i.ReadU8 ();

  m_srcCompression = LowPanHc1Addr_e (encoding >> 6);
  m_dstCompression = LowPanHc1Addr_e ( (encoding >> 4) & 0x3);
  m_tcflCompression = (encoding >> 3) & 0x1;
  m_nextHeaderCompression = LowPanHc1NextHeader_e ( (encoding >> 1) & 0x3);
  m_hc2HeaderPresent = encoding & 0x1;

  switch (m_srcCompression)
    {
    case HC1_PIII:
      for ( int j = 0; j < 8; j++)
        {
          m_srcPrefix[j] = i.ReadU8 ();
        }
      for ( int j = 0; j < 8; j++)
        {
          m_srcInterface[j] = i.ReadU8 ();
        }
      serializedSize += 16;
      break;
    case HC1_PIIC:
      for ( int j = 0; j < 8; j++)
        {
          m_srcPrefix[j] = i.ReadU8 ();
        }
      serializedSize += 8;
      break;
    case HC1_PCII:
      for ( int j = 0; j < 8; j++)
        {
          m_srcInterface[j] = i.ReadU8 ();
        }
      serializedSize += 8;
      break;
    case HC1_PCIC:
      break;
    }
  switch (m_dstCompression)
    {
    case HC1_PIII:
      for ( int j = 0; j < 8; j++)
        {
          m_dstPrefix[j] = i.ReadU8 ();
        }
      for ( int j = 0; j < 8; j++)
        {
          m_dstInterface[j] = i.ReadU8 ();
        }
      serializedSize += 16;
      break;
    case HC1_PIIC:
      for ( int j = 0; j < 8; j++)
        {
          m_dstPrefix[j] = i.ReadU8 ();
        }
      serializedSize += 8;
      break;
    case HC1_PCII:
      for ( int j = 0; j < 8; j++)
        {
          m_dstInterface[j] = i.ReadU8 ();
        }
      serializedSize += 8;
      break;
    case HC1_PCIC:
      break;
    }

  if ( m_tcflCompression == false )
    {
      m_trafficClass = i.ReadU8 ();
      uint8_t temp[3];
      i.Read (temp, 3);
      m_flowLabel = temp[2];
      m_flowLabel = (m_flowLabel << 8) | temp[1];
      m_flowLabel = (m_flowLabel << 8) | temp[0];
      serializedSize += 4;
    }

  switch ( m_nextHeaderCompression )
    {
    case HC1_NC:
      m_nextHeader = i.ReadU8 ();
      serializedSize++;
      break;
    case HC1_TCP:
      m_nextHeader = Ipv6Header::IPV6_TCP;
      break;
    case HC1_UDP:
      m_nextHeader = Ipv6Header::IPV6_UDP;
      break;
    case HC1_ICMP:
      m_nextHeader = Ipv6Header::IPV6_ICMPV6;
      break;
    }

  NS_ASSERT_MSG ( m_hc2HeaderPresent != true, "Can not compress HC2, exiting. Very sorry." );

  return GetSerializedSize ();
}

void SixLowPanHc1::SetHopLimit (uint8_t limit)
{
  m_hopLimit = limit;
}

uint8_t SixLowPanHc1::GetHopLimit () const
{
  return m_hopLimit;
}

SixLowPanHc1::LowPanHc1Addr_e SixLowPanHc1::GetDstCompression () const
{
  return m_dstCompression;
}

const uint8_t* SixLowPanHc1::GetDstInterface () const
{
  return m_dstInterface;
}

const uint8_t* SixLowPanHc1::GetDstPrefix () const
{
  return m_dstPrefix;
}

uint32_t SixLowPanHc1::GetFlowLabel () const
{
  return m_flowLabel;
}

uint8_t SixLowPanHc1::GetNextHeader () const
{
  return m_nextHeader;
}

SixLowPanHc1::LowPanHc1Addr_e SixLowPanHc1::GetSrcCompression () const
{
  return m_srcCompression;
}

const uint8_t* SixLowPanHc1::GetSrcInterface () const
{
  return m_srcInterface;
}

const uint8_t* SixLowPanHc1::GetSrcPrefix () const
{
  return m_srcPrefix;
}

uint8_t SixLowPanHc1::GetTrafficClass () const
{
  return m_trafficClass;
}

bool SixLowPanHc1::IsTcflCompression () const
{
  return m_tcflCompression;
}

bool SixLowPanHc1::IsHc2HeaderPresent () const
{
  return m_hc2HeaderPresent;
}

void SixLowPanHc1::SetDstCompression (LowPanHc1Addr_e dstCompression)
{
  m_dstCompression = dstCompression;
}

void SixLowPanHc1::SetDstInterface (const uint8_t* dstInterface)
{
  for ( int i = 0; i < 8; i++)
    {
      m_dstInterface[i] = dstInterface[i];
    }
}

void SixLowPanHc1::SetDstPrefix (const uint8_t* dstPrefix)
{
  for ( int i = 0; i < 8; i++)
    {
      m_dstPrefix[i] = dstPrefix[i];
    }
}

void SixLowPanHc1::SetFlowLabel (uint32_t flowLabel)
{
  m_flowLabel = flowLabel;
}

void SixLowPanHc1::SetNextHeader (uint8_t nextHeader)
{
  m_nextHeader = nextHeader;

  switch (m_nextHeader)
    {
    case Ipv6Header::IPV6_UDP:
      m_nextHeaderCompression = HC1_UDP;
      break;
    case Ipv6Header::IPV6_TCP:
      m_nextHeaderCompression = HC1_TCP;
      break;
    case Ipv6Header::IPV6_ICMPV6:
      m_nextHeaderCompression = HC1_ICMP;
      break;
    default:
      m_nextHeaderCompression = HC1_NC;
      break;
    }
}

void SixLowPanHc1::SetSrcCompression (LowPanHc1Addr_e srcCompression)
{
  m_srcCompression = srcCompression;
}

void SixLowPanHc1::SetSrcInterface (const uint8_t* srcInterface)
{
  for ( int i = 0; i < 8; i++)
    {
      m_srcInterface[i] = srcInterface[i];
    }
}

void SixLowPanHc1::SetSrcPrefix (const uint8_t* srcPrefix)
{
  for ( int i = 0; i < 8; i++)
    {
      m_srcPrefix[i] = srcPrefix[i];
    }
}

void SixLowPanHc1::SetTcflCompression (bool tcflCompression)
{
  m_tcflCompression = tcflCompression;
}

void SixLowPanHc1::SetTrafficClass (uint8_t trafficClass)
{
  m_trafficClass = trafficClass;
}


void SixLowPanHc1::SetHc2HeaderPresent (bool hc2HeaderPresent)
{
  m_hc2HeaderPresent = hc2HeaderPresent;
}


std::ostream & operator << (std::ostream & os, const SixLowPanHc1 & h)
{
  h.Print (os);
  return os;
}

/*
 * SixLowPanFrag1
 */
NS_OBJECT_ENSURE_REGISTERED (SixLowPanFrag1);

SixLowPanFrag1::SixLowPanFrag1 ()
  : m_datagramSize (0),
  m_datagramTag (0)
{
}

TypeId SixLowPanFrag1::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanFrag1")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanFrag1> ();
  return tid;
}

TypeId SixLowPanFrag1::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanFrag1::Print (std::ostream & os) const
{
  os << "datagram size " << m_datagramSize << " tag " << m_datagramTag;
}

uint32_t SixLowPanFrag1::GetSerializedSize () const
{
  return 4;
}

void SixLowPanFrag1::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  uint16_t temp = m_datagramSize | ( uint16_t (SixLowPanDispatch::LOWPAN_FRAG1) << 8 );

  i.WriteU8 (uint8_t (temp >> 8));
  i.WriteU8 (uint8_t (temp & 0xff));

  i.WriteU16 (m_datagramTag);
}

uint32_t SixLowPanFrag1::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  uint8_t temp = i.ReadU8 ();
  m_datagramSize = (uint16_t (temp) << 8) | i.ReadU8 ();
  m_datagramSize &= 0x7FF;

  m_datagramTag = i.ReadU16 ();
  return GetSerializedSize ();
}

void SixLowPanFrag1::SetDatagramSize (uint16_t datagramSize)
{
  m_datagramSize = datagramSize & 0x7FF;
}

uint16_t SixLowPanFrag1::GetDatagramSize () const
{
  return m_datagramSize & 0x7FF;
}

void SixLowPanFrag1::SetDatagramTag (uint16_t datagramTag)
{
  m_datagramTag = datagramTag;
}

uint16_t SixLowPanFrag1::GetDatagramTag () const
{
  return m_datagramTag;
}

std::ostream & operator << (std::ostream & os, const SixLowPanFrag1 & h)
{
  h.Print (os);
  return os;
}


/*
 * SixLowPanFragN
 */

NS_OBJECT_ENSURE_REGISTERED (SixLowPanFragN);

SixLowPanFragN::SixLowPanFragN ()
  : m_datagramSize (0),
  m_datagramTag (0),
  m_datagramOffset (0)
{
}
/*
 * SixLowPanFragmentOffset
 */
TypeId SixLowPanFragN::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanFragN")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanFragN> ();
  return tid;
}

TypeId SixLowPanFragN::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanFragN::Print (std::ostream & os) const
{
  os << "datagram size " << m_datagramSize << " tag " << m_datagramTag << " offset " << +m_datagramOffset;
}

uint32_t SixLowPanFragN::GetSerializedSize () const
{
  return 5;
}

void SixLowPanFragN::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  uint16_t temp = m_datagramSize | ( uint16_t (SixLowPanDispatch::LOWPAN_FRAGN) << 8 );

  i.WriteU8 (uint8_t (temp >> 8));
  i.WriteU8 (uint8_t (temp & 0xff));

  i.WriteU16 (m_datagramTag);
  i.WriteU8 (m_datagramOffset);
}

uint32_t SixLowPanFragN::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  uint8_t temp = i.ReadU8 ();
  m_datagramSize = (uint16_t (temp) << 8) | i.ReadU8 ();
  m_datagramSize &= 0x7FF;

  m_datagramTag = i.ReadU16 ();
  m_datagramOffset = i.ReadU8 ();

  return GetSerializedSize ();
}

void SixLowPanFragN::SetDatagramSize (uint16_t datagramSize)
{
  m_datagramSize = datagramSize & 0x7FF;
}

uint16_t SixLowPanFragN::GetDatagramSize () const
{
  return m_datagramSize & 0x7FF;
}

void SixLowPanFragN::SetDatagramTag (uint16_t datagramTag)
{
  m_datagramTag = datagramTag;
}

uint16_t SixLowPanFragN::GetDatagramTag () const
{
  return m_datagramTag;
}

void SixLowPanFragN::SetDatagramOffset (uint8_t datagramOffset)
{
  m_datagramOffset = datagramOffset;
}

uint8_t SixLowPanFragN::GetDatagramOffset () const
{
  return m_datagramOffset;
}

std::ostream & operator << (std::ostream & os, const SixLowPanFragN & h)
{
  h.Print (os);
  return os;
}

/*
 * SixLowPanIpv6
 */

NS_OBJECT_ENSURE_REGISTERED (SixLowPanIpv6);

SixLowPanIpv6::SixLowPanIpv6 ()
{
}

TypeId SixLowPanIpv6::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanIpv6")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanIpv6> ();
  return tid;
}

TypeId SixLowPanIpv6::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanIpv6::Print (std::ostream & os) const
{
  os << "Uncompressed IPv6";
}

uint32_t SixLowPanIpv6::GetSerializedSize () const
{
  return 1;
}

void SixLowPanIpv6::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (uint8_t (SixLowPanDispatch::LOWPAN_IPv6));
}

uint32_t SixLowPanIpv6::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  i.ReadU8 ();

  return GetSerializedSize ();
}

std::ostream & operator << (std::ostream & os, const SixLowPanIpv6 & h)
{
  h.Print (os);
  return os;
}

/*
 * SixLowPanIphcHeader
 */
NS_OBJECT_ENSURE_REGISTERED (SixLowPanIphc);

SixLowPanIphc::SixLowPanIphc ()
{
  // 011x xxxx xxxx xxxx
  m_baseFormat = 0x6000;
  m_srcdstContextId = 0;
}

SixLowPanIphc::SixLowPanIphc (uint8_t dispatch)
{
  // 011x xxxx xxxx xxxx
  m_baseFormat = dispatch;
  m_baseFormat <<= 8;
  m_srcdstContextId = 0;
}

TypeId SixLowPanIphc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanIphc")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanIphc> ();
  return tid;
}

TypeId SixLowPanIphc::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanIphc::Print (std::ostream & os) const
{
  switch ( GetTf () )
    {
    case TF_FULL:
      os << "TF_FULL(" << +m_ecn << ", " << +m_dscp << ", " << m_flowLabel << ")";
      break;
    case TF_DSCP_ELIDED:
      os << "TF_DSCP_ELIDED(" << +m_ecn << ", " << m_flowLabel << ")";
      break;
    case TF_FL_ELIDED:
      os << "TF_FL_ELIDED(" << +m_ecn << ", " << +m_dscp << ")";
      break;
    default:
      os << "TF_ELIDED";
      break;
    }

  GetNh () ? os << " NH(1)" : os << " NH(0)";

  switch ( GetHlim () )
    {
    case HLIM_INLINE:
      os << " HLIM_INLINE(" << +m_hopLimit << ")";
      break;
    case HLIM_COMPR_1:
      os << " HLIM_COMPR_1(1)";
      break;
    case HLIM_COMPR_64:
      os << " HLIM_COMPR_64(64)";
      break;
    default:
      os << " HLIM_COMPR_255(255)";
      break;
    }

  GetCid () ? os << " CID(" << +m_srcdstContextId << ")" : os << " CID(0)";

  GetSac () ? os << " SAC(1)" : os << " SAC(0)";
  os << " SAM (" << GetSam () << ")";

  GetM () ? os << " M(1)" : os << " M(0)";
  GetDac () ? os << " DAC(1)" : os << " DAC(0)";
  os << " DAM (" << GetDam () << ")";
}

uint32_t SixLowPanIphc::GetSerializedSize () const
{
  uint32_t serializedSize = 2;

  if ( GetCid () )
    {
      serializedSize++;
    }
  switch ( GetTf () )
    {
    case TF_FULL:
      serializedSize += 4;
      break;
    case TF_DSCP_ELIDED:
      serializedSize += 3;
      break;
    case TF_FL_ELIDED:
      serializedSize++;
      break;
    default:
      break;
    }
  if ( GetNh () == false )
    {
      serializedSize++;
    }
  if ( GetHlim () == HLIM_INLINE)
    {
      serializedSize++;
    }
  switch (GetSam () )
    {
    case HC_INLINE:
      if ( GetSac () == false )
        {
          serializedSize += 16;
        }
      break;
    case HC_COMPR_64:
      serializedSize += 8;
      break;
    case HC_COMPR_16:
      serializedSize += 2;
      break;
    case HC_COMPR_0:
    default:
      break;
    }
  if ( GetM () == false)
    {
      switch (GetDam () )
        {
        case HC_INLINE:
          if ( GetDac () == false )
            {
              serializedSize += 16;
            }
          break;
        case HC_COMPR_64:
          serializedSize += 8;
          break;
        case HC_COMPR_16:
          serializedSize += 2;
          break;
        case HC_COMPR_0:
        default:
          break;
        }
    }
  else
    {
      switch (GetDam () )
        {
        case HC_INLINE:
          if ( GetDac () == false )
            {
              serializedSize += 16;
            }
          else
            {
              serializedSize += 6;
            }
          break;
        case HC_COMPR_64:
          if ( GetDac () == false )
            {
              serializedSize += 6;
            }
          break;
        case HC_COMPR_16:
          if ( GetDac () == false )
            {
              serializedSize += 4;
            }
          break;
        case HC_COMPR_0:
        default:
          if ( GetDac () == false )
            {
              serializedSize++;
            }
          break;
        }
    }



  return serializedSize;
}

void SixLowPanIphc::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteHtonU16 (m_baseFormat);

  if ( GetCid () )
    {
      i.WriteU8 (m_srcdstContextId);
    }
  // Traffic Class and Flow Label
  switch ( GetTf () )
    {
      uint8_t temp;
    case TF_FULL:
      temp = (m_ecn << 6) | m_dscp;
      i.WriteU8 (temp);
      temp = m_flowLabel >> 16;
      i.WriteU8 (temp);
      temp = (m_flowLabel >> 8) & 0xff;
      i.WriteU8 (temp);
      temp = m_flowLabel & 0xff;
      i.WriteU8 (temp);
      break;
    case TF_DSCP_ELIDED:
      temp = (m_ecn << 6) | (m_flowLabel >> 16 );
      i.WriteU8 (temp);
      temp = (m_flowLabel >> 8) & 0xff;
      i.WriteU8 (temp);
      temp = m_flowLabel & 0xff;
      i.WriteU8 (temp);
      break;
    case TF_FL_ELIDED:
      temp = (m_ecn << 6) | m_dscp;
      i.WriteU8 (temp);
      break;
    default:
      break;
    }
  // Next Header
  if ( GetNh () == false )
    {
      i.WriteU8 (m_nextHeader);
    }
  // Hop Limit
  if ( GetHlim () == HLIM_INLINE )
    {
      i.WriteU8 (m_hopLimit);
    }
  // Source Address
  switch (GetSam () )
    {
    case HC_INLINE:
      if ( GetSac () == false )
        {
          i.Write (m_srcInlinePart, 16);
        }
      break;
    case HC_COMPR_64:
      i.Write (m_srcInlinePart, 8);
      break;
    case HC_COMPR_16:
      i.Write (m_srcInlinePart, 2);
      break;
    case HC_COMPR_0:
    default:
      break;
    }
  // Destination Address
  if ( GetM () == false)
    {
      // unicast
      switch (GetDam () )
        {
        case HC_INLINE:
          i.Write (m_dstInlinePart, 16);
          break;
        case HC_COMPR_64:
          i.Write (m_dstInlinePart, 8);
          break;
        case HC_COMPR_16:
          i.Write (m_dstInlinePart, 2);
          break;
        case HC_COMPR_0:
        default:
          break;
        }
    }
  else
    {
      // multicast
      switch (GetDam () )
        {
        case HC_INLINE:
          i.Write (m_dstInlinePart, 16);
          break;
        case HC_COMPR_64:
          i.Write (m_dstInlinePart, 6);
          break;
        case HC_COMPR_16:
          i.Write (m_dstInlinePart, 4);
          break;
        case HC_COMPR_0:
          i.Write (m_dstInlinePart, 1);
          break;
       default:
          break;
        }
    }
}

uint32_t SixLowPanIphc::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_baseFormat = i.ReadNtohU16 ();

  if ( GetCid () )
    {
      m_srcdstContextId = i.ReadU8 ();
    }
  else
    {
      m_srcdstContextId = 0;
    }
  // Traffic Class and Flow Label
  switch ( GetTf () )
    {
      uint8_t temp;
    case TF_FULL:
      temp = i.ReadU8 ();
      m_ecn = temp >> 6;
      m_dscp = temp & 0x3F;
      temp = i.ReadU8 ();
      m_flowLabel = temp;
      temp = i.ReadU8 ();
      m_flowLabel = (m_flowLabel << 8) | temp;
      temp = i.ReadU8 ();
      m_flowLabel = (m_flowLabel << 8) | temp;
      break;
    case TF_DSCP_ELIDED:
      temp = i.ReadU8 ();
      m_ecn = temp >> 6;
      m_flowLabel = temp & 0x3F;
      temp = i.ReadU8 ();
      m_flowLabel = (m_flowLabel << 8) | temp;
      temp = i.ReadU8 ();
      m_flowLabel = (m_flowLabel << 8) | temp;
      break;
    case TF_FL_ELIDED:
      temp = i.ReadU8 ();
      m_ecn = temp >> 6;
      m_dscp = temp & 0x3F;
      break;
    default:
      break;
    }
  // Next Header
  if ( GetNh () == false )
    {
      m_nextHeader = i.ReadU8 ();
    }
  // Hop Limit
  switch ( GetHlim () )
    {
    case HLIM_INLINE:
      m_hopLimit = i.ReadU8 ();
      break;
    case HLIM_COMPR_1:
      m_hopLimit = 1;
      break;
    case HLIM_COMPR_64:
      m_hopLimit = 64;
      break;
    case HLIM_COMPR_255:
    default:
      m_hopLimit = 255;
      break;

    }
  // Source Address
  memset (m_srcInlinePart, 0x00, sizeof (m_srcInlinePart));
  switch (GetSam () )
    {
    case HC_INLINE:
      if ( GetSac () == false )
        {
          i.Read (m_srcInlinePart, 16);
        }
      break;
    case HC_COMPR_64:
      i.Read (m_srcInlinePart, 8);
      break;
    case HC_COMPR_16:
      i.Read (m_srcInlinePart, 2);
      break;
    case HC_COMPR_0:
    default:
      break;
    }

  // Destination Address
  memset (m_dstInlinePart, 0x00, sizeof (m_dstInlinePart));
  if ( GetM () == false)
    {
      // unicast
      switch (GetDam () )
        {
        case HC_INLINE:
          i.Read (m_dstInlinePart, 16);
          break;
        case HC_COMPR_64:
          i.Read (m_dstInlinePart, 8);
          break;
        case HC_COMPR_16:
          i.Read (m_dstInlinePart, 2);
          break;
        case HC_COMPR_0:
        default:
          break;
        }
    }
  else
    {
      // multicast
      switch (GetDam () )
        {
        case HC_INLINE:
          i.Read (m_dstInlinePart, 16);
          break;
        case HC_COMPR_64:
          i.Read (m_dstInlinePart, 6);
          break;
        case HC_COMPR_16:
          i.Read (m_dstInlinePart, 4);
          break;
        case HC_COMPR_0:
          i.Read (m_dstInlinePart, 1);
          break;
        default:
          break;
        }
    }

  return GetSerializedSize ();
}

void SixLowPanIphc::SetTf (TrafficClassFlowLabel_e tfField)
{
  uint16_t field = tfField;
  m_baseFormat |= (field << 11);
}

SixLowPanIphc::TrafficClassFlowLabel_e SixLowPanIphc::GetTf (void) const
{
  return TrafficClassFlowLabel_e ((m_baseFormat >> 11) & 0x3);
}

void SixLowPanIphc::SetNh (bool nhField)
{
  uint16_t field = nhField;
  m_baseFormat |= (field << 10);
}

bool SixLowPanIphc::GetNh (void) const
{
  return ((m_baseFormat >> 10) & 0x1);
}

void SixLowPanIphc::SetHlim (Hlim_e hlimField)
{
  uint16_t field = hlimField;
  m_baseFormat |= (field << 8);
}

SixLowPanIphc::Hlim_e SixLowPanIphc::GetHlim (void) const
{
  return Hlim_e ((m_baseFormat >> 8) & 0x3);
}

void SixLowPanIphc::SetCid (bool cidField)
{
  uint16_t field = cidField;
  m_baseFormat |= (field << 7);
}

bool SixLowPanIphc::GetCid (void) const
{
  return ((m_baseFormat >> 7) & 0x1);
}

void SixLowPanIphc::SetSac (bool sacField)
{
  uint16_t field = sacField;
  m_baseFormat |= (field << 6);
}

bool SixLowPanIphc::GetSac (void) const
{
  return ((m_baseFormat >> 6) & 0x1);
}

void SixLowPanIphc::SetSam (HeaderCompression_e samField)
{
  uint16_t field = samField;
  m_baseFormat |= (field << 4);
}

SixLowPanIphc::HeaderCompression_e SixLowPanIphc::GetSam (void) const
{
  return HeaderCompression_e ((m_baseFormat >> 4) & 0x3);
}

const uint8_t* SixLowPanIphc::GetSrcInlinePart (void) const
{
  return m_srcInlinePart;
}

void SixLowPanIphc::SetSrcInlinePart (uint8_t srcInlinePart[16], uint8_t size)
{
  NS_ASSERT_MSG (size <= 16,  "Src inline part too large");

  memcpy (m_srcInlinePart, srcInlinePart, size);
  return;
}

void SixLowPanIphc::SetM (bool mField)
{
  uint16_t field = mField;
  m_baseFormat |= (field << 3);
}

bool SixLowPanIphc::GetM (void) const
{
  return ((m_baseFormat >> 3) & 0x1);
}

void SixLowPanIphc::SetDac (bool dacField)
{
  uint16_t field = dacField;
  m_baseFormat |= (field << 2);
}

bool SixLowPanIphc::GetDac (void) const
{
  return ((m_baseFormat >> 2) & 0x1);
}

void SixLowPanIphc::SetDam (HeaderCompression_e damField)
{
  uint16_t field = damField;
  m_baseFormat |= field;
}

SixLowPanIphc::HeaderCompression_e SixLowPanIphc::GetDam (void) const
{
  return HeaderCompression_e (m_baseFormat & 0x3);
}

const uint8_t* SixLowPanIphc::GetDstInlinePart (void) const
{
  return m_dstInlinePart;
}

void SixLowPanIphc::SetDstInlinePart (uint8_t dstInlinePart[16], uint8_t size)
{
  NS_ASSERT_MSG (size <= 16,  "Dst inline part too large");

  memcpy (m_dstInlinePart, dstInlinePart, size);
  return;
}

void SixLowPanIphc::SetSrcContextId (uint8_t srcContextId)
{
  NS_ASSERT_MSG (srcContextId < 16,  "Src Context ID too large");
  m_srcdstContextId |= srcContextId << 4;
}

uint8_t SixLowPanIphc::GetSrcContextId (void) const
{
  return ( m_srcdstContextId >> 4);
}

void SixLowPanIphc::SetDstContextId (uint8_t dstContextId)
{
  NS_ASSERT_MSG (dstContextId < 16,  "Dst Context ID too large");
  m_srcdstContextId |= (dstContextId & 0xF);
}

uint8_t SixLowPanIphc::GetDstContextId (void) const
{
  return (m_srcdstContextId & 0xF);
}

void SixLowPanIphc::SetEcn (uint8_t ecn)
{
  NS_ASSERT_MSG (ecn < 4,  "ECN too large");
  m_ecn = ecn;
}

uint8_t SixLowPanIphc::GetEcn (void) const
{
  return m_ecn;
}

void SixLowPanIphc::SetDscp (uint8_t dscp)
{
  NS_ASSERT_MSG (dscp < 64,  "DSCP too large");
  m_dscp = dscp;
}

uint8_t SixLowPanIphc::GetDscp (void) const
{
  return m_dscp;
}

void SixLowPanIphc::SetFlowLabel (uint32_t flowLabel)
{
  NS_ASSERT_MSG (flowLabel < 0x100000,  "Flow Label too large");
  m_flowLabel = flowLabel;
}

uint32_t SixLowPanIphc::GetFlowLabel (void) const
{
  return m_flowLabel;
}

void SixLowPanIphc::SetNextHeader (uint8_t nextHeader)
{
  m_nextHeader = nextHeader;
}

uint8_t SixLowPanIphc::GetNextHeader (void) const
{
  return m_nextHeader;
}

void SixLowPanIphc::SetHopLimit (uint8_t hopLimit)
{
  m_hopLimit = hopLimit;
}

uint8_t SixLowPanIphc::GetHopLimit (void) const
{
  return m_hopLimit;
}

std::ostream & operator << (std::ostream & os, const SixLowPanIphc & h)
{
  h.Print (os);
  return os;
}

/*
 * SixLowPanNhcExtensionHeader
 */
NS_OBJECT_ENSURE_REGISTERED (SixLowPanNhcExtension);

SixLowPanNhcExtension::SixLowPanNhcExtension ()
{
  // 1110 xxxx
  m_nhcExtensionHeader = 0xE0;
  m_nhcNextHeader = 0;
  m_nhcBlobLength = 0;
}

TypeId SixLowPanNhcExtension::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanNhcExtension")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanNhcExtension> ();
  return tid;
}

TypeId SixLowPanNhcExtension::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanNhcExtension::Print (std::ostream & os) const
{
  os << "Compression kind: " << +m_nhcExtensionHeader << " Size: " << GetSerializedSize ();
}

uint32_t SixLowPanNhcExtension::GetSerializedSize () const
{
  uint32_t serializedSize = 2;
  if ( GetNh () == false )
    {
      serializedSize++;
    }
  return serializedSize + m_nhcBlobLength;
}

void SixLowPanNhcExtension::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_nhcExtensionHeader);
  if ( GetNh () == false )
    {
      i.WriteU8 (m_nhcNextHeader);
    }
  i.WriteU8 (m_nhcBlobLength);
  i.Write (m_nhcBlob, m_nhcBlobLength);
}

uint32_t SixLowPanNhcExtension::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_nhcExtensionHeader = i.ReadU8 ();
  if ( GetNh () == false )
    {
      m_nhcNextHeader = i.ReadU8 ();
    }
  m_nhcBlobLength = i.ReadU8 ();
  i.Read (m_nhcBlob, m_nhcBlobLength);

  return GetSerializedSize ();
}

SixLowPanDispatch::NhcDispatch_e
SixLowPanNhcExtension::GetNhcDispatchType (void) const
{
  return SixLowPanDispatch::LOWPAN_NHC;
}

void SixLowPanNhcExtension::SetEid (Eid_e extensionHeaderType)
{
  uint8_t field = extensionHeaderType;
  m_nhcExtensionHeader |= (field << 1);
}

SixLowPanNhcExtension::Eid_e SixLowPanNhcExtension::GetEid (void) const
{
  return Eid_e ((m_nhcExtensionHeader >> 1) & 0x7);
}

void SixLowPanNhcExtension::SetNextHeader (uint8_t nextHeader)
{
  m_nhcNextHeader = nextHeader;
}

uint8_t SixLowPanNhcExtension::GetNextHeader (void) const
{
  return m_nhcNextHeader;
}

void SixLowPanNhcExtension::SetNh (bool nhField)
{
  uint8_t field = nhField;
  m_nhcExtensionHeader |= field;
}

bool SixLowPanNhcExtension::GetNh (void) const
{
  return m_nhcExtensionHeader & 0x01;
}

void SixLowPanNhcExtension::SetBlob (const uint8_t* blob, uint32_t size)
{
  NS_ASSERT_MSG ( size < 255, "Buffer too long" );

  m_nhcBlobLength = size;
  std::memcpy (m_nhcBlob, blob, size);
}

uint32_t SixLowPanNhcExtension::CopyBlob (uint8_t* blob, uint32_t size) const
{
  NS_ASSERT_MSG ( size > m_nhcBlobLength, "Buffer too short" );

  std::memcpy (blob, m_nhcBlob, m_nhcBlobLength);
  return m_nhcBlobLength;
}

std::ostream & operator << (std::ostream & os, const SixLowPanNhcExtension & h)
{
  h.Print (os);
  return os;
}

/*
 * SixLowPanUdpNhcExtension
 */
NS_OBJECT_ENSURE_REGISTERED (SixLowPanUdpNhcExtension);

SixLowPanUdpNhcExtension::SixLowPanUdpNhcExtension ()
{
  // 1111 0xxx
  m_baseFormat = 0xF0;
  m_checksum = 0;
  m_srcPort = 0;
  m_dstPort = 0;
}

TypeId SixLowPanUdpNhcExtension::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanUdpNhcExtension")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanUdpNhcExtension> ();
  return tid;
}

TypeId SixLowPanUdpNhcExtension::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanUdpNhcExtension::Print (std::ostream & os) const
{
  os << "Compression kind: " << +m_baseFormat;
}

uint32_t SixLowPanUdpNhcExtension::GetSerializedSize () const
{
  uint32_t serializedSize = 1;
  if ( !GetC () )
    {
      serializedSize += 2;
    }
  switch (GetPorts ())
    {
    case PORTS_INLINE:
      serializedSize += 4;
      break;
    case PORTS_ALL_SRC_LAST_DST:
    case PORTS_LAST_SRC_ALL_DST:
      serializedSize += 3;
      break;
    case PORTS_LAST_SRC_LAST_DST:
      serializedSize += 1;
      break;
    default:
      break;
    }
  return serializedSize;
}

void SixLowPanUdpNhcExtension::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_baseFormat);
  uint8_t temp;

  // Ports
  switch ( GetPorts () )
    {
    case PORTS_INLINE:
      i.WriteHtonU16 (m_srcPort);
      i.WriteHtonU16 (m_dstPort);
      break;
    case PORTS_ALL_SRC_LAST_DST:
      i.WriteHtonU16 (m_srcPort);
      i.WriteU8 (m_dstPort & 0xff);
      break;
    case PORTS_LAST_SRC_ALL_DST:
      i.WriteU8 (m_srcPort & 0xff);
      i.WriteHtonU16 (m_dstPort);
      break;
    case PORTS_LAST_SRC_LAST_DST:
      temp = ((m_srcPort & 0xf) << 4) | (m_dstPort & 0xf);
      i.WriteU8 (temp);
      break;
    default:
      break;
    }

  // Checksum
  if ( !GetC () )
    {
      i.WriteU16 (m_checksum);
    }

}

uint32_t SixLowPanUdpNhcExtension::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_baseFormat = i.ReadU8 ();
  uint8_t temp;

  // Ports
  switch ( GetPorts () )
    {
    case PORTS_INLINE:
      m_srcPort = i.ReadNtohU16 ();
      m_dstPort = i.ReadNtohU16 ();
      break;
    case PORTS_ALL_SRC_LAST_DST:
      m_srcPort = i.ReadNtohU16 ();
      m_dstPort = i.ReadU8 ();
      break;
    case PORTS_LAST_SRC_ALL_DST:
      m_srcPort = i.ReadU8 ();
      m_dstPort = i.ReadNtohU16 ();
      break;
    case PORTS_LAST_SRC_LAST_DST:
      temp = i.ReadU8 ();
      m_srcPort = temp >> 4;
      m_dstPort = temp & 0xf;
      break;
    default:
      break;
    }

  // Checksum
  if ( !GetC () )
    {
      m_checksum = i.ReadU16 ();
    }

  return GetSerializedSize ();
}

SixLowPanDispatch::NhcDispatch_e
SixLowPanUdpNhcExtension::GetNhcDispatchType (void) const
{
  return SixLowPanDispatch::LOWPAN_UDPNHC;
}

void SixLowPanUdpNhcExtension::SetPorts (Ports_e ports)
{
  uint16_t field = ports;
  m_baseFormat |= field;
}

SixLowPanUdpNhcExtension::Ports_e SixLowPanUdpNhcExtension::GetPorts (void) const
{
  return Ports_e (m_baseFormat & 0x3);
}

void SixLowPanUdpNhcExtension::SetSrcPort (uint16_t srcport)
{
  m_srcPort = srcport;
}

uint16_t SixLowPanUdpNhcExtension::GetSrcPort (void) const
{
  return m_srcPort;
}

void SixLowPanUdpNhcExtension::SetDstPort (uint16_t dstport)
{
  m_dstPort = dstport;
}

uint16_t SixLowPanUdpNhcExtension::GetDstPort (void) const
{
  return m_dstPort;
}

void SixLowPanUdpNhcExtension::SetC (bool cField)
{
  uint16_t field = cField;
  m_baseFormat |= (field << 2);
}

bool SixLowPanUdpNhcExtension::GetC (void) const
{
  return ((m_baseFormat >> 2) & 0x1);
}

void SixLowPanUdpNhcExtension::SetChecksum (uint16_t checksum)
{
  m_checksum = checksum;
}

uint16_t SixLowPanUdpNhcExtension::GetChecksum (void) const
{
  return m_checksum;
}

std::ostream & operator << (std::ostream & os, const SixLowPanUdpNhcExtension & h)
{
  h.Print (os);
  return os;
}

/*
 * SixLowPanBc0
 */
NS_OBJECT_ENSURE_REGISTERED (SixLowPanBc0);

SixLowPanBc0::SixLowPanBc0 ()
{
  m_seqNumber = 66;
}

TypeId SixLowPanBc0::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanBc0")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanBc0> ();
  return tid;
}

TypeId SixLowPanBc0::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanBc0::Print (std::ostream & os) const
{
  os << "Sequence number: " << +m_seqNumber;
}

uint32_t SixLowPanBc0::GetSerializedSize () const
{
  return 2;
}

void SixLowPanBc0::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (0x50);
  i.WriteU8 (m_seqNumber);

}

uint32_t SixLowPanBc0::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t dispatch = i.ReadU8 ();

  if (dispatch != 0x50)
    {
      return 0;
    }

  m_seqNumber = i.ReadU8 ();

  return GetSerializedSize ();
}

void SixLowPanBc0::SetSequenceNumber (uint8_t seqNumber)
{
  m_seqNumber = seqNumber;
}

uint8_t SixLowPanBc0::GetSequenceNumber (void) const
{
  return m_seqNumber;
}

std::ostream & operator << (std::ostream & os, const SixLowPanBc0 & h)
{
  h.Print (os);
  return os;
}

/*
 * SixLowPanMesh
 */
NS_OBJECT_ENSURE_REGISTERED (SixLowPanMesh);

SixLowPanMesh::SixLowPanMesh ()
{
  m_hopsLeft = 0;
  m_src = Address ();
  m_dst = Address ();
  m_v = false;
  m_f = false;
}

TypeId SixLowPanMesh::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanMesh")
    .SetParent<Header> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanMesh> ();
  return tid;
}

TypeId SixLowPanMesh::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void SixLowPanMesh::Print (std::ostream & os) const
{
  os << "Hops left: " << +m_hopsLeft << ", src: ";
  if (Mac64Address::IsMatchingType (m_src))
    {
      os << Mac64Address::ConvertFrom (m_src);
    }
  else
    {
      os << Mac16Address::ConvertFrom (m_src);
    }
  os << ", dst: ";
  if (Mac64Address::IsMatchingType (m_dst))
    {
      os << Mac64Address::ConvertFrom (m_dst);
    }
  else
    {
      os << Mac16Address::ConvertFrom (m_dst);
    }
}

uint32_t SixLowPanMesh::GetSerializedSize () const
{
  uint32_t serializedSize = 1;

  if (m_hopsLeft >= 0xF)
    {
      serializedSize++;
    }

  if (m_v)
    {
      serializedSize += 2;
    }
  else
    {
      serializedSize += 8;
    }

  if (m_f)
    {
      serializedSize += 2;
    }
  else
    {
      serializedSize += 8;
    }

  return serializedSize;
}

void SixLowPanMesh::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  uint8_t dispatch = 0x80;

  if (m_v)
    {
      dispatch |= 0x20;
    }
  if (m_f)
    {
      dispatch |= 0x10;
    }

  if (m_hopsLeft < 0xF)
    {
      dispatch |= m_hopsLeft;
      i.WriteU8 (dispatch);
    }
  else
    {
      dispatch |= 0xF;
      i.WriteU8 (dispatch);
      i.WriteU8 (m_hopsLeft);
    }

  uint8_t buffer[8];

  m_src.CopyTo (buffer);
  if (m_v)
    {
      i.Write (buffer, 2);
    }
  else
    {
      i.Write (buffer, 8);
    }

  m_dst.CopyTo (buffer);
  if (m_f)
    {
      i.Write (buffer, 2);
    }
  else
    {
      i.Write (buffer, 8);
    }
}

uint32_t SixLowPanMesh::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t temp = i.ReadU8 ();

  if ((temp & 0xC0) != 0x80)
    {
      return 0;
    }

  m_v = temp & 0x20;
  m_f = temp & 0x10;
  m_hopsLeft = temp & 0xF;

  if (m_hopsLeft == 0xF)
    {
      m_hopsLeft = i.ReadU8 ();
    }

  uint8_t buffer[8];
  uint8_t addrSize;

  if (m_v)
    {
      addrSize = 2;
    }
  else
    {
      addrSize = 8;
    }
  i.Read (buffer, addrSize);
  m_src.CopyFrom (buffer, addrSize);

  if (m_f)
    {
      addrSize = 2;
    }
  else
    {
      addrSize = 8;
    }
  i.Read (buffer, addrSize);
  m_dst.CopyFrom (buffer, addrSize);

  return GetSerializedSize ();
}

void SixLowPanMesh::SetOriginator (Address originator)
{
  if (Mac64Address::IsMatchingType (originator))
    {
      m_v = false;
    }
  else if (Mac16Address::IsMatchingType (originator))
    {
      m_v = true;
    }
  else
    {
      NS_ABORT_MSG ("SixLowPanMesh::SetOriginator - incompatible address");
    }

  m_src = originator;
}

Address SixLowPanMesh::GetOriginator (void) const
{
  return m_src;
}

void SixLowPanMesh::SetFinalDst (Address finalDst)
{
  if (Mac64Address::IsMatchingType (finalDst))
    {
      m_f = false;
    }
  else if (Mac16Address::IsMatchingType (finalDst))
    {
      m_f = true;
    }
  else
    {
      NS_ABORT_MSG ("SixLowPanMesh::SetFinalDst - incompatible address");
    }

  m_dst = finalDst;
}

Address SixLowPanMesh::GetFinalDst (void) const
{
  return m_dst;
}

void SixLowPanMesh::SetHopsLeft (uint8_t hopsLeft)
{
  m_hopsLeft = hopsLeft;
}

uint8_t SixLowPanMesh::GetHopsLeft (void) const
{
  return m_hopsLeft;
}

std::ostream & operator << (std::ostream & os, const SixLowPanMesh & h)
{
  h.Print (os);
  return os;
}

}

