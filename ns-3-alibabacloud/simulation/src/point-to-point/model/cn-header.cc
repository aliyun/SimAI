/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 New York University
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
 * Author: Adrian S.-W. Tam <adrian.sw.tam@gmail.com>
 */

#include <stdint.h>
#include <iostream>
#include "cn-header.h"
#include "ns3/buffer.h"
#include "ns3/address-utils.h"
#include "ns3/log.h"

NS_LOG_COMPONENT_DEFINE ("CnHeader");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (CnHeader);

CnHeader::CnHeader (const uint16_t fid, uint8_t qIndex, uint8_t ecnbits, uint16_t qfb, uint16_t total)
  : m_fid(fid), m_qIndex(qIndex), m_qfb(qfb), m_ecnBits(ecnbits), m_total(total)
{
  //NS_LOG_LOGIC("CN got the flow id " << std::hex << m_fid.hi << "+" << m_fid.lo << std::dec);
}

/*
CnHeader::CnHeader (const uint16_t fid, uint8_t qIndex, uint8_t qfb)
	: m_fid(fid), m_qIndex(qIndex), m_qfb(qfb), m_ecnBits(0)
{
	//NS_LOG_LOGIC("CN got the flow id " << std::hex << m_fid.hi << "+" << m_fid.lo << std::dec);
}
*/

CnHeader::CnHeader ()
  : m_fid(), m_qIndex(), m_qfb(0), m_ecnBits(0)
{}

CnHeader::~CnHeader ()
{}

void CnHeader::SetFlow (const uint16_t fid)
{
  m_fid = fid;
}

void CnHeader::SetQindex (const uint8_t qIndex)
{
	m_qIndex = qIndex;
}

void CnHeader::SetQfb (uint16_t q)
{
  m_qfb = q;
}

void CnHeader::SetTotal (uint16_t total)
{
	m_total = total;
}

void CnHeader::SetECNBits (const uint8_t ecnbits)
{
	m_ecnBits = ecnbits;
}


uint16_t CnHeader::GetFlow () const
{
  return m_fid;
}

uint8_t CnHeader::GetQindex () const
{
	return m_qIndex;
}

uint16_t CnHeader::GetQfb () const
{
  return m_qfb;
}

uint16_t CnHeader::GetTotal() const
{
	return m_total;
}

uint8_t CnHeader::GetECNBits() const
{
	return m_ecnBits;
}

void CnHeader::SetSeq (const uint32_t seq){
	m_seq = seq;
}

uint32_t CnHeader::GetSeq () const{
	return m_seq;
}
TypeId 
CnHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::CnHeader")
    .SetParent<Header> ()
    .AddConstructor<CnHeader> ()
    ;
  return tid;
}
TypeId 
CnHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
void CnHeader::Print (std::ostream &os) const
{
  //m_fid.Print(os);
  os << " qFb=" << (unsigned) m_qfb << "/" << (unsigned) m_total;
}
uint32_t CnHeader::GetSerializedSize (void)  const
{
  return 8;
}
void CnHeader::Serialize (Buffer::Iterator start)  const
{
  //uint64_t hibyte = m_fid.hi;
  //uint64_t lobyte = m_fid.lo;
  //lobyte = (lobyte & 0x00FFFFFFFFFFFFFFLLU) | (static_cast<uint64_t>(m_qfb)<<56);
  //uint32_t lobyte = (m_qIndex &  0x00FFFFFFLLU) | (static_cast<uint32_t>(m_qfb)<<24);
  //start.WriteU64 (hibyte);
  //start.WriteU64 (lobyte);
  start.WriteU8(m_qIndex);
  start.WriteU16(m_fid);
  start.WriteU8(m_ecnBits);
  start.WriteU16(m_qfb);
  start.WriteU16(m_total);
  //NS_LOG_LOGIC("CN Seriealized as " << std::hex << hibyte << "+" << lobyte << std::dec);
}

uint32_t CnHeader::Deserialize (Buffer::Iterator start)
{
  //uint64_t hibyte = start.ReadU64 ();
  //uint64_t lobyte = start.ReadU64 ();
  //NS_LOG_LOGIC("CN deseriealized as " << std::hex << hibyte << "+" << lobyte << std::dec);
  //m_qfb = static_cast<uint8_t>(lobyte>>56);
  //m_fid.hi = hibyte;
  //m_fid.lo = lobyte & 0x00FFFFFFFFFFFFFFLLU;
  
  //uint32_t lobyte = start.ReadU32();
  

  m_qIndex = start.ReadU8();
  m_fid = start.ReadU16();
  m_ecnBits = start.ReadU8();
  m_qfb = start.ReadU16();
  m_total = start.ReadU16();

  //m_qfb = static_cast<uint8_t>(lobyte>>24);
  //m_qIndex = lobyte & 0x00FFFFFFLLU;

  return GetSerializedSize ();
}


}; // namespace ns3
