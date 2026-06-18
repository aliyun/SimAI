/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
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
#include "tcp-error-model.h"
#include "ns3/ipv4-header.h"
#include "ns3/packet.h"
#include "ns3/log.h"
namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpGeneralErrorModel");

NS_OBJECT_ENSURE_REGISTERED (TcpGeneralErrorModel);

TypeId
TcpGeneralErrorModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpGeneralErrorModel")
    .SetParent<ErrorModel> ()
  ;
  return tid;
}

TcpGeneralErrorModel::TcpGeneralErrorModel ()
{
  NS_LOG_FUNCTION (this);
}

bool
TcpGeneralErrorModel::DoCorrupt (Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this << p);

  if (!IsEnabled ())
    {
      return false;
    }

  Ipv4Header ipHeader;
  TcpHeader tcpHeader;

  p->RemoveHeader (ipHeader);
  p->RemoveHeader (tcpHeader);

  bool toDrop = ShouldDrop (ipHeader, tcpHeader, p->GetSize ());

  if (toDrop && ! m_dropCallback.IsNull ())
    {
      m_dropCallback (ipHeader, tcpHeader, p);
    }

  p->AddHeader (tcpHeader);
  p->AddHeader (ipHeader);

  return toDrop;
}

NS_OBJECT_ENSURE_REGISTERED (TcpSeqErrorModel);

TypeId
TcpSeqErrorModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpSeqErrorModel")
    .SetParent<TcpGeneralErrorModel> ()
    .AddConstructor<TcpSeqErrorModel> ()
  ;
  return tid;
}

bool
TcpSeqErrorModel::ShouldDrop (const Ipv4Header &ipHeader, const TcpHeader &tcpHeader,
                              uint32_t packetSize)
{
  NS_LOG_FUNCTION (this << ipHeader << tcpHeader);

  bool toDrop = false;

  if (m_seqToKill.begin() != m_seqToKill.end() && packetSize != 0)
    {
      SequenceNumber32 toKill = m_seqToKill.front();
      NS_LOG_INFO ("Analyzing seq=" << tcpHeader.GetSequenceNumber () <<
                   " killing=" << toKill);
      if (tcpHeader.GetSequenceNumber() == toKill)
        {
          NS_LOG_INFO ("segment " << toKill << " dropped");
          toDrop = true;
          m_seqToKill.pop_front();
        }
    }

  return toDrop;
}

void
TcpSeqErrorModel::DoReset()
{
  m_seqToKill.erase (m_seqToKill.begin(), m_seqToKill.end());
}

NS_OBJECT_ENSURE_REGISTERED (TcpFlagErrorModel);

TypeId
TcpFlagErrorModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpFlagErrorModel")
    .SetParent<TcpGeneralErrorModel> ()
    .AddConstructor<TcpFlagErrorModel> ()
  ;
  return tid;
}

TcpFlagErrorModel::TcpFlagErrorModel ()
  : TcpGeneralErrorModel (),
  m_flagsToKill (TcpHeader::NONE),
  m_killNumber (0)
{
}

bool
TcpFlagErrorModel::ShouldDrop (const Ipv4Header &ipHeader, const TcpHeader &tcpHeader,
                               uint32_t packetSize)
{
  NS_LOG_FUNCTION (this << ipHeader << tcpHeader);

  (void) packetSize;

  bool toDrop = false;

  if ((tcpHeader.GetFlags () & m_flagsToKill) == m_flagsToKill)
    {
      if (m_killNumber > 0)
        {
          m_killNumber--;
          if (m_killNumber > 0)
            {
              toDrop = true;
            }
        }
      else if (m_killNumber < 0)
        {
          toDrop = true;
        }
    }

  return toDrop;
}

void
TcpFlagErrorModel::DoReset (void)
{
  m_flagsToKill = TcpHeader::NONE;
  m_killNumber = 0;
}

} //namespace ns3

