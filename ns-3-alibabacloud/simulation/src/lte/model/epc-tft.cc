/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */




#include "epc-tft.h"
#include "ns3/abort.h"
#include "ns3/log.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EpcTft");

/**
 * Output stream operator for EPC TFT direction
 *
 * \param os output stream
 * \param d EPC TFT direction
 * \return ostream
 */
std::ostream& operator<< (std::ostream& os, EpcTft::Direction& d)
{
  switch (d)
  {
    case EpcTft::DOWNLINK:
      os << "DOWNLINK";
      break;
    case EpcTft::UPLINK:
      os << "UPLINK";
      break;
    default:
      os << "BIDIRECTIONAL";
      break;
  }
  return os;
}


/**
 * Output stream for EPC TFT packet filter
 *
 * \param os output stream
 * \param f EPC TFT packet filter
 * \return ostream
 */
std::ostream& operator<< (std::ostream& os, EpcTft::PacketFilter& f)
{
  os << " direction: " << f.direction
      << " remoteAddress: "  << f.remoteAddress
      << " remoteMask: "  << f.remoteMask
      << " remoteIpv6Address: "  << f.remoteIpv6Address
      << " remoteIpv6Prefix: "  << f.remoteIpv6Prefix
      << " localAddress: "  << f.localAddress
      << " localMask: "  << f.localMask
      << " localIpv6Address: "  << f.localIpv6Address
      << " localIpv6Prefix: "  << f.localIpv6Prefix
      << " remotePortStart: "   << f.remotePortStart
      << " remotePortEnd: "   << f.remotePortEnd
      << " localPortStart: "   << f.localPortStart
      << " localPortEnd: "   << f.localPortEnd
      << " typeOfService: 0x" << std::hex << (uint16_t) f.typeOfService << std::dec
      << " typeOfServiceMask: 0x" << std::hex << (uint16_t) f.typeOfServiceMask << std::dec;
  return os;
}

EpcTft::PacketFilter::PacketFilter ()
: precedence (255),
  direction (BIDIRECTIONAL),
  remoteMask ("0.0.0.0"),
  localMask ("0.0.0.0"),
  remotePortStart (0),
  remotePortEnd (65535),
  localPortStart (0),
  localPortEnd (65535),
  typeOfService (0),
  typeOfServiceMask (0)
{
  NS_LOG_FUNCTION (this);
}

bool 
EpcTft::PacketFilter::Matches (Direction d,
                               Ipv4Address ra,
                               Ipv4Address la,
                               uint16_t rp,
                               uint16_t lp,
                               uint8_t tos)
{
  NS_LOG_FUNCTION (this << d << ra << la << rp << lp << (uint16_t) tos);
  if (d & direction)
    {
      NS_LOG_LOGIC ("d matches");
      if (remoteMask.IsMatch (remoteAddress, ra))
        {
          NS_LOG_LOGIC ("ra matches");
          if (localMask.IsMatch (localAddress, la))
            {
              NS_LOG_LOGIC ("la matches");
              if (remotePortStart <= rp && rp <= remotePortEnd)
                {
                  NS_LOG_LOGIC ("rp matches");
                  if (localPortStart <= lp && lp <= localPortEnd)
                    {
                      NS_LOG_LOGIC ("lp matches");
                      if ((tos & typeOfServiceMask) == (typeOfService & typeOfServiceMask))
                        {
                          NS_LOG_LOGIC ("tos matches --> have match!");
                          return true;
                        }
                      else
                        {
                          NS_LOG_LOGIC ("tos doesn't match: tos=" << tos << " f.tos=" << typeOfService << " f.tosmask=" << typeOfServiceMask);
                        }
                    }
                  else
                    {
                      NS_LOG_LOGIC ("lp doesn't match: lp=" << lp << " f.lps=" << localPortStart << " f.lpe=" << localPortEnd);
                    }
                }
              else
                {
                  NS_LOG_LOGIC ("rp doesn't match: rp=" << rp << " f.rps=" << remotePortStart << " f.lpe=" << remotePortEnd);
                }
            }
          else
            {
              NS_LOG_LOGIC ("la doesn't match: la=" << la << " f.la=" << localAddress << " f.lmask=" << localMask);
            }
        }
      else
        {
          NS_LOG_LOGIC ("ra doesn't match: ra=" << ra << " f.ra=" << remoteAddress << " f.rmask=" << remoteMask);
        }
    }
  else
    {
      NS_LOG_LOGIC ("d doesn't match: d=0x" << std::hex << d << " f.d=0x" << std::hex << direction << std::dec);
    }
  return false;
}

bool 
EpcTft::PacketFilter::Matches (Direction d,
                               Ipv6Address ra,
                               Ipv6Address la,
                               uint16_t rp,
                               uint16_t lp,
                               uint8_t tos)
{
  NS_LOG_FUNCTION (this << d << ra << la << rp << lp << (uint16_t) tos);
  if (d & direction)
    {
      NS_LOG_LOGIC ("d matches");
      if (remoteIpv6Prefix.IsMatch (remoteIpv6Address, ra))
        {
          NS_LOG_LOGIC ("ra matches");
          if (localIpv6Prefix.IsMatch (localIpv6Address, la))
            {
              NS_LOG_LOGIC ("la matches");
              if (remotePortStart <= rp && rp <= remotePortEnd)
                {
                  NS_LOG_LOGIC ("rp matches");
                  if (localPortStart <= lp && lp <= localPortEnd)
                    {
                      NS_LOG_LOGIC ("lp matches");
                      if ((tos & typeOfServiceMask) == (typeOfService & typeOfServiceMask))
                        {
                          NS_LOG_LOGIC ("tos matches --> have match!");
                          return true;
                        }
                      else
                        {
                          NS_LOG_LOGIC ("tos doesn't match: tos=" << tos << " f.tos=" << typeOfService << " f.tosmask=" << typeOfServiceMask);
                        }
                    }
                  else
                    {
                      NS_LOG_LOGIC ("lp doesn't match: lp=" << lp << " f.lps=" << localPortStart << " f.lpe=" << localPortEnd);
                    }
                }
              else
                {
                  NS_LOG_LOGIC ("rp doesn't match: rp=" << rp << " f.rps=" << remotePortStart << " f.lpe=" << remotePortEnd);
                }
            }
          else
            {
              NS_LOG_LOGIC ("la doesn't match: la=" << la << " f.la=" << localIpv6Address << " f.lprefix=" << localIpv6Prefix);
            }
        }
      else
        {
          NS_LOG_LOGIC ("ra doesn't match: ra=" << ra << " f.ra=" << remoteIpv6Address << " f.rprefix=" << remoteIpv6Prefix);
        }
    }
  else
    {
      NS_LOG_LOGIC ("d doesn't match: d=0x" << std::hex << d << " f.d=0x" << std::hex << direction << std::dec);
    }
  return false;
}


Ptr<EpcTft> 
EpcTft::Default ()
{
  Ptr<EpcTft> tft = Create<EpcTft> ();
  EpcTft::PacketFilter defaultPacketFilter;
  tft->Add (defaultPacketFilter);
  return tft;
}


EpcTft::EpcTft ()
: m_numFilters (0)
{
  NS_LOG_FUNCTION (this);
}

uint8_t 
EpcTft::Add (PacketFilter f)
{
  NS_LOG_FUNCTION (this << f);
  NS_ABORT_IF (m_numFilters >= 16);

  std::list<PacketFilter>::iterator it;
  for (it = m_filters.begin ();
      (it != m_filters.end ()) && (it->precedence <= f.precedence);
      ++it)
    {
    }  
  m_filters.insert (it, f);  
  ++m_numFilters;
  return (m_numFilters - 1);
}

bool 
EpcTft::Matches (Direction direction,
                 Ipv4Address remoteAddress,
                 Ipv4Address localAddress,
                 uint16_t remotePort,
                 uint16_t localPort,
                 uint8_t typeOfService)
{
  NS_LOG_FUNCTION (this << direction << remoteAddress << localAddress << std::dec << remotePort << localPort << (uint16_t) typeOfService);
  for (std::list<PacketFilter>::iterator it = m_filters.begin ();
      it != m_filters.end ();
      ++it)
    {
      if (it->Matches (direction, remoteAddress, localAddress, remotePort, localPort, typeOfService))
        {
          return true;
        }
    }  
  return false;
}

bool 
EpcTft::Matches (Direction direction,
                 Ipv6Address remoteAddress,
                 Ipv6Address localAddress,
                 uint16_t remotePort,
                 uint16_t localPort,
                 uint8_t typeOfService)
{
  NS_LOG_FUNCTION (this << direction << remoteAddress << localAddress << std::dec << remotePort << localPort << (uint16_t) typeOfService);
  for (std::list<PacketFilter>::iterator it = m_filters.begin ();
      it != m_filters.end ();
      ++it)
    {
      if (it->Matches (direction, remoteAddress, localAddress, remotePort, localPort, typeOfService))
        {
          return true;
        }
    }  
  return false;
}

std::list<EpcTft::PacketFilter>
EpcTft::GetPacketFilters () const
{
  NS_LOG_FUNCTION (this);
  return m_filters;
};

} // namespace ns3
