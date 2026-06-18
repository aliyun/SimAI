/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Louis Pasteur University / Telecom Bretagne
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
 * Author: Angelos Chatzipapas <Angelos.CHATZIPAPAS@enst-bretagne.fr> /
 * <chatzipa@ceid.upatras.gr>
 */

#include "ns3/log.h"
#include "ns3/ipv6-address.h"
#include "ns3/node.h"
#include "ns3/mac48-address.h"
#include "ns3/simulator.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TestIpv6");

int 
main (int argc, char *argv[])
{
  LogComponentEnable ("TestIpv6", LOG_LEVEL_ALL);

  NS_LOG_INFO ("Test Ipv6");

  Mac48Address m_addresses[10];

  m_addresses[0] = ("00:00:00:00:00:01");
  m_addresses[1] = ("00:00:00:00:00:02");
  m_addresses[2] = ("00:00:00:00:00:03");
  m_addresses[3] = ("00:00:00:00:00:04");
  m_addresses[4] = ("00:00:00:00:00:05");
  m_addresses[5] = ("00:00:00:00:00:06");
  m_addresses[6] = ("00:00:00:00:00:07");
  m_addresses[7] = ("00:00:00:00:00:08");
  m_addresses[8] = ("00:00:00:00:00:09");
  m_addresses[9] = ("00:00:00:00:00:10");

  Ipv6Address prefix1 ("2001:1::");
  NS_LOG_INFO ("prefix = " << prefix1);
  for (uint32_t i = 0; i < 10; ++i)
    {
      NS_LOG_INFO ("address = " << m_addresses[i]);
      Ipv6Address ipv6address = Ipv6Address::MakeAutoconfiguredAddress (m_addresses[i], prefix1);
      NS_LOG_INFO ("address = " << ipv6address);
    }

  Ipv6Address prefix2 ("2002:1:1::");

  NS_LOG_INFO ("prefix = " << prefix2);
  for (uint32_t i = 0; i < 10; ++i)
    {
      Ipv6Address ipv6address = Ipv6Address::MakeAutoconfiguredAddress (m_addresses[i], prefix2);
      NS_LOG_INFO ("address = " << ipv6address);
    }
}

