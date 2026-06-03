/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
 * Copyright (c) 2013 University of Surrey
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 *          Konstantinos Katsaros <dinos.katsaros@gmail.com>
 */

#include "ns3/double.h"
#include "snr-tag.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (SnrTag);

TypeId
SnrTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SnrTag")
    .SetParent<Tag> ()
    .SetGroupName ("Wifi")
    .AddConstructor<SnrTag> ()
    .AddAttribute ("Snr", "The SNR of the last packet received",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&SnrTag::Get),
                   MakeDoubleChecker<double> ())
  ;
  return tid;
}

TypeId
SnrTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

SnrTag::SnrTag ()
  : m_snr (0)
{
}

uint32_t
SnrTag::GetSerializedSize (void) const
{
  return sizeof (double);
}

void
SnrTag::Serialize (TagBuffer i) const
{
  i.WriteDouble (m_snr);
}

void
SnrTag::Deserialize (TagBuffer i)
{
  m_snr = i.ReadDouble ();
}

void
SnrTag::Print (std::ostream &os) const
{
  os << "Snr=" << m_snr;
}

void
SnrTag::Set (double snr)
{
  m_snr = snr;
}

double
SnrTag::Get (void) const
{
  return m_snr;
}

}
