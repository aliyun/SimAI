/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "ie-dot11s-metric-report.h"
#include "ns3/assert.h"

namespace ns3 {
namespace dot11s {
IeLinkMetricReport::IeLinkMetricReport () :
  m_metric (0)
{
}
IeLinkMetricReport::IeLinkMetricReport (uint32_t metric)
{
  m_metric = metric;
}
WifiInformationElementId
IeLinkMetricReport::ElementId () const
{
  return IE_MESH_LINK_METRIC_REPORT;
}
uint8_t
IeLinkMetricReport::GetInformationFieldSize (void) const
{
  return sizeof (uint32_t);
}
uint32_t
IeLinkMetricReport::GetMetric ()
{
  return m_metric;
}
void
IeLinkMetricReport::SetMetric (uint32_t metric)
{
  m_metric = metric;
}
void
IeLinkMetricReport::SerializeInformationField (Buffer::Iterator i) const
{
  i.WriteHtolsbU32 (m_metric);
}
uint8_t
IeLinkMetricReport::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  m_metric = i.ReadLsbtohU32 ();
  return i.GetDistanceFrom (start);
}
void
IeLinkMetricReport::Print (std::ostream& os) const
{
  os << "Metric=" << m_metric;
}
bool
operator== (const IeLinkMetricReport & a, const IeLinkMetricReport & b)
{
  return (a.m_metric == b.m_metric);
}
bool
operator< (const IeLinkMetricReport & a, const IeLinkMetricReport & b)
{
  return (a.m_metric < b.m_metric);
}
bool
operator> (const IeLinkMetricReport & a, const IeLinkMetricReport & b)
{
  return (a.m_metric > b.m_metric);
}
std::ostream &
operator << (std::ostream &os, const IeLinkMetricReport &a)
{
  a.Print (os);
  return os;
}
} // namespace dot11s
} // namespace ns3
