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

#ifndef METRIC_REPORT_H
#define METRIC_REPORT_H

#include <stdint.h>
#include "ns3/buffer.h"
#include "ns3/mesh-information-element-vector.h"

namespace ns3 {
namespace dot11s {
/**
 * \brief a IEEE 802.11s Mesh ID 7.3.2.88 of 802.11s draft 3.0
 *
 */
class IeLinkMetricReport : public WifiInformationElement
{
public:
  IeLinkMetricReport ();
  /**
   * Constructor
   *
   * \param metric the metric
   */
  IeLinkMetricReport (uint32_t metric);
  /**
   * Set metric value
   * \param metric the metric
   */
  void SetMetric (uint32_t metric);
  /**
   * Get metric value
   * \returns the metric
   */
  uint32_t GetMetric ();

  // Inherited from WifiInformationElement
  virtual WifiInformationElementId ElementId () const;
  virtual void SerializeInformationField (Buffer::Iterator i) const;
  virtual uint8_t DeserializeInformationField (Buffer::Iterator start, uint8_t length);
  virtual void Print (std::ostream& os) const;
  virtual uint8_t GetInformationFieldSize () const;

private:
  uint32_t m_metric; ///< metric
  /**
   * equality operator
   *
   * \param a lhs
   * \param b lhs
   * \returns true if equal
   */
  friend bool operator== (const IeLinkMetricReport & a, const IeLinkMetricReport & b);
  /**
   * greater than operator
   *
   * \param a lhs
   * \param b lhs
   * \returns true if equal
   */
  friend bool operator>  (const IeLinkMetricReport & a, const IeLinkMetricReport & b);
  /**
   * less than operator
   *
   * \param a lhs
   * \param b lhs
   * \returns true if equal
   */
  friend bool operator<  (const IeLinkMetricReport & a, const IeLinkMetricReport & b);
};

bool operator== (const IeLinkMetricReport & a, const IeLinkMetricReport & b);
bool operator>  (const IeLinkMetricReport & a, const IeLinkMetricReport & b);
bool operator<  (const IeLinkMetricReport & a, const IeLinkMetricReport & b);
std::ostream &operator << (std::ostream &os, const IeLinkMetricReport &linkMetricReport);
} // namespace dot11s
} // namespace ns3
#endif /* METRIC_REPORT_H */
