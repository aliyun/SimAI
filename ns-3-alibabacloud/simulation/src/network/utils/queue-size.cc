/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2018 Universita' degli Studi di Napoli Federico II
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Stefano Avallone <stavallo@unina.it>
//

#include "queue-size.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("QueueSize");

ATTRIBUTE_HELPER_CPP (QueueSize);

/* static */
bool
QueueSize::DoParse (const std::string s, QueueSizeUnit *unit, uint32_t *value)
{
  NS_LOG_FUNCTION (s << unit << value);
  std::string::size_type n = s.find_first_not_of ("0123456789.");
  if (n != std::string::npos)
    { // Found non-numeric
      std::istringstream iss;
      iss.str (s.substr (0, n));
      double r;
      iss >> r;
      std::string trailer = s.substr (n, std::string::npos);
      if (trailer == "B")
        {
          // bytes
          *unit = QueueSizeUnit::BYTES;
          *value = static_cast<uint32_t>(r);
        }
      else if (trailer == "kB" || trailer == "KB")
        {
          // kilobytes
          *unit = QueueSizeUnit::BYTES;
          *value = static_cast<uint32_t>(r * 1000);
        }
      else if (trailer == "KiB")
        {
          // kibibytes
          *unit = QueueSizeUnit::BYTES;
          *value = static_cast<uint32_t>(r * 1024);
        }
      else if (trailer == "MB")
        {
          // MegaBytes
          *unit = QueueSizeUnit::BYTES;
          *value = static_cast<uint32_t>(r * 1000000);
        }
      else if (trailer == "MiB")
        {
          // MebiBytes
          *unit = QueueSizeUnit::BYTES;
          *value = static_cast<uint32_t>(r * 1048576);
        }
      else if (trailer == "p")
        {
          // packets
          *unit = QueueSizeUnit::PACKETS;
          *value = static_cast<uint32_t>(r);
        }
      else if (trailer == "kp" || trailer == "Kp")
        {
          // kilopackets
          *unit = QueueSizeUnit::PACKETS;
          *value = static_cast<uint32_t>(r * 1000);
        }
      else if (trailer == "Kip")
        {
          // kibipackets
          *unit = QueueSizeUnit::PACKETS;
          *value = static_cast<uint32_t>(r * 1024);
        }
      else if (trailer == "Mp")
        {
          // MegaPackets
          *unit = QueueSizeUnit::PACKETS;
          *value = static_cast<uint32_t>(r * 1000000);
        }
      else if (trailer == "Mip")
        {
          // MebiPackets
          *unit = QueueSizeUnit::PACKETS;
          *value = static_cast<uint32_t>(r * 1048576);
        }
      else
        {
          return false;   // unsupported unit string
        }
      return true;
    }
  return false;   // a unit string is required
}

QueueSize::QueueSize ()
  : m_unit (QueueSizeUnit::PACKETS),
    m_value (0)
{
  NS_LOG_FUNCTION (this);
}

QueueSize::QueueSize (QueueSizeUnit unit, uint32_t value)
  : m_unit (unit),
    m_value (value)
{
  NS_LOG_FUNCTION (this << static_cast<uint16_t>(unit) << value);
}

bool QueueSize::operator < (const QueueSize& rhs) const
{
  NS_ABORT_MSG_IF (m_unit != rhs.GetUnit (), "Cannot compare heterogeneous sizes");

  return m_value<rhs.m_value;
}

bool QueueSize::operator <= (const QueueSize& rhs) const
{
  NS_ABORT_MSG_IF (m_unit != rhs.GetUnit (), "Cannot compare heterogeneous sizes");

  return m_value<=rhs.m_value;
}

bool QueueSize::operator >  (const QueueSize& rhs) const
{
  NS_ABORT_MSG_IF (m_unit != rhs.GetUnit (), "Cannot compare heterogeneous sizes");

  return m_value>rhs.m_value;
}

bool QueueSize::operator >= (const QueueSize& rhs) const
{
  NS_ABORT_MSG_IF (m_unit != rhs.GetUnit (), "Cannot compare heterogeneous sizes");

  return m_value>=rhs.m_value;
}

bool QueueSize::operator == (const QueueSize& rhs) const
{
  NS_ABORT_MSG_IF (m_unit != rhs.GetUnit (), "Cannot compare heterogeneous sizes");

  return m_value==rhs.m_value;
}

bool QueueSize::operator != (const QueueSize& rhs) const
{
  NS_ABORT_MSG_IF (m_unit != rhs.GetUnit (), "Cannot compare heterogeneous sizes");

  return m_value!=rhs.m_value;
}

QueueSizeUnit QueueSize::GetUnit () const
{
  NS_LOG_FUNCTION (this);
  return m_unit;
}

uint32_t QueueSize::GetValue () const
{
  NS_LOG_FUNCTION (this);
  return m_value;
}

QueueSize::QueueSize (std::string size)
{
  NS_LOG_FUNCTION (this << size);
  [[maybe_unused]] bool ok = DoParse (size, &m_unit, &m_value);
  NS_ABORT_MSG_IF (!ok, "Could not parse queue size: " << size);
}

/* For printing of queue size */
std::ostream &operator << (std::ostream &os, const QueueSize &size)
{
  os << size.GetValue () << (size.GetUnit () == QueueSizeUnit::PACKETS ? "p" : "B");
  return os;
}
/* Initialize a queue size from an input stream */
std::istream &operator >> (std::istream &is, QueueSize &size)
{
  std::string value;
  is >> value;
  QueueSizeUnit m;
  uint32_t l;
  bool ok = QueueSize::DoParse (value, &m, &l);
  if (!ok)
    {
      is.setstate (std::ios_base::failbit);
    }
  size = QueueSize (m, l);
  return is;
}

} // namespace ns3
