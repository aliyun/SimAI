/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Universita' di Napoli
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "block-ack-type.h"
#include "ns3/fatal-error.h"

namespace ns3 {

BlockAckType::BlockAckType (Variant v)
  : m_variant (v)
{
  switch (m_variant)
    {
    case BASIC:
      m_bitmapLen.push_back (128);
      break;
    case COMPRESSED:
    case EXTENDED_COMPRESSED:
      m_bitmapLen.push_back (8);
      break;
    case MULTI_TID:
    case MULTI_STA:
      // m_bitmapLen is left empty.
      break;
    default:
      NS_FATAL_ERROR ("Unknown block ack type");
    }
}

BlockAckType::BlockAckType ()
  : BlockAckType (BASIC)
{
}

BlockAckType::BlockAckType (Variant v, std::vector<uint8_t> l)
  : m_variant (v),
    m_bitmapLen (l)
{
}

BlockAckReqType::BlockAckReqType (Variant v)
  : m_variant (v)
{
  switch (m_variant)
    {
    case BASIC:
    case COMPRESSED:
    case EXTENDED_COMPRESSED:
      m_nSeqControls = 1;
      break;
    case MULTI_TID:
      m_nSeqControls = 0;
      break;
    default:
      NS_FATAL_ERROR ("Unknown block ack request type");
    }
}

BlockAckReqType::BlockAckReqType ()
  : BlockAckReqType (BASIC)
{
}

BlockAckReqType::BlockAckReqType (Variant v, uint8_t nSeqControls)
  : m_variant (v),
    m_nSeqControls (nSeqControls)
{
}

std::ostream &operator << (std::ostream &os, const BlockAckType &type)
{
  switch (type.m_variant)
    {
    case BlockAckType::BASIC:
      os << "basic-block-ack";
      break;
    case BlockAckType::COMPRESSED:
      os << "compressed-block-ack";
      break;
    case BlockAckType::EXTENDED_COMPRESSED:
      os << "extended-compressed-block-ack";
      break;
    case BlockAckType::MULTI_TID:
      os << "multi-tid-block-ack[" << type.m_bitmapLen.size () << "]";
      break;
    case BlockAckType::MULTI_STA:
      os << "multi-sta-block-ack[" << type.m_bitmapLen.size () << "]";
      break;
    default:
      NS_FATAL_ERROR ("Unknown block ack type");
    }
  return os;
}

std::ostream &operator << (std::ostream &os, const BlockAckReqType &type)
{
  switch (type.m_variant)
    {
    case BlockAckReqType::BASIC:
      os << "basic-block-ack-req";
      break;
    case BlockAckReqType::COMPRESSED:
      os << "compressed-block-ack-req";
      break;
    case BlockAckReqType::EXTENDED_COMPRESSED:
      os << "extended-compressed-block-ack-req";
      break;
    case BlockAckReqType::MULTI_TID:
      os << "multi-tid-block-ack-req[" << type.m_nSeqControls << "]";
      break;
    default:
      NS_FATAL_ERROR ("Unknown block ack request type");
    }
  return os;
}

} //namespace ns3
