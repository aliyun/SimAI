/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Lawrence Livermore National Laboratory
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include "log.h"
#include "hash-function.h"

/**
 * \file
 * \ingroup hash
 * \brief ns3::Hash::Implementation::GetHash64 default implementation.
 */


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HashFunction");

namespace Hash {

uint64_t
Implementation::GetHash64  (const char * buffer, const std::size_t size)
{
  NS_LOG_WARN ("64-bit hash requested, only 32-bit implementation available");
  return GetHash32 (buffer, size);
}

}  // namespace Hash

}  // namespace ns3
