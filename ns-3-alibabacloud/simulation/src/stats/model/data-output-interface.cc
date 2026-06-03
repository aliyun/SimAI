/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Drexel University
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
 * Author: Joe Kopena (tjkopena@cs.drexel.edu)
 */

#include "ns3/log.h"

#include "data-output-interface.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("DataOutputInterface");

//--------------------------------------------------------------
//----------------------------------------------
DataOutputInterface::DataOutputInterface()
{
  NS_LOG_FUNCTION (this);
}
DataOutputInterface::~DataOutputInterface()
{
  NS_LOG_FUNCTION (this);
}
/* static */
TypeId
DataOutputInterface::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DataOutputInterface")
    .SetParent<Object> ()
    .SetGroupName ("Stats")
    // No AddConstructor because this is an abstract class.
    ;
  return tid;
}
  
void
DataOutputInterface::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  Object::DoDispose ();
  // end DataOutputInterface::DoDispose
}

void
DataOutputInterface::SetFilePrefix (const std::string prefix)
{
  NS_LOG_FUNCTION (this << prefix);

  m_filePrefix = prefix;
}

std::string
DataOutputInterface::GetFilePrefix () const
{
  NS_LOG_FUNCTION (this);

  return m_filePrefix;
}
