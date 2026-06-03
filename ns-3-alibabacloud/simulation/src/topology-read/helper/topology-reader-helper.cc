/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella (tommaso.pecorella@unifi.it)
 * Author: Valerio Sartini (valesar@gmail.com)
 */

#include "ns3/object.h"
#include "ns3/topology-reader-helper.h"
#include "ns3/inet-topology-reader.h"
#include "ns3/orbis-topology-reader.h"
#include "ns3/rocketfuel-topology-reader.h"
#include "ns3/log.h"

/**
 * \file
 * \ingroup topology
 * ns3::TopologyHelper implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TopologyReaderHelper");

TopologyReaderHelper::TopologyReaderHelper ()
{
  m_inputModel = 0;
}

void
TopologyReaderHelper::SetFileName (const std::string fileName)
{
  m_fileName = fileName;
}

void
TopologyReaderHelper::SetFileType (const std::string fileType)
{
  m_fileType = fileType;
}


Ptr<TopologyReader>
TopologyReaderHelper::GetTopologyReader ()
{
  if (!m_inputModel)
    {
      NS_ASSERT_MSG (!m_fileType.empty (), "Missing File Type");
      NS_ASSERT_MSG (!m_fileName.empty (), "Missing File Name");

      if (m_fileType == "Orbis")
        {
          NS_LOG_INFO ("Creating Orbis formatted data input.");
          m_inputModel = CreateObject<OrbisTopologyReader> ();
        }
      else if (m_fileType == "Inet")
        {
          NS_LOG_INFO ("Creating Inet formatted data input.");
          m_inputModel = CreateObject<InetTopologyReader> ();
        }
      else if (m_fileType == "Rocketfuel")
        {
          NS_LOG_INFO ("Creating Rocketfuel formatted data input.");
          m_inputModel = CreateObject<RocketfuelTopologyReader> ();
        }
      else
        {
          NS_ASSERT_MSG (false, "Wrong (unknown) File Type");
        }

      m_inputModel->SetFileName (m_fileName);
    }
  return m_inputModel;
}



} // namespace ns3
