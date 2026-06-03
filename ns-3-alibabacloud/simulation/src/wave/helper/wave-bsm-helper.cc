/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 North Carolina State University
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
 * Author: Scott E. Carpenter <scarpen@ncsu.edu>
 *
 */

#include "ns3/wave-bsm-helper.h"
#include "ns3/log.h"

NS_LOG_COMPONENT_DEFINE ("WaveBsmHelper");

namespace ns3 {

std::vector<int> WaveBsmHelper::nodesMoving;

WaveBsmHelper::WaveBsmHelper ()
  : m_waveBsmStats ()
{
  m_txSafetyRangesSq.resize (10, 0);
  m_txSafetyRangesSq[0] = 50.0 * 50.0;
  m_txSafetyRangesSq[1] = 100.0 * 100.0;
  m_txSafetyRangesSq[2] = 200.0 * 200.0;
  m_txSafetyRangesSq[3] = 300.0 * 300.0;
  m_txSafetyRangesSq[4] = 400.0 * 400.0;
  m_txSafetyRangesSq[5] = 500.0 * 500.0;
  m_txSafetyRangesSq[6] = 600.0 * 600.0;
  m_txSafetyRangesSq[7] = 800.0 * 800.0;
  m_txSafetyRangesSq[8] = 1000.0 * 1000.0;
  m_txSafetyRangesSq[9] = 1500.0 * 1500.0;

  m_factory.SetTypeId ("ns3::BsmApplication");
}

void
WaveBsmHelper::SetAttribute (std::string name, const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ApplicationContainer
WaveBsmHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
WaveBsmHelper::Install (Ipv4InterfaceContainer i) const
{
  ApplicationContainer apps;
  for (Ipv4InterfaceContainer::Iterator itr = i.Begin (); itr != i.End (); ++itr)
    {
      std::pair<Ptr<Ipv4>, uint32_t> interface = (*itr);
      Ptr<Ipv4> pp = interface.first;
      Ptr<Node> node = pp->GetObject<Node> ();
      apps.Add (InstallPriv (node));
    }

  return apps;
}

Ptr<Application>
WaveBsmHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Application> app = m_factory.Create<Application> ();
  node->AddApplication (app);

  return app;
}

void
WaveBsmHelper::Install (Ipv4InterfaceContainer & i,
                        Time totalTime,          // seconds
                        uint32_t wavePacketSize, // bytes
                        Time waveInterval,       // seconds
                        double gpsAccuracyNs,    // clock drift range in number of ns
                        std::vector <double> ranges,           // m
                        int chAccessMode,        // channel access mode
                        Time txMaxDelay)         // max delay prior to transmit
{
  int size = ranges.size ();
  m_txSafetyRangesSq.clear ();
  m_txSafetyRangesSq.resize (size, 0);
  for (int index = 0; index < size; index++)
    {
      // stored as square of value, for optimization
      m_txSafetyRangesSq[index] = ranges[index] * ranges[index];
    }

  // install a BsmApplication on each node
  ApplicationContainer bsmApps = Install (i);
  // start BSM app immediately (BsmApplication will
  // delay transmission of first BSM by 1.0 seconds)
  bsmApps.Start (Seconds (0));
  bsmApps.Stop (totalTime);

  // for each app, setup the app parameters
  ApplicationContainer::Iterator aci;
  int nodeId = 0;
  for (aci = bsmApps.Begin (); aci != bsmApps.End (); ++aci)
    {
      Ptr<BsmApplication> bsmApp = DynamicCast<BsmApplication> (*aci);
      bsmApp->Setup (i,
                     nodeId,
                     totalTime,
                     wavePacketSize,
                     waveInterval,
                     gpsAccuracyNs,
                     m_txSafetyRangesSq,
                     GetWaveBsmStats (),
                     &nodesMoving,
                     chAccessMode,
                     txMaxDelay);
      nodeId++;
    }
}

Ptr<WaveBsmStats>
WaveBsmHelper::GetWaveBsmStats ()
{
  return &m_waveBsmStats;
}

int64_t
WaveBsmHelper::AssignStreams (NodeContainer c, int64_t stream)
{
  int64_t currentStream = stream;
  Ptr<Node> node;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      node = (*i);
      for (uint32_t j = 0; j < node->GetNApplications (); j++)
        {
          Ptr<BsmApplication> bsmApp = DynamicCast<BsmApplication> (node->GetApplication (j));
          if (bsmApp)
            {
              currentStream += bsmApp->AssignStreams (currentStream);
            }
        }
    }
  return (currentStream - stream);
}

std::vector<int>&
WaveBsmHelper::GetNodesMoving ()
{
  return nodesMoving;
}

} // namespace ns3
