/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 University of Washington
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/double.h"
#include "obss-pd-algorithm.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "he-phy.h"
#include "ns3/wifi-utils.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ObssPdAlgorithm");
NS_OBJECT_ENSURE_REGISTERED (ObssPdAlgorithm);

TypeId
ObssPdAlgorithm::GetTypeId (void)
{
  static ns3::TypeId tid = ns3::TypeId ("ns3::ObssPdAlgorithm")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddAttribute ("ObssPdLevel",
                   "The current OBSS PD level (dBm).",
                   DoubleValue (-82.0),
                   MakeDoubleAccessor (&ObssPdAlgorithm::m_obssPdLevel),
                   MakeDoubleChecker<double> (-101, -62))
    .AddAttribute ("ObssPdLevelMin",
                   "Minimum value (dBm) of OBSS PD level.",
                   DoubleValue (-82.0),
                   MakeDoubleAccessor (&ObssPdAlgorithm::m_obssPdLevelMin),
                   MakeDoubleChecker<double> (-101, -62))
    .AddAttribute ("ObssPdLevelMax",
                   "Maximum value (dBm) of OBSS PD level.",
                   DoubleValue (-62.0),
                   MakeDoubleAccessor (&ObssPdAlgorithm::m_obssPdLevelMax),
                   MakeDoubleChecker<double> (-101, -62))
    .AddAttribute ("TxPowerRefSiso",
                   "The SISO reference TX power level (dBm).",
                   DoubleValue (21),
                   MakeDoubleAccessor (&ObssPdAlgorithm::m_txPowerRefSiso),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TxPowerRefMimo",
                   "The MIMO reference TX power level (dBm).",
                   DoubleValue (25),
                   MakeDoubleAccessor (&ObssPdAlgorithm::m_txPowerRefMimo),
                   MakeDoubleChecker<double> ())
    .AddTraceSource ("Reset", "Trace CCA Reset event",
                     MakeTraceSourceAccessor (&ObssPdAlgorithm::m_resetEvent),
                     "ns3::ObssPdAlgorithm::ResetTracedCallback")
  ;
  return tid;
}

void
ObssPdAlgorithm::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_device = 0;
}

void
ObssPdAlgorithm::ConnectWifiNetDevice (const Ptr<WifiNetDevice> device)
{
  NS_LOG_FUNCTION (this << device);
  m_device = device;
}

void
ObssPdAlgorithm::ResetPhy (HeSigAParameters params)
{
  double txPowerMaxSiso = 0;
  double txPowerMaxMimo = 0;
  bool powerRestricted = false;
  // Fetch my BSS color
  Ptr<HeConfiguration> heConfiguration = m_device->GetHeConfiguration ();
  NS_ASSERT (heConfiguration);
  uint8_t bssColor = heConfiguration->GetBssColor ();
  NS_LOG_DEBUG ("My BSS color " << (uint16_t) bssColor << " received frame " << (uint16_t) params.bssColor);

  Ptr<WifiPhy> phy = m_device->GetPhy ();
  if ((m_obssPdLevel > m_obssPdLevelMin) && (m_obssPdLevel <= m_obssPdLevelMax))
    {
      txPowerMaxSiso = m_txPowerRefSiso - (m_obssPdLevel - m_obssPdLevelMin);
      txPowerMaxMimo = m_txPowerRefMimo - (m_obssPdLevel - m_obssPdLevelMin);
      powerRestricted = true;
    }
  m_resetEvent (bssColor, WToDbm (params.rssiW), powerRestricted, txPowerMaxSiso, txPowerMaxMimo);
  phy->ResetCca (powerRestricted, txPowerMaxSiso, txPowerMaxMimo);
}

} //namespace ns3
