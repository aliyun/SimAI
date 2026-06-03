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

#ifndef CONSTANT_OBSS_PD_ALGORITHM_H
#define CONSTANT_OBSS_PD_ALGORITHM_H

#include "obss-pd-algorithm.h"

namespace ns3 {

/**
 * \brief Constant OBSS PD algorithm
 * \ingroup wifi
 *
 * This constant OBSS_PD algorithm is a simple OBSS_PD algorithm which evaluates if a receiving
 * signal should be accepted or rejected based on a constant threshold.
 *
 * Once a HE-SIG-A has been received by the PHY, the ReceiveHeSigA method is
 * triggered. The algorithm then checks whether this is an OBSS frame by comparing its own BSS
 * color with the BSS color of the received preamble. If this is an OBSS frame, it compares the
 * received RSSI with its configured OBSS_PD level value. The PHY then gets reset to IDLE state
 * in case the received RSSI is lower than that constant OBSS PD level value, and is informed
 * about TX power restrictions that might be applied to the next transmission.
 */
class ConstantObssPdAlgorithm : public ObssPdAlgorithm
{
public:
  ConstantObssPdAlgorithm ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  void ConnectWifiNetDevice (const Ptr<WifiNetDevice> device) override;
  void ReceiveHeSigA (HeSigAParameters params) override;
};

} //namespace ns3

#endif /* CONSTANT_OBSS_PD_ALGORITHM_H */
