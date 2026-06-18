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

#ifndef OBSS_PD_ALGORITHM_H
#define OBSS_PD_ALGORITHM_H

#include "ns3/object.h"
#include "ns3/traced-callback.h"
#include "he-configuration.h"

namespace ns3 {

struct HeSigAParameters;

class WifiNetDevice;

/**
 * \brief OBSS PD algorithm interface
 * \ingroup wifi
 *
 * This object provides the interface for all OBSS_PD algorithms
 * and is designed to be subclassed.
 *
 * OBSS_PD stands for Overlapping Basic Service Set Preamble-Detection.
 * OBSS_PD is an 802.11ax feature that allows a STA, under specific
 * conditions, to ignore an inter-BSS PPDU.
 */
class ObssPdAlgorithm : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Connect the WifiNetDevice and setup eventual callbacks.
   *
   * \param device the WifiNetDevice
   */
  virtual void ConnectWifiNetDevice (const Ptr<WifiNetDevice> device);

  /**
   * Reset PHY to IDLE.
   * \param params HeSigAParameters causing PHY reset
   */
  void ResetPhy (HeSigAParameters params);

  /**
   * \param params the HE-SIG-A parameters
   *
   * Evaluate the receipt of HE-SIG-A.
   */
  virtual void ReceiveHeSigA (HeSigAParameters params) = 0;

  /**
   * TracedCallback signature for OBSS_PD reset events.
   *
   * \param [in] bssColor The BSS color of frame triggering the reset
   * \param [in] rssiDbm The RSSI (dBm) of frame triggering the reset
   * \param [in] powerRestricted Whether a TX power restriction is triggered
   * \param [in] txPowerMaxDbmSiso The SISO TX power restricted level (dBm)
   * \param [in] txPowerMaxDbmMimo The MIMO TX power restricted level (dBm)
   */
  typedef void (* ResetTracedCallback)(uint8_t bssColor, double rssiDbm, bool powerRestricted, double txPowerMaxDbmSiso, double txPowerMaxDbmMimo);


protected:
  void DoDispose (void) override;

  Ptr<WifiNetDevice> m_device; ///< Pointer to the WifiNetDevice
  double m_obssPdLevel;        ///< Current OBSS PD level (dBm)


private:
  double m_obssPdLevelMin; ///< Minimum OBSS PD level (dBm)
  double m_obssPdLevelMax; ///< Maximum OBSS PD level (dBm)
  double m_txPowerRefSiso; ///< SISO reference TX power level (dBm)
  double m_txPowerRefMimo; ///< MIMO reference TX power level (dBm)

  /**
   * TracedCallback signature for PHY reset events.
   */
  TracedCallback<uint8_t, double, bool, double, double>  m_resetEvent;
};

} //namespace ns3

#endif /* OBSS_PD_ALGORITHM_H */
