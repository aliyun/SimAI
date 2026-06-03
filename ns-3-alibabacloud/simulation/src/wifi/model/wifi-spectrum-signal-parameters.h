/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 * Modified by Marco Miozzo <mmiozzo@cttc.es> (add data and ctrl diversity)
 */

#ifndef WIFI_SPECTRUM_SIGNAL_PARAMETERS_H
#define WIFI_SPECTRUM_SIGNAL_PARAMETERS_H

#include "ns3/spectrum-signal-parameters.h"

namespace ns3 {

class WifiPpdu;

/**
 * \ingroup wifi
 *
 * Signal parameters for wifi
 */
struct WifiSpectrumSignalParameters : public SpectrumSignalParameters
{

  Ptr<SpectrumSignalParameters> Copy () override;

  /**
   * default constructor
   */
  WifiSpectrumSignalParameters ();

  /**
   * copy constructor
   *
   * \param p the wifi spectrum signal parameters
   */
  WifiSpectrumSignalParameters (const WifiSpectrumSignalParameters& p);

  Ptr<WifiPpdu> ppdu;                  ///< The PPDU being transmitted
  uint16_t txCenterFreq;               ///< the center frequency of the transmitted signal in MHz
};

}  // namespace ns3

#endif /* WIFI_SPECTRUM_SIGNAL_PARAMETERS_H */
