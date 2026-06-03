/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Author: Gary Pei <guangyu.pei@boeing.com>
 */
#ifndef LR_WPAN_SPECTRUM_VALUE_HELPER_H
#define LR_WPAN_SPECTRUM_VALUE_HELPER_H

#include <ns3/ptr.h>

namespace ns3 {

class SpectrumValue;

/**
 * \ingroup lr-wpan
 *
 * \brief This class defines all functions to create spectrum model for LrWpan
 */
class LrWpanSpectrumValueHelper
{
public:
  LrWpanSpectrumValueHelper (void);
  virtual ~LrWpanSpectrumValueHelper (void);

  /**
   * \brief create spectrum value
   * \param txPower the power transmission in dBm
   * \param channel the channel number per IEEE802.15.4
   * \return a Ptr to a newly created SpectrumValue instance
   */
  Ptr<SpectrumValue> CreateTxPowerSpectralDensity (double txPower, uint32_t channel);

  /**
   * \brief create spectrum value for noise
   * \param channel the channel number per IEEE802.15.4
   * \return a Ptr to a newly created SpectrumValue instance
   */
  Ptr<SpectrumValue> CreateNoisePowerSpectralDensity (uint32_t channel);

  /**
   * \brief total average power of the signal is the integral of the PSD using
   * the limits of the given channel
   * \param psd spectral density
   * \param channel the channel number per IEEE802.15.4
   * \return total power (using composite trap. rule to numerally integrate)
   */
  static double TotalAvgPower (Ptr<const SpectrumValue> psd, uint32_t channel);

private:
  /**
   * A scaling factor for the noise power.
   */
  double m_noiseFactor;

};

} //namespace ns3

#endif /*  LR_WPAN_SPECTRUM_VALUE_HELPER_H */
