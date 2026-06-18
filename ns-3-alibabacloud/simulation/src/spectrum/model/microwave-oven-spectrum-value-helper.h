/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 */

#ifndef MICROWAVE_OVEN_SPECTRUM_VALUE_HELPER_H
#define MICROWAVE_OVEN_SPECTRUM_VALUE_HELPER_H


#include <ns3/spectrum-value.h>

namespace ns3 {



/**
 * \ingroup spectrum
 *
 * This class provides methods for the creation of SpectrumValue
 * instances that mimic the Power Spectral Density of commercial
 * microwave ovens based on the measurements reported in the following paper:
 * Tanim M. Taher, Matthew J. Misurac, Joseph L. LoCicero, and Donald R. Ucci,
 * "MICROWAVE OVEN SIGNAL MODELING", in Proc. of IEEE WCNC, 2008
 *
 */
class MicrowaveOvenSpectrumValueHelper
{
public:
  /**
   *
   * @return the Power Spectral Density of Micro Wave Oven #1 in the
   * cited paper
   */
  static Ptr<SpectrumValue> CreatePowerSpectralDensityMwo1 ();

  /**
   *
   * @return the Power Spectral Density of Micro Wave Oven #2 in the
   * cited paper
   */
  static Ptr<SpectrumValue> CreatePowerSpectralDensityMwo2 ();

};





} // namespace ns3



#endif /*  MICROWAVE_OVEN_SPECTRUM_VALUE_HELPER_H */
