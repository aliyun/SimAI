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


#include <ns3/spectrum-model-ism2400MHz-res1MHz.h>

namespace ns3 {

/**
 * \ingroup spectrum
 * Spectrum model logger for frequencies in the 2.4 GHz ISM band
 * with 1 MHz resolution.
 */
Ptr<SpectrumModel> SpectrumModelIsm2400MhzRes1Mhz;


/**
 * \ingroup spectrum
 * 
 * Static initializer class for Spectrum model logger for 
 * frequencies in the 2.4 GHz ISM band with 1 MHz resolution.
 */
class static_SpectrumModelIsm2400MhzRes1Mhz_initializer
{
public:
  static_SpectrumModelIsm2400MhzRes1Mhz_initializer ()
  {

    std::vector<double> freqs;
    for (int i = 0; i < 100; ++i)
      {
        freqs.push_back ((i + 2400) * 1e6);
      }


    SpectrumModelIsm2400MhzRes1Mhz = Create<SpectrumModel> (freqs);
  }

};

/**
 * \ingroup spectrum
 * Static variable for analyzer initialization
 */
static_SpectrumModelIsm2400MhzRes1Mhz_initializer g_static_SpectrumModelIsm2400MhzRes1Mhz_initializer_instance;

}
