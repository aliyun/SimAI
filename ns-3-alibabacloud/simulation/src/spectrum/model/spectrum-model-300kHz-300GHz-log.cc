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


#include <ns3/spectrum-model-300kHz-300GHz-log.h>

namespace ns3 {


/**
 * \ingroup spectrum
 * Spectrum model logger for frequencies between 300 Khz 300 Ghz
 */
Ptr<SpectrumModel> SpectrumModel300Khz300GhzLog;


/**
 * \ingroup spectrum
 * 
 * Static initializer class for Spectrum model logger 
 * for frequencies between 300 Khz 300 Ghz
 */
class static_SpectrumModel300Khz300GhzLog_initializer
{
public:
  static_SpectrumModel300Khz300GhzLog_initializer ()
  {

    std::vector<double> freqs;
    for (double f = 3e5; f < 3e11; f = f * 2)
      {
        freqs.push_back (f);
      }
    SpectrumModel300Khz300GhzLog = Create<SpectrumModel> (freqs);
  }

};

/**
 * \ingroup spectrum
 * Static variable for analyzer initialization
 */
static_SpectrumModel300Khz300GhzLog_initializer g_static_SpectrumModel300Khz300GhzLog_initializer_instance;

}
