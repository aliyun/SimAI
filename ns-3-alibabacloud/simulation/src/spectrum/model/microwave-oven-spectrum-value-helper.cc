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

#include "ns3/log.h"
#include "microwave-oven-spectrum-value-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MicrowaveOvenSpectrumValue");

/// Spectrum model for the 5MHz model (model #2)
static Ptr<SpectrumModel> g_MicrowaveOvenSpectrumModel5Mhz;
/// Spectrum model for the 6MHz model (model #1)
static Ptr<SpectrumModel> g_MicrowaveOvenSpectrumModel6Mhz;

/**
 * \ingroup spectrum
 * Init a static class containing the MWO #2 model (5Mhz)
 */
static class MicrowaveOvenSpectrumModel5MhzInitializer
{
public:
  MicrowaveOvenSpectrumModel5MhzInitializer ()
  {
    NS_LOG_FUNCTION (this);
    Bands bands;
    for (double fl = 2400e6; fl < 2499e6; fl += 5e6)
      {
        BandInfo bi;
        bi.fl = fl;
        bi.fc = fl + 5e6;
        bi.fh = fl + 10e6;
        bands.push_back (bi);
      }
    NS_LOG_LOGIC ("bands.size () :" << bands.size ());
    g_MicrowaveOvenSpectrumModel5Mhz = Create<SpectrumModel> (bands);
  }
} g_MicrowaveOvenSpectrumModel5MhzInitializerInstance; //!< MWO model #2



/**
 * \ingroup spectrum
 * Init a static class containing the MWO #1 model (6Mhz)
 */
static class MicrowaveOvenSpectrumModel6MhzInitializer
{
public:
  MicrowaveOvenSpectrumModel6MhzInitializer ()
  {
    NS_LOG_FUNCTION (this);
    Bands bands;
    for (double fl = 2360e6; fl < 2479e6; fl += 6e6)
      {
        BandInfo bi;
        bi.fl = fl;
        bi.fc = fl + 6e6;
        bi.fh = fl + 12e6;
        bands.push_back (bi);
      }
    NS_LOG_LOGIC ("bands.size () :" << bands.size ());
    g_MicrowaveOvenSpectrumModel6Mhz = Create<SpectrumModel> (bands);
  }
} g_MicrowaveOvenSpectrumModel6MhzInitializerInstance; //!< MWO model #1





Ptr<SpectrumValue>
MicrowaveOvenSpectrumValueHelper::CreatePowerSpectralDensityMwo1 ()
{
  Ptr<SpectrumValue> psd = Create <SpectrumValue> (g_MicrowaveOvenSpectrumModel6Mhz);

  // values from this paper:
  // Tanim M. Taher, Matthew J. Misurac, Joseph L. LoCicero, and Donald R. Ucci,
  // "MICROWAVE OVEN SIGNAL MODELING", in Proc. of IEEE WCNC, 2008,
  // see Figure 3, "Experimental PSD of MWO #1"
  // the figure has a resolution of 12 MHz per division; we use a
  // SpectrumModel with a per-subband bandwidth of 6MHz, so we have
  // two samples per division. The values used here are an approximation
  // of what appears in the figure.

  (*psd)[0] = -67.5;
  (*psd)[1] = -67.5;
  (*psd)[2] = -67.5;
  (*psd)[3] = -67.5;
  (*psd)[4] = -67.5;
  (*psd)[5] = -66;
  (*psd)[6] = -64;
  (*psd)[7] = -63;
  (*psd)[8] = -62.5;
  (*psd)[9] = -63;
  (*psd)[10] = -62.5;
  (*psd)[11] = -62.5;
  (*psd)[12] = -58;
  (*psd)[13] = -53.5;
  (*psd)[14] = -44;
  (*psd)[15] = -38;
  (*psd)[16] = -45;
  (*psd)[17] = -65;
  (*psd)[18] = -67.5;
  (*psd)[19] = -67.5;

  // convert to W/Hz
  (*psd) = Pow (10.0, ((*psd) - 30) / 10.0);

  return psd;
}



Ptr<SpectrumValue>
MicrowaveOvenSpectrumValueHelper::CreatePowerSpectralDensityMwo2 ()
{
  // values from this paper:
  // Tanim M. Taher, Matthew J. Misurac, Joseph L. LoCicero, and Donald R. Ucci,
  // "MICROWAVE OVEN SIGNAL MODELING", in Proc. of IEEE WCNC, 2008,
  // see Figure 9, "Experimental PSD of actual MWO #2"
  // the figure has a resolution of 10 MHz per division; we use a
  // SpectrumModel with a per-subband bandwidth of 5MHz, so we have
  // two samples per division. The values used here are an approximation
  // of what appears in the figure.

  Ptr<SpectrumValue> psd = Create <SpectrumValue> (g_MicrowaveOvenSpectrumModel5Mhz);

  (*psd)[0] = -68;
  (*psd)[1] = -68;
  (*psd)[2] = -68;
  (*psd)[3] = -68;
  (*psd)[4] = -65;
  (*psd)[5] = -62;
  (*psd)[6] = -56;
  (*psd)[7] = -55;
  (*psd)[8] = -47;
  (*psd)[9] = -40;
  (*psd)[10] = -37;
  (*psd)[11] = -33;
  (*psd)[12] = -45;
  (*psd)[13] = -67;
  (*psd)[14] = -68;
  (*psd)[15] = -68;
  (*psd)[16] = -68;
  (*psd)[17] = -68;
  (*psd)[18] = -68;
  (*psd)[19] = -68;

  // convert to W/Hz
  (*psd) = Pow (10.0, ((*psd) - 30) / 10.0);

  return psd;
}



} // namespace ns3
