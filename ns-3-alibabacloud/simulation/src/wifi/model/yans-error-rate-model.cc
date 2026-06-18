/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include <cmath>
#include "ns3/log.h"
#include "yans-error-rate-model.h"
#include "wifi-utils.h"
#include "wifi-tx-vector.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("YansErrorRateModel");

NS_OBJECT_ENSURE_REGISTERED (YansErrorRateModel);

TypeId
YansErrorRateModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::YansErrorRateModel")
    .SetParent<ErrorRateModel> ()
    .SetGroupName ("Wifi")
    .AddConstructor<YansErrorRateModel> ()
  ;
  return tid;
}

YansErrorRateModel::YansErrorRateModel ()
{
}

double
YansErrorRateModel::GetBpskBer (double snr, uint32_t signalSpread, uint64_t phyRate) const
{
  NS_LOG_FUNCTION (this << snr << signalSpread << phyRate);
  double EbNo = snr * signalSpread / phyRate;
  double z = std::sqrt (EbNo);
  double ber = 0.5 * erfc (z);
  NS_LOG_INFO ("bpsk snr=" << snr << " ber=" << ber);
  return ber;
}

double
YansErrorRateModel::GetQamBer (double snr, unsigned int m, uint32_t signalSpread, uint64_t phyRate) const
{
  NS_LOG_FUNCTION (this << snr << m << signalSpread << phyRate);
  double EbNo = snr * signalSpread / phyRate;
  double z = std::sqrt ((1.5 * log2 (m) * EbNo) / (m - 1.0));
  double z1 = ((1.0 - 1.0 / std::sqrt (m)) * erfc (z));
  double z2 = 1 - std::pow ((1 - z1), 2);
  double ber = z2 / log2 (m);
  NS_LOG_INFO ("Qam m=" << m << " rate=" << phyRate << " snr=" << snr << " ber=" << ber);
  return ber;
}

uint32_t
YansErrorRateModel::Factorial (uint32_t k) const
{
  uint32_t fact = 1;
  while (k > 0)
    {
      fact *= k;
      k--;
    }
  return fact;
}

double
YansErrorRateModel::Binomial (uint32_t k, double p, uint32_t n) const
{
  double retval = Factorial (n) / (Factorial (k) * Factorial (n - k)) * std::pow (p, static_cast<double> (k)) * std::pow (1 - p, static_cast<double> (n - k));
  return retval;
}

double
YansErrorRateModel::CalculatePdOdd (double ber, unsigned int d) const
{
  NS_ASSERT ((d % 2) == 1);
  unsigned int dstart = (d + 1) / 2;
  unsigned int dend = d;
  double pd = 0;

  for (unsigned int i = dstart; i < dend; i++)
    {
      pd += Binomial (i, ber, d);
    }
  return pd;
}

double
YansErrorRateModel::CalculatePdEven (double ber, unsigned int d) const
{
  NS_ASSERT ((d % 2) == 0);
  unsigned int dstart = d / 2 + 1;
  unsigned int dend = d;
  double pd = 0;

  for (unsigned int i = dstart; i < dend; i++)
    {
      pd +=  Binomial (i, ber, d);
    }
  pd += 0.5 * Binomial (d / 2, ber, d);

  return pd;
}

double
YansErrorRateModel::CalculatePd (double ber, unsigned int d) const
{
  NS_LOG_FUNCTION (this << ber << d);
  double pd;
  if ((d % 2) == 0)
    {
      pd = CalculatePdEven (ber, d);
    }
  else
    {
      pd = CalculatePdOdd (ber, d);
    }
  return pd;
}

double
YansErrorRateModel::GetFecBpskBer (double snr, uint64_t nbits,
                                   uint32_t signalSpread, uint64_t phyRate,
                                   uint32_t dFree, uint32_t adFree) const
{
  NS_LOG_FUNCTION (this << snr << nbits << signalSpread << phyRate << dFree << adFree);
  double ber = GetBpskBer (snr, signalSpread, phyRate);
  if (ber == 0.0)
    {
      return 1.0;
    }
  double pd = CalculatePd (ber, dFree);
  double pmu = adFree * pd;
  pmu = std::min (pmu, 1.0);
  double pms = std::pow (1 - pmu, nbits);
  return pms;
}

double
YansErrorRateModel::GetFecQamBer (double snr, uint64_t nbits,
                                  uint32_t signalSpread,
                                  uint64_t phyRate,
                                  uint32_t m, uint32_t dFree,
                                  uint32_t adFree, uint32_t adFreePlusOne) const
{
  NS_LOG_FUNCTION (this << snr << nbits << signalSpread << phyRate << m << dFree << adFree << adFreePlusOne);
  double ber = GetQamBer (snr, m, signalSpread, phyRate);
  if (ber == 0.0)
    {
      return 1.0;
    }
  /* first term */
  double pd = CalculatePd (ber, dFree);
  double pmu = adFree * pd;
  /* second term */
  pd = CalculatePd (ber, dFree + 1);
  pmu += adFreePlusOne * pd;
  pmu = std::min (pmu, 1.0);
  double pms = std::pow (1 - pmu, nbits);
  return pms;
}

double
YansErrorRateModel::DoGetChunkSuccessRate (WifiMode mode, const WifiTxVector& txVector, double snr, uint64_t nbits, uint8_t numRxAntennas, WifiPpduField field, uint16_t staId) const
{
  NS_LOG_FUNCTION (this << mode << txVector << snr << nbits << +numRxAntennas << field << staId);
  if (mode.GetModulationClass () >= WIFI_MOD_CLASS_ERP_OFDM)
    {
      uint64_t phyRate;
      if ((txVector.IsMu () && (staId == SU_STA_ID)) || (mode != txVector.GetMode ()))
        {
          phyRate = mode.GetPhyRate (txVector.GetChannelWidth () >= 40 ? 20 : txVector.GetChannelWidth ()); //This is the PHY header
        }
      else
        {
          phyRate = mode.GetPhyRate (txVector, staId);
        }
      if (mode.GetConstellationSize () == 2)
        {
          if (mode.GetCodeRate () == WIFI_CODE_RATE_1_2)
            {
              return GetFecBpskBer (snr,
                                    nbits,
                                    txVector.GetChannelWidth () * 1000000, //signal spread
                                    phyRate, //PHY rate
                                    10, //dFree
                                    11); //adFree
            }
          else
            {
              return GetFecBpskBer (snr,
                                    nbits,
                                    txVector.GetChannelWidth () * 1000000, //signal spread
                                    phyRate, //PHY rate
                                    5, //dFree
                                    8); //adFree
            }
        }
      else if (mode.GetConstellationSize () == 4)
        {
          if (mode.GetCodeRate () == WIFI_CODE_RATE_1_2)
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, //signal spread
                                   phyRate, //PHY rate
                                   4, //m
                                   10, //dFree
                                   11, //adFree
                                   0); //adFreePlusOne
            }
          else
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, //signal spread
                                   phyRate, //PHY rate
                                   4, //m
                                   5, //dFree
                                   8, //adFree
                                   31); //adFreePlusOne
            }
        }
      else if (mode.GetConstellationSize () == 16)
        {
          if (mode.GetCodeRate () == WIFI_CODE_RATE_1_2)
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, //signal spread
                                   phyRate, //PHY rate
                                   16, //m
                                   10, //dFree
                                   11, //adFree
                                   0); //adFreePlusOne
            }
          else
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, //signal spread
                                   phyRate, //PHY rate
                                   16, //m
                                   5, //dFree
                                   8, //adFree
                                   31); //adFreePlusOne
            }
        }
      else if (mode.GetConstellationSize () == 64)
        {
          if (mode.GetCodeRate () == WIFI_CODE_RATE_2_3)
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, //signal spread
                                   phyRate, //PHY rate
                                   64, //m
                                   6, //dFree
                                   1, //adFree
                                   16); //adFreePlusOne
            }
          if (mode.GetCodeRate () == WIFI_CODE_RATE_5_6)
            {
              //Table B.32  in Pâl Frenger et al., "Multi-rate Convolutional Codes".
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, //signal spread
                                   phyRate, //PHY rate
                                   64, //m
                                   4, //dFree
                                   14, //adFree
                                   69); //adFreePlusOne
            }
          else
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, //signal spread
                                   phyRate, //PHY rate
                                   64, //m
                                   5, //dFree
                                   8, //adFree
                                   31); //adFreePlusOne
            }
        }
      else if (mode.GetConstellationSize () == 256)
        {
          if (mode.GetCodeRate () == WIFI_CODE_RATE_5_6)
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, // signal spread
                                   phyRate, //PHY rate
                                   256, // m
                                   4,  // dFree
                                   14,  // adFree
                                   69  // adFreePlusOne
                                   );
            }
          else
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, // signal spread
                                   phyRate, //PHY rate
                                   256, // m
                                   5,  // dFree
                                   8,  // adFree
                                   31  // adFreePlusOne
                                   );
            }
        }
      else if (mode.GetConstellationSize () == 1024)
        {
          if (mode.GetCodeRate () == WIFI_CODE_RATE_5_6)
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, // signal spread
                                   phyRate, //PHY rate
                                   1024, // m
                                   4,  // dFree
                                   14,  // adFree
                                   69  // adFreePlusOne
                                   );
            }
          else
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, // signal spread
                                   phyRate, //PHY rate
                                   1024, // m
                                   5,  // dFree
                                   8,  // adFree
                                   31  // adFreePlusOne
                                   );
            }
        }
      else if (mode.GetConstellationSize () == 4096)
        {
          if (mode.GetCodeRate () == WIFI_CODE_RATE_5_6)
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, // signal spread
                                   mode.GetPhyRate (txVector), //PHY rate
                                   4096, // m
                                   4,  // dFree
                                   14,  // adFree
                                   69  // adFreePlusOne
                                   );
            }
          else
            {
              return GetFecQamBer (snr,
                                   nbits,
                                   txVector.GetChannelWidth () * 1000000, // signal spread
                                   mode.GetPhyRate (txVector), //PHY rate
                                   4096, // m
                                   5,  // dFree
                                   8,  // adFree
                                   31  // adFreePlusOne
                                   );
            }
        }
    }
  return 0;
}

} //namespace ns3
