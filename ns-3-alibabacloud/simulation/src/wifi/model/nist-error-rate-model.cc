/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 The Boeing Company
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
 * Authors: Gary Pei <guangyu.pei@boeing.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include <cmath>
#include <bitset>
#include "ns3/log.h"
#include "nist-error-rate-model.h"
#include "wifi-tx-vector.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NistErrorRateModel");

NS_OBJECT_ENSURE_REGISTERED (NistErrorRateModel);

TypeId
NistErrorRateModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::NistErrorRateModel")
    .SetParent<ErrorRateModel> ()
    .SetGroupName ("Wifi")
    .AddConstructor<NistErrorRateModel> ()
  ;
  return tid;
}

NistErrorRateModel::NistErrorRateModel ()
{
}

double
NistErrorRateModel::GetBpskBer (double snr) const
{
  NS_LOG_FUNCTION (this << snr);
  double z = std::sqrt (snr);
  double ber = 0.5 * erfc (z);
  NS_LOG_INFO ("bpsk snr=" << snr << " ber=" << ber);
  return ber;
}

double
NistErrorRateModel::GetQpskBer (double snr) const
{
  NS_LOG_FUNCTION (this << snr);
  double z = std::sqrt (snr / 2.0);
  double ber = 0.5 * erfc (z);
  NS_LOG_INFO ("qpsk snr=" << snr << " ber=" << ber);
  return ber;
}

double
NistErrorRateModel::GetQamBer (uint16_t constellationSize, double snr) const
{
  NS_LOG_FUNCTION (this << constellationSize << snr);
  NS_ASSERT (std::bitset<16> (constellationSize).count () == 1); //constellationSize has to be a power of 2
  double z = std::sqrt (snr / ((2 * (constellationSize - 1)) / 3));
  uint8_t bitsPerSymbol = std::sqrt (constellationSize);
  double ber = ((bitsPerSymbol - 1) / (bitsPerSymbol * std::log2 (bitsPerSymbol))) * erfc (z);
  NS_LOG_INFO (constellationSize << "-QAM: snr=" << snr << " ber=" << ber);
  return ber;
}

double
NistErrorRateModel::GetFecBpskBer (double snr, uint64_t nbits, uint8_t bValue) const
{
  NS_LOG_FUNCTION (this << snr << nbits << +bValue);
  double ber = GetBpskBer (snr);
  if (ber == 0.0)
    {
      return 1.0;
    }
  double pe = CalculatePe (ber, bValue);
  pe = std::min (pe, 1.0);
  double pms = std::pow (1 - pe, nbits);
  return pms;
}

double
NistErrorRateModel::GetFecQpskBer (double snr, uint64_t nbits, uint8_t bValue) const
{
  NS_LOG_FUNCTION (this << snr << nbits << +bValue);
  double ber = GetQpskBer (snr);
  if (ber == 0.0)
    {
      return 1.0;
    }
  double pe = CalculatePe (ber, bValue);
  pe = std::min (pe, 1.0);
  double pms = std::pow (1 - pe, nbits);
  return pms;
}

double
NistErrorRateModel::CalculatePe (double p, uint8_t bValue) const
{
  NS_LOG_FUNCTION (this << p << +bValue);
  double D = std::sqrt (4.0 * p * (1.0 - p));
  double pe = 1.0;
  if (bValue == 1)
    {
      //code rate 1/2, use table 3.1.1
      pe = 0.5 * (36.0 * std::pow (D, 10)
                  + 211.0 * std::pow (D, 12)
                  + 1404.0 * std::pow (D, 14)
                  + 11633.0 * std::pow (D, 16)
                  + 77433.0 * std::pow (D, 18)
                  + 502690.0 * std::pow (D, 20)
                  + 3322763.0 * std::pow (D, 22)
                  + 21292910.0 * std::pow (D, 24)
                  + 134365911.0 * std::pow (D, 26));
    }
  else if (bValue == 2)
    {
      //code rate 2/3, use table 3.1.2
      pe = 1.0 / (2.0 * bValue) *
        (3.0 * std::pow (D, 6)
         + 70.0 * std::pow (D, 7)
         + 285.0 * std::pow (D, 8)
         + 1276.0 * std::pow (D, 9)
         + 6160.0 * std::pow (D, 10)
         + 27128.0 * std::pow (D, 11)
         + 117019.0 * std::pow (D, 12)
         + 498860.0 * std::pow (D, 13)
         + 2103891.0 * std::pow (D, 14)
         + 8784123.0 * std::pow (D, 15));
    }
  else if (bValue == 3)
    {
      //code rate 3/4, use table 3.1.2
      pe = 1.0 / (2.0 * bValue) *
        (42.0 * std::pow (D, 5)
         + 201.0 * std::pow (D, 6)
         + 1492.0 * std::pow (D, 7)
         + 10469.0 * std::pow (D, 8)
         + 62935.0 * std::pow (D, 9)
         + 379644.0 * std::pow (D, 10)
         + 2253373.0 * std::pow (D, 11)
         + 13073811.0 * std::pow (D, 12)
         + 75152755.0 * std::pow (D, 13)
         + 428005675.0 * std::pow (D, 14));
    }
  else if (bValue == 5)
    {
      //code rate 5/6, use table V from D. Haccoun and G. Begin, "High-Rate Punctured Convolutional Codes
      //for Viterbi Sequential Decoding", IEEE Transactions on Communications, Vol. 32, Issue 3, pp.315-319.
      pe = 1.0 / (2.0 * bValue) *
        (92.0 * std::pow (D, 4.0)
         + 528.0 * std::pow (D, 5.0)
         + 8694.0 * std::pow (D, 6.0)
         + 79453.0 * std::pow (D, 7.0)
         + 792114.0 * std::pow (D, 8.0)
         + 7375573.0 * std::pow (D, 9.0)
         + 67884974.0 * std::pow (D, 10.0)
         + 610875423.0 * std::pow (D, 11.0)
         + 5427275376.0 * std::pow (D, 12.0)
         + 47664215639.0 * std::pow (D, 13.0));
    }
  else
    {
      NS_ASSERT (false);
    }
  return pe;
}

double
NistErrorRateModel::GetFecQamBer (uint16_t constellationSize, double snr, uint64_t nbits, uint8_t bValue) const
{
  NS_LOG_FUNCTION (this << constellationSize << snr << nbits << +bValue);
  double ber = GetQamBer (constellationSize, snr);
  if (ber == 0.0)
    {
      return 1.0;
    }
  double pe = CalculatePe (ber, bValue);
  pe = std::min (pe, 1.0);
  double pms = std::pow (1 - pe, nbits);
  return pms;
}

uint8_t
NistErrorRateModel::GetBValue (WifiCodeRate codeRate) const
{
  switch (codeRate)
    {
    case WIFI_CODE_RATE_3_4:
      return 3;
    case WIFI_CODE_RATE_2_3:
      return 2;
    case WIFI_CODE_RATE_1_2:
      return 1;
    case WIFI_CODE_RATE_5_6:
      return 5;
    case WIFI_CODE_RATE_UNDEFINED:
    default:
      NS_FATAL_ERROR ("Unknown code rate");
      break;
    }
  return 0;
}

double
NistErrorRateModel::DoGetChunkSuccessRate (WifiMode mode, const WifiTxVector& txVector, double snr, uint64_t nbits, uint8_t numRxAntennas, WifiPpduField field, uint16_t staId) const
{
  NS_LOG_FUNCTION (this << mode << snr << nbits << +numRxAntennas << field << staId);
  if (mode.GetModulationClass () >= WIFI_MOD_CLASS_ERP_OFDM)
    {
      if (mode.GetConstellationSize () == 2)
        {
          return GetFecBpskBer (snr, nbits, GetBValue (mode.GetCodeRate ()));
        }
      else if (mode.GetConstellationSize () == 4)
        {
          return GetFecQpskBer (snr, nbits, GetBValue (mode.GetCodeRate ()));
        }
      else
        {
          return GetFecQamBer (mode.GetConstellationSize (), snr, nbits, GetBValue (mode.GetCodeRate ()));
        }
    }
  return 0;
}

} //namespace ns3
