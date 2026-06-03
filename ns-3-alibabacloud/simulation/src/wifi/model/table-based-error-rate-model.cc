/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 University of Washington
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
 * Authors: Rohan Patidar <rpatidar@uw.edu>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include <cmath>
#include <algorithm>
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/uinteger.h"
#include "table-based-error-rate-model.h"
#include "wifi-utils.h"
#include "wifi-tx-vector.h"
#include "ns3/dsss-error-rate-model.h"
#include "yans-error-rate-model.h"

namespace ns3 {

static const double SNR_PRECISION = 2; //!< precision for SNR
static const double TABLED_BASED_ERROR_MODEL_PRECISION = 1e-5; //!< precision for PER

NS_OBJECT_ENSURE_REGISTERED (TableBasedErrorRateModel);

NS_LOG_COMPONENT_DEFINE ("TableBasedErrorRateModel");

TypeId
TableBasedErrorRateModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TableBasedErrorRateModel")
    .SetParent<ErrorRateModel> ()
    .SetGroupName ("Wifi")
    .AddConstructor<TableBasedErrorRateModel> ()
    .AddAttribute ("FallbackErrorRateModel",
                   "Ptr to the fallback error rate model to be used when no matching value is found in a table",
                   PointerValue (CreateObject<YansErrorRateModel> ()),
                   MakePointerAccessor (&TableBasedErrorRateModel::m_fallbackErrorModel),
                   MakePointerChecker <ErrorRateModel> ())
    .AddAttribute ("SizeThreshold",
                   "Threshold in bytes over which the table for large size frames is used",
                   UintegerValue (400),
                   MakeUintegerAccessor (&TableBasedErrorRateModel::m_threshold),
                   MakeUintegerChecker<uint64_t> ())
  ;
  return tid;
}

TableBasedErrorRateModel::TableBasedErrorRateModel ()
{
  NS_LOG_FUNCTION (this);
}

TableBasedErrorRateModel::~TableBasedErrorRateModel ()
{
  NS_LOG_FUNCTION (this);
  m_fallbackErrorModel = 0;
}


double
TableBasedErrorRateModel::RoundSnr (double snr, double precision) const
{
  NS_LOG_FUNCTION (this << snr);
  double multiplier = std::round (std::pow (10.0, precision));
  return std::floor (snr * multiplier + 0.5) / multiplier;
}

std::optional<uint8_t>
TableBasedErrorRateModel::GetMcsForMode (WifiMode mode)
{
  std::optional<uint8_t> mcs;
  WifiModulationClass modulationClass = mode.GetModulationClass ();
  WifiCodeRate codeRate = mode.GetCodeRate ();
  uint16_t constellationSize = mode.GetConstellationSize ();

  if (modulationClass == WIFI_MOD_CLASS_OFDM || modulationClass == WIFI_MOD_CLASS_ERP_OFDM)
    {
      if (constellationSize == 2)    // BPSK
        {
          if (codeRate == WIFI_CODE_RATE_1_2)
            {
              mcs = 0;
            }
          if (codeRate == WIFI_CODE_RATE_3_4)
            {
              // No MCS uses BPSK and a Coding Rate of 3/4
            }
        }
      else if (constellationSize == 4)    // QPSK
        {
          if (codeRate == WIFI_CODE_RATE_1_2)
            {
              mcs = 1;
            }
          else if (codeRate == WIFI_CODE_RATE_3_4)
            {
              mcs = 2;
            }
        }
      else if (constellationSize == 16)    // 16-QAM
        {
          if (codeRate == WIFI_CODE_RATE_1_2)
            {
              mcs = 3;
            }
          else if (codeRate == WIFI_CODE_RATE_3_4)
            {
              mcs = 4;
            }
        }
      else if (constellationSize == 64)    // 64-QAM
        {
          if (codeRate == WIFI_CODE_RATE_2_3)
            {
              mcs = 5;
            }
          else if (codeRate == WIFI_CODE_RATE_3_4)
            {
              mcs = 6;
            }
        }
    }
  else if (modulationClass >= WIFI_MOD_CLASS_HT)
    {
      mcs = mode.GetMcsValue ();
    }
  return mcs;
}

double
TableBasedErrorRateModel::DoGetChunkSuccessRate (WifiMode mode, const WifiTxVector& txVector, double snr, uint64_t nbits, uint8_t numRxAntennas, WifiPpduField field, uint16_t staId) const
{
  NS_LOG_FUNCTION (this << mode << txVector << snr << nbits << +numRxAntennas << field << staId);
  uint64_t size = std::max<uint64_t> (1, (nbits / 8));
  double roundedSnr = RoundSnr (RatioToDb (snr), SNR_PRECISION);
  uint8_t mcs;
  if (auto ret = GetMcsForMode (mode); ret.has_value ())
    {
      mcs = ret.value ();
    }
  else
    {
      NS_LOG_DEBUG ("No MCS found for mode " << mode << ": use fallback error rate model");
      return m_fallbackErrorModel->GetChunkSuccessRate (mode, txVector, snr, nbits, staId);
    }
  bool ldpc = txVector.IsLdpc ();
  NS_LOG_FUNCTION (this << +mcs << roundedSnr << size << ldpc);

  // HT: for MCS greater than 7, use 0 - 7 curves for data rate
  if (mode.GetModulationClass () == WIFI_MOD_CLASS_HT)
    {
      mcs = mcs % 8;
    }

  if (mcs >= (ldpc ? ERROR_TABLE_LDPC_MAX_NUM_MCS : ERROR_TABLE_BCC_MAX_NUM_MCS))
    {
      NS_LOG_WARN ("Table missing for MCS: " << +mcs << " in TableBasedErrorRateModel: use fallback error rate model");
      return m_fallbackErrorModel->GetChunkSuccessRate (mode, txVector, snr, nbits, staId);
    }

  auto errorTable = (ldpc ? AwgnErrorTableLdpc1458 : (size < m_threshold ? AwgnErrorTableBcc32 : AwgnErrorTableBcc1458));
  const auto& itVector = errorTable[mcs];
  auto itTable = std::find_if (itVector.cbegin (), itVector.cend (),
      [&roundedSnr](const std::pair<double, double>& element) {
          return element.first == roundedSnr;
      });
  double minSnr = itVector.cbegin ()->first;
  double maxSnr = (--itVector.cend ())->first;
  double per;
  if (itTable == itVector.cend ())
    {
      if (roundedSnr < minSnr)
        {
          per = 1.0;
        }
      else if (roundedSnr > maxSnr)
        {
          per = 0.0;
        }
      else
        {
          double a = 0.0, b = 0.0, previousSnr = 0.0, nextSnr = 0.0;
          for (auto i = itVector.cbegin (); i != itVector.cend (); ++i)
            {
              if (i->first < roundedSnr)
                {
                  previousSnr = i->first;
                  a = i->second;
                }
              else
                {
                  nextSnr = i->first;
                  b = i->second;
                  break;
                }
            }
          per = a + (roundedSnr - previousSnr) * (b - a) / (nextSnr - previousSnr);
        }
    }
  else
    {
      per = itTable->second;
    }

  uint16_t tableSize = (ldpc ? ERROR_TABLE_LDPC_FRAME_SIZE : (size < m_threshold ? ERROR_TABLE_BCC_SMALL_FRAME_SIZE : ERROR_TABLE_BCC_LARGE_FRAME_SIZE));
  if (size != tableSize)
    {
      // From IEEE document 11-14/0803r1 (Packet Length for Box 0 Calibration)
      per = (1.0 - std::pow ((1 - per), (static_cast<double> (size) / tableSize)));
    }

  if (per < TABLED_BASED_ERROR_MODEL_PRECISION)
    {
      per = 0.0;
    }

  return 1.0 - per;
}

} //namespace ns3
