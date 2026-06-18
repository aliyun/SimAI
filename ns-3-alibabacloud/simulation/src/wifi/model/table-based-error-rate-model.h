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

#ifndef TABLE_BASED_ERROR_RATE_MODEL_H
#define TABLE_BASED_ERROR_RATE_MODEL_H

#include "ns3/error-rate-tables.h"
#include "error-rate-model.h"
#include "wifi-mode.h"
#include <optional>

namespace ns3 {

class WifiTxVector;

/*
 * \ingroup wifi
 * \brief the interface for the table-driven OFDM error model
 *
 */
class TableBasedErrorRateModel : public ErrorRateModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TableBasedErrorRateModel ();
  ~TableBasedErrorRateModel ();

  /**
   * \brief Utility function to convert WifiMode to an MCS value
   * \param mode the WifiMode
   * \return the equivalent MCS value, if found
   */
  static std::optional<uint8_t> GetMcsForMode (WifiMode mode);


private:
  double DoGetChunkSuccessRate (WifiMode mode, const WifiTxVector& txVector, double snr, uint64_t nbits,
                                uint8_t numRxAntennas, WifiPpduField field, uint16_t staId) const override;

  /**
   * Round SNR (in dB) to the specified precision
   *
   * \param snr the SNR to round  (in dB)
   * \param precision the precision to use
   * \return the rounded SNR (in dB) to the specified precision
   */
  double RoundSnr (double snr, double precision) const;

  /**
   * Fetch the frame success rate for a given Wi-Fi mode, TXVECTOR, SNR and frame size.
   * \param mode the Wi-Fi mode
   * \param txVector the TXVECTOR
   * \param snr the SNR (linear scale)
   * \param nbits the number of bits
   * \return the frame success rate for a given Wi-Fi mode, TXVECTOR, SNR and frame size
   */
  double FetchFsr (WifiMode mode, const WifiTxVector& txVector, double snr, uint64_t nbits) const;

  Ptr<ErrorRateModel> m_fallbackErrorModel; //!< Error rate model to fallback to if no value is found in the table

  uint64_t m_threshold; //!< Threshold in bytes over which the table for large size frames is used
};

} //namespace ns3

#endif /* TABLE_BASED_ERROR_RATE_MODEL_H */
