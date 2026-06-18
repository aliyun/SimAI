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
 * Author: Gary Pei <guangyu.pei@boeing.com>
 */

#ifndef NIST_ERROR_RATE_MODEL_H
#define NIST_ERROR_RATE_MODEL_H

#include "error-rate-model.h"
#include "wifi-mode.h"

namespace ns3 {

/**
 * \ingroup wifi
 *
 * A model for the error rate for different modulations.  For OFDM modulation,
 * the model description and validation can be found in
 * http://www.nsnam.org/~pei/80211ofdm.pdf.  For DSSS modulations (802.11b),
 * the model uses the DsssErrorRateModel.
 */
class NistErrorRateModel : public ErrorRateModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  NistErrorRateModel ();


private:
  double DoGetChunkSuccessRate (WifiMode mode, const WifiTxVector& txVector, double snr, uint64_t nbits,
                                uint8_t numRxAntennas, WifiPpduField field, uint16_t staId) const override;
  /**
   * Return the bValue such that coding rate = bValue / (bValue + 1).
   *
   * \param codeRate the coding rate
   *
   * \return the bValue such that coding rate = bValue / (bValue + 1)
   */
  uint8_t GetBValue (WifiCodeRate codeRate) const;
  /**
   * Return the coded BER for the given p and b.
   *
   * \param p the SNR ratio (in linear scale)
   * \param bValue such that coding rate = bValue / (bValue + 1)
   *
   * \return the coded BER
   */
  double CalculatePe (double p, uint8_t bValue) const;
  /**
   * Return BER of BPSK at the given SNR.
   *
   * \param snr SNR ratio (in linear scale)
   *
   * \return BER of BPSK at the given SNR
   */
  double GetBpskBer (double snr) const;
  /**
   * Return BER of QPSK at the given SNR.
   *
   * \param snr SNR ratio (in linear scale)
   *
   * \return BER of QPSK at the given SNR
   */
  double GetQpskBer (double snr) const;
  /**
   * Return BER of QAM for a given constellation size at the given SNR.
   *
   * \param constellationSize the constellation size (M)
   * \param snr SNR ratio (in linear scale)
   * \return BER of QAM for a given constellation size at the given SNR
   */
  double GetQamBer (uint16_t constellationSize, double snr) const;
  /**
   * Return BER of BPSK at the given SNR after applying FEC.
   *
   * \param snr SNR ratio (in linear scale)
   * \param nbits the number of bits in the chunk
   * \param bValue the bValue such that coding rate = bValue / (bValue + 1)
   *
   * \return BER of BPSK at the given SNR after applying FEC
   */
  double GetFecBpskBer (double snr, uint64_t nbits, uint8_t bValue) const;
  /**
   * Return BER of QPSK at the given SNR after applying FEC.
   *
   * \param snr SNR ratio (in linear scale)
   * \param nbits the number of bits in the chunk
   * \param bValue the bValue such that coding rate = bValue / (bValue + 1)
   *
   * \return BER of QPSK at the given SNR after applying FEC
   */
  double GetFecQpskBer (double snr, uint64_t nbits, uint8_t bValue) const;
  /**
   * Return BER of QAM for a given constellation size at the given SNR after applying FEC.
   *
   * \param constellationSize the constellation size (M)
   * \param snr SNR ratio (in linear scale)
   * \param nbits the number of bits in the chunk
   * \param bValue the bValue such that coding rate = bValue / (bValue + 1)
   *
   * \return BER of QAM for a given constellation size at the given SNR after applying FEC
   */
  double GetFecQamBer (uint16_t constellationSize, double snr, uint64_t nbits, uint8_t bValue) const;
};

} //namespace ns3

#endif /* NIST_ERROR_RATE_MODEL_H */
