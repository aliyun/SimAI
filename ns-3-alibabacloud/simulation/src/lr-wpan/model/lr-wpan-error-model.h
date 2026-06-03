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
#ifndef LR_WPAN_ERROR_MODEL_H
#define LR_WPAN_ERROR_MODEL_H

#include <ns3/object.h>

namespace ns3 {

/**
 * \ingroup lr-wpan
 *
 * Model the error rate for IEEE 802.15.4 2.4 GHz AWGN channel for OQPSK
 * the model description can be found in IEEE Std 802.15.4-2006, section
 * E.4.1.7
 */
class LrWpanErrorModel : public Object
{
public:
  /**
   * Get the type ID.
   *
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  LrWpanErrorModel (void);

  /**
   * Return chunk success rate for given SNR.
   *
   * \return success rate (i.e. 1 - chunk error rate)
   * \param snr SNR expressed as a power ratio (i.e. not in dB)
   * \param nbits number of bits in the chunk
   */
  double GetChunkSuccessRate (double snr, uint32_t nbits) const;

private:
  /**
   * Array of precalculated binomial coefficients.
   */
  double m_binomialCoefficients[17];

};


} // namespace ns3

#endif /* LR_WPAN_ERROR_MODEL_H */
