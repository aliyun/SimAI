/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Author: Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 */

#ifndef SNR_TO_BLOCK_ERROR_RATE_RECORD_H
#define SNR_TO_BLOCK_ERROR_RATE_RECORD_H

namespace ns3 {

/**
 * \ingroup wimax
 * \brief This class represents a record (handled by SnrToBlockErrorRate manager) that keeps a mapping between
 * an SNR value and its corresponding (1) Bit Error Rate, (2) Block Error Rate, (3) Standard deviation, (4 and 5)
 * confidence interval.
 *
 */
class SNRToBlockErrorRateRecord
{
public:
  /**
   * \brief creates a record and sets its values
   * \param snrValue The SNR value
   * \param bitErrorRate the BER
   * \param BlockErrorRate the BlcER
   * \param sigma2 the standard deviation
   * \param I1 the lower boundary of the confidence interval
   * \param I2 the upper boundary of the confidence interval
   */
  SNRToBlockErrorRateRecord (double snrValue, double bitErrorRate,
                             double BlockErrorRate, double sigma2, double I1, double I2);
  ~SNRToBlockErrorRateRecord (void);

  /**
   * \return the SNR value
   */
  double GetSNRValue (void);
  /**
   * \return the BER value
   */
  double GetBitErrorRate (void);
  /**
   * \return the BlcER value
   */
  double GetBlockErrorRate (void);
  /**
   * \return the standard deviation
   */
  double GetSigma2 (void);
  /**
   * \return the lower boundary of the confidence interval
   */
  double GetI1 (void);
  /**
   * \return the upper boundary of the confidence interval
   */
  double GetI2 (void);
  /**
   * \brief copy a record
   * \return a copy of this record
   */
  SNRToBlockErrorRateRecord *
  Copy ();

  /**
   * \brief set the snr value
   * \param snrValue the SNR value
   */
  void SetSNRValue (double snrValue);
  /**
   * \brief set the BER value
   * \param bitErrorRate the bit error rate value
   */
  void SetBitErrorRate (double bitErrorRate);
  /**
   * \brief set the BlcER value
   * \param blockErrorRate the block error rate value
   */
  void SetBlockErrorRate (double blockErrorRate);
  /**
   * \brief set the lower boundary of the confidence interval
   * \param i1 the lower boundary of the confidence interval
   */
  void SetI1 (double i1);
  /**
   * \brief set the upper boundary of the confidence interval
   * \param i2 the upper boundary of the confidence interval
   */
  void SetI2 (double i2);

private:
  double m_snrValue; ///< SNR value
  double m_bitErrorRate; ///< bit error rate
  double m_blockErrorRate; ///< block error rate
  double m_sigma2; ///< sigma2
  double m_i1; ///< The lower boundary of the confidence interval
  double m_i2; ///< The upper boundary of the confidence interval
};

}

#endif /* SNR_TO_BLOCK_ERROR_RATE_RECORD_H */
