/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 SIGNET LAB. Department of Information Engineering (DEI), University of Padua
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
 *
 * Original Work Authors:
 *      Marco Mezzavilla <mezzavil@dei.unipd.it>
 *      Giovanni Tomasi <tomasigv@gmail.com>
 * Original Work Acknowldegments:
 *      This work was supported by the MEDIEVAL (MultiMEDia transport
 *      for mobIlE Video AppLications) project, which is a
 *      medium-scale focused research project (STREP) of the 7th
 *      Framework Programme (FP7)
 *
 * Subsequent integration in LENA and extension done by:
 *      Marco Miozzo <marco.miozzo@cttc.es>
 */ 

#ifndef LTE_MI_ERROR_MODEL_H
#define LTE_MI_ERROR_MODEL_H


#include <list>
#include <vector>
#include <ns3/ptr.h>
#include <stdint.h>
#include <ns3/spectrum-value.h>
#include <ns3/lte-harq-phy.h>




namespace ns3 {
  
  /// PDCCH PCFICH curve size
  const uint16_t PDCCH_PCFICH_CURVE_SIZE = 46;
  /// MI map QPSK size 
  const uint16_t MI_MAP_QPSK_SIZE = 797;
  /// MI map 16QAM size
  const uint16_t MI_MAP_16QAM_SIZE = 994;
  /// MI map 64QAM size
  const uint16_t MI_MAP_64QAM_SIZE = 752;
  /// MI QPSK maximum ID
  const uint16_t MI_QPSK_MAX_ID = 9;
  /// MI 16QAM maximum ID
  const uint16_t MI_16QAM_MAX_ID = 16;
  /// MI 64QAM maximum ID
  const uint16_t MI_64QAM_MAX_ID = 28;  // 29,30 and 31 are reserved
  /// MI QPSK BLER maximum ID
  const uint16_t MI_QPSK_BLER_MAX_ID = 12; // MI_QPSK_MAX_ID + 3 RETX
  /// MI 16QAM BLER maximum ID
  const uint16_t MI_16QAM_BLER_MAX_ID = 22;
  /// MI 64QAM BLER maximum ID
  const uint16_t MI_64QAM_BLER_MAX_ID = 37;

/// TbStats_t structure
struct TbStats_t
{
  double tbler; ///< Transport block BLER
  double mi; ///< Mutual information
};
  


/**
 * This class provides the BLER estimation based on mutual information metrics
 */
class LteMiErrorModel
{

public:

  /** 
   * \brief find the mmib (mean mutual information per bit) for different modulations of the specified TB
   * \param sinr the perceived sinr values in the whole bandwidth in Watt
   * \param map the active RBs for the TB
   * \param mcs the MCS of the TB
   * \return the mmib
   */
  static double Mib (const SpectrumValue& sinr, const std::vector<int>& map, uint8_t mcs);
  /** 
   * \brief map the mmib (mean mutual information per bit) for different MCS
   * \param mib mean mutual information per bit of a code-block
   * \param ecrId Effective Code Rate ID
   * \param cbSize the size of the CB
   * \return the code block error rate
   */
  static double MappingMiBler (double mib, uint8_t ecrId, uint16_t cbSize);

  /**
   * \brief run the error-model algorithm for the specified TB
   * \param sinr the perceived sinr values in the whole bandwidth in Watt
   * \param map the active RBs for the TB
   * \param size the size in bytes of the TB
   * \param mcs the MCS of the TB
   * \param miHistory MI of past transmissions (in case of retx)
   * \return the TB error rate and MI
   */
  static TbStats_t GetTbDecodificationStats (const SpectrumValue& sinr, const std::vector<int>& map, uint16_t size, uint8_t mcs, HarqProcessInfoList_t miHistory);
  
  /** 
  * \brief run the error-model algorithm for the specified PCFICH+PDCCH channels
  * \param sinr the perceived sinr values in the whole bandwidth in Watt
  * \return the decodification error of the PCFICH+PDCCH channels
  */  
  static double GetPcfichPdcchError (const SpectrumValue& sinr);


//private:



};


} // namespace ns3

#endif /* LTE_MI_ERROR_MODEL_H */
