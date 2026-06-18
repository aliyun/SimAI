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

#ifndef SPECTRUM_CONVERTER_H
#define SPECTRUM_CONVERTER_H

#include <ns3/spectrum-value.h>


namespace ns3 {

/**
 * \ingroup spectrum
 *
 * Class which implements a converter between SpectrumValue which are
 * defined over different SpectrumModel. In more formal terms, this class
 * allows conversion between different function spaces. In practical
 * terms, this allows you to mix different spectrum representation
 * within the same channel, such as a device using a coarse spectrum
 * representation (e.g., one frequency for each IEEE 802.11 channel)
 * and devices using a finer representation (e.g., one frequency for
 * each OFDM subcarrier).
 *
 */
class SpectrumConverter : public SimpleRefCount<SpectrumConverter>
{
public:
  /**
   * Create a SpectrumConverter class that will be able to convert ValueVsFreq
   * instances defined over one SpectrumModel to corresponding ValueVsFreq
   * instances defined over a diffent SpectrumModel
   *
   * @param fromSpectrumModel the SpectrumModel to convert from
   * @param toSpectrumModel the SpectrumModel to convert to
   */
  SpectrumConverter (Ptr<const SpectrumModel> fromSpectrumModel, Ptr<const SpectrumModel> toSpectrumModel);

  SpectrumConverter ();



  /**
   * Convert a particular ValueVsFreq instance to
   *
   * @param vvf the ValueVsFreq instance to be converted
   *
   * @return the converted version of the provided ValueVsFreq
   */
  Ptr<SpectrumValue> Convert (Ptr<const SpectrumValue> vvf) const;


private:
  /**
   * Calculate the coefficient for value conversion between elements
   *
   * @param from BandInfo to convert from
   * @param to  BandInfo to convert to
   *
   * @return the fraction of the value of the "from" BandInfos that is
   * mapped to the "to" BandInfo
   */
  double GetCoefficient (const BandInfo& from, const BandInfo& to) const;

  std::vector<double> m_conversionMatrix; //!< matrix of conversion coefficients stored in Compressed Row Storage format
  std::vector<size_t> m_conversionRowPtr; //!< offset of rows in m_conversionMatrix
  std::vector<size_t> m_conversionColInd; //!< column of each non-zero element in m_conversionMatrix

  Ptr<const SpectrumModel> m_fromSpectrumModel;  //!<  the SpectrumModel this SpectrumConverter instance can convert from
  Ptr<const SpectrumModel> m_toSpectrumModel;    //!<  the SpectrumModel this SpectrumConverter instance can convert to

};



} // namespace ns3



#endif /*  SPECTRUM_CONVERTER_H */


