/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2009 INESC Porto
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Pedro Fortuna  <pedro.fortuna@inescporto.pt> <pedro.fortuna@gmail.com>
//

#ifndef NS3_HISTOGRAM_H
#define NS3_HISTOGRAM_H

#include <vector>
#include <stdint.h>
#include <ostream>

namespace ns3 {

/**
 * \brief Class used to store data and make an histogram of the data frequency.
 *
 * Data are grouped in "bins", i.e., intervals. Each value is assigned to the
 * bin according to the following formula: floor(value/binWidth).
 * Hence, bin \a i groups the data from [i*binWidth, (i+1)binWidth).
 *
 * This class only handles \a positive bins, i.e., it does \a not handles negative data.
 *
 * \todo Add support for negative data.
 *
 * \todo Add method(s) to estimate parameters from the histogram,
 * see http://www.dspguide.com/ch2/4.htm
 *
 */
class Histogram
{
public:

  // --- basic methods ---
  /**
   * \brief Constructor
   * \param binWidth width of the histogram "bin".
   */
  Histogram (double binWidth);
  Histogram ();

  // Methods for Getting the Histogram Results
  /**
   * \brief Returns the number of bins in the histogram.
   * \return the number of bins in the histogram
   */
  uint32_t GetNBins () const;
  /**
   * \brief Returns the bin start, i.e., index*binWidth
   * \param index the bin index
   * \return the bin start
   */
  double GetBinStart (uint32_t index);
  /**
   * \brief Returns the bin end, i.e., (index+1)*binWidth
   * \param index the bin index
   * \return the bin start
   */
  double GetBinEnd (uint32_t index);
  /**
   * \brief Returns the bin width.
   *
   * Note that all the bins have the same width.
   *
   * \param index the bin index
   * \return the bin width
   */
  double GetBinWidth (uint32_t index) const;
  /**
   * \brief Set the bin width.
   *
   * Note that you can change the bin width only if the histogram is empty.
   *
   * \param binWidth the bin width
   */
  void SetDefaultBinWidth (double binWidth);
  /**
   * \brief Get the number of data added to the bin.
   * \param index the bin index
   * \return the number of data added to the bin
   */
  uint32_t GetBinCount (uint32_t index);

  // Method for adding values
  /**
   * \brief Add a value to the histogram
   * \param value the value to add
   */
  void AddValue (double value);

  /**
   * \brief Serializes the results to an std::ostream in XML format.
   * \param os the output stream
   * \param indent number of spaces to use as base indentation level
   * \param elementName name of the element to serialize.
   */
  void SerializeToXmlStream (std::ostream &os, uint16_t indent, std::string elementName) const;


private:
  std::vector<uint32_t> m_histogram; //!< Histogram data
  double m_binWidth; //!< Bin width
};


} // namespace ns3

#endif /* NS3_HISTOGRAM_H */
