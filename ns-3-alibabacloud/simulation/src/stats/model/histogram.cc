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

#include <cmath>

#include "histogram.h"
#include "ns3/simulator.h"
#include "ns3/log.h"

#define DEFAULT_BIN_WIDTH       1
// #define RESERVED_BINS_INC	10


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Histogram");

// uint32_t 
// Histogram::GetSize () const
// {
//   return m_histogram.size ();
// }

uint32_t 
Histogram::GetNBins () const
{
  return m_histogram.size ();
}

double 
Histogram::GetBinStart (uint32_t index)
{
  return index*m_binWidth;
}

double 
Histogram::GetBinEnd (uint32_t index)
{
  return (index + 1) * m_binWidth;
}

double 
Histogram::GetBinWidth (uint32_t index) const
{
  return m_binWidth;
}

void 
Histogram::SetDefaultBinWidth (double binWidth)
{
  NS_ASSERT (m_histogram.size () == 0); //we can only change the bin width if no values were added
  m_binWidth = binWidth;
}

uint32_t 
Histogram::GetBinCount (uint32_t index) 
{
  NS_ASSERT (index < m_histogram.size ());
  return m_histogram[index];
}

void 
Histogram::AddValue (double value)
{
  uint32_t index = (uint32_t)std::floor (value/m_binWidth);

  //check if we need to resize the vector
  NS_LOG_DEBUG ("AddValue: index=" << index << ", m_histogram.size()=" << m_histogram.size ());

  if (index >= m_histogram.size ())
    {
      m_histogram.resize (index + 1, 0);
    }
  m_histogram[index]++;
}

Histogram::Histogram (double binWidth)
{
  m_binWidth = binWidth;
}

Histogram::Histogram ()
{
  m_binWidth = DEFAULT_BIN_WIDTH;
}

void
Histogram::SerializeToXmlStream (std::ostream &os, uint16_t indent, std::string elementName) const
{
  os << std::string ( indent, ' ' ) << "<" << elementName // << " binWidth=\"" << m_binWidth << "\""
     << " nBins=\"" << m_histogram.size () << "\""
     << " >\n";
  indent += 2;

#if 1 // two alternative forms of representing bin data, one more verbose than the other one
  for (uint32_t index = 0; index < m_histogram.size (); index++)
    {
      if (m_histogram[index])
        {
          os << std::string ( indent, ' ' );
          os << "<bin"
             << " index=\"" << (index) << "\""
             << " start=\"" << (index*m_binWidth) << "\""
             << " width=\"" << m_binWidth << "\""
             << " count=\"" << m_histogram[index] << "\""
             << " />\n";
        }
    }
#else
  os << std::string ( indent+2, ' ' );
  for (uint32_t index = 0; index < m_histogram.size (); index++)
    {
      if (index > 0)
        {
          os << " ";
        }
      os << m_histogram[index];
    }
  os << "\n";
#endif
  indent -= 2;
  os << std::string ( indent, ' ' ) << "</" << elementName << ">\n";
}




} // namespace ns3


