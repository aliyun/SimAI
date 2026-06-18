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

#include <ns3/spectrum-converter.h>
#include <ns3/assert.h>
#include <ns3/log.h>
#include <algorithm>



namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SpectrumConverter");

SpectrumConverter::SpectrumConverter ()
{
}

SpectrumConverter::SpectrumConverter (Ptr<const SpectrumModel> fromSpectrumModel, Ptr<const SpectrumModel> toSpectrumModel)
{
  NS_LOG_FUNCTION (this);
  m_fromSpectrumModel = fromSpectrumModel;
  m_toSpectrumModel = toSpectrumModel;

  size_t rowPtr = 0;
  for (Bands::const_iterator toit = toSpectrumModel->Begin (); toit != toSpectrumModel->End (); ++toit)
    {
      size_t colInd = 0;
      for (Bands::const_iterator fromit = fromSpectrumModel->Begin (); fromit != fromSpectrumModel->End (); ++fromit)
        {
          double c = GetCoefficient (*fromit, *toit);
          NS_LOG_LOGIC ("(" << fromit->fl << ","  << fromit->fh << ")"
                            << " --> " <<
                        "(" << toit->fl << "," << toit->fh << ")"
                            << " = " << c);
          if (c > 0)
            {
              m_conversionMatrix.push_back (c);
              m_conversionColInd.push_back (colInd);
              rowPtr++;
            }
          colInd++;
        }
      m_conversionRowPtr.push_back (rowPtr);
    }

}


double SpectrumConverter::GetCoefficient (const BandInfo& from, const BandInfo& to) const
{
  NS_LOG_FUNCTION (this);
  double coeff = std::min (from.fh, to.fh) - std::max (from.fl, to.fl);
  coeff = std::max (0.0, coeff);
  coeff = std::min (1.0, coeff / (to.fh - to.fl));
  return coeff;
}



Ptr<SpectrumValue>
SpectrumConverter::Convert (Ptr<const SpectrumValue> fvvf) const
{
  NS_ASSERT ( *(fvvf->GetSpectrumModel ()) == *m_fromSpectrumModel);

  Ptr<SpectrumValue> tvvf = Create<SpectrumValue> (m_toSpectrumModel);

  Values::iterator tvit = tvvf->ValuesBegin ();
  size_t i = 0; // Index of conversion coefficient

  for (std::vector<size_t>::const_iterator convIt = m_conversionRowPtr.begin ();
       convIt != m_conversionRowPtr.end ();
       ++convIt)
    {
      double sum = 0;
      while (i < *convIt)
        {
          sum += (*fvvf)[m_conversionColInd.at (i)] * m_conversionMatrix.at (i);
          i++;
        }
      *tvit = sum;
      ++tvit;
    }

  return tvvf;
}





} // namespace ns3
