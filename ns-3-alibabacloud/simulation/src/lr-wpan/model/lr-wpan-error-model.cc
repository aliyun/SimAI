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
#include "lr-wpan-error-model.h"
#include <ns3/log.h>

#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LrWpanErrorModel");

NS_OBJECT_ENSURE_REGISTERED (LrWpanErrorModel);

TypeId
LrWpanErrorModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LrWpanErrorModel")
    .SetParent<Object> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<LrWpanErrorModel> ()
  ;
  return tid;
}

LrWpanErrorModel::LrWpanErrorModel (void)
{
  m_binomialCoefficients[0]  = 1;
  m_binomialCoefficients[1]  = -16;
  m_binomialCoefficients[2]  = 120;
  m_binomialCoefficients[3]  = -560;
  m_binomialCoefficients[4]  = 1820;
  m_binomialCoefficients[5]  = -4368;
  m_binomialCoefficients[6]  = 8008;
  m_binomialCoefficients[7]  = -11440;
  m_binomialCoefficients[8]  = 12870;
  m_binomialCoefficients[9]  = -11440;
  m_binomialCoefficients[10] = 8008;
  m_binomialCoefficients[11] = -4368;
  m_binomialCoefficients[12] = 1820;
  m_binomialCoefficients[13] = -560;
  m_binomialCoefficients[14] = 120;
  m_binomialCoefficients[15] = -16;
  m_binomialCoefficients[16] = 1;
}

double
LrWpanErrorModel::GetChunkSuccessRate (double snr, uint32_t nbits) const
{
  double ber = 0.0;

  for (uint32_t k = 2; k <= 16; k++)
    {
      ber += m_binomialCoefficients[k] * exp (20.0 * snr * (1.0 / k - 1.0));
    }

  ber = ber * 8.0 / 15.0 / 16.0;

  ber = std::min (ber, 1.0);
  double retval = pow (1.0 - ber, nbits);
  return retval;
}

} // namespace ns3
