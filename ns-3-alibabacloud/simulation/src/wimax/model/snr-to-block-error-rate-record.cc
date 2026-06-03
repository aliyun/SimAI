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

#include "ns3/simulator.h"
#include "ns3/snr-to-block-error-rate-record.h"
#include "ns3/assert.h"

namespace ns3 {

SNRToBlockErrorRateRecord::SNRToBlockErrorRateRecord (double snrValue,
                                                      double bitErrorRate, double blockErrorRate, double sigma2, double I1,
                                                      double I2)
{
  m_snrValue = snrValue;
  m_bitErrorRate = bitErrorRate;
  m_blockErrorRate = blockErrorRate;
  m_sigma2 = sigma2;
  m_i1 = I1;
  m_i2 = I2;
}

SNRToBlockErrorRateRecord *
SNRToBlockErrorRateRecord::Copy ()
{
  return (new SNRToBlockErrorRateRecord (m_snrValue, m_bitErrorRate,
                                         m_blockErrorRate, m_sigma2, m_i1, m_i2));

}

double
SNRToBlockErrorRateRecord::GetSNRValue (void)
{
  return m_snrValue;
}

SNRToBlockErrorRateRecord::~SNRToBlockErrorRateRecord (void)
{
  m_snrValue = 0;
  m_bitErrorRate = 0;
  m_blockErrorRate = 0;
  m_sigma2 = 0;
  m_i1 = 0;
  m_i2 = 0;
}

double
SNRToBlockErrorRateRecord::GetBitErrorRate (void)
{
  return m_bitErrorRate;
}
double
SNRToBlockErrorRateRecord::GetBlockErrorRate (void)
{
  return m_blockErrorRate;
}
double
SNRToBlockErrorRateRecord::GetSigma2 (void)
{
  return m_sigma2;
}
double
SNRToBlockErrorRateRecord::GetI1 (void)
{
  return m_i1;
}
double
SNRToBlockErrorRateRecord::GetI2 (void)
{
  return m_i2;
}

void
SNRToBlockErrorRateRecord::SetSNRValue (double snrValue)
{
  m_snrValue = snrValue;
}
void
SNRToBlockErrorRateRecord::SetBitErrorRate (double bitErrorRate)
{
  m_bitErrorRate = bitErrorRate;
}
void
SNRToBlockErrorRateRecord::SetBlockErrorRate (double blockErrorRate)
{
  m_blockErrorRate = blockErrorRate;
}
void
SNRToBlockErrorRateRecord::SetI1 (double i1)
{
  m_i1 = i1;
}
void
SNRToBlockErrorRateRecord::SetI2 (double i2)
{
  m_i2 = i2;
}

}
