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
 * Modified by: Marco Miozzo <mmiozzo@cttc.es> convert to
 *               LteSpectrumSignalParametersDlCtrlFrame framework
 */


#include <ns3/object-factory.h>
#include <ns3/log.h>
#include <ns3/double.h>
#include <ns3/simulator.h>
#include <ns3/trace-source-accessor.h>
#include <ns3/lte-spectrum-signal-parameters.h>
#include <ns3/antenna-model.h>

#include "rem-spectrum-phy.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RemSpectrumPhy");

NS_OBJECT_ENSURE_REGISTERED (RemSpectrumPhy);

RemSpectrumPhy::RemSpectrumPhy ()
  : m_mobility (0),
    m_referenceSignalPower (0),
    m_sumPower (0),
    m_active (true),
    m_useDataChannel (false),
    m_rbId (-1)
{
  NS_LOG_FUNCTION (this);
}



RemSpectrumPhy::~RemSpectrumPhy ()
{
  NS_LOG_FUNCTION (this);
}

void
RemSpectrumPhy::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_mobility = 0;
  SpectrumPhy::DoDispose ();
}

TypeId
RemSpectrumPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RemSpectrumPhy")
    .SetParent<SpectrumPhy> ()
    .SetGroupName("Lte")
    .AddConstructor<RemSpectrumPhy> ()
  ;
  return tid;
}



void
RemSpectrumPhy::SetChannel (Ptr<SpectrumChannel> c)
{
  // this is a no-op, RemSpectrumPhy does not transmit hence it does not need a reference to the channel
}

void
RemSpectrumPhy::SetMobility (Ptr<MobilityModel> m)
{
  NS_LOG_FUNCTION (this << m);
  m_mobility = m;
}

void
RemSpectrumPhy::SetDevice (Ptr<NetDevice> d)
{
  NS_LOG_FUNCTION (this << d);
  // this is a no-op, RemSpectrumPhy does not handle any data hence it does not support the use of a NetDevice
}

Ptr<MobilityModel>
RemSpectrumPhy::GetMobility () const
{
  return m_mobility;
}

Ptr<NetDevice>
RemSpectrumPhy::GetDevice () const
{
  return 0;
}

Ptr<const SpectrumModel>
RemSpectrumPhy::GetRxSpectrumModel () const
{
  return m_rxSpectrumModel;
}

Ptr<Object>
RemSpectrumPhy::GetAntenna () const
{
  return 0;
}


void
RemSpectrumPhy::StartRx (Ptr<SpectrumSignalParameters> params)
{
  NS_LOG_FUNCTION ( this << params);

  if (m_active)
    {
      if (m_useDataChannel)
        {
          Ptr<LteSpectrumSignalParametersDataFrame> lteDlDataRxParams = DynamicCast<LteSpectrumSignalParametersDataFrame> (params);
          if (lteDlDataRxParams != 0)
            {
              NS_LOG_DEBUG ("StartRx data");
              double power = 0;
              if (m_rbId >= 0)
                {
                  power = (*(params->psd))[m_rbId] * 180000;
                }
              else
                {
                  power = Integral (*(params->psd));
                }

              m_sumPower += power;
              if (power > m_referenceSignalPower)
                {
                  m_referenceSignalPower = power;
                }
            }
        }
      else
        {
          Ptr<LteSpectrumSignalParametersDlCtrlFrame> lteDlCtrlRxParams = DynamicCast<LteSpectrumSignalParametersDlCtrlFrame> (params);
          if (lteDlCtrlRxParams != 0)
            {
              NS_LOG_DEBUG ("StartRx control");
              double power = 0;
              if (m_rbId >= 0)
                {
                  power = (*(params->psd))[m_rbId] * 180000;
                }
              else
                {
                  power = Integral (*(params->psd));
                }

              m_sumPower += power;
              if (power > m_referenceSignalPower)
                {
                  m_referenceSignalPower = power;
                }
            }
        }
    }
}

void
RemSpectrumPhy::SetRxSpectrumModel (Ptr<const SpectrumModel> m)
{
  NS_LOG_FUNCTION (this << m);
  m_rxSpectrumModel = m;
}

double
RemSpectrumPhy::GetSinr (double noisePower)
{
  return m_referenceSignalPower / (m_sumPower - m_referenceSignalPower + noisePower);
}

void
RemSpectrumPhy::Deactivate ()
{
  m_active = false;
}

bool
RemSpectrumPhy::IsActive ()
{
  return m_active;
}

void
RemSpectrumPhy::Reset ()
{
  m_referenceSignalPower = 0;
  m_sumPower = 0;
}

void
RemSpectrumPhy::SetUseDataChannel (bool value)
{
  m_useDataChannel = value;
}

void
RemSpectrumPhy::SetRbId (int32_t rbId)
{
  m_rbId = rbId;
}


} // namespace ns3
