/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 *
 */

#include <ns3/log.h>
#include <cmath>
#include <ns3/simulator.h>
#include <ns3/antenna-model.h>
#include "lte-simple-spectrum-phy.h"
#include "ns3/lte-spectrum-signal-parameters.h"
#include "ns3/lte-net-device.h"
#include "ns3/lte-phy-tag.h"
#include <ns3/boolean.h>
#include <ns3/double.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteSimpleSpectrumPhy");

NS_OBJECT_ENSURE_REGISTERED (LteSimpleSpectrumPhy);

LteSimpleSpectrumPhy::LteSimpleSpectrumPhy ()
  : m_cellId (0)
{
}


LteSimpleSpectrumPhy::~LteSimpleSpectrumPhy ()
{
  NS_LOG_FUNCTION (this);
}

void LteSimpleSpectrumPhy::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_channel = 0;
  m_mobility = 0;
  m_device = 0;
  SpectrumPhy::DoDispose ();
}


TypeId
LteSimpleSpectrumPhy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteSimpleSpectrumPhy")
    .SetParent<SpectrumPhy> ()
    .AddTraceSource ("RxStart",
                     "Data reception start",
                     MakeTraceSourceAccessor (&LteSimpleSpectrumPhy::m_rxStart),
                     "ns3::SpectrumValue::TracedCallback")
  ;
  return tid;
}



Ptr<NetDevice>
LteSimpleSpectrumPhy::GetDevice () const
{
  NS_LOG_FUNCTION (this);
  return m_device;
}


Ptr<MobilityModel>
LteSimpleSpectrumPhy::GetMobility () const
{
  NS_LOG_FUNCTION (this);
  return m_mobility;
}


void
LteSimpleSpectrumPhy::SetDevice (Ptr<NetDevice> d)
{
  NS_LOG_FUNCTION (this << d);
  m_device = d;
}


void
LteSimpleSpectrumPhy::SetMobility (Ptr<MobilityModel> m)
{
  NS_LOG_FUNCTION (this << m);
  m_mobility = m;
}


void
LteSimpleSpectrumPhy::SetChannel (Ptr<SpectrumChannel> c)
{
  NS_LOG_FUNCTION (this << c);
  m_channel = c;
}

Ptr<const SpectrumModel>
LteSimpleSpectrumPhy::GetRxSpectrumModel () const
{
  return m_rxSpectrumModel;
}


Ptr<Object>
LteSimpleSpectrumPhy::GetAntenna () const
{
  return m_antenna;
}

void
LteSimpleSpectrumPhy::StartRx (Ptr<SpectrumSignalParameters> spectrumRxParams)
{
  NS_LOG_DEBUG ("LteSimpleSpectrumPhy::StartRx");

  NS_LOG_FUNCTION (this << spectrumRxParams);
  Ptr <const SpectrumValue> rxPsd = spectrumRxParams->psd;
  Time duration = spectrumRxParams->duration;

  // the device might start RX only if the signal is of a type
  // understood by this device - in this case, an LTE signal.
  Ptr<LteSpectrumSignalParametersDataFrame> lteDataRxParams = DynamicCast<LteSpectrumSignalParametersDataFrame> (spectrumRxParams);
  if (lteDataRxParams != 0)
    {
      if ( m_cellId > 0 )
        {
          if (m_cellId == lteDataRxParams->cellId)
            {
              m_rxStart (rxPsd);
            }
        }
      else
        {
          m_rxStart (rxPsd);
        }
    }
}

void
LteSimpleSpectrumPhy::SetRxSpectrumModel (Ptr<const SpectrumModel> model)
{
  NS_LOG_FUNCTION (this);
  m_rxSpectrumModel = model;
}

void
LteSimpleSpectrumPhy::SetCellId (uint16_t cellId)
{
  NS_LOG_FUNCTION (this);
  m_cellId = cellId;
}



} // namespace ns3
