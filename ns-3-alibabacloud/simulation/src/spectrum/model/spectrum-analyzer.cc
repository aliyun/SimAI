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


#include <ns3/object-factory.h>
#include <ns3/log.h>
#include <ns3/double.h>
#include <ns3/simulator.h>
#include <ns3/trace-source-accessor.h>
#include <ns3/antenna-model.h>

#include "spectrum-analyzer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SpectrumAnalyzer");

NS_OBJECT_ENSURE_REGISTERED (SpectrumAnalyzer);

SpectrumAnalyzer::SpectrumAnalyzer ()
  : m_mobility (0),
    m_netDevice (0),
    m_channel (0),
    m_spectrumModel (0),
    m_sumPowerSpectralDensity (0),
    m_resolution (MilliSeconds (50)),
    m_active (false)
{
  NS_LOG_FUNCTION (this);
}



SpectrumAnalyzer::~SpectrumAnalyzer ()
{
  NS_LOG_FUNCTION (this);
}

void
SpectrumAnalyzer::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_mobility = 0;
  m_netDevice = 0;
  m_channel = 0;
  m_spectrumModel = 0;
  m_sumPowerSpectralDensity = 0;
  m_energySpectralDensity = 0;
  SpectrumPhy::DoDispose ();
}

TypeId
SpectrumAnalyzer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SpectrumAnalyzer")
    .SetParent<SpectrumPhy> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<SpectrumAnalyzer> ()
    .AddAttribute ("Resolution",
                   "The length of the time interval over which the "
                   "power spectral density of incoming signals is averaged",
                   TimeValue (MilliSeconds (1)),
                   MakeTimeAccessor (&SpectrumAnalyzer::m_resolution),
                   MakeTimeChecker ())
    .AddAttribute ("NoisePowerSpectralDensity",
                   "The power spectral density of the measuring instrument "
                   "noise, in Watt/Hz. Mostly useful to make spectrograms "
                   "look more similar to those obtained by real devices. "
                   "Defaults to the value for thermal noise at 300K.",
                   DoubleValue (1.38e-23 * 300),
                   MakeDoubleAccessor (&SpectrumAnalyzer::m_noisePowerSpectralDensity),
                   MakeDoubleChecker<double> ())
    .AddTraceSource ("AveragePowerSpectralDensityReport",
                     "Trace fired whenever a new value for the average "
                     "Power Spectral Density is calculated",
                     MakeTraceSourceAccessor (&SpectrumAnalyzer::m_averagePowerSpectralDensityReportTrace),
                     "ns3::SpectrumValue::TracedCallback")
  ;
  return tid;
}



Ptr<NetDevice>
SpectrumAnalyzer::GetDevice () const
{
  return m_netDevice;
}


Ptr<MobilityModel>
SpectrumAnalyzer::GetMobility () const
{
  return m_mobility;
}


Ptr<const SpectrumModel>
SpectrumAnalyzer::GetRxSpectrumModel () const
{
  return m_spectrumModel;
}

void
SpectrumAnalyzer::SetDevice (Ptr<NetDevice> d)
{
  NS_LOG_FUNCTION (this << d);
  m_netDevice = d;
}


void
SpectrumAnalyzer::SetMobility (Ptr<MobilityModel> m)
{
  NS_LOG_FUNCTION (this << m);
  m_mobility = m;
}


void
SpectrumAnalyzer::SetChannel (Ptr<SpectrumChannel> c)
{
  NS_LOG_FUNCTION (this << c);
  m_channel = c;
}


Ptr<Object>
SpectrumAnalyzer::GetAntenna () const
{
  return m_antenna;
}

void
SpectrumAnalyzer::SetAntenna (Ptr<AntennaModel> a)
{
  NS_LOG_FUNCTION (this << a);
  m_antenna = a;
}



void
SpectrumAnalyzer::StartRx (Ptr<SpectrumSignalParameters> params)
{
  NS_LOG_FUNCTION ( this << params);
  AddSignal (params->psd);
  Simulator::Schedule (params->duration, &SpectrumAnalyzer::SubtractSignal, this, params->psd);
}


void
SpectrumAnalyzer::AddSignal  (Ptr<const SpectrumValue> psd)
{
  NS_LOG_FUNCTION (this << *psd);
  UpdateEnergyReceivedSoFar ();
  (*m_sumPowerSpectralDensity) += (*psd);
}

void
SpectrumAnalyzer::SubtractSignal  (Ptr<const SpectrumValue> psd)
{
  NS_LOG_FUNCTION (this << *psd);
  UpdateEnergyReceivedSoFar ();
  (*m_sumPowerSpectralDensity) -= (*psd);
}

void
SpectrumAnalyzer::UpdateEnergyReceivedSoFar  ()
{
  NS_LOG_FUNCTION (this);
  if (m_lastChangeTime < Now ())
    {
      (*m_energySpectralDensity) += (*m_sumPowerSpectralDensity) * ((Now () - m_lastChangeTime).GetSeconds ());
      m_lastChangeTime = Now ();
    }
  else
    {
      NS_ASSERT (m_lastChangeTime == Now ());
    }
}

void
SpectrumAnalyzer::GenerateReport ()
{
  NS_LOG_FUNCTION (this);

  UpdateEnergyReceivedSoFar ();
  Ptr<SpectrumValue> avgPowerSpectralDensity = Create<SpectrumValue> (m_sumPowerSpectralDensity->GetSpectrumModel ());
  (*avgPowerSpectralDensity) = (*m_energySpectralDensity) / m_resolution.GetSeconds ();
  (*avgPowerSpectralDensity) += m_noisePowerSpectralDensity;
  (*m_energySpectralDensity) = 0;

  NS_LOG_INFO ("generating report");
  m_averagePowerSpectralDensityReportTrace (avgPowerSpectralDensity);

  *avgPowerSpectralDensity = 0;

  if (m_active)
    {
      Simulator::Schedule (m_resolution, &SpectrumAnalyzer::GenerateReport, this);
    }
}



void
SpectrumAnalyzer::SetRxSpectrumModel (Ptr<SpectrumModel> f)
{
  NS_LOG_FUNCTION (this << f);
  m_spectrumModel = f;
  NS_ASSERT (!m_sumPowerSpectralDensity);
  m_sumPowerSpectralDensity = Create<SpectrumValue> (f);
  m_energySpectralDensity = Create<SpectrumValue> (f);
  NS_ASSERT (m_sumPowerSpectralDensity);
}




void
SpectrumAnalyzer::Start ()
{
  NS_LOG_FUNCTION (this);
  if (!m_active)
    {
      NS_LOG_LOGIC ("activating");
      m_active = true;
      Simulator::Schedule (m_resolution, &SpectrumAnalyzer::GenerateReport, this);
    }
}


void
SpectrumAnalyzer::Stop ()
{
  m_active = false;
}

} // namespace ns3
