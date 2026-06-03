/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 CTTC
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
#include "ns3/propagation-delay-model.h"
#include "ns3/spectrum-propagation-loss-model.h"
#include "ns3/mobility-model.h"
#include "ns3/log.h"
#include "ns3/config.h"
#include "ns3/simulator.h"
#include "ns3/names.h"
#include "ns3/antenna-model.h"
#include "ns3/spectrum-channel.h"
#include "ns3/spectrum-analyzer.h"
#include "ns3/non-communicating-net-device.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/trace-helper.h"
#include "spectrum-analyzer-helper.h"




namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SpectrumAnalyzerHelper");

/**
 * \ingroup spectrum
 * \brief Writes a report of the Average Power Spectral Density
 * \param streamWrapper the wrapper to the output stream
 * \param avgPowerSpectralDensity Average Power Spectral Density
 */
static void
WriteAveragePowerSpectralDensityReport (Ptr<OutputStreamWrapper> streamWrapper,
                                        Ptr<const SpectrumValue> avgPowerSpectralDensity)
{
  NS_LOG_FUNCTION (streamWrapper << avgPowerSpectralDensity);
  std::ostream* ostream = streamWrapper->GetStream ();
  if (ostream->good ())
    {
      Bands::const_iterator fi = avgPowerSpectralDensity->ConstBandsBegin ();
      Values::const_iterator vi = avgPowerSpectralDensity->ConstValuesBegin ();
      while (fi != avgPowerSpectralDensity->ConstBandsEnd ())
        {
          NS_ASSERT (vi != avgPowerSpectralDensity->ConstValuesEnd ());
          *ostream << Now ().GetSeconds () << " "
                   << fi->fc << " "
                   << *vi << std::endl;
          ++fi;
          ++vi;
        }
      // An additional line separates different spectrums sweeps
      *ostream << std::endl;
    }
}



SpectrumAnalyzerHelper::SpectrumAnalyzerHelper ()
{
  NS_LOG_FUNCTION (this);
  m_phy.SetTypeId ("ns3::SpectrumAnalyzer");
  m_device.SetTypeId ("ns3::NonCommunicatingNetDevice");
  m_antenna.SetTypeId ("ns3::IsotropicAntennaModel");
}

SpectrumAnalyzerHelper::~SpectrumAnalyzerHelper ()
{
  NS_LOG_FUNCTION (this);
}

void
SpectrumAnalyzerHelper::SetChannel (Ptr<SpectrumChannel> channel)
{
  NS_LOG_FUNCTION (this);
  m_channel = channel;
}

void
SpectrumAnalyzerHelper::SetChannel (std::string channelName)
{
  NS_LOG_FUNCTION (this);
  Ptr<SpectrumChannel> channel = Names::Find<SpectrumChannel> (channelName);
  m_channel = channel;
}


void
SpectrumAnalyzerHelper::SetPhyAttribute (std::string name, const AttributeValue &v)
{
  NS_LOG_FUNCTION (this);
  m_phy.Set (name, v);
}


void
SpectrumAnalyzerHelper::SetDeviceAttribute (std::string name, const AttributeValue &v)
{
  NS_LOG_FUNCTION (this);
  m_device.Set (name, v);
}

void
SpectrumAnalyzerHelper::SetAntenna (std::string type,
                                    std::string n0, const AttributeValue &v0,
                                    std::string n1, const AttributeValue &v1,
                                    std::string n2, const AttributeValue &v2,
                                    std::string n3, const AttributeValue &v3,
                                    std::string n4, const AttributeValue &v4,
                                    std::string n5, const AttributeValue &v5,
                                    std::string n6, const AttributeValue &v6,
                                    std::string n7, const AttributeValue &v7)
{
  ObjectFactory factory;
  factory.SetTypeId (type);
  factory.Set (n0, v0);
  factory.Set (n1, v1);
  factory.Set (n2, v2);
  factory.Set (n3, v3);
  factory.Set (n4, v4);
  factory.Set (n5, v5);
  factory.Set (n6, v6);
  factory.Set (n7, v7);
  m_antenna = factory;
}


void
SpectrumAnalyzerHelper::SetRxSpectrumModel (Ptr<SpectrumModel> m)
{
  NS_LOG_FUNCTION (this);
  m_rxSpectrumModel = m;
}

void
SpectrumAnalyzerHelper::EnableAsciiAll (std::string prefix)
{
  NS_LOG_FUNCTION (this);
  m_prefix = prefix;
}



NetDeviceContainer
SpectrumAnalyzerHelper::Install (NodeContainer c) const
{
  NS_LOG_FUNCTION (this);
  NetDeviceContainer devices;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      Ptr<Node> node = *i;

      Ptr<NonCommunicatingNetDevice> dev = m_device.Create ()->GetObject<NonCommunicatingNetDevice> ();

      Ptr<SpectrumAnalyzer>  phy = m_phy.Create ()->GetObject<SpectrumAnalyzer> ();
      NS_ASSERT (phy);

      dev->SetPhy (phy);

      NS_ASSERT (node);
      phy->SetMobility (node->GetObject<MobilityModel> ());

      NS_ASSERT (dev);
      phy->SetDevice (dev);

      NS_ASSERT_MSG (m_rxSpectrumModel, "you forgot to call SpectrumAnalyzerHelper::SetRxSpectrumModel ()");
      phy->SetRxSpectrumModel (m_rxSpectrumModel);

      NS_ASSERT_MSG (m_channel, "you forgot to call SpectrumAnalyzerHelper::SetChannel ()");
      m_channel->AddRx (phy);

      dev->SetChannel (m_channel);

      Ptr<AntennaModel> antenna = (m_antenna.Create ())->GetObject<AntennaModel> ();
      NS_ASSERT_MSG (antenna, "error in creating the AntennaModel object");
      phy->SetAntenna (antenna);

      uint32_t devId = node->AddDevice (dev);
      devices.Add (dev);

      if (!m_prefix.empty ())
        {
          NS_LOG_LOGIC ("creating new output stream and binding it to the callback");
          AsciiTraceHelper asciiTraceHelper;
          std::string filename;
          filename = asciiTraceHelper.GetFilenameFromDevice (m_prefix, dev);
          Ptr<OutputStreamWrapper> stream = asciiTraceHelper.CreateFileStream (filename);

          // note that we don't use AsciiTraceHelper to connect the trace sink, since we use a custom trace sink

          // the following is inspired from YansWifiPhyHelper::EnableAsciiInternal
          std::ostringstream oss;
          oss.str ("");
          oss << "/NodeList/" << node->GetId () << "/DeviceList/" << devId << "/$ns3::NonCommunicatingNetDevice/Phy/AveragePowerSpectralDensityReport";
          Config::ConnectWithoutContext (oss.str (), MakeBoundCallback (&WriteAveragePowerSpectralDensityReport, stream));

          phy->Start ();

        }
    }
  return devices;
}

NetDeviceContainer
SpectrumAnalyzerHelper::Install (Ptr<Node> node) const
{
  NS_LOG_FUNCTION (this);
  return Install (NodeContainer (node));
}

NetDeviceContainer
SpectrumAnalyzerHelper::Install (std::string nodeName) const
{
  NS_LOG_FUNCTION (this);
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return Install (node);
}


} // namespace ns3
