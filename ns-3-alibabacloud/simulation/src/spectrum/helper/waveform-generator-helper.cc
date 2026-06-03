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
#include "ns3/waveform-generator.h"
#include "ns3/non-communicating-net-device.h"
#include "waveform-generator-helper.h"




namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WaveformGeneratorHelper");

WaveformGeneratorHelper::WaveformGeneratorHelper ()
{
  m_phy.SetTypeId ("ns3::WaveformGenerator");
  m_device.SetTypeId ("ns3::NonCommunicatingNetDevice");
  m_antenna.SetTypeId ("ns3::IsotropicAntennaModel");
}

WaveformGeneratorHelper::~WaveformGeneratorHelper ()
{
}

void
WaveformGeneratorHelper::SetChannel (Ptr<SpectrumChannel> channel)
{
  m_channel = channel;
}

void
WaveformGeneratorHelper::SetChannel (std::string channelName)
{
  Ptr<SpectrumChannel> channel = Names::Find<SpectrumChannel> (channelName);
  m_channel = channel;
}

void
WaveformGeneratorHelper::SetTxPowerSpectralDensity (Ptr<SpectrumValue> txPsd)
{
  NS_LOG_FUNCTION (this << txPsd);
  m_txPsd = txPsd;
}


void
WaveformGeneratorHelper::SetPhyAttribute (std::string name, const AttributeValue &v)
{
  m_phy.Set (name, v);
}


void
WaveformGeneratorHelper::SetDeviceAttribute (std::string name, const AttributeValue &v)
{
  m_device.Set (name, v);
}

void
WaveformGeneratorHelper::SetAntenna (std::string type,
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

NetDeviceContainer
WaveformGeneratorHelper::Install (NodeContainer c) const
{
  NetDeviceContainer devices;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      Ptr<Node> node = *i;

      Ptr<NonCommunicatingNetDevice> dev = m_device.Create ()->GetObject<NonCommunicatingNetDevice> ();

      Ptr<WaveformGenerator>  phy = m_phy.Create ()->GetObject<WaveformGenerator> ();
      NS_ASSERT (phy);

      dev->SetPhy (phy);

      NS_ASSERT (node);
      phy->SetMobility (node->GetObject<MobilityModel> ());

      NS_ASSERT (dev);
      phy->SetDevice (dev);

      NS_ASSERT_MSG (m_txPsd, "you forgot to call WaveformGeneratorHelper::SetTxPowerSpectralDensity ()");
      phy->SetTxPowerSpectralDensity (m_txPsd);

      NS_ASSERT_MSG (m_channel, "you forgot to call WaveformGeneratorHelper::SetChannel ()");
      phy->SetChannel (m_channel);
      dev->SetChannel (m_channel);

      Ptr<AntennaModel> antenna = (m_antenna.Create ())->GetObject<AntennaModel> ();
      NS_ASSERT_MSG (antenna, "error in creating the AntennaModel object");
      phy->SetAntenna (antenna);

      node->AddDevice (dev);
      devices.Add (dev);
    }
  return devices;
}

NetDeviceContainer
WaveformGeneratorHelper::Install (Ptr<Node> node) const
{
  return Install (NodeContainer (node));
}

NetDeviceContainer
WaveformGeneratorHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return Install (node);
}


} // namespace ns3
