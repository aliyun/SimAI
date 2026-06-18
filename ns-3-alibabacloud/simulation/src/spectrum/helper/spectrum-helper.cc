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
#include "spectrum-helper.h"
#include <ns3/simulator.h>
#include <ns3/config.h>
#include <ns3/names.h>
#include <ns3/spectrum-channel.h>
#include <ns3/spectrum-phy.h>
#include <ns3/single-model-spectrum-channel.h>
#include <ns3/multi-model-spectrum-channel.h>
#include <ns3/half-duplex-ideal-phy.h>


namespace ns3 {


SpectrumChannelHelper
SpectrumChannelHelper::Default (void)
{
  SpectrumChannelHelper h;
  h.SetChannel ("ns3::SingleModelSpectrumChannel");
  h.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  h.AddSpectrumPropagationLoss ("ns3::FriisSpectrumPropagationLossModel");
  return h;
}

void
SpectrumChannelHelper::SetChannel (std::string type,
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
  m_channel.SetTypeId (type);
  m_channel.Set (n0, v0);
  m_channel.Set (n1, v1);
  m_channel.Set (n2, v2);
  m_channel.Set (n3, v3);
  m_channel.Set (n4, v4);
  m_channel.Set (n5, v5);
  m_channel.Set (n6, v6);
  m_channel.Set (n7, v7);
}

void
SpectrumChannelHelper::AddPropagationLoss (std::string type,
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
  Ptr<PropagationLossModel> m = factory.Create<PropagationLossModel> ();
  AddPropagationLoss (m);
}


void 
SpectrumChannelHelper::AddPropagationLoss (Ptr<PropagationLossModel> m)
{
  m->SetNext (m_propagationLossModel);
  m_propagationLossModel = m;
}

void
SpectrumChannelHelper::AddSpectrumPropagationLoss (std::string type,
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
  Ptr<SpectrumPropagationLossModel> m = factory.Create<SpectrumPropagationLossModel> ();
  AddSpectrumPropagationLoss (m);
}

void 
SpectrumChannelHelper::AddSpectrumPropagationLoss (Ptr<SpectrumPropagationLossModel> m)
{
  m->SetNext (m_spectrumPropagationLossModel);
  m_spectrumPropagationLossModel = m;
}

void
SpectrumChannelHelper::SetPropagationDelay (std::string type,
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
  m_propagationDelay = factory;
}

Ptr<SpectrumChannel>
SpectrumChannelHelper::Create (void) const
{
  Ptr<SpectrumChannel> channel = (m_channel.Create ())->GetObject<SpectrumChannel> ();
  channel->AddSpectrumPropagationLossModel (m_spectrumPropagationLossModel);
  channel->AddPropagationLossModel (m_propagationLossModel);
  Ptr<PropagationDelayModel> delay = m_propagationDelay.Create<PropagationDelayModel> ();
  channel->SetPropagationDelayModel (delay);
  return channel;
}


void
SpectrumPhyHelper::SetPhy (std::string type,
                           std::string n0, const AttributeValue &v0,
                           std::string n1, const AttributeValue &v1,
                           std::string n2, const AttributeValue &v2,
                           std::string n3, const AttributeValue &v3,
                           std::string n4, const AttributeValue &v4,
                           std::string n5, const AttributeValue &v5,
                           std::string n6, const AttributeValue &v6,
                           std::string n7, const AttributeValue &v7)
{
  m_phy.SetTypeId (type);
  m_phy.Set (n0, v0);
  m_phy.Set (n1, v1);
  m_phy.Set (n2, v2);
  m_phy.Set (n3, v3);
  m_phy.Set (n4, v4);
  m_phy.Set (n5, v5);
  m_phy.Set (n6, v6);
  m_phy.Set (n7, v7);
}



void
SpectrumPhyHelper::SetChannel (Ptr<SpectrumChannel> channel)
{
  m_channel = channel;
}

void
SpectrumPhyHelper::SetChannel (std::string channelName)
{
  Ptr<SpectrumChannel> channel = Names::Find<SpectrumChannel> (channelName);
  m_channel = channel;
}

void
SpectrumPhyHelper::SetPhyAttribute (std::string name, const AttributeValue &v)
{
  m_phy.Set (name, v);
}


Ptr<SpectrumPhy>
SpectrumPhyHelper::Create (Ptr<Node> node, Ptr<NetDevice> device) const
{
  NS_ASSERT (m_channel);
  Ptr<SpectrumPhy> phy = (m_phy.Create ())->GetObject<SpectrumPhy> ();
  phy->SetChannel (m_channel);
  phy->SetMobility (node->GetObject<MobilityModel> ());
  phy->SetDevice (device);
  return phy;
}


} // namespace ns3
