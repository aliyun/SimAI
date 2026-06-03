/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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
 * Author: Benjamin Cizdziel <ben.cizdziel@gmail.com>
 */

#include <ns3/log.h>
#include <cmath>
#include <list>
#include <vector>
#include <ns3/uinteger.h>
#include <ns3/double.h>
#include <ns3/isotropic-antenna-model.h>
#include <ns3/mobility-helper.h>
#include <ns3/position-allocator.h>
#include <ns3/geographic-positions.h>
#include "tv-spectrum-transmitter-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TvSpectrumTransmitterHelper");

/// NORTH AMERICA: 84 elements (index 0 - 83); valid channels = 2 - 83
const int northAmericaArrayLength = 84;
/// NORTH AMERICA start frequencies
const double northAmericaStartFrequencies[84] = {0, 0, 54e6, 60e6, 66e6, 76e6, 
  82e6, 174e6, 180e6, 186e6, 192e6, 198e6, 204e6, 210e6, 470e6, 476e6, 482e6, 
  488e6, 494e6, 500e6, 506e6, 512e6, 518e6, 524e6, 530e6, 536e6, 542e6, 548e6, 
  554e6, 560e6, 566e6, 572e6, 578e6, 584e6, 590e6, 596e6, 602e6, 608e6, 614e6, 
  620e6, 626e6, 632e6, 638e6, 644e6, 650e6, 656e6, 662e6, 668e6, 674e6, 680e6, 
  686e6, 692e6, 698e6, 704e6, 710e6, 716e6, 722e6, 728e6, 734e6, 740e6, 746e6, 
  752e6, 758e6, 764e6, 770e6, 776e6, 782e6, 788e6, 794e6, 800e6, 806e6, 812e6, 
  818e6, 824e6, 830e6, 836e6, 842e6, 848e6, 854e6, 860e6, 866e6, 872e6, 878e6, 
  884e6};
/// NORTH AMERICA end frequencies
const double northAmericaEndFrequencies[84] = {0, 0, 60e6, 66e6, 72e6, 82e6, 
  88e6, 180e6, 186e6, 192e6, 198e6, 204e6, 210e6, 216e6, 476e6, 482e6, 488e6, 
  494e6, 500e6, 506e6, 512e6, 518e6, 524e6, 530e6, 536e6, 542e6, 548e6, 554e6, 
  560e6, 566e6, 572e6, 578e6, 584e6, 590e6, 596e6, 602e6, 608e6, 614e6, 620e6, 
  626e6, 632e6, 638e6, 644e6, 650e6, 656e6, 662e6, 668e6, 674e6, 680e6, 686e6, 
  692e6, 698e6, 704e6, 710e6, 716e6, 722e6, 728e6, 734e6, 740e6, 746e6, 752e6, 
  758e6, 764e6, 770e6, 776e6, 782e6, 788e6, 794e6, 800e6, 806e6, 812e6, 818e6, 
  824e6, 830e6, 836e6, 842e6, 848e6, 854e6, 860e6, 866e6, 872e6, 878e6, 884e6, 
  890e6};

/// EUROPE: 70 elements (index 0 - 69); valid channels = 5 - 12, 21 - 69
const int europeArrayLength = 70;
/// EUROPE start frequencies
const double europeStartFrequencies[70] = {0, 0, 0, 0, 0, 174e6, 181e6, 188e6, 
  195e6, 202e6, 209e6, 216e6, 223e6, 0, 0, 0, 0, 0, 0, 0, 0, 470e6, 478e6, 
  486e6, 494e6, 502e6, 510e6, 518e6, 526e6, 534e6, 542e6, 550e6, 558e6, 566e6, 
  574e6, 582e6, 590e6, 598e6, 606e6, 614e6, 622e6, 630e6, 638e6, 646e6, 654e6, 
  662e6, 670e6, 678e6, 686e6, 694e6, 702e6, 710e6, 718e6, 726e6, 734e6, 742e6, 
  750e6, 758e6, 766e6, 774e6, 782e6, 790e6, 798e6, 806e6, 814e6, 822e6, 830e6, 
  838e6, 846e6, 854e6};
/// EUROPE end frequencies
const double europeEndFrequencies[70] = {0, 0, 0, 0, 0, 181e6, 188e6, 195e6, 
  202e6, 209e6, 216e6, 223e6, 230e6, 0, 0, 0, 0, 0, 0, 0, 0, 478e6, 486e6, 
  494e6, 502e6, 510e6, 518e6, 526e6, 534e6, 542e6, 550e6, 558e6, 566e6, 574e6, 
  582e6, 590e6, 598e6, 606e6, 614e6, 622e6, 630e6, 638e6, 646e6, 654e6, 662e6, 
  670e6, 678e6, 686e6, 694e6, 702e6, 710e6, 718e6, 726e6, 734e6, 742e6, 750e6, 
  758e6, 766e6, 774e6, 782e6, 790e6, 798e6, 806e6, 814e6, 822e6, 830e6, 838e6, 
  846e6, 854e6, 862e6};

/// JAPAN: 63 elements (index 0 - 62); valid channels = 1 - 62
const int japanArrayLength = 63;
/// JAPAN start frequencies
const double japanStartFrequencies[63] = {0, 90e6, 96e6, 102e6, 170e6, 176e6, 
  182e6, 188e6, 192e6, 198e6, 204e6, 210e6, 216e6, 470e6, 476e6, 482e6, 488e6, 
  494e6, 500e6, 506e6, 512e6, 518e6, 524e6, 530e6, 536e6, 542e6, 548e6, 554e6, 
  560e6, 566e6, 572e6, 578e6, 584e6, 590e6, 596e6, 602e6, 608e6, 614e6, 620e6, 
  626e6, 632e6, 638e6, 644e6, 650e6, 656e6, 662e6, 668e6, 674e6, 680e6, 686e6, 
  692e6, 698e6, 704e6, 710e6, 716e6, 722e6, 728e6, 734e6, 740e6, 746e6, 752e6, 
  758e6, 764e6};
/// JAPAN end frequencies
const double japanEndFrequencies[63] = {0, 96e6, 102e6, 108e6, 176e6, 182e6, 
  188e6, 194e6, 198e6, 204e6, 210e6, 216e6, 222e6, 476e6, 482e6, 488e6, 494e6, 
  500e6, 506e6, 512e6, 518e6, 524e6, 530e6, 536e6, 542e6, 548e6, 554e6, 560e6, 
  566e6, 572e6, 578e6, 584e6, 590e6, 596e6, 602e6, 608e6, 614e6, 620e6, 626e6, 
  632e6, 638e6, 644e6, 650e6, 656e6, 662e6, 668e6, 674e6, 680e6, 686e6, 692e6, 
  698e6, 704e6, 710e6, 716e6, 722e6, 728e6, 734e6, 740e6, 746e6, 752e6, 758e6, 
  764e6, 770e6};

TvSpectrumTransmitterHelper::TvSpectrumTransmitterHelper ()
  : m_channel (0),
    m_uniRand (CreateObject<UniformRandomVariable> ())
{
  NS_LOG_FUNCTION (this);
  m_factory.SetTypeId ("ns3::TvSpectrumTransmitter");
}

TvSpectrumTransmitterHelper::~TvSpectrumTransmitterHelper ()
{
  m_channel = 0;
  m_uniRand = 0;
  NS_LOG_FUNCTION (this);
}

void
TvSpectrumTransmitterHelper::SetChannel (Ptr<SpectrumChannel> c)
{
  NS_LOG_FUNCTION (this << c);
  m_channel = c;
}

void
TvSpectrumTransmitterHelper::SetAttribute (std::string name, const AttributeValue &val)
{
  m_factory.Set (name, val);
}


NetDeviceContainer
TvSpectrumTransmitterHelper::Install (NodeContainer nodeCont)
{
  NS_LOG_FUNCTION (this);
  NetDeviceContainer devCont;
  //iterate over node container to make one transmitter for each given node
  for (NodeContainer::Iterator i = nodeCont.Begin (); i != nodeCont.End (); ++i)
    {
      Ptr<Node> node = *i;
      Ptr<TvSpectrumTransmitter> phy = m_factory.Create ()->GetObject<TvSpectrumTransmitter> ();
      phy->CreateTvPsd ();
      Ptr<NonCommunicatingNetDevice> dev = CreateObject<NonCommunicatingNetDevice> ();
      NS_ASSERT (phy);
      dev->SetPhy (phy);
      NS_ASSERT (node);
      phy->SetMobility (node->GetObject<MobilityModel> ());
      NS_ASSERT (dev);
      phy->SetDevice (dev);
      NS_ASSERT (m_channel);
      phy->SetChannel (m_channel);
      dev->SetChannel (m_channel);
      node->AddDevice (dev);
      devCont.Add (dev);
      phy->Start ();
    }
  return devCont;
}


NetDeviceContainer
TvSpectrumTransmitterHelper::Install (NodeContainer nodeCont,
                                      Region region,
                                      uint16_t channelNumber)
{
  NS_LOG_FUNCTION (this);
  NetDeviceContainer devCont;
  double startFrequency;
  double channelBandwidth;
  if (region == REGION_NORTH_AMERICA)
    {
      NS_ASSERT_MSG (channelNumber < northAmericaArrayLength,
                     "channel number " << channelNumber <<
                     " does not exist for this region");
      NS_ASSERT_MSG (northAmericaStartFrequencies[channelNumber] != 0,
                     "channel number " << channelNumber <<
                     " does not exist for this region");
      startFrequency = northAmericaStartFrequencies[channelNumber];    
      channelBandwidth = northAmericaEndFrequencies[channelNumber] - 
                         northAmericaStartFrequencies[channelNumber];
    }
  else if (region == REGION_EUROPE)
    {
      NS_ASSERT_MSG (channelNumber < europeArrayLength,
                     "channel number " << channelNumber <<
                     " does not exist for this region");
      NS_ASSERT_MSG (europeStartFrequencies[channelNumber] != 0,
                     "channel number " << channelNumber <<
                     " does not exist for this region");
      startFrequency = europeStartFrequencies[channelNumber];    
      channelBandwidth = europeEndFrequencies[channelNumber] - 
                         europeStartFrequencies[channelNumber];
    }
  else if (region == REGION_JAPAN)
    {
      NS_ASSERT_MSG (channelNumber < japanArrayLength,
                     "channel number " << channelNumber <<
                     " does not exist for this region");
      NS_ASSERT_MSG (japanStartFrequencies[channelNumber] != 0,
                     "channel number " << channelNumber <<
                     " does not exist for this region");
      startFrequency = japanStartFrequencies[channelNumber];    
      channelBandwidth = japanEndFrequencies[channelNumber] - 
                         japanStartFrequencies[channelNumber];
    }
  //iterate over node container to make one transmitter for each given node
  for (NodeContainer::Iterator i = nodeCont.Begin (); i != nodeCont.End (); ++i)
    {
      Ptr<Node> node = *i;
      Ptr<TvSpectrumTransmitter> phy = m_factory.Create ()->GetObject<TvSpectrumTransmitter> ();
      phy->SetAttribute ("StartFrequency", DoubleValue (startFrequency));
      phy->SetAttribute ("ChannelBandwidth", DoubleValue (channelBandwidth));
      phy->CreateTvPsd ();
      Ptr<NonCommunicatingNetDevice> dev = CreateObject<NonCommunicatingNetDevice> ();
      NS_ASSERT (phy);
      dev->SetPhy (phy);
      NS_ASSERT (node);
      phy->SetMobility (node->GetObject<MobilityModel> ());
      NS_ASSERT (dev);
      phy->SetDevice (dev);
      NS_ASSERT (m_channel);
      phy->SetChannel (m_channel);
      dev->SetChannel (m_channel);
      node->AddDevice (dev);
      devCont.Add (dev);
      phy->Start ();
    }
  return devCont;
}


NetDeviceContainer
TvSpectrumTransmitterHelper::InstallAdjacent (NodeContainer nodeCont)
{
  NS_LOG_FUNCTION (this);
  NetDeviceContainer devCont;
  int index = 0;
  DoubleValue startFrequency;
  DoubleValue channelBandwidth;
  //iterate over node container to make one transmitter for each given node
  for (NodeContainer::Iterator i = nodeCont.Begin (); i != nodeCont.End (); ++i)
    {
      Ptr<Node> node = *i;
      Ptr<TvSpectrumTransmitter> phy = m_factory.Create ()->GetObject<TvSpectrumTransmitter> ();
      phy->GetAttribute ("StartFrequency", startFrequency);
      phy->GetAttribute ("ChannelBandwidth", channelBandwidth);
      phy->SetAttribute ("StartFrequency", DoubleValue (startFrequency.Get () + 
                         (index * channelBandwidth.Get ())));
      phy->CreateTvPsd ();
      Ptr<NonCommunicatingNetDevice> dev = CreateObject<NonCommunicatingNetDevice> ();
      NS_ASSERT (phy);
      dev->SetPhy (phy);
      NS_ASSERT (node);
      phy->SetMobility (node->GetObject<MobilityModel> ());
      NS_ASSERT (dev);
      phy->SetDevice (dev);
      NS_ASSERT (m_channel);
      phy->SetChannel (m_channel);
      dev->SetChannel (m_channel);
      node->AddDevice (dev);
      devCont.Add (dev);
      phy->Start ();
      index++;
    }
  return devCont;
}


NetDeviceContainer
TvSpectrumTransmitterHelper::InstallAdjacent (NodeContainer nodeCont,
                                              Region region,
                                              uint16_t channelNumber)
{
  NS_LOG_FUNCTION (this);
  NetDeviceContainer devCont;
  double startFrequency;
  double channelBandwidth;
  uint16_t currChannelNumber;
  int index = 0;
  //iterate over node container to make one transmitter for each given node
  for (NodeContainer::Iterator i = nodeCont.Begin (); i != nodeCont.End (); ++i)
    {
      currChannelNumber = channelNumber + index;
      if (region == REGION_NORTH_AMERICA)
        {
          NS_ASSERT_MSG (currChannelNumber < northAmericaArrayLength,
                         "channel number " << currChannelNumber <<
                         " does not exist for this region");
          NS_ASSERT_MSG (northAmericaStartFrequencies[currChannelNumber] != 0,
                         "channel number " << currChannelNumber <<
                         " does not exist for this region");
          startFrequency = northAmericaStartFrequencies[currChannelNumber];    
          channelBandwidth = northAmericaEndFrequencies[currChannelNumber] 
                             - northAmericaStartFrequencies[currChannelNumber];
        }
      else if (region == REGION_EUROPE)
        {
          NS_ASSERT_MSG (currChannelNumber < europeArrayLength,
                         "channel number " << currChannelNumber <<
                         " does not exist for this region");
          NS_ASSERT_MSG (europeStartFrequencies[currChannelNumber] != 0,
                         "channel number " << currChannelNumber <<
                         " does not exist for this region");
          startFrequency = europeStartFrequencies[currChannelNumber];    
          channelBandwidth = europeEndFrequencies[currChannelNumber] - 
                             europeStartFrequencies[currChannelNumber];
        }
      else if (region == REGION_JAPAN)
        {
          NS_ASSERT_MSG (currChannelNumber < japanArrayLength,
                         "channel number " << currChannelNumber <<
                         " does not exist for this region");
          NS_ASSERT_MSG (japanStartFrequencies[currChannelNumber] != 0,
                         "channel number " << currChannelNumber <<
                         " does not exist for this region");
          startFrequency = japanStartFrequencies[currChannelNumber];    
          channelBandwidth = japanEndFrequencies[currChannelNumber] - 
                             japanStartFrequencies[currChannelNumber];
        }
      Ptr<Node> node = *i;
      Ptr<TvSpectrumTransmitter> phy = m_factory.Create ()->GetObject<TvSpectrumTransmitter> ();
      phy->SetAttribute ("StartFrequency", DoubleValue (startFrequency));
      phy->SetAttribute ("ChannelBandwidth", DoubleValue (channelBandwidth));
      phy->CreateTvPsd ();
      Ptr<NonCommunicatingNetDevice> dev = CreateObject<NonCommunicatingNetDevice> ();
      NS_ASSERT (phy);
      dev->SetPhy (phy);
      NS_ASSERT (node);
      phy->SetMobility (node->GetObject<MobilityModel> ());
      NS_ASSERT (dev);
      phy->SetDevice (dev);
      NS_ASSERT (m_channel);
      phy->SetChannel (m_channel);
      dev->SetChannel (m_channel);
      node->AddDevice (dev);
      devCont.Add (dev);
      phy->Start ();
      index++;
    }
  return devCont;
}


int64_t
TvSpectrumTransmitterHelper::AssignStreams (int64_t streamNum)
{
  m_uniRand->SetStream (streamNum);
  return 1;
}


void
TvSpectrumTransmitterHelper::CreateRegionalTvTransmitters (Region region, 
                                                           Density density, 
                                                           double originLatitude, 
                                                           double originLongitude,
                                                           double maxAltitude, 
                                                           double maxRadius)
{
  NS_LOG_FUNCTION (this);
  std::list<int> transmitterIndicesToCreate;
  if (region == REGION_NORTH_AMERICA)
    {
      transmitterIndicesToCreate = GenerateRegionalTransmitterIndices 
                                   (northAmericaStartFrequencies, 
                                    northAmericaArrayLength, 
                                    density);
    }
  else if (region == REGION_EUROPE)
    {
      transmitterIndicesToCreate = GenerateRegionalTransmitterIndices 
                                   (europeStartFrequencies, 
                                    europeArrayLength, 
                                    density);
    }
  else if (region == REGION_JAPAN)
    {
      transmitterIndicesToCreate = GenerateRegionalTransmitterIndices 
                                   (japanStartFrequencies, 
                                    japanArrayLength, 
                                    density);
    }
  std::list<Vector> tvTransmitterLocations = 
    GeographicPositions::RandCartesianPointsAroundGeographicPoint (originLatitude, 
                                                                   originLongitude, 
                                                                   maxAltitude,
                                                                   transmitterIndicesToCreate.size(), 
                                                                   maxRadius,
                                                                   m_uniRand);
  InstallRandomRegionalTransmitters (region, 
                                     transmitterIndicesToCreate, 
                                     tvTransmitterLocations);
}


std::list<int>
TvSpectrumTransmitterHelper::GenerateRegionalTransmitterIndices (const double startFrequencies[], 
                                                                 const int startFrequenciesLength, 
                                                                 Density density)
{
  std::vector<double> startFreqVector; //stores all non-zero start frequencies
  for (int i = 0; i < startFrequenciesLength; i++)
    {
      double element = startFrequencies[i];
      //add all non-zero frequencies to vector (0 means unused channel)
      if (element != 0) startFreqVector.push_back(element);
    }

  //randomly generate number of transmitters to create based on density
  uint32_t freqVectorSize = startFreqVector.size();
  int randNumTransmitters = GetRandomNumTransmitters (density, freqVectorSize);
 
  //stores start frequencies that transmitters will be created to transmit
  std::vector<double> transmitterStartFreqsToCreate; 
  for (int i = 0; i < randNumTransmitters; i++)
    {
      //get random index from start frequency vector
      uint32_t randIndex = m_uniRand->GetInteger (0, startFreqVector.size () - 1);
      //add start frequency corresponding to random index to vector 
      transmitterStartFreqsToCreate.push_back(startFreqVector[randIndex]); 
      //remove selected start frequency from vector so it is not selected again
      startFreqVector.erase(startFreqVector.begin() + randIndex); 
    }

  //find indices on startFrequencies[] containing each start frequency that is 
  //selected to be transmitted and add to list
  std::list<int> transmitterIndicesToCreate;
  for (int i = 0; i < (int)transmitterStartFreqsToCreate.size(); i++)
    {
      for (int channelNumberIndex = 0; 
           channelNumberIndex < startFrequenciesLength; channelNumberIndex++)
        {
          if (startFrequencies[channelNumberIndex] == transmitterStartFreqsToCreate[i])
            {
              transmitterIndicesToCreate.push_back(channelNumberIndex);
              break;
            }
        }
    }
  return transmitterIndicesToCreate;
}


int
TvSpectrumTransmitterHelper::GetRandomNumTransmitters (Density density,
                                                       uint32_t numChannels)
{
  int numTransmitters;
  if (density == DENSITY_LOW)
    {
      numTransmitters = m_uniRand->GetInteger (1, ceil (0.33 * numChannels));
    }
  else if (density == DENSITY_MEDIUM)
    {
      numTransmitters = m_uniRand->GetInteger (ceil (0.33 * numChannels) + 1, ceil (0.66 * numChannels));
    }  
  else
    {
      numTransmitters = m_uniRand->GetInteger (ceil (0.66 * numChannels) + 1, numChannels);
    }  
  return numTransmitters;
}


void
TvSpectrumTransmitterHelper::InstallRandomRegionalTransmitters (Region region, 
                                                                std::list<int> transmitterIndicesToCreate, 
                                                                std::list<Vector> transmitterLocations)
{
  int numTransmitters = (int)transmitterIndicesToCreate.size();
  for (int transNum = 0; transNum < numTransmitters; transNum++)
    {
      Ptr<ListPositionAllocator> nodePosition = CreateObject<ListPositionAllocator> ();
      // add generated coordinate point to node position
      nodePosition->Add (transmitterLocations.front()); 
      MobilityHelper mobility;
      mobility.SetPositionAllocator (nodePosition);
      mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
      NodeContainer tvNode; // contains position of transmitter to be created
      tvNode.Create (1);
      mobility.Install (tvNode);
      // set channel number for this transmitter
      uint16_t channelNumber = (uint16_t) transmitterIndicesToCreate.front(); 
      Install (tvNode, region, channelNumber); //install tv transmitter
      transmitterLocations.pop_front(); // remove created transmitter location
      transmitterIndicesToCreate.pop_front(); // remove created transmitter index
    }
}

} // namespace ns3

