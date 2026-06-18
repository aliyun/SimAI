/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 CTTC
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


#include "radio-environment-map-helper.h"

#include <ns3/abort.h>
#include <ns3/log.h>
#include <ns3/double.h>
#include <ns3/integer.h>
#include <ns3/uinteger.h>
#include <ns3/string.h>
#include <ns3/boolean.h>
#include <ns3/pointer.h>
#include <ns3/spectrum-channel.h>
#include <ns3/config.h>
#include <ns3/rem-spectrum-phy.h>
#include <ns3/mobility-building-info.h>
#include <ns3/constant-position-mobility-model.h>
#include <ns3/simulator.h>
#include <ns3/node.h>
#include <ns3/buildings-helper.h>
#include <ns3/lte-spectrum-value-helper.h>

#include <fstream>
#include <limits>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RadioEnvironmentMapHelper");

NS_OBJECT_ENSURE_REGISTERED (RadioEnvironmentMapHelper);

RadioEnvironmentMapHelper::RadioEnvironmentMapHelper ()
{
}


RadioEnvironmentMapHelper::~RadioEnvironmentMapHelper ()
{
}



void
RadioEnvironmentMapHelper::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
RadioEnvironmentMapHelper::GetTypeId (void)
{
  NS_LOG_FUNCTION ("RadioEnvironmentMapHelper::GetTypeId");
  static TypeId tid = TypeId ("ns3::RadioEnvironmentMapHelper")
    .SetParent<Object> ()
    .SetGroupName("Lte")
    .AddConstructor<RadioEnvironmentMapHelper> ()
    .AddAttribute ("Channel",
                   "The DL spectrum channel for which the RadioEnvironment Map is to be generated. "
                   "Alternatively ChannelPath attribute can be used."
                   "Only one of the two (Channel or ChannelPath) should be set.",
                   PointerValue (nullptr),
                   MakePointerAccessor (&RadioEnvironmentMapHelper::m_channel),
                   MakePointerChecker<SpectrumChannel> ())
    .AddAttribute ("ChannelPath", "The path to the channel for which the Radio Environment Map is to be generated."
                   "This attribute is an alternative to Channel attribute and is only used if Channel is not set (equal to nullptr). "
                   "Only one of the two (Channel or ChannelPath) should be set.",
                   StringValue ("/ChannelList/0"),
                   MakeStringAccessor (&RadioEnvironmentMapHelper::m_channelPath),
                   MakeStringChecker ())
    .AddAttribute ("OutputFile", "the filename to which the Radio Environment Map is saved",
                   StringValue ("rem.out"),
                   MakeStringAccessor (&RadioEnvironmentMapHelper::m_outputFile),
                   MakeStringChecker ())
    .AddAttribute ("XMin", "The min x coordinate of the map.",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&RadioEnvironmentMapHelper::m_xMin),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("YMin", "The min y coordinate of the map.",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&RadioEnvironmentMapHelper::m_yMin),
                   MakeDoubleChecker<double> ())
   .AddAttribute ("XMax", "The max x coordinate of the map.",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&RadioEnvironmentMapHelper::m_xMax),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("YMax", "The max y coordinate of the map.",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&RadioEnvironmentMapHelper::m_yMax),
                   MakeDoubleChecker<double> ())
   .AddAttribute ("XRes", "The resolution (number of points) of the map along the x axis.",
                   UintegerValue (100),
                   MakeUintegerAccessor (&RadioEnvironmentMapHelper::m_xRes),
                  MakeUintegerChecker<uint32_t> (2,std::numeric_limits<uint16_t>::max ()))
    .AddAttribute ("YRes", "The resolution (number of points) of the map along the y axis.",
                   UintegerValue (100),
                   MakeUintegerAccessor (&RadioEnvironmentMapHelper::m_yRes),
                   MakeUintegerChecker<uint16_t> (2,std::numeric_limits<uint16_t>::max ()))
    .AddAttribute ("Z", "The value of the z coordinate for which the map is to be generated",
		   DoubleValue (0.0),
                   MakeDoubleAccessor (&RadioEnvironmentMapHelper::m_z),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("StopWhenDone", "If true, Simulator::Stop () will be called as soon as the REM has been generated",
		   BooleanValue (true),
                   MakeBooleanAccessor (&RadioEnvironmentMapHelper::m_stopWhenDone),
                   MakeBooleanChecker ())
    .AddAttribute ("NoisePower",
                   "the power of the measuring instrument noise, in Watts. Default to a kT of -174 dBm with a noise figure of 9 dB and a bandwidth of 25 LTE Resource Blocks",
                   DoubleValue (1.4230e-13),
                   MakeDoubleAccessor (&RadioEnvironmentMapHelper::m_noisePower),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("MaxPointsPerIteration", "Maximum number of REM points to be calculated per iteration. Every point consumes approximately 5KB of memory.",
                   UintegerValue (20000),
                   MakeUintegerAccessor (&RadioEnvironmentMapHelper::m_maxPointsPerIteration),
                   MakeUintegerChecker<uint32_t> (1,std::numeric_limits<uint32_t>::max ()))
    .AddAttribute ("Earfcn",
                   "E-UTRA Absolute Radio Frequency Channel Number (EARFCN) "
                   "as per 3GPP 36.101 Section 5.7.3. ",
                   UintegerValue (100),
                   MakeUintegerAccessor (&RadioEnvironmentMapHelper::m_earfcn),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("Bandwidth",
                   "Transmission Bandwidth Configuration (in number of RBs) over which the SINR will be calculated",
                   UintegerValue (25),
                   MakeUintegerAccessor (&RadioEnvironmentMapHelper::SetBandwidth, 
                                         &RadioEnvironmentMapHelper::GetBandwidth),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("UseDataChannel",
                   "If true, REM will be generated for PDSCH and for PDCCH otherwise ",
                   BooleanValue (false),
                   MakeBooleanAccessor (&RadioEnvironmentMapHelper::m_useDataChannel),
                   MakeBooleanChecker ())
    .AddAttribute ("RbId",
                   "Resource block Id, for which REM will be generated,"
                   "default value is -1, what means REM will be averaged from all RBs",
                   IntegerValue (-1),
                   MakeIntegerAccessor (&RadioEnvironmentMapHelper::m_rbId),
                   MakeIntegerChecker<int32_t> ())
  ;
  return tid;
}


uint16_t
RadioEnvironmentMapHelper::GetBandwidth () const
{
  return m_bandwidth;
}

void 
RadioEnvironmentMapHelper::SetBandwidth (uint16_t bw)
{
  switch (bw)
    { 
    case 6:
    case 15:
    case 25:
    case 50:
    case 75:
    case 100:
      m_bandwidth = bw;
      break;

    default:
      NS_FATAL_ERROR ("invalid bandwidth value " << bw);
      break;
    }
}



void 
RadioEnvironmentMapHelper::Install ()
{
  NS_LOG_FUNCTION (this);
  if (!m_rem.empty ())
    {
      NS_FATAL_ERROR ("only one REM supported per instance of RadioEnvironmentMapHelper");
    }

  if (m_channel == nullptr) // if Channel attribute is not set, then use the ChannelPath attribute
    {
      Config::MatchContainer match = Config::LookupMatches (m_channelPath);
      if (match.GetN () != 1)
        {
          NS_FATAL_ERROR ("Lookup " << m_channelPath << " should have exactly one match");
        }
      m_channel = match.Get (0)->GetObject<SpectrumChannel> ();
      NS_ABORT_MSG_IF (m_channel == 0, "object at " << m_channelPath << " is not of type SpectrumChannel");
    }

  m_outFile.open (m_outputFile.c_str ());
  if (!m_outFile.is_open ())
    {
      NS_FATAL_ERROR ("Can't open file " << (m_outputFile));
      return;
    }
  
  double startDelay = 0.0026;

  if (m_useDataChannel)
    {
      //need time to start transmission of data channel
      startDelay = 0.5001;
    }

  Simulator::Schedule (Seconds (startDelay),
                       &RadioEnvironmentMapHelper::DelayedInstall,
                       this);
}


void 
RadioEnvironmentMapHelper::DelayedInstall ()
{
  NS_LOG_FUNCTION (this);
  m_xStep = (m_xMax - m_xMin)/(m_xRes-1);
  m_yStep = (m_yMax - m_yMin)/(m_yRes-1);
  
  if ((double)m_xRes * (double) m_yRes < (double) m_maxPointsPerIteration)
    {
      m_maxPointsPerIteration = m_xRes * m_yRes;
    }
  
  for (uint32_t i = 0; i < m_maxPointsPerIteration; ++i)
    {
      RemPoint p;
      p.phy = CreateObject<RemSpectrumPhy> ();
      p.bmm = CreateObject<ConstantPositionMobilityModel> ();
      Ptr<MobilityBuildingInfo> buildingInfo = CreateObject<MobilityBuildingInfo> ();
      p.bmm->AggregateObject (buildingInfo); // operation usually done by BuildingsHelper::Install
      p.phy->SetRxSpectrumModel (LteSpectrumValueHelper::GetSpectrumModel (m_earfcn, m_bandwidth));
      p.phy->SetMobility (p.bmm);
      p.phy->SetUseDataChannel (m_useDataChannel);
      p.phy->SetRbId (m_rbId);
      m_channel->AddRx (p.phy);
      m_rem.push_back (p);
    }

  double remIterationStartTime = 0.0001;
  double xMinNext = m_xMin;
  double yMinNext = m_yMin;
  uint32_t numPointsCurrentIteration = 0;
  bool justScheduled = false;
  for (double x = m_xMin; x < m_xMax + 0.5*m_xStep; x += m_xStep)
    {
      for (double y = m_yMin; y < m_yMax + 0.5*m_yStep ; y += m_yStep)
        {
          if (justScheduled)
            {
              xMinNext = x;
              yMinNext = y;
              justScheduled = false;
            }
          
          ++numPointsCurrentIteration;
          if ((numPointsCurrentIteration == m_maxPointsPerIteration)
              || ((x > m_xMax - 0.5*m_xStep) && (y > m_yMax - 0.5*m_yStep)) )
            {
              Simulator::Schedule (Seconds (remIterationStartTime), 
                                   &RadioEnvironmentMapHelper::RunOneIteration,
                                   this, xMinNext, x, yMinNext, y);
              remIterationStartTime += 0.001;
              justScheduled = true;
              numPointsCurrentIteration = 0;
            }
        }      
    }

  Simulator::Schedule (Seconds (remIterationStartTime), 
                       &RadioEnvironmentMapHelper::Finalize,
                       this);
}

  
void 
RadioEnvironmentMapHelper::RunOneIteration (double xMin, double xMax, double yMin, double yMax)
{
  NS_LOG_FUNCTION (this << xMin << xMax << yMin << yMax);
  std::list<RemPoint>::iterator remIt = m_rem.begin ();
  double x = 0.0;
  double y = 0.0;
  for (x = xMin; x < xMax + 0.5*m_xStep; x += m_xStep)
    {
      for (y = (x == xMin) ? yMin : m_yMin;
           y < ((x == xMax) ? yMax : m_yMax) + 0.5*m_yStep;
           y += m_yStep)
        {
          NS_ASSERT (remIt != m_rem.end ());
          remIt->bmm->SetPosition (Vector (x, y, m_z));
          Ptr <MobilityBuildingInfo> buildingInfo = (remIt->bmm)->GetObject <MobilityBuildingInfo> ();
          buildingInfo->MakeConsistent (remIt->bmm);
          ++remIt;
        }      
    }

  if (remIt != m_rem.end ())
    {
      NS_ASSERT ((x > m_xMax - 0.5*m_xStep) && (y > m_yMax - 0.5*m_yStep));
      NS_LOG_LOGIC ("deactivating RemSpectrumPhys that are unneeded in the last iteration");
      while (remIt != m_rem.end ())
        {
          remIt->phy->Deactivate ();
          ++remIt;
        }
    }

  Simulator::Schedule (Seconds (0.0005), &RadioEnvironmentMapHelper::PrintAndReset, this);  
}

void 
RadioEnvironmentMapHelper::PrintAndReset ()
{
  NS_LOG_FUNCTION (this);
  
  for (std::list<RemPoint>::iterator it = m_rem.begin ();
       it != m_rem.end ();
       ++it)
    {
      if (!(it->phy->IsActive ()))
        {
          // should occur only upon last iteration when some RemPoint
          // at the end of the list can be unused
          break;
        }
      Vector pos = it->bmm->GetPosition ();
      NS_LOG_LOGIC ("output: " << pos.x << "\t" 
                    << pos.y << "\t" 
                    << pos.z << "\t" 
                    << it->phy->GetSinr (m_noisePower));
      m_outFile << pos.x << "\t" 
                << pos.y << "\t" 
                << pos.z << "\t" 
                << it->phy->GetSinr (m_noisePower)
                << std::endl;
      it->phy->Reset ();
    }
}

void 
RadioEnvironmentMapHelper::Finalize ()
{
  NS_LOG_FUNCTION (this);
  m_outFile.close ();
  if (m_stopWhenDone)
    {
      Simulator::Stop ();
    }
}


} // namespace ns3
