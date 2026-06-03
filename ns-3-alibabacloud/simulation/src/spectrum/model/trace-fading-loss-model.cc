/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Giuseppe Piro  <g.piro@poliba.it>
 *         Marco Miozzo <mmiozzo@cttc.es>
 */


#include <ns3/trace-fading-loss-model.h>
#include <ns3/mobility-model.h>
#include <ns3/spectrum-value.h>
#include <ns3/log.h>
#include <ns3/string.h>
#include <ns3/double.h>
#include "ns3/uinteger.h"
#include <fstream>
#include <ns3/simulator.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TraceFadingLossModel");

NS_OBJECT_ENSURE_REGISTERED (TraceFadingLossModel);
  


TraceFadingLossModel::TraceFadingLossModel ()
  : m_streamsAssigned (false)
{
  NS_LOG_FUNCTION (this);
  SetNext (NULL);
}


TraceFadingLossModel::~TraceFadingLossModel ()
{
  m_fadingTrace.clear ();
  m_windowOffsetsMap.clear ();
  m_startVariableMap.clear ();
}


TypeId
TraceFadingLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TraceFadingLossModel")
    .SetParent<SpectrumPropagationLossModel> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<TraceFadingLossModel> ()
    .AddAttribute ("TraceFilename",
                   "Name of file to load a trace from.",
                   StringValue (""),
                   MakeStringAccessor (&TraceFadingLossModel::SetTraceFileName),
                   MakeStringChecker ())
    .AddAttribute ("TraceLength",
                  "The total length of the fading trace (default value 10 s.)",
                  TimeValue (Seconds (10.0)),
                  MakeTimeAccessor (&TraceFadingLossModel::SetTraceLength),
                  MakeTimeChecker ())
    .AddAttribute ("SamplesNum",
                  "The number of samples the trace is made of (default 10000)",
                   UintegerValue (10000),
                   MakeUintegerAccessor (&TraceFadingLossModel::m_samplesNum),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("WindowSize",
                  "The size of the window for the fading trace (default value 0.5 s.)",
                  TimeValue (Seconds (0.5)),
                  MakeTimeAccessor (&TraceFadingLossModel::m_windowSize),
                  MakeTimeChecker ())
    .AddAttribute ("RbNum",
                    "The number of RB the trace is made of (default 100)",
                    UintegerValue (100),
                   MakeUintegerAccessor (&TraceFadingLossModel::m_rbNum),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("RngStreamSetSize",
                    "The number of RNG streams reserved for the fading model. The maximum number of streams that are needed for an LTE FDD scenario is 2 * numUEs * numeNBs.",
                    UintegerValue (200000),
                   MakeUintegerAccessor (&TraceFadingLossModel::m_streamSetSize),
                   MakeUintegerChecker<uint64_t> ())
  ;
  return tid;
}

void
TraceFadingLossModel::SetTraceFileName (std::string fileName)
{
  NS_LOG_FUNCTION (this << "Set Fading Trace " << fileName);
  
  m_traceFile = fileName;
}

void 
TraceFadingLossModel::SetTraceLength (Time t)
{
  m_traceLength = t;
}

void 
TraceFadingLossModel::DoInitialize ()
{
  LoadTrace ();
}


void
TraceFadingLossModel::LoadTrace ()
{
  NS_LOG_FUNCTION (this << "Loading Fading Trace " << m_traceFile);
  std::ifstream ifTraceFile;
  ifTraceFile.open (m_traceFile.c_str (), std::ifstream::in);
  m_fadingTrace.clear ();
  if (!ifTraceFile.good ())
    {
      NS_LOG_INFO (this << " File: " << m_traceFile);
      NS_ASSERT_MSG(ifTraceFile.good (), " Fading trace file not found");
    }

//   NS_LOG_INFO (this << " length " << m_traceLength.GetSeconds ());
//   NS_LOG_INFO (this << " RB " << (uint32_t)m_rbNum << " samples " << m_samplesNum);
  for (uint32_t i = 0; i < m_rbNum; i++)
    {
      FadingTraceSample rbTimeFadingTrace;
      for (uint32_t j = 0; j < m_samplesNum; j++)
        {
          double sample;
          ifTraceFile >> sample;
          rbTimeFadingTrace.push_back (sample);
        }
      m_fadingTrace.push_back (rbTimeFadingTrace);
    }
  m_timeGranularity = m_traceLength.GetMilliSeconds () / m_samplesNum;
  m_lastWindowUpdate = Simulator::Now ();
}


Ptr<SpectrumValue>
TraceFadingLossModel::DoCalcRxPowerSpectralDensity (
  Ptr<const SpectrumValue> txPsd,
  Ptr<const MobilityModel> a,
  Ptr<const MobilityModel> b) const
{
  NS_LOG_FUNCTION (this << *txPsd << a << b);
  
  std::map <ChannelRealizationId_t, int >::iterator itOff;
  ChannelRealizationId_t mobilityPair = std::make_pair (a,b);
  itOff = m_windowOffsetsMap.find (mobilityPair);
  if (itOff!=m_windowOffsetsMap.end ())
    {
      if (Simulator::Now ().GetSeconds () >= m_lastWindowUpdate.GetSeconds () + m_windowSize.GetSeconds ())
        {
          // update all the offsets
          NS_LOG_INFO ("Fading Windows Updated");
          std::map <ChannelRealizationId_t, int >::iterator itOff2;
          for (itOff2 = m_windowOffsetsMap.begin (); itOff2 != m_windowOffsetsMap.end (); itOff2++)
            {
              std::map <ChannelRealizationId_t, Ptr<UniformRandomVariable> >::iterator itVar;
              itVar = m_startVariableMap.find ((*itOff2).first);
              (*itOff2).second = (*itVar).second->GetValue ();
            }
          m_lastWindowUpdate = Simulator::Now ();
        }
    }
  else
    {
      NS_LOG_LOGIC (this << "insert new channel realization, m_windowOffsetMap.size () = " << m_windowOffsetsMap.size ());
      Ptr<UniformRandomVariable> startV = CreateObject<UniformRandomVariable> ();
      startV->SetAttribute ("Min", DoubleValue (1.0));
      startV->SetAttribute ("Max", DoubleValue ((m_traceLength.GetSeconds () - m_windowSize.GetSeconds ()) * 1000.0));
      if (m_streamsAssigned)
        {
          NS_ASSERT_MSG (m_currentStream <= m_lastStream, "not enough streams, consider increasing the StreamSetSize attribute");
          startV->SetStream (m_currentStream);
          m_currentStream += 1;
        }
      ChannelRealizationId_t mobilityPair = std::make_pair (a,b);
      m_startVariableMap.insert (std::pair<ChannelRealizationId_t,Ptr<UniformRandomVariable> > (mobilityPair, startV));
      itOff = m_windowOffsetsMap.insert (std::pair<ChannelRealizationId_t,int> (mobilityPair, startV->GetValue ())).first;
    }

  
  Ptr<SpectrumValue> rxPsd = Copy<SpectrumValue> (txPsd);
  Values::iterator vit = rxPsd->ValuesBegin ();
  
  //Vector aSpeedVector = a->GetVelocity ();
  //Vector bSpeedVector = b->GetVelocity ();
  
  //double speed = std::sqrt (std::pow (aSpeedVector.x-bSpeedVector.x,2) + std::pow (aSpeedVector.y-bSpeedVector.y,2));

  NS_LOG_LOGIC (this << *rxPsd);
  NS_ASSERT (!m_fadingTrace.empty ());
  int now_ms = static_cast<int> (Simulator::Now ().GetMilliSeconds () * m_timeGranularity);
  int lastUpdate_ms = static_cast<int> (m_lastWindowUpdate.GetMilliSeconds () * m_timeGranularity);
  int index = ((*itOff).second + now_ms - lastUpdate_ms) % m_samplesNum;
  int subChannel = 0;
  while (vit != rxPsd->ValuesEnd ())
    {
      NS_ASSERT (subChannel < 100);
      if (*vit != 0.)
        {
          double fading = m_fadingTrace.at (subChannel).at (index);
          NS_LOG_INFO (this << " FADING now " << now_ms << " offset " << (*itOff).second << " id " << index << " fading " << fading);
          double power = *vit; // in Watt/Hz
          power = 10 * std::log10 (180000 * power); // in dB

          NS_LOG_LOGIC (this << subChannel << *vit  << power << fading);

          *vit = std::pow (10., ((power + fading) / 10)) / 180000; // in Watt

          NS_LOG_LOGIC (this << subChannel << *vit);

        }

      ++vit;
      ++subChannel;

    }

  NS_LOG_LOGIC (this << *rxPsd);
  return rxPsd;
}

int64_t
TraceFadingLossModel::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  NS_ASSERT (m_streamsAssigned == false);  
  m_streamsAssigned = true;
  m_currentStream = stream;
  m_lastStream = stream + m_streamSetSize - 1;
  std::map <ChannelRealizationId_t, Ptr<UniformRandomVariable> >::iterator itVar;
  itVar = m_startVariableMap.begin ();
  // the following loop is for eventually pre-existing ChannelRealization instances
  // note that more instances are expected to be created at run time
  while (itVar!=m_startVariableMap.end ())
    {
      NS_ASSERT_MSG (m_currentStream <= m_lastStream, "not enough streams, consider increasing the StreamSetSize attribute");
      (*itVar).second->SetStream (m_currentStream);
      m_currentStream += 1;
    }
  return m_streamSetSize;
}



} // namespace ns3
