/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Modified by : Marco Miozzo <mmiozzo@cttc.es>
 *        (move from CQI to Ctrl and Data SINR Chunk processors
 */


#include <ns3/log.h>
#include <ns3/spectrum-value.h>
#include "lte-chunk-processor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteChunkProcessor");

LteChunkProcessor::LteChunkProcessor ()
{
  NS_LOG_FUNCTION (this);
}

LteChunkProcessor::~LteChunkProcessor ()
{
  NS_LOG_FUNCTION (this);
}

void
LteChunkProcessor::AddCallback (LteChunkProcessorCallback c)
{
  NS_LOG_FUNCTION (this);
  m_lteChunkProcessorCallbacks.push_back (c);
}

void
LteChunkProcessor::Start ()
{
  NS_LOG_FUNCTION (this);
  m_sumValues = 0;
  m_totDuration = MicroSeconds (0);
}


void
LteChunkProcessor::EvaluateChunk (const SpectrumValue& sinr, Time duration)
{
  NS_LOG_FUNCTION (this << sinr << duration);
  if (m_sumValues == 0)
    {
      m_sumValues = Create<SpectrumValue> (sinr.GetSpectrumModel ());
    }
  (*m_sumValues) += sinr * duration.GetSeconds ();
  m_totDuration += duration;
}

void
LteChunkProcessor::End ()
{
  NS_LOG_FUNCTION (this);
  if (m_totDuration.GetSeconds () > 0)
    {
      std::vector<LteChunkProcessorCallback>::iterator it;
      for (it = m_lteChunkProcessorCallbacks.begin (); it != m_lteChunkProcessorCallbacks.end (); it++)
        {
          (*it)((*m_sumValues) / m_totDuration.GetSeconds ());
        }
    }
  else
    {
      NS_LOG_WARN ("m_numSinr == 0");
    }
}



void
LteSpectrumValueCatcher::ReportValue (const SpectrumValue& value)
{
  m_value = value.Copy ();
}

Ptr<SpectrumValue>
LteSpectrumValueCatcher::GetValue ()
{
  return m_value;
}


} // namespace ns3
