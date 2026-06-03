/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 University of Washington
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
 * Author: Mitch Watrous (watrous@u.washington.edu)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "gnuplot-helper.h"
#include "ns3/abort.h"
#include "ns3/assert.h"
#include "ns3/config.h"
#include "ns3/log.h"
#include "ns3/get-wildcard-matches.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GnuplotHelper");

GnuplotHelper::GnuplotHelper ()
  : m_aggregator                     (0),
    m_plotProbeCount                 (0),
    m_outputFileNameWithoutExtension ("gnuplot-helper"),
    m_title                          ("Gnuplot Helper Plot"),
    m_xLegend                        ("X Values"),
    m_yLegend                        ("Y Values"),
    m_terminalType                   ("png")
{
  NS_LOG_FUNCTION (this);

  // Note that this does not construct an aggregator. It will be
  // constructed later when needed.
}

GnuplotHelper::GnuplotHelper (const std::string &outputFileNameWithoutExtension,
                              const std::string &title,
                              const std::string &xLegend,
                              const std::string &yLegend,
                              const std::string &terminalType)
  : m_aggregator                     (0),
    m_plotProbeCount                 (0),
    m_outputFileNameWithoutExtension (outputFileNameWithoutExtension),
    m_title                          (title),
    m_xLegend                        (xLegend),
    m_yLegend                        (yLegend),
    m_terminalType                   (terminalType)
{
  NS_LOG_FUNCTION (this);

  // Construct the aggregator.
  ConstructAggregator ();
}

GnuplotHelper::~GnuplotHelper ()
{
  NS_LOG_FUNCTION (this);
}

void
GnuplotHelper::ConfigurePlot (const std::string &outputFileNameWithoutExtension,
                              const std::string &title,
                              const std::string &xLegend,
                              const std::string &yLegend,
                              const std::string &terminalType)
{
  NS_LOG_FUNCTION (this << outputFileNameWithoutExtension << title
                        << xLegend << yLegend <<  terminalType);

  // See if an aggregator has already been constructed.
  if (m_aggregator != 0)
    {
      NS_LOG_WARN ("An existing aggregator object " << m_aggregator <<
                   " may be destroyed if no references remain.");
    }

  // Store these so that they can be used to construct the aggregator.
  m_outputFileNameWithoutExtension = outputFileNameWithoutExtension;
  m_title                          = title;
  m_xLegend                        = xLegend;
  m_yLegend                        = yLegend;
  m_terminalType                   = terminalType;

  // Construct the aggregator.
  ConstructAggregator ();
}

void
GnuplotHelper::PlotProbe (const std::string &typeId,
                          const std::string &path,
                          const std::string &probeTraceSource,
                          const std::string &title,
                          enum GnuplotAggregator::KeyLocation keyLocation)
{
  NS_LOG_FUNCTION (this << typeId << path << probeTraceSource << title << keyLocation);

  // Get a pointer to the aggregator.
  Ptr<GnuplotAggregator> aggregator = GetAggregator ();

  // Add a subtitle to the title to show the trace source's path.
  aggregator->SetTitle ( m_title + " \\n\\nTrace Source Path: " + path);

  // Set the default dataset plotting style for the values.
  aggregator->Set2dDatasetDefaultStyle (Gnuplot2dDataset::LINES_POINTS);

  // Set the location of the key in the plot.
  aggregator->SetKeyLocation (keyLocation);

  std::string pathWithoutLastToken;
  std::string lastToken;

  // See if the path has any wildcards.
  bool pathHasNoWildcards = path.find ("*") == std::string::npos;

  // Remove the last token from the path; this should correspond to the 
  // trace source attribute.
  size_t lastSlash = path.find_last_of ("/");
  if (lastSlash == std::string::npos)
    {
      pathWithoutLastToken = path;
      lastToken = "";
    }
  else
    {
      // Chop off up to last token.
      pathWithoutLastToken = path.substr (0, lastSlash);

      // Save the last token without the last slash.
      lastToken = path.substr (lastSlash + 1, std::string::npos);
    }

  // See if there are any matches for the probe's path with the last
  // token removed; this corresponds to the traced object itself.
  NS_LOG_DEBUG ("Searching config database for trace source " << path);
  Config::MatchContainer matches = Config::LookupMatches (pathWithoutLastToken);
  uint32_t matchCount = matches.GetN ();
  NS_LOG_DEBUG ("Found " << matchCount << " matches for trace source " << path);

  // This is used to make the probe's context be unique.
  std::string matchIdentifier;

  // Hook one or more probes and the aggregator together.
  if (matchCount == 1 && pathHasNoWildcards)
    {
      // Connect the probe to the aggregator only once because there
      // is only one matching config path.  There is no need to find
      // the wildcard matches because the passed in path has none.
      matchIdentifier = "0";
      ConnectProbeToAggregator (typeId,
                                matchIdentifier,
                                path,
                                probeTraceSource,
                                title);
    }
  else if (matchCount > 0)
    {
      // Handle all of the matches if there are more than one.
      for (uint32_t i = 0; i < matchCount; i++)
        {
          // Set the match identifier.
          std::ostringstream matchIdentifierStream;
          matchIdentifierStream << i;
          matchIdentifier = matchIdentifierStream.str ();

          // Construct the matched path and get the matches for each
          // of the wildcards.
          std::string wildcardSeparator = " ";
          std::string matchedPath = matches.GetMatchedPath (i) + lastToken;
          std::string wildcardMatches = GetWildcardMatches (path,
                                                            matchedPath,
                                                            wildcardSeparator);

          // Connect the probe to the aggregator for this match.
          ConnectProbeToAggregator (typeId,
                                    matchIdentifier,
                                    matchedPath,
                                    probeTraceSource,
                                    title + "-" + wildcardMatches);
        }
    }
  else
    {
      // There is a problem if there are no matching config paths.
      NS_FATAL_ERROR ("Lookup of " << path << " got no matches");
    }
}

void
GnuplotHelper::AddProbe (const std::string &typeId,
                         const std::string &probeName,
                         const std::string &path)
{
  NS_LOG_FUNCTION (this << typeId << probeName << path);

  // See if this probe had already been added.
  if (m_probeMap.count (probeName) > 0)
    {
      NS_ABORT_MSG ("That probe has already been added");
    }

  // Prepare the factory to create an object with the requested type.
  m_factory.SetTypeId (typeId);

  // Create a base class object in order to validate the type.
  Ptr<Probe> probe = m_factory.Create ()->GetObject<Probe> ();
  if (probe == 0)
    {
      NS_ABORT_MSG ("The requested type is not a probe");
    }

  // Set the probe's name.
  probe->SetName (probeName);

  // Set the path.  Note that no return value is checked here.
  probe->ConnectByPath (path);

  // Enable logging of data for the probe.
  probe->Enable ();

  // Add this probe to the map so that its values can be used.
  m_probeMap[probeName] = std::make_pair (probe, typeId);
}

void
GnuplotHelper::AddTimeSeriesAdaptor (const std::string &adaptorName)
{
  NS_LOG_FUNCTION (this << adaptorName);

  // See if this time series adaptor had already been added.
  if (m_timeSeriesAdaptorMap.count (adaptorName) > 0)
    {
      NS_ABORT_MSG ("That time series adaptor has already been added");
    }

  // Create the time series adaptor.
  Ptr<TimeSeriesAdaptor> timeSeriesAdaptor = CreateObject<TimeSeriesAdaptor> ();

  // Enable logging of data for the time series adaptor.
  timeSeriesAdaptor->Enable ();

  // Add this time series adaptor to the map so that can be used.
  m_timeSeriesAdaptorMap[adaptorName] = timeSeriesAdaptor;
}

Ptr<Probe>
GnuplotHelper::GetProbe (std::string probeName) const
{
  // Look for the probe.
  std::map<std::string, std::pair <Ptr<Probe>, std::string> >::const_iterator mapIterator = m_probeMap.find (probeName);

  // Return the probe if it has been added.
  if (mapIterator != m_probeMap.end ())
    {
      return mapIterator->second.first;
    }
  else
    {
      NS_ABORT_MSG ("That probe has not been added");
    }
}

Ptr<GnuplotAggregator>
GnuplotHelper::GetAggregator ()
{
  NS_LOG_FUNCTION (this);

  // Do a lazy construction of the aggregator if it hasn't already
  // been constructed.
  if (!m_aggregator)
    {
      ConstructAggregator ();
    }
  return m_aggregator;
}

void
GnuplotHelper::ConstructAggregator ()
{
  NS_LOG_FUNCTION (this);

  // Create the aggregator.
  m_aggregator = CreateObject<GnuplotAggregator> (m_outputFileNameWithoutExtension);

  // Set the aggregator's properties.
  m_aggregator->SetTerminal (m_terminalType);
  m_aggregator->SetTitle (m_title);
  m_aggregator->SetLegend (m_xLegend, m_yLegend);

  // Enable logging of data for the aggregator.
  m_aggregator->Enable ();
}

void
GnuplotHelper::ConnectProbeToAggregator (const std::string &typeId,
                                         const std::string &matchIdentifier,
                                         const std::string &path,
                                         const std::string &probeTraceSource,
                                         const std::string &title)
{
  NS_LOG_FUNCTION (this << typeId << matchIdentifier << path << probeTraceSource
                        << title);

  Ptr<GnuplotAggregator> aggregator = GetAggregator ();

  // Increment the total number of plot probes that have been created.
  m_plotProbeCount++;

  // Create a unique name for this probe.
  std::ostringstream probeNameStream;
  probeNameStream << "PlotProbe-" << m_plotProbeCount;
  std::string probeName = probeNameStream.str ();

  // Create a unique dataset context string for this probe.
  std::string probeContext = probeName
    + "/" + matchIdentifier + "/" + probeTraceSource;

  // Add the probe to the map of probes, which will keep the probe in
  // memory after this function ends.
  AddProbe (typeId, probeName, path);

  // Because the callbacks to the probes' trace sources don't use the
  // probe's context, a unique adaptor needs to be created for each
  // probe context so that information is not lost.
  AddTimeSeriesAdaptor (probeContext);

  // Connect the probe to the adaptor.
  if (m_probeMap[probeName].second == "ns3::DoubleProbe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkDouble,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::BooleanProbe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkBoolean,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::PacketProbe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkUinteger32,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::ApplicationPacketProbe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkUinteger32,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::Ipv4PacketProbe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkUinteger32,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::Ipv6PacketProbe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkUinteger32,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::Uinteger8Probe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkUinteger8,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::Uinteger16Probe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkUinteger16,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::Uinteger32Probe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkUinteger32,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else if (m_probeMap[probeName].second == "ns3::TimeProbe")
    {
      m_probeMap[probeName].first->TraceConnectWithoutContext
        (probeTraceSource,
        MakeCallback (&TimeSeriesAdaptor::TraceSinkDouble,
                      m_timeSeriesAdaptorMap[probeContext]));
    }
  else
    {
      NS_FATAL_ERROR ("Unknown probe type " << m_probeMap[probeName].second << "; need to add support in the helper for this");
    }

  // Connect the adaptor to the aggregator.
  std::string adaptorTraceSource = "Output";
  m_timeSeriesAdaptorMap[probeContext]->TraceConnect
    (adaptorTraceSource,
    probeContext,
    MakeCallback (&GnuplotAggregator::Write2d, aggregator));

  // Add the dataset to the plot.
  aggregator->Add2dDataset (probeContext, title);
}

} // namespace ns3

