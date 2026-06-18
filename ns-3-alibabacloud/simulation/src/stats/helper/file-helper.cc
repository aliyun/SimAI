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

#include "file-helper.h"
#include "ns3/abort.h"
#include "ns3/log.h"
#include "ns3/config.h"
#include "ns3/get-wildcard-matches.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FileHelper");

FileHelper::FileHelper ()
  : m_aggregator                     (0),
    m_fileProbeCount                 (0),
    m_fileType                       (FileAggregator::SPACE_SEPARATED),
    m_outputFileNameWithoutExtension ("file-helper"),
    m_hasHeadingBeenSet              (false)
{
  NS_LOG_FUNCTION (this);

  // Note that this does not construct an aggregator. It will be
  // constructed later when needed.
}

FileHelper::FileHelper (const std::string &outputFileNameWithoutExtension,
                        enum FileAggregator::FileType fileType)
  : m_aggregator                     (0),
    m_fileProbeCount                 (0),
    m_fileType                       (fileType),
    m_outputFileNameWithoutExtension (outputFileNameWithoutExtension),
    m_hasHeadingBeenSet              (false)
{
  NS_LOG_FUNCTION (this);

  // Note that this does not construct an aggregator. It will be
  // constructed later when needed.
}

FileHelper::~FileHelper ()
{
  NS_LOG_FUNCTION (this);
}

void
FileHelper::ConfigureFile (const std::string &outputFileNameWithoutExtension,
                           enum FileAggregator::FileType fileType)
{
  NS_LOG_FUNCTION (this << outputFileNameWithoutExtension << fileType);

  // See if an aggregator has already been constructed.
  if (m_aggregator != 0)
    {
      NS_LOG_WARN ("An existing aggregator object " << m_aggregator <<
                   " may be destroyed if no references remain.");
    }

  // Store these so that they can be used to construct the aggregator.
  m_fileType = fileType;
  m_outputFileNameWithoutExtension = outputFileNameWithoutExtension;
  m_hasHeadingBeenSet = false;

  // Note that this does not construct an aggregator. It will be
  // constructed later when needed.
}

void
FileHelper::WriteProbe (const std::string &typeId,
                        const std::string &path,
                        const std::string &probeTraceSource)
{
  NS_LOG_FUNCTION (this << typeId << path << probeTraceSource);

  std::string pathWithoutLastToken;
  std::string lastToken;

  // See if the path has any wildcards.
  bool pathHasNoWildcards = path.find ("*") == std::string::npos;

  // Remove the last token from the path.
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
  // token removed.
  Config::MatchContainer matches = Config::LookupMatches (pathWithoutLastToken);
  uint32_t matchCount = matches.GetN ();

  // This is used to make the probe's context be unique.
  std::string matchIdentifier;

  /// This is used to indicate if multiple aggregators are needed.
  bool onlyOneAggregator;

  // Hook one or more probes and one or more aggregators together.
  if (matchCount == 1 && pathHasNoWildcards)
    {
      // Connect the probe to the aggregator only once because there
      // is only one matching config path.  There is no need to find
      // the wildcard matches because the passed in path has none.
      matchIdentifier = "0";
      onlyOneAggregator = true;
      ConnectProbeToAggregator (typeId,
                                matchIdentifier,
                                path,
                                probeTraceSource,
                                m_outputFileNameWithoutExtension,
                                onlyOneAggregator);
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
          onlyOneAggregator = false;

          // Construct the matched path and get the matches for each
          // of the wildcards.
          std::string wildcardSeparator = "-";
          std::string matchedPath = matches.GetMatchedPath (i) + lastToken;
          std::string wildcardMatches = GetWildcardMatches (path,
                                                            matchedPath,
                                                            wildcardSeparator);

          // Connect the probe to the aggregator for this match.
          ConnectProbeToAggregator (typeId,
                                    matchIdentifier,
                                    matchedPath,
                                    probeTraceSource,
                                    m_outputFileNameWithoutExtension + "-" + wildcardMatches,
                                    onlyOneAggregator);
        }
    }
  else
    {
      // There is a problem if there are no matching config paths.
      NS_FATAL_ERROR ("Lookup of " << path << " got no matches");
    }
}

void
FileHelper::AddProbe (const std::string &typeId,
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
FileHelper::AddTimeSeriesAdaptor (const std::string &adaptorName)
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

  // Add this time series adaptor to the map so that it can be used.
  m_timeSeriesAdaptorMap[adaptorName] = timeSeriesAdaptor;
}

void
FileHelper::AddAggregator (const std::string &aggregatorName,
                           const std::string &outputFileName,
                           bool onlyOneAggregator)
{
  NS_LOG_FUNCTION (this << aggregatorName << outputFileName << onlyOneAggregator);

  // See if this file aggregator had already been added.
  if (m_aggregatorMap.count (aggregatorName) > 0)
    {
      NS_ABORT_MSG ("That file aggregator has already been added");
    }

  // If there is only going to be 1 file aggregator, then use the one
  // already constructed in the map.
  if (onlyOneAggregator)
    {
      // Get a pointer to the aggregator.
      Ptr<FileAggregator> singleAggregator = GetAggregatorSingle ();

      m_aggregatorMap[aggregatorName] = singleAggregator;
      return;
    }

  // Create the file aggregator with the proper file type.
  Ptr<FileAggregator> multipleAggregator =
    CreateObject<FileAggregator> (outputFileName, m_fileType);

  // Set all of the format strings for the aggregator.
  multipleAggregator->Set1dFormat (m_1dFormat);
  multipleAggregator->Set2dFormat (m_2dFormat);
  multipleAggregator->Set3dFormat (m_3dFormat);
  multipleAggregator->Set4dFormat (m_4dFormat);
  multipleAggregator->Set5dFormat (m_5dFormat);
  multipleAggregator->Set6dFormat (m_6dFormat);
  multipleAggregator->Set7dFormat (m_7dFormat);
  multipleAggregator->Set8dFormat (m_8dFormat);
  multipleAggregator->Set9dFormat (m_9dFormat);
  multipleAggregator->Set10dFormat (m_10dFormat);

  // Set the heading
  multipleAggregator->SetHeading (m_heading);

  // Enable logging of data for the file aggregator.
  multipleAggregator->Enable ();

  // Add this file aggregator to the map so that it can be used.
  m_aggregatorMap[aggregatorName] = multipleAggregator;
}

Ptr<Probe>
FileHelper::GetProbe (std::string probeName) const
{
  NS_LOG_FUNCTION (this << probeName);

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

Ptr<FileAggregator>
FileHelper::GetAggregatorSingle ()
{
  NS_LOG_FUNCTION (this);

  // Do a lazy construction of the single aggregator if it hasn't
  // already been constructed.
  if (!m_aggregator)
    {
      // Create the aggregator.
      std::string outputFileName = m_outputFileNameWithoutExtension + ".txt";
      m_aggregator = CreateObject<FileAggregator> (outputFileName, m_fileType);

      // Set all of the format strings for the aggregator.
      m_aggregator->Set1dFormat (m_1dFormat);
      m_aggregator->Set2dFormat (m_2dFormat);
      m_aggregator->Set3dFormat (m_3dFormat);
      m_aggregator->Set4dFormat (m_4dFormat);
      m_aggregator->Set5dFormat (m_5dFormat);
      m_aggregator->Set6dFormat (m_6dFormat);
      m_aggregator->Set7dFormat (m_7dFormat);
      m_aggregator->Set8dFormat (m_8dFormat);
      m_aggregator->Set9dFormat (m_9dFormat);
      m_aggregator->Set10dFormat (m_10dFormat);

      // Set the heading
      m_aggregator->SetHeading (m_heading);

      // Enable logging of data for the aggregator.
      m_aggregator->Enable ();
    }
  return m_aggregator;
}

Ptr<FileAggregator>
FileHelper::GetAggregatorMultiple (const std::string &aggregatorName,
                                   const std::string &outputFileName)
{
  NS_LOG_FUNCTION (this);

  // See if this file aggregator had already been added.
  if (m_aggregatorMap.count (aggregatorName) > 0)
    {
      return m_aggregatorMap[aggregatorName];
    }

  // Do a lazy construction of the aggregator if it hasn't already
  // been constructed.
  bool onlyOneAggregator = false;
  AddAggregator (aggregatorName,
                 outputFileName,
                 onlyOneAggregator);

  return m_aggregatorMap[aggregatorName];
}

void
FileHelper::SetHeading (const std::string &heading)
{
  NS_LOG_FUNCTION (this << heading);

  m_hasHeadingBeenSet = true;
  m_heading = heading;
}

void
FileHelper::Set1dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_1dFormat = format;
}

void
FileHelper::Set2dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_2dFormat = format;
}

void
FileHelper::Set3dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_3dFormat = format;
}

void
FileHelper::Set4dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_4dFormat = format;
}

void
FileHelper::Set5dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_5dFormat = format;
}

void
FileHelper::Set6dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_6dFormat = format;
}

void
FileHelper::Set7dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_7dFormat = format;
}

void
FileHelper::Set8dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_8dFormat = format;
}

void
FileHelper::Set9dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_9dFormat = format;
}

void
FileHelper::Set10dFormat (const std::string &format)
{
  NS_LOG_FUNCTION (this << format);

  m_10dFormat = format;
}

void
FileHelper::ConnectProbeToAggregator (const std::string &typeId,
                                      const std::string &matchIdentifier,
                                      const std::string &path,
                                      const std::string &probeTraceSource,
                                      const std::string &outputFileNameWithoutExtension,
                                      bool onlyOneAggregator)
{
  NS_LOG_FUNCTION (this << typeId << matchIdentifier << path << probeTraceSource
                        << outputFileNameWithoutExtension << onlyOneAggregator);

  // Increment the total number of file probes that have been created.
  m_fileProbeCount++;

  // Create a unique name for this probe.
  std::ostringstream probeNameStream;
  probeNameStream << "FileProbe-" << m_fileProbeCount;
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

  // Add the aggregator to the map of aggregators, which will keep the
  // aggregator in memory after this function ends.
  std::string outputFileName = outputFileNameWithoutExtension + ".txt";
  AddAggregator (probeContext, outputFileName, onlyOneAggregator);

  // Connect the adaptor to the aggregator.
  std::string adaptorTraceSource = "Output";
  m_timeSeriesAdaptorMap[probeContext]->TraceConnect
    (adaptorTraceSource,
    probeContext,
    MakeCallback (&FileAggregator::Write2d,
                  m_aggregatorMap[probeContext]));
}

} // namespace ns3

