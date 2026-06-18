/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Drexel University
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
 * Author: Joe Kopena (tjkopena@cs.drexel.edu)
 */

#include "ns3/object.h"
#include "ns3/log.h"

#include "data-collector.h"
#include "data-calculator.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("DataCollector");

//--------------------------------------------------------------
//----------------------------------------------
DataCollector::DataCollector() {
  NS_LOG_FUNCTION (this);
  // end DataCollector::DataCollector
}

DataCollector::~DataCollector() {
  NS_LOG_FUNCTION (this);
  // end DataCollector::~DataCollector
}

/* static */
TypeId
DataCollector::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DataCollector")
    .SetParent<Object> ()
    .SetGroupName ("Stats")
    .AddConstructor<DataCollector> ()
    ;
  return tid;
}
  
void DataCollector::DoDispose () {
  NS_LOG_FUNCTION (this);

  m_calcList.clear ();
  m_metadata.clear ();

  Object::DoDispose ();
  // end DataCollector::DoDispose
}

void
DataCollector::DescribeRun (std::string experiment,
                            std::string strategy,
                            std::string input,
                            std::string runID,
                            std::string description)
{
  NS_LOG_FUNCTION (this << experiment << strategy << input << runID << description);

  m_experimentLabel = experiment;
  m_strategyLabel = strategy;
  m_inputLabel = input;
  m_runLabel = runID;
  m_description = description;

  // end DataCollector::DescribeRun
}

void
DataCollector::AddDataCalculator (Ptr<DataCalculator> datac)
{
  NS_LOG_FUNCTION (this << datac);

  m_calcList.push_back (datac);

  // end DataCollector::AddDataCalculator
}

DataCalculatorList::iterator
DataCollector::DataCalculatorBegin ()
{
  return m_calcList.begin ();
  // end DataCollector::DataCalculatorBegin
}
DataCalculatorList::iterator
DataCollector::DataCalculatorEnd ()
{
  return m_calcList.end ();
  // end DataCollector::DataCalculatorEnd
}

void
DataCollector::AddMetadata (std::string key, std::string value)
{
  NS_LOG_FUNCTION (this << key << value);

  std::pair<std::string, std::string> blob (key, value);
  m_metadata.push_back (blob);
  // end DataCollector::AddMetadata
}
void
DataCollector::AddMetadata (std::string key, uint32_t value)
{
  NS_LOG_FUNCTION (this << key << value);

  std::stringstream st;
  st << value;

  std::pair<std::string, std::string> blob (key, st.str ());
  m_metadata.push_back (blob);
  // end DataCollector::AddMetadata
}
void
DataCollector::AddMetadata (std::string key, double value)
{
  NS_LOG_FUNCTION (this << key << value);

  std::stringstream st;
  st << value;

  std::pair<std::string, std::string> blob (key, st.str ());
  m_metadata.push_back (blob);
  // end DataCollector::AddMetadata
}

MetadataList::iterator
DataCollector::MetadataBegin ()
{
  return m_metadata.begin ();
  // end DataCollector::MetadataBegin
}
MetadataList::iterator
DataCollector::MetadataEnd ()
{
  return m_metadata.end ();
  // end DataCollector::MetadataEnd
}
