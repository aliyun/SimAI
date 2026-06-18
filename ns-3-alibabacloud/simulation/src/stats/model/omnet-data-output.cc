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

#include <fstream>
#include <cstdlib>

#include "ns3/log.h"
#include "ns3/nstime.h"

#include "data-collector.h"
#include "data-calculator.h"
#include "omnet-data-output.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OmnetDataOutput");

//--------------------------------------------------------------
//----------------------------------------------
OmnetDataOutput::OmnetDataOutput()
{
  NS_LOG_FUNCTION (this);

  m_filePrefix = "data";
}
OmnetDataOutput::~OmnetDataOutput()
{
  NS_LOG_FUNCTION (this);
}
/* static */
TypeId
OmnetDataOutput::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OmnetDataOutput")
    .SetParent<DataOutputInterface> ()
    .SetGroupName ("Stats")
    .AddConstructor<OmnetDataOutput> ()
    ;
  return tid;
}
  
void
OmnetDataOutput::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  DataOutputInterface::DoDispose ();
  // end OmnetDataOutput::DoDispose
}

//----------------------------------------------

inline bool isNumeric (const std::string& s) {
  bool decimalPtSeen = false;
  bool exponentSeen = false;
  char last = '\0';

  for (std::string::const_iterator it = s.begin (); it != s.end (); it++)
    {
      if ((*it == '.') && (decimalPtSeen))
        return false;
      else if (*it == '.')
        decimalPtSeen = true;
      else if ((*it == 'e') && exponentSeen)
        return false;
      else if (*it == 'e')
        {
          exponentSeen = true;
          decimalPtSeen = false;
        }
      else if (*it == '-' && it != s.begin () && last != 'e')
        return false;

      last = *it;
    }
  return true;
}

void
OmnetDataOutput::Output (DataCollector &dc)
{
  NS_LOG_FUNCTION (this << &dc);

  std::ofstream scalarFile;
  std::string fn = m_filePrefix +"-"+dc.GetRunLabel ()+ ".sca";
  scalarFile.open (fn.c_str (), std::ios_base::out);

  /// \todo add timestamp to the runlevel
  scalarFile << "run " << dc.GetRunLabel () << std::endl;
  scalarFile << "attr experiment \"" << dc.GetExperimentLabel ()
             << "\"" << std::endl;
  scalarFile << "attr strategy \"" << dc.GetStrategyLabel ()
             << "\"" << std::endl;
  scalarFile << "attr measurement \"" << dc.GetInputLabel ()
             << "\"" << std::endl;
  scalarFile << "attr description \"" << dc.GetDescription ()
             << "\"" << std::endl;

  for (MetadataList::iterator i = dc.MetadataBegin ();
       i != dc.MetadataEnd (); i++) {
      std::pair<std::string, std::string> blob = (*i);
      scalarFile << "attr \"" << blob.first << "\" \"" << blob.second << "\""
                 << std::endl;
    }

  scalarFile << std::endl;
  if (isNumeric (dc.GetInputLabel ())) {
      scalarFile << "scalar . measurement \"" << dc.GetInputLabel ()
                 << "\"" << std::endl;
    }
  for (MetadataList::iterator i = dc.MetadataBegin ();
       i != dc.MetadataEnd (); i++) {
      std::pair<std::string, std::string> blob = (*i);
      if (isNumeric (blob.second)) {
          scalarFile << "scalar . \"" << blob.first << "\" \"" << blob.second << "\""
                     << std::endl;
        }
    }
  OmnetOutputCallback callback (&scalarFile);

  for (DataCalculatorList::iterator i = dc.DataCalculatorBegin ();
       i != dc.DataCalculatorEnd (); i++) {
      (*i)->Output (callback);
    }

  scalarFile << std::endl << std::endl;
  scalarFile.close ();

  // end OmnetDataOutput::Output
}


OmnetDataOutput::OmnetOutputCallback::OmnetOutputCallback
  (std::ostream *scalar) :
  m_scalar (scalar)
{
  NS_LOG_FUNCTION (this << scalar);
}

void
OmnetDataOutput::OmnetOutputCallback::OutputStatistic (std::string context,
                                                       std::string name,
                                                       const StatisticalSummary *statSum)
{
  NS_LOG_FUNCTION (this << context << name << statSum);

  if (context == "")
    context = ".";
  if (name == "")
    name = "\"\"";
  (*m_scalar) << "statistic " << context << " " << name << std::endl;
  if (!isNaN (statSum->getCount ()))
    (*m_scalar) << "field count " << statSum->getCount () << std::endl;
  if (!isNaN (statSum->getSum ()))
    (*m_scalar) << "field sum " << statSum->getSum () << std::endl;
  if (!isNaN (statSum->getMean ()))
    (*m_scalar) << "field mean " << statSum->getMean () << std::endl;
  if (!isNaN (statSum->getMin ()))
    (*m_scalar) << "field min " << statSum->getMin () << std::endl;
  if (!isNaN (statSum->getMax ()))
    (*m_scalar) << "field max " << statSum->getMax () << std::endl;
  if (!isNaN (statSum->getSqrSum ()))
    (*m_scalar) << "field sqrsum " << statSum->getSqrSum () << std::endl;
  if (!isNaN (statSum->getStddev ()))
    (*m_scalar) << "field stddev " << statSum->getStddev () << std::endl;
}

void
OmnetDataOutput::OmnetOutputCallback::OutputSingleton (std::string context,
                                                       std::string name,
                                                       int val)
{
  NS_LOG_FUNCTION (this << context << name << val);

  if (context == "")
    context = ".";
  if (name == "")
    name = "\"\"";
  (*m_scalar) << "scalar " << context << " " << name << " " << val << std::endl;
  // end OmnetDataOutput::OmnetOutputCallback::OutputSingleton
}

void
OmnetDataOutput::OmnetOutputCallback::OutputSingleton (std::string context,
                                                       std::string name,
                                                       uint32_t val)
{
  NS_LOG_FUNCTION (this << context << name << val);

  if (context == "")
    context = ".";
  if (name == "")
    name = "\"\"";
  (*m_scalar) << "scalar " << context << " " << name << " " << val << std::endl;
  // end OmnetDataOutput::OmnetOutputCallback::OutputSingleton
}

void
OmnetDataOutput::OmnetOutputCallback::OutputSingleton (std::string context,
                                                       std::string name,
                                                       double val)
{
  NS_LOG_FUNCTION (this << context << name << val);

  if (context == "")
    context = ".";
  if (name == "")
    name = "\"\"";
  (*m_scalar) << "scalar " << context << " " << name << " " << val << std::endl;
  // end OmnetDataOutput::OmnetOutputCallback::OutputSingleton
}

void
OmnetDataOutput::OmnetOutputCallback::OutputSingleton (std::string context,
                                                       std::string name,
                                                       std::string val)
{
  NS_LOG_FUNCTION (this << context << name << val);

  if (context == "")
    context = ".";
  if (name == "")
    name = "\"\"";
  (*m_scalar) << "scalar " << context << " " << name << " " << val << std::endl;
  // end OmnetDataOutput::OmnetOutputCallback::OutputSingleton
}

void
OmnetDataOutput::OmnetOutputCallback::OutputSingleton (std::string context,
                                                       std::string name,
                                                       Time val)
{
  NS_LOG_FUNCTION (this << context << name << val);

  if (context == "")
    context = ".";
  if (name == "")
    name = "\"\"";
  (*m_scalar) << "scalar " << context << " " << name << " " << val.GetTimeStep () << std::endl;
  // end OmnetDataOutput::OmnetOutputCallback::OutputSingleton
}
