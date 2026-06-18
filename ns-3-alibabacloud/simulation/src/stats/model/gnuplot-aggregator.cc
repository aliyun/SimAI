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

#include "gnuplot-aggregator.h"
#include "ns3/abort.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GnuplotAggregator");

NS_OBJECT_ENSURE_REGISTERED (GnuplotAggregator);

TypeId
GnuplotAggregator::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::GnuplotAggregator")
    .SetParent<DataCollectionObject> ()
    .SetGroupName ("Stats")
  ;

  return tid;
}

GnuplotAggregator::GnuplotAggregator (const std::string &outputFileNameWithoutExtension)
  : m_outputFileNameWithoutExtension (outputFileNameWithoutExtension),
    m_graphicsFileName               (m_outputFileNameWithoutExtension + ".png"),
    m_title                          ("Data Values"),
    m_xLegend                        ("X Values"),
    m_yLegend                        ("Y Values"),
    m_titleSet                       (false),
    m_xAndYLegendsSet                (false),
    m_gnuplot                        (m_graphicsFileName)
{
  NS_LOG_FUNCTION (this);
}

GnuplotAggregator::~GnuplotAggregator ()
{
  NS_LOG_FUNCTION (this);
  if (!m_titleSet)
    {
      NS_LOG_WARN ("Warning: The plot title was not set for the gnuplot aggregator");
    }
  if (!m_xAndYLegendsSet)
    {
      NS_LOG_WARN ("Warning: The axis legends were not set for the gnuplot aggregator");
    }

  std::string dataFileName     = m_outputFileNameWithoutExtension + ".dat";
  std::string plotFileName     = m_outputFileNameWithoutExtension + ".plt";
  std::string scriptFileName   = m_outputFileNameWithoutExtension + ".sh";

  // Open the gnuplot plot and data files.
  std::ofstream plotFile;
  plotFile.open (plotFileName.c_str ());
  std::ofstream dataFile;
  dataFile.open (dataFileName.c_str ());

  // Skip any NaN's that appear in data.
  m_gnuplot.AppendExtra ("set datafile missing \"-nan\"");

  // Write the gnuplot plot and data files.
  m_gnuplot.GenerateOutput (plotFile, dataFile, dataFileName);

  // Close the gnuplot plot and data files.
  plotFile.close ();
  dataFile.close ();

  // Open the shell script file.
  std::ofstream scriptFile;
  scriptFile.open (scriptFileName.c_str ());

  // Write the shell script file.
  scriptFile << "#!/bin/sh" << std::endl;
  scriptFile << std::endl;
  scriptFile << "gnuplot " << plotFileName << std::endl;

  // Close the shell script file.
  scriptFile.close ();
}

void
GnuplotAggregator::Write2d (std::string context, double x, double y)
{
  NS_LOG_FUNCTION (this << context << x << y);

  if (m_2dDatasetMap.count (context) == 0)
    {
      NS_ABORT_MSG ("Dataset " << context << " has not been added");
    }

  if (m_enabled)
    {
      // Add this 2D data point to its dataset.
      m_2dDatasetMap[context].Add (x, y);
    }
}

void
GnuplotAggregator::Write2dWithXErrorDelta (std::string context,
                                           double x,
                                           double y,
                                           double errorDelta)
{
  NS_LOG_FUNCTION (this << context << x << y << errorDelta);

  if (m_2dDatasetMap.count (context) == 0)
    {
      NS_ABORT_MSG ("Dataset " << context << " has not been added");
    }

  if (m_enabled)
    {
      // Add this 2D data point with its error bar to its dataset.
      m_2dDatasetMap[context].Add (x, y, errorDelta);
    }
}

void
GnuplotAggregator::Write2dWithYErrorDelta (std::string context,
                                           double x,
                                           double y,
                                           double errorDelta)
{
  NS_LOG_FUNCTION (this << context << x << y << errorDelta);

  if (m_2dDatasetMap.count (context) == 0)
    {
      NS_ABORT_MSG ("Dataset " << context << " has not been added");
    }

  if (m_enabled)
    {
      // Add this 2D data point with its error bar to its dataset.
      m_2dDatasetMap[context].Add (x, y, errorDelta);
    }
}

void
GnuplotAggregator::Write2dWithXYErrorDelta (std::string context,
                                            double x,
                                            double y,
                                            double xErrorDelta,
                                            double yErrorDelta)
{
  NS_LOG_FUNCTION (this << context << x << y << xErrorDelta << yErrorDelta);

  if (m_2dDatasetMap.count (context) == 0)
    {
      NS_ABORT_MSG ("Dataset " << context << " has not been added");
    }

  if (m_enabled)
    {
      // Add this 2D data point with its error bar to its dataset.
      m_2dDatasetMap[context].Add (x, y, xErrorDelta, yErrorDelta);
    }
}

void
GnuplotAggregator::SetTerminal (const std::string &terminal)
{
  // Change the extension for the graphics file.
  m_graphicsFileName = m_outputFileNameWithoutExtension + "." + terminal;

  // Update the gnuplot, too.
  m_gnuplot.SetTerminal (terminal);
  m_gnuplot.SetOutputFilename (m_graphicsFileName);
}

void
GnuplotAggregator::SetTitle (const std::string &title)
{
  NS_LOG_FUNCTION (this << title);
  m_gnuplot.SetTitle (title);
  m_titleSet = true;
}

void
GnuplotAggregator::SetLegend (const std::string &xLegend, const std::string &yLegend)
{
  NS_LOG_FUNCTION (this << xLegend << yLegend);
  m_gnuplot.SetLegend (xLegend, yLegend);
  m_xAndYLegendsSet = true;
}

void
GnuplotAggregator::SetExtra (const std::string &extra)
{
  NS_LOG_FUNCTION (this << extra);
  m_gnuplot.SetExtra (extra);
}

void
GnuplotAggregator::AppendExtra (const std::string &extra)
{
  NS_LOG_FUNCTION (this << extra);
  m_gnuplot.AppendExtra (extra);
}

void
GnuplotAggregator::Add2dDataset (const std::string &dataset, const std::string &title)
{
  NS_LOG_FUNCTION (this << dataset << title);

  if (m_2dDatasetMap.count (dataset) > 0)
    {
      NS_ABORT_MSG ("Dataset " << dataset << " has already been added");
    }

  // Add this dataset to the map so that its values can be saved.
  Gnuplot2dDataset gnuplot2dDataset (title);
  m_2dDatasetMap[dataset] = gnuplot2dDataset;

  // Add this dataset to the plot so that its values can be plotted.
  m_gnuplot.AddDataset (m_2dDatasetMap[dataset]);
}

void
GnuplotAggregator::Set2dDatasetDefaultExtra (const std::string &extra)
{
  NS_LOG_FUNCTION (extra);
  Gnuplot2dDataset::SetDefaultExtra (extra);
}

void
GnuplotAggregator::Set2dDatasetExtra (const std::string &dataset, const std::string &extra)
{
  NS_LOG_FUNCTION (this << dataset << extra);
  if (m_2dDatasetMap.count (dataset) == 0)
    {
      NS_ABORT_MSG ("Dataset " << dataset << " has not been added");
    }

  // Set the extra parameters for the dataset.
  m_2dDatasetMap[dataset].SetExtra (extra);
}

void
GnuplotAggregator::Write2dDatasetEmptyLine (const std::string &dataset)
{
  NS_LOG_FUNCTION (this << dataset);
  if (m_2dDatasetMap.count (dataset) == 0)
    {
      NS_ABORT_MSG ("Dataset " << dataset << " has not been added");
    }

  if (m_enabled)
    {
      // Add an empty line to the dataset.
      m_2dDatasetMap[dataset].AddEmptyLine ();
    }
}

void
GnuplotAggregator::Set2dDatasetDefaultStyle (enum Gnuplot2dDataset::Style style)
{
  NS_LOG_FUNCTION (style);
  Gnuplot2dDataset::SetDefaultStyle (style);
}

void
GnuplotAggregator::Set2dDatasetStyle (const std::string &dataset, enum Gnuplot2dDataset::Style style)
{
  NS_LOG_FUNCTION (this << dataset << style);
  if (m_2dDatasetMap.count (dataset) == 0)
    {
      NS_ABORT_MSG ("Dataset " << dataset << " has not been added");
    }

  // Set the style for the dataset.
  m_2dDatasetMap[dataset].SetStyle (style);
}

void
GnuplotAggregator::Set2dDatasetDefaultErrorBars (enum Gnuplot2dDataset::ErrorBars errorBars)
{
  NS_LOG_FUNCTION (errorBars);
  Gnuplot2dDataset::SetDefaultErrorBars (errorBars);
}

void
GnuplotAggregator::Set2dDatasetErrorBars (const std::string &dataset, enum Gnuplot2dDataset::ErrorBars errorBars)
{
  NS_LOG_FUNCTION (this << dataset << errorBars);
  if (m_2dDatasetMap.count (dataset) == 0)
    {
      NS_ABORT_MSG ("Dataset " << dataset << " has not been added");
    }

  // Set the error bars for the dataset.
  m_2dDatasetMap[dataset].SetErrorBars (errorBars);
}

void
GnuplotAggregator::SetKeyLocation (enum GnuplotAggregator::KeyLocation keyLocation)
{
  NS_LOG_FUNCTION (this << keyLocation);
  // Set the specified key location.
  switch (keyLocation)
    {
    case NO_KEY:
      m_gnuplot.AppendExtra ("set key off");
      break;
    case KEY_ABOVE:
      m_gnuplot.AppendExtra ("set key outside center above");
      break;
    case KEY_BELOW:
      m_gnuplot.AppendExtra ("set key outside center below");
      break;
    default:
      m_gnuplot.AppendExtra ("set key inside");
      break;
    }
}

} // namespace ns3

