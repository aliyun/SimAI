/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Author: L. Felipe Perrone (perrone@bucknell.edu)
 *
 * Modified by: Mitch Watrous (watrous@u.washington.edu)
 *
 */

#ifndef GNUPLOT_AGGREGATOR_H
#define GNUPLOT_AGGREGATOR_H

#include <map>
#include <string>
#include "ns3/gnuplot.h"
#include "ns3/data-collection-object.h"

namespace ns3 {

/**
 * \ingroup aggregator
 * This aggregator produces output used to make gnuplot plots.
 **/
class GnuplotAggregator : public DataCollectionObject
{
public:
  /// The location of the key in the plot.
  enum KeyLocation
  {
    NO_KEY,
    KEY_INSIDE,
    KEY_ABOVE,
    KEY_BELOW
  };

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  /**
   * \param outputFileNameWithoutExtension name of gnuplot related
   * files to write with no extension
   *
   * Constructs a gnuplot file aggregator that will create a space
   * separated gnuplot data file named outputFileNameWithoutExtension
   * + ".dat", a gnuplot control file named
   * outputFileNameWithoutExtension + ".plt", and a shell script to
   * generate the gnuplot named outputFileNameWithoutExtension +
   * ".sh".
   */
  GnuplotAggregator (const std::string &outputFileNameWithoutExtension);

  virtual ~GnuplotAggregator ();

  // Below are hooked to connectors exporting data
  // They are not overloaded since it confuses the compiler when made
  // into callbacks

  /**
   * \param context specifies the gnuplot 2D dataset for these values
   * \param x x coordinate for the new data point
   * \param y y coordinate for the new data point
   *
   * \brief Writes a 2D value to a 2D gnuplot dataset.
   *
   * Use this method with error bar style NONE.
   */
  void Write2d (std::string context, double x, double y);

  /**
   * \param context specifies the gnuplot 2D dataset for these values
   * \param x x coordinate for the new data point
   * \param y y coordinate for the new data point
   * \param errorDelta x data point uncertainty
   *
   * \brief Writes a 2D value to a 2D gnuplot dataset with error bars
   * in the x direction.
   *
   * Use this method with error bar style X.
   */
  void Write2dWithXErrorDelta (std::string context,
                               double x,
                               double y,
                               double errorDelta);

  /**
   * \param context specifies the gnuplot 2D dataset for these values
   * \param x x coordinate for the new data point
   * \param y y coordinate for the new data point
   * \param errorDelta y data point uncertainty
   *
   * \brief Writes a 2D value to a 2D gnuplot dataset with error bars
   * in the y direction.
   *
   * Use this method with error bar style Y.
   */
  void Write2dWithYErrorDelta (std::string context,
                               double x,
                               double y,
                               double errorDelta);

  /**
   * \param context specifies the gnuplot 2D dataset for these values
   * \param x x coordinate for the new data point
   * \param y y coordinate for the new data point
   * \param xErrorDelta x data point uncertainty
   * \param yErrorDelta y data point uncertainty
   *
   * \brief Writes a 2D value to a 2D gnuplot dataset with error bars
   * in the x and y directions.
   *
   * Use this method with error bar style XY.
   */
  void Write2dWithXYErrorDelta (std::string context,
                                double x,
                                double y,
                                double xErrorDelta,
                                double yErrorDelta);

  // Methods to configure the plot

  /**
   * \param terminal terminal setting string for output. The default terminal
   * string is "png"
   */
  void SetTerminal (const std::string &terminal);

  /**
   * \param title set new plot title string to use for this plot.
   */
  void SetTitle (const std::string &title);

  /**
   * \param xLegend the legend for the x horizontal axis
   * \param yLegend the legend for the y vertical axis
   */
  void SetLegend (const std::string &xLegend, const std::string &yLegend);

  /**
   * \param extra set extra gnuplot directive for output.
   */
  void SetExtra (const std::string &extra);

  /**
   * \param extra append extra gnuplot directive for output.
   */
  void AppendExtra (const std::string &extra);

  // Methods for datasets

  /**
   * \param dataset the gnuplot 2D dataset to be plotted.
   * \param title the title to be associated to this dataset.
   *
   * \brief Adds a 2D dataset to the plot.
   *
   * Creates an empty dataset. Usually, the dataset's title is
   * displayed in the legend box.
   *
   * The string in dataset should match the context for the Collector
   * it is connected to.
   */
  void Add2dDataset (const std::string &dataset, const std::string &title);

  /**
   * \param extra       extra formatting
   *
   * \brief Change extra formatting style parameters for newly created
   * objects.
   */
  static void Set2dDatasetDefaultExtra (const std::string &extra);

  /**
   * \param dataset the gnuplot 2D dataset to be plotted.
   * \param extra       extra formatting
   *
   * \brief Add extra formatting parameters to this dataset.
   *
   * The string in dataset should match the context for the Collector
   * it is connected to.
   */
  void Set2dDatasetExtra (const std::string &dataset, const std::string &extra);

  /**
   * \param dataset the gnuplot 2D dataset to be plotted.
   *
   * \brief Add an empty line in the data output sequence.
   *
   * Writes an empty line in the plot data, which breaks continuous
   * lines and does other things in the output.
   *
   * The string in dataset should match the context for the Collector
   * it is connected to.
   */
  void Write2dDatasetEmptyLine (const std::string &dataset);

  /**
   * \param style the style of plotting to use for newly created datasets.
   *
   * \brief Change default style for all newly created objects.
   */
  static void Set2dDatasetDefaultStyle (enum Gnuplot2dDataset::Style style);

  /**
   * \param dataset the gnuplot 2D dataset to be plotted.
   * \param style the style of plotting to use for this dataset.
   *
   * \brief Set the style of plotting to use for this dataset.
   *
   * The string in dataset should match the context for the Collector
   * it is connected to.
   */
  void Set2dDatasetStyle (const std::string &dataset, enum Gnuplot2dDataset::Style style);

  /**
   * \param errorBars the style of errorbars to use for newly created datasets.
   *
   * \brief Change default errorbars style for all newly created objects.
   */
  static void Set2dDatasetDefaultErrorBars (enum Gnuplot2dDataset::ErrorBars errorBars);

  /**
   * \param dataset the gnuplot 2D dataset to be plotted.
   * \param errorBars the style of errorbars to display.
   *
   * \brief Set the error bars to use for this dataset.
   *
   * If you use any style other than none, you need
   * to make sure you store the delta information in
   * this dataset with the right GnuplotDataset::Add
   * method.
   *
   * The string in dataset should match the context for the Collector
   * it is connected to.
   */
  void Set2dDatasetErrorBars (const std::string &dataset, enum Gnuplot2dDataset::ErrorBars errorBars);

  /**
   * \param keyLocation the location of the key in the plot.
   *
   * \brief Set the location of the key in the plot.
   */
  void SetKeyLocation (enum KeyLocation keyLocation);

private:
  /// The output file name without any extension.
  std::string m_outputFileNameWithoutExtension;

  /// The graphics file name with its extension.
  std::string m_graphicsFileName;

  /// The title.
  std::string m_title;

  /// The terminal.
  std::string m_terminal;

  /// The x legend.
  std::string m_xLegend;

  /// The y legend.
  std::string m_yLegend;

  /// The extra gnuplot command that will be added to the gnuplot
  /// control statements.
  std::string m_extra;

  /// Set equal to true after setting the title.
  bool m_titleSet;

  /// Set equal to true after setting the x and y legends.
  bool m_xAndYLegendsSet;

  /// Used to create gnuplot files.
  Gnuplot m_gnuplot;

  /// Maps context strings to 2D datasets.
  std::map<std::string, Gnuplot2dDataset> m_2dDatasetMap;

}; // class GnuplotAggregator


} // namespace ns3

#endif // GNUPLOT_AGGREGATOR_H
