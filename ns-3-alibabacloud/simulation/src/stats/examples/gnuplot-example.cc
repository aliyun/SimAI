/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 University of Washington
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

#include <fstream>

#include "ns3/gnuplot.h"

using namespace ns3;

namespace {

/**
 * This function creates a 2-D plot file.
 */
void Create2DPlotFile ()
{
  std::string fileNameWithNoExtension = "plot-2d";
  std::string graphicsFileName        = fileNameWithNoExtension + ".png";
  std::string plotFileName            = fileNameWithNoExtension + ".plt";
  std::string plotTitle               = "2-D Plot";
  std::string dataTitle               = "2-D Data";

  // Instantiate the plot and set its title.
  Gnuplot plot (graphicsFileName);
  plot.SetTitle (plotTitle);

  // Make the graphics file, which the plot file will create when it
  // is used with Gnuplot, be a PNG file.
  plot.SetTerminal ("png");

  // Set the labels for each axis.
  plot.SetLegend ("X Values", "Y Values");

  // Set the range for the x axis.
  plot.AppendExtra ("set xrange [-6:+6]");

  // Instantiate the dataset, set its title, and make the points be
  // plotted along with connecting lines.
  Gnuplot2dDataset dataset;
  dataset.SetTitle (dataTitle);
  dataset.SetStyle (Gnuplot2dDataset::LINES_POINTS);

  double x;
  double y;

  // Create the 2-D dataset.
  for (x = -5.0; x <= +5.0; x += 1.0)
    {
      // Calculate the 2-D curve
      // 
      //            2
      //     y  =  x   .
      //  
      y = x * x;

      // Add this point.
      dataset.Add (x, y);
    }

  // Add the dataset to the plot.
  plot.AddDataset (dataset);

  // Open the plot file.
  std::ofstream plotFile (plotFileName.c_str());

  // Write the plot file.
  plot.GenerateOutput (plotFile);

  // Close the plot file.
  plotFile.close ();
}


/**
 * This function creates a 2-D plot with error bars file.
 */
void Create2DPlotWithErrorBarsFile ()
{
  std::string fileNameWithNoExtension = "plot-2d-with-error-bars";
  std::string graphicsFileName        = fileNameWithNoExtension + ".png";
  std::string plotFileName            = fileNameWithNoExtension + ".plt";
  std::string plotTitle               = "2-D Plot With Error Bars";
  std::string dataTitle               = "2-D Data With Error Bars";

  // Instantiate the plot and set its title.
  Gnuplot plot (graphicsFileName);
  plot.SetTitle (plotTitle);

  // Make the graphics file, which the plot file will create when it
  // is used with Gnuplot, be a PNG file.
  plot.SetTerminal ("png");

  // Set the labels for each axis.
  plot.SetLegend ("X Values", "Y Values");

  // Set the range for the x axis.
  plot.AppendExtra ("set xrange [-6:+6]");

  // Instantiate the dataset, set its title, and make the points be
  // plotted with no connecting lines.
  Gnuplot2dDataset dataset;
  dataset.SetTitle (dataTitle);
  dataset.SetStyle (Gnuplot2dDataset::POINTS);

  // Make the dataset have error bars in both the x and y directions.
  dataset.SetErrorBars (Gnuplot2dDataset::XY);

  double x;
  double xErrorDelta;
  double y;
  double yErrorDelta;

  // Create the 2-D dataset.
  for (x = -5.0; x <= +5.0; x += 1.0)
    {
      // Calculate the 2-D curve
      // 
      //            2
      //     y  =  x   .
      //  
      y = x * x;

      // Make the uncertainty in the x direction be constant and make
      // the uncertainty in the y direction be a constant fraction of
      // y's value.
      xErrorDelta = 0.25;
      yErrorDelta = 0.1 * y;

      // Add this point with uncertainties in both the x and y
      // direction.
      dataset.Add (x, y, xErrorDelta, yErrorDelta);
    }

  // Add the dataset to the plot.
  plot.AddDataset (dataset);

  // Open the plot file.
  std::ofstream plotFile (plotFileName.c_str());

  // Write the plot file.
  plot.GenerateOutput (plotFile);

  // Close the plot file.
  plotFile.close ();
}


/**
 * This function creates a 3-D plot file.
 */
void Create3DPlotFile ()
{
  std::string fileNameWithNoExtension = "plot-3d";
  std::string graphicsFileName        = fileNameWithNoExtension + ".png";
  std::string plotFileName            = fileNameWithNoExtension + ".plt";
  std::string plotTitle               = "3-D Plot";
  std::string dataTitle               = "3-D Data";

  // Instantiate the plot and set its title.
  Gnuplot plot (graphicsFileName);
  plot.SetTitle (plotTitle);

  // Make the graphics file, which the plot file will create when it
  // is used with Gnuplot, be a PNG file.
  plot.SetTerminal ("png");

  // Rotate the plot 30 degrees around the x axis and then rotate the
  // plot 120 degrees around the new z axis.
  plot.AppendExtra ("set view 30, 120, 1.0, 1.0");

  // Make the zero for the z-axis be in the x-axis and y-axis plane.
  plot.AppendExtra ("set ticslevel 0");

  // Set the labels for each axis.
  plot.AppendExtra ("set xlabel \"X Values\"");
  plot.AppendExtra ("set ylabel \"Y Values\"");
  plot.AppendExtra ("set zlabel \"Z Values\"");

  // Set the ranges for the x and y axis.
  plot.AppendExtra ("set xrange [-5:+5]");
  plot.AppendExtra ("set yrange [-5:+5]");

  // Instantiate the dataset, set its title, and make the points be
  // connected by lines.
  Gnuplot3dDataset dataset;
  dataset.SetTitle (dataTitle);
  dataset.SetStyle ("with lines");

  double x;
  double y;
  double z;

  // Create the 3-D dataset.
  for (x = -5.0; x <= +5.0; x += 1.0)
    {
    for (y = -5.0; y <= +5.0; y += 1.0)
	{
	  // Calculate the 3-D surface
	  // 
	  //            2      2
	  //     z  =  x   *  y   .
	  //  
	  z = x * x * y * y;

	  // Add this point.
	  dataset.Add (x, y, z);
	}

    // The blank line is necessary at the end of each x value's data
    // points for the 3-D surface grid to work.
    dataset.AddEmptyLine ();
    }

  // Add the dataset to the plot.
  plot.AddDataset (dataset);

  // Open the plot file.
  std::ofstream plotFile (plotFileName.c_str());

  // Write the plot file.
  plot.GenerateOutput (plotFile);

  // Close the plot file.
  plotFile.close ();
}

}  // unnamed namespace


int main (int argc, char *argv[])
{
  // Create a 2-D plot file.
  Create2DPlotFile();

  // Create a 2-D plot with error bars file.
  Create2DPlotWithErrorBarsFile();

  // Create a 3-D plot file.
  Create3DPlotFile();

  return 0;
}
