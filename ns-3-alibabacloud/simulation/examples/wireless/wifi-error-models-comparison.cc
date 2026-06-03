/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 University of Washington
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
 * Authors: Sébastien Deronne <sebastien.deronne@gmail.com>
 *          Rohan Patidar <rpatidar@uw.edu>
 */

// This example is to show difference among Nist, Yans and Table-based error rate models.
//
// It outputs plots of the Frame Error Rate versus the Signal-to-noise ratio for
// Nist, Yans and Table-based error rate models and for MCS 0, 4 and 7 value.

#include <fstream>
#include <cmath>
#include "ns3/gnuplot.h"
#include "ns3/command-line.h"
#include "ns3/yans-error-rate-model.h"
#include "ns3/nist-error-rate-model.h"
#include "ns3/table-based-error-rate-model.h"
#include "ns3/wifi-tx-vector.h"

using namespace ns3;

int main (int argc, char *argv[])
{
  uint32_t size = 1500 * 8; //bits
  bool tableErrorModelEnabled = true;
  bool yansErrorModelEnabled = true;
  bool nistErrorModelEnabled = true;
  uint8_t beginMcs = 0;
  uint8_t endMcs = 7;
  uint8_t stepMcs = 4;
  std::string format ("Ht");

  CommandLine cmd (__FILE__);
  cmd.AddValue ("size", "The size in bits", size);
  cmd.AddValue ("frameFormat", "The frame format to use: Ht, Vht or He", format);
  cmd.AddValue ("beginMcs", "The first MCS to test", beginMcs);
  cmd.AddValue ("endMcs", "The last MCS to test", endMcs);
  cmd.AddValue ("stepMcs", "The step between two MCSs to test", stepMcs);
  cmd.AddValue ("includeTableErrorModel", "Flag to include/exclude Table-based error model", tableErrorModelEnabled);
  cmd.AddValue ("includeYansErrorModel", "Flag to include/exclude Yans error model", yansErrorModelEnabled);
  cmd.AddValue ("includeNistErrorModel", "Flag to include/exclude Nist error model", nistErrorModelEnabled);
  cmd.Parse (argc, argv);

  std::ofstream errormodelfile ("wifi-error-rate-models.plt");
  Gnuplot plot = Gnuplot ("wifi-error-rate-models.eps");

  Ptr <YansErrorRateModel> yans = CreateObject<YansErrorRateModel> ();
  Ptr <NistErrorRateModel> nist = CreateObject<NistErrorRateModel> ();
  Ptr <TableBasedErrorRateModel> table = CreateObject<TableBasedErrorRateModel> ();
  WifiTxVector txVector;
  std::vector <std::string> modes;

  std::stringstream mode;
  mode << format << "Mcs" << +beginMcs;
  modes.push_back (mode.str ());
  for (uint8_t mcs = (beginMcs + stepMcs); mcs < endMcs; mcs += stepMcs)
    {
      mode.str ("");
      mode << format << "Mcs" << +mcs;
      modes.push_back (mode.str ());
    }
  mode.str ("");
  mode << format << "Mcs" << +endMcs;
  modes.push_back (mode.str ());

  for (uint32_t i = 0; i < modes.size (); i++)
    {
      std::cout << modes[i] << std::endl;
      Gnuplot2dDataset yansdataset (modes[i]);
      Gnuplot2dDataset nistdataset (modes[i]);
      Gnuplot2dDataset tabledataset (modes[i]);
      txVector.SetMode (modes[i]);

      for (double snr = -5.0; snr <= (endMcs * 5); snr += 0.1)
        {
          double ps = yans->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), size);
          if (ps < 0 || ps > 1)
            {
              //error
              exit (1);
            }
          yansdataset.Add (snr, 1 - ps);
          ps = nist->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), size);
          if (ps < 0 || ps > 1)
            {
              //error
              exit (1);
            }
          nistdataset.Add (snr, 1 - ps);
          ps = table->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), size);
          if (ps < 0 || ps > 1)
            {
              //error
              exit (1);
            }
          tabledataset.Add (snr, 1 - ps);
        }
      
      if (tableErrorModelEnabled)
        {
          std::stringstream ss;
          ss << "Table-" << modes[i];
          tabledataset.SetTitle (ss.str());
          plot.AddDataset (tabledataset);
        }
      if (yansErrorModelEnabled)
        {
          std::stringstream ss;
          ss << "Yans-" << modes[i];
          yansdataset.SetTitle (ss.str ());
          plot.AddDataset (yansdataset);
        }
      if (nistErrorModelEnabled)
        {
          std::stringstream ss;
          ss << "Nist-" << modes[i];
          nistdataset.SetTitle (ss.str ());
          plot.AddDataset (nistdataset);
        }
    }

  plot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  plot.SetLegend ("SNR(dB)", "Frame Error Rate");

  std::stringstream plotExtra;
  plotExtra << "set xrange [-5:" << endMcs * 5 << "]\n\
set log y\n\
set yrange [0.0001:1]\n";

  uint8_t lineNumber = 1;
  for (uint32_t i = 0; i < modes.size (); i++)
    {
      if (tableErrorModelEnabled)
        {
          plotExtra << "set style line " << +lineNumber++ << " linewidth 5 linecolor rgb \"red\" \n";
        }
      if (yansErrorModelEnabled)
        {
          plotExtra << "set style line " << +lineNumber++ << " linewidth 5 linecolor rgb \"green\" \n";
        }
      if (nistErrorModelEnabled)
        {
          plotExtra << "set style line " << +lineNumber++ << " linewidth 5 linecolor rgb \"blue\" \n";
        }
    }

  plotExtra << "set style increment user";
  plot.SetExtra (plotExtra.str ());

  plot.GenerateOutput (errormodelfile);
  errormodelfile.close ();
}
