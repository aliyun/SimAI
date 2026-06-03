/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

// This example is used to validate error rate models for DSSS rates.
//
// It outputs plots of the Frame Success Rate versus the Signal-to-noise ratio
// for the DSSS error rate models and for every DSSS mode.

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
  uint32_t FrameSize = 1500; //bytes
  std::ofstream file ("frame-success-rate-dsss.plt");
  std::vector <std::string> modes;

  modes.push_back ("DsssRate1Mbps");
  modes.push_back ("DsssRate2Mbps");
  modes.push_back ("DsssRate5_5Mbps");
  modes.push_back ("DsssRate11Mbps");

  CommandLine cmd (__FILE__);
  cmd.AddValue ("FrameSize", "The frame size in bytes", FrameSize);
  cmd.Parse (argc, argv);

  Gnuplot plot = Gnuplot ("frame-success-rate-dsss.eps");

  Ptr <YansErrorRateModel> yans = CreateObject<YansErrorRateModel> ();
  Ptr <NistErrorRateModel> nist = CreateObject<NistErrorRateModel> ();
  Ptr <TableBasedErrorRateModel> table = CreateObject<TableBasedErrorRateModel> ();
  WifiTxVector txVector;

  for (uint32_t i = 0; i < modes.size (); i++)
    {
      std::cout << modes[i] << std::endl;
      Gnuplot2dDataset dataset (modes[i]);
      txVector.SetMode (modes[i]);

      for (double snr = -10.0; snr <= 20.0; snr += 0.1)
        {
          double psYans = yans->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), FrameSize * 8);
          if (psYans < 0.0 || psYans > 1.0)
            {
              //error
              exit (1);
            }
          double psNist = nist->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), FrameSize * 8);
          if (psNist < 0.0 || psNist > 1.0)
            {
              std::cout<<psNist<<std::endl;
              //error
              exit (1);
            }
          if (psNist != psYans)
            {
              exit (1);
            }
          double psTable = table->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), FrameSize * 8);
          if (psTable < 0.0 || psTable > 1.0)
            {
              std::cout << psTable << std::endl;
              //error
              exit (1);
            }
          if (psTable != psYans)
            {
              exit (1);
            }
          dataset.Add (snr, psYans);
        }

      plot.AddDataset (dataset);
    }

  plot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  plot.SetLegend ("SNR(dB)", "Frame Success Rate");
  plot.SetExtra  ("set xrange [-10:20]\n\
set yrange [0:1.2]\n\
set style line 1 linewidth 5\n\
set style line 2 linewidth 5\n\
set style line 3 linewidth 5\n\
set style line 4 linewidth 5\n\
set style line 5 linewidth 5\n\
set style line 6 linewidth 5\n\
set style line 7 linewidth 5\n\
set style line 8 linewidth 5\n\
set style increment user"                                                                                                                                                                                                                                                                                                                                   );
  plot.GenerateOutput (file);
  file.close ();
}
