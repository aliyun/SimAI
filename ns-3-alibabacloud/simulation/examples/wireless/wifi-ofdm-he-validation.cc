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

// This example is used to validate Nist, Yans and Table-based error rate models for HE rates.
//
// It outputs plots of the Frame Success Rate versus the Signal-to-noise ratio for
// Nist, Yans and Table-based error rate models and for every HE MCS value.

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
  std::ofstream yansfile ("yans-frame-success-rate-ax.plt");
  std::ofstream nistfile ("nist-frame-success-rate-ax.plt");
  std::ofstream tablefile ("table-frame-success-rate-ax.plt");
  std::vector <std::string> modes;

  modes.push_back ("HeMcs0");
  modes.push_back ("HeMcs1");
  modes.push_back ("HeMcs2");
  modes.push_back ("HeMcs3");
  modes.push_back ("HeMcs4");
  modes.push_back ("HeMcs5");
  modes.push_back ("HeMcs6");
  modes.push_back ("HeMcs7");
  modes.push_back ("HeMcs8");
  modes.push_back ("HeMcs9");
  modes.push_back ("HeMcs10");
  modes.push_back ("HeMcs11");

  CommandLine cmd (__FILE__);
  cmd.AddValue ("FrameSize", "The frame size", FrameSize);
  cmd.Parse (argc, argv);

  Gnuplot yansplot = Gnuplot ("yans-frame-success-rate-ax.eps");
  Gnuplot nistplot = Gnuplot ("nist-frame-success-rate-ax.eps");
  Gnuplot tableplot = Gnuplot ("table-frame-success-rate-ax.eps");

  Ptr <YansErrorRateModel> yans = CreateObject<YansErrorRateModel> ();
  Ptr <NistErrorRateModel> nist = CreateObject<NistErrorRateModel> ();
  Ptr <TableBasedErrorRateModel> table = CreateObject<TableBasedErrorRateModel> ();
  WifiTxVector txVector;

  for (uint32_t i = 0; i < modes.size (); i++)
    {
      std::cout << modes[i] << std::endl;
      Gnuplot2dDataset yansdataset (modes[i]);
      Gnuplot2dDataset nistdataset (modes[i]);
      Gnuplot2dDataset tabledataset (modes[i]);
      txVector.SetMode (modes[i]);

      for (double snr = -5.0; snr <= 40.0; snr += 0.1)
        {
          double ps = yans->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), FrameSize * 8);
          if (ps < 0.0 || ps > 1.0)
            {
              //error
              exit (1);
            }
          yansdataset.Add (snr, ps);

          ps = nist->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), FrameSize * 8);
          if (ps < 0.0 || ps > 1.0)
            {
              //error
              exit (1);
            }
          nistdataset.Add (snr, ps);

          ps = table->GetChunkSuccessRate (WifiMode (modes[i]), txVector, std::pow (10.0, snr / 10.0), FrameSize * 8);
          if (ps < 0.0 || ps > 1.0)
            {
              //error
              exit (1);
            }
          tabledataset.Add (snr, ps);
        }

      yansplot.AddDataset (yansdataset);
      nistplot.AddDataset (nistdataset);
      tableplot.AddDataset (tabledataset);
    }

  yansplot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  yansplot.SetLegend ("SNR(dB)", "Frame Success Rate");
  yansplot.SetExtra  ("set xrange [-5:55]\n\
set yrange [0:1]\n\
set style line 1 linewidth 5\n\
set style line 2 linewidth 5\n\
set style line 3 linewidth 5\n\
set style line 4 linewidth 5\n\
set style line 5 linewidth 5\n\
set style line 6 linewidth 5\n\
set style line 7 linewidth 5\n\
set style line 8 linewidth 5\n\
set style line 9 linewidth 5\n\
set style line 10 linewidth 5\n\
set style line 11 linewidth 5\n\
set style line 12 linewidth 5\n\
set style increment user");
  yansplot.GenerateOutput (yansfile);
  yansfile.close ();

  nistplot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  nistplot.SetLegend ("SNR(dB)", "Frame Success Rate");
  nistplot.SetExtra  ("set xrange [-5:55]\n\
set yrange [0:1]\n\
set style line 1 linewidth 5\n\
set style line 2 linewidth 5\n\
set style line 3 linewidth 5\n\
set style line 4 linewidth 5\n\
set style line 5 linewidth 5\n\
set style line 6 linewidth 5\n\
set style line 7 linewidth 5\n\
set style line 8 linewidth 5\n\
set style line 9 linewidth 5\n\
set style line 10 linewidth 5\n\
set style line 11 linewidth 5\n\
set style line 12 linewidth 5\n\
set style increment user");

  nistplot.GenerateOutput (nistfile);
  nistfile.close ();

  tableplot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  tableplot.SetLegend ("SNR(dB)", "Frame Success Rate");
  tableplot.SetExtra  ("set xrange [-5:55]\n\
set yrange [0:1]\n\
set style line 1 linewidth 5\n\
set style line 2 linewidth 5\n\
set style line 3 linewidth 5\n\
set style line 4 linewidth 5\n\
set style line 5 linewidth 5\n\
set style line 6 linewidth 5\n\
set style line 7 linewidth 5\n\
set style line 8 linewidth 5\n\
set style line 9 linewidth 5\n\
set style line 10 linewidth 5\n\
set style line 11 linewidth 5\n\
set style line 12 linewidth 5\n\
set style increment user");

  tableplot.GenerateOutput (tablefile);
  tablefile.close ();
}
