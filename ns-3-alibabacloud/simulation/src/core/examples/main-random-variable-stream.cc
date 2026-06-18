/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Timo Bingmann
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
 * Author: Timo Bingmann <timo.bingmann@student.kit.edu>
 */
#include "ns3/random-variable-stream.h"
#include "ns3/ptr.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/integer.h"
#include "ns3/gnuplot.h"
#include "ns3/command-line.h"
#include <map>
#include <cmath>

/**
 * \file
 * \ingroup core-examples
 * \ingroup randomvariable
 * Example program illustrating use of RandomVariableStream
 */

using namespace ns3;

namespace {

/**
 * Round a double number to the given precision.
 * For example, `dround(0.234, 0.1) = 0.2`
 * and `dround(0.257, 0.1) = 0.3`
 * \param [in] number The number to round.
 * \param [in] precision The least significant digit to keep in the rounding.
 * \returns \pname{number} rounded to \pname{precision}.
 */
double dround (double number, double precision)
{
  number /= precision;
  if (number >= 0)
    {
      number = std::floor (number + 0.5);
    }
  else
    {
      number = std::ceil (number - 0.5);
    }
  number *= precision;
  return number;
}

/**
 * Generate a histogram from a RandomVariableStream.
 * \param [in] rndvar The RandomVariableStream to sample.
 * \param [in] probes The number of samples.
 * \param [in] precision The precision to round samples to.
 * \param [in] title The title for the histogram.
 * \param [in] impulses Set the plot style to IMPULSES.
 * \return The histogram as a GnuPlot data set.
 */
static GnuplotDataset
Histogram (Ptr<RandomVariableStream> rndvar,
           unsigned int probes, double precision,
           const std::string& title, bool impulses = false)
{
  typedef std::map<double, unsigned int> histogram_maptype;
  histogram_maptype histogram;

  for (unsigned int i = 0; i < probes; ++i)
    {
      double val = dround ( rndvar->GetValue (), precision );

      ++histogram[val];
    }

  Gnuplot2dDataset data;
  data.SetTitle (title);

  if (impulses)
    {
      data.SetStyle (Gnuplot2dDataset::IMPULSES);
    }

  for (histogram_maptype::const_iterator hi = histogram.begin ();
       hi != histogram.end (); ++hi)
    {
      data.Add (hi->first, (double)hi->second / (double)probes / precision);
    }

  return data;
}

}  // unnamed namespace


int main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  unsigned int probes = 1000000;
  double precision = 0.01;

  GnuplotCollection gnuplots ("main-random-variables.pdf");
  gnuplots.SetTerminal ("pdf enhanced");

  {
    Gnuplot plot;
    plot.SetTitle ("UniformRandomVariable");
    plot.AppendExtra ("set yrange [0:]");

    Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
    x->SetAttribute ("Min", DoubleValue (0.0));
    x->SetAttribute ("Max", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x, probes, precision,
                                 "UniformRandomVariable [0.0 .. 1.0)") );
    plot.AddDataset ( Gnuplot2dFunction ("1.0",
                                         "0 <= x && x <= 1 ? 1.0 : 0") );

    gnuplots.AddPlot (plot);
  }

  {
    Gnuplot plot;
    plot.SetTitle ("ExponentialRandomVariable");
    plot.AppendExtra ("set xrange [0:8]");
    plot.AppendExtra ("ExpDist(x,l) = 1/l * exp(-1/l * x)");

    Ptr<ExponentialRandomVariable> x1 = CreateObject<ExponentialRandomVariable> ();
    x1->SetAttribute ("Mean", DoubleValue (0.5));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "ExponentialRandomVariable m=0.5") );

    plot.AddDataset ( Gnuplot2dFunction ("ExponentialDistribution mean 0.5",
                                         "ExpDist(x, 0.5)") );

    Ptr<ExponentialRandomVariable> x2 = CreateObject<ExponentialRandomVariable> ();
    x2->SetAttribute ("Mean", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "ExponentialRandomVariable m=1") );

    plot.AddDataset ( Gnuplot2dFunction ("ExponentialDistribution mean 1.0",
                                         "ExpDist(x, 1.0)") );

    Ptr<ExponentialRandomVariable> x3 = CreateObject<ExponentialRandomVariable> ();
    x3->SetAttribute ("Mean", DoubleValue (1.5));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "ExponentialRandomVariable m=1.5") );

    plot.AddDataset ( Gnuplot2dFunction ("ExponentialDistribution mean 1.5",
                                         "ExpDist(x, 1.5)") );

    gnuplots.AddPlot (plot);
  }

  {
    Gnuplot plot;
    plot.SetTitle ("ParetoRandomVariable");
    plot.AppendExtra ("set xrange [0:2]");

    Ptr<ParetoRandomVariable> x1 = CreateObject<ParetoRandomVariable> ();
    x1->SetAttribute ("Scale", DoubleValue (1.0));
    x1->SetAttribute ("Shape", DoubleValue (1.5));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "ParetoRandomVariable scale=1.0 shape=1.5") );

    Ptr<ParetoRandomVariable> x2 = CreateObject<ParetoRandomVariable> ();
    x2->SetAttribute ("Scale", DoubleValue (1.0));
    x2->SetAttribute ("Shape", DoubleValue (2.0));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "ParetoRandomVariable scale=1.0 shape=2.0") );

    Ptr<ParetoRandomVariable> x3 = CreateObject<ParetoRandomVariable> ();
    x3->SetAttribute ("Scale", DoubleValue (1.0));
    x3->SetAttribute ("Shape", DoubleValue (2.5));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "ParetoRandomVariable scale=1.0 shape=2.5") );

    gnuplots.AddPlot (plot);
  }

  {
    Gnuplot plot;
    plot.SetTitle ("WeibullRandomVariable");
    plot.AppendExtra ("set xrange [0:3]");

    Ptr<WeibullRandomVariable> x1 = CreateObject<WeibullRandomVariable> ();
    x1->SetAttribute ("Scale", DoubleValue (1.0));
    x1->SetAttribute ("Shape", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "WeibullRandomVariable scale=1.0 shape=1.0") );

    Ptr<WeibullRandomVariable> x2 = CreateObject<WeibullRandomVariable> ();
    x2->SetAttribute ("Scale", DoubleValue (1.0));
    x2->SetAttribute ("Shape", DoubleValue (2.0));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "WeibullRandomVariable scale=1.0 shape=2.0") );

    Ptr<WeibullRandomVariable> x3 = CreateObject<WeibullRandomVariable> ();
    x3->SetAttribute ("Scale", DoubleValue (1.0));
    x3->SetAttribute ("Shape", DoubleValue (3.0));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "WeibullRandomVariable scale=1.0 shape=3.0") );

    gnuplots.AddPlot (plot);
  }

  {
    Gnuplot plot;
    plot.SetTitle ("NormalRandomVariable");
    plot.AppendExtra ("set xrange [-3:3]");
    plot.AppendExtra ("NormalDist(x,m,s) = 1 / (s * sqrt(2*pi)) * exp(-1.0 / 2.0 * ((x-m) / s)**2)");

    Ptr<NormalRandomVariable> x1 = CreateObject<NormalRandomVariable> ();
    x1->SetAttribute ("Mean", DoubleValue (0.0));
    x1->SetAttribute ("Variance", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "NormalRandomVariable m=0.0 v=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("NormalDist {/Symbol m}=0.0 {/Symbol s}=1.0",
                                         "NormalDist(x,0.0,1.0)") );

    Ptr<NormalRandomVariable> x2 = CreateObject<NormalRandomVariable> ();
    x2->SetAttribute ("Mean", DoubleValue (0.0));
    x2->SetAttribute ("Variance", DoubleValue (2.0));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "NormalRandomVariable m=0.0 v=2.0") );

    plot.AddDataset ( Gnuplot2dFunction ("NormalDist {/Symbol m}=0.0 {/Symbol s}=sqrt(2.0)",
                                         "NormalDist(x,0.0,sqrt(2.0))") );

    Ptr<NormalRandomVariable> x3 = CreateObject<NormalRandomVariable> ();
    x3->SetAttribute ("Mean", DoubleValue (0.0));
    x3->SetAttribute ("Variance", DoubleValue (3.0));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "NormalRandomVariable m=0.0 v=3.0") );

    plot.AddDataset ( Gnuplot2dFunction ("NormalDist {/Symbol m}=0.0 {/Symbol s}=sqrt(3.0)",
                                         "NormalDist(x,0.0,sqrt(3.0))") );

    gnuplots.AddPlot (plot);
  }

  /** \todo Turn this plot back on once its distribution has been finished. */
  /*
  {
    Gnuplot plot;
    plot.SetTitle ("EmpiricalRandomVariable");
    plot.AppendExtra ("set xrange [*:*]");

    EmpiricalRandomVariable emp1;
    emp1.CDF (0.0,  0.0 / 15.0);
    emp1.CDF (0.2,  1.0 / 15.0);
    emp1.CDF (0.4,  3.0 / 15.0);
    emp1.CDF (0.6,  6.0 / 15.0);
    emp1.CDF (0.8, 10.0 / 15.0);
    emp1.CDF (1.0, 15.0 / 15.0);

    plot.AddDataset ( Histogram (emp1, probes, precision,
                                  "EmpiricalRandomVariable (Stairs)") );

    gnuplots.AddPlot (plot);
  }
  */

  /** \todo Turn this plot back on once its distribution has been finished. */
  /*
  {
    Gnuplot plot;
    plot.SetTitle ("DeterministicRandomVariable");
    plot.AppendExtra ("set xrange [*:*]");

    double values[] = { 0.0, 0.2, 0.2, 0.4, 0.2, 0.6, 0.8, 0.8, 1.0 };
    DeterministicRandomVariable det1 (values, sizeof(values) / sizeof(values[0]));

    plot.AddDataset ( Histogram (det1, probes, precision,
                                  "DeterministicRandomVariable", true) );

    gnuplots.AddPlot (plot);
  }
  */

  {
    Gnuplot plot;
    plot.SetTitle ("LogNormalRandomVariable");
    plot.AppendExtra ("set xrange [0:3]");

    plot.AppendExtra ("LogNormalDist(x,m,s) = 1.0/x * NormalDist(log(x), m, s)");

    Ptr<LogNormalRandomVariable> x1 = CreateObject<LogNormalRandomVariable> ();
    x1->SetAttribute ("Mu", DoubleValue (0.0));
    x1->SetAttribute ("Sigma", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "LogNormalRandomVariable m=0.0 s=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("LogNormalDist(x, 0.0, 1.0)",
                                         "LogNormalDist(x, 0.0, 1.0)") );

    Ptr<LogNormalRandomVariable> x2 = CreateObject<LogNormalRandomVariable> ();
    x2->SetAttribute ("Mu", DoubleValue (0.0));
    x2->SetAttribute ("Sigma", DoubleValue (0.5));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "LogNormalRandomVariable m=0.0 s=0.5") );

    Ptr<LogNormalRandomVariable> x3 = CreateObject<LogNormalRandomVariable> ();
    x3->SetAttribute ("Mu", DoubleValue (0.0));
    x3->SetAttribute ("Sigma", DoubleValue (0.25));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "LogNormalRandomVariable m=0.0 s=0.25") );

    plot.AddDataset ( Gnuplot2dFunction ("LogNormalDist(x, 0.0, 0.25)",
                                         "LogNormalDist(x, 0.0, 0.25)") );

    Ptr<LogNormalRandomVariable> x4 = CreateObject<LogNormalRandomVariable> ();
    x4->SetAttribute ("Mu", DoubleValue (0.0));
    x4->SetAttribute ("Sigma", DoubleValue (0.125));

    plot.AddDataset ( Histogram (x4, probes, precision,
                                 "LogNormalRandomVariable m=0.0 s=0.125") );

    Ptr<LogNormalRandomVariable> x5 = CreateObject<LogNormalRandomVariable> ();
    x5->SetAttribute ("Mu", DoubleValue (0.0));
    x5->SetAttribute ("Sigma", DoubleValue (2.0));

    plot.AddDataset ( Histogram (x5, probes, precision,
                                 "LogNormalRandomVariable m=0.0 s=2.0") );

    plot.AddDataset ( Gnuplot2dFunction ("LogNormalDist(x, 0.0, 2.0)",
                                         "LogNormalDist(x, 0.0, 2.0)") );

    Ptr<LogNormalRandomVariable> x6 = CreateObject<LogNormalRandomVariable> ();
    x6->SetAttribute ("Mu", DoubleValue (0.0));
    x6->SetAttribute ("Sigma", DoubleValue (2.5));

    plot.AddDataset ( Histogram (x6, probes, precision,
                                 "LogNormalRandomVariable m=0.0 s=2.5") );

    gnuplots.AddPlot (plot);
  }

  {
    Gnuplot plot;
    plot.SetTitle ("TriangularRandomVariable");
    plot.AppendExtra ("set xrange [*:*]");

    Ptr<TriangularRandomVariable> x1 = CreateObject<TriangularRandomVariable> ();
    x1->SetAttribute ("Min", DoubleValue (0.0));
    x1->SetAttribute ("Max", DoubleValue (1.0));
    x1->SetAttribute ("Mean", DoubleValue (0.5));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "TriangularRandomVariable [0.0 .. 1.0) m=0.5") );

    Ptr<TriangularRandomVariable> x2 = CreateObject<TriangularRandomVariable> ();
    x2->SetAttribute ("Min", DoubleValue (0.0));
    x2->SetAttribute ("Max", DoubleValue (1.0));
    x2->SetAttribute ("Mean", DoubleValue (0.4));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "TriangularRandomVariable [0.0 .. 1.0) m=0.4") );

    Ptr<TriangularRandomVariable> x3 = CreateObject<TriangularRandomVariable> ();
    x3->SetAttribute ("Min", DoubleValue (0.0));
    x3->SetAttribute ("Max", DoubleValue (1.0));
    x3->SetAttribute ("Mean", DoubleValue (0.65));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "TriangularRandomVariable [0.0 .. 1.0) m=0.65") );

    gnuplots.AddPlot (plot);
  }

  {
    Gnuplot plot;
    plot.SetTitle ("GammaRandomVariable");
    plot.AppendExtra ("set xrange [0:10]");
    plot.AppendExtra ("set yrange [0:1]");
    plot.AppendExtra ("GammaDist(x,a,b) = x**(a-1) * 1/b**a * exp(-x/b) / gamma(a)");

    plot.AppendExtra ("set label 1 '{/Symbol g}(x,{/Symbol a},{/Symbol b}) = x^{/Symbol a-1} e^{-x {/Symbol b}^{-1}} ( {/Symbol b}^{/Symbol a} {/Symbol G}({/Symbol a}) )^{-1}' at 0.7, 0.9");

    Ptr<GammaRandomVariable> x1 = CreateObject<GammaRandomVariable> ();
    x1->SetAttribute ("Alpha", DoubleValue (1.0));
    x1->SetAttribute ("Beta", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "GammaRandomVariable a=1.0 b=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("{/Symbol g}(x, 1.0, 1.0)",
                                         "GammaDist(x, 1.0, 1.0)") );

    Ptr<GammaRandomVariable> x2 = CreateObject<GammaRandomVariable> ();
    x2->SetAttribute ("Alpha", DoubleValue (1.5));
    x2->SetAttribute ("Beta", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "GammaRandomVariable a=1.5 b=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("{/Symbol g}(x, 1.5, 1.0)",
                                         "GammaDist(x, 1.5, 1.0)") );

    Ptr<GammaRandomVariable> x3 = CreateObject<GammaRandomVariable> ();
    x3->SetAttribute ("Alpha", DoubleValue (2.0));
    x3->SetAttribute ("Beta", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "GammaRandomVariable a=2.0 b=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("{/Symbol g}(x, 2.0, 1.0)",
                                         "GammaDist(x, 2.0, 1.0)") );

    Ptr<GammaRandomVariable> x4 = CreateObject<GammaRandomVariable> ();
    x4->SetAttribute ("Alpha", DoubleValue (4.0));
    x4->SetAttribute ("Beta", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x4, probes, precision,
                                 "GammaRandomVariable a=4.0 b=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("{/Symbol g}(x, 4.0, 1.0)",
                                         "GammaDist(x, 4.0, 1.0)") );

    Ptr<GammaRandomVariable> x5 = CreateObject<GammaRandomVariable> ();
    x5->SetAttribute ("Alpha", DoubleValue (2.0));
    x5->SetAttribute ("Beta", DoubleValue (2.0));

    plot.AddDataset ( Histogram (x5, probes, precision,
                                 "GammaRandomVariable a=2.0 b=2.0") );

    plot.AddDataset ( Gnuplot2dFunction ("{/Symbol g}(x, 2.0, 2.0)",
                                         "GammaDist(x, 2.0, 2.0)") );

    Ptr<GammaRandomVariable> x6 = CreateObject<GammaRandomVariable> ();
    x6->SetAttribute ("Alpha", DoubleValue (2.5));
    x6->SetAttribute ("Beta", DoubleValue (3.0));

    plot.AddDataset ( Histogram (x6, probes, precision,
                                 "GammaRandomVariable a=2.5 b=3.0") );

    plot.AddDataset ( Gnuplot2dFunction ("{/Symbol g}(x, 2.5, 3.0)",
                                         "GammaDist(x, 2.5, 3.0)") );

    Ptr<GammaRandomVariable> x7 = CreateObject<GammaRandomVariable> ();
    x7->SetAttribute ("Alpha", DoubleValue (2.5));
    x7->SetAttribute ("Beta", DoubleValue (4.5));

    plot.AddDataset ( Histogram (x7, probes, precision,
                                 "GammaRandomVariable a=2.5 b=4.5") );

    plot.AddDataset ( Gnuplot2dFunction ("{/Symbol g}(x, 2.5, 4.5)",
                                         "GammaDist(x, 2.5, 4.5)") );

    gnuplots.AddPlot (plot);
  }

  {
    Gnuplot plot;
    plot.SetTitle ("ErlangRandomVariable");
    plot.AppendExtra ("set xrange [0:10]");
    plot.AppendExtra ("ErlangDist(x,k,l) = x**(k-1) * 1/l**k * exp(-x/l) / (k-1)!");

    plot.AppendExtra ("set label 1 'Erlang(x,k,{/Symbol l}) = x^{k-1} e^{-x {/Symbol l}^{-1}} ( {/Symbol l}^k (k-1)! )^{-1}' at 0.7, 0.9");

    Ptr<ErlangRandomVariable> x1 = CreateObject<ErlangRandomVariable> ();
    x1->SetAttribute ("K", IntegerValue (1));
    x1->SetAttribute ("Lambda", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x1, probes, precision,
                                 "ErlangRandomVariable k=1 {/Symbol l}=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("Erlang(x, 1, 1.0)",
                                         "ErlangDist(x, 1, 1.0)") );

    Ptr<ErlangRandomVariable> x2 = CreateObject<ErlangRandomVariable> ();
    x2->SetAttribute ("K", IntegerValue (2));
    x2->SetAttribute ("Lambda", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x2, probes, precision,
                                 "ErlangRandomVariable k=2 {/Symbol l}=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("Erlang(x, 2, 1.0)",
                                         "ErlangDist(x, 2, 1.0)") );

    Ptr<ErlangRandomVariable> x3 = CreateObject<ErlangRandomVariable> ();
    x3->SetAttribute ("K", IntegerValue (3));
    x3->SetAttribute ("Lambda", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x3, probes, precision,
                                 "ErlangRandomVariable k=3 {/Symbol l}=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("Erlang(x, 3, 1.0)",
                                         "ErlangDist(x, 3, 1.0)") );

    Ptr<ErlangRandomVariable> x4 = CreateObject<ErlangRandomVariable> ();
    x4->SetAttribute ("K", IntegerValue (5));
    x4->SetAttribute ("Lambda", DoubleValue (1.0));

    plot.AddDataset ( Histogram (x4, probes, precision,
                                 "ErlangRandomVariable k=5 {/Symbol l}=1.0") );

    plot.AddDataset ( Gnuplot2dFunction ("Erlang(x, 5, 1.0)",
                                         "ErlangDist(x, 5, 1.0)") );

    Ptr<ErlangRandomVariable> x5 = CreateObject<ErlangRandomVariable> ();
    x5->SetAttribute ("K", IntegerValue (2));
    x5->SetAttribute ("Lambda", DoubleValue (2.0));

    plot.AddDataset ( Histogram (x5, probes, precision,
                                 "ErlangRandomVariable k=2 {/Symbol l}=2.0") );

    plot.AddDataset ( Gnuplot2dFunction ("Erlang(x, 2, 2.0)",
                                         "ErlangDist(x, 2, 2.0)") );

    Ptr<ErlangRandomVariable> x6 = CreateObject<ErlangRandomVariable> ();
    x6->SetAttribute ("K", IntegerValue (2));
    x6->SetAttribute ("Lambda", DoubleValue (3.0));

    plot.AddDataset ( Histogram (x6, probes, precision,
                                 "ErlangRandomVariable k=2 {/Symbol l}=3.0") );

    plot.AddDataset ( Gnuplot2dFunction ("Erlang(x, 2, 3.0)",
                                         "ErlangDist(x, 2, 3.0)") );

    Ptr<ErlangRandomVariable> x7 = CreateObject<ErlangRandomVariable> ();
    x7->SetAttribute ("K", IntegerValue (2));
    x7->SetAttribute ("Lambda", DoubleValue (5.0));

    plot.AddDataset ( Histogram (x7, probes, precision,
                                 "ErlangRandomVariable k=2 {/Symbol l}=5.0") );

    plot.AddDataset ( Gnuplot2dFunction ("Erlang(x, 2, 5.0)",
                                         "ErlangDist(x, 2, 5.0)") );

    gnuplots.AddPlot (plot);
  }

  gnuplots.GenerateOutput (std::cout);

  return 0;
}
