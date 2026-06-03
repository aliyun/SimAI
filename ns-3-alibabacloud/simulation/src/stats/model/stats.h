/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#ifndef STATS_H
#define STATS_H

// Header file used only to define the stats group in Doxygen

/**
 * \defgroup stats Statistics
 *
 * The statistics module includes some useful features to ease data collection from experiments.
 * In particular the following features are included:
 * <ul>
 * <li> The core framework and two basic data collectors: A counter, and a min/max/avg/total observer.</li>
 * <li> Extensions of those to easily work with times and packets.</li>
 * <li> Plaintext output formatted for OMNet++.</li>
 * <li> Database output using SQLite, a standalone, lightweight, high performance SQL engine.</li>
 * <li> Mandatory and open ended metadata for describing and working with runs.</li>
 * </ul>
 *
 * See the manual for a complete documentation.
 */

/**
 * \ingroup stats
 * \defgroup aggregator Data Aggregators
 *
 * Data aggregators are classes used to collect data and produce output
 * specialized for various purpose, e.g., Gnuplot, file output, etc.
 */

/**
 * \ingroup stats
 * \defgroup probes Probes
 *
 * Probes are used to probe an underlying ns3 TraceSource exporting
 * its value.  This probe usually exports a trace source "Output".
 * The Output trace source emits a value when either the trace source
 * emits a new value, or when SetValue () is called.
 *
 * Probes are a special kind of Trace Source.
 */

/**
 * \ingroup stats
 * \defgroup gnuplot Gnuplot
 *
 * Classes in Gnuplot group are used to collect and prepare and output data
 * for subsequent processing by Gnuplot.
 */

/**
 * \ingroup stats
 * \defgroup dataoutput Data Output
 *
 * Classes in Data Output group are used to collect and prepare and output data
 * for subsequent output in a specific format, e.g., Omnet++, SQLite, etc.
 */

/**
 * \ingroup stats
 * \ingroup tests
 * \defgroup stats-tests Statistics module tests
 */

#endif /* STATS_H */
