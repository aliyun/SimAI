/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 LLNL
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#ifndef DESMETRICS_H
#define DESMETRICS_H

/**
 * @file
 * @ingroup simulator
 * ns3::DesMetrics declaration.
 */

#include "nstime.h"
#include "singleton.h"

#include <stdint.h>    // uint32_t
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace ns3 {

/**
 * @ingroup simulator
 *
 * @brief Event trace data collector for the DES Metrics project.
 *
 * This feature generates a JSON file with event trace data,
 * including the source and destination context for each
 * event, and the (virtual) times when the event was scheduled and
 * when it will execute.
 *
 * See the DES Metrics Project page: https://wilseypa.github.io/desMetrics/
 * for more information and analysis tools.
 *
 * If enabled (see below), ns-3 scripts should use CommandLine to
 * parse arguments, which will open the JSON file with the same name
 * as the script, and write the JSON header.  Failure to use CommandLine when
 * DES Metrics is enabled will put the trace data in the file
 * \c desTraceFile.json instead. All examples accessible from \c test.py
 * use CommandLine, and so generate JSON files.
 *
 * Output from scripts ends up in the current working directory (normally the
 * top level directory). When \c test.py is used to run tests or examples
 * the trace files are generated in a time-stamped subdirectory of
 * \c testpy-output/,  which \c test.py normally deletes.
 * To keep the output of examples, use the \c --retain argument to \c test.py.
 *
 * The output file has the following form:
 * \verbatim
{
 "simulator_name" : "ns-3",
 "model_name" : "ipv4-raw",
 "capture_date" : "Fri May 27 00:34:27 2016",
 "command_line_arguments" : "ipv4-raw [ns3-dev-test-runner-debug] --test-name=ipv4-raw --stop-on-failure --fullness=QUICK --xml --tempdir=testpy-output/2016-05-27-04-33-35-CUT --out=testpy-output/2016-05-27-04-33-35-CUT/ipv4-raw.xml",
 "events" : [
  ["0",0,"0",0],
  ["1",0,"0",0],
  ["0",0,"0",0],
  ...
  ["0",0,"0",0]
 ]
} \endverbatim
 * The first few fields are self-explanatory. The \c event record consists of
 * the source context, the event send time, the destination context,
 * and the event execution time.  Times are given in the
 * current Time resolution.
 *
 * <b> Enabling DES Metrics </b>
 *
 * Enable DES Metrics at configure time with
 * \verbatim
   $ ns3 configure ... --enable-des-metrics \endverbatim
 *
 * <b> Working with DES Metrics </b>
 *
 * Some useful shell pipelines:
 *
 * \li Run everything, retaining the results directory: <br/>
 *   \code  ./test.py --nobuild --retain \endcode
 * \li Example traces end up in \c testpy-output/, so move there: <br/>
 *   \code cd testpy-output/$(date +"%F")*_/  \endcode
 *   (Remove the `_', which is to work around a Doxygen limitation.)
 * \li Remove the traces with less than 10 events: <br/>
 *   \code wc -l *.json | sort -nr | grep "^ *[789] " | cut -d ' ' -f 9 | xargs rm -f \endcode
 * \li Show the largest file, and total number of trace files: <br/>
 *   \code wc -l *.json | sort -n | tail -2 \endcode
 *
 */
class DesMetrics : public Singleton<DesMetrics>
{
public:

  /**
   * Open the DesMetrics trace file and print the header.
   *
   * The trace file will have the same base name as the main program,
   * '.json' as the extension.
   *
   * \param args [in] Command line arguments.
   * \param outDir [in] Directory where the trace file should be written.
   */
  void Initialize (std::vector<std::string> args, std::string outDir = "");

  /**
   * Trace an event to self at the time it is scheduled.
   *
   * \param now [in] The local simulation time.
   * \param delay [in] The delay to the event.
   */
  void Trace (const Time & now, const Time & delay);

  /**
   * Trace an event (with context) at the time it is scheduled.
   *
   * \param context [in] The context (NodeId) which will receive the event.
   * \param now [in] The local simulation time.
   * \param delay [in] The delay to the event.
   */
  void TraceWithContext (uint32_t context,  const Time & now, const Time & delay);

  /**
   * Destructor, closes the trace file.
   */
  ~DesMetrics (void);

private:

  /** Close the output file. */
  void Close (void);

  /**
   * Cache the last-used output directory.
   *
   * This is enables repeated/re-entrant use of CommandLine, for example
   * in \c command-line-test-suite.cc
   */
  static std::string m_outputDir;

  bool m_initialized;    //!< Have we been initialized.
  std::ofstream m_os;    //!< The output JSON trace file stream.
  char m_separator;      //!< The separator between event records.

  /** Mutex to control access to the output file. */
  std::mutex m_mutex;

};  // class DesMetrics


} // namespace ns3

#endif /* DESMETRICS_H */
