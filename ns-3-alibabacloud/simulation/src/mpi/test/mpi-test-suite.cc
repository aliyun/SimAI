/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Lawrence Livermore National Laboratory
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

#include "ns3/example-as-test.h"

#include <sstream>

using namespace ns3;

/**
 * \ingroup mpi-tests
 * 
 * This version of ns3::ExampleTestCase is specialized for MPI
 * by accepting the number of ranks as a parameter,
 * then building a `--command-template` string which
 * invokes `mpiexec` correctly to execute MPI examples.
 */
class MpiTestCase : public ExampleAsTestCase
{
public:
  /**
   * \copydoc ns3::ExampleAsTestCase::ExampleAsTestCase
   *
   * \param [in] ranks The number of ranks to use
   */
  MpiTestCase (const std::string name,
               const std::string program,
               const std::string dataDir,
               const int ranks,
               const std::string args = "");

  /** Destructor */
  virtual ~MpiTestCase (void) {};

  /**
   * Produce the `--command-template` argument which will invoke
   * `mpiexec` with the requested number of ranks.
   *
   * \returns The `--command-template` string.
   */
  std::string GetCommandTemplate (void) const;

  /**
   * Sort the output from parallel execution.
   * stdout from multiple ranks is not ordered.
   *
   * \returns Sort command
   */
  std::string
  GetPostProcessingCommand (void) const;
  
private:
  /** The number of ranks. */
  int m_ranks;            
};

MpiTestCase::MpiTestCase (const std::string name,
                          const std::string program,
                          const std::string dataDir,
                          const int ranks,
                          const std::string args /* = "" */)
  : ExampleAsTestCase (name, program, dataDir, args),
    m_ranks (ranks)
{
}

std::string
MpiTestCase::GetCommandTemplate (void) const
{
  std::stringstream ss;
  ss << "mpiexec -n " << m_ranks  << " %s --test " << m_args;
  return ss.str ();
}

std::string
MpiTestCase::GetPostProcessingCommand (void) const
{
  std::string command ("| grep TEST | sort ");
  return command;
}

/**
 * \ingroup mpi-tests
 * MPI specialization of ns3::ExampleTestSuite.
 */
class MpiTestSuite : public TestSuite
{
public:
  /**
   * \copydoc MpiTestCase::MpiTestCase
   *
   * \param [in] duration Amount of time this test takes to execute
   *             (defaults to QUICK).
   */
  MpiTestSuite (const std::string name,
                const std::string program,
                const std::string dataDir,
                const int ranks,
                const std::string args = "",
                    const TestDuration duration=QUICK)
    : TestSuite (name, EXAMPLE)
  {
    AddTestCase (new MpiTestCase (name, program, dataDir, ranks, args), duration);
  }

};  // class MpiTestSuite

/* Tests using SimpleDistributedSimulatorImpl */
static MpiTestSuite g_mpiNms2      ("mpi-example-nms-2",       "nms-p2p-nix-distributed", NS_TEST_SOURCEDIR, 2);
static MpiTestSuite g_mpiComm2     ("mpi-example-comm-2",      "simple-distributed-mpi-comm", NS_TEST_SOURCEDIR, 2);
static MpiTestSuite g_mpiComm2comm ("mpi-example-comm-2-init", "simple-distributed-mpi-comm", NS_TEST_SOURCEDIR, 2, "--init");
static MpiTestSuite g_mpiComm3comm ("mpi-example-comm-3-init", "simple-distributed-mpi-comm", NS_TEST_SOURCEDIR, 3, "--init");
static MpiTestSuite g_mpiEmpty2    ("mpi-example-empty-2",     "simple-distributed-empty-node", NS_TEST_SOURCEDIR, 2);
static MpiTestSuite g_mpiEmpty3    ("mpi-example-empty-3",     "simple-distributed-empty-node", NS_TEST_SOURCEDIR, 3);
static MpiTestSuite g_mpiSimple2   ("mpi-example-simple-2",    "simple-distributed", NS_TEST_SOURCEDIR, 2);
static MpiTestSuite g_mpiThird2    ("mpi-example-third-2",     "third-distributed", NS_TEST_SOURCEDIR, 2);

/* Tests using NullMessageSimulatorImpl */
static MpiTestSuite g_mpiSimple2NullMsg ("mpi-example-simple-2-nullmsg",    "simple-distributed", NS_TEST_SOURCEDIR, 2, "--nullmsg");
static MpiTestSuite g_mpiEmpty2NullMsg  ("mpi-example-empty-2-nullmsg",     "simple-distributed-empty-node", NS_TEST_SOURCEDIR, 2, "-nullmsg");
static MpiTestSuite g_mpiEmpty3NullMsg  ("mpi-example-empty-3-nullmsg",     "simple-distributed-empty-node", NS_TEST_SOURCEDIR, 3, "-nullmsg");

