/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Lawrence Livermore National Laboratory
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

#include "example-as-test.h"
#include "ascii-test.h"
#include "log.h"
#include "assert.h"

#include <string>
#include <sstream>
#include <cstdlib>  // itoa(), system (), getenv ()
#include <cstring>

/**
 * \file
 * \ingroup testing
 * Implementation of classes ns3::ExampleAsTestSuite and ns3::ExampleTestCase.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ExampleAsTestCase");

// Running tests as examples currently requires bash shell; uses Unix
// piping that does not work on Windows.
#if defined(NS3_ENABLE_EXAMPLES) && !defined (__win32__)

ExampleAsTestCase::ExampleAsTestCase (const std::string name,
                                      const std::string program,
                                      const std::string dataDir,
                                      const std::string args /* = "" */)
  : TestCase (name),
    m_program (program),
    m_dataDir (dataDir),
    m_args (args)
{
  NS_LOG_FUNCTION (this << name << program << dataDir << args);
}

ExampleAsTestCase::~ExampleAsTestCase (void)
{
  NS_LOG_FUNCTION_NOARGS ();
}

std::string
ExampleAsTestCase::GetCommandTemplate (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  std::string command ("%s ");
  command += m_args;
  return command;
}

std::string
ExampleAsTestCase::GetPostProcessingCommand (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  std::string command ("");
  return command;
}

void
ExampleAsTestCase::DoRun (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  // Set up the output file names
  SetDataDir (m_dataDir);
  std::string refFile = CreateDataDirFilename (GetName () + ".reflog");
  std::string testFile = CreateTempDirFilename  (GetName () + ".reflog");

  std::stringstream ss;

  // Use bash as shell to allow use of PIPESTATUS
  ss << "bash -c './ns3 run " << m_program
     << " --no-build --command-template=\"" << GetCommandTemplate () << "\""

    // redirect std::clog, std::cerr to std::cout
     << " 2>&1 "

    // Suppress the waf lines from output; waf output contains directory paths which will
    // obviously differ during a test run
     << " | grep -v 'Waf:' "
     << GetPostProcessingCommand ()
     << " > " << testFile

    // Get the status of ns3
     << "; exit ${PIPESTATUS[0]}'";

  int status = std::system (ss.str ().c_str ());

  std::cout << "command:  " << ss.str () << "\n"
            << "status:   " << status << "\n"
            << "refFile:  " << refFile   << "\n"
            << "testFile: " << testFile  << "\n"
            << std::endl;
  std::cout << "testFile contents:" << std::endl;

  std::ifstream logF (testFile);
  std::string line;
  while (getline (logF, line))
    {
      std::cout << line << "\n";
    }
  logF.close ();

  // Make sure the example didn't outright crash
  NS_TEST_ASSERT_MSG_EQ (status, 0, "example " + m_program + " failed");

  // Check that we're not just introspecting the command-line
  const char * envVar = std::getenv ("NS_COMMANDLINE_INTROSPECTION");
  if (envVar != 0 && std::strlen (envVar) != 0)
    {
      return;
    }

  // Compare the testFile to the reference file
  NS_ASCII_TEST_EXPECT_EQ (testFile, refFile);
}

ExampleAsTestSuite::ExampleAsTestSuite (const std::string name,
                                        const std::string program,
                                        const std::string dataDir,
                                        const std::string args /* = "" */,
                                        const TestDuration duration /* =QUICK */)
  : TestSuite (name, EXAMPLE)
{
  NS_LOG_FUNCTION (this << name << program << dataDir << args << duration);
  AddTestCase (new ExampleAsTestCase (name, program, dataDir, args), duration);
}

#endif // NS3_ENABLE_EXAMPLES && !defined (__win32__)

}  // namespace ns3
