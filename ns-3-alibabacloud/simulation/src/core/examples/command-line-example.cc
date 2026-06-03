/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Lawrence Livermore National Laboratory
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

#include <iostream>
#include <iomanip>
#include <string>

#include "ns3/core-module.h"

/**
 * \file
 * \ingroup core-examples
 * \ingroup commandline
 * Example program illustrating use of ns3::CommandLine.
 */

using namespace ns3;

namespace {

/**
 * Global variable to illustrate command line arguments handled by a
 * Callback function.
 */
std::string g_cbArg = "cbArg default";

/**
 * Function to illustrate command line arguments handled by a
 * Callback function.
 *
 * \param [in] val New value for \pname{g_cbArg}.
 * \returns \c true.
 */
bool SetCbArg (std::string val)
{
  g_cbArg = val;
  return true;
}

}  // unnamed namespace


int main (int argc, char *argv[])
{

  int         intArg  = 1;
  bool        boolArg = false;
  std::string strArg  = "strArg default";
  // Attribute path
  const std::string attrClass = "ns3::RandomVariableStream";
  const std::string attrName  = "Antithetic";
  const std::string attrPath  = attrClass + "::" + attrName;
  // Non-option arguments
  int         nonOpt1 = 1;
  int         nonOpt2 = 1;

  // Cache the initial values.  Normally one wouldn't do this,
  // but we want to demonstrate that CommandLine has changed them.
  const int intDef = intArg;
  const bool boolDef = boolArg;
  const std::string strDef = strArg;
  const std::string cbDef  = g_cbArg;
  // Look up default value for attribute
  const TypeId tid = TypeId::LookupByName (attrClass);
  std::string attrDef;
  {
    struct TypeId::AttributeInformation info;
    tid.LookupAttributeByName (attrName, &info);
    attrDef = info.originalInitialValue->SerializeToString (info.checker);
  }
  const int nonOpt1Def = nonOpt1;
  const int nonOpt2Def = nonOpt2;


  CommandLine cmd (__FILE__);
  cmd.Usage ("CommandLine example program.\n"
             "\n"
             "This little program demonstrates how to use CommandLine.");
  cmd.AddValue ("intArg",  "an int argument",       intArg);
  cmd.AddValue ("boolArg", "a bool argument",       boolArg);
  cmd.AddValue ("strArg",  "a string argument",     strArg);
  cmd.AddValue ("anti",    attrPath);
  cmd.AddValue ("cbArg",   "a string via callback", MakeCallback (SetCbArg));
  cmd.AddNonOption ("nonOpt1", "first non-option",  nonOpt1);
  cmd.AddNonOption ("nonOpt2", "second non-option", nonOpt2);
  cmd.Parse (argc, argv);

  // Show initial values:
  std::cout << std::endl;
  std::cout << cmd.GetName () << ":" << std::endl;

  //Print the source version used to build this example
  std::cout << "Program Version: ";
  cmd.PrintVersion (std::cout);
  std::cout << std::endl;

  std::cout << "Initial values:" << std::endl;

  std::cout << std::left << std::setw (10) << "intArg:"
            <<                   intDef
            << std::endl;
  std::cout << std::setw (10)              << "boolArg:"
            << std::boolalpha << boolDef   << std::noboolalpha
            << std::endl;

  std::cout << std::setw (10)              << "strArg:"
            << "\""           << strDef    << "\""
            << std::endl;
  std::cout << std::setw (10)              << "anti:"
            << "\""           << attrDef   << "\""
            << std::endl;
  std::cout << std::setw (10)              << "cbArg:"
            << "\""           << cbDef     << "\""
            << std::endl;
  std::cout << std::left << std::setw (10) << "nonOpt1:"
            <<                   nonOpt1Def
            << std::endl;
  std::cout << std::left << std::setw (10) << "nonOpt2:"
            <<                   nonOpt2Def
            << std::endl;
  std::cout << std::endl;


  // Show final values
  std::cout << "Final values:" << std::endl;
  std::cout << std::left << std::setw (10) << "intArg:"
            <<                   intArg
            << std::endl;
  std::cout << std::setw (10)              << "boolArg:"
            << std::boolalpha << boolArg   << std::noboolalpha
            << std::endl;

  std::cout << std::setw (10)              << "strArg:"
            << "\""           << strArg    << "\""
            << std::endl;

  // Look up new default value for attribute
  {
    struct TypeId::AttributeInformation info;
    tid.LookupAttributeByName (attrName, &info);

    std::cout << std::setw (10)            << "anti:"
              << "\""
              << info.initialValue->SerializeToString (info.checker)
              << "\""
              << std::endl;
  }
  std::cout << std::setw (10)              << "cbArg:"
            << "\""           << g_cbArg   << "\""
            << std::endl;
  std::cout << std::left << std::setw (10) << "nonOpt1:"
            <<                   nonOpt1
            << std::endl;
  std::cout << std::left << std::setw (10) << "nonOpt2:"
            <<                   nonOpt2
            << std::endl;
  std::cout << std::left << "Number of extra non-option arguments:"
            <<                   cmd.GetNExtraNonOptions ()
            << std::endl;

  for (std::size_t i = 0; i < cmd.GetNExtraNonOptions (); ++i)
    {
      std::cout << std::left << std::setw (10) << "extra:"
                << "\""       << cmd.GetExtraNonOption (i) << "\""
                << std::endl;
    }


  return 0;
}
