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

#include "ns3/command-line.h"
#include "ns3/version.h"

#include <iostream>
#include <iomanip>
#include <string>

/**
 * \file
 * \ingroup core-examples
 * Example program illustrating use of ns3::Version.
 */

using namespace ns3;

int main (int argc, char *argv[])
{

  CommandLine cmd (__FILE__);
  cmd.Usage ("Version class example program.\n"
             "\n"
             "This program demonstrates the various outputs from the Version class");
  cmd.Parse (argc, argv);

  std::cout << std::endl;
  std::cout << cmd.GetName () << ":" << std::endl;

  //Print the source version used to build this example
  std::cout << "Program Version (according to CommandLine): ";
  cmd.PrintVersion (std::cout);
  std::cout << std::endl;

  Version version;
  std::cout << "Version fields:\n"
            << "LongVersion:        " << version.LongVersion () << "\n"
            << "ShortVersion:       " << version.ShortVersion () << "\n"
            << "BuildSummary:       " << version.BuildSummary () << "\n"
            << "VersionTag:         " << version.VersionTag () << "\n"
            << "Major:              " << version.Major () << "\n"
            << "Minor:              " << version.Minor () << "\n"
            << "Patch:              " << version.Patch () << "\n"
            << "ReleaseCandidate:   " << version.ReleaseCandidate () << "\n"
            << "ClosestAncestorTag: " << version.ClosestAncestorTag () << "\n"
            << "TagDistance:        " << version.TagDistance () << "\n"
            << "CommitHash:         " << version.CommitHash () << "\n"
            << "BuildProfile:       " << version.BuildProfile () << "\n"
            << "WorkingTree:        " << (version.DirtyWorkingTree () ? "dirty" : "clean")
            << std::endl;

  return 0;
}
