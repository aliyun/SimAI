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

#include <iostream>
#include <iomanip>
#include <string>

#include "ns3/core-module.h"

/**
 * \file
 * \ingroup core-examples
 * \ingroup systempath
 * Example program illustrating use of ns3::SystemPath
 */

using namespace ns3;
using namespace ns3::SystemPath;

int main (int argc, char *argv[])
{
  std::string path = "/usr/share/dict/";

  CommandLine cmd;
  cmd.Usage ("SystemPath examples.\n");

  cmd.AddValue ("path", "Path to demonstrate SystemPath functions.", path);
  cmd.Parse (argc, argv);

  // Show initial values:
  std::cout << std::endl;
  std::cout << cmd.GetName () << ":" << std::endl;

  std::cout << "FindSelfDirectory:   " << FindSelfDirectory () << std::endl;

  std::cout << "Demonstration path:  " << path << std::endl;
  std::cout << "Exists?              "
            << (Exists (path) ? "yes" : "no") << std::endl;

  auto foo = Append (path, "foo");
  std::cout << "Append 'foo':        " << foo << std::endl;
  std::cout << "Exists?              "
            << (Exists (foo) ? "yes" : "no") << std::endl;

  std::cout << "Split path:\n";
  auto items = Split (path);
  for (auto item : items)
    {
      std::cout << "    '" << item << "'\n";
    }
  std::cout << std::endl;

  std::cout << "Successive Joins: \n";
  for (auto it = items.begin (); it != items.end (); ++it)
    {
      auto partial = Join (items.begin (), it);
      std::cout << "    '" << partial << "'\n";
    }

  std::cout << "Files in the directory: \n";
  auto files = ReadFiles (path);
  for (auto item : files)
    {
      std::cout << "    '" << item << "'\n";
    }


  return 0;
}
