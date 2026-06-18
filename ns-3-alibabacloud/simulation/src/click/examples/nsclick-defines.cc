/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
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
 * Authors: Sascha Jopen <jopen@cs.uni-bonn.de>
 */


#include "ns3/core-module.h"
#include "ns3/ipv4-click-routing.h"
#include "ns3/click-internet-stack-helper.h"
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NsclickRouting");

int
main (int argc, char *argv[])
{
#ifdef NS3_CLICK
  std::string clickConfigFolder = "src/click/examples";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("clickConfigFolder", "Base folder for click configuration files", clickConfigFolder);
  cmd.Parse (argc, argv);

//
// Explicitly create the nodes required by the topology (shown above).
//
  NS_LOG_INFO ("Create a node.");
  NodeContainer n;
  n.Create (1);

//
// Install Click on the nodes
//
  std::map<std::string, std::string> defines;
// Strings, especially with blanks in it, have to be enclosed in quotation
// marks, like in click configuration files.
  defines["OUTPUT"] = "\"Hello World!\"";
  
  ClickInternetStackHelper clickinternet;
  clickinternet.SetClickFile (n, clickConfigFolder + "/nsclick-defines.click");
  clickinternet.SetRoutingTableElement (n, "rt");
  clickinternet.SetDefines(n, defines);
  clickinternet.Install (n);

//
// Now, do the actual simulation.
//
  NS_LOG_INFO ("Run Simulation.");
  Simulator::Stop (Seconds (20.0));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
#else
  NS_FATAL_ERROR ("Can't use ns-3-click without NSCLICK compiled in");
#endif
}
