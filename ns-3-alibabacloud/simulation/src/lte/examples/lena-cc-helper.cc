/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it>
 */


#include "ns3/core-module.h"
#include "ns3/config-store.h"
#include <ns3/buildings-helper.h>
#include "ns3/component-carrier.h"
#include "ns3/cc-helper.h"

using namespace ns3;


void Print ( ComponentCarrier cc);

int main (int argc, char *argv[])
{	
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
	
  Config::SetDefault ("ns3::ComponentCarrier::UlBandwidth", UintegerValue (50));
  Config::SetDefault ("ns3::ComponentCarrier::PrimaryCarrier", BooleanValue (true));

  // Parse again so you can override default values from the command line
  cmd.Parse (argc, argv);

  Ptr<CcHelper> cch = CreateObject<CcHelper> ();
  cch->SetNumberOfComponentCarriers(2);

  std::map< uint8_t, ComponentCarrier > ccm = cch->EquallySpacedCcs ();
 
  std::map< uint8_t, ComponentCarrier >::iterator it;
  
  std::cout << " CcMap size " << ccm.size () << std::endl;
  for (it = ccm.begin (); it !=ccm.end (); it++)
    {
      Print (it->second);
    }
 

  Simulator::Stop (Seconds (1.05));

  Simulator::Run ();

  // GtkConfigStore config;
  // config.ConfigureAttributes ();

  Simulator::Destroy ();
  return 0;
}

void Print ( ComponentCarrier cc)
{
  std::cout << " UlBandwidth " << uint16_t (cc.GetUlBandwidth ())
            << " DlBandwidth " << uint16_t (cc.GetDlBandwidth ())
            << " Dl Earfcn " << cc.GetDlEarfcn ()
            << " Ul Earfcn " << cc.GetUlEarfcn ()
            << " - Is this the Primary Channel? " << cc.IsPrimary () 
            << std::endl;
}
