/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Jaume Nin <jnin@cttc.es>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/config-store.h"
#include <ns3/buildings-propagation-loss-model.h>
#include <ns3/buildings-helper.h>
#include <ns3/radio-environment-map-helper.h>
#include <iomanip>
#include <string>
#include <vector>
//#include "ns3/gtk-config-store.h"

using namespace ns3;
using std::vector;

int
main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults ();

  cmd.Parse (argc, argv);

  // Geometry of the scenario (in meters)
  // Assume squared building
  double nodeHeight = 1.5;
  double roomHeight = 3;
  double roomLength = 500;
  uint32_t nRooms = 2;
  // Create one eNodeB per room + one 3 sector eNodeB (i.e. 3 eNodeB) + one regular eNodeB
  uint32_t nEnb = nRooms*nRooms + 4;
  uint32_t nUe = 1;

  Ptr < LteHelper > lteHelper = CreateObject<LteHelper> ();
  //lteHelper->EnableLogComponents ();
  lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisPropagationLossModel"));

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  NodeContainer oneSectorNodes;
  NodeContainer threeSectorNodes;
  vector < NodeContainer > ueNodes;

  oneSectorNodes.Create (nEnb-3);
  threeSectorNodes.Create (3);

  enbNodes.Add (oneSectorNodes);
  enbNodes.Add (threeSectorNodes);

  for (uint32_t i = 0; i < nEnb; i++)
    {
      NodeContainer ueNode;
      ueNode.Create (nUe);
      ueNodes.push_back (ueNode);
    }

  MobilityHelper mobility;
  vector<Vector> enbPosition;
  Ptr < ListPositionAllocator > positionAlloc = CreateObject<ListPositionAllocator> ();
  Ptr < Building > building;
  building = Create<Building> ();
  building->SetBoundaries (Box (0.0, nRooms * roomLength,
                                0.0, nRooms * roomLength,
                                0.0, roomHeight));
  building->SetBuildingType (Building::Residential);
  building->SetExtWallsType (Building::ConcreteWithWindows);
  building->SetNFloors (1);
  building->SetNRoomsX (nRooms);
  building->SetNRoomsY (nRooms);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  BuildingsHelper::Install (enbNodes);
  uint32_t plantedEnb = 0;
  for (uint32_t row = 0; row < nRooms; row++)
    {
      for (uint32_t column = 0; column < nRooms; column++, plantedEnb++)
        {
          Vector v (roomLength * (column + 0.5),
                    roomLength * (row + 0.5),
                    nodeHeight );
          positionAlloc->Add (v);
          enbPosition.push_back (v);
          Ptr<MobilityModel> mmEnb = enbNodes.Get (plantedEnb)->GetObject<MobilityModel> ();
          mmEnb->SetPosition (v);
        }
    }

  // Add a 1-sector site
  Vector v (500, 3000, nodeHeight);
  positionAlloc->Add (v);
  enbPosition.push_back (v);
  mobility.Install (ueNodes.at(plantedEnb));
  plantedEnb++;

  // Add the 3-sector site
  for (uint32_t index = 0; index < 3; index++, plantedEnb++)
    {
      Vector v (500, 2000, nodeHeight);
      positionAlloc->Add (v);
      enbPosition.push_back (v);
      mobility.Install (ueNodes.at(plantedEnb));
    }


  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (enbNodes);

  // Position of UEs attached to eNB
  for (uint32_t i = 0; i < nEnb; i++)
    {
      Ptr<UniformRandomVariable> posX = CreateObject<UniformRandomVariable> ();
      posX->SetAttribute ("Min", DoubleValue (enbPosition.at(i).x - roomLength * 0));
      posX->SetAttribute ("Max", DoubleValue (enbPosition.at(i).x + roomLength * 0));
      Ptr<UniformRandomVariable> posY = CreateObject<UniformRandomVariable> ();
      posY->SetAttribute ("Min", DoubleValue (enbPosition.at(i).y - roomLength * 0));
      posY->SetAttribute ("Max", DoubleValue (enbPosition.at(i).y + roomLength * 0));
      positionAlloc = CreateObject<ListPositionAllocator> ();
      for (uint32_t j = 0; j < nUe; j++)
        {
          if ( i == nEnb - 3 )
            {
              positionAlloc->Add (Vector (enbPosition.at(i).x + 10, enbPosition.at(i).y, nodeHeight));
            }
          else if ( i == nEnb - 2 )
            {
              positionAlloc->Add (Vector (enbPosition.at(i).x - std::sqrt (10), enbPosition.at(i).y + std::sqrt (10), nodeHeight));
            }
          else if ( i == nEnb - 1 )
            {
              positionAlloc->Add (Vector (enbPosition.at(i).x - std::sqrt (10), enbPosition.at(i).y - std::sqrt (10), nodeHeight));
            }
          else
            {
              positionAlloc->Add (Vector (posX->GetValue (), posY->GetValue (), nodeHeight));
            }
          mobility.SetPositionAllocator (positionAlloc);
        }
      mobility.Install (ueNodes.at(i));
      BuildingsHelper::Install (ueNodes.at(i));
    }

  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  vector < NetDeviceContainer > ueDevs;

  // power setting in dBm for small cells
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (20.0));
  enbDevs = lteHelper->InstallEnbDevice (oneSectorNodes);


  // power setting for three-sector macrocell
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (43.0));

  // Beam width is made quite narrow so sectors can be noticed in the REM
  lteHelper->SetEnbAntennaModelType ("ns3::CosineAntennaModel");
  lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (0));
  lteHelper->SetEnbAntennaModelAttribute ("HorizontalBeamwidth", DoubleValue (100));
  lteHelper->SetEnbAntennaModelAttribute ("MaxGain", DoubleValue (0.0));
  enbDevs.Add ( lteHelper->InstallEnbDevice (threeSectorNodes.Get (0)));

  lteHelper->SetEnbAntennaModelType ("ns3::CosineAntennaModel");
  lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (360/3));
  lteHelper->SetEnbAntennaModelAttribute ("HorizontalBeamwidth", DoubleValue (100));
  lteHelper->SetEnbAntennaModelAttribute ("MaxGain", DoubleValue (0.0));
  enbDevs.Add ( lteHelper->InstallEnbDevice (threeSectorNodes.Get (1)));

  lteHelper->SetEnbAntennaModelType ("ns3::CosineAntennaModel");
  lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (2*360/3));
  lteHelper->SetEnbAntennaModelAttribute ("HorizontalBeamwidth", DoubleValue (100));
  lteHelper->SetEnbAntennaModelAttribute ("MaxGain", DoubleValue (0.0));
  enbDevs.Add ( lteHelper->InstallEnbDevice (threeSectorNodes.Get (2)));

  for (uint32_t i = 0; i < nEnb; i++)
    {
      NetDeviceContainer ueDev = lteHelper->InstallUeDevice (ueNodes.at(i));
      ueDevs.push_back (ueDev);
      lteHelper->Attach (ueDev, enbDevs.Get (i));
      enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
      EpsBearer bearer (q);
      lteHelper->ActivateDataRadioBearer (ueDev, bearer);
    }

  // by default, simulation will anyway stop right after the REM has been generated
  Simulator::Stop (Seconds (0.0069));

  Ptr<RadioEnvironmentMapHelper> remHelper = CreateObject<RadioEnvironmentMapHelper> ();
  remHelper->SetAttribute ("ChannelPath", StringValue ("/ChannelList/0"));
  remHelper->SetAttribute ("OutputFile", StringValue ("rem.out"));
  remHelper->SetAttribute ("XMin", DoubleValue (-2000.0));
  remHelper->SetAttribute ("XMax", DoubleValue (+2000.0));
  remHelper->SetAttribute ("YMin", DoubleValue (-500.0));
  remHelper->SetAttribute ("YMax", DoubleValue (+3500.0));
  remHelper->SetAttribute ("Z", DoubleValue (1.5));
  remHelper->Install ();

  Simulator::Run ();

//  GtkConfigStore config;
//  config.ConfigureAttributes ();

  lteHelper = 0;
  Simulator::Destroy ();
  return 0;
}
