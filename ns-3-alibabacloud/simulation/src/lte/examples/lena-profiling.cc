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
#include <ns3/buildings-module.h>
#include <iomanip>
#include <string>
#include <vector>
//#include "ns3/gtk-config-store.h"

using namespace ns3;

int
main (int argc, char *argv[])
{
  uint32_t nEnbPerFloor = 1;
  uint32_t nUe = 1;
  uint32_t nFloors = 0;
  double simTime = 1.0;
  CommandLine cmd (__FILE__);

  cmd.AddValue ("nEnb", "Number of eNodeBs per floor", nEnbPerFloor);
  cmd.AddValue ("nUe", "Number of UEs", nUe);
  cmd.AddValue ("nFloors", "Number of floors, 0 for Friis propagation model",
                nFloors);
  cmd.AddValue ("simTime", "Total duration of the simulation (in seconds)",
                simTime);
  cmd.Parse (argc, argv);

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults ();

  // parse again so you can override default values from the command line
  cmd.Parse (argc, argv);

  // Geometry of the scenario (in meters)
  // Assume squared building
  double nodeHeight = 1.5;
  double roomHeight = 3;
  double roomLength = 8;
  uint32_t nRooms = std::ceil (std::sqrt (nEnbPerFloor));
  uint32_t nEnb;

  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  //lteHelper->EnableLogComponents ();
  //LogComponentEnable ("BuildingsPropagationLossModel", LOG_LEVEL_ALL);
  if (nFloors == 0)
    {
      lteHelper->SetAttribute ("PathlossModel",
                               StringValue ("ns3::FriisPropagationLossModel"));
      nEnb = nEnbPerFloor;
    }
  else
    {
      lteHelper->SetAttribute ("PathlossModel",
                               StringValue ("ns3::HybridBuildingsPropagationLossModel"));
      nEnb = nFloors * nEnbPerFloor;
    }

  // Create Nodes: eNodeB and UE
  NodeContainer enbNodes;
  std::vector<NodeContainer> ueNodes;

  enbNodes.Create (nEnb);
  for (uint32_t i = 0; i < nEnb; i++)
    {
      NodeContainer ueNode;
      ueNode.Create (nUe);
      ueNodes.push_back (ueNode);
    }

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  std::vector<Vector> enbPosition;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  Ptr<Building> building;

  if (nFloors == 0)
    {
      // Position of eNBs
      uint32_t plantedEnb = 0;
      for (uint32_t row = 0; row < nRooms; row++)
        {
          for (uint32_t column = 0; column < nRooms && plantedEnb < nEnbPerFloor; column++, plantedEnb++)
            {
              Vector v (roomLength * (column + 0.5), roomLength * (row + 0.5), nodeHeight);
              positionAlloc->Add (v);
              enbPosition.push_back (v);
              mobility.Install (ueNodes.at(plantedEnb));
            }
        }
      mobility.SetPositionAllocator (positionAlloc);
      mobility.Install (enbNodes);
      BuildingsHelper::Install (enbNodes);

      // Position of UEs attached to eNB
      for (uint32_t i = 0; i < nEnb; i++)
        {
          Ptr<UniformRandomVariable> posX = CreateObject<UniformRandomVariable> ();
          posX->SetAttribute ("Min", DoubleValue (enbPosition.at(i).x - roomLength * 0.5));
          posX->SetAttribute ("Max", DoubleValue (enbPosition.at(i).x + roomLength * 0.5));
          Ptr<UniformRandomVariable> posY = CreateObject<UniformRandomVariable> ();
          posY->SetAttribute ("Min", DoubleValue (enbPosition.at(i).y - roomLength * 0.5));
          posY->SetAttribute ("Max", DoubleValue (enbPosition.at(i).y + roomLength * 0.5));
          positionAlloc = CreateObject<ListPositionAllocator> ();
          for (uint32_t j = 0; j < nUe; j++)
            {
              positionAlloc->Add (Vector (posX->GetValue (), posY->GetValue (), nodeHeight));
              mobility.SetPositionAllocator (positionAlloc);
            }
          mobility.Install (ueNodes.at(i));
          BuildingsHelper::Install (ueNodes.at(i));
        }

    }
  else
    {
      building = CreateObject<Building> ();
      building->SetBoundaries (Box (0.0, nRooms * roomLength,
                                    0.0, nRooms * roomLength,
                                    0.0, nFloors* roomHeight));
      building->SetBuildingType (Building::Residential);
      building->SetExtWallsType (Building::ConcreteWithWindows);
      building->SetNFloors (nFloors);
      building->SetNRoomsX (nRooms);
      building->SetNRoomsY (nRooms);
      mobility.Install (enbNodes);
      BuildingsHelper::Install (enbNodes);
      uint32_t plantedEnb = 0;
      for (uint32_t floor = 0; floor < nFloors; floor++)
        {
          uint32_t plantedEnbPerFloor = 0;
          for (uint32_t row = 0; row < nRooms; row++)
            {
              for (uint32_t column = 0; column < nRooms && plantedEnbPerFloor < nEnbPerFloor; column++, plantedEnb++, plantedEnbPerFloor++)
                {
                  Vector v (roomLength * (column + 0.5),
                            roomLength * (row + 0.5),
                            nodeHeight + roomHeight * floor);
                  positionAlloc->Add (v);
                  enbPosition.push_back (v);
                  Ptr<MobilityModel> mmEnb = enbNodes.Get (plantedEnb)->GetObject<MobilityModel> ();
                  mmEnb->SetPosition (v);

                  // Positioning UEs attached to eNB
                  mobility.Install (ueNodes.at(plantedEnb));
                  BuildingsHelper::Install (ueNodes.at(plantedEnb));
                  for (uint32_t ue = 0; ue < nUe; ue++)
                    {
                      Ptr<MobilityModel> mmUe = ueNodes.at(plantedEnb).Get (ue)->GetObject<MobilityModel> ();
                      Vector vUe (v.x, v.y, v.z);
                      mmUe->SetPosition (vUe);
                    }
                }
            }
        }
    }


  // Create Devices and install them in the Nodes (eNB and UE)
  NetDeviceContainer enbDevs;
  std::vector<NetDeviceContainer> ueDevs;
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  for (uint32_t i = 0; i < nEnb; i++)
    {
      NetDeviceContainer ueDev = lteHelper->InstallUeDevice (ueNodes.at(i));
      ueDevs.push_back (ueDev);
      lteHelper->Attach (ueDev, enbDevs.Get (i));
      enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
      EpsBearer bearer (q);
      lteHelper->ActivateDataRadioBearer (ueDev, bearer);
    }


  Simulator::Stop (Seconds (simTime));
  lteHelper->EnableTraces ();

  Simulator::Run ();

  /*GtkConfigStore config;
  config.ConfigureAttributes ();*/

  Simulator::Destroy ();
  return 0;
}
