# /*
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License version 2 as
#  * published by the Free Software Foundation;
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#  *
#  */

# 
#  This ns-3 example demonstrates the use of helper functions to ease 
#  the construction of simulation scenarios.  
#  
#  The simulation topology consists of a mixed wired and wireless
#  scenario in which a hierarchical mobility model is used.
# 
#  The simulation layout consists of N backbone routers interconnected
#  by an ad hoc wifi network.
#  Each backbone router also has a local 802.11 network and is connected
#  to a local LAN.  An additional set of(K-1) nodes are connected to
#  this backbone.  Finally, a local LAN is connected to each router
#  on the backbone, with L-1 additional hosts.  
# 
#  The nodes are populated with TCP/IP stacks, and OLSR unicast routing
#  on the backbone.  An example UDP transfer is shown.  The simulator
#  be configured to output tcpdumps or traces from different nodes.
# 
# 
#           +--------------------------------------------------------+
#           |                                                        |
#           |              802.11 ad hoc, ns-2 mobility              | 
#           |                                                        |
#           +--------------------------------------------------------+
#                    |       o o o(N backbone routers)       |
#                +--------+                               +--------+
#      wired LAN | mobile |                     wired LAN | mobile |
#     -----------| router |                    -----------| router |
#                ---------                                ---------
#                    |                                        |
#           +----------------+                       +----------------+
#           |     802.11     |                       |     802.11     |
#           |      net       |                       |       net      |
#           |   K-1 hosts    |                       |   K-1 hosts    |
#           +----------------+                       +----------------+
# 

import ns.applications
import ns.core
import ns.csma
import ns.internet
import ns.mobility
import ns.network
import ns.olsr
import ns.wifi

# # 
# #  This function will be used below as a trace sink
# #  
# static void
# CourseChangeCallback(std.string path, Ptr<const MobilityModel> model)
# {
#   Vector position = model.GetPosition();
#   std.cout << "CourseChange " << path << " x=" << position.x << ", y=" << position.y << ", z=" << position.z << std.endl;
# }

def main(argv): 
    # 
    #  First, we initialize a few local variables that control some 
    #  simulation parameters.
    #

    cmd = ns.core.CommandLine()
    cmd.backboneNodes = 10
    cmd.infraNodes = 2
    cmd.lanNodes = 2
    cmd.stopTime = 20

    # 
    #  Simulation defaults are typically set next, before command line
    #  arguments are parsed.
    # 
    ns.core.Config.SetDefault("ns3::OnOffApplication::PacketSize", ns.core.StringValue("1472"))
    ns.core.Config.SetDefault("ns3::OnOffApplication::DataRate", ns.core.StringValue("100kb/s"))

    # 
    #  For convenience, we add the local variables to the command line argument
    #  system so that they can be overridden with flags such as 
    #  "--backboneNodes=20"
    # 
    
    cmd.AddValue("backboneNodes", "number of backbone nodes")
    cmd.AddValue("infraNodes", "number of leaf nodes")
    cmd.AddValue("lanNodes", "number of LAN nodes")
    cmd.AddValue("stopTime", "simulation stop time(seconds)")
    
    # 
    #  The system global variables and the local values added to the argument
    #  system can be overridden by command line arguments by using this call.
    # 
    cmd.Parse(argv)

    backboneNodes = int(cmd.backboneNodes)
    infraNodes = int(cmd.infraNodes) 
    lanNodes = int(cmd.lanNodes)
    stopTime = int(cmd.stopTime)

    if (stopTime < 10):
        print ("Use a simulation stop time >= 10 seconds")
        exit(1)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 
    #                                                                        # 
    #  Construct the backbone                                                # 
    #                                                                        # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 

    # 
    #  Create a container to manage the nodes of the adhoc(backbone) network.
    #  Later we'll create the rest of the nodes we'll need.
    # 
    backbone = ns.network.NodeContainer()
    backbone.Create(backboneNodes)
    # 
    #  Create the backbone wifi net devices and install them into the nodes in 
    #  our container
    # 
    wifi = ns.wifi.WifiHelper()
    mac = ns.wifi.WifiMacHelper()
    mac.SetType("ns3::AdhocWifiMac")
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode", ns.core.StringValue("OfdmRate54Mbps"))
    wifiPhy = ns.wifi.YansWifiPhyHelper()
    wifiPhy.SetPcapDataLinkType(wifiPhy.DLT_IEEE802_11_RADIO)
    wifiChannel = ns.wifi.YansWifiChannelHelper.Default()
    wifiPhy.SetChannel(wifiChannel.Create())
    backboneDevices = wifi.Install(wifiPhy, mac, backbone)
    # 
    #  Add the IPv4 protocol stack to the nodes in our container
    # 
    print ("Enabling OLSR routing on all backbone nodes")
    internet = ns.internet.InternetStackHelper()
    olsr = ns.olsr.OlsrHelper()
    internet.SetRoutingHelper(olsr); # has effect on the next Install ()
    internet.Install(backbone);
    # re-initialize for non-olsr routing.
    # internet.Reset()
    # 
    #  Assign IPv4 addresses to the device drivers(actually to the associated
    #  IPv4 interfaces) we just created.
    # 
    ipAddrs = ns.internet.Ipv4AddressHelper()
    ipAddrs.SetBase(ns.network.Ipv4Address("192.168.0.0"), ns.network.Ipv4Mask("255.255.255.0"))
    ipAddrs.Assign(backboneDevices)

    # 
    #  The ad-hoc network nodes need a mobility model so we aggregate one to 
    #  each of the nodes we just finished building.  
    # 
    mobility = ns.mobility.MobilityHelper()
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", ns.core.DoubleValue(20.0),
                                  "MinY", ns.core.DoubleValue(20.0),
                                  "DeltaX", ns.core.DoubleValue(20.0),
                                  "DeltaY", ns.core.DoubleValue(20.0),
                                  "GridWidth", ns.core.UintegerValue(5),
                                  "LayoutType", ns.core.StringValue("RowFirst"))
    mobility.SetMobilityModel("ns3::RandomDirection2dMobilityModel",
                               "Bounds", ns.mobility.RectangleValue(ns.mobility.Rectangle(-500, 500, -500, 500)),
                               "Speed", ns.core.StringValue ("ns3::ConstantRandomVariable[Constant=2]"),
                               "Pause", ns.core.StringValue ("ns3::ConstantRandomVariable[Constant=0.2]"))
    mobility.Install(backbone)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 
    #                                                                        # 
    #  Construct the LANs                                                    # 
    #                                                                        # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 

    #  Reset the address base-- all of the CSMA networks will be in
    #  the "172.16 address space
    ipAddrs.SetBase(ns.network.Ipv4Address("172.16.0.0"), ns.network.Ipv4Mask("255.255.255.0"))

    for i in range(backboneNodes):
        print ("Configuring local area network for backbone node ", i)
        # 
        #  Create a container to manage the nodes of the LAN.  We need
        #  two containers here; one with all of the new nodes, and one
        #  with all of the nodes including new and existing nodes
        # 
        newLanNodes = ns.network.NodeContainer()
        newLanNodes.Create(lanNodes - 1)
        #  Now, create the container with all nodes on this link
        lan = ns.network.NodeContainer(ns.network.NodeContainer(backbone.Get(i)), newLanNodes)
        # 
        #  Create the CSMA net devices and install them into the nodes in our 
        #  collection.
        # 
        csma = ns.csma.CsmaHelper()
        csma.SetChannelAttribute("DataRate", ns.network.DataRateValue(ns.network.DataRate(5000000)))
        csma.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.MilliSeconds(2)))
        lanDevices = csma.Install(lan)
        # 
        #  Add the IPv4 protocol stack to the new LAN nodes
        # 
        internet.Install(newLanNodes)
        # 
        #  Assign IPv4 addresses to the device drivers(actually to the 
        #  associated IPv4 interfaces) we just created.
        # 
        ipAddrs.Assign(lanDevices)
        # 
        #  Assign a new network prefix for the next LAN, according to the
        #  network mask initialized above
        # 
        ipAddrs.NewNetwork()
        #
        # The new LAN nodes need a mobility model so we aggregate one
        # to each of the nodes we just finished building.
        #
        mobilityLan = ns.mobility.MobilityHelper() 
        positionAlloc = ns.mobility.ListPositionAllocator()
        for j in range(newLanNodes.GetN()):
            positionAlloc.Add(ns.core.Vector(0.0, (j*10 + 10), 0.0))

        mobilityLan.SetPositionAllocator(positionAlloc)
        mobilityLan.PushReferenceMobilityModel(backbone.Get(i))
        mobilityLan.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobilityLan.Install(newLanNodes);

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 
    #                                                                        # 
    #  Construct the mobile networks                                         # 
    #                                                                        # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 

    #  Reset the address base-- all of the 802.11 networks will be in
    #  the "10.0" address space
    ipAddrs.SetBase(ns.network.Ipv4Address("10.0.0.0"), ns.network.Ipv4Mask("255.255.255.0"))

    for i in range(backboneNodes):
        print ("Configuring wireless network for backbone node ", i)
        # 
        #  Create a container to manage the nodes of the LAN.  We need
        #  two containers here; one with all of the new nodes, and one
        #  with all of the nodes including new and existing nodes
        # 
        stas = ns.network.NodeContainer()
        stas.Create(infraNodes - 1)
        #  Now, create the container with all nodes on this link
        infra = ns.network.NodeContainer(ns.network.NodeContainer(backbone.Get(i)), stas)
        # 
        #  Create another ad hoc network and devices
        # 
        ssid = ns.wifi.Ssid('wifi-infra' + str(i))
        wifiInfra = ns.wifi.WifiHelper()
        wifiPhy.SetChannel(wifiChannel.Create())
        macInfra = ns.wifi.WifiMacHelper();
        macInfra.SetType("ns3::StaWifiMac",
                         "Ssid", ns.wifi.SsidValue(ssid))

        # setup stas
        staDevices = wifiInfra.Install(wifiPhy, macInfra, stas)
        # setup ap.
        macInfra.SetType("ns3::ApWifiMac",
                         "Ssid", ns.wifi.SsidValue(ssid))
        apDevices = wifiInfra.Install(wifiPhy, macInfra, backbone.Get(i))
        # Collect all of these new devices
        infraDevices = ns.network.NetDeviceContainer(apDevices, staDevices)

        #  Add the IPv4 protocol stack to the nodes in our container
        # 
        internet.Install(stas)
        # 
        #  Assign IPv4 addresses to the device drivers(actually to the associated
        #  IPv4 interfaces) we just created.
        # 
        ipAddrs.Assign(infraDevices)
        # 
        #  Assign a new network prefix for each mobile network, according to 
        #  the network mask initialized above
        # 
        ipAddrs.NewNetwork()
        # 
        #  The new wireless nodes need a mobility model so we aggregate one 
        #  to each of the nodes we just finished building.
        # 
        subnetAlloc = ns.mobility.ListPositionAllocator()
        for j in range(infra.GetN()):
            subnetAlloc.Add(ns.core.Vector(0.0, j, 0.0))

        mobility.PushReferenceMobilityModel(backbone.Get(i))
        mobility.SetPositionAllocator(subnetAlloc)
        mobility.SetMobilityModel("ns3::RandomDirection2dMobilityModel",
                                  "Bounds", ns.mobility.RectangleValue(ns.mobility.Rectangle(-10, 10, -10, 10)),
                                  "Speed", ns.core.StringValue ("ns3::ConstantRandomVariable[Constant=3]"),
                                  "Pause", ns.core.StringValue ("ns3::ConstantRandomVariable[Constant=0.4]"))
        mobility.Install(stas)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 
    #                                                                        # 
    #  Application configuration                                             # 
    #                                                                        # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 

    #  Create the OnOff application to send UDP datagrams of size
    #  210 bytes at a rate of 448 Kb/s, between two nodes
    print ("Create Applications.")
    port = 9   #  Discard port(RFC 863)

    appSource = ns.network.NodeList.GetNode(backboneNodes)
    lastNodeIndex = backboneNodes + backboneNodes*(lanNodes - 1) + backboneNodes*(infraNodes - 1) - 1
    appSink = ns.network.NodeList.GetNode(lastNodeIndex)
    # Let's fetch the IP address of the last node, which is on Ipv4Interface 1
    remoteAddr = appSink.GetObject(ns.internet.Ipv4.GetTypeId()).GetAddress(1,0).GetLocal()

    onoff = ns.applications.OnOffHelper("ns3::UdpSocketFactory", 
                            ns.network.Address(ns.network.InetSocketAddress(remoteAddr, port)))
    apps = onoff.Install(ns.network.NodeContainer(appSource))
    apps.Start(ns.core.Seconds(3))
    apps.Stop(ns.core.Seconds(stopTime - 1))

    #  Create a packet sink to receive these packets
    sink = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory", 
                                ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port))
    apps = sink.Install(ns.network.NodeContainer(appSink))
    apps.Start(ns.core.Seconds(3))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 
    #                                                                        # 
    #  Tracing configuration                                                 # 
    #                                                                        # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # / 

    print ("Configure Tracing.")
    csma = ns.csma.CsmaHelper()
    # 
    #  Let's set up some ns-2-like ascii traces, using another helper class
    # 
    ascii = ns.network.AsciiTraceHelper();
    stream = ascii.CreateFileStream("mixed-wireless.tr");
    wifiPhy.EnableAsciiAll(stream);
    csma.EnableAsciiAll(stream);
    internet.EnableAsciiIpv4All(stream);

    #  Csma captures in non-promiscuous mode
    csma.EnablePcapAll("mixed-wireless", False)
    #  Let's do a pcap trace on the backbone devices
    wifiPhy.EnablePcap("mixed-wireless", backboneDevices)
    wifiPhy.EnablePcap("mixed-wireless", appSink.GetId(), 0)

#   #ifdef ENABLE_FOR_TRACING_EXAMPLE
#     Config.Connect("/NodeList/*/$MobilityModel/CourseChange",
#       MakeCallback(&CourseChangeCallback))
#   #endif


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    #                                                                        # 
    #  Run simulation                                                        # 
    #                                                                        # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

    print ("Run Simulation.")
    ns.core.Simulator.Stop(ns.core.Seconds(stopTime))
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()


if __name__ == '__main__':
    import sys
    main(sys.argv)


