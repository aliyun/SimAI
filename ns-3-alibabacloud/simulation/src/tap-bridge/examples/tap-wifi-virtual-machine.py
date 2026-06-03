# -*-  Mode: Python; -*-
#
# Copyright 2010 University of Washington
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

import sys
import ns.core
import ns.internet
import ns.mobility
import ns.network
import ns.tap_bridge
import ns.wifi

def main(argv):

    ns.core.CommandLine().Parse(argv)
    
    #
    # We are interacting with the outside, real, world.  This means we have to 
    # interact in real-time and therefore we have to use the real-time simulator
    # and take the time to calculate checksums.
    #
    ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::RealtimeSimulatorImpl"))
    ns.core.GlobalValue.Bind("ChecksumEnabled", ns.core.BooleanValue("true"))

    #
    # Create two ghost nodes.  The first will represent the virtual machine host
    # on the left side of the network; and the second will represent the VM on 
    # the right side.
    #
    nodes = ns.network.NodeContainer()
    nodes.Create (2);

    #
    # We're going to use 802.11 A so set up a wifi helper to reflect that.
    #
    wifi = ns.wifi.WifiHelper()
    wifi.SetStandard (ns.wifi.WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode", ns.core.StringValue ("OfdmRate54Mbps"));

    #
    # No reason for pesky access points, so we'll use an ad-hoc network.
    #
    wifiMac = ns.wifi.WifiMacHelper()
    wifiMac.SetType ("ns3::AdhocWifiMac");

    #
    # Configure the physical layer.
    #
    wifiChannel = ns.wifi.YansWifiChannelHelper.Default()
    wifiPhy = ns.wifi.YansWifiPhyHelper()
    wifiPhy.SetChannel(wifiChannel.Create())

    #
    # Install the wireless devices onto our ghost nodes.
    #
    devices = wifi.Install(wifiPhy, wifiMac, nodes)

    #
    # We need location information since we are talking about wifi, so add a
    # constant position to the ghost nodes.
    #
    mobility = ns.mobility.MobilityHelper()
    positionAlloc = ns.mobility.ListPositionAllocator()
    positionAlloc.Add(ns.core.Vector(0.0, 0.0, 0.0))
    positionAlloc.Add(ns.core.Vector(5.0, 0.0, 0.0))
    mobility.SetPositionAllocator(positionAlloc)
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel")
    mobility.Install(nodes)

    #
    # Use the TapBridgeHelper to connect to the pre-configured tap devices for 
    # the left side.  We go with "UseLocal" mode since the wifi devices do not
    # support promiscuous mode (because of their natures0.  This is a special
    # case mode that allows us to extend a linux bridge into ns-3 IFF we will
    # only see traffic from one other device on that bridge.  That is the case
    # for this configuration.
    #
    tapBridge = ns.tap_bridge.TapBridgeHelper()
    tapBridge.SetAttribute ("Mode", ns.core.StringValue ("UseLocal"));
    tapBridge.SetAttribute ("DeviceName", ns.core.StringValue ("tap-left"));
    tapBridge.Install (nodes.Get (0), devices.Get (0));

    #
    # Connect the right side tap to the right side wifi device on the right-side
    # ghost node.
    #
    tapBridge.SetAttribute ("DeviceName", ns.core.StringValue ("tap-right"));
    tapBridge.Install (nodes.Get (1), devices.Get (1));

    #
    # Run the simulation for ten minutes to give the user time to play around
    #
    ns.core.Simulator.Stop (ns.core.Seconds (600));
    ns.core.Simulator.Run(signal_check_frequency = -1)
    ns.core.Simulator.Destroy()
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

