#
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License version 2 as
# * published by the Free Software Foundation;
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

# Network topology
#
#       n0    n1   n2   n3
#       |     |    |    |
#       =================
#              LAN
#
# - UDP flows from n0 to n1 and back
# - DropTail queues 
# - Tracing of queues and packet receptions to file "udp-echo.tr"

import ns.applications
import ns.core
import ns.csma
import ns.internet
import ns.network

def main(argv):
  #
  # Allow the user to override any of the defaults and the above Bind() at
  # run-time, via command-line arguments
  #
  cmd = ns.core.CommandLine()
  cmd.Parse(argv)

  #
  # But since this is a realtime script, don't allow the user to mess with
  # that.
  #
  ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::RealtimeSimulatorImpl"))

  #
  # Explicitly create the nodes required by the topology (shown above).
  #
  print ("Create nodes.")
  n = ns.network.NodeContainer()
  n.Create(4)

  internet = ns.internet.InternetStackHelper()
  internet.Install(n)

  #
  # Explicitly create the channels required by the topology (shown above).
  #
  print ("Create channels.")
  csma = ns.csma.CsmaHelper()
  csma.SetChannelAttribute("DataRate", ns.network.DataRateValue(ns.network.DataRate(5000000)))
  csma.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.MilliSeconds(2)));
  csma.SetDeviceAttribute("Mtu", ns.core.UintegerValue(1400))
  d = csma.Install(n)

  #
  # We've got the "hardware" in place.  Now we need to add IP addresses.
  #
  print ("Assign IP Addresses.")
  ipv4 = ns.internet.Ipv4AddressHelper()
  ipv4.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
  i = ipv4.Assign(d)

  print ("Create Applications.")

  #
  # Create a UdpEchoServer application on node one.
  #
  port = 9  # well-known echo port number
  server = ns.applications.UdpEchoServerHelper(port)
  apps = server.Install(n.Get(1))
  apps.Start(ns.core.Seconds(1.0))
  apps.Stop(ns.core.Seconds(10.0))

  #
  # Create a UdpEchoClient application to send UDP datagrams from node zero to
  # node one.
  #
  packetSize = 1024
  maxPacketCount = 500
  interPacketInterval = ns.core.Seconds(0.01)
  client = ns.applications.UdpEchoClientHelper(i.GetAddress (1), port)
  client.SetAttribute("MaxPackets", ns.core.UintegerValue(maxPacketCount))
  client.SetAttribute("Interval", ns.core.TimeValue(interPacketInterval))
  client.SetAttribute("PacketSize", ns.core.UintegerValue(packetSize))
  apps = client.Install(n.Get(0))
  apps.Start(ns.core.Seconds(2.0))
  apps.Stop(ns.core.Seconds(10.0))

  ascii = ns.network.AsciiTraceHelper()
  csma.EnableAsciiAll(ascii.CreateFileStream("realtime-udp-echo.tr"))
  csma.EnablePcapAll("realtime-udp-echo", False)

  #
  # Now, do the actual simulation.
  #
  print ("Run Simulation.")
  ns.core.Simulator.Run()
  ns.core.Simulator.Destroy()
  print ("Done.")

if __name__ == '__main__':
  import sys
  main(sys.argv)
