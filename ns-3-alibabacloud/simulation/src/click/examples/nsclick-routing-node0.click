// nsclick-lan-single-interface.click
//
// Copyright (c) 2011, Deutsche Telekom Laboratories
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License version
// 2 as published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Ruben Merz <ruben@net.t-labs.tu-berlin.de>
//
// This is a single host Click configuration for a LAN.
// The node broadcasts ARP requests if it wants to find a destination
// address, and it responds to ARP requests made for it.

elementclass LanSimHost {
  $ipaddr, $hwaddr |

  cl::Classifier(12/0806 20/0001,12/0806 20/0002, -);
  forhost::IPClassifier(dst host $ipaddr,-);
  arpquerier::ARPQuerier(eth0);
  arpresponder::ARPResponder(eth0);

  ethout::Queue 	
    -> ToDump(out_eth0.pcap,PER_NODE 1)
    -> ToSimDevice(eth0);

  // All packets received on eth0 are silently
  // dropped if they are destined for another location
  FromSimDevice(eth0,SNAPLEN 4096)
    -> ToDump(in_eth0.pcap,PER_NODE 1,ENCAP ETHER)
    -> cl;

  // ARP queries from other nodes go to the ARP responder element
  cl[0] -> arpresponder;

  // ARP responses go to our ARP query element
  cl[1] -> [1]arpquerier;

  // All other packets get checked whether they are meant for us
  cl[2]				
    -> Strip(14)
    -> CheckIPHeader2
    -> MarkIPHeader
    -> GetIPAddress(16) // Sets destination IP address annotation from packet data
    -> forhost; 

  // Packets for us are pushed outside
  forhost[0]
    ->[0]output;

  // Packets for other folks or broadcast packets get sent to output 1
  forhost[1]
    -> ToDump(discard.pcap,2000,PER_NODE 1,ENCAP IP)
    -> [1]output;

  // Incoming packets get pushed into the ARP query module
  input[0]
    -> arpquerier;

  // Both the ARP query and response modules send data out to
  // the simulated network device, eth0.
  arpquerier
    -> ToDump(out_arpquery.pcap,PER_NODE 1)
    -> ethout;

  arpresponder
    -> ToDump(out_arprespond.pcap,PER_NODE 1)
    -> ethout;

}

elementclass TapSimHost {
  $dev |

  // It is mandatory to use an IPRouteTable element with ns-3-click
  rt :: LinearIPLookup (172.16.1.0/24 0.0.0.0 1, 172.16.2.0/24 172.16.1.2 1);

  // Packets go to "tap0" which sends them to the kernel
  input[0]
    -> ToDump(tokernel.pcap,2000,IP,PER_NODE 1) 
    -> ToSimDevice($dev,IP);

  // Packets sent out by the "kernel" get pushed outside
  FromSimDevice($dev,SNAPLEN 4096)
    -> CheckIPHeader2
    -> ToDump(fromkernel.pcap,2000,IP,PER_NODE 1)
    -> GetIPAddress(16)
    -> rt
    -> [0]output;

  rt[1] -> [0] output;
}

// Instantiate elements
lan::LanSimHost(eth0:ip,eth0:eth);
kernel::TapSimHost(tap0);

// Users can do some processing between the two elements
lan[0] -> kernel;
kernel -> lan;
// Packets for others or broadcasts are discarded
lan[1] -> Discard;

