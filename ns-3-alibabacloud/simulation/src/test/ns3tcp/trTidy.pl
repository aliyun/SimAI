#!/usr/bin/perl
###
# *
# * Copyright (c) 2010 Adrian Sai-wah Tam
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License version 2 as
# * published by the Free Software Foundation;
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public
# * License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# * This perl script is used within plot.gp, a gnuplot file also
# * in this directory, for parsing the Ns3TcpLossTest logging output.
# * It can also be used, stand-alone, to tidy up the logging output 
# * from Ns3TcpStateTestCases, if logging is enabled in these tests.
### 

  while(<>) {
    s|ns3::PppHeader \(Point-to-Point Protocol: IP \(0x0021\)\) ||;
    s|/TxQueue||;
    s|/TxQ/|Q|;
    s|NodeList/|N|;
    s|/DeviceList/|D|;
    s|/MacRx||;
    s|/Enqueue||;
    s|/Dequeue||;
    s|/\$ns3::QbbNetDevice||;
    s|/\$ns3::PointToPointNetDevice||;
    s| /|\t|;
    s| ns3::|\t|g;
    s|tos 0x0 ||;
    s|protocol 6 ||;
    s|offset 0 ||;
    s|flags \[none\] ||;
    s|length:|len|;
    s|Header||g;
    s|/PhyRxDrop||;
    print;
 };
