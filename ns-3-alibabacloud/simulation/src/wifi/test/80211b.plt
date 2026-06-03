# Copyright (c) 2010 The Boeing Company
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
# Author: Gary Pei <guangyu.pei@boeing.com>

set term postscript eps color enh "Times-BoldItalic"
set output '80211b.ieee.pkt.eps'
set xlabel "RSS (dBm)"
set ylabel "Packet Received"
set yrange [0:200]
set xrange [-102:-83]
plot "80211b.txt" using 1:11 title '1M IEEE', \
"80211b.txt" using 1:12 title '2M IEEE', \
"80211b.txt" using 1:13 title '5.5M IEEE', \
"80211b.txt" using 1:14 title '11M IEEE'
set term postscript eps color enh "Times-BoldItalic"
set output '80211b.ns3.pkt.eps'
set xlabel "RSS (dBm)"
set ylabel "Packet Received"
set yrange [0:200]
set xrange [-102:-83]
plot "80211b.txt" using 1:19 title '1M DBPSK', \
"80211b.txt" using 1:20 title '2M DQPSK', \
"80211b.txt" using 1:21 title '5.5M CCK16', \
"80211b.txt" using 1:22 title '11M CCK256'
set term postscript eps color enh "Times-BoldItalic"
set output '80211b.ieee.sir.eps'
set xlabel "SIR"
set ylabel "BER"
set yrange [10e-9:1]
set xrange [-2:10]
set logscale y
plot "80211b.txt" using 2:7 title '1M IEEE', \
"80211b.txt" using 2:8 title '2M IEEE', \
"80211b.txt" using 2:9 title '5.5M IEEE', \
"80211b.txt" using 2:10 title '11M IEEE'
