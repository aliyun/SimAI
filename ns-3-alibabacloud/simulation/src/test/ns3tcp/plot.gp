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
# * This gnuplot file is used to plot a number of graphs from the 
# * Ns3TcpLossTestCases. In order to run this properly, the 
# * logging output from Ns3TcpLossTest case must first be enabled 
# * by manually editing ns3-tcp-loss-test-suite.cc and setting 
# * WRITE_LOGGING to true. Then, the test suite should be re-run 
# * with ./ns3 run "test-runner --suite='ns3-tcp-loss'"
# * This will generate a number of log files which are parsed 
# * below for eventual plotting to .eps format using gnuplot.
# * To run this file in gnuplot, simply: gnuplot plot.gp
### 

set key left
set terminal postscript eps enhanced color
set yrange [0:100000]
set xrange [0:3.5]
set xtics 0.5
set mxtics 2
set mytics 2
set grid mxtics xtics ytics mytics
set output "TcpNewReno.0.eps"
plot \
  "< grep ^r TcpNewReno.0.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%100000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpNewReno.0.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"
 
set output "TcpReno.0.eps"
plot \
  "< grep ^r TcpReno.0.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%100000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpReno.0.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"
 
set output "TcpTahoe.0.eps"
plot \
  "< grep ^r TcpTahoe.0.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%100000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpTahoe.0.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"
 
set yrange [0:25000]
set xrange [0:5.5]
set output "TcpNewReno.1.eps"
plot \
  "< grep ^r TcpNewReno.1.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%25000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpNewReno.1.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"
 
set output "TcpReno.1.eps"
plot \
  "< grep ^r TcpReno.1.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%25000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpReno.1.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set output "TcpTahoe.1.eps"
plot \
  "< grep ^r TcpTahoe.1.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%25000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpTahoe.1.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set yrange [0:30000]
set output "TcpNewReno.2.eps"
plot \
  "< grep ^r TcpNewReno.2.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%30000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpNewReno.2.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set output "TcpReno.2.eps"
plot \
  "< grep ^r TcpReno.2.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%30000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpReno.2.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set output "TcpTahoe.2.eps"
plot \
  "< grep ^r TcpTahoe.2.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%30000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpTahoe.2.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set yrange [0:35000]
set xrange [0:6.0]
set output "TcpNewReno.3.eps"
plot \
  "< grep ^r TcpNewReno.3.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%35000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpNewReno.3.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set output "TcpReno.3.eps"
plot \
  "< grep ^r TcpReno.3.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%35000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpReno.3.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set output "TcpTahoe.3.eps"
plot \
  "< grep ^r TcpTahoe.3.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%35000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpTahoe.3.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set yrange [0:40000]
set xrange [0:6.5]
set output "TcpNewReno.4.eps"
plot \
  "< grep ^r TcpNewReno.4.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%40000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpNewReno.4.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set output "TcpReno.4.eps"
plot \
  "< grep ^r TcpReno.4.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%40000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpReno.4.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"

set output "TcpTahoe.4.eps"
plot \
  "< grep ^r TcpTahoe.4.log | perl trTidy.pl | grep N0 | perl -ne 's/^..//;s/\t.*Ack=/ /;s/ Win.*$//;split;print $_[0].q{ }.($_[1]%40000).q{\n}'" using 1:2 w p pt 2 lc rgb "#00FF00" title "ack", \
  "< grep cwnd TcpTahoe.4.log | grep seconds | perl -pe 's/^.*to //;s/ at time / /;s/ seconds//'" using 2:1 w p pt 1 lc rgb "#FF0000" title "cwnd"
