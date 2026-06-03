#!/bin/bash

#
# Copyright (c) 2017 Orange Labs
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
# Author: Rediet <getachew.redieteab@orange.com>
#

# This script runs wifi-trans-example for all combinations, plots transmitted spectra
# using the generated .plt file and puts result png images in wifi-trans-results folder.
#
# Inspired from Tom's and Nicola's scripts for LAA-WiFi coexistence

#Catch abort signal
control_c()
{
  echo "Aborted, exiting..."
  exit $?
}
trap control_c SIGINT

#Save ns3 and script locations. Have to be run from where they are (otherwise won't find executables)
scriptDir=`pwd`
cd ../../../
ns3Dir=`pwd`
cd $scriptDir

#Test where script is run from
if test ! -f ${ns3Dir}/ns3 ; then
  echo "Please run this script from within the directory `dirname $0`, like this:"
  echo "cd `dirname $0`"
  echo "./`basename $0`"
  exit 1
fi

#Folder where PNGs should be grouped
outputDir=`pwd`/wifi-trans-results
if [ -d $outputDir ]; then
  echo "$outputDir directory exists."
else
  mkdir -p "$outputDir"
  echo "$outputDir directory created."
fi

#List of available standards with corresponding bandwidth
std_leg=("11a" "11_10MHZ" "11_5MHZ")
std_n=("11n_2_4GHZ" "11n_5GHZ")
std_ac_ax=("11ac" "11ax_2_4GHZ" "11ax_5GHZ")
bw_leg=(20 10 5)
bw_n=(20 40)
bw_ac_ax=(20 40 80 160)

#power spectrum density filename
pre="spectrum-analyzer-wifi-"
suf="-2-0"

#Iteratively run simulation for all combinations
#Legacy combinations
for i in 0 1 2
do
    std=${std_leg[${i}]}
    bw=${bw_leg[${i}]}
    echo "==============================================="
    echo "Run for wifi-trans-example for ${std} and ${bw} MHz"
    cd $ns3Dir
    ./ns3 run "wifi-trans-example --standard=$std --bw=$bw"
    echo "Generate PSD using ${file}.tr"
    file="${pre}${std}-${bw}MHz${suf}"
    gnuplot ${file}.plt
    if test ! -f ${file}.png ; then
      echo "PNG file (${file}.png) was not generated, something went wrong!"
      exit 1
    else
      echo "PNG file (${file}.png) generated, remove .tr and .plt files"
      mv ${file}.png $outputDir
      rm -f ${file}.*
      echo ""
    fi
done

#11n combinations
for std in "${std_n[@]}"
do
  for bw in "${bw_n[@]}"
  do
    echo "==============================================="
    echo "Run for wifi-trans-example for ${std} and ${bw} MHz"
    cd $ns3Dir
    ./ns3 run "wifi-trans-example --standard=$std --bw=$bw"
    echo "Generate PSD using ${file}.tr"
    file="${pre}${std}-${bw}MHz${suf}"
    gnuplot ${file}.plt
    if test ! -f ${file}.png ; then
      echo "PNG file (${file}.png) was not generated, something went wrong!"
      exit 1
    else
      echo "PNG file (${file}.png) generated, remove .tr and .plt files"
      mv ${file}.png $outputDir
      rm -f ${file}.*
      echo ""
    fi
  done
done

#11ac/11ax combinations
for std in "${std_ac_ax[@]}"
do
  for bw in "${bw_ac_ax[@]}"
  do
    #160 MHz not available in 2.4 GHz
    [ ${std} = "11ax_2_4GHZ" ] && ([ ${bw} = "80" ] || [ ${bw} = "160" ]) && continue
    #for all other combinations continue
    echo "==============================================="
    echo "Run for wifi-trans-example for ${std} and ${bw} MHz"
    cd $ns3Dir
    ./ns3 run "wifi-trans-example --standard=$std --bw=$bw"
    echo "Generate PSD using ${file}.tr"
    file="${pre}${std}-${bw}MHz${suf}"
    gnuplot ${file}.plt
    if test ! -f ${file}.png ; then
      echo "PNG file (${file}.png) was not generated, something went wrong!"
      exit 1
    else
      echo "PNG file (${file}.png) generated, remove .tr and .plt files"
      mv ${file}.png $outputDir
      rm -f ${file}.*
      echo ""
    fi
  done
done

echo "Finished"
