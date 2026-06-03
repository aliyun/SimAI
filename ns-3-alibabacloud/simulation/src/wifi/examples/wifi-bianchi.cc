/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 The Boeing Company
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
 * Author: Gary Pei <guangyu.pei@boeing.com>
 *
 * Updated by Tom Henderson, Rohan Patidar, Hao Yin and Sébastien Deronne
 */

// This program conducts a Bianchi analysis of a wifi network.
// Bianchi analysis involves saturating the network and observing how
// the maximum achievable throughput is governed by the DCF or ECDA
// channel access mechanisms.  This program offers many configurable
// options, traces, and a validation option; the main output is a Gnuplot
// plot file plotting throughput vs. number of nodes.

#include <fstream>
#include "ns3/log.h"
#include "ns3/config.h"
#include "ns3/gnuplot.h"
#include "ns3/boolean.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/integer.h"
#include "ns3/uinteger.h"
#include "ns3/command-line.h"
#include "ns3/node-list.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/queue-size.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-delay-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/mobility-helper.h"
#include "ns3/wifi-net-device.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-server.h"
#include "ns3/application-container.h"
#include "ns3/ampdu-subframe-header.h"
#include "ns3/wifi-mac.h"

#define PI 3.1415926535

NS_LOG_COMPONENT_DEFINE ("WifiBianchi");

using namespace ns3;

std::ofstream cwTraceFile;         ///< File that traces CW over time
std::ofstream backoffTraceFile;    ///< File that traces backoff over time
std::ofstream phyTxTraceFile;      ///< File that traces PHY transmissions  over time
std::ofstream macTxTraceFile;      ///< File that traces MAC transmissions  over time
std::ofstream macRxTraceFile;      ///< File that traces MAC receptions  over time
std::ofstream socketSendTraceFile; ///< File that traces packets transmitted by the application  over time

std::map<Mac48Address, uint64_t> packetsReceived;              ///< Map that stores the total packets received per STA (and addressed to that STA)
std::map<Mac48Address, uint64_t> bytesReceived;                ///< Map that stores the total bytes received per STA (and addressed to that STA)
std::map<Mac48Address, uint64_t> packetsTransmitted;           ///< Map that stores the total packets transmitted per STA
std::map<Mac48Address, uint64_t> psduFailed;                   ///< Map that stores the total number of unsuccessfully received PSDUS (for which the PHY header was successfully received)  per STA (including PSDUs not addressed to that STA)
std::map<Mac48Address, uint64_t> psduSucceeded;                ///< Map that stores the total number of successfully received PSDUs per STA (including PSDUs not addressed to that STA)
std::map<Mac48Address, uint64_t> phyHeaderFailed;              ///< Map that stores the total number of unsuccessfully received PHY headers per STA
std::map<Mac48Address, uint64_t> rxEventWhileTxing;            ///< Map that stores the number of reception events per STA that occured while PHY was already transmitting a PPDU
std::map<Mac48Address, uint64_t> rxEventWhileRxing;            ///< Map that stores the number of reception events per STA that occured while PHY was already receiving a PPDU
std::map<Mac48Address, uint64_t> rxEventWhileDecodingPreamble; ///< Map that stores the number of reception events per STA that occured while PHY was already decoding a preamble
std::map<Mac48Address, uint64_t> rxEventAbortedByTx;           ///< Map that stores the number of reception events aborted per STA because the PHY has started to transmit

std::map<Mac48Address, Time> timeFirstReceived;    ///< Map that stores the time at which the first packet was received per STA (and the packet is addressed to that STA)
std::map<Mac48Address, Time> timeLastReceived;     ///< Map that stores the time at which the last packet was received per STA (and the packet is addressed to that STA)
std::map<Mac48Address, Time> timeFirstTransmitted; ///< Map that stores the time at which the first packet was transmitted per STA
std::map<Mac48Address, Time> timeLastTransmitted;  ///< Map that stores the time at which the last packet was transmitted per STA

std::set<uint32_t> associated; ///< Contains the IDs of the STAs that successfully associated to the access point (in infrastructure mode only)

bool tracing = false;    ///< Flag to enable/disable generation of tracing files
uint32_t pktSize = 1500; ///< packet size used for the simulation (in bytes)
uint8_t maxMpdus = 0;    ///< The maximum number of MPDUs in A-MPDUs (0 to disable MPDU aggregation)

std::map<std::string /* mode */, std::map<unsigned int /* number of nodes */, double /* calculated throughput */> > bianchiResultsEifs =
{
/* 11b */
    {"DsssRate1Mbps", {
        {5, 0.8418}, {10, 0.7831}, {15, 0.7460}, {20, 0.7186}, {25, 0.6973}, {30, 0.6802}, {35, 0.6639}, {40, 0.6501}, {45, 0.6386}, {50, 0.6285},
    }},
    {"DsssRate2Mbps", {
        {5, 1.6170}, {10, 1.5075}, {15, 1.4371}, {20, 1.3849}, {25, 1.3442}, {30, 1.3115}, {35, 1.2803}, {40, 1.2538}, {45, 1.2317}, {50, 1.2124},
    }},
    {"DsssRate5_5Mbps", {
        {5, 3.8565}, {10, 3.6170}, {15, 3.4554}, {20, 3.3339}, {25, 3.2385}, {30, 3.1613}, {35, 3.0878}, {40, 3.0249}, {45, 2.9725}, {50, 2.9266},
    }},
    {"DsssRate11Mbps", {
        {5, 6.3821}, {10, 6.0269}, {15, 5.7718}, {20, 5.5765}, {25, 5.4217}, {30, 5.2958}, {35, 5.1755}, {40, 5.0722}, {45, 4.9860}, {50, 4.9103},
    }},
/* 11a */
    {"OfdmRate6Mbps", {
        {5, 4.6899}, {10, 4.3197}, {15, 4.1107}, {20, 3.9589}, {25, 3.8478}, {30, 3.7490}, {35, 3.6618}, {40, 3.5927}, {45, 3.5358}, {50, 3.4711},
    }},
    {"OfdmRate9Mbps", {
        {5, 6.8188}, {10, 6.2885}, {15, 5.9874}, {20, 5.7680}, {25, 5.6073}, {30, 5.4642}, {35, 5.3378}, {40, 5.2376}, {45, 5.1551}, {50, 5.0612},
    }},
    {"OfdmRate12Mbps", {
        {5, 8.8972}, {10, 8.2154}, {15, 7.8259}, {20, 7.5415}, {25, 7.3329}, {30, 7.1469}, {35, 6.9825}, {40, 6.8521}, {45, 6.7447}, {50, 6.6225},
    }},
    {"OfdmRate18Mbps", {
        {5, 12.6719}, {10, 11.7273}, {15, 11.1814}, {20, 10.7810}, {25, 10.4866}, {30, 10.2237}, {35, 9.9910}, {40, 9.8061}, {45, 9.6538}, {50, 9.4804},
    }},
    {"OfdmRate24Mbps", {
        {5, 16.0836}, {10, 14.9153}, {15, 14.2327}, {20, 13.7300}, {25, 13.3595}, {30, 13.0281}, {35, 12.7343}, {40, 12.5008}, {45, 12.3083}, {50, 12.0889},
    }},
    {"OfdmRate36Mbps", {
        {5, 22.0092}, {10, 20.4836}, {15, 19.5743}, {20, 18.8997}, {25, 18.4002}, {30, 17.9524}, {35, 17.5545}, {40, 17.2377}, {45, 16.9760}, {50, 16.6777},
    }},
    {"OfdmRate48Mbps", {
        {5, 26.8382}, {10, 25.0509}, {15, 23.9672}, {20, 23.1581}, {25, 22.5568}, {30, 22.0165}, {35, 21.5355}, {40, 21.1519}, {45, 20.8348}, {50, 20.4729},
    }},
    {"OfdmRate54Mbps", {
        {5, 29.2861}, {10, 27.3763}, {15, 26.2078}, {20, 25.3325}, {25, 24.6808}, {30, 24.0944}, {35, 23.5719}, {40, 23.1549}, {45, 22.8100}, {50, 22.4162},
    }},
/* 11g */
    {"ErpOfdmRate6Mbps", {
        {5, 4.6899}, {10, 4.3197}, {15, 4.1107}, {20, 3.9589}, {25, 3.8478}, {30, 3.7490}, {35, 3.6618}, {40, 3.5927}, {45, 3.5358}, {50, 3.4711},
    }},
    {"ErpOfdmRate9Mbps", {
        {5, 6.8188}, {10, 6.2885}, {15, 5.9874}, {20, 5.7680}, {25, 5.6073}, {30, 5.4642}, {35, 5.3378}, {40, 5.2376}, {45, 5.1551}, {50, 5.0612},
    }},
    {"ErpOfdmRate12Mbps", {
        {5, 8.8972}, {10, 8.2154}, {15, 7.8259}, {20, 7.5415}, {25, 7.3329}, {30, 7.1469}, {35, 6.9825}, {40, 6.8521}, {45, 6.7447}, {50, 6.6225},
    }},
    {"ErpOfdmRate18Mbps", {
        {5, 12.6719}, {10, 11.7273}, {15, 11.1814}, {20, 10.7810}, {25, 10.4866}, {30, 10.2237}, {35, 9.9910}, {40, 9.8061}, {45, 9.6538}, {50, 9.4804},
    }},
    {"ErpOfdmRate24Mbps", {
        {5, 16.0836}, {10, 14.9153}, {15, 14.2327}, {20, 13.7300}, {25, 13.3595}, {30, 13.0281}, {35, 12.7343}, {40, 12.5008}, {45, 12.3083}, {50, 12.0889},
    }},
    {"ErpOfdmRate36Mbps", {
        {5, 22.0092}, {10, 20.4836}, {15, 19.5743}, {20, 18.8997}, {25, 18.4002}, {30, 17.9524}, {35, 17.5545}, {40, 17.2377}, {45, 16.9760}, {50, 16.6777},
    }},
    {"ErpOfdmRate48Mbps", {
        {5, 26.8382}, {10, 25.0509}, {15, 23.9672}, {20, 23.1581}, {25, 22.5568}, {30, 22.0165}, {35, 21.5355}, {40, 21.1519}, {45, 20.8348}, {50, 20.4729},
    }},
    {"ErpOfdmRate54Mbps", {
        {5, 29.2861}, {10, 27.3763}, {15, 26.2078}, {20, 25.3325}, {25, 24.6808}, {30, 24.0944}, {35, 23.5719}, {40, 23.1549}, {45, 22.8100}, {50, 22.4162},
    }},
/* 11ax, no frame aggregation */
    {"HeMcs0_20MHz", {
        {5, 6.3381}, {10, 5.8172}, {15, 5.5223}, {20, 5.3146}, {25, 5.1525}, {30, 5.0187}, {35, 4.9039}, {40, 4.8034}, {45, 4.7134}, {50, 4.6317},
    }},
    {"HeMcs1_20MHz", {
        {5, 11.6580}, {10, 10.7369}, {15, 10.2068}, {20, 9.8309}, {25, 9.5365}, {30, 9.2930}, {35, 9.0837}, {40, 8.9001}, {45, 8.7355}, {50, 8.5860},
    }},
    {"HeMcs2_20MHz", {
        {5, 15.8572}, {10, 14.6445}, {15, 13.9367}, {20, 13.4323}, {25, 13.0361}, {30, 12.7076}, {35, 12.4249}, {40, 12.1766}, {45, 11.9538}, {50, 11.7511},
    }},
    {"HeMcs3_20MHz", {
        {5, 19.7457}, {10, 18.2820}, {15, 17.4163}, {20, 16.7963}, {25, 16.3078}, {30, 15.9021}, {35, 15.5524}, {40, 15.2449}, {45, 14.9687}, {50, 14.7173},
    }},
    {"HeMcs4_20MHz", {
        {5, 25.8947}, {10, 24.0721}, {15, 22.9698}, {20, 22.1738}, {25, 21.5437}, {30, 21.0186}, {35, 20.5650}, {40, 20.1654}, {45, 19.8059}, {50, 19.4784},
    }},
    {"HeMcs5_20MHz", {
        {5, 30.0542}, {10, 28.0155}, {15, 26.7625}, {20, 25.8523}, {25, 25.1295}, {30, 24.5258}, {35, 24.0034}, {40, 23.5426}, {45, 23.1277}, {50, 22.7492},
    }},
    {"HeMcs6_20MHz", {
        {5, 32.6789}, {10, 30.5150}, {15, 29.1708}, {20, 28.1907}, {25, 27.4107}, {30, 26.7583}, {35, 26.1931}, {40, 25.6941}, {45, 25.2446}, {50, 24.8343},
    }},
    {"HeMcs7_20MHz", {
        {5, 34.1710}, {10, 31.9398}, {15, 30.5451}, {20, 29.5261}, {25, 28.7140}, {30, 28.0342}, {35, 27.4449}, {40, 26.9245}, {45, 26.4554}, {50, 26.0271},
    }},
    {"HeMcs8_20MHz", {
        {5, 37.6051}, {10, 35.2296}, {15, 33.7228}, {20, 32.6160}, {25, 31.7314}, {30, 30.9895}, {35, 30.3455}, {40, 29.7760}, {45, 29.2623}, {50, 28.7929},
    }},
    {"HeMcs9_20MHz", {
        {5, 39.5947}, {10, 37.1424}, {15, 35.5731}, {20, 34.4169}, {25, 33.4911}, {30, 32.7138}, {35, 32.0385}, {40, 31.4410}, {45, 30.9016}, {50, 30.4086},
    }},
    {"HeMcs10_20MHz", {
        {5, 39.5947}, {10, 37.1424}, {15, 35.5731}, {20, 34.4169}, {25, 33.4911}, {30, 32.7138}, {35, 32.0385}, {40, 31.4410}, {45, 30.9016}, {50, 30.4086},
    }},
    {"HeMcs11_20MHz", {
        {5, 41.8065}, {10, 39.2749}, {15, 37.6383}, {20, 36.4282}, {25, 35.4575}, {30, 34.6414}, {35, 33.9316}, {40, 33.3031}, {45, 32.7355}, {50, 32.2164},
    }},
    {"HeMcs0_40MHz", {
        {5, 11.4999}, {10, 10.5902}, {15, 10.0669}, {20, 9.6960}, {25, 9.4055}, {30, 9.1652}, {35, 8.9587}, {40, 8.7775}, {45, 8.6151}, {50, 8.4676},
    }},
    {"HeMcs1_40MHz", {
        {5, 19.5937}, {10, 18.1394}, {15, 17.2798}, {20, 16.6642}, {25, 16.1793}, {30, 15.7766}, {35, 15.4295}, {40, 15.1242}, {45, 14.8502}, {50, 14.6007},
    }},
    {"HeMcs2_40MHz", {
        {5, 25.6338}, {10, 23.8255}, {15, 22.7329}, {20, 21.9442}, {25, 21.3200}, {30, 20.7999}, {35, 20.3506}, {40, 19.9549}, {45, 19.5990}, {50, 19.2746},
    }},
    {"HeMcs3_40MHz", {
        {5, 30.0542}, {10, 28.0155}, {15, 26.7625}, {20, 25.8523}, {25, 25.1295}, {30, 24.5258}, {35, 24.0034}, {40, 23.5426}, {45, 23.1277}, {50, 22.7492},
    }},
    {"HeMcs4_40MHz", {
        {5, 37.6051}, {10, 35.2296}, {15, 33.7228}, {20, 32.6160}, {25, 31.7314}, {30, 30.9895}, {35, 30.3455}, {40, 29.7760}, {45, 29.2623}, {50, 28.7929},
    }},
    {"HeMcs5_40MHz", {
        {5, 41.8065}, {10, 39.2749}, {15, 37.6383}, {20, 36.4282}, {25, 35.4575}, {30, 34.6414}, {35, 33.9316}, {40, 33.3031}, {45, 32.7355}, {50, 32.2164},
    }},
    {"HeMcs6_40MHz", {
        {5, 44.2801}, {10, 41.6672}, {15, 39.9580}, {20, 38.6892}, {25, 37.6692}, {30, 36.8103}, {35, 36.0625}, {40, 35.3998}, {45, 34.8008}, {50, 34.2528},
    }},
    {"HeMcs7_40MHz", {
        {5, 44.2801}, {10, 41.6672}, {15, 39.9580}, {20, 38.6892}, {25, 37.6692}, {30, 36.8103}, {35, 36.0625}, {40, 35.3998}, {45, 34.8008}, {50, 34.2528},
    }},
    {"HeMcs8_40MHz", {
        {5, 47.0648}, {10, 44.3699}, {15, 42.5825}, {20, 41.2495}, {25, 40.1751}, {30, 39.2689}, {35, 38.4790}, {40, 37.7781}, {45, 37.1443}, {50, 36.5639},
    }},
    {"HeMcs9_40MHz", {
        {5, 50.2233}, {10, 47.4474}, {15, 45.5760}, {20, 44.1727}, {25, 43.0382}, {30, 42.0794}, {35, 41.2425}, {40, 40.4991}, {45, 39.8262}, {50, 39.2095},
    }},
    {"HeMcs10_40MHz", {
        {5, 50.2233}, {10, 47.4474}, {15, 45.5760}, {20, 44.1727}, {25, 43.0382}, {30, 42.0794}, {35, 41.2425}, {40, 40.4991}, {45, 39.8262}, {50, 39.2095},
    }},
    {"HeMcs11_40MHz", {
        {5, 50.2233}, {10, 47.4474}, {15, 45.5760}, {20, 44.1727}, {25, 43.0382}, {30, 42.0794}, {35, 41.2425}, {40, 40.4991}, {45, 39.8262}, {50, 39.2095},
    }},
    {"HeMcs0_80MHz", {
        {5, 19.6542}, {10, 18.1962}, {15, 17.3342}, {20, 16.7168}, {25, 16.2305}, {30, 15.8265}, {35, 15.4784}, {40, 15.1723}, {45, 14.8973}, {50, 14.6471},
    }},
    {"HeMcs1_80MHz", {
        {5, 30.9311}, {10, 28.8495}, {15, 27.5657}, {20, 26.6320}, {25, 25.8899}, {30, 25.2699}, {35, 24.7332}, {40, 24.2595}, {45, 23.8330}, {50, 23.4439},
    }},
    {"HeMcs2_80MHz", {
        {5, 37.0575}, {10, 34.7039}, {15, 33.2146}, {20, 32.1216}, {25, 31.2485}, {30, 30.5164}, {35, 29.8811}, {40, 29.3194}, {45, 28.8127}, {50, 28.3499},
    }},
    {"HeMcs3_80MHz", {
        {5, 41.8065}, {10, 39.2749}, {15, 37.6383}, {20, 36.4282}, {25, 35.4575}, {30, 34.6414}, {35, 33.9316}, {40, 33.3031}, {45, 32.7355}, {50, 32.2164},
    }},
    {"HeMcs4_80MHz", {
        {5, 47.0648}, {10, 44.3699}, {15, 42.5825}, {20, 41.2495}, {25, 40.1751}, {30, 39.2689}, {35, 38.4790}, {40, 37.7781}, {45, 37.1443}, {50, 36.5639},
    }},
    {"HeMcs5_80MHz", {
        {5, 50.2233}, {10, 47.4474}, {15, 45.5760}, {20, 44.1727}, {25, 43.0382}, {30, 42.0794}, {35, 41.2425}, {40, 40.4991}, {45, 39.8262}, {50, 39.2095},
    }},
    {"HeMcs6_80MHz", {
        {5, 53.8362}, {10, 50.9837}, {15, 49.0221}, {20, 47.5418}, {25, 46.3407}, {30, 45.3233}, {35, 44.4337}, {40, 43.6425}, {45, 42.9255}, {50, 42.2678},
    }},
    {"HeMcs7_80MHz", {
        {5, 53.8362}, {10, 50.9837}, {15, 49.0221}, {20, 47.5418}, {25, 46.3407}, {30, 45.3233}, {35, 44.4337}, {40, 43.6425}, {45, 42.9255}, {50, 42.2678},
    }},
    {"HeMcs8_80MHz", {
        {5, 53.8362}, {10, 50.9837}, {15, 49.0221}, {20, 47.5418}, {25, 46.3407}, {30, 45.3233}, {35, 44.4337}, {40, 43.6425}, {45, 42.9255}, {50, 42.2678},
    }},
    {"HeMcs9_80MHz", {
        {5, 58.0092}, {10, 55.0896}, {15, 53.0321}, {20, 51.4672}, {25, 50.1922}, {30, 49.1091}, {35, 48.1601}, {40, 47.3148}, {45, 46.5478}, {50, 45.8436},
    }},
    {"HeMcs10_80MHz", {
        {5, 58.0092}, {10, 55.0896}, {15, 53.0321}, {20, 51.4672}, {25, 50.1922}, {30, 49.1091}, {35, 48.1601}, {40, 47.3148}, {45, 46.5478}, {50, 45.8436},
    }},
    {"HeMcs11_80MHz", {
        {5, 58.0092}, {10, 55.0896}, {15, 53.0321}, {20, 51.4672}, {25, 50.1922}, {30, 49.1091}, {35, 48.1601}, {40, 47.3148}, {45, 46.5478}, {50, 45.8436},
    }},
    {"HeMcs0_160MHz", {
        {5, 29.8428}, {10, 27.8145}, {15, 26.5689}, {20, 25.6645}, {25, 24.9463}, {30, 24.3466}, {35, 23.8276}, {40, 23.3699}, {45, 22.9578}, {50, 22.5819},
    }},
    {"HeMcs1_160MHz", {
        {5, 41.1308}, {10, 38.6227}, {15, 37.0064}, {20, 35.8126}, {25, 34.8556}, {30, 34.0513}, {35, 33.3520}, {40, 32.7329}, {45, 32.1739}, {50, 31.6628},
    }},
    {"HeMcs2_160MHz", {
        {5, 46.2101}, {10, 43.5393}, {15, 41.7755}, {20, 40.4620}, {25, 39.4041}, {30, 38.5123}, {35, 37.7353}, {40, 37.0461}, {45, 36.4229}, {50, 35.8524},
    }},
    {"HeMcs3_160MHz", {
        {5, 50.2233}, {10, 47.4474}, {15, 45.5760}, {20, 44.1727}, {25, 43.0382}, {30, 42.0794}, {35, 41.2425}, {40, 40.4991}, {45, 39.8262}, {50, 39.2095},
    }},
    {"HeMcs4_160MHz", {
        {5, 53.8362}, {10, 50.9837}, {15, 49.0221}, {20, 47.5418}, {25, 46.3407}, {30, 45.3233}, {35, 44.4337}, {40, 43.6425}, {45, 42.9255}, {50, 42.2678},
    }},
    {"HeMcs5_160MHz", {
        {5, 58.0092}, {10, 55.0896}, {15, 53.0321}, {20, 51.4672}, {25, 50.1922}, {30, 49.1091}, {35, 48.1601}, {40, 47.3148}, {45, 46.5478}, {50, 45.8436},
    }},
    {"HeMcs6_160MHz", {
        {5, 58.0092}, {10, 55.0896}, {15, 53.0321}, {20, 51.4672}, {25, 50.1922}, {30, 49.1091}, {35, 48.1601}, {40, 47.3148}, {45, 46.5478}, {50, 45.8436},
    }},
    {"HeMcs7_160MHz", {
        {5, 58.0092}, {10, 55.0896}, {15, 53.0321}, {20, 51.4672}, {25, 50.1922}, {30, 49.1091}, {35, 48.1601}, {40, 47.3148}, {45, 46.5478}, {50, 45.8436},
    }},
    {"HeMcs8_160MHz", {
        {5, 58.0092}, {10, 55.0896}, {15, 53.0321}, {20, 51.4672}, {25, 50.1922}, {30, 49.1091}, {35, 48.1601}, {40, 47.3148}, {45, 46.5478}, {50, 45.8436},
    }},
    {"HeMcs9_160MHz", {
        {5, 62.8834}, {10, 59.9147}, {15, 57.7564}, {20, 56.0992}, {25, 54.7419}, {30, 53.5850}, {35, 52.5689}, {40, 51.6620}, {45, 50.8379}, {50, 50.0803},
    }},
    {"HeMcs10_160MHz", {
        {5, 62.8834}, {10, 59.9147}, {15, 57.7564}, {20, 56.0992}, {25, 54.7419}, {30, 53.5850}, {35, 52.5689}, {40, 51.6620}, {45, 50.8379}, {50, 50.0803},
    }},
    {"HeMcs11_160MHz", {
        {5, 62.8834}, {10, 59.9147}, {15, 57.7564}, {20, 56.0992}, {25, 54.7419}, {30, 53.5850}, {35, 52.5689}, {40, 51.6620}, {45, 50.8379}, {50, 50.0803},
    }},
};

std::map<std::string /* mode */, std::map<unsigned int /* number of nodes */, double /* calculated throughput */> > bianchiResultsDifs =
{
/* 11b */
    {"DsssRate1Mbps", {
        {5, 0.8437}, {10, 0.7861}, {15, 0.7496}, {20, 0.7226}, {25, 0.7016}, {30, 0.6847}, {35, 0.6686}, {40, 0.6549}, {45, 0.6435}, {50, 0.6336},
    }},
    {"DsssRate2Mbps", {
        {5, 1.6228}, {10, 1.5168}, {15, 1.4482}, {20, 1.3972}, {25, 1.3574}, {30, 1.3253}, {35, 1.2947}, {40, 1.2687}, {45, 1.2469}, {50, 1.2279},
    }},
    {"DsssRate5_5Mbps", {
        {5, 3.8896}, {10, 3.6707}, {15, 3.5203}, {20, 3.4063}, {25, 3.3161}, {30, 3.2429}, {35, 3.1729}, {40, 3.1128}, {45, 3.0625}, {50, 3.0184},
    }},
    {"DsssRate11Mbps", {
        {5, 6.4734}, {10, 6.1774}, {15, 5.9553}, {20, 5.7819}, {25, 5.6429}, {30, 5.5289}, {35, 5.4191}, {40, 5.3243}, {45, 5.2446}, {50, 5.1745},
    }},
/* 11a */
    {"OfdmRate6Mbps", {
        {5, 4.7087}, {10, 4.3453}, {15, 4.1397}, {20, 3.9899}, {25, 3.8802}, {30, 3.7824}, {35, 3.6961}, {40, 3.6276}, {45, 3.5712}, {50, 3.5071},
    }},
    {"OfdmRate9Mbps", {
        {5, 6.8586}, {10, 6.3431}, {15, 6.0489}, {20, 5.8340}, {25, 5.6762}, {30, 5.5355}, {35, 5.4110}, {40, 5.3122}, {45, 5.2307}, {50, 5.1380},
    }},
    {"OfdmRate12Mbps", {
        {5, 8.9515}, {10, 8.2901}, {15, 7.9102}, {20, 7.6319}, {25, 7.4274}, {30, 7.2447}, {35, 7.0829}, {40, 6.9544}, {45, 6.8485}, {50, 6.7278},
    }},
    {"OfdmRate18Mbps", {
        {5, 12.7822}, {10, 11.8801}, {15, 11.3543}, {20, 10.9668}, {25, 10.6809}, {30, 10.4249}, {35, 10.1978}, {40, 10.0171}, {45, 9.8679}, {50, 9.6978},
    }},
    {"OfdmRate24Mbps", {
        {5, 16.2470}, {10, 15.1426}, {15, 14.4904}, {20, 14.0072}, {25, 13.6496}, {30, 13.3288}, {35, 13.0436}, {40, 12.8164}, {45, 12.6286}, {50, 12.4144},
    }},
    {"OfdmRate36Mbps", {
        {5, 22.3164}, {10, 20.9147}, {15, 20.0649}, {20, 19.4289}, {25, 18.9552}, {30, 18.5284}, {35, 18.1476}, {40, 17.8434}, {45, 17.5915}, {50, 17.3036},
    }},
    {"OfdmRate48Mbps", {
        {5, 27.2963}, {10, 25.6987}, {15, 24.7069}, {20, 23.9578}, {25, 23.3965}, {30, 22.8891}, {35, 22.4350}, {40, 22.0713}, {45, 21.7696}, {50, 21.4243},
    }},
    {"OfdmRate54Mbps", {
        {5, 29.8324}, {10, 28.1519}, {15, 27.0948}, {20, 26.2925}, {25, 25.6896}, {30, 25.1434}, {35, 24.6539}, {40, 24.2613}, {45, 23.9353}, {50, 23.5618},
    }},
/* 11g */
    {"ErpOfdmRate6Mbps", {
        {5, 4.7087}, {10, 4.3453}, {15, 4.1397}, {20, 3.9899}, {25, 3.8802}, {30, 3.7824}, {35, 3.6961}, {40, 3.6276}, {45, 3.5712}, {50, 3.5071},
    }},
    {"ErpOfdmRate9Mbps", {
        {5, 6.8586}, {10, 6.3431}, {15, 6.0489}, {20, 5.8340}, {25, 5.6762}, {30, 5.5355}, {35, 5.4110}, {40, 5.3122}, {45, 5.2307}, {50, 5.1380},
    }},
    {"ErpOfdmRate12Mbps", {
        {5, 8.9515}, {10, 8.2901}, {15, 7.9102}, {20, 7.6319}, {25, 7.4274}, {30, 7.2447}, {35, 7.0829}, {40, 6.9544}, {45, 6.8485}, {50, 6.7278},
    }},
    {"ErpOfdmRate18Mbps", {
        {5, 12.7822}, {10, 11.8801}, {15, 11.3543}, {20, 10.9668}, {25, 10.6809}, {30, 10.4249}, {35, 10.1978}, {40, 10.0171}, {45, 9.8679}, {50, 9.6978},
    }},
    {"ErpOfdmRate24Mbps", {
        {5, 16.2470}, {10, 15.1426}, {15, 14.4904}, {20, 14.0072}, {25, 13.6496}, {30, 13.3288}, {35, 13.0436}, {40, 12.8164}, {45, 12.6286}, {50, 12.4144},
    }},
    {"ErpOfdmRate36Mbps", {
        {5, 22.3164}, {10, 20.9147}, {15, 20.0649}, {20, 19.4289}, {25, 18.9552}, {30, 18.5284}, {35, 18.1476}, {40, 17.8434}, {45, 17.5915}, {50, 17.3036},
    }},
    {"ErpOfdmRate48Mbps", {
        {5, 27.2963}, {10, 25.6987}, {15, 24.7069}, {20, 23.9578}, {25, 23.3965}, {30, 22.8891}, {35, 22.4350}, {40, 22.0713}, {45, 21.7696}, {50, 21.4243},
    }},
    {"ErpOfdmRate54Mbps", {
        {5, 29.8324}, {10, 28.1519}, {15, 27.0948}, {20, 26.2925}, {25, 25.6896}, {30, 25.1434}, {35, 24.6539}, {40, 24.2613}, {45, 23.9353}, {50, 23.5618},
    }},
/* 11ax, no frame aggregation */
    {"HeMcs0_20MHz", {
        {5, 6.3746}, {10, 5.8670}, {15, 5.5782}, {20, 5.3742}, {25, 5.2147}, {30, 5.0829}, {35, 4.9696}, {40, 4.8703}, {45, 4.7813}, {50, 4.7004},
    }},
    {"HeMcs1_20MHz", {
        {5, 11.7574}, {10, 10.8735}, {15, 10.3606}, {20, 9.9954}, {25, 9.7084}, {30, 9.4704}, {35, 9.2654}, {40, 9.0853}, {45, 8.9235}, {50, 8.7763},
    }},
    {"HeMcs2_20MHz", {
        {5, 16.0419}, {10, 14.8998}, {15, 14.2252}, {20, 13.7413}, {25, 13.3594}, {30, 13.0417}, {35, 12.7674}, {40, 12.5258}, {45, 12.3086}, {50, 12.1107},
    }},
    {"HeMcs3_20MHz", {
        {5, 20.0089}, {10, 18.6480}, {15, 17.8309}, {20, 17.2410}, {25, 16.7736}, {30, 16.3837}, {35, 16.0465}, {40, 15.7491}, {45, 15.4813}, {50, 15.2369},
    }},
    {"HeMcs4_20MHz", {
        {5, 26.3492}, {10, 24.7107}, {15, 23.6964}, {20, 22.9553}, {25, 22.3640}, {30, 21.8683}, {35, 21.4379}, {40, 21.0571}, {45, 20.7134}, {50, 20.3991},
    }},
    {"HeMcs5_20MHz", {
        {5, 30.6683}, {10, 28.8843}, {15, 27.7540}, {20, 26.9210}, {25, 26.2528}, {30, 25.6906}, {35, 25.2012}, {40, 24.7671}, {45, 24.3746}, {50, 24.0151},
    }},
    {"HeMcs6_20MHz", {
        {5, 33.4062}, {10, 31.5485}, {15, 30.3527}, {20, 29.4662}, {25, 28.7527}, {30, 28.1508}, {35, 27.6259}, {40, 27.1597}, {45, 26.7376}, {50, 26.3507},
    }},
    {"HeMcs7_20MHz", {
        {5, 34.9671}, {10, 33.0739}, {15, 31.8436}, {20, 30.9282}, {25, 30.1900}, {30, 29.5665}, {35, 29.0221}, {40, 28.5382}, {45, 28.0997}, {50, 27.6975},
    }},
    {"HeMcs8_20MHz", {
        {5, 38.5714}, {10, 36.6144}, {15, 35.3124}, {20, 34.3355}, {25, 33.5438}, {30, 32.8728}, {35, 32.2854}, {40, 31.7623}, {45, 31.2874}, {50, 30.8512},
    }},
    {"HeMcs9_20MHz", {
        {5, 40.6674}, {10, 38.6851}, {15, 37.3466}, {20, 36.3371}, {25, 35.5165}, {30, 34.8197}, {35, 34.2087}, {40, 33.6638}, {45, 33.1688}, {50, 32.7137},
    }},
    {"HeMcs10_20MHz", {
        {5, 40.6674}, {10, 38.6851}, {15, 37.3466}, {20, 36.3371}, {25, 35.5165}, {30, 34.8197}, {35, 34.2087}, {40, 33.6638}, {45, 33.1688}, {50, 32.7137},
    }},
    {"HeMcs11_20MHz", {
        {5, 43.0043}, {10, 41.0039}, {15, 39.6294}, {20, 38.5865}, {25, 37.7358}, {30, 37.0116}, {35, 36.3756}, {40, 35.8076}, {45, 35.2909}, {50, 34.8154},
    }},
    {"HeMcs0_40MHz", {
        {5, 11.6208}, {10, 10.7566}, {15, 10.2544}, {20, 9.8965}, {25, 9.6151}, {30, 9.3815}, {35, 9.1804}, {40, 9.0035}, {45, 8.8446}, {50, 8.7000},
    }},
    {"HeMcs1_40MHz", {
        {5, 19.8764}, {10, 18.5328}, {15, 17.7255}, {20, 17.1424}, {25, 16.6803}, {30, 16.2947}, {35, 15.9612}, {40, 15.6668}, {45, 15.4018}, {50, 15.1599},
    }},
    {"HeMcs2_40MHz", {
        {5, 26.1198}, {10, 24.5088}, {15, 23.5107}, {20, 22.7810}, {25, 22.1986}, {30, 21.7101}, {35, 21.2858}, {40, 20.9104}, {45, 20.5714}, {50, 20.2613},
    }},
    {"HeMcs3_40MHz", {
        {5, 30.6683}, {10, 28.8843}, {15, 27.7540}, {20, 26.9210}, {25, 26.2528}, {30, 25.6906}, {35, 25.2012}, {40, 24.7671}, {45, 24.3746}, {50, 24.0151},
    }},
    {"HeMcs4_40MHz", {
        {5, 38.5714}, {10, 36.6144}, {15, 35.3124}, {20, 34.3355}, {25, 33.5438}, {30, 32.8728}, {35, 32.2854}, {40, 31.7623}, {45, 31.2874}, {50, 30.8512},
    }},
    {"HeMcs5_40MHz", {
        {5, 43.0043}, {10, 41.0039}, {15, 39.6294}, {20, 38.5865}, {25, 37.7358}, {30, 37.0116}, {35, 36.3756}, {40, 35.8076}, {45, 35.2909}, {50, 34.8154},
    }},
    {"HeMcs6_40MHz", {
        {5, 45.6261}, {10, 43.6185}, {15, 42.2095}, {20, 41.1328}, {25, 40.2509}, {30, 39.4981}, {35, 38.8356}, {40, 38.2430}, {45, 37.7032}, {50, 37.2058},
    }},
    {"HeMcs7_40MHz", {
        {5, 45.6261}, {10, 43.6185}, {15, 42.2095}, {20, 41.1328}, {25, 40.2509}, {30, 39.4981}, {35, 38.8356}, {40, 38.2430}, {45, 37.7032}, {50, 37.2058},
    }},
    {"HeMcs8_40MHz", {
        {5, 48.5883}, {10, 46.5892}, {15, 45.1489}, {20, 44.0388}, {25, 43.1252}, {30, 42.3428}, {35, 41.6525}, {40, 41.0338}, {45, 40.4694}, {50, 39.9486},
    }},
    {"HeMcs9_40MHz", {
        {5, 51.9619}, {10, 49.9941}, {15, 48.5284}, {20, 47.3867}, {25, 46.4416}, {30, 45.6290}, {35, 44.9099}, {40, 44.2640}, {45, 43.6736}, {50, 43.1279},
    }},
    {"HeMcs10_40MHz", {
        {5, 51.9619}, {10, 49.9941}, {15, 48.5284}, {20, 47.3867}, {25, 46.4416}, {30, 45.6290}, {35, 44.9099}, {40, 44.2640}, {45, 43.6736}, {50, 43.1279},
    }},
    {"HeMcs11_40MHz", {
        {5, 51.9619}, {10, 49.9941}, {15, 48.5284}, {20, 47.3867}, {25, 46.4416}, {30, 45.6290}, {35, 44.9099}, {40, 44.2640}, {45, 43.6736}, {50, 43.1279},
    }},
    {"HeMcs0_80MHz", {
        {5, 20.0101}, {10, 18.6928}, {15, 17.8976}, {20, 17.3219}, {25, 16.8648}, {30, 16.4830}, {35, 16.1523}, {40, 15.8603}, {45, 15.5971}, {50, 15.3567},
    }},
    {"HeMcs1_80MHz", {
        {5, 31.6415}, {10, 29.8575}, {15, 28.7177}, {20, 27.8747}, {25, 27.1971}, {30, 26.6261}, {35, 26.1283}, {40, 25.6865}, {45, 25.2866}, {50, 24.9200},
    }},
    {"HeMcs2_80MHz", {
        {5, 38.0818}, {10, 36.1730}, {15, 34.9016}, {20, 33.9470}, {25, 33.1729}, {30, 32.5165}, {35, 31.9417}, {40, 31.4295}, {45, 30.9645}, {50, 30.5372},
    }},
    {"HeMcs3_80MHz", {
        {5, 43.0043}, {10, 41.0039}, {15, 39.6294}, {20, 38.5865}, {25, 37.7358}, {30, 37.0116}, {35, 36.3756}, {40, 35.8076}, {45, 35.2909}, {50, 34.8154},
    }},
    {"HeMcs4_80MHz", {
        {5, 48.5883}, {10, 46.5892}, {15, 45.1489}, {20, 44.0388}, {25, 43.1252}, {30, 42.3428}, {35, 41.6525}, {40, 41.0338}, {45, 40.4694}, {50, 39.9486},
    }},
    {"HeMcs5_80MHz", {
        {5, 51.9619}, {10, 49.9941}, {15, 48.5284}, {20, 47.3867}, {25, 46.4416}, {30, 45.6290}, {35, 44.9099}, {40, 44.2640}, {45, 43.6736}, {50, 43.1279},
    }},
    {"HeMcs6_80MHz", {
        {5, 55.8389}, {10, 53.9360}, {15, 52.4548}, {20, 51.2855}, {25, 50.3106}, {30, 49.4682}, {35, 48.7201}, {40, 48.0462}, {45, 47.4288}, {50, 46.8571},
    }},
    {"HeMcs7_80MHz", {
        {5, 55.8389}, {10, 53.9360}, {15, 52.4548}, {20, 51.2855}, {25, 50.3106}, {30, 49.4682}, {35, 48.7201}, {40, 48.0462}, {45, 47.4288}, {50, 46.8571},
    }},
    {"HeMcs8_80MHz", {
        {5, 55.8389}, {10, 53.9360}, {15, 52.4548}, {20, 51.2855}, {25, 50.3106}, {30, 49.4682}, {35, 48.7201}, {40, 48.0462}, {45, 47.4288}, {50, 46.8571},
    }},
    {"HeMcs9_80MHz", {
        {5, 60.3411}, {10, 58.5527}, {15, 57.0724}, {20, 55.8834}, {25, 54.8827}, {30, 54.0128}, {35, 53.2368}, {40, 52.5352}, {45, 51.8906}, {50, 51.2922},
    }},
    {"HeMcs10_80MHz", {
        {5, 60.3411}, {10, 58.5527}, {15, 57.0724}, {20, 55.8834}, {25, 54.8827}, {30, 54.0128}, {35, 53.2368}, {40, 52.5352}, {45, 51.8906}, {50, 51.2922},
    }},
    {"HeMcs11_80MHz", {
        {5, 60.3411}, {10, 58.5527}, {15, 57.0724}, {20, 55.8834}, {25, 54.8827}, {30, 54.0128}, {35, 53.2368}, {40, 52.5352}, {45, 51.8906}, {50, 51.2922},
    }},
    {"HeMcs0_160MHz", {
        {5, 30.6710}, {10, 28.9919}, {15, 27.9160}, {20, 27.1188}, {25, 26.4770}, {30, 25.9355}, {35, 25.4630}, {40, 25.0432}, {45, 24.6629}, {50, 24.3141},
    }},
    {"HeMcs1_160MHz", {
        {5, 42.3965}, {10, 40.4510}, {15, 39.1127}, {20, 38.0965}, {25, 37.2670}, {30, 36.5606}, {35, 35.9398}, {40, 35.3852}, {45, 34.8806}, {50, 34.4160},
    }},
    {"HeMcs2_160MHz", {
        {5, 47.8139}, {10, 45.8767}, {15, 44.4795}, {20, 43.4017}, {25, 42.5141}, {30, 41.7535}, {35, 41.0821}, {40, 40.4801}, {45, 39.9307}, {50, 39.4236},
    }},
    {"HeMcs3_160MHz", {
        {5, 51.9619}, {10, 49.9941}, {15, 48.5284}, {20, 47.3867}, {25, 46.4416}, {30, 45.6290}, {35, 44.9099}, {40, 44.2640}, {45, 43.6736}, {50, 43.1279},
    }},
    {"HeMcs4_160MHz", {
        {5, 55.8389}, {10, 53.9360}, {15, 52.4548}, {20, 51.2855}, {25, 50.3106}, {30, 49.4682}, {35, 48.7201}, {40, 48.0462}, {45, 47.4288}, {50, 46.8571},
    }},
    {"HeMcs5_160MHz", {
        {5, 60.3411}, {10, 58.5527}, {15, 57.0724}, {20, 55.8834}, {25, 54.8827}, {30, 54.0128}, {35, 53.2368}, {40, 52.5352}, {45, 51.8906}, {50, 51.2922},
    }},
    {"HeMcs6_160MHz", {
        {5, 60.3411}, {10, 58.5527}, {15, 57.0724}, {20, 55.8834}, {25, 54.8827}, {30, 54.0128}, {35, 53.2368}, {40, 52.5352}, {45, 51.8906}, {50, 51.2922},
    }},
    {"HeMcs7_160MHz", {
        {5, 60.3411}, {10, 58.5527}, {15, 57.0724}, {20, 55.8834}, {25, 54.8827}, {30, 54.0128}, {35, 53.2368}, {40, 52.5352}, {45, 51.8906}, {50, 51.2922},
    }},
    {"HeMcs8_160MHz", {
        {5, 60.3411}, {10, 58.5527}, {15, 57.0724}, {20, 55.8834}, {25, 54.8827}, {30, 54.0128}, {35, 53.2368}, {40, 52.5352}, {45, 51.8906}, {50, 51.2922},
    }},
    {"HeMcs9_160MHz", {
        {5, 65.6329}, {10, 64.0336}, {15, 62.5814}, {20, 61.3869}, {25, 60.3690}, {30, 59.4769}, {35, 58.6764}, {40, 57.9495}, {45, 57.2790}, {50, 56.6548},
    }},
    {"HeMcs10_160MHz", {
        {5, 65.6329}, {10, 64.0336}, {15, 62.5814}, {20, 61.3869}, {25, 60.3690}, {30, 59.4769}, {35, 58.6764}, {40, 57.9495}, {45, 57.2790}, {50, 56.6548},
    }},
    {"HeMcs11_160MHz", {
        {5, 65.6329}, {10, 64.0336}, {15, 62.5814}, {20, 61.3869}, {25, 60.3690}, {30, 59.4769}, {35, 58.6764}, {40, 57.9495}, {45, 57.2790}, {50, 56.6548},
    }},
};

// Parse context strings of the form "/NodeList/x/DeviceList/x/..." to extract the NodeId integer
uint32_t
ContextToNodeId (std::string context)
{
  std::string sub = context.substr (10);
  uint32_t pos = sub.find ("/Device");
  return atoi (sub.substr (0, pos).c_str ());
}

// Parse context strings of the form "/NodeList/x/DeviceList/x/..." and fetch the Mac address
Mac48Address
ContextToMac (std::string context)
{
  std::string sub = context.substr (10);
  uint32_t pos = sub.find ("/Device");
  uint32_t nodeId = atoi (sub.substr (0, pos).c_str ());
  Ptr<Node> n = NodeList::GetNode (nodeId);
  Ptr<WifiNetDevice> d;
  for (uint32_t i = 0; i < n->GetNDevices (); i++)
    {
      d = n->GetDevice (i)->GetObject<WifiNetDevice> ();
      if (d)
        {
          break;
        }
    }
  return Mac48Address::ConvertFrom (d->GetAddress ());
}

// Functions for tracing.

void
IncrementCounter (std::map<Mac48Address, uint64_t> & counter, Mac48Address addr, uint64_t increment = 1)
{
  auto it = counter.find (addr);
  if (it != counter.end ())
   {
     it->second += increment;
   }
  else
   {
     counter.insert (std::make_pair (addr, increment));
   }
}

void
TracePacketReception (std::string context, Ptr<const Packet> p, uint16_t channelFreqMhz, WifiTxVector txVector, MpduInfo aMpdu, SignalNoiseDbm signalNoise, uint16_t staId)
{
  Ptr<Packet> packet = p->Copy ();
  if (txVector.IsAggregation ())
    {
      AmpduSubframeHeader subHdr;
      uint32_t extractedLength;
      packet->RemoveHeader (subHdr);
      extractedLength = subHdr.GetLength ();
      packet = packet->CreateFragment (0, static_cast<uint32_t> (extractedLength));
    }
  WifiMacHeader hdr;
  packet->PeekHeader (hdr);
  // hdr.GetAddr1() is the receiving MAC address
  if (hdr.GetAddr1 () != ContextToMac (context))
    {
      return;
    }
  // hdr.GetAddr2() is the sending MAC address
  if (packet->GetSize () >= pktSize) // ignore non-data frames
    {
      IncrementCounter (packetsReceived, hdr.GetAddr2 ());
      IncrementCounter (bytesReceived, hdr.GetAddr2 (), pktSize);
      auto itTimeFirstReceived = timeFirstReceived.find (hdr.GetAddr2 ());
      if (itTimeFirstReceived == timeFirstReceived.end ())
        {
          timeFirstReceived.insert (std::make_pair (hdr.GetAddr2 (), Simulator::Now ()));
        }
      auto itTimeLastReceived = timeLastReceived.find (hdr.GetAddr2 ());
      if (itTimeLastReceived != timeLastReceived.end ())
        {
          itTimeLastReceived->second = Simulator::Now ();
        }
      else
        {
          timeLastReceived.insert (std::make_pair (hdr.GetAddr2 (), Simulator::Now ()));
        }
    }
}

void
CwTrace (std::string context, uint32_t oldVal, uint32_t newVal)
{
  NS_LOG_INFO ("CW time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " val=" << newVal);
  if (tracing)
    {
      cwTraceFile << Simulator::Now ().GetSeconds () << " " << ContextToNodeId (context) << " " << newVal << std::endl;
    }
}

void
BackoffTrace (std::string context, uint32_t newVal)
{
  NS_LOG_INFO ("Backoff time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " val=" << newVal);
  if (tracing)
    {
      backoffTraceFile << Simulator::Now ().GetSeconds () << " " << ContextToNodeId (context) << " " << newVal << std::endl;
    }
}

void
PhyRxTrace (std::string context, Ptr<const Packet> p, RxPowerWattPerChannelBand power)
{
  NS_LOG_INFO ("PHY-RX-START time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " size=" << p->GetSize ());
}

void
PhyRxPayloadTrace (std::string context, WifiTxVector txVector, Time psduDuration)
{
  NS_LOG_INFO ("PHY-RX-PAYLOAD-START time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " psduDuration=" << psduDuration);
}

void
PhyRxDropTrace (std::string context, Ptr<const Packet> p, WifiPhyRxfailureReason reason)
{
  NS_LOG_INFO ("PHY-RX-DROP time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " size=" << p->GetSize () << " reason=" << reason);
  Mac48Address addr = ContextToMac (context);
  switch (reason)
    {
    case UNSUPPORTED_SETTINGS:
      NS_FATAL_ERROR ("RX packet with unsupported settings!");
      break;
    case CHANNEL_SWITCHING:
      NS_FATAL_ERROR ("Channel is switching!");
      break;
    case BUSY_DECODING_PREAMBLE:
    {
      if (p->GetSize () >= pktSize) // ignore non-data frames
        {
          IncrementCounter (rxEventWhileDecodingPreamble, addr);
        }
      break;
    }
    case RXING:
    {
      if (p->GetSize () >= pktSize) // ignore non-data frames
        {
          IncrementCounter (rxEventWhileRxing, addr);
        }
      break;
    }
    case TXING:
    {
      if (p->GetSize () >= pktSize) // ignore non-data frames
        {
          IncrementCounter (rxEventWhileTxing, addr);
        }
      break;
    }
    case SLEEPING:
      NS_FATAL_ERROR ("Device is sleeping!");
      break;
    case PREAMBLE_DETECT_FAILURE:
      NS_FATAL_ERROR ("Preamble should always be detected!");
      break;
    case RECEPTION_ABORTED_BY_TX:
    {
      if (p->GetSize () >= pktSize) // ignore non-data frames
        {
          IncrementCounter (rxEventAbortedByTx, addr);
        }
      break;
    }
    case L_SIG_FAILURE:
    {
      if (p->GetSize () >= pktSize) // ignore non-data frames
        {
          IncrementCounter (phyHeaderFailed, addr);
        }
      break;
    }
    case HT_SIG_FAILURE:
    case SIG_A_FAILURE:
    case SIG_B_FAILURE:
      NS_FATAL_ERROR ("Unexpected PHY header failure!");
    case PREAMBLE_DETECTION_PACKET_SWITCH:
      NS_FATAL_ERROR ("All devices should send with same power, so no packet switch during preamble detection should occur!");
      break;
    case FRAME_CAPTURE_PACKET_SWITCH:
      NS_FATAL_ERROR ("Frame capture should be disabled!");
      break;
    case OBSS_PD_CCA_RESET:
      NS_FATAL_ERROR ("Unexpected CCA reset!");
      break;
    case UNKNOWN:
    default:
      NS_FATAL_ERROR ("Unknown drop reason!");
      break;
    }
}

void
PhyRxDoneTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_INFO ("PHY-RX-END time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " size=" << p->GetSize ());
}

void
PhyRxOkTrace (std::string context, Ptr<const Packet> p, double snr, WifiMode mode, WifiPreamble preamble)
{
  uint8_t nMpdus = (p->GetSize () / pktSize);
  NS_LOG_INFO ("PHY-RX-OK time=" << Simulator::Now ().As (Time::S)
                                 << " node=" << ContextToNodeId (context)
                                 << " size=" << p->GetSize ()
                                 << " nMPDUs=" << +nMpdus
                                 << " snr=" << snr
                                 << " mode=" << mode
                                 << " preamble=" << preamble);
  if ((maxMpdus != 0) && (nMpdus != 0) && (nMpdus != maxMpdus))
    {
      if (nMpdus > maxMpdus)
        {
          NS_FATAL_ERROR ("A-MPDU settings not properly applied: maximum configured MPDUs is " << +maxMpdus << " but received an A-MPDU containing " << +nMpdus << " MPDUs");
        }
      NS_LOG_WARN ("Warning: less MPDUs aggregated in a received A-MPDU (" << +nMpdus << ") than configured (" << +maxMpdus << ")");
    }
  if (p->GetSize () >= pktSize) // ignore non-data frames
    {
      Mac48Address addr = ContextToMac (context);
      IncrementCounter (psduSucceeded, addr);
    }
}

void
PhyRxErrorTrace (std::string context, Ptr<const Packet> p, double snr)
{
  NS_LOG_INFO ("PHY-RX-ERROR time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " size=" << p->GetSize () << " snr=" << snr);
  if (p->GetSize () >= pktSize) // ignore non-data frames
    {
      Mac48Address addr = ContextToMac (context);
      IncrementCounter (psduFailed, addr);
    }
}

void
PhyTxTrace (std::string context, Ptr<const Packet> p, double txPowerW)
{
  NS_LOG_INFO ("PHY-TX-START time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " size=" << p->GetSize () << " " << txPowerW);
  if (tracing)
    {
      phyTxTraceFile << Simulator::Now ().GetSeconds () << " " << ContextToNodeId (context) << " size=" << p->GetSize () << " " << txPowerW << std::endl;
    }
  if (p->GetSize () >= pktSize) // ignore non-data frames
    {
      Mac48Address addr = ContextToMac (context);
      IncrementCounter (packetsTransmitted, addr);
    }
}

void
PhyTxDoneTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_INFO ("PHY-TX-END time=" << Simulator::Now () << " node=" << ContextToNodeId (context) << " " << p->GetSize ());
}

void
MacTxTrace (std::string context, Ptr<const Packet> p)
{
  if (tracing)
    {
      macTxTraceFile << Simulator::Now ().GetSeconds () << " " << ContextToNodeId (context) << " " << p->GetSize () << std::endl;
    }
}

void
MacRxTrace (std::string context, Ptr<const Packet> p)
{
  if (tracing)
    {
      macRxTraceFile << Simulator::Now ().GetSeconds () << " " << ContextToNodeId (context) << " " << p->GetSize () << std::endl;
    }
}

void
SocketSendTrace (std::string context, Ptr<const Packet> p, const Address &addr)
{
  if (tracing)
    {
      socketSendTraceFile << Simulator::Now ().GetSeconds () << " " << ContextToNodeId (context) << " " << p->GetSize () << " " << addr << std::endl;
    }
}

void
AssociationLog (std::string context, Mac48Address address)
{
  uint32_t nodeId = ContextToNodeId (context);
  auto it = associated.find (nodeId);
  if (it == associated.end ())
    {
      NS_LOG_DEBUG ("Association: time=" << Simulator::Now () << " node=" << nodeId);
      associated.insert (it, nodeId);
    }
  else
    {
      NS_FATAL_ERROR (nodeId << " is already associated!");
    }
}

void
DisassociationLog (std::string context, Mac48Address address)
{
  uint32_t nodeId = ContextToNodeId (context);
  NS_LOG_DEBUG ("Disassociation: time=" << Simulator::Now () << " node=" << nodeId);
  NS_FATAL_ERROR ("Device should not disassociate!");
}

void
RestartCalc ()
{
  bytesReceived.clear ();
  packetsReceived.clear ();
  packetsTransmitted.clear ();
  psduFailed.clear ();
  psduSucceeded.clear ();
  phyHeaderFailed.clear ();
  timeFirstReceived.clear ();
  timeLastReceived.clear ();
  rxEventWhileDecodingPreamble.clear ();
  rxEventWhileRxing.clear ();
  rxEventWhileTxing.clear ();
  rxEventAbortedByTx.clear ();
}

class Experiment
{
public:
  Experiment ();

  /**
   * Configure and run the experiment.
   *
   * \param wifi the pre-configured WifiHelper
   * \param wifiPhy the pre-configured YansWifiPhyHelper
   * \param wifiMac the pre-configured WifiMacHelper
   * \param wifiChannel the pre-configured YansWifiChannelHelper
   * \param trialNumber the trial index
   * \param networkSize the number of stations
   * \param duration the duration of each simulation run
   * \param pcap flag to enable/disable PCAP files generation
   * \param infra flag to enable infrastructure model, ring adhoc network if not set
   * \param guardIntervalNs the guard interval in ns
   * \param distanceM the distance in meters
   * \param apTxPowerDbm the AP transmit power in dBm
   * \param staTxPowerDbm the STA transmit power in dBm
   * \param pktInterval the packet interval
   * \return 0 if all went well
   */
  int Run (const WifiHelper &wifi, const YansWifiPhyHelper &wifiPhy, const WifiMacHelper &wifiMac, const YansWifiChannelHelper &wifiChannel,
           uint32_t trialNumber, uint32_t networkSize, Time duration, bool pcap, bool infra, uint16_t guardIntervalNs,
           double distanceM, double apTxPowerDbm, double staTxPowerDbm, Time pktInterval);
};

Experiment::Experiment ()
{
}

int
Experiment::Run (const WifiHelper &helper, const YansWifiPhyHelper &wifiPhy, const WifiMacHelper &wifiMac, const YansWifiChannelHelper &wifiChannel,
                 uint32_t trialNumber, uint32_t networkSize, Time duration, bool pcap, bool infra, uint16_t guardIntervalNs,
                 double distance, double apTxPowerDbm, double staTxPowerDbm, Time pktInterval)
{
  RngSeedManager::SetSeed (10);
  RngSeedManager::SetRun (10);

  NodeContainer wifiNodes;
  if (infra)
    {
      wifiNodes.Create (networkSize + 1);
    }
  else
    {
      wifiNodes.Create (networkSize);
    }

  YansWifiPhyHelper phy = wifiPhy;
  phy.SetErrorRateModel ("ns3::NistErrorRateModel");
  phy.SetChannel (wifiChannel.Create ());
  phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  WifiMacHelper mac = wifiMac;
  WifiHelper wifi = helper;
  NetDeviceContainer devices;
  uint32_t nNodes = wifiNodes.GetN ();
  if (infra)
    {
      Ssid ssid = Ssid ("wifi-bianchi");
      uint64_t beaconInterval = std::min<uint64_t> ((ceil ((duration.GetSeconds () * 1000000) / 1024) * 1024), (65535 * 1024)); //beacon interval needs to be a multiple of time units (1024 us)
      mac.SetType ("ns3::ApWifiMac",
                   "BeaconInterval", TimeValue (MicroSeconds (beaconInterval)),
                   "Ssid", SsidValue (ssid));
      phy.Set ("TxPowerStart", DoubleValue (apTxPowerDbm));
      phy.Set ("TxPowerEnd", DoubleValue (apTxPowerDbm));
      devices = wifi.Install (phy, mac, wifiNodes.Get (0));

      mac.SetType ("ns3::StaWifiMac",
                   "MaxMissedBeacons", UintegerValue (std::numeric_limits<uint32_t>::max ()),
                   "Ssid", SsidValue (ssid));
      phy.Set ("TxPowerStart", DoubleValue (staTxPowerDbm));
      phy.Set ("TxPowerEnd", DoubleValue (staTxPowerDbm));
      for (uint32_t i = 1; i < nNodes; ++i)
        {
          devices.Add (wifi.Install (phy, mac, wifiNodes.Get (i)));
        }
    }
  else
    {
      mac.SetType ("ns3::AdhocWifiMac");
      phy.Set ("TxPowerStart", DoubleValue (staTxPowerDbm));
      phy.Set ("TxPowerEnd", DoubleValue (staTxPowerDbm));
      devices = wifi.Install (phy, mac, wifiNodes);
    }

  wifi.AssignStreams (devices, trialNumber);

  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (guardIntervalNs == 400));
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval", TimeValue (NanoSeconds (guardIntervalNs)));

  // Configure aggregation
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<NetDevice> dev = wifiNodes.Get (i)->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      wifi_dev->GetMac ()->SetAttribute ("BE_MaxAmpduSize", UintegerValue (maxMpdus * (pktSize + 50)));
      wifi_dev->GetMac ()->SetAttribute ("BK_MaxAmpduSize", UintegerValue (maxMpdus * (pktSize + 50)));
      wifi_dev->GetMac ()->SetAttribute ("VO_MaxAmpduSize", UintegerValue (maxMpdus * (pktSize + 50)));
      wifi_dev->GetMac ()->SetAttribute ("VI_MaxAmpduSize", UintegerValue (maxMpdus * (pktSize + 50)));
    }

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  // Set postion for AP
  positionAlloc->Add (Vector (1.0, 1.0, 0.0));

  // Set postion for STAs
  double angle = (static_cast<double> (360) / (nNodes - 1));
  for (uint32_t i = 0; i < (nNodes - 1); ++i)
    {
      positionAlloc->Add (Vector (1.0 + (distance * cos ((i * angle * PI) / 180)), 1.0 + (distance * sin ((i * angle * PI) / 180)), 0.0));
    }

  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (wifiNodes);

  PacketSocketHelper packetSocket;
  packetSocket.Install (wifiNodes);

  ApplicationContainer apps;
  Ptr<UniformRandomVariable> startTime = CreateObject<UniformRandomVariable> ();
  startTime->SetAttribute ("Stream", IntegerValue (trialNumber));
  startTime->SetAttribute ("Max", DoubleValue (5.0));

  uint32_t i = infra ? 1 : 0;
  for (; i < nNodes; ++i)
    {
      uint32_t j = infra ? 0 : (i + 1) % nNodes;
      PacketSocketAddress socketAddr;
      socketAddr.SetSingleDevice (devices.Get (i)->GetIfIndex ());
      socketAddr.SetPhysicalAddress (devices.Get (j)->GetAddress ());
      socketAddr.SetProtocol (1);

      Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
      client->SetRemote (socketAddr);
      wifiNodes.Get (i)->AddApplication (client);
      client->SetAttribute ("PacketSize", UintegerValue (pktSize));
      client->SetAttribute ("MaxPackets", UintegerValue (0));
      client->SetAttribute ("Interval", TimeValue (pktInterval));
      double start = startTime->GetValue ();
      NS_LOG_DEBUG ("Client " << i << " starting at " << start);
      client->SetStartTime (Seconds (start));

      Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
      server->SetLocal (socketAddr);
      wifiNodes.Get (j)->AddApplication (server);
    }

  // Log packet receptions
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/MonitorSnifferRx", MakeCallback (&TracePacketReception));

  // Log association and disassociation
  if (infra)
    {
      Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::StaWifiMac/Assoc", MakeCallback (&AssociationLog));
      Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::StaWifiMac/DeAssoc", MakeCallback (&DisassociationLog));
    }

  // Trace CW evolution
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/Txop/CwTrace", MakeCallback (&CwTrace));
  // Trace backoff evolution
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/Txop/BackoffTrace", MakeCallback (&BackoffTrace));
  // Trace PHY Tx start events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxBegin", MakeCallback (&PhyTxTrace));
  // Trace PHY Tx end events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxEnd", MakeCallback (&PhyTxDoneTrace));
  // Trace PHY Rx start events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyRxBegin", MakeCallback (&PhyRxTrace));
  // Trace PHY Rx payload start events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyRxPayloadBegin", MakeCallback (&PhyRxPayloadTrace));
  // Trace PHY Rx drop events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyRxDrop", MakeCallback (&PhyRxDropTrace));
  // Trace PHY Rx end events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyRxEnd", MakeCallback (&PhyRxDoneTrace));
  // Trace PHY Rx error events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/State/RxError", MakeCallback (&PhyRxErrorTrace));
  // Trace PHY Rx success events
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/State/RxOk", MakeCallback (&PhyRxOkTrace));
  // Trace packet transmission by the device
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx", MakeCallback (&MacTxTrace));
  // Trace packet receptions to the device
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx", MakeCallback (&MacRxTrace));
  // Trace packets transmitted by the application
  Config::Connect ("/NodeList/*/$ns3::Node/ApplicationList/*/$ns3::PacketSocketClient/Tx", MakeCallback (&SocketSendTrace));

  Simulator::Schedule (Seconds (10), &RestartCalc);
  Simulator::Stop (Seconds (10) + duration);

  if (pcap)
    {
      phy.EnablePcap ("wifi_bianchi_pcap", devices);
    }

  Simulator::Run ();
  Simulator::Destroy ();

  if (tracing)
    {
      cwTraceFile.flush ();
      backoffTraceFile.flush ();
      phyTxTraceFile.flush ();
      macTxTraceFile.flush ();
      macRxTraceFile.flush ();
      socketSendTraceFile.flush ();
    }

  return 0;
}

uint64_t
GetCount (const std::map<Mac48Address, uint64_t> & counter, Mac48Address addr)
{
  uint64_t count = 0;
  auto it = counter.find (addr);
  if (it != counter.end ())
    {
      count = it->second;
    }
  return count;
}

int main (int argc, char *argv[])
{
  uint32_t nMinStas = 5;                  ///< Minimum number of STAs to start with
  uint32_t nMaxStas = 50;                 ///< Maximum number of STAs to end with
  uint32_t nStepSize = 5;                 ///< Number of stations to add at each step
  uint32_t verbose = 0;                   ///< verbosity level that increases the number of debugging traces
  double duration = 100;                  ///< duration (in seconds) of each simulation run (i.e. per trial and per number of stations)
  uint32_t trials = 1;                    ///< Number of runs per point in the plot
  bool pcap = false;                      ///< Flag to enable/disable PCAP files generation
  bool infra = false;                     ///< Flag to enable infrastructure model, ring adhoc network if not set
  std::string workDir = "./";             ///< the working directory to store generated files
  std::string phyMode = "OfdmRate54Mbps"; ///< the constant PHY mode string used to transmit frames
  std::string standard ("11a");           ///< the 802.11 standard
  bool validate = false;                  ///< Flag used for regression in order to verify ns-3 results are in the expected boundaries
  uint16_t plotBianchiModel = 0x01;        ///< First bit corresponds to the DIFS model, second bit to the EIFS model
  double maxRelativeError = 0.015;        ///< Maximum relative error tolerated between ns-3 results and the Bianchi model (used for regression, i.e. when the validate flag is set)
  double frequency = 5;                   ///< The operating frequency band in GHz: 2.4, 5 or 6
  uint16_t channelWidth = 20;             ///< The constant channel width in MHz (only for 11n/ac/ax)
  uint16_t guardIntervalNs = 800;         ///< The guard interval in nanoseconds (800 or 400 for 11n/ac, 800 or 1600 or 3200 for 11 ax)
  uint16_t pktInterval = 1000;            ///< The socket packet interval in microseconds (a higher value is needed to reach saturation conditions as the channel bandwidth or the MCS increases)
  double distance = 0.001;                ///< The distance in meters between the AP and the STAs
  double apTxPower = 16;                  ///< The transmit power of the AP in dBm (if infrastructure only)
  double staTxPower = 16;                 ///< The transmit power of each STA in dBm (or all STAs if adhoc)

  // Disable fragmentation and RTS/CTS
  Config::SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold", StringValue ("22000"));
  Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue ("22000"));
  // Disable short retransmission failure (make retransmissions persistent)
  Config::SetDefault ("ns3::WifiRemoteStationManager::MaxSlrc", UintegerValue (std::numeric_limits<uint32_t>::max ()));
  Config::SetDefault ("ns3::WifiRemoteStationManager::MaxSsrc", UintegerValue (std::numeric_limits<uint32_t>::max ()));
  // Set maximum queue size to the largest value and set maximum queue delay to be larger than the simulation time
  Config::SetDefault ("ns3::WifiMacQueue::MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, std::numeric_limits<uint32_t>::max ())));
  Config::SetDefault ("ns3::WifiMacQueue::MaxDelay", TimeValue (Seconds (2 * duration)));

  CommandLine cmd (__FILE__);
  cmd.AddValue ("verbose", "Logging level (0: no log - 1: simulation script logs - 2: all logs)", verbose);
  cmd.AddValue ("tracing", "Generate trace files", tracing);
  cmd.AddValue ("pktSize", "The packet size in bytes", pktSize);
  cmd.AddValue ("trials", "The maximal number of runs per network size", trials);
  cmd.AddValue ("duration", "Time duration for each trial in seconds", duration);
  cmd.AddValue ("pcap", "Enable/disable PCAP tracing", pcap);
  cmd.AddValue ("infra", "True to use infrastructure mode, false to use ring adhoc mode", infra);
  cmd.AddValue ("workDir", "The working directory used to store generated files", workDir);
  cmd.AddValue ("phyMode", "Set the constant PHY mode string used to transmit frames", phyMode);
  cmd.AddValue ("standard", "Set the standard (11a, 11b, 11g, 11n, 11ac, 11ax)", standard);
  cmd.AddValue ("nMinStas", "Minimum number of stations to start with", nMinStas);
  cmd.AddValue ("nMaxStas", "Maximum number of stations to start with", nMaxStas);
  cmd.AddValue ("nStepSize", "Number of stations to add at each step", nStepSize);
  cmd.AddValue ("plotBianchiModel", "First bit corresponds to the DIFS model, second bit to the EIFS model", plotBianchiModel);
  cmd.AddValue ("validate", "Enable/disable validation of the ns-3 simulations against the Bianchi model", validate);
  cmd.AddValue ("maxRelativeError", "The maximum relative error tolerated between ns-3 results and the Bianchi model (used for regression, i.e. when the validate flag is set)", maxRelativeError);
  cmd.AddValue ("frequency", "Set the operating frequency band in GHz: 2.4, 5 or 6", frequency);
  cmd.AddValue ("channelWidth", "Set the constant channel width in MHz (only for 11n/ac/ax)", channelWidth);
  cmd.AddValue ("guardIntervalNs", "Set the the guard interval in nanoseconds (800 or 400 for 11n/ac, 800 or 1600 or 3200 for 11 ax)", guardIntervalNs);
  cmd.AddValue ("maxMpdus", "Set the maximum number of MPDUs in A-MPDUs (0 to disable MPDU aggregation)", maxMpdus);
  cmd.AddValue ("distance", "Set the distance in meters between the AP and the STAs", distance);
  cmd.AddValue ("apTxPower", "Set the transmit power of the AP in dBm (if infrastructure only)", apTxPower);
  cmd.AddValue ("staTxPower", "Set the transmit power of each STA in dBm (or all STAs if adhoc)", staTxPower);
  cmd.AddValue ("pktInterval", "Set the socket packet interval in microseconds", pktInterval);
  cmd.Parse (argc, argv);

  if (tracing)
    {
      cwTraceFile.open ("wifi-bianchi-cw-trace.out");
      if (!cwTraceFile.is_open ())
        {
          NS_FATAL_ERROR ("Failed to open file wifi-bianchi-cw-trace.out");
        }
      backoffTraceFile.open ("wifi-bianchi-backoff-trace.out");
      if (!backoffTraceFile.is_open ())
        {
          NS_FATAL_ERROR ("Failed to open file wifi-bianchi-backoff-trace.out");
        }
      phyTxTraceFile.open ("wifi-bianchi-phy-tx-trace.out");
      if (!phyTxTraceFile.is_open ())
        {
          NS_FATAL_ERROR ("Failed to open file wifi-bianchi-phy-tx-trace.out");
        }
      macTxTraceFile.open ("wifi-bianchi-mac-tx-trace.out");
      if (!macTxTraceFile.is_open ())
        {
          NS_FATAL_ERROR ("Failed to open file wifi-bianchi-mac-tx-trace.out");
        }
      macRxTraceFile.open ("wifi-bianchi-mac-rx-trace.out");
      if (!macRxTraceFile.is_open ())
        {
          NS_FATAL_ERROR ("Failed to open file wifi-bianchi-mac-rx-trace.out");
        }
      socketSendTraceFile.open ("wifi-bianchi-socket-send-trace.out");
      if (!socketSendTraceFile.is_open ())
        {
          NS_FATAL_ERROR ("Failed to open file wifi-bianchi-socket-send-trace.out");
        }
    }

  if (verbose >= 1)
    {
      LogComponentEnable ("WifiBianchi", LOG_LEVEL_ALL);
    }
  else
    {
      LogComponentEnable ("WifiBianchi", LOG_LEVEL_WARN);
    }
  if (verbose >= 2)
    {
      WifiHelper::EnableLogComponents ();
    }

  std::stringstream phyModeStr;
  phyModeStr << phyMode;
  if (phyMode.find ("Mcs") != std::string::npos)
    {
      phyModeStr << "_" << channelWidth << "MHz";
    }

  std::stringstream ss;
  ss << "wifi-"<< standard << "-p-" << pktSize << (infra ? "-infrastructure" : "-adhoc") << "-r-" << phyModeStr.str () << "-min-" << nMinStas << "-max-" << nMaxStas << "-step-" << nStepSize << "-throughput.plt";
  std::ofstream throughputPlot (ss.str ().c_str ());
  ss.str ("");
  ss << "wifi-" << standard << "-p-" << pktSize << (infra ? "-infrastructure" : "-adhoc") <<"-r-" << phyModeStr.str () << "-min-" << nMinStas << "-max-" << nMaxStas << "-step-" << nStepSize << "-throughput.eps";
  Gnuplot gnuplot = Gnuplot (ss.str ());

  WifiStandard wifiStandard;
  if (standard == "11a")
    {
      wifiStandard = WIFI_STANDARD_80211a;
      frequency = 5;
      channelWidth = 20;
    }
  else if (standard == "11b")
    {
      wifiStandard = WIFI_STANDARD_80211b;
      frequency = 2.4;
      channelWidth = 22;
    }
  else if (standard == "11g")
    {
      wifiStandard = WIFI_STANDARD_80211g;
      frequency = 2.4;
      channelWidth = 20;
    }
  else if (standard == "11n")
    {
      if (frequency == 2.4)
        {
          wifiStandard = WIFI_STANDARD_80211n;
        }
      else if (frequency == 5)
        {
          wifiStandard = WIFI_STANDARD_80211n;
        }
      else
        {
          NS_FATAL_ERROR ("Unsupported frequency band " << frequency << " GHz for standard " << standard);
        }
    }
  else if (standard == "11ac")
    {
      wifiStandard = WIFI_STANDARD_80211ac;
      frequency = 5;
    }
  else if (standard == "11ax")
    {
      if (frequency == 2.4)
        {
          wifiStandard = WIFI_STANDARD_80211ax;
        }
      else if (frequency == 5)
        {
          wifiStandard = WIFI_STANDARD_80211ax;
        }
      else if (frequency == 6)
        {
          wifiStandard = WIFI_STANDARD_80211ax;
        }
      else
        {
          NS_FATAL_ERROR ("Unsupported frequency band " << frequency << " GHz for standard " << standard);
        }
    }
  else
    {
      NS_FATAL_ERROR ("Unsupported standard: " << standard);
    }

  std::string channelStr = "{0, " + std::to_string (channelWidth) + ", BAND_"
                           + (frequency == 2.4 ? "2_4" : (frequency == 5 ? "5" : "6"))
                           + "GHZ, 0}";
  Config::SetDefault ("ns3::WifiPhy::ChannelSettings", StringValue (channelStr));

  YansWifiPhyHelper wifiPhy;
  wifiPhy.DisablePreambleDetectionModel ();

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  if (frequency == 6)
    {
      // Reference Loss for Friss at 1 m with 6.0 GHz
      wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                      "Exponent", DoubleValue (2.0),
                                      "ReferenceDistance", DoubleValue (1.0),
                                      "ReferenceLoss", DoubleValue (49.013));
    }
  else if (frequency == 5)
    {
      // Reference Loss for Friss at 1 m with 5.15 GHz
      wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                      "Exponent", DoubleValue (2.0),
                                      "ReferenceDistance", DoubleValue (1.0),
                                      "ReferenceLoss", DoubleValue (46.6777));
    }
  else
    {
      // Reference Loss for Friss at 1 m with 2.4 GHz
      wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                      "Exponent", DoubleValue (2.0),
                                      "ReferenceDistance", DoubleValue (1.0),
                                      "ReferenceLoss", DoubleValue (40.046));
    }

  WifiHelper wifi;
  wifi.SetStandard (wifiStandard);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue (phyMode),
                                "ControlMode", StringValue (phyMode));

  Gnuplot2dDataset dataset;
  Gnuplot2dDataset datasetBianchiEifs;
  Gnuplot2dDataset datasetBianchiDifs;
  dataset.SetErrorBars (Gnuplot2dDataset::Y);
  dataset.SetStyle (Gnuplot2dDataset::LINES_POINTS);
  datasetBianchiEifs.SetStyle (Gnuplot2dDataset::LINES_POINTS);
  datasetBianchiDifs.SetStyle (Gnuplot2dDataset::LINES_POINTS);

  Experiment experiment;
  WifiMacHelper wifiMac;
  double averageThroughput, throughputArray[trials];
  for (uint32_t n = nMinStas; n <= nMaxStas; n += nStepSize)
    {
      averageThroughput = 0;
      double throughput;
      for (uint32_t runIndex = 0; runIndex < trials; runIndex++)
        {
          packetsReceived.clear ();
          bytesReceived.clear ();
          packetsTransmitted.clear ();
          psduFailed.clear ();
          psduSucceeded.clear ();
          phyHeaderFailed.clear ();
          timeFirstReceived.clear ();
          timeLastReceived.clear ();
          rxEventWhileDecodingPreamble.clear ();
          rxEventWhileRxing.clear ();
          rxEventWhileTxing.clear ();
          rxEventAbortedByTx.clear ();
          associated.clear ();
          throughput = 0;
          std::cout << "Trial " << runIndex + 1 << " of " << trials << "; "<< phyModeStr.str () << " for " << n << " nodes " << std::endl;
          if (tracing)
            {
              cwTraceFile << "# Trial " << runIndex + 1 << " of " << trials << "; "<< phyModeStr.str () << " for " << n << " nodes" << std::endl;
              backoffTraceFile << "# Trial " << runIndex + 1 << " of " << trials << "; "<< phyModeStr.str () << " for " << n << " nodes" << std::endl;
              phyTxTraceFile << "# Trial " << runIndex + 1 << " of " << trials << "; " << phyModeStr.str () << " for " << n << " nodes" << std::endl;
              macTxTraceFile << "# Trial " << runIndex + 1 << " of " << trials << "; " << phyModeStr.str () << " for " << n << " nodes" << std::endl;
              macRxTraceFile << "# Trial " << runIndex + 1 << " of " << trials << "; " << phyModeStr.str () << " for " << n << " nodes" << std::endl;
              socketSendTraceFile << "# Trial " << runIndex + 1 << " of " << trials << "; " << phyModeStr.str () << " for " << n << " nodes" << std::endl;
            }
          experiment.Run (wifi, wifiPhy, wifiMac, wifiChannel, runIndex, n, Seconds (duration), pcap, infra, guardIntervalNs, distance, apTxPower, staTxPower, MicroSeconds (pktInterval));
          uint32_t k = 0;
          if (bytesReceived.size () != n)
            {
              NS_FATAL_ERROR ("Not all stations got traffic!");
            }
          for (auto it = bytesReceived.begin (); it != bytesReceived.end (); it++, k++)
            {
              Time first = timeFirstReceived.find (it->first)->second;
              Time last = timeLastReceived.find (it->first)->second;
              Time dataTransferDuration = last - first;
              double nodeThroughput = (it->second * 8 / static_cast<double> (dataTransferDuration.GetMicroSeconds ()));
              throughput += nodeThroughput;
              uint64_t nodeTxPackets = GetCount (packetsTransmitted, it->first);
              uint64_t nodeRxPackets = GetCount (packetsReceived, it->first);
              uint64_t nodePhyHeaderFailures = GetCount (phyHeaderFailed, it->first);
              uint64_t nodePsduFailures = GetCount (psduFailed, it->first);
              uint64_t nodePsduSuccess = GetCount (psduSucceeded, it->first);
              uint64_t nodeRxEventWhileDecodingPreamble = GetCount (rxEventWhileDecodingPreamble, it->first);
              uint64_t nodeRxEventWhileRxing = GetCount (rxEventWhileRxing, it->first);
              uint64_t nodeRxEventWhileTxing = GetCount (rxEventWhileTxing, it->first);
              uint64_t nodeRxEventAbortedByTx = GetCount (rxEventAbortedByTx, it->first);
              uint64_t nodeRxEvents =
                  nodePhyHeaderFailures + nodePsduFailures + nodePsduSuccess + nodeRxEventWhileDecodingPreamble +
                  nodeRxEventWhileRxing + nodeRxEventWhileTxing + nodeRxEventAbortedByTx;
              std::cout << "Node " << it->first
                        << ": TX packets " << nodeTxPackets
                        << "; RX packets " << nodeRxPackets
                        << "; PHY header failures " << nodePhyHeaderFailures
                        << "; PSDU failures " << nodePsduFailures
                        << "; PSDU success " << nodePsduSuccess
                        << "; RX events while decoding preamble " << nodeRxEventWhileDecodingPreamble
                        << "; RX events while RXing " << nodeRxEventWhileRxing
                        << "; RX events while TXing " << nodeRxEventWhileTxing
                        << "; RX events aborted by TX " << nodeRxEventAbortedByTx
                        << "; total RX events " << nodeRxEvents
                        << "; total events " << nodeTxPackets + nodeRxEvents
                        << "; time first RX " << first
                        << "; time last RX " << last
                        << "; dataTransferDuration " << dataTransferDuration
                        << "; throughput " << nodeThroughput  << " Mbps" << std::endl;
            }
          std::cout << "Total throughput: " << throughput << " Mbps" << std::endl;
          averageThroughput += throughput;
          throughputArray[runIndex] = throughput;
        }
      averageThroughput = averageThroughput / trials;

      bool rateFound = false;
      double relativeErrorDifs = 0;
      double relativeErrorEifs = 0;
      auto itDifs = bianchiResultsDifs.find (phyModeStr.str ());
      if (itDifs != bianchiResultsDifs.end ())
        {
          rateFound = true;
          auto it = itDifs->second.find (n);
          if (it != itDifs->second.end ())
            {
              relativeErrorDifs = (std::abs (averageThroughput - it->second) / it->second);
              std::cout << "Relative error (DIFS): " << 100 * relativeErrorDifs << "%" << std::endl;
            }
          else if (validate)
            {
              NS_FATAL_ERROR ("No Bianchi results (DIFS) calculated for that number of stations!");
            }
        }
      auto itEifs = bianchiResultsEifs.find (phyModeStr.str ());
      if (itEifs != bianchiResultsEifs.end ())
        {
          rateFound = true;
          auto it = itEifs->second.find (n);
          if (it != itEifs->second.end ())
            {
              relativeErrorEifs = (std::abs (averageThroughput - it->second) / it->second);
              std::cout << "Relative error (EIFS): " << 100 * relativeErrorEifs << "%" << std::endl;
            }
          else if (validate)
            {
              NS_FATAL_ERROR ("No Bianchi results (EIFS) calculated for that number of stations!");
            }
        }
      if (!rateFound && validate)
        {
          NS_FATAL_ERROR ("No Bianchi results calculated for that rate!");
        }
      double relativeError = std::min (relativeErrorDifs, relativeErrorEifs);
      if (validate && (relativeError > maxRelativeError))
        {
          NS_FATAL_ERROR ("Relative error is too high!");
        }

      double stDev = 0;
      for (uint32_t i = 0; i < trials; ++i)
        {
          stDev += pow (throughputArray[i] - averageThroughput, 2);
        }
      stDev = sqrt (stDev / (trials - 1));
      dataset.Add (n, averageThroughput, stDev);
    }
  dataset.SetTitle ("ns-3");

  auto itDifs = bianchiResultsDifs.find (phyModeStr.str ());
  if (itDifs != bianchiResultsDifs.end ())
    {
      for (uint32_t i = nMinStas; i <= nMaxStas; i += nStepSize)
        {
          double value = 0.0;
          auto it = itDifs->second.find (i);
          if (it != itDifs->second.end ())
            {
              value = it->second;
            }
          datasetBianchiDifs.Add (i, value);
        }
    }
  else
    {
      for (uint32_t i = nMinStas; i <= nMaxStas; i += nStepSize)
        {
          datasetBianchiDifs.Add (i, 0.0);
        }
    }

  auto itEifs = bianchiResultsEifs.find (phyModeStr.str ());
  if (itEifs != bianchiResultsEifs.end ())
    {
      for (uint32_t i = nMinStas; i <= nMaxStas; i += nStepSize)
        {
          double value = 0.0;
          auto it = itEifs->second.find (i);
          if (it != itEifs->second.end ())
            {
              value = it->second;
            }
          datasetBianchiEifs.Add (i, value);
        }
    }
  else
    {
      for (uint32_t i = nMinStas; i <= nMaxStas; i += nStepSize)
        {
          datasetBianchiEifs.Add (i, 0.0);
        }
    }

  datasetBianchiEifs.SetTitle ("Bianchi (EIFS - lower bound)");
  datasetBianchiDifs.SetTitle ("Bianchi (DIFS - upper bound)");
  gnuplot.AddDataset (dataset);
  gnuplot.SetTerminal ("postscript eps color enh \"Times-BoldItalic\"");
  gnuplot.SetLegend ("Number of competing stations", "Throughput (Mbps)");
  ss.str ("");
  ss << "Frame size " << pktSize << " bytes";
  gnuplot.SetTitle (ss.str ());
  ss.str ("");
  ss << "set xrange [" << nMinStas << ":" << nMaxStas << "]\n"
     << "set xtics " << nStepSize << "\n"
     << "set grid xtics ytics\n"
     << "set mytics\n"
     << "set style line 1 linewidth 5\n"
     << "set style line 2 linewidth 5\n"
     << "set style line 3 linewidth 5\n"
     << "set style line 4 linewidth 5\n"
     << "set style line 5 linewidth 5\n"
     << "set style line 6 linewidth 5\n"
     << "set style line 7 linewidth 5\n"
     << "set style line 8 linewidth 5\n"
     << "set style increment user";
  gnuplot.SetExtra (ss.str ());
  if (plotBianchiModel & 0x0001)
    {
      datasetBianchiDifs.SetTitle ("Bianchi");
      gnuplot.AddDataset (datasetBianchiDifs);
    }
  if (plotBianchiModel & 0x0002)
    {
      datasetBianchiEifs.SetTitle ("Bianchi");
      gnuplot.AddDataset (datasetBianchiEifs);
    }
  if (plotBianchiModel == 0x0003)
    {
      datasetBianchiEifs.SetTitle ("Bianchi (EIFS - lower bound)");
      datasetBianchiDifs.SetTitle ("Bianchi (DIFS - upper bound)");
    }
  gnuplot.GenerateOutput (throughputPlot);
  throughputPlot.close ();

  if (tracing)
    {
      cwTraceFile.close ();
      backoffTraceFile.close ();
      phyTxTraceFile.close ();
      macTxTraceFile.close ();
      macRxTraceFile.close ();
      socketSendTraceFile.close ();
    }

  return 0;
}
