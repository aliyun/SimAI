/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021
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
 * Authors: Stefano Avallone <stavallo@unina.it>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/assert.h"
#include "wifi-phy-operating-channel.h"
#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiPhyOperatingChannel");

const std::set<FrequencyChannelInfo> WifiPhyOperatingChannel::m_frequencyChannels =
{
  //2.4 GHz channels
  // 802.11b uses width of 22, while OFDM modes use width of 20
  { std::make_tuple (1, 2412, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (1, 2412, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (2, 2417, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (2, 2417, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (3, 2422, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (3, 2422, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (4, 2427, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (4, 2427, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (5, 2432, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (5, 2432, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (6, 2437, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (6, 2437, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (7, 2442, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (7, 2442, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (8, 2447, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (8, 2447, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (9, 2452, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (9, 2452, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (10, 2457, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (10, 2457, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (11, 2462, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (11, 2462, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (12, 2467, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (12, 2467, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (13, 2472, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (13, 2472, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  // Only defined for 802.11b
  { std::make_tuple (14, 2484, 22, WIFI_PHY_DSSS_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  // 40 MHz channels
  { std::make_tuple (3, 2422, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (4, 2427, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (5, 2432, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (6, 2437, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (7, 2442, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (8, 2447, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (9, 2452, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (10, 2457, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },
  { std::make_tuple (11, 2462, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_2_4GHZ) },

  // Now the 5 GHz channels used for 802.11a/n/ac/ax
  // 20 MHz channels
  { std::make_tuple (36, 5180, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (40, 5200, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (44, 5220, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (48, 5240, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (52, 5260, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (56, 5280, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (60, 5300, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (64, 5320, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (100, 5500, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (104, 5520, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (108, 5540, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (112, 5560, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (116, 5580, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (120, 5600, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (124, 5620, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (128, 5640, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (132, 5660, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (136, 5680, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (140, 5700, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (144, 5720, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (149, 5745, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (153, 5765, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (157, 5785, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (161, 5805, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (165, 5825, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (169, 5845, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (173, 5865, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (177, 5885, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (181, 5905, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  // 40 MHz channels
  { std::make_tuple (38, 5190, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (46, 5230, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (54, 5270, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (62, 5310, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (102, 5510, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (110, 5550, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (118, 5590, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (126, 5630, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (134, 5670, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (142, 5710, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (151, 5755, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (159, 5795, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (167, 5835, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (175, 5875, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  // 80 MHz channels
  { std::make_tuple (42, 5210, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (58, 5290, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (106, 5530, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (122, 5610, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (138, 5690, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (155, 5775, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (171, 5855, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  // 160 MHz channels
  { std::make_tuple (50, 5250, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (114, 5570, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (163, 5815, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_5GHZ) },

  // 802.11p 10 MHz channels at the 5.855-5.925 band
  { std::make_tuple (172, 5860, 10, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (174, 5870, 10, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (176, 5880, 10, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (178, 5890, 10, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (180, 5900, 10, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (182, 5910, 10, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (184, 5920, 10, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },

  // 802.11p 5 MHz channels at the 5.855-5.925 band (for simplification, we consider the same center frequencies as the 10 MHz channels)
  { std::make_tuple (171, 5860, 5, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (173, 5870, 5, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (175, 5880, 5, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (177, 5890, 5, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (179, 5900, 5, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (181, 5910, 5, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },
  { std::make_tuple (183, 5920, 5, WIFI_PHY_80211p_CHANNEL, WIFI_PHY_BAND_5GHZ) },

  // Now the 6 GHz channels (802.11ax only)
  // 20 MHz channels
  { std::make_tuple (1, 5945, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (5, 5965, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (9, 5985, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (13, 6005, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (17, 6025, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (21, 6045, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (25, 6065, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (29, 6085, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (33, 6105, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (37, 6125, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (41, 6145, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (45, 6165, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (49, 6185, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (53, 6205, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (57, 6225, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (61, 6245, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (65, 6265, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (69, 6285, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (73, 6305, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (77, 6325, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (81, 6345, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (85, 6365, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (89, 6385, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (93, 6405, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (97, 6425, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (101, 6445, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (105, 6465, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (109, 6485, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (113, 6505, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (117, 6525, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (121, 6545, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (125, 6565, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (129, 6585, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (133, 6605, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (137, 6625, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (141, 6645, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (145, 6665, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (149, 6685, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (153, 6705, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (157, 6725, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (161, 6745, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (165, 6765, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (169, 6785, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (173, 6805, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (177, 6825, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (181, 6845, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (185, 6865, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (189, 6885, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (193, 6905, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (197, 6925, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (201, 6945, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (205, 6965, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (209, 6985, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (213, 7005, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (217, 7025, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (221, 7045, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (225, 7065, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (229, 7085, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (233, 7105, 20, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  // 40 MHz channels
  { std::make_tuple (3, 5955, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (11, 5995, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (19, 6035, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (27, 6075, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (35, 6115, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (43, 6155, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (51, 6195, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (59, 6235, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (67, 6275, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (75, 6315, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (83, 6355, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (91, 6395, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (99, 6435, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (107, 6475, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (115, 6515, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (123, 6555, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (131, 6595, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (139, 6635, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (147, 6675, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (155, 6715, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (163, 6755, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (171, 6795, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (179, 6835, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (187, 6875, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (195, 6915, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (203, 6955, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (211, 6995, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (219, 7035, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (227, 7075, 40, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  // 80 MHz channels
  { std::make_tuple (7, 5975, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (23, 6055, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (39, 6135, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (55, 6215, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (71, 6295, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (87, 6375, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (103, 6455, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (119, 6535, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (135, 6615, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (151, 6695, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (167, 6775, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (183, 6855, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (199, 6935, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (215, 7015, 80, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  // 160 MHz channels
  { std::make_tuple (15, 6015, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (47, 6175, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (79, 6335, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (111, 6495, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (143, 6655, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (175, 6815, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) },
  { std::make_tuple (207, 6975, 160, WIFI_PHY_OFDM_CHANNEL, WIFI_PHY_BAND_6GHZ) }
};

WifiPhyOperatingChannel::WifiPhyOperatingChannel ()
  : m_channelIt (m_frequencyChannels.end ()),
    m_primary20Index (0)
{
  NS_LOG_FUNCTION (this);
}

WifiPhyOperatingChannel::~WifiPhyOperatingChannel ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

bool
WifiPhyOperatingChannel::IsSet (void) const
{
  return m_channelIt != m_frequencyChannels.end ();
}

void
WifiPhyOperatingChannel::Set (uint8_t number, uint16_t frequency, uint16_t width,
                              WifiStandard standard, WifiPhyBand band)
{
  NS_LOG_FUNCTION (this << +number << frequency << width << standard << band);

  auto channelIt = FindFirst (number, frequency, width, standard, band);

  if (channelIt != m_frequencyChannels.end ()
      && FindFirst (number, frequency, width, standard, band, std::next (channelIt))
         == m_frequencyChannels.end ())
    {
      // a unique channel matches the specified criteria
      m_channelIt = channelIt;
      m_primary20Index = 0;
      return;
    }

  // if a unique channel was not found, throw an exception (mainly for unit testing this code)
  throw std::runtime_error ("WifiPhyOperatingChannel: No unique channel found given the specified criteria");
}

void
WifiPhyOperatingChannel::SetDefault (uint16_t width, WifiStandard standard, WifiPhyBand band)
{
  NS_LOG_FUNCTION (this << width << standard << band);

  Set (GetDefaultChannelNumber (width, standard, band), 0, width, standard, band);
}

uint8_t
WifiPhyOperatingChannel::GetDefaultChannelNumber (uint16_t width, WifiStandard standard, WifiPhyBand band)
{
  auto channelIt = FindFirst (0, 0, width, standard, band);

  if (channelIt != m_frequencyChannels.end ())
    {
      // a channel matches the specified criteria
      return std::get<0> (*channelIt);
    }

  // if a default channel was not found, throw an exception (mainly for unit testing this code)
  throw std::runtime_error ("WifiPhyOperatingChannel: No default channel found of the given width and for the given PHY standard and band");
}

WifiPhyOperatingChannel::ConstIterator
WifiPhyOperatingChannel::FindFirst (uint8_t number, uint16_t frequency, uint16_t width,
                                    WifiStandard standard, WifiPhyBand band,
                                    ConstIterator start)
{
  // lambda used to match channels against the specified criteria
  auto predicate = [&](const FrequencyChannelInfo& channel)
    {
      if (number != 0 && std::get<0> (channel) != number)
        {
          return false;
        }
      if (frequency != 0 && std::get<1> (channel) != frequency)
        {
          return false;
        }
      if (width != 0 && std::get<2> (channel) != width)
        {
          return false;
        }
      if (std::get<3> (channel) != GetFrequencyChannelType (standard))
        {
          return false;
        }
      if (std::get<4> (channel) != band)
        {
          return false;
        }
      return true;
    };

  // Do not search for a channel matching the specified criteria if the given PHY band
  // is not allowed for the given standard or the given channel width is not allowed
  // for the given standard
  if (const auto standardIt = wifiStandards.find (standard);
      standardIt == wifiStandards.cend ()
      || std::find (standardIt->second.cbegin (), standardIt->second.cend (), band) == standardIt->second.cend ()
      || width > GetMaximumChannelWidth (standard))
    {
      return m_frequencyChannels.cend ();
    }

  return std::find_if (start, m_frequencyChannels.cend (), predicate);
}

uint8_t
WifiPhyOperatingChannel::GetNumber (void) const
{
  NS_ASSERT (IsSet ());
  return std::get<0> (*m_channelIt);
}

uint16_t
WifiPhyOperatingChannel::GetFrequency (void) const
{
  NS_ASSERT (IsSet ());
  return std::get<1> (*m_channelIt);
}

uint16_t
WifiPhyOperatingChannel::GetWidth (void) const
{
  NS_ASSERT (IsSet ());
  return std::get<2> (*m_channelIt);
}

uint8_t
WifiPhyOperatingChannel::GetPrimaryChannelIndex (uint16_t primaryChannelWidth) const
{
  NS_LOG_FUNCTION (this << primaryChannelWidth);

  if (primaryChannelWidth % 20 != 0)
    {
      NS_LOG_DEBUG ("The operating channel width is not a multiple of 20 MHz; return 0");
      return 0;
    }

  NS_ASSERT (primaryChannelWidth <= GetWidth ());

  // the index of primary40 is half the index of primary20; the index of
  // primary80 is half the index of primary40, ...
  uint16_t width = 20;
  uint8_t index = m_primary20Index;

  while (width < primaryChannelWidth)
    {
      index /= 2;
      width *= 2;
    }
  NS_LOG_LOGIC ("Return " << +index);
  return index;
}

void
WifiPhyOperatingChannel::SetPrimary20Index (uint8_t index)
{
  NS_LOG_FUNCTION (this << +index);

  NS_ABORT_MSG_IF (index > 0 && index >= GetWidth () / 20, "Primary20 index out of range");
  m_primary20Index = index;
}

uint16_t
WifiPhyOperatingChannel::GetPrimaryChannelCenterFrequency (uint16_t primaryChannelWidth) const
{
  uint16_t freq = GetFrequency () - GetWidth () / 2.
                  + (GetPrimaryChannelIndex (primaryChannelWidth) + 0.5) * primaryChannelWidth;

  NS_LOG_FUNCTION (this << primaryChannelWidth << freq);
  return freq;
}

uint8_t
WifiPhyOperatingChannel::GetPrimaryChannelNumber (uint16_t     primaryChannelWidth,
                                                  WifiStandard standard) const
{
  auto frequency = GetPrimaryChannelCenterFrequency (primaryChannelWidth);
  auto& [chanNumber, centerFreq, channelWidth, channelType, band] = *m_channelIt;
  auto primaryChanIt = FindFirst (0, frequency, primaryChannelWidth, standard, band);
  return std::get<0> (*primaryChanIt);
}

} //namespace ns3
