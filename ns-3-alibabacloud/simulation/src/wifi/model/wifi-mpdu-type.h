/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef WIFI_MPDU_TYPE_H
#define WIFI_MPDU_TYPE_H

namespace ns3 {

/**
 * \ingroup wifi
 * The type of an MPDU.
 */
enum MpduType
{
  /** The MPDU is not part of an A-MPDU */
  NORMAL_MPDU,
  /** The MPDU is a single MPDU */
  SINGLE_MPDU,
  /** The MPDU is the first aggregate in an A-MPDU with multiple MPDUs, but is not the last aggregate */
  FIRST_MPDU_IN_AGGREGATE,
  /** The MPDU is part of an A-MPDU with multiple MPDUs, but is neither the first nor the last aggregate */
  MIDDLE_MPDU_IN_AGGREGATE,
  /** The MPDU is the last aggregate in an A-MPDU with multiple MPDUs */
  LAST_MPDU_IN_AGGREGATE
};

} //namespace ns3

#endif /* WIFI_MPDU_TYPE_H */
