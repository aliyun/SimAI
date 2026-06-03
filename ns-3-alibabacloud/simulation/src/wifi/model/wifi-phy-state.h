/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef WIFI_PHY_STATE_H
#define WIFI_PHY_STATE_H

#include "ns3/fatal-error.h"

/**
  * The state of the PHY layer.
  */
/// State enumeration
enum WifiPhyState
{
  /**
   * The PHY layer is IDLE.
    */
  IDLE,
  /**
    * The PHY layer has sense the medium busy through the CCA mechanism
    */
  CCA_BUSY,
  /**
    * The PHY layer is sending a packet.
    */
  TX,
  /**
    * The PHY layer is receiving a packet.
    */
  RX,
  /**
    * The PHY layer is switching to other channel.
    */
  SWITCHING,
  /**
    * The PHY layer is sleeping.
    */
  SLEEP,
  /**
    * The PHY layer is switched off.
    */
  OFF
};

/**
* \brief Stream insertion operator.
*
* \param os the stream
* \param state the state
* \returns a reference to the stream
*/
inline std::ostream& operator<< (std::ostream& os, WifiPhyState state)
{
  switch (state)
    {
    case IDLE:
      return (os << "IDLE");
    case CCA_BUSY:
      return (os << "CCA_BUSY");
    case TX:
      return (os << "TX");
    case RX:
      return (os << "RX");
    case SWITCHING:
      return (os << "SWITCHING");
    case SLEEP:
      return (os << "SLEEP");
    case OFF:
      return (os << "OFF");
    default:
      NS_FATAL_ERROR ("Invalid state");
      return (os << "INVALID");
    }
}

#endif /* WIFI_PHY_STATE_H */
