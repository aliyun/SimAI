/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c)  2010 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef GENERIC_PHY_H
#define GENERIC_PHY_H

#include <ns3/callback.h>

namespace ns3 {


/**
 * This method allows the MAC to instruct the PHY to start a
 * transmission of a given packet
 *
 * @param packet the Packet to be transmitted
 * @return this method returns false if the PHY will start TX,
 * true if the PHY refuses to start the TX. If false, the MAC layer
 * will expect that GenericPhyTxEndCallback is invoked at some point later.
 */
typedef Callback< bool, Ptr<Packet> > GenericPhyTxStartCallback;

/**
 * this method is invoked by the PHY to notify the MAC that the
 * transmission of a given packet has been completed.
 *
 * @param packet the Packet whose TX has been completed.
 */
typedef Callback< void, Ptr<const Packet> > GenericPhyTxEndCallback;

/**
 * This method is used by the PHY to notify the MAC that a RX
 * attempt is being started, i.e., a valid signal has been
 * recognized by the PHY.
 *
 */
typedef Callback< void > GenericPhyRxStartCallback;

/**
 * This method is used by the PHY to notify the MAC that a
 * previously started RX attempt has terminated without success.
 */
typedef Callback< void > GenericPhyRxEndErrorCallback;

/**
 * This method is used by the PHY to notify the MAC that a
 * previously started RX attempt has been successfully completed.
 *
 * @param packet the received Packet
 */
typedef Callback< void, Ptr<Packet> > GenericPhyRxEndOkCallback;

} // namespace ns3


#endif /* GENERIC_PHY_H */
