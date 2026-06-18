/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Drexel University
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
 * Author: Joe Kopena (tjkopena@cs.drexel.edu)
 */

#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/mac48-address.h"

#include "ns3/basic-data-calculators.h"
#include "packet-data-calculators.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("PacketDataCalculators");

//--------------------------------------------------------------
//----------------------------------------------
PacketCounterCalculator::PacketCounterCalculator()
{
  NS_LOG_FUNCTION_NOARGS ();
}

PacketCounterCalculator::~PacketCounterCalculator()
{
  NS_LOG_FUNCTION_NOARGS ();
}
/* static */
TypeId
PacketCounterCalculator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PacketCounterCalculator")
    .SetParent<CounterCalculator<uint32_t> > ()
    .SetGroupName ("Network")
    .AddConstructor<PacketCounterCalculator> ()
    ;
  return tid;
}
void
PacketCounterCalculator::DoDispose (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  CounterCalculator<uint32_t>::DoDispose ();
  // PacketCounterCalculator::DoDispose
}

void
PacketCounterCalculator::PacketUpdate (std::string path,
                                       Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION_NOARGS ();

  CounterCalculator<uint32_t>::Update ();

  // PacketCounterCalculator::Update
}
void
PacketCounterCalculator::FrameUpdate (std::string path,
                                      Ptr<const Packet> packet,
                                      Mac48Address realto)
{
  NS_LOG_FUNCTION_NOARGS ();

  CounterCalculator<uint32_t>::Update ();

  // PacketCounterCalculator::Update
}




//--------------------------------------------------------------
//----------------------------------------------
PacketSizeMinMaxAvgTotalCalculator::PacketSizeMinMaxAvgTotalCalculator()
{
  NS_LOG_FUNCTION_NOARGS ();
}

PacketSizeMinMaxAvgTotalCalculator::~PacketSizeMinMaxAvgTotalCalculator()
{
  NS_LOG_FUNCTION_NOARGS ();
}
/* static */
TypeId
PacketSizeMinMaxAvgTotalCalculator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PacketSizeMinMaxAvgTotalCalculator")
    .SetParent<MinMaxAvgTotalCalculator<uint32_t> > ()
    .SetGroupName ("Network")
    .AddConstructor<PacketSizeMinMaxAvgTotalCalculator> ()
    ;
  return tid;
}
void
PacketSizeMinMaxAvgTotalCalculator::DoDispose (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  MinMaxAvgTotalCalculator<uint32_t>::DoDispose ();
  // end PacketSizeMinMaxAvgTotalCalculator::DoDispose
}

void
PacketSizeMinMaxAvgTotalCalculator::PacketUpdate (std::string path,
                                                  Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION_NOARGS ();

  MinMaxAvgTotalCalculator<uint32_t>::Update (packet->GetSize ());

  // end PacketSizeMinMaxAvgTotalCalculator::Update
}
void
PacketSizeMinMaxAvgTotalCalculator::FrameUpdate (std::string path,
                                                 Ptr<const Packet> packet,
                                                 Mac48Address realto)
{
  NS_LOG_FUNCTION_NOARGS ();

  MinMaxAvgTotalCalculator<uint32_t>::Update (packet->GetSize ());

  // end PacketSizeMinMaxAvgTotalCalculator::Update
}
