/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#ifndef LTE_SIMPLE_NET_DEVICE_H
#define LTE_SIMPLE_NET_DEVICE_H

#include "ns3/event-id.h"
#include "ns3/node.h"
#include "ns3/error-model.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/lte-rlc.h"

namespace ns3 {


/**
 * \ingroup lte
 * The LteSimpleNetDevice class implements the LTE simple net device. 
 * This class is used to provide a limited LteNetDevice functionalities that 
 * are necessary for testing purposes.
 */
class LteSimpleNetDevice : public SimpleNetDevice
{

public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  LteSimpleNetDevice (void);
  /**
   * Constructor
   *
   * \param node the Node
   */
  LteSimpleNetDevice (Ptr<Node> node);

  virtual ~LteSimpleNetDevice (void);
  virtual void DoDispose ();


  // inherited from NetDevice
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);


protected:
  // inherited from Object
  virtual void DoInitialize (void);

};

} // namespace ns3


#endif // LTE_SIMPLE_NET_DEVICE_H
