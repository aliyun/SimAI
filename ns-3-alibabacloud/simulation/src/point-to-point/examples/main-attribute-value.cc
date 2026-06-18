/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
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
 * Author: Tom Henderson <tomh@tomh.org>
 */

#include "ns3/log.h"
#include "ns3/command-line.h"
#include "ns3/ptr.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"

#include "ns3/node.h"
#include "ns3/queue.h"
#include "ns3/drop-tail-queue.h"
#include "ns3/point-to-point-net-device.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("AttributeValueSample");

//
// This is a basic example of how to use the attribute system to
// set and get a value in the underlying system; namely, the maximum
// size of the FIFO queue in the PointToPointNetDevice
//

int 
main (int argc, char *argv[])
{
  LogComponentEnable ("AttributeValueSample", LOG_LEVEL_INFO);

  // Queues in ns-3 are objects that hold items (other objects) in 
  // a queue structure.  The C++ implementation uses templates to
  // allow queues to hold various types of items, but the most
  // common is a pointer to a packet (Ptr<Packet>).
  //
  // The maximum queue size can either be enforced in bytes ('b') or
  // packets ('p').  A special type called the ns3::QueueSize can
  // hold queue size values in either unit (bytes or packets).  The
  // DropTailQueue<Packet> class has a MaxSize attribute that can
  // be set to a QueueSize.

  // By default, the MaxSize attribute has a value of 100 packets ('100p')
  // (this default can be observed in the function DropTail<Item>::GetTypeId)
  // 
  // Here, we set it to 80 packets.  We could use one of two value types:
  // a string-based value or a QueueSizeValue value
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize", StringValue ("80p"));
  // The below function call is redundant
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, 80)));

  // Allow the user to override any of the defaults and the above
  // SetDefaults() at run-time, via command-line arguments
  // For example, via "--ns3::DropTailQueue<Packet>::MaxSize=80p"
  CommandLine cmd (__FILE__);
  // This provides yet another way to set the value from the command line:
  cmd.AddValue ("maxSize", "ns3::DropTailQueue<Packet>::MaxSize");
  cmd.Parse (argc, argv);

  // Now, we will create a few objects using the low-level API
  Ptr<Node> n0 = CreateObject<Node> ();

  Ptr<PointToPointNetDevice> net0 = CreateObject<PointToPointNetDevice> ();
  n0->AddDevice (net0);

  Ptr<Queue<Packet> > q = CreateObject<DropTailQueue<Packet> > ();
  net0->SetQueue (q);

  // At this point, we have created a single node (Node 0) and a 
  // single PointToPointNetDevice (NetDevice 0) and added a 
  // DropTailQueue to it.

  // Now, we can manipulate the MaxSize value of the already 
  // instantiated DropTailQueue.  Here are various ways to do that.

  // We assume that a smart pointer (Ptr) to a relevant network device
  // is in hand; here, it is the net0 pointer. 

  // 1.  Pointer-based access
  //
  // One way to change the value is to access a pointer to the
  // underlying queue and modify its attribute.
  // 
  // First, we observe that we can get a pointer to the (base class)
  // queue via the PointToPointNetDevice attributes, where it is called
  // TxQueue 
  PointerValue ptr;
  net0->GetAttribute ("TxQueue", ptr);
  Ptr<Queue<Packet> > txQueue = ptr.Get<Queue<Packet> > ();

  // Using the GetObject function, we can perform a safe downcast
  // to a DropTailQueue
  Ptr<DropTailQueue<Packet> > dtq = txQueue->GetObject <DropTailQueue<Packet> > ();
  NS_ASSERT (dtq);

  // Next, we can get the value of an attribute on this queue
  // We have introduced wrapper "Value" classes for the underlying
  // data types, similar to Java wrappers around these types, since
  // the attribute system stores values and not disparate types.
  // Here, the attribute value is assigned to a QueueSizeValue, and
  // the Get() method on this value produces the (unwrapped) QueueSize.
  QueueSizeValue limit;
  dtq->GetAttribute ("MaxSize", limit);
  NS_LOG_INFO ("1.  dtq limit: " << limit.Get ());

  // Note that the above downcast is not really needed; we could have
  // done the same using the Ptr<Queue> even though the attribute
  // is a member of the subclass
  txQueue->GetAttribute ("MaxSize", limit);
  NS_LOG_INFO ("2.  txQueue limit: " << limit.Get ());

  // Now, let's set it to another value (60 packets).  Let's also make
  // use of the StringValue shorthand notation to set the size by
  // passing in a string (the string must be a positive integer suffixed
  // by either the 'p' or 'b' character).
  txQueue->SetAttribute ("MaxSize", StringValue ("60p"));
  txQueue->GetAttribute ("MaxSize", limit);
  NS_LOG_INFO ("3.  txQueue limit changed: " << limit.Get ());

  // 2.  Namespace-based access
  //
  // An alternative way to get at the attribute is to use the configuration
  // namespace.  Here, this attribute resides on a known path in this
  // namespace; this approach is useful if one doesn't have access to
  // the underlying pointers and would like to configure a specific
  // attribute with a single statement.
  Config::Set ("/NodeList/0/DeviceList/0/TxQueue/MaxSize", StringValue ("25p"));
  txQueue->GetAttribute ("MaxSize", limit); 
  NS_LOG_INFO ("4.  txQueue limit changed through namespace: " << 
               limit.Get ());

  // we could have also used wildcards to set this value for all nodes
  // and all net devices (which in this simple example has the same
  // effect as the previous Set())
  Config::Set ("/NodeList/*/DeviceList/*/TxQueue/MaxSize", StringValue ("15p"));
  txQueue->GetAttribute ("MaxSize", limit); 
  NS_LOG_INFO ("5.  txQueue limit changed through wildcarded namespace: " << 
               limit.Get ());

  Simulator::Destroy ();
}
