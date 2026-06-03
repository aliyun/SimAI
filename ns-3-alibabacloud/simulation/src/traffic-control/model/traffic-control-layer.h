/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
 *               2016 Stefano Avallone <stavallo@unina.it>
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
 */
#ifndef TRAFFICCONTROLLAYER_H
#define TRAFFICCONTROLLAYER_H

#include "ns3/object.h"
#include "ns3/address.h"
#include "ns3/net-device.h"
#include "ns3/traced-callback.h"
#include "ns3/node.h"
#include "ns3/queue-item.h"
#include <map>
#include <vector>

namespace ns3 {

class Packet;
class QueueDisc;
class NetDeviceQueueInterface;

/**
 * \defgroup traffic-control
 *
 * The Traffic Control layer aims at introducing an equivalent of the Linux Traffic
 * Control infrastructure into ns-3. The Traffic Control layer sits in between
 * the NetDevices (L2) and any network protocol (e.g., IP). It is in charge of
 * processing packets and performing actions on them: scheduling, dropping,
 * marking, policing, etc.
 *
 * \ingroup traffic-control
 *
 * \brief Traffic control layer class
 *
 * This object represents the main interface of the Traffic Control Module.
 * Basically, we manage both IN and OUT directions (sometimes called RX and TX,
 * respectively). The OUT direction is easy to follow, since it involves
 * direct calls: upper layer (e.g. IP) calls the Send method on an instance of
 * this class, which then calls the Enqueue method of the QueueDisc associated
 * with the device. The Dequeue method of the QueueDisc finally calls the Send
 * method of the NetDevice.
 *
 * The IN direction uses a little trick to reduce dependencies between modules.
 * In simple words, we use Callbacks to connect upper layer (which should register
 * their Receive callback through RegisterProtocolHandler) and NetDevices.
 *
 * An example of the IN connection between this layer and IP layer is the following:
 *\verbatim
  Ptr<TrafficControlLayer> tc = m_node->GetObject<TrafficControlLayer> ();

  NS_ASSERT (tc != 0);

  m_node->RegisterProtocolHandler (MakeCallback (&TrafficControlLayer::Receive, tc),
                                   Ipv4L3Protocol::PROT_NUMBER, device);
  m_node->RegisterProtocolHandler (MakeCallback (&TrafficControlLayer::Receive, tc),
                                   ArpL3Protocol::PROT_NUMBER, device);

  tc->RegisterProtocolHandler (MakeCallback (&Ipv4L3Protocol::Receive, this),
                               Ipv4L3Protocol::PROT_NUMBER, device);
  tc->RegisterProtocolHandler (MakeCallback (&ArpL3Protocol::Receive, PeekPointer (GetObject<ArpL3Protocol> ())),
                               ArpL3Protocol::PROT_NUMBER, device);
  \endverbatim
 * On the node, for IPv4 and ARP packet, is registered the
 * TrafficControlLayer::Receive callback. At the same time, on the TrafficControlLayer
 * object, is registered the callbacks associated to the upper layers (IPv4 or ARP).
 *
 * When the node receives an IPv4 or ARP packet, it calls the Receive method
 * on TrafficControlLayer, that calls the right upper-layer callback once it
 * finishes the operations on the packet received.
 *
 * Discrimination through callbacks (in other words: what is the right upper-layer
 * callback for this packet?) is done through checks over the device and the
 * protocol number.
 */
class TrafficControlLayer : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Get the type ID for the instance
   * \return the instance TypeId
   */
  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * \brief Constructor
   */
  TrafficControlLayer ();

  virtual ~TrafficControlLayer ();

  // Delete copy constructor and assignment operator to avoid misuse
  TrafficControlLayer (TrafficControlLayer const &) = delete;
  TrafficControlLayer & operator = (TrafficControlLayer const &) = delete;

  /**
   * \brief Register an IN handler
   *
   * The handler will be invoked when a packet is received to pass it to
   * upper  layers.
   *
   * \param handler the handler to register
   * \param protocolType the type of protocol this handler is
   *        interested in. This protocol type is a so-called
   *        EtherType, as registered here:
   *        http://standards.ieee.org/regauth/ethertype/eth.txt
   *        the value zero is interpreted as matching all
   *        protocols.
   * \param device the device attached to this handler. If the
   *        value is zero, the handler is attached to all
   *        devices.
   */
  void RegisterProtocolHandler (Node::ProtocolHandler handler,
                                uint16_t protocolType,
                                Ptr<NetDevice> device);

  /// Typedef for queue disc vector
  typedef std::vector<Ptr<QueueDisc> > QueueDiscVector;

  /**
   * \brief Collect information needed to determine how to handle packets
   *        destined to each of the NetDevices of this node
   *
   * Checks whether a NetDeviceQueueInterface objects is aggregated to each of
   * the NetDevices of this node and sets the required callbacks properly.
   */
  virtual void ScanDevices (void);

  /**
   * \brief This method can be used to set the root queue disc installed on a device
   *
   * \param device the device on which the provided queue disc will be installed
   * \param qDisc the queue disc to be installed as root queue disc on device
   */
  virtual void SetRootQueueDiscOnDevice (Ptr<NetDevice> device, Ptr<QueueDisc> qDisc);

  /**
   * \brief This method can be used to get the root queue disc installed on a device
   *
   * \param device the device on which the requested root queue disc is installed
   * \return the root queue disc installed on the given device
   */
  virtual Ptr<QueueDisc> GetRootQueueDiscOnDevice (Ptr<NetDevice> device) const;

  /**
   * \brief This method can be used to remove the root queue disc (and associated
   *        filters, classes and queues) installed on a device
   *
   * \param device the device on which the installed queue disc will be deleted
   */
  virtual void DeleteRootQueueDiscOnDevice (Ptr<NetDevice> device);

  /**
   * \brief Set node associated with this stack.
   * \param node node to set
   */
  void SetNode (Ptr<Node> node);

  /**
   * \brief Called by NetDevices, incoming packet
   *
   * After analyses and scheduling, this method will call the right handler
   * to pass the packet up in the stack.
   *
   * \param device network device
   * \param p the packet
   * \param protocol next header value
   * \param from address of the correspondent
   * \param to address of the destination
   * \param packetType type of the packet
   */
  virtual void Receive (Ptr<NetDevice> device, Ptr<const Packet> p,
                        uint16_t protocol, const Address &from,
                        const Address &to, NetDevice::PacketType packetType);
  /**
   * \brief Called from upper layer to queue a packet for the transmission.
   *
   * \param device the device the packet must be sent to
   * \param item a queue item including a packet and additional information
   */
  virtual void Send (Ptr<NetDevice> device, Ptr<QueueDiscItem> item);

protected:

  virtual void DoDispose (void);
  virtual void DoInitialize (void);
  virtual void NotifyNewAggregate (void);

private:
  /**
   * \brief Protocol handler entry.
   * This structure is used to demultiplex all the protocols.
   */
  struct ProtocolHandlerEntry {
    Node::ProtocolHandler handler; //!< the protocol handler
    Ptr<NetDevice> device;         //!< the NetDevice
    uint16_t protocol;             //!< the protocol number
    bool promiscuous;              //!< true if it is a promiscuous handler
  };

  /**
   * \brief Information to store for each device
   */
  struct NetDeviceInfo
  {
    Ptr<QueueDisc> m_rootQueueDisc;       //!< the root queue disc on the device
    Ptr<NetDeviceQueueInterface> m_ndqi;  //!< the netdevice queue interface
    QueueDiscVector m_queueDiscsToWake;   //!< the vector of queue discs to wake
  };

  /// Typedef for protocol handlers container
  typedef std::vector<struct ProtocolHandlerEntry> ProtocolHandlerList;

  /**
   * \brief Required by the object map accessor
   * \return the number of devices in the m_netDevices map
   */
  uint32_t GetNDevices (void) const;
  /**
   * \brief Required by the object map accessor
   * \param index the index of the device in the node's device list
   * \return the root queue disc installed on the specified device
   */
  Ptr<QueueDisc> GetRootQueueDiscOnDeviceByIndex (uint32_t index) const;

  /// The node this TrafficControlLayer object is aggregated to
  Ptr<Node> m_node;
  /// Map storing the required information for each device with a queue disc installed
  std::map<Ptr<NetDevice>, NetDeviceInfo> m_netDevices;
  ProtocolHandlerList m_handlers;  //!< List of upper-layer handlers

  /**
   * The trace source fired when the Traffic Control layer drops a packet because
   * no queue disc is installed on the device, the device supports flow control and
   * the device queue is stopped
   */
  TracedCallback<Ptr<const Packet>> m_dropped;
};

} // namespace ns3

#endif // TRAFFICCONTROLLAYER_H
