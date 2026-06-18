/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */
#ifndef TRAFFIC_CONTROL_HELPER_H
#define TRAFFIC_CONTROL_HELPER_H

#include <string>
#include <vector>
#include <map>
#include "ns3/object-factory.h"
#include "ns3/net-device-container.h"
#include "ns3/queue-disc-container.h"
#include "ns3/queue.h"

namespace ns3 {

/**
 * \ingroup traffic-control
 *
 * This class stores object factories required to create a queue disc and all of
 * its components (packet filters, internal queues, classes).
 */
class QueueDiscFactory
{
public:
  /**
   * \brief Constructor
   *
   * \param factory the factory used to create this queue disc
   */
  QueueDiscFactory (ObjectFactory factory);

  virtual ~QueueDiscFactory () {}

  // Delete default constructor to avoid misuse
  QueueDiscFactory () = delete;

  /**
   * \brief Add a factory to create an internal queue
   *
   * \param factory the factory used to create an internal queue
   */
  void AddInternalQueue (ObjectFactory factory);

  /**
   * \brief Add a factory to create a packet filter
   *
   * \param factory the factory used to create a packet filter
   */
  void AddPacketFilter (ObjectFactory factory);

  /**
   * \brief Add a factory to create a queue disc class
   *
   * \param factory the factory used to create a queue disc class
   * \return the class ID of the created queue disc class
   */
  uint16_t AddQueueDiscClass (ObjectFactory factory);

  /**
   * \brief Set the (child) queue disc to attach to a class
   *
   * \param classId the id of the class to attach a child queue disc to
   * \param handle the handle of the child queue disc to attach to the class
   */
  void SetChildQueueDisc (uint16_t classId, uint16_t handle);

  /**
   * \brief Create a queue disc with the currently stored configuration.
   *
   * \param queueDiscs the vector of queue discs held by the helper
   * \return the created queue disc
   */
  Ptr<QueueDisc> CreateQueueDisc (const std::vector<Ptr<QueueDisc> > & queueDiscs);

private:
  /// Factory to create this queue disc
  ObjectFactory m_queueDiscFactory;
  /// Vector of factories to create internal queues
  std::vector<ObjectFactory> m_internalQueuesFactory;
  /// Vector of factories to create packet filters
  std::vector<ObjectFactory> m_packetFiltersFactory;
  /// Vector of factories to create queue disc classes
  std::vector<ObjectFactory> m_queueDiscClassesFactory;
  /// Map storing the associations between class IDs and child queue disc handles
  std::map<uint16_t, uint16_t> m_classIdChildHandleMap;
};


/**
 * \ingroup traffic-control
 *
 * \brief Build a set of QueueDisc objects
 *
 * This class can help to create QueueDisc objects and map them to
 * the corresponding devices. This map is stored at the Traffic Control
 * layer.
 */
class TrafficControlHelper
{
public:
  /**
   * Create a TrafficControlHelper to make life easier when creating QueueDisc
   * objects.
   */
  TrafficControlHelper ();
  virtual ~TrafficControlHelper () {}

  /**
   * \param nTxQueues the number of Tx queue disc classes
   * \returns a new TrafficControlHelper with a default configuration
   *
   * The default configuration is an FqCoDelQueueDisc, if the device has a single
   * queue, or an MqQueueDisc with as many FqCoDelQueueDiscs as the number of
   * device queues, otherwise.
   */
  static TrafficControlHelper Default (std::size_t nTxQueues = 1);

  /**
   * Helper function used to set a root queue disc of the given type and with the
   * given attributes. To set the InternalQueueList, PacketFilterList and ChildQueueDiscList
   * attributes, use the AddInternalQueue, AddPacketFilter and AddChildQueueDisc methods.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param type the type of queue disc
   * \param args A sequence of name-value pairs of the attributes to set.
   * \return the handle of the root queue disc (zero)
   */
  template <typename... Args>
  uint16_t SetRootQueueDisc (const std::string& type, Args&&... args);

  /**
   * Helper function used to add the given number of internal queues (of the given
   * type and with the given attributes) to the queue disc having the given handle.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param handle the handle of the parent queue disc
   * \param count the number of queues to add
   * \param type the type of queue
   * \param args A sequence of name-value pairs of the attributes to set.
   */
  template <typename... Args>
  void AddInternalQueues (uint16_t handle, uint16_t count, std::string type, Args&&... args);

  /**
   * Helper function used to add a packet filter (of the given type and with
   * the given attributes) to the queue disc having the given handle.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param handle the handle of the parent queue disc
   * \param type the type of packet filter
   * \param args A sequence of name-value pairs of the attributes to set.
   */
  template <typename... Args>
  void AddPacketFilter (uint16_t handle, const std::string& type, Args&&... args);

  /**
   * Container type for Class IDs
   */
  typedef std::vector<uint16_t> ClassIdList;

  /**
   * Helper function used to add the given number of queue disc classes (of the given
   * type and with the given attributes) to the queue disc having the given handle.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param handle the handle of the parent queue disc
   * \param count the number of queue disc classes to add
   * \param type the type of queue disc class
   * \param args A sequence of name-value pairs of the attributes to set.
   * \return the list of class IDs
   */
  template <typename... Args>
  ClassIdList AddQueueDiscClasses (uint16_t handle, uint16_t count, const std::string& type, Args&&... args);

  /**
   * Helper function used to attach a child queue disc (of the given type and with
   * the given attributes) to a given class (included in the queue disc
   * having the given handle).
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param handle the handle of the parent queue disc
   * \param classId the class ID of the class to attach the queue disc to
   * \param type the type of queue disc
   * \param args A sequence of name-value pairs of the attributes to set.
   * \return the handle of the created child queue disc
   */
  template <typename... Args>
  uint16_t AddChildQueueDisc (uint16_t handle, uint16_t classId, const std::string& type, Args&&... args);

  /**
   * Container type for Handlers
   */
  typedef std::vector<uint16_t> HandleList;

  /**
   * Helper function used to attach a child queue disc (of the given type and with
   * the given attributes) to each of the given classes (included in the queue disc
   * having the given handle).
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param handle the handle of the parent queue disc
   * \param classes the class IDs of the classes to attach a queue disc to
   * \param type the type of queue disc
   * \param args A sequence of name-value pairs of the attributes to set.
   * \return the list of handles of the created child queue discs
   */
  template <typename... Args>
  HandleList AddChildQueueDiscs (uint16_t handle, const ClassIdList &classes,
                                 const std::string& type, Args&&... args);

  /**
   * Helper function used to add a queue limits object to the transmission
   * queues of the devices
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
   * \param type the type of queue
   * \param args A sequence of name-value pairs of the attributes to set.
   */
  template <typename... Args>
  void SetQueueLimits (std::string type, Args&&... args);

  /**
   * \param c set of devices
   * \returns a QueueDisc container with the root queue discs installed on the devices
   *
   * This method creates a QueueDisc object of the type and with the
   * attributes configured by TrafficControlHelper::SetQueueDisc for
   * each device in the container. Then, stores the mapping between a
   * device and the associated queue disc into the traffic control layer
   * of the corresponding node.
   * This method creates the queue discs (along with their packet filters,
   * internal queues, classes) configured with the methods provided by this
   * class and installs them on each device in the given container. Additionally,
   * if configured, a queue limits object is installed on each transmission queue
   * of the devices.
   */
  QueueDiscContainer Install (NetDeviceContainer c);

  /**
   * \param d device
   * \returns a QueueDisc container with the root queue disc installed on the device
   *
   * This method creates the queue discs (along with their packet filters,
   * internal queues, classes) configured with the methods provided by this
   * class and installs them on the given device. Additionally, if configured,
   * a queue limits object is installed on each transmission queue of the device.
   */
  QueueDiscContainer Install (Ptr<NetDevice> d);

  /**
   * \param c set of devices
   *
   * This method removes the root queue discs (and associated filters, classes
   * and queues) installed on the given devices.
   * Note that the traffic control layer will continue to perform flow control
   * if the device has an aggregated NetDeviceQueueInterface. If you really
   * want that the Traffic Control layer forwards packets down to the NetDevice
   * even if there is no room for them in the NetDevice queue(s), then disable
   * the flow control by using the DisableFlowControl method of the NetDevice
   * helper.
   */
  void Uninstall (NetDeviceContainer c);

  /**
   * \param d device
   *
   * This method removes the root queue disc (and associated filters, classes
   * and queues) installed on the given device.
   * Note that the traffic control layer will continue to perform flow control
   * if the device has an aggregated NetDeviceQueueInterface. If you really
   * want that the Traffic Control layer forwards packets down to the NetDevice
   * even if there is no room for them in the NetDevice queue(s), then disable
   * the flow control by using the DisableFlowControl method of the NetDevice
   * helper.
   */
  void Uninstall (Ptr<NetDevice> d);

private:
  /**
   * Actual implementation of the SetRootQueueDisc method.
   *
   * \param factory the factory used to create the root queue disc
   * \returns zero on success
   */
  uint16_t DoSetRootQueueDisc (ObjectFactory factory);

  /**
   * Actual implementation of the AddInternalQueues method.
   *
   * \param handle the handle of the parent queue disc
   * \param count the number of queues to add
   * \param factory the factory used to add internal queues
   */
  void DoAddInternalQueues (uint16_t handle, uint16_t count, ObjectFactory factory);

  /**
   * Actual implementation of the AddPacketFilter method.
   *
   * \param handle the handle of the parent queue disc
   * \param factory the factory used to add a packet filter
   */
  void DoAddPacketFilter (uint16_t handle, ObjectFactory factory);

  /**
   * Actual implementation of the AddQueueDiscClasses method.
   *
   * \param handle the handle of the parent queue disc
   * \param count the number of queue disc classes to add
   * \param factory the factory used to add queue disc classes
   * \return the list of class IDs
   */
  ClassIdList DoAddQueueDiscClasses (uint16_t handle, uint16_t count, ObjectFactory factory);

  /**
   * Actual implementation of the AddChildQueueDisc method.
   *
   * \param handle the handle of the parent queue disc
   * \param classId the class ID of the class to attach the queue disc to
   * \param factory the factory used to add a child queue disc
   * \return the handle of the created child queue disc
   */
  uint16_t DoAddChildQueueDisc (uint16_t handle, uint16_t classId, ObjectFactory factory);

  /**
   * Actual implementation of the AddChildQueueDiscs method.
   *
   * \param handle the handle of the parent queue disc
   * \param classes the class IDs of the classes to attach a queue disc to
   * \param factory the factory used to add child queue discs
   * \return the list of handles of the created child queue discs
   */
  HandleList DoAddChildQueueDiscs (uint16_t handle, const ClassIdList &classes, ObjectFactory factory);

  /// QueueDisc factory, stores the configuration of all the queue discs
  std::vector<QueueDiscFactory> m_queueDiscFactory;
  /// Vector of all the created queue discs
  std::vector<Ptr<QueueDisc> > m_queueDiscs;
  /// Factory to create a queue limits object
  ObjectFactory m_queueLimitsFactory;
};

} // namespace ns3


/***************************************************************
 *  Implementation of the templates declared above.
 ***************************************************************/

namespace ns3 {

template <typename... Args>
uint16_t
TrafficControlHelper::SetRootQueueDisc (const std::string& type, Args&&... args)
{
  return DoSetRootQueueDisc (ObjectFactory (type, args...));
}

template <typename... Args>
void
TrafficControlHelper::AddInternalQueues (uint16_t handle, uint16_t count,
                                         std::string type, Args&&... args)
{
  QueueBase::AppendItemTypeIfNotPresent (type, "QueueDiscItem");
  DoAddInternalQueues (handle, count, ObjectFactory (type, args...));
}

template <typename... Args>
void
TrafficControlHelper::AddPacketFilter (uint16_t handle, const std::string& type, Args&&... args)
{
  DoAddPacketFilter (handle, ObjectFactory (type, args...));
}

template <typename... Args>
TrafficControlHelper::ClassIdList
TrafficControlHelper::AddQueueDiscClasses (uint16_t handle, uint16_t count,
                                           const std::string& type, Args&&... args)
{
  return DoAddQueueDiscClasses (handle, count, ObjectFactory (type, args...));
}

template <typename... Args>
uint16_t
TrafficControlHelper::AddChildQueueDisc (uint16_t handle, uint16_t classId,
                                         const std::string& type, Args&&... args)
{
  return DoAddChildQueueDisc (handle, classId, ObjectFactory (type, args...));
}

template <typename... Args>
TrafficControlHelper::HandleList
TrafficControlHelper::AddChildQueueDiscs (uint16_t handle, const ClassIdList &classes,
                                          const std::string& type, Args&&... args)
{
  return DoAddChildQueueDiscs (handle, classes, ObjectFactory (type, args...));
}

template <typename... Args>
void
TrafficControlHelper::SetQueueLimits (std::string type, Args&&... args)
{
  m_queueLimitsFactory.SetTypeId (type);
  m_queueLimitsFactory.Set (args...);
}

} // namespace ns3

#endif /* TRAFFIC_CONTROL_HELPER_H */
