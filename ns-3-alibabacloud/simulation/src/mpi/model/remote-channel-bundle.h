/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright 2013. Lawrence Livermore National Security, LLC.
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
 * Author: Steven Smith <smith84@llnl.gov>
 *
 */

/**
 * \file
 * \ingroup mpi
 * Declaration of class ns3::RemoteChannelBundle.
 */

#ifndef NS3_REMOTE_CHANNEL_BUNDLE
#define NS3_REMOTE_CHANNEL_BUNDLE

#include "null-message-simulator-impl.h"

#include <ns3/channel.h>
#include <ns3/ptr.h>
#include <ns3/pointer.h>

#include <unordered_map>

namespace ns3 {

/**
 * \ingroup mpi
 * 
 * \brief Collection of ns-3 channels between local and remote nodes.
 * 
 * An instance exists for each remote system that the local system is
 * in communication with.  These are created and managed by the
 * RemoteChannelBundleManager class.  Stores time information for each
 * bundle.
 */
class RemoteChannelBundle : public Object
{
public:
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /** Default constructor. */
  RemoteChannelBundle ();

  /**
   * Construct and assing system Id.
   * \param [in] remoteSystemId The system id.
   */
  RemoteChannelBundle (const uint32_t remoteSystemId);

  /** Destructor. */
  ~RemoteChannelBundle ()
  {
  }

  /**
   * Add a channel to this bundle.
   * \param channel to add to the bundle
   * \param delay time for the channel (usually the latency)
   */
  void AddChannel (Ptr<Channel> channel, Time delay);

  /**
   * Get the system Id for this side.
   * \return SystemID for remote side of this bundle
   */
  uint32_t GetSystemId () const;

  /**
   * Get the current guarantee time for this bundle.
   * \return guarantee time
   */
  Time GetGuaranteeTime (void) const;

  /**
   * \param time The guarantee time.
   *
   * Set the guarantee time for the bundle.  This should be called
   * after a packet or Null Message received.
   */
  void SetGuaranteeTime (Time time);

  /**
   * Get the minimum delay along any channel in this bundle
   * \return The minimum delay.
   */
  Time GetDelay (void) const;

  /**
   * Set the event ID of the Null Message send event currently scheduled
   * for this channel.
   *
   * \param [in] id The null message event id.
   */
  void SetEventId (EventId id);

  /**
   * Get the event ID of the Null Message send event for this bundle
   * \return The null message event id.
   */
  EventId GetEventId (void) const;

  /**
   * Get the number of ns-3 channels in this bundle
   * \return The number of channels.
   */
  std::size_t GetSize (void) const;

  /**
   * \param time The delay from now when the null message should be received.
   *
   * Send Null Message to the remote task associated with this bundle.
   * Message will be delivered at current simulation time + the time
   * passed in.
   */
  void Send(Time time);

  /**
   * Output for debugging purposes.
   *
   * \param [in,out] out The stream.
   * \param [in] bundle The bundle to print.
   * \return The stream.
   */
  friend std::ostream& operator<< (std::ostream& out, ns3::RemoteChannelBundle& bundle );

private:
  /** Remote rank. */
  uint32_t m_remoteSystemId;

  /**
   * Container of channels that are connected from nodes in this MPI task
   * to nodes in a remote rank.
   */
  typedef std::unordered_map < uint32_t, Ptr < Channel > > ChannelMap;
  ChannelMap m_channels; /**< ChannelId to Channel map */

  /**
   * Guarantee time for the incoming Channels from MPI task remote_rank.
   * No PacketMessage will ever arrive on any incoming channel
   * in this bundle with a ReceiveTime less than this.
   * Initialized to StartTime.
   */
  Time m_guaranteeTime;

  /**
   * Delay for this Channel bundle, which is
   * the min link delay over all incoming channels;
   */
  Time m_delay;

  /** Event scheduled to send Null Message for this bundle. */
  EventId m_nullEventId;

};

}

#endif
