/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef DEFAULT_CHANNEL_SCHEDULER_H
#define DEFAULT_CHANNEL_SCHEDULER_H

#include "wave-net-device.h"
namespace ns3 {
class WaveNetDevice;

/**
 * \ingroup wave
 * \brief This class uses a simple mechanism to assign channel access with following features:
 * (1) only in the context of single-PHY device;
 * (2) FCFS (First come First service) strategy, which seems against the description of
 * the standard (preemptive strategy).
 */
class DefaultChannelScheduler : public ChannelScheduler
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  DefaultChannelScheduler ();
  virtual ~DefaultChannelScheduler ();

  /**
   * \param device enable channel scheduler associated with WaveNetDevice
   */
  virtual void SetWaveNetDevice (Ptr<WaveNetDevice> device);
  /**
   * \param channelNumber the specified channel number
   * \return  the type of current assigned channel access for the specific channel.
   */
  virtual enum ChannelAccess GetAssignedAccessType (uint32_t channelNumber) const;

  /**
   * Notify CCH slot start
   * \param duration the CCH slot duration
   */
  void NotifyCchSlotStart (Time duration);
  /**
   * Notify SCH slot start
   * \param duration the SCH slot duration
   */
  void NotifySchSlotStart (Time duration);
  /**
   * Notify guard slot start
   * \param duration the CCH slot duration
   * \param cchi if true, switch to next channel
   */
  void NotifyGuardSlotStart (Time duration, bool cchi);
private:
  virtual void DoInitialize (void);
  virtual void DoDispose (void);
  /**
   * \param channelNumber the specific channel
   * \param immediate indicate whether channel switch to channel
   * \return whether the channel access is assigned successfully
   *
   * This method will assign alternating access for SCHs and CCH.
   */
  virtual bool AssignAlternatingAccess (uint32_t channelNumber, bool immediate);
  /**
   * \param channelNumber the specific channel
   * \param immediate indicate whether channel switch to channel
   * \return whether the channel access is assigned successfully
   *
   * This method will assign continuous SCH access CCH.
   */
  virtual bool AssignContinuousAccess (uint32_t channelNumber, bool immediate);
  /**
   * \param channelNumber the specific channel
   * \param extends extension duration
   * \param immediate indicate whether channel switch to channel
   * \return whether the channel access is assigned successfully
   *
   * This method will assign extended SCH access for SCHs.
   */
  virtual bool AssignExtendedAccess (uint32_t channelNumber, uint32_t extends, bool immediate);
  /**
   * This method will assign default CCH access for CCH.
   * \return whether the channel access is assigned successfully
   */
  virtual bool AssignDefaultCchAccess (void);
  /**
   * \param channelNumber indicating for which channel should release
   * the assigned channel access resource.
   * \return whether the channel access is released successfully
   */
  virtual bool ReleaseAccess (uint32_t channelNumber);
  /**
   * \param curChannelNumber switch from MAC activity for current channel
   * \param nextChannelNumber switch to MAC activity for next channel
   */
  void SwitchToNextChannel (uint32_t curChannelNumber, uint32_t nextChannelNumber);

  Ptr<ChannelManager> m_manager; ///< channel manager
  Ptr<ChannelCoordinator> m_coordinator; ///< channel coordinator
  Ptr<WifiPhy> m_phy; ///< Phy

  /**
   *  when m_channelAccess is ContinuousAccess, m_channelNumber
   *   is continuous channel number;
   *  when m_channelAccess is AlternatingAccess, m_channelNumber
   *   is SCH channel number, another alternating channel is CCH;
   *  when m_channelAccess is ExtendedAccess, m_channelNumber
   *   is extended access, extends is the number of extends access.
   *  when m_channelAccess is DefaultCchAccess, m_channelNumber is CCH.
   */
  uint32_t m_channelNumber; ///< channel number
  uint32_t m_extend; ///< extend
  EventId m_extendEvent; ///< extend event
  enum ChannelAccess m_channelAccess; ///< channel access

  EventId m_waitEvent; ///< wait event
  uint32_t m_waitChannelNumber; ///< wait channel number
  uint32_t m_waitExtend; ///< wait extend

  Ptr<ChannelCoordinationListener> m_coordinationListener; ///< coordination listener
};

}
#endif /* DEFAULT_CHANNEL_SCHEDULER_H */
