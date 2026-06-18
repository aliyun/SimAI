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
#ifndef CHANNEL_COORDINATOR_H
#define CHANNEL_COORDINATOR_H
#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/event-id.h"

namespace ns3 {

/**
 * \ingroup wave
 * \brief receive notifications about channel coordination events.
 */
class ChannelCoordinationListener : public SimpleRefCount<ChannelCoordinationListener>
{
public:
  virtual ~ChannelCoordinationListener (void);
  /**
   * \param duration the CCH access time (CCHI - GI) continues,
   */
  virtual void NotifyCchSlotStart (Time duration) = 0;
  /**
   * \param duration the SCH access time (SCHI - GI) continues,
   */
  virtual void NotifySchSlotStart (Time duration) = 0;
  /**
   * \param duration the time Guard access time (GI) continues
   * \param cchi indicate whether the guard slot is in the GI of CCHI or SCHI.
   */
  virtual void NotifyGuardSlotStart (Time duration, bool cchi) = 0;
};
/**
 * \ingroup wave
 * \brief ChannelCoordinator deals with channel coordination in data plane (see 1609.4 chapter 5.2)
 *
 *      <------------   -----SYNCI-----------------> <-------------------SYNCI------------------>
 *                   CCHI                      SCHI                         CCHI                     SCHI
 *     |..************|..************|..************|..************|
 *      GI                             GI                            GI                             GI
 *
 *  The relation among CCHI(CCH Interval), SCHI(SCH Interval), GI(Guard Interval), SYNCI(Sync Interval):
 *  1. they are all time durations, gegerally default CCHI is 50ms,
 *  default SCHI is 50ms, default GI is 4ms, default SYNCI is 100ms.
 *  2. Every UTC second shall be an integer number of sync interval, and the
 *  beginning of a sync interval shall align with beginning of UTC second;
 *  3. SYNCI is the sum of CCHI and SCHI.
 *  GI is the sum of SyncTolerance and MaxSwitchTime defined for
 *  multi-channel synchronization (see 1609.4 chapter 6.2). And since all
 *  devices the in simulation could be supposed to synchronize by "perfect GPS",
 *  here the synchronization mechanism defined in the standard will be not implemented.
 *  4. Although the real channel switch time of wifi PHY layer is very fast, and the "ChannelSwitchDelay"
 *  of YansWifiPhy is 250 microseconds, here in 4ms guard interval WAVE devices
 *  cannot transmit packets while may receive packets.
 */
class ChannelCoordinator : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  ChannelCoordinator ();
  virtual ~ChannelCoordinator ();

  /**
   * \return the default control channel interval for multi-channel operation (50 milliseconds).
   */
  static Time GetDefaultCchInterval (void);
  /**
   * \return the default service channel interval for multi-channel operation (50 milliseconds).
   */
  static Time GetDefaultSchInterval (void);
  /**
   * \return the default sync interval for multi-channel operation (100 milliseconds).
   */
  static Time GetDefaultSyncInterval (void);
  /**
   * \return the default guard channel interval for multi-channel operation (4 milliseconds).
   */
  static Time GetDefaultGuardInterval (void);

  /**
   * \param cchi the CCH interval for multi-channel operation.
   */
  void SetCchInterval (Time cchi);
  /**
   * \return the CCH interval for multi-channel operation.
   */
  Time GetCchInterval (void) const;
  /**
   * \param schi the SCH interval for multi-channel operation.
   */
  void SetSchInterval (Time schi);
  /**
   * \return the SCH interval for multi-channel operation.
   */
  Time GetSchInterval (void) const;
  /**
   * \return the Sync interval for multi-channel operation.
   */
  Time GetSyncInterval (void) const;
  /**
   * \param guardi the guard interval for multi-channel operation.
   */
  void SetGuardInterval (Time guardi);
  /**
   * \return the guard interval for multi-channel operation.
   */
  Time GetGuardInterval (void) const;
  /**
   * \return whether current channel interval configuration is valid.
   *
   * If users set the channel intervals different from default values here,
   * it should be better to test whether the configuration is valid.
   */
  bool IsValidConfig (void) const;

  /**
   * \param duration the future time after duration
   * \returns whether future time is in CCH Interval
   */
  bool IsCchInterval (Time duration = Seconds (0.0)) const;
  /**
   * \param duration the future time after duration
   * \returns whether future time is in SCH Interval
   */
  bool IsSchInterval (Time duration = Seconds (0.0)) const;
  /**
   * \param duration the future time after duration
   * \returns whether future time is in Guard Interval
   */
  bool IsGuardInterval (Time duration = Seconds (0.0)) const;
  /**
   * \param duration the future time after duration
   * \returns the time to the next CCH interval.
   * If current time is already in CCH interval, return 0;
   */
  Time NeedTimeToCchInterval (Time duration = Seconds (0.0)) const;
  /**
   * \param duration the future time after duration
   * \returns the duration time to the next SCH interval.
   * If current time is already in SCH interval, return 0;
   */
  Time NeedTimeToSchInterval (Time duration = Seconds (0.0)) const;
  /**
   * \param duration the future time after duration
   * \returns the duration time to next Guard Interval;
   * return 0 if current time is already in Guard Interval, ;
   */
  Time NeedTimeToGuardInterval (Time duration = Seconds (0.0)) const;
  /**
   * \param duration the future time after duration
   * \return the time in a Sync Interval of future time
   *  for example:
   *  SyncInterval = 100ms;
   *  Now = 5s20ms;
   *  duration = 50ms;
   *  then GetIntervalTime (duration) = 70ms.
   */
  Time GetIntervalTime (Time duration = Seconds (0.0)) const;
  /**
   * \param duration the future time after duration
   * \return the remain time in a Sync Interval of future time
   *  for example:
   *  SyncInterval = 100ms;
   *  Now = 5s20ms;
   *  duration = 50ms;
   *  then GetRemainTime (duration) = 30ms.
   */
  Time GetRemainTime (Time duration = Seconds (0.0)) const;
  /**
   * \param listener the new listener for channel coordination events.
   *
   * Add the input listener to the list of objects to be notified of
   * channel coordination events.
   */
  void RegisterListener (Ptr<ChannelCoordinationListener> listener);
  /**
   * \param listener the current attached listener
   *
   * Remove the specified listener.
   */
  void UnregisterListener (Ptr<ChannelCoordinationListener> listener);
  /**
   * Remove all listeners.
   */
  void UnregisterAllListeners (void);

private:
  virtual void DoDispose (void);
  virtual void DoInitialize (void);

  /**
   * start to make channel coordination events
   */
  void StartChannelCoordination (void);
  /**
   * stop channel coordination events
   */
  void StopChannelCoordination (void);
  /**
   * notify listeners of a SCH slot start
   */
  void NotifySchSlot (void);
  /**
   * notify listeners of a CCH slot start
   */
  void NotifyCchSlot (void);
  /**
   * notify listeners of a guard slot start
   */
  void NotifyGuardSlot (void);
  /**
   * \return SCH channel access time which is SchInterval - GuardInterval, default 46ms
   */
  Time GetSchSlot (void) const;
  /**
   * \return CCH channel access time which is SchInterval - GuardInterval, default 46ms
   */
  Time GetCchSlot (void) const;

  Time m_cchi;  ///< CchInterval
  Time m_schi;  ///< SchInterval
  Time m_gi;    ///< GuardInterval

  /// Listeners typdef
  typedef std::vector<Ptr<ChannelCoordinationListener> > Listeners;
  /// Listeners iterator typedef
  typedef std::vector<Ptr<ChannelCoordinationListener> >::iterator ListenersI;
  Listeners m_listeners; ///< listeners

  uint32_t m_guardCount; ///< guard count
  EventId m_coordination; ///< coordination event
};

}
#endif /* CHANNEL_COORDINATOR_H */
