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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "channel-access-manager.h"
#include "txop.h"
#include "wifi-phy-listener.h"
#include "wifi-phy.h"
#include "frame-exchange-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ChannelAccessManager");

/**
 * Listener for PHY events. Forwards to ChannelAccessManager
 */
class PhyListener : public ns3::WifiPhyListener
{
public:
  /**
   * Create a PhyListener for the given ChannelAccessManager.
   *
   * \param cam the ChannelAccessManager
   */
  PhyListener (ns3::ChannelAccessManager *cam)
    : m_cam (cam)
  {
  }
  virtual ~PhyListener ()
  {
  }
  void NotifyRxStart (Time duration)
  {
    m_cam->NotifyRxStartNow (duration);
  }
  void NotifyRxEndOk (void)
  {
    m_cam->NotifyRxEndOkNow ();
  }
  void NotifyRxEndError (void)
  {
    m_cam->NotifyRxEndErrorNow ();
  }
  void NotifyTxStart (Time duration, double txPowerDbm)
  {
    m_cam->NotifyTxStartNow (duration);
  }
  void NotifyMaybeCcaBusyStart (Time duration)
  {
    m_cam->NotifyMaybeCcaBusyStartNow (duration);
  }
  void NotifySwitchingStart (Time duration)
  {
    m_cam->NotifySwitchingStartNow (duration);
  }
  void NotifySleep (void)
  {
    m_cam->NotifySleepNow ();
  }
  void NotifyOff (void)
  {
    m_cam->NotifyOffNow ();
  }
  void NotifyWakeup (void)
  {
    m_cam->NotifyWakeupNow ();
  }
  void NotifyOn (void)
  {
    m_cam->NotifyOnNow ();
  }

private:
  ns3::ChannelAccessManager *m_cam;  //!< ChannelAccessManager to forward events to
};


/****************************************************************
 *      Implement the channel access manager of all Txop holders
 ****************************************************************/

ChannelAccessManager::ChannelAccessManager ()
  : m_lastAckTimeoutEnd (MicroSeconds (0)),
    m_lastCtsTimeoutEnd (MicroSeconds (0)),
    m_lastNavStart (MicroSeconds (0)),
    m_lastNavDuration (MicroSeconds (0)),
    m_lastRxStart (MicroSeconds (0)),
    m_lastRxDuration (MicroSeconds (0)),
    m_lastRxReceivedOk (true),
    m_lastTxStart (MicroSeconds (0)),
    m_lastTxDuration (MicroSeconds (0)),
    m_lastBusyStart (MicroSeconds (0)),
    m_lastBusyDuration (MicroSeconds (0)),
    m_lastSwitchingStart (MicroSeconds (0)),
    m_lastSwitchingDuration (MicroSeconds (0)),
    m_sleeping (false),
    m_off (false),
    m_phyListener (0)
{
  NS_LOG_FUNCTION (this);
}

ChannelAccessManager::~ChannelAccessManager ()
{
  NS_LOG_FUNCTION (this);
  delete m_phyListener;
  m_phyListener = 0;
}

void
ChannelAccessManager::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  for (Ptr<Txop> i : m_txops)
    {
      i->Dispose ();
      i = 0;
    }
  m_phy = 0;
  m_feManager = 0;
}

void
ChannelAccessManager::SetupPhyListener (Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  NS_ASSERT (m_phyListener == 0);
  m_phyListener = new PhyListener (this);
  phy->RegisterListener (m_phyListener);
  m_phy = phy;
}

void
ChannelAccessManager::RemovePhyListener (Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  if (m_phyListener != 0)
    {
      phy->UnregisterListener (m_phyListener);
      delete m_phyListener;
      m_phyListener = 0;
      m_phy = 0;
    }
}

void
ChannelAccessManager::SetupFrameExchangeManager (Ptr<FrameExchangeManager> feManager)
{
  NS_LOG_FUNCTION (this << feManager);
  m_feManager = feManager;
  m_feManager->SetChannelAccessManager (this);
}

Time
ChannelAccessManager::GetSlot (void) const
{
  return m_phy->GetSlot ();
}

Time
ChannelAccessManager::GetSifs (void) const
{
  return m_phy->GetSifs ();
}

Time
ChannelAccessManager::GetEifsNoDifs () const
{
  return m_phy->GetSifs () + m_phy->GetAckTxTime ();
}

void
ChannelAccessManager::Add (Ptr<Txop> txop)
{
  NS_LOG_FUNCTION (this << txop);
  m_txops.push_back (txop);
}

Time
ChannelAccessManager::MostRecent (std::initializer_list<Time> list) const
{
  NS_ASSERT (list.size () > 0);
  return *std::max_element (list.begin (), list.end ());
}

bool
ChannelAccessManager::IsBusy (void) const
{
  NS_LOG_FUNCTION (this);
  // PHY busy
  Time lastRxEnd = m_lastRxStart + m_lastRxDuration;
  if (lastRxEnd > Simulator::Now ())
    {
      return true;
    }
  Time lastTxEnd = m_lastTxStart + m_lastTxDuration;
  if (lastTxEnd > Simulator::Now ())
    {
      return true;
    }
  // NAV busy
  Time lastNavEnd = m_lastNavStart + m_lastNavDuration;
  if (lastNavEnd > Simulator::Now ())
    {
      return true;
    }
  // CCA busy
  Time lastCCABusyEnd = m_lastBusyStart + m_lastBusyDuration;
  if (lastCCABusyEnd > Simulator::Now ())
    {
      return true;
    }
  return false;
}

bool
ChannelAccessManager::NeedBackoffUponAccess (Ptr<Txop> txop)
{
  NS_LOG_FUNCTION (this << txop);

  // No backoff needed if in sleep mode or off
  if (m_sleeping || m_off)
    {
      return false;
    }

  // the Txop might have a stale value of remaining backoff slots
  UpdateBackoff ();

  /*
   * From section 10.3.4.2 "Basic access" of IEEE 802.11-2016:
   *
   * A STA may transmit an MPDU when it is operating under the DCF access
   * method, either in the absence of a PC, or in the CP of the PCF access
   * method, when the STA determines that the medium is idle when a frame is
   * queued for transmission, and remains idle for a period of a DIFS, or an
   * EIFS (10.3.2.3.7) from the end of the immediately preceding medium-busy
   * event, whichever is the greater, and the backoff timer is zero. Otherwise
   * the random backoff procedure described in 10.3.4.3 shall be followed.
   *
   * From section 10.22.2.2 "EDCA backoff procedure" of IEEE 802.11-2016:
   *
   * The backoff procedure shall be invoked by an EDCAF when any of the following
   * events occurs:
   * a) An MA-UNITDATA.request primitive is received that causes a frame with that AC
   *    to be queued for transmission such that one of the transmit queues associated
   *    with that AC has now become non-empty and any other transmit queues
   *    associated with that AC are empty; the medium is busy on the primary channel
   */
  if (!txop->HasFramesToTransmit () && txop->GetAccessStatus () != Txop::GRANTED
      && txop->GetBackoffSlots () == 0)
    {
      if (!IsBusy ())
        {
          // medium idle. If this is a DCF, use immediate access (we can transmit
          // in a DIFS if the medium remains idle). If this is an EDCAF, update
          // the backoff start time kept by the EDCAF to the current time in order
          // to correctly align the backoff start time at the next slot boundary
          // (performed by the next call to ChannelAccessManager::RequestAccess())
          Time delay = (txop->IsQosTxop () ? Seconds (0)
                                           : GetSifs () + txop->GetAifsn () * GetSlot ());
          txop->UpdateBackoffSlotsNow (0, Simulator::Now () + delay);
        }
      else
        {
          // medium busy, backoff is needed
          return true;
        }
    }
  return false;
}

void
ChannelAccessManager::RequestAccess (Ptr<Txop> txop)
{
  NS_LOG_FUNCTION (this << txop);
  if (m_phy)
    {
      m_phy->NotifyChannelAccessRequested ();
    }
  //Deny access if in sleep mode or off
  if (m_sleeping || m_off)
    {
      return;
    }
  /*
   * EDCAF operations shall be performed at slot boundaries (Sec. 10.22.2.4 of 802.11-2016)
   */
  Time accessGrantStart = GetAccessGrantStart () + (txop->GetAifsn () * GetSlot ());

  if (txop->IsQosTxop () && txop->GetBackoffStart () > accessGrantStart)
    {
      // The backoff start time reported by the EDCAF is more recent than the last
      // time the medium was busy plus an AIFS, hence we need to align it to the
      // next slot boundary.
      Time diff = txop->GetBackoffStart () - accessGrantStart;
      uint32_t nIntSlots = (diff / GetSlot ()).GetHigh () + 1;
      txop->UpdateBackoffSlotsNow (0, accessGrantStart + (nIntSlots * GetSlot ()));
    }

  UpdateBackoff ();
  NS_ASSERT (txop->GetAccessStatus () != Txop::REQUESTED);
  txop->NotifyAccessRequested ();
  DoGrantDcfAccess ();
  DoRestartAccessTimeoutIfNeeded ();
}

void
ChannelAccessManager::DoGrantDcfAccess (void)
{
  NS_LOG_FUNCTION (this);
  uint32_t k = 0;
  for (Txops::iterator i = m_txops.begin (); i != m_txops.end (); k++)
    {
      Ptr<Txop> txop = *i;
      if (txop->GetAccessStatus () == Txop::REQUESTED
          && (!txop->IsQosTxop () || !StaticCast<QosTxop> (txop)->EdcaDisabled ())
          && GetBackoffEndFor (txop) <= Simulator::Now () )
        {
          /**
           * This is the first Txop we find with an expired backoff and which
           * needs access to the medium. i.e., it has data to send.
           */
          NS_LOG_DEBUG ("dcf " << k << " needs access. backoff expired. access granted. slots=" << txop->GetBackoffSlots ());
          i++; //go to the next item in the list.
          k++;
          std::vector<Ptr<Txop> > internalCollisionTxops;
          for (Txops::iterator j = i; j != m_txops.end (); j++, k++)
            {
              Ptr<Txop> otherTxop = *j;
              if (otherTxop->GetAccessStatus () == Txop::REQUESTED
                  && GetBackoffEndFor (otherTxop) <= Simulator::Now ())
                {
                  NS_LOG_DEBUG ("dcf " << k << " needs access. backoff expired. internal collision. slots=" <<
                                otherTxop->GetBackoffSlots ());
                  /**
                   * all other Txops with a lower priority whose backoff
                   * has expired and which needed access to the medium
                   * must be notified that we did get an internal collision.
                   */
                  internalCollisionTxops.push_back (otherTxop);
                }
            }

          /**
           * Now, we notify all of these changes in one go if the EDCAF winning
           * the contention actually transmitted a frame. It is necessary to
           * perform first the calculations of which Txops are colliding and then
           * only apply the changes because applying the changes through notification
           * could change the global state of the manager, and, thus, could change
           * the result of the calculations.
           */
          NS_ASSERT (m_feManager != 0);

          if (m_feManager->StartTransmission (txop))
            {
              for (auto& collidingTxop : internalCollisionTxops)
                {
                  m_feManager->NotifyInternalCollision (collidingTxop);
                }
              break;
            }
          else
            {
              // reset the current state to the EDCAF that won the contention
              // but did not transmit anything
              i--;
              k = std::distance (m_txops.begin (), i);
            }
        }
      i++;
    }
}

void
ChannelAccessManager::AccessTimeout (void)
{
  NS_LOG_FUNCTION (this);
  UpdateBackoff ();
  DoGrantDcfAccess ();
  DoRestartAccessTimeoutIfNeeded ();
}

Time
ChannelAccessManager::GetAccessGrantStart (bool ignoreNav) const
{
  NS_LOG_FUNCTION (this);
  Time lastRxEnd = m_lastRxStart + m_lastRxDuration;
  const Time& sifs = GetSifs();
  Time rxAccessStart = lastRxEnd + sifs;
  if ((lastRxEnd <= Simulator::Now ()) && !m_lastRxReceivedOk)
    {
      rxAccessStart += GetEifsNoDifs ();
    }
  Time busyAccessStart = m_lastBusyStart + m_lastBusyDuration + sifs;
  Time txAccessStart = m_lastTxStart + m_lastTxDuration + sifs;
  Time navAccessStart = m_lastNavStart + m_lastNavDuration + sifs;
  Time ackTimeoutAccessStart = m_lastAckTimeoutEnd + sifs;
  Time ctsTimeoutAccessStart = m_lastCtsTimeoutEnd + sifs;
  Time switchingAccessStart = m_lastSwitchingStart + m_lastSwitchingDuration + sifs;
  Time accessGrantedStart;
  if (ignoreNav)
    {
      accessGrantedStart = MostRecent ({rxAccessStart,
                                        busyAccessStart,
                                        txAccessStart,
                                        ackTimeoutAccessStart,
                                        ctsTimeoutAccessStart,
                                        switchingAccessStart}
                                       );
    }
  else
    {
      accessGrantedStart = MostRecent ({rxAccessStart,
                                        busyAccessStart,
                                        txAccessStart,
                                        navAccessStart,
                                        ackTimeoutAccessStart,
                                        ctsTimeoutAccessStart,
                                        switchingAccessStart}
                                       );
    }
  NS_LOG_INFO ("access grant start=" << accessGrantedStart <<
               ", rx access start=" << rxAccessStart <<
               ", busy access start=" << busyAccessStart <<
               ", tx access start=" << txAccessStart <<
               ", nav access start=" << navAccessStart);
  return accessGrantedStart;
}

Time
ChannelAccessManager::GetBackoffStartFor (Ptr<Txop> txop)
{
  NS_LOG_FUNCTION (this << txop);
  Time mostRecentEvent = MostRecent ({txop->GetBackoffStart (),
                                     GetAccessGrantStart () + (txop->GetAifsn () * GetSlot ())});
  NS_LOG_DEBUG ("Backoff start: " << mostRecentEvent.As (Time::US));

  return mostRecentEvent;
}

Time
ChannelAccessManager::GetBackoffEndFor (Ptr<Txop> txop)
{
  NS_LOG_FUNCTION (this << txop);
  Time backoffEnd = GetBackoffStartFor (txop) + (txop->GetBackoffSlots () * GetSlot ());
  NS_LOG_DEBUG ("Backoff end: " << backoffEnd.As (Time::US));

  return backoffEnd;
}

void
ChannelAccessManager::UpdateBackoff (void)
{
  NS_LOG_FUNCTION (this);
  uint32_t k = 0;
  for (auto txop : m_txops)
    {
      Time backoffStart = GetBackoffStartFor (txop);
      if (backoffStart <= Simulator::Now ())
        {
          uint32_t nIntSlots = ((Simulator::Now () - backoffStart) / GetSlot ()).GetHigh ();
          /*
           * EDCA behaves slightly different to DCA. For EDCA we
           * decrement once at the slot boundary at the end of AIFS as
           * well as once at the end of each clear slot
           * thereafter. For DCA we only decrement at the end of each
           * clear slot after DIFS. We account for the extra backoff
           * by incrementing the slot count here in the case of
           * EDCA. The if statement whose body we are in has confirmed
           * that a minimum of AIFS has elapsed since last busy
           * medium.
           */
          if (txop->IsQosTxop ())
            {
              nIntSlots++;
            }
          uint32_t n = std::min (nIntSlots, txop->GetBackoffSlots ());
          NS_LOG_DEBUG ("dcf " << k << " dec backoff slots=" << n);
          Time backoffUpdateBound = backoffStart + (n * GetSlot ());
          txop->UpdateBackoffSlotsNow (n, backoffUpdateBound);
        }
      ++k;
    }
}

void
ChannelAccessManager::DoRestartAccessTimeoutIfNeeded (void)
{
  NS_LOG_FUNCTION (this);
  /**
   * Is there a Txop which needs to access the medium, and,
   * if there is one, how many slots for AIFS+backoff does it require ?
   */
  bool accessTimeoutNeeded = false;
  Time expectedBackoffEnd = Simulator::GetMaximumSimulationTime ();
  for (auto txop : m_txops)
    {
      if (txop->GetAccessStatus () == Txop::REQUESTED)
        {
          Time tmp = GetBackoffEndFor (txop);
          if (tmp > Simulator::Now ())
            {
              accessTimeoutNeeded = true;
              expectedBackoffEnd = std::min (expectedBackoffEnd, tmp);
            }
        }
    }
  NS_LOG_DEBUG ("Access timeout needed: " << accessTimeoutNeeded);
  if (accessTimeoutNeeded)
    {
      NS_LOG_DEBUG ("expected backoff end=" << expectedBackoffEnd);
      Time expectedBackoffDelay = expectedBackoffEnd - Simulator::Now ();
      if (m_accessTimeout.IsRunning ()
          && Simulator::GetDelayLeft (m_accessTimeout) > expectedBackoffDelay)
        {
          m_accessTimeout.Cancel ();
        }
      if (m_accessTimeout.IsExpired ())
        {
          m_accessTimeout = Simulator::Schedule (expectedBackoffDelay,
                                                 &ChannelAccessManager::AccessTimeout, this);
        }
    }
}

void
ChannelAccessManager::DisableEdcaFor (Ptr<Txop> qosTxop, Time duration)
{
  NS_LOG_FUNCTION (this << qosTxop << duration);
  NS_ASSERT (qosTxop->IsQosTxop ());
  UpdateBackoff ();
  Time resume = Simulator::Now () + duration;
  NS_LOG_DEBUG ("Backoff will resume at time " << resume << " with "
                << qosTxop->GetBackoffSlots () << " remaining slot(s)");
  qosTxop->UpdateBackoffSlotsNow (0, resume);
  DoRestartAccessTimeoutIfNeeded ();
}

void
ChannelAccessManager::NotifyRxStartNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  NS_LOG_DEBUG ("rx start for=" << duration);
  UpdateBackoff ();
  m_lastRxStart = Simulator::Now ();
  m_lastRxDuration = duration;
  m_lastRxReceivedOk = true;
}

void
ChannelAccessManager::NotifyRxEndOkNow (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("rx end ok");
  m_lastRxDuration = Simulator::Now () - m_lastRxStart;
  m_lastRxReceivedOk = true;
}

void
ChannelAccessManager::NotifyRxEndErrorNow (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("rx end error");
  Time now = Simulator::Now ();
  Time lastRxEnd = m_lastRxStart + m_lastRxDuration;
  if (lastRxEnd > now)
    {
      m_lastBusyStart = now;
      m_lastBusyDuration = lastRxEnd - m_lastBusyStart;
    }
  m_lastRxDuration = now - m_lastRxStart;
  m_lastRxReceivedOk = false;
}

void
ChannelAccessManager::NotifyTxStartNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  m_lastRxReceivedOk = true;
  Time now = Simulator::Now ();
  Time lastRxEnd = m_lastRxStart + m_lastRxDuration;
  if (lastRxEnd > now)
    {
      //this may be caused only if PHY has started to receive a packet
      //inside SIFS, so, we check that lastRxStart was maximum a SIFS ago
      NS_ASSERT (now - m_lastRxStart <= GetSifs ());
      m_lastRxDuration = now - m_lastRxStart;
    }
  NS_LOG_DEBUG ("tx start for " << duration);
  UpdateBackoff ();
  m_lastTxStart = now;
  m_lastTxDuration = duration;
}

void
ChannelAccessManager::NotifyMaybeCcaBusyStartNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  NS_LOG_DEBUG ("busy start for " << duration);
  UpdateBackoff ();
  m_lastBusyStart = Simulator::Now ();
  m_lastBusyDuration = duration;
}

void
ChannelAccessManager::NotifySwitchingStartNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  Time now = Simulator::Now ();
  NS_ASSERT (m_lastTxStart + m_lastTxDuration <= now);
  NS_ASSERT (m_lastSwitchingStart + m_lastSwitchingDuration <= now);

  m_lastRxReceivedOk = true;

  if (m_lastRxStart + m_lastRxDuration > now)
    {
      //channel switching during packet reception
      m_lastRxDuration = now - m_lastRxStart;
    }
  if (m_lastNavStart + m_lastNavDuration > now)
    {
      m_lastNavDuration = now - m_lastNavStart;
    }
  if (m_lastBusyStart + m_lastBusyDuration > now)
    {
      m_lastBusyDuration = now - m_lastBusyStart;
    }
  if (m_lastAckTimeoutEnd > now)
    {
      m_lastAckTimeoutEnd = now;
    }
  if (m_lastCtsTimeoutEnd > now)
    {
      m_lastCtsTimeoutEnd = now;
    }

  //Cancel timeout
  if (m_accessTimeout.IsRunning ())
    {
      m_accessTimeout.Cancel ();
    }

  // Notify the FEM, which will in turn notify the MAC
  m_feManager->NotifySwitchingStartNow (duration);

  //Reset backoffs
  for (auto txop : m_txops)
    {
      uint32_t remainingSlots = txop->GetBackoffSlots ();
      if (remainingSlots > 0)
        {
          txop->UpdateBackoffSlotsNow (remainingSlots, now);
          NS_ASSERT (txop->GetBackoffSlots () == 0);
        }
      txop->ResetCw ();
      txop->m_access = Txop::NOT_REQUESTED;
    }

  NS_LOG_DEBUG ("switching start for " << duration);
  m_lastSwitchingStart = Simulator::Now ();
  m_lastSwitchingDuration = duration;
}

void
ChannelAccessManager::NotifySleepNow (void)
{
  NS_LOG_FUNCTION (this);
  m_sleeping = true;
  //Cancel timeout
  if (m_accessTimeout.IsRunning ())
    {
      m_accessTimeout.Cancel ();
    }

  //Reset backoffs
  for (auto txop : m_txops)
    {
      txop->NotifySleep ();
    }
}

void
ChannelAccessManager::NotifyOffNow (void)
{
  NS_LOG_FUNCTION (this);
  m_off = true;
  //Cancel timeout
  if (m_accessTimeout.IsRunning ())
    {
      m_accessTimeout.Cancel ();
    }

  //Reset backoffs
  for (auto txop : m_txops)
    {
      txop->NotifyOff ();
    }
}

void
ChannelAccessManager::NotifyWakeupNow (void)
{
  NS_LOG_FUNCTION (this);
  m_sleeping = false;
  for (auto txop : m_txops)
    {
      uint32_t remainingSlots = txop->GetBackoffSlots ();
      if (remainingSlots > 0)
        {
          txop->UpdateBackoffSlotsNow (remainingSlots, Simulator::Now ());
          NS_ASSERT (txop->GetBackoffSlots () == 0);
        }
      txop->ResetCw ();
      txop->m_access = Txop::NOT_REQUESTED;
      txop->NotifyWakeUp ();
    }
}

void
ChannelAccessManager::NotifyOnNow (void)
{
  NS_LOG_FUNCTION (this);
  m_off = false;
  for (auto txop : m_txops)
    {
      uint32_t remainingSlots = txop->GetBackoffSlots ();
      if (remainingSlots > 0)
        {
          txop->UpdateBackoffSlotsNow (remainingSlots, Simulator::Now ());
          NS_ASSERT (txop->GetBackoffSlots () == 0);
        }
      txop->ResetCw ();
      txop->m_access = Txop::NOT_REQUESTED;
      txop->NotifyOn ();
    }
}

void
ChannelAccessManager::NotifyNavResetNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  NS_LOG_DEBUG ("nav reset for=" << duration);
  UpdateBackoff ();
  m_lastNavStart = Simulator::Now ();
  m_lastNavDuration = duration;
  /**
   * If the NAV reset indicates an end-of-NAV which is earlier
   * than the previous end-of-NAV, the expected end of backoff
   * might be later than previously thought so, we might need
   * to restart a new access timeout.
   */
  DoRestartAccessTimeoutIfNeeded ();
}

void
ChannelAccessManager::NotifyNavStartNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  NS_ASSERT (m_lastNavStart <= Simulator::Now ());
  NS_LOG_DEBUG ("nav start for=" << duration);
  UpdateBackoff ();
  Time newNavEnd = Simulator::Now () + duration;
  Time lastNavEnd = m_lastNavStart + m_lastNavDuration;
  if (newNavEnd > lastNavEnd)
    {
      m_lastNavStart = Simulator::Now ();
      m_lastNavDuration = duration;
    }
}

void
ChannelAccessManager::NotifyAckTimeoutStartNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  NS_ASSERT (m_lastAckTimeoutEnd < Simulator::Now ());
  m_lastAckTimeoutEnd = Simulator::Now () + duration;
}

void
ChannelAccessManager::NotifyAckTimeoutResetNow (void)
{
  NS_LOG_FUNCTION (this);
  m_lastAckTimeoutEnd = Simulator::Now ();
  DoRestartAccessTimeoutIfNeeded ();
}

void
ChannelAccessManager::NotifyCtsTimeoutStartNow (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  m_lastCtsTimeoutEnd = Simulator::Now () + duration;
}

void
ChannelAccessManager::NotifyCtsTimeoutResetNow (void)
{
  NS_LOG_FUNCTION (this);
  m_lastCtsTimeoutEnd = Simulator::Now ();
  DoRestartAccessTimeoutIfNeeded ();
}

} //namespace ns3
