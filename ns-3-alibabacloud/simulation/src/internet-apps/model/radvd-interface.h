/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#ifndef RADVD_INTERFACE_H
#define RADVD_INTERFACE_H

#include "radvd-prefix.h"
#include <list>
#include "ns3/simple-ref-count.h"
#include "ns3/nstime.h"

namespace ns3
{

/**
 * \ingroup radvd
 * \brief Radvd interface configuration.
 */
class RadvdInterface : public SimpleRefCount<RadvdInterface>
{
public:
  /// Container: Ptr to RadvdPrefix
  typedef std::list<Ptr<RadvdPrefix> > RadvdPrefixList;
  /// Container Iterator: Ptr to RadvdPrefix
  typedef std::list<Ptr<RadvdPrefix> >::iterator RadvdPrefixListI;
  /// Container Const Iterator: Ptr to RadvdPrefix
  typedef std::list<Ptr<RadvdPrefix> >::const_iterator RadvdPrefixListCI;

  /**
   * \brief Constructor.
   * \param interface interface index
   */
  RadvdInterface (uint32_t interface);

  /**
   * \brief Constructor.
   * \param interface interface index
   * \param maxRtrAdvInterval maximum RA interval (ms)
   * \param minRtrAdvInterval minimum RA interval (ms)
   */
  RadvdInterface (uint32_t interface, uint32_t maxRtrAdvInterval, uint32_t minRtrAdvInterval);

  /**
   * \brief Destructor.
   */
  ~RadvdInterface ();

  /**
   * \brief Get interface index for this configuration.
   * \return interface index
   */
  uint32_t GetInterface () const;

  /**
   * \brief Get list of prefixes advertised for this interface.
   * \return list of IPv6 prefixes
   */
  RadvdPrefixList GetPrefixes () const;

  /**
   * \brief Add a prefix to advertise on interface.
   * \param routerPrefix prefix to advertise
   */
  void AddPrefix (Ptr<RadvdPrefix> routerPrefix);

  /**
   * \brief Is send advert enabled (periodic RA and reply to RS) ?
   * \return send advert flag
   */
  bool IsSendAdvert () const;

  /**
   * \brief Set send advert flag.
   * \param sendAdvert value
   */
  void SetSendAdvert (bool sendAdvert);

  /**
   * \brief Get maximum RA interval.
   * \return RA interval (ms)
   */
  uint32_t GetMaxRtrAdvInterval () const;

  /**
   * \brief Get maximum RA interval.
   * \param maxRtrAdvInterval RA interval (ms)
   */
  void SetMaxRtrAdvInterval (uint32_t maxRtrAdvInterval);

  /**
   * \brief Get minimum RA interval 
   * \return RA interval (ms)
   */
  uint32_t GetMinRtrAdvInterval () const;

  /**
   * \brief Get minimum RA interval 
   * \param minRtrAdvInterval RA interval (ms).
   */
  void SetMinRtrAdvInterval (uint32_t minRtrAdvInterval);

  /**
   * \brief Get minimum delay between RAs.
   * \return minimum delay (ms)
   */
  uint32_t GetMinDelayBetweenRAs () const;

  /**
   * \brief Set minimum delay between RAs.
   * \param minDelayBetweenRAs minimum delay (ms)
   */
  void SetMinDelayBetweenRAs (uint32_t minDelayBetweenRAs);

  /**
   * \brief Is managed flag enabled ?
   * \return managed flag
   */
  bool IsManagedFlag () const;

  /**
   * \brief Set managed flag
   * \param managedFlag value
   */
  void SetManagedFlag (bool managedFlag);

  /**
   * \brief Is "other config" flag enabled ?
   * \return other config flag
   */
  bool IsOtherConfigFlag () const;

  /**
   * \brief Set "other config" flag
   * \param otherConfigFlag value
   */
  void SetOtherConfigFlag (bool otherConfigFlag);

  /**
   * \brief Get link MTU.
   * \return link MTU
   */
  uint32_t GetLinkMtu () const;

  /**
   * \brief Set link MTU.
   * \param linkMtu link MTU
   */
  void SetLinkMtu (uint32_t linkMtu); 

  /**
   * \brief Get reachable time.
   * \return reachable time
   */
  uint32_t GetReachableTime () const;

  /**
   * \brief Set reachable time.
   * \param reachableTime reachable time
   */
  void SetReachableTime (uint32_t reachableTime);

  /**
   * \brief Get default lifetime.
   * \return default lifetime
   */
  uint32_t GetDefaultLifeTime () const;

  /**
   * \brief Set default lifetime.
   * \param defaultLifeTime default lifetime
   */
  void SetDefaultLifeTime (uint32_t defaultLifeTime);

  /**
   * \brief Get retransmission timer.
   * \return retransmission timer
   */
  uint32_t GetRetransTimer () const;

  /**
   * \brief Set retransmission timer.
   * \param retransTimer retransmission timer
   */
  void SetRetransTimer (uint32_t retransTimer);

  /**
   * \brief Get current hop limit.
   * \return current hop limit for the link
   */
  uint8_t GetCurHopLimit () const;

  /**
   * \brief Set current hop limit.
   * \param curHopLimit current hop limit for the link
   */
  void SetCurHopLimit (uint8_t curHopLimit); 

  /**
   * \brief Get default preference.
   * \return default preference
   */
  uint8_t GetDefaultPreference () const;

  /**
   * \brief Set default preference.
   * \param defaultPreference default preference
   */
  void SetDefaultPreference (uint8_t defaultPreference);

  /**
   * \brief Is source LLA option should be included in RA ?
   * \return true if source address is added in RA, false otherwise
   */
  bool IsSourceLLAddress () const;

  /**
   * \brief Set flag to add or not LLA to RA.
   * \param sourceLLAddress value
   */
  void SetSourceLLAddress (bool sourceLLAddress);

  /**
   * \brief Is "home agent" flag enabled ?
   * \return "home agent" flag
   */
  bool IsHomeAgentFlag () const;

  /**
   * \brief Set "home agent" flag.
   * \param homeAgentFlag value
   */
  void SetHomeAgentFlag (bool homeAgentFlag);

  /**
   * \brief Is Home Agent Information option should be included in RA ?
   * \return true if HA information option is added in RA, false otherwise
   */
  bool IsHomeAgentInfo () const;

  /**
   * \brief Set flag to add or not HA information option to RA.
   * \param homeAgentFlag value
   */
  void SetHomeAgentInfo (bool homeAgentFlag);

  /**
   * \brief Get home agent lifetime.
   * \return home agent lifetime
   */
  uint32_t GetHomeAgentLifeTime () const;

  /**
   * \brief Set home agent lifetime.
   * \param homeAgentLifeTime home agent lifetime
   */
  void SetHomeAgentLifeTime (uint32_t homeAgentLifeTime);

  /**
   * \brief Get home agent preference.
   * \return home agent preference
   */
  uint32_t GetHomeAgentPreference () const;

  /**
   * \brief Set home agent preference.
   * \param homeAgentPreference home agent preference
   */
  void SetHomeAgentPreference (uint32_t homeAgentPreference);

  /**
   * \brief Is "mobile router support" flag enabled ?
   * \return "mobile router support" flag
   */
  bool IsMobRtrSupportFlag () const;

  /**
   * \brief Set "mobile router support" flag.
   * \param mobRtrSupportFlag value
   */
  void SetMobRtrSupportFlag (bool mobRtrSupportFlag);

  /**
   * \brief Is advertisement interval option should be included in RA ?
   * \return true if advertisement interval option is added in RA, false otherwise
   */
  bool IsIntervalOpt () const;

  /**
   * \brief Set flag to add or not advertisement interval to RA.
   * \param intervalOpt value
   */
  void SetIntervalOpt (bool intervalOpt);

  /**
   * \brief Get the last time a RA has been sent.
   * \returns the last RA send time
   */
  Time GetLastRaTxTime ();

  /**
   * \brief Set the last RA send time. It also decrements the initial Rtr Advertisements counter.
   * \param now the last RA send time
   */
  void SetLastRaTxTime (Time now);

  /**
   * \brief Checks if the interface is subject to the initial Rtr Advertisements rule.
   * \returns true if the initial Rtr Advertisements counter is greater than zero.
   */
  bool IsInitialRtrAdv ();

private:

  /**
   * \brief Interface to advertise RA.
   */
  uint32_t m_interface;

  /**
   * \brief List of prefixes to advertise.
   */
  RadvdPrefixList m_prefixes;

  /**
   * \brief Flag whether or not router sends periodic RA and respond to RS.
   */
  bool m_sendAdvert;

  /**
   * \brief Maximum RA interval in milliseconds.
   */
  uint32_t m_maxRtrAdvInterval;

  /**
   * \brief Minimum RA interval in milliseconds.
   */
  uint32_t m_minRtrAdvInterval;

  /**
   * \brief Minimum delay between RA in milliseconds.
   */
  uint32_t m_minDelayBetweenRAs;

  /**
   * \brief Managed flag. If true host use the stateful protocol for address autoconfiguration.
   */
  bool m_managedFlag;

  /**
   * \brief Other configuration flag. If true host use stateful protocol for other (non-address) information.
   */
  bool m_otherConfigFlag;

  /**
   * \brief Link MTU to use.
   */
  uint32_t m_linkMtu;

  /**
   * \brief Reachable time in milliseconds.
   */
  uint32_t m_reachableTime;

  /**
   * \brief Retransmission timer in milliseconds.
   */
  uint32_t m_retransTimer;

  /**
   * \brief Current hop limit (TTL).
   */
  uint32_t m_curHopLimit;

  /**
   * \brief Default life time in seconds.
   */
  uint32_t m_defaultLifeTime;

  /**
   * \brief Preference associated with default router.
   * 0 = low
   * 1 = medium
   * 2 = high
   */
  uint8_t m_defaultPreference;

  /**
   * \brief Flag to add link-layer address in RA.
   */
  bool m_sourceLLAddress;

  /**
   * \brief Flag to add HA (home agent) flag in RA.
   */
  bool m_homeAgentFlag;

  /**
   * \brief Flag to add Home Agent Information option (Mobile IPv6).
   * Currently not implemented.
   */
  bool m_homeAgentInfo;

  /**
   * \brief Home agent lifetime in seconds. Ignored if home agent info is not set.
   */
  uint32_t m_homeAgentLifeTime;

  /**
   * \brief Home agent preference. Ignored if home agent info is not set.
   */
  uint32_t m_homeAgentPreference;

  /**
   * \brief Flag for HA to signals it supports Mobile Router registrations (NEMO Basic).
   */
  bool m_mobRtrSupportFlag;

  /**
   * \brief Flag to add Advertisement Interval option in RA.
   */
  bool m_intervalOpt;

  /**
   * \brief Last RA send time.
   */
  Time m_lastSendTime;

  /**
   * \brief Number of fast announcement to do
   */
  uint8_t m_initialRtrAdvertisementsLeft;

};

} /* namespace ns3 */

#endif /* RADVD_INTERFACE_H */

