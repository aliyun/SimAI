/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008-2009 Telecom Bretagne 
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
 * Author: Mehdi Benamor <benamor.mehdi@ensi.rnu.tn>
 */

#ifndef IPV6_AUTOCONFIGURED_PREFIX_H
#define IPV6_AUTOCONFIGURED_PREFIX_H

#include <stdint.h>

#include <list>
#include <vector>
#include <ostream>

#include "ns3/timer.h"
#include "ns3/ipv6-address.h"

namespace ns3
{

/**
 * \ingroup ipv6
 *
 * \brief Router prefix information.
 */
class Ipv6AutoconfiguredPrefix : public Object
{
public:
  /** 
   * \brief Constructor.
   * \param node node
   * \param interface interface index
   * \param prefix IPv6 address
   * \param mask bitmask prefix
   * \param preferredLifeTime the preferred life time
   * \param validLifeTime the valid life time
   * \param router if it the prefix that configure the default gateway
   */
  Ipv6AutoconfiguredPrefix (Ptr<Node> node, uint32_t interface, Ipv6Address prefix, Ipv6Prefix mask, uint32_t preferredLifeTime, uint32_t validLifeTime, Ipv6Address router = Ipv6Address ("::"));

  /**
   * \brief Destructor.
   */
  ~Ipv6AutoconfiguredPrefix ();

  /**
   * \brief Set the default gateway router.
   * \param router IPv6 link-local address of the default router
   */
  void SetDefaultGatewayRouter (Ipv6Address router);

  /**
   * \brief Get the default gateway address.
   * \return IPv6 link-local address of the default router
   */
  Ipv6Address GetDefaultGatewayRouter () const;

  /**
   * \brief Get the interface index.
   * \return interface index
   */
  uint32_t GetInterface () const;

  /**
   * \brief Set the interface.
   * \param interface interface index to set
   */
  void SetInterface (uint32_t interface);

  /**
   * \brief Get the prefix preferred life time.
   * \return preferred life time
   */
  uint32_t GetPreferredLifeTime () const;

  /**
   * \brief Set the prefix preferred life time.
   * \param p the prefix preferred life time
   */
  void SetPreferredLifeTime (uint32_t p);

  /**
   * \brief Get the prefix valid life time.
   * \return valid life time
   */
  uint32_t GetValidLifeTime (void) const;

  /**
   * \brief Set the prefix valid life time.
   * \param v the prefix valid life time
   */
  void SetValidLifeTime (uint32_t v);

  /**
   * \brief Test if the prefix is preferred.
   * \return true if prefix is in preferred state, false otherwise
   */
  bool IsPreferred () const;

  /**
   * \brief Test if the prefix is valid.
   * \return true if prefix is in valid state, false otherwise
   */
  bool IsValid () const;

  /**
   * \brief Set the prefix as preferred.
   */
  void SetPreferred ();

  /**
   * \brief Set the prefix as valid.
   */
  void SetValid ();

  /**
   * \brief Start the preferred timer.
   */
  void StartPreferredTimer ();

  /**
   * \brief Start the valid timer.
   */
  void StartValidTimer ();

  /**
   * \brief Stop the preferred timer.
   */
  void StopPreferredTimer ();

  /**
   * \brief Stop the valid timer.
   */
  void StopValidTimer ();

  /**
   * \brief Set the prefix as preferred.
   */
  void MarkPreferredTime ();

  /**
   * \brief Set the prefix as valid.
   */
  void MarkValidTime ();

  /**
   * \brief Signal that the preferred time expired and start the valid timer.
   */
  void FunctionPreferredTimeout ();

  /**
   * \brief Signal that the valid time expired.
   */
  void FunctionValidTimeout ();

  /**
   * \brief Remove this prefix from the prefix list.
   */
  void RemoveMe ();

  /**
   * \brief Get the prefix identifier.
   * \return id of the prefix.
   */
  uint32_t GetId () const;

  /**
   * \brief Get the prefix address.
   * \return prefix address
   */
  Ipv6Address GetPrefix () const;

  /**
   * \brief Set the prefix address.
   * \param prefix prefix address to set
   */
  void SetPrefix (Ipv6Address prefix);

  /**
   * \brief Get the bitmask prefix.
   * \return bitmask prefix
   */
  Ipv6Prefix GetMask () const;

  /**
   * \brief Set the bitmask prefix.
   * \param mask prefix
   */
  void SetMask (Ipv6Prefix mask);

private:
  /**
   * \brief a static identifier.
   */
  static uint32_t m_prefixId;

  /**
   * \brief the identifier of this prefix.
   */
  uint32_t m_id;

  /**
   * \brief The node.
   */
  Ptr<Node> m_node;

  /**
   * \brief The prefix IPv6 address.
   */
  Ipv6Address m_prefix;

  /**
   * \brief The prefix bitmask (length).
   */
  Ipv6Prefix m_mask;

  /**
   * \brief Default gateway router.
   *
   * If the RA received also configured the default gateway, 
   * this variable has the link-local address. Otherwise this
   * is "::"
   */
  Ipv6Address m_defaultGatewayRouter;

  /**
   * \brief The interface index (which is stored the address
   * corresponding of the prefix).
   */
  uint32_t m_interface;

  /**
   * \brief the valid life time.
   */
  uint32_t m_validLifeTime;

  /**
   * \brief the preferred life time.
   */
  uint32_t m_preferredLifeTime;

  /**
   * \brief true if the prefix is preferred.
   */
  bool m_preferred;

  /**
   * \brief true if the prefix is valid.
   */
  bool m_valid;

  /**
   * \brief the timer for preferred life time.
   */
  Timer m_preferredTimer;

  /**
   * \brief the timer for valid life time.
   */
  Timer m_validTimer;
};

} /* namespace ns3 */

#endif /* IPV6_AUTOCONFIGURED_PREFIX_H */

