/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013
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
 * Authors: Ghada Badawy <gbadawy@gmail.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef AMPDU_TAG_H
#define AMPDU_TAG_H

#include "ns3/nstime.h"
#include "ns3/tag.h"

namespace ns3 {

/**
 * \ingroup wifi
 *
 * The aim of the AmpduTag is to provide means for a MAC to specify that a packet includes A-MPDU
 * since this is done in HT-SIG and there is no HT-SIG representation in ns-3
 */
class AmpduTag : public Tag
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TypeId GetInstanceTypeId (void) const override;
  void Serialize (TagBuffer i) const override;
  void Deserialize (TagBuffer i) override;
  uint32_t GetSerializedSize () const override;
  void Print (std::ostream &os) const override;

  /**
   * Create a AmpduTag with the default =0 no A-MPDU
   */
  AmpduTag ();
  /**
   * \param nbOfMpdus the remaining number of MPDUs
   *
   * Set the remaining number of MPDUs in the A-MPDU.
   */
  void SetRemainingNbOfMpdus (uint8_t nbOfMpdus);
  /**
   * \param duration the remaining duration of the A-MPDU
   *
   * Set the remaining duration of the A-MPDU.
   */
  void SetRemainingAmpduDuration (Time duration);

  /**
   * \return the remaining number of MPDUs in an A-MPDU
   *
   * Returns the remaining number of MPDUs in an A-MPDU
   */
  uint8_t GetRemainingNbOfMpdus (void) const;
  /**
   * \return the remaining duration of an A-MPDU
   *
   * Returns the remaining duration of an A-MPDU
   */
  Time GetRemainingAmpduDuration (void) const;


private:
  uint8_t m_nbOfMpdus; //!< Remaining number of MPDUs in the A-MPDU
  Time m_duration;     //!< Remaining duration of the A-MPDU
};

} //namespace ns3

#endif /* AMPDU_TAG_H */
