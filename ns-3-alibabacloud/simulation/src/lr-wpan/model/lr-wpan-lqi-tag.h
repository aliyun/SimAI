/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Fraunhofer FKIE
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
 * Author:
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 */
#ifndef LR_WPAN_LQI_TAG_H
#define LR_WPAN_LQI_TAG_H

#include <ns3/tag.h>

namespace ns3 {

/**
 * \ingroup lr-wpan
 * Represent the LQI (Link Quality Estination).
 * 
 * The LQI Tag is added to each received packet, and can be 
 * used by upper layers to estimate the channel conditions.
 * 
 * The LQI is the total packet success rate scaled to 0-255.
 */
class LrWpanLqiTag : public Tag
{
public:
  /**
   * Get the type ID.
   *
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * Create a LrWpanLqiTag with the default LQI 0.
   */
  LrWpanLqiTag (void);

  /**
   * Create a LrWpanLqiTag with the given LQI value.
   * \param lqi The LQI.
   */
  LrWpanLqiTag (uint8_t lqi);

  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

  /**
   * Set the LQI to the given value.
   *
   * \param lqi the value of the LQI to set
   */
  void Set (uint8_t lqi);

  /**
   * Get the LQI value.
   *
   * \return the LQI value
   */
  uint8_t Get (void) const;
private:
  /**
   * The current LQI value of the tag.
   */
  uint8_t m_lqi;
};


}
#endif /* LR_WPAN_LQI_TAG_H */
