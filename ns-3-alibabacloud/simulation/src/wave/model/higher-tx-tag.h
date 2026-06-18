/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2013 Dalian University of Technology
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
 *         Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef HIGHER_LAYER_TX_VECTOR_TAG_H
#define HIGHER_LAYER_TX_VECTOR_TAG_H

#include "ns3/tag.h"
#include "ns3/wifi-tx-vector.h"

namespace ns3 {

class Tag;
class TypeId;

/**
 * \ingroup packet
 * \ingroup wave
 *
 * \brief This tag will be used to support higher layer control DataRate
 * and TxPwr_Level for transmission.
 * If the high layer enables adaptable mode,  DataRate will be the
 * minimum allowable value and TxPwr_Level will be the maximum
 * allowable value for transmission.
 * If the higher layer does not enable adaptable parameter, the
 * DataRate and TxPwr_Level will be actual values for transmission.
 * However, if this tag is not used and inserted in the packet, the actual
 * DataRate and TxPwr_Level for transmission will be determined by MAC layer.
 */
class HigherLayerTxVectorTag : public Tag
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  HigherLayerTxVectorTag (void);
  /**
   * Constructor
   *
   * \param txVector wifi transmit vector
   * \param adaptable is adaptable?
   */
  HigherLayerTxVectorTag (WifiTxVector txVector, bool adaptable);
  /**
   * \returns the tx vector for transmission
   */
  WifiTxVector GetTxVector (void) const;
  /**
   * \returns the adaptable mode for transmission
   */
  bool IsAdaptable (void) const;

  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

private:
  WifiTxVector m_txVector; ///< transmit vector
  bool m_adaptable; ///< adaptable
};

} // namespace ns3

#endif /* HIGHER_LAYER_TX_VECTOR_TAG_H*/
