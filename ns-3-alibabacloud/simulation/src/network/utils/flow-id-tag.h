/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
#ifndef FLOW_ID_TAG_H
#define FLOW_ID_TAG_H

#include "ns3/tag.h"

namespace ns3 {

class FlowIdTag : public Tag
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer buf) const;
  virtual void Deserialize (TagBuffer buf);
  virtual void Print (std::ostream &os) const;
  FlowIdTag ();
  
  /**
   *  Constructs a FlowIdTag with the given flow id
   *
   *  \param flowId Id to use for the tag
   */
  FlowIdTag (uint32_t flowId);
  /**
   *  Sets the flow id for the tag
   *  \param flowId Id to assign to the tag
   */
  void SetFlowId (uint32_t flowId);
  /**
   *  Gets the flow id for the tag
   *  \returns current flow id for this tag
   */
  uint32_t GetFlowId (void) const;
  /**
   *  Uses a static variable to generate sequential flow id
   *  \returns flow id allocated
   */
  static uint32_t AllocateFlowId (void);
private:
  uint32_t m_flowId; //!< Flow ID
};

} // namespace ns3

#endif /* FLOW_ID_TAG_H */
