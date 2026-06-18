/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2021 University of Washington: Group mobility changes
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
 * Adapted from 'mobility-helper.h' for group mobility by Tom Henderson
 */

#ifndef GROUP_MOBILITY_HELPER_H
#define GROUP_MOBILITY_HELPER_H

#include <vector>
#include "ns3/object-factory.h"
#include "ns3/attribute.h"
#include "ns3/node-container.h"

namespace ns3 {

class PositionAllocator;
class MobilityModel;

/**
 * \ingroup mobility
 * \brief Helper class used to assign positions and mobility models to nodes
 * for a group mobility configuration.
 *
 * This helper can be used for group mobility configuration and installation
 * onto a group of nodes, in which there is a reference (parent) mobility
 * model that is the same for all nodes, and similarly configured (but
 * distinct) child mobility models.
 */
class GroupMobilityHelper
{
public:
  /**
   * Construct a group mobility helper 
   */
  GroupMobilityHelper ();

  /**
   * Destroy a group mobility helper
   */
  ~GroupMobilityHelper ();

  /**
   * Set the position allocator which will be used to allocate the initial 
   * position of the reference mobility model.
   *
   * \param allocator allocate initial reference mobility model position
   */
  void SetReferencePositionAllocator (Ptr<PositionAllocator> allocator);

  /**
   * Configure the position allocator which will be used to allocate
   * the initial position of the reference mobility model.
   *
   * \param type the type of mobility model to use.
   * \param n1 the name of the attribute to set in the mobility model.
   * \param v1 the value of the attribute to set in the mobility model.
   * \param n2 the name of the attribute to set in the mobility model.
   * \param v2 the value of the attribute to set in the mobility model.
   * \param n3 the name of the attribute to set in the mobility model.
   * \param v3 the value of the attribute to set in the mobility model.
   * \param n4 the name of the attribute to set in the mobility model.
   * \param v4 the value of the attribute to set in the mobility model.
   * \param n5 the name of the attribute to set in the mobility model.
   * \param v5 the value of the attribute to set in the mobility model.
   * \param n6 the name of the attribute to set in the mobility model.
   * \param v6 the value of the attribute to set in the mobility model.
   * \param n7 the name of the attribute to set in the mobility model.
   * \param v7 the value of the attribute to set in the mobility model.
   * \param n8 the name of the attribute to set in the mobility model.
   * \param v8 the value of the attribute to set in the mobility model.
   * \param n9 the name of the attribute to set in the mobility model.
   * \param v9 the value of the attribute to set in the mobility model.
   */
  void SetReferencePositionAllocator (std::string type,
                             std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                             std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                             std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                             std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                             std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                             std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                             std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue (),
                             std::string n8 = "", const AttributeValue &v8 = EmptyAttributeValue (),
                             std::string n9 = "", const AttributeValue &v9 = EmptyAttributeValue ());

  /**
   * Set the position allocator which will be used to allocate the initial 
   * position of the member mobility models.
   *
   * \param allocator allocate initial member mobility model positions
   */
  void SetMemberPositionAllocator (Ptr<PositionAllocator> allocator);

  /**
   * Configure the position allocator which will be used to allocate the
   * initial position of the member mobility models.
   *
   * \param type the type of mobility model to use.
   * \param n1 the name of the attribute to set in the mobility model.
   * \param v1 the value of the attribute to set in the mobility model.
   * \param n2 the name of the attribute to set in the mobility model.
   * \param v2 the value of the attribute to set in the mobility model.
   * \param n3 the name of the attribute to set in the mobility model.
   * \param v3 the value of the attribute to set in the mobility model.
   * \param n4 the name of the attribute to set in the mobility model.
   * \param v4 the value of the attribute to set in the mobility model.
   * \param n5 the name of the attribute to set in the mobility model.
   * \param v5 the value of the attribute to set in the mobility model.
   * \param n6 the name of the attribute to set in the mobility model.
   * \param v6 the value of the attribute to set in the mobility model.
   * \param n7 the name of the attribute to set in the mobility model.
   * \param v7 the value of the attribute to set in the mobility model.
   * \param n8 the name of the attribute to set in the mobility model.
   * \param v8 the value of the attribute to set in the mobility model.
   * \param n9 the name of the attribute to set in the mobility model.
   * \param v9 the value of the attribute to set in the mobility model.
   */
  void SetMemberPositionAllocator (std::string type,
                             std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                             std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                             std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                             std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                             std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                             std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                             std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue (),
                             std::string n8 = "", const AttributeValue &v8 = EmptyAttributeValue (),
                             std::string n9 = "", const AttributeValue &v9 = EmptyAttributeValue ());

  /**
   * Set the reference mobility model which will be installed as the parent
   * mobility model during GroupMobilityModel::Install.
   *
   * \param mobility reference mobility model
   */
  void SetReferenceMobilityModel (Ptr<MobilityModel> mobility);

  /**
   * Configure the reference mobility model which will be installed as the
   * parent mobility model during GroupMobilityModel::Install.
   *
   * \param type the type of mobility model to use.
   * \param n1 the name of the attribute to set in the mobility model.
   * \param v1 the value of the attribute to set in the mobility model.
   * \param n2 the name of the attribute to set in the mobility model.
   * \param v2 the value of the attribute to set in the mobility model.
   * \param n3 the name of the attribute to set in the mobility model.
   * \param v3 the value of the attribute to set in the mobility model.
   * \param n4 the name of the attribute to set in the mobility model.
   * \param v4 the value of the attribute to set in the mobility model.
   * \param n5 the name of the attribute to set in the mobility model.
   * \param v5 the value of the attribute to set in the mobility model.
   * \param n6 the name of the attribute to set in the mobility model.
   * \param v6 the value of the attribute to set in the mobility model.
   * \param n7 the name of the attribute to set in the mobility model.
   * \param v7 the value of the attribute to set in the mobility model.
   * \param n8 the name of the attribute to set in the mobility model.
   * \param v8 the value of the attribute to set in the mobility model.
   * \param n9 the name of the attribute to set in the mobility model.
   * \param v9 the value of the attribute to set in the mobility model.
   */
  void SetReferenceMobilityModel (std::string type,
                         std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                         std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                         std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                         std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                         std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                         std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                         std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue (),
                         std::string n8 = "", const AttributeValue &v8 = EmptyAttributeValue (),
                         std::string n9 = "", const AttributeValue &v9 = EmptyAttributeValue ());

  /**
   * Configure the mobility model which will be installed as the
   * member (child) mobility model during GroupMobilityModel::Install.
   * 
   * \param type the type of mobility model to use.
   * \param n1 the name of the attribute to set in the mobility model.
   * \param v1 the value of the attribute to set in the mobility model.
   * \param n2 the name of the attribute to set in the mobility model.
   * \param v2 the value of the attribute to set in the mobility model.
   * \param n3 the name of the attribute to set in the mobility model.
   * \param v3 the value of the attribute to set in the mobility model.
   * \param n4 the name of the attribute to set in the mobility model.
   * \param v4 the value of the attribute to set in the mobility model.
   * \param n5 the name of the attribute to set in the mobility model.
   * \param v5 the value of the attribute to set in the mobility model.
   * \param n6 the name of the attribute to set in the mobility model.
   * \param v6 the value of the attribute to set in the mobility model.
   * \param n7 the name of the attribute to set in the mobility model.
   * \param v7 the value of the attribute to set in the mobility model.
   * \param n8 the name of the attribute to set in the mobility model.
   * \param v8 the value of the attribute to set in the mobility model.
   * \param n9 the name of the attribute to set in the mobility model.
   * \param v9 the value of the attribute to set in the mobility model.
   *
   * Calls to MobilityHelper::Install will create an instance of a matching 
   * mobility model for each node.
   */
  void SetMemberMobilityModel (std::string type,
                         std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                         std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                         std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                         std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                         std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                         std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                         std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue (),
                         std::string n8 = "", const AttributeValue &v8 = EmptyAttributeValue (),
                         std::string n9 = "", const AttributeValue &v9 = EmptyAttributeValue ());

  /**
   * \brief Install and configure a hierarchical mobility model to the
   * given node, based on the configured reference and member models.
   *
   * If position allocators are configured, they will be invoked to
   * set the initial position.
   *
   * \param node The node to configure
   */
  void Install (Ptr<Node> node);
  /**
   * \brief Install and configure a hierarchical mobility model to the
   * given node, based on the configured reference and member models.
   *
   * If position allocators are configured, they will be invoked to
   * set the initial position.
   *
   * \param nodeName The name of the node to configure
   */
  void Install (std::string nodeName);

  /**
   * \brief Install and configure a hierarchical mobility model to all nodes
   * in the container, based on the configured reference and member models.
   *
   * If position allocators are configured, they will be invoked to
   * set the initial positions.
   *
   * \param container The set of nodes to configure
   */
  void Install (NodeContainer container);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by the mobility models on these nodes. Return the number of 
   * streams (possibly zero) that have been assigned. The Install() 
   * method should have previously been called by the user.
   *
   * \note If the PositionAllocator used contains random variables, they
   * will not be affected by this call to AssignStreams because they are
   * used earlier during Install() time.  If the user needs to assign a fixed
   * stream number to a PositionAllocator used with this helper, the user
   * should instantiate it outside of the helper, call AssignStreams() on
   * it, and then pass the pointer of it to this helper.
   *
   * \param c NodeContainer of the set of nodes containing the MobilityModels
   * that should be modified to use a fixed stream
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this helper
   */
  int64_t AssignStreams (NodeContainer c, int64_t stream);

private:

  bool m_referencePositionSet {false}; //!< flag for avoiding multiple SetPosition calls on the reference model
  Ptr<MobilityModel> m_referenceMobility; //!< Reference mobility model
  Ptr<PositionAllocator> m_referencePosition; //!< Position allocator for use as reference position allocator
  ObjectFactory  m_memberMobilityFactory; //!< Object factory to create member mobility models
  Ptr<PositionAllocator> m_memberPosition; //!< Position allocator for use as member position allocator
};

} // namespace ns3

#endif /* GROUP_MOBILITY_HELPER_H */
