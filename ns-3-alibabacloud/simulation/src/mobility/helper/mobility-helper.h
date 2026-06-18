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

#ifndef MOBILITY_HELPER_H
#define MOBILITY_HELPER_H

#include <vector>
#include "ns3/object-factory.h"
#include "ns3/attribute.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/position-allocator.h"
#include "ns3/node-container.h"

namespace ns3 {

class PositionAllocator;
class MobilityModel;

/**
 * \ingroup mobility
 * \brief Helper class used to assign positions and mobility models to nodes.
 *
 * MobilityHelper::Install is the most important method here.
 */
class MobilityHelper
{
public:
  /**
   * Construct a Mobility Helper which is used to make life easier when working
   * with mobility models.
   */
  MobilityHelper ();

  /**
   * Destroy a Mobility Helper
   */
  ~MobilityHelper ();

  /**
   * Set the position allocator which will be used to allocate the initial 
   * position of every node initialized during MobilityModel::Install.
   *
   * \param allocator allocate initial node positions
   */
  void SetPositionAllocator (Ptr<PositionAllocator> allocator);

  /**
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
  void SetPositionAllocator (std::string type,
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
  void SetMobilityModel (std::string type,
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
   * \param reference item to push.
   *
   * Push an item on the top of the stack of "reference mobility models".
   * The input item should be a node instance to which a mobility model
   * has already been aggregated (usually by a call to Install).
   *
   * If this stack is not empty when MobilityHelper::Install
   * is called, the model from the top of the stack is used
   * to create a ns3::HierarchicalMobilityModel to make the
   * newly-created models define their positions relative to that
   * of the parent mobility model.
   *
   * This method is typically used to create hierarchical mobility
   * patterns and positions by starting with the large-scale mobility
   * features, and, then, defining the smaller-scale movements relative
   * to a few reference points in the large-scale model.
   */
  void PushReferenceMobilityModel (Ptr<Object> reference);
  /**
   * \param referenceName named item to push.
   *
   * Push an item on the top of the stack of "reference mobility models".
   * The input item should be a node instance to which a mobility model
   * has already been aggregated (usually by a call to Install).
   *
   * If this stack is not empty when MobilityHelper::Install
   * is called, the model from the top of the stack is used
   * to create a ns3::HierarchicalMobilityModel to make the
   * newly-created models define their positions relative to that
   * of the parent mobility model.
   *
   * This method is typically used to create hierarchical mobility
   * patterns and positions by starting with the large-scale mobility
   * features, and, then, defining the smaller-scale movements relative
   * to a few reference points in the large-scale model.
   */
  void PushReferenceMobilityModel (std::string referenceName);
  /**
   * Remove the top item from the top of the stack of
   * "reference mobility models".
   */
  void PopReferenceMobilityModel (void);

  /**
   * \return a string which contains the TypeId of the currently-selected
   *          mobility model.
   */
  std::string GetMobilityModelType (void) const;

  /**
   * \brief "Layout" a single node according to the current position allocator type.
   *
   * This method creates an instance of a ns3::MobilityModel subclass (the 
   * type of which was set with MobilityHelper::SetMobilityModel), aggregates
   * it to the provided node, and sets an initial position based on the current
   * position allocator (set through MobilityHelper::SetPositionAllocator). 
   *
   * \param node The node to "layout."
   */
  void Install (Ptr<Node> node) const;
  /**
   * \brief "Layout" a single node according to the current position allocator type.
   *
   * This method creates an instance of a ns3::MobilityModel subclass (the 
   * type of which was set with MobilityHelper::SetMobilityModel), aggregates
   * it to the provided node, and sets an initial position based on the current
   * position allocator (set through MobilityHelper::SetPositionAllocator). 
   *
   * \param nodeName The name of the node to "layout."
   */
  void Install (std::string nodeName) const;

  /**
   * \brief Layout a collection of nodes according to the current position allocator type.
   *
   * For each node in the provided NodeContainer, this method creates an instance 
   * of a ns3::MobilityModel subclass (the type of which was set with 
   * MobilityHelper::SetMobilityModel), aggregates it to the node, and sets an 
   * initial position based on the current position allocator (set through 
   * MobilityHelper::SetPositionAllocator). 
   *
   * \param container The set of nodes to layout.
   */
  void Install (NodeContainer container) const;

  /**
   * Perform the work of MobilityHelper::Install on _all_ nodes which
   * exist in the simulation.
   */
  void InstallAll (void);

  /**
   * \param stream an output stream wrapper
   * \param nodeid the id of the node to generate ascii output for.
   *
   * Enable ascii output to record course changes from the mobility model
   * associated with the specified nodeid and dump that to the specified output 
   * stream.  If the Node does not have a MobilityModel aggregated,
   * this method will not produce any output.
   */
  static void EnableAscii (Ptr<OutputStreamWrapper> stream, uint32_t nodeid);
  /**
   * \param stream an output stream wrapper
   * \param n node container
   *
   * Enable ascii output to record course changes from the mobility models
   * associated to the the nodes in the input container and dump that to the 
   * specified output stream.  Nodes that do not have a MobilityModel
   * aggregated will not result in any output.
   */
  static void EnableAscii (Ptr<OutputStreamWrapper> stream, NodeContainer n);
  /**
   * \param stream an output stream wrapper
   *
   * Enable ascii output to record course changes from the mobility models
   * associated to every node in the system and dump that to the specified 
   * output stream.  Nodes that do not have a MobilityModel aggregated
   * will not result in any output.
   */
  static void EnableAsciiAll (Ptr<OutputStreamWrapper> stream);
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
   * \param c NodeContainer of the set of nodes for which the MobilityModels
   * should be modified to use a fixed stream
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this helper
   */
  int64_t AssignStreams (NodeContainer c, int64_t stream);

  /**
   * \param n1 node 1
   * \param n2 node 2
   * \return the distance (squared), in meters, between two nodes
   */
  static double GetDistanceSquaredBetween (Ptr<Node> n1, Ptr<Node> n2);

private:

  /**
   * Output course change events from mobility model to output stream
   * \param stream output stream
   * \param mobility mobility model
   */
  static void CourseChanged (Ptr<OutputStreamWrapper> stream, Ptr<const MobilityModel> mobility);
  std::vector<Ptr<MobilityModel> > m_mobilityStack; //!< Internal stack of mobility models
  ObjectFactory m_mobility; //!< Object factory to create mobility objects
  Ptr<PositionAllocator> m_position; //!< Position allocator for use in hierarchical mobility model
};

} // namespace ns3

#endif /* MOBILITY_HELPER_H */
