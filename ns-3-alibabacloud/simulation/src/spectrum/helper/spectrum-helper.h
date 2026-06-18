/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef SPECTRUM_HELPER_H
#define SPECTRUM_HELPER_H

#include <string>
#include <ns3/attribute.h>
#include <ns3/object-factory.h>
#include <ns3/node-container.h>
#include <ns3/net-device-container.h>
#include <ns3/propagation-loss-model.h>
#include <ns3/spectrum-propagation-loss-model.h>

namespace ns3 {

class SpectrumPhy;
class SpectrumChannel;
class Node;
class NetDevice;


/**
 * \ingroup spectrum
 * \brief Setup a SpectrumChannel
 */
class SpectrumChannelHelper
{
public:
  /**
   * \brief Setup a default SpectrumChannel. The Default mode is:
   * Channel: "ns3::SingleModelSpectrumChannel",
   * PropagationDelay: "ns3::ConstantSpeedPropagationDelayModel", and
   * SpectrumPropagationLoss: "ns3::FriisSpectrumPropagationLossModel".
   *
   * \returns a Default-configured SpectrumChannelHelper
   */
  static SpectrumChannelHelper Default ();

  /**
   * \param type the type of the SpectrumChannel to use
   * \param n0 the name of the attribute to set
   * \param v0 the value of the attribute to set
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   * \param n2 the name of the attribute to set
   * \param v2 the value of the attribute to set
   * \param n3 the name of the attribute to set
   * \param v3 the value of the attribute to set
   * \param n4 the name of the attribute to set
   * \param v4 the value of the attribute to set
   * \param n5 the name of the attribute to set
   * \param v5 the value of the attribute to set
   * \param n6 the name of the attribute to set
   * \param v6 the value of the attribute to set
   * \param n7 the name of the attribute to set
   * \param v7 the value of the attribute to set
   */
  void SetChannel (std::string type,
                   std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                   std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                   std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                   std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                   std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                   std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                   std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                   std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue ());
  /**
   * \param name the name of the model to set
   * \param n0 the name of the attribute to set
   * \param v0 the value of the attribute to set
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   * \param n2 the name of the attribute to set
   * \param v2 the value of the attribute to set
   * \param n3 the name of the attribute to set
   * \param v3 the value of the attribute to set
   * \param n4 the name of the attribute to set
   * \param v4 the value of the attribute to set
   * \param n5 the name of the attribute to set
   * \param v5 the value of the attribute to set
   * \param n6 the name of the attribute to set
   * \param v6 the value of the attribute to set
   * \param n7 the name of the attribute to set
   * \param v7 the value of the attribute to set
   *
   * Add a new single-frequency propagation loss model to this channel helper.
   */
  void AddPropagationLoss (std::string name,
                           std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                           std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                           std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                           std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                           std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                           std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                           std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                           std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue ());


  /** 
   * Add a new single-frequency propagation loss model instance to this channel helper.
   * 
   * \param m a pointer to the instance of the propagation loss model
   */
  void AddPropagationLoss (Ptr<PropagationLossModel> m);

  /**
   * \param name the name of the model to set
   * \param n0 the name of the attribute to set
   * \param v0 the value of the attribute to set
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   * \param n2 the name of the attribute to set
   * \param v2 the value of the attribute to set
   * \param n3 the name of the attribute to set
   * \param v3 the value of the attribute to set
   * \param n4 the name of the attribute to set
   * \param v4 the value of the attribute to set
   * \param n5 the name of the attribute to set
   * \param v5 the value of the attribute to set
   * \param n6 the name of the attribute to set
   * \param v6 the value of the attribute to set
   * \param n7 the name of the attribute to set
   * \param v7 the value of the attribute to set
   *
   * Add a new frequency-dependent propagation loss model to this channel helper.
   */
  void AddSpectrumPropagationLoss (std::string name,
                                   std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                                   std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                                   std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                                   std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                                   std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                                   std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                                   std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                                   std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue ());

  /** 
   * Add a new frequency-dependent propagation loss model instance to this channel helper.
   * 
   * \param m a pointer to the instance of the propagation loss model
   */
  void AddSpectrumPropagationLoss (Ptr<SpectrumPropagationLossModel> m);

  /**
   * \param name the name of the model to set
   * \param n0 the name of the attribute to set
   * \param v0 the value of the attribute to set
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   * \param n2 the name of the attribute to set
   * \param v2 the value of the attribute to set
   * \param n3 the name of the attribute to set
   * \param v3 the value of the attribute to set
   * \param n4 the name of the attribute to set
   * \param v4 the value of the attribute to set
   * \param n5 the name of the attribute to set
   * \param v5 the value of the attribute to set
   * \param n6 the name of the attribute to set
   * \param v6 the value of the attribute to set
   * \param n7 the name of the attribute to set
   * \param v7 the value of the attribute to set
   *
   * Configure a propagation delay for this channel.
   */
  void SetPropagationDelay (std::string name,
                            std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                            std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                            std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                            std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                            std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                            std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                            std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                            std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue ());

  /**
   * \returns a new channel
   *
   * Create a channel based on the configuration parameters set previously.
   */
  Ptr<SpectrumChannel> Create (void) const;

private:
  Ptr<SpectrumPropagationLossModel> m_spectrumPropagationLossModel; //!< Spectrum propagation loss model
  Ptr<PropagationLossModel> m_propagationLossModel; //!< Propagation loss model
  ObjectFactory m_propagationDelay; //!< Propagation delay
  ObjectFactory m_channel; //!< Channel
};



/**
 * \ingroup spectrum
 *
 * Create and configure several SpectrumPhy instances and connect them to a channel.
 */
class SpectrumPhyHelper
{
public:
  /**
   * \param name the type of SpectrumPhy to use
   * \param n0 the name of the attribute to set
   * \param v0 the value of the attribute to set
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   * \param n2 the name of the attribute to set
   * \param v2 the value of the attribute to set
   * \param n3 the name of the attribute to set
   * \param v3 the value of the attribute to set
   * \param n4 the name of the attribute to set
   * \param v4 the value of the attribute to set
   * \param n5 the name of the attribute to set
   * \param v5 the value of the attribute to set
   * \param n6 the name of the attribute to set
   * \param v6 the value of the attribute to set
   * \param n7 the name of the attribute to set
   * \param v7 the value of the attribute to set
   */
  void SetPhy (std::string name,
               std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
               std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
               std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
               std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
               std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
               std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
               std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
               std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue ());

  /**
   * set the channel that will be used by SpectrumPhy instances created by this helper
   *
   * @param channel
   */
  void SetChannel (Ptr<SpectrumChannel> channel);

  /**
   * set the channel that will be used by SpectrumPhy instances created by this helper
   *
   * @param channelName
   */
  void SetChannel (std::string channelName);

  /**
   * \param name the name of the attribute to set
   * \param v the value of the attribute
   *
   * Set an attribute of the SpectrumPhy instances to be created
   */
  void SetPhyAttribute (std::string name, const AttributeValue &v);

  /**
   *
   * @param node
   * @param device
   *
   * @return a  newly created SpectrumPhy instance
   */
  Ptr<SpectrumPhy> Create (Ptr<Node> node, Ptr<NetDevice> device) const;



private:
  ObjectFactory m_phy;      //!< Object factory for the phy objects
  Ptr<SpectrumChannel> m_channel; //!< Channel
};


} // namespace ns3

#endif /* SPECTRUM_HELPER_H */
