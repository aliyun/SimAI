/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Author: Mirko Banchi <mk.banchi@gmail.com>
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef WAVE_MAC_HELPER_H
#define WAVE_MAC_HELPER_H
#include "ns3/wifi-mac-helper.h"

namespace ns3 {

/**
 * \ingroup wave
 * \brief Nqos Wave Mac Helper class
 */
class NqosWaveMacHelper : public WifiMacHelper
{
public:
  /**
   * Create a NqosWaveMacHelper to make life easier for people who want to
   * work with non-QOS Wave MAC layers.
   */
  NqosWaveMacHelper (void);

  /**
   * Destroy a NqosWaveMacHelper.
   */
  virtual ~NqosWaveMacHelper (void);
  /**
   * Create a mac helper in a default working state.
   * i.e., this is an ocb mac by default.
   * \returns NqosWaveMacHelper
   */
  static NqosWaveMacHelper Default (void);
  /**
   * \param type the type of ns3::WifiMac to create.
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
   * \param n8 the value of the attribute to set
   * \param v8 the value of the attribute to set
   * \param n9 the value of the attribute to set
   * \param v9 the value of the attribute to set
   * \param n10 the value of the attribute to set
   * \param v10 the value of the attribute to set
   *
   * All the attributes specified in this method should exist
   * in the requested mac.
   *
   * note: Here we require users set type with OcbWifiMac or its
   * subclass, otherwise it will become error
   */
  virtual void SetType (std::string type,
                        std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                        std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                        std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                        std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                        std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                        std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                        std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                        std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue (),
                        std::string n8 = "", const AttributeValue &v8 = EmptyAttributeValue (),
                        std::string n9 = "", const AttributeValue &v9 = EmptyAttributeValue (),
                        std::string n10 = "", const AttributeValue &v10 = EmptyAttributeValue ());
};

/**
 * \ingroup wave
 * \brief Qos Wave Mac Helper class
 */
class QosWaveMacHelper : public WifiMacHelper
{
public:
  /**
   * Create a QosWaveMacHelper that is used to make life easier when working
   * with Wifi 802.11p devices using a QOS MAC layer.
   */
  QosWaveMacHelper (void);

  /**
   * Destroy a QosWaveMacHelper
   */
  virtual ~QosWaveMacHelper (void);

  /**
   * Create a mac helper in a default working state.
   * \return A mac helper
   */
  static QosWaveMacHelper Default (void);

  /**
   * \param type the type of ns3::WifiMac to create.
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
   * \param n8 the value of the attribute to set
   * \param v8 the value of the attribute to set
   * \param n9 the value of the attribute to set
   * \param v9 the value of the attribute to set
   * \param n10 the value of the attribute to set
   * \param v10 the value of the attribute to set
   *
   * All the attributes specified in this method should exist
   * in the requested mac.
   *
   * note: Here we require users set type with OcbWifiMac or its
   * subclass, otherwise it will become error
   */
  virtual void SetType (std::string type,
                        std::string n0 = "", const AttributeValue &v0 = EmptyAttributeValue (),
                        std::string n1 = "", const AttributeValue &v1 = EmptyAttributeValue (),
                        std::string n2 = "", const AttributeValue &v2 = EmptyAttributeValue (),
                        std::string n3 = "", const AttributeValue &v3 = EmptyAttributeValue (),
                        std::string n4 = "", const AttributeValue &v4 = EmptyAttributeValue (),
                        std::string n5 = "", const AttributeValue &v5 = EmptyAttributeValue (),
                        std::string n6 = "", const AttributeValue &v6 = EmptyAttributeValue (),
                        std::string n7 = "", const AttributeValue &v7 = EmptyAttributeValue (),
                        std::string n8 = "", const AttributeValue &v8 = EmptyAttributeValue (),
                        std::string n9 = "", const AttributeValue &v9 = EmptyAttributeValue (),
                        std::string n10 = "", const AttributeValue &v10 = EmptyAttributeValue ());
};

}

#endif /* WAVE_MAC_HELPER_H */
