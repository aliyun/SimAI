/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Wireless Communications and Networking Group (WCNG),
 * University of Rochester, Rochester, NY, USA.
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
 * Author: Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */

#ifndef ENERGY_HARVESTER_H
#define ENERGY_HARVESTER_H

#include <iostream>

// include from ns-3
#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/type-id.h"
#include "ns3/node.h"
#include "ns3/energy-source-container.h"

namespace ns3 {
    
class EnergySource;

/**
 * \ingroup energy
 *
 * \brief Energy harvester base class.
 */

class EnergyHarvester : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  EnergyHarvester ();

  virtual ~EnergyHarvester ();

  /**
   * \brief Sets pointer to node containing this EnergyHarvester.
   *
   * \param node Pointer to node containing this EnergyHarvester.
   */
  void SetNode (Ptr<Node> node);

  /**
   * \returns Pointer to node containing this EnergyHarvester.
   *
   * When a subclass needs to get access to the underlying node base class to
   * print the nodeId for example, it can invoke this method.
   */
  Ptr<Node> GetNode (void) const;

  /**
   * \param source Pointer to energy source to which this EnergyHarvester is
   * installed.
   *
   * This function sets the pointer to the energy source connected to the energy
   * harvester.
   */
  void SetEnergySource (Ptr<EnergySource> source);

  /**
   * \returns Pointer to energy source connected to the harvester.
   *
   * When a subclass needs to get access to the connected energy source,
   * it can invoke this method.
   */
  Ptr<EnergySource> GetEnergySource (void) const;

  /**
   * \returns Amount of power currently provided by the harvester.
   *
   * This method is called by the energy source connected to the harvester in order
   * to determine the amount of energy that the harvester provided since last update.
   */
  double GetPower (void) const;

private:
  /**
   *
   * Defined in ns3::Object
   */
  virtual void DoDispose (void);

  /**
   * This method is called by the GetPower method and it needs to be implemented by the
   * subclasses of the energy harvester. It returns the actual amount of power that is
   * currently provided by the energy harvester.
   *
   * This method should be used to connect the logic behind the particular implementation
   * of the energy harvester with the energy source.
   * 
   * \returns Amount of power currently provided by the harvester.
   */
  virtual double DoGetPower (void) const;

private:
  /**
   * Pointer to node containing this EnergyHarvester. Used by helper class to make
   * sure energy harvesters are installed onto the corresponding node.
   */
  Ptr<Node> m_node;

  /**
   * Pointer to the Energy Source to which this EnergyHarvester is connected. Used
   * by helper class to make sure energy harvesters are installed onto the
   * corresponding energy source.
   */
  Ptr<EnergySource> m_energySource;

protected:

};
    
} // namespace ns3

#endif /* defined(ENERGY_HARVESTER_H) */
