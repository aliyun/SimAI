/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (C) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 *         Michele Polese <michele.polese@gmail.com> for the OutdoorPositionAllocator class
 */
#ifndef BUILDING_POSITION_ALLOCATOR_H
#define BUILDING_POSITION_ALLOCATOR_H

#include <ns3/ptr.h>
#include <ns3/position-allocator.h>
#include <ns3/node-container.h>
#include "ns3/random-variable-stream.h"

namespace ns3 {

class Building;
class UniformRandomVariable;


/**
 * Allocate each position by randomly choosing a building from the list
 * of all buildings, and then randomly choosing a position inside the building.
 */
class RandomBuildingPositionAllocator : public PositionAllocator
{
public:
  RandomBuildingPositionAllocator ();

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  // inherited from PositionAllocator
  virtual Vector GetNext (void) const;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

private:

  bool m_withReplacement; //!< If true, the building will be randomly selected with replacement
  mutable std::vector< Ptr<Building> > m_buildingListWithoutReplacement; //!< List of building without replacement

  /// Provides uniform random variables.
  Ptr<UniformRandomVariable> m_rand;
};

/**
 * \ingroup buildings
 * \brief allocate outdoor positions
 *
 * Allocate positions outside of existing buildings using rejection sampling.
 * This class extracts a random position in a box defined by the three 
 * RandomVariableStreams for the X, Y and Z dimensions (similarly to 
 * RandomBoxPositionAllocator), until a position is found that is outdoors 
 * with respect to all of the buildings in the scenario, or a maximum number 
 * of attempts is reached.  The RandomVariableStream and the maximum number 
 * of attempts can be set using attributes.  If the maximum number of 
 * attempts is reached, then the simulation aborts due to failure of properly
 * positioning the node.
 */
class OutdoorPositionAllocator : public PositionAllocator
{
public:
  OutdoorPositionAllocator ();

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  // inherited from PositionAllocator
  virtual Vector GetNext (void) const;

  /**
   * \brief Set the random variable stream object that generates x-positions
   * \param x pointer to a RandomVariableStream object
   */
  void SetX (Ptr<RandomVariableStream> x);
  /**
   * \brief Set the random variable stream object that generates y-positions
   * \param y pointer to a RandomVariableStream object
   */
  void SetY (Ptr<RandomVariableStream> y);
  /**
   * \brief Set the random variable stream object that generates z-positions
   * \param z pointer to a RandomVariableStream object
   */
  void SetZ (Ptr<RandomVariableStream> z);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

private:
  Ptr<RandomVariableStream> m_x; //!< pointer to x's random variable stream
  Ptr<RandomVariableStream> m_y; //!< pointer to y's random variable stream
  Ptr<RandomVariableStream> m_z; //!< pointer to z's random variable stream

  uint32_t m_maxAttempts; //!< maximum number of attempts before giving up
};

/**
 * Allocate each position by randomly choosing a room from the list
 * of all buildings, and then randomly choosing a position inside the room.
 * The selection of the room is always done without replacement.
 * 
 */
class RandomRoomPositionAllocator : public PositionAllocator
{
public:
  RandomRoomPositionAllocator ();

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  // inherited from PositionAllocator
  virtual Vector GetNext (void) const;

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  int64_t AssignStreams (int64_t stream);

private:

  /**
   * Room informations
   */
  struct RoomInfo 
  {
    Ptr<Building> b; //!< Building
    uint32_t roomx;  //!< Room (x coord)
    uint32_t roomy;  //!< Room (y coord)
    uint32_t floor;  //!< Room (floor number)
  };
  mutable std::vector<RoomInfo> m_roomListWithoutReplacement; //!< Container of rooms

  /// Provides uniform random variables.
  Ptr<UniformRandomVariable> m_rand;
};



/**
 * Walks a given NodeContainer sequentially, and for each node allocate a new
 * position randomly in the same room of that node
 *
 */
class SameRoomPositionAllocator : public PositionAllocator
{
public:
  SameRoomPositionAllocator ();

  /**
   * Constructor
   * \param c Node container
   */
  SameRoomPositionAllocator (NodeContainer c);

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  // inherited from PositionAllocator
  virtual Vector GetNext (void) const;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t);

private:

  NodeContainer m_nodes; //!< Nodes container
  mutable NodeContainer::Iterator m_nodeIt; //!< Nodes iterator

  /// Provides uniform random variables.
  Ptr<UniformRandomVariable> m_rand;
};

/**
 * Generate a random position uniformly distributed in the volume of a
 * chosen room inside a chosen building.  
 */
class FixedRoomPositionAllocator : public PositionAllocator
{
public:

  /** 
   * 
   * 
   * \param x index of the room on the x-axis 
   * \param y index of the room on the y-axis 
   * \param z index of the room on the z-axis (i.e., floor number)
   * \param b pointer to the chosen building
   * 
   */
  FixedRoomPositionAllocator (uint32_t x,
                              uint32_t y,
                              uint32_t z,
                              Ptr<Building> b);
  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  // inherited from PositionAllocator
  virtual Vector GetNext (void) const;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t);

private:

  uint32_t roomx; //!< Index of the room on the x-axis 
  uint32_t roomy; //!< Index of the room on the y-axis 
  uint32_t floor; //!< Index of the room on the z-axis (i.e., floor number)

  Ptr<Building> bptr; //!< Pointer to the chosen building

  /// Provides uniform random variables.
  Ptr<UniformRandomVariable> m_rand;
};


} // namespace ns3

#endif /* BUILDING_POSITION_ALLOCATOR_H */
