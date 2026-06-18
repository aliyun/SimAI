/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Georgia Institute of Technology 
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
 * Authors: Josh Pelkey <jpelkey@gatech.edu>
 */

#ifndef NIX_VECTOR_H
#define NIX_VECTOR_H

#include "ns3/ptr.h"
#include "ns3/simple-ref-count.h"
#include "ns3/buffer.h"

namespace ns3 {

/**
 * \ingroup packet
 *
 * \brief Neighbor-index data structure for nix-vector routing
 *
 * This data structure holds a vector of "neighbor-indexes" for 
 * a simulation specific routing protocol, nix-vector routing.
 * Theses neighbor-indexes correspond to the net-device which a 
 * node should use to route a packet.  A nix-vector is built
 * (or fetched from a cache) on-demand. The nix-vector is 
 * transmitted with the packet, and along each hop of the 
 * route, the current node extracts the appropriate 
 * neighbor-index and routes the packet.
 *
 * \internal
 * The implementation of NixVector uses a vector to store 
 * the neighbor-indexes.  Each entry in the vector is 32 
 * bits long and can store multiple neighbor-indexes.  A 
 * fair amount of bit manipulation is used to store these 
 * neighbor-indexes efficiently.  A vector is used so that 
 * the nix-vector can grow arbitrarily if the topology and 
 * route requires a large number of neighbor-indexes.
 *
 * As the nix-vector travels along the route, an internal 
 * private member variable keeps track of how many bits 
 * have been used.  At a particular node, the nix-vector 
 * is used to return the next neighbor-index. This 
 * neighbor-index is used to determine which net-device 
 * to use.  The number of bits used would then be 
 * incremented accordingly, and the packet would be 
 * routed.
 */

class NixVector : public SimpleRefCount<NixVector>
{
public:
  NixVector ();
  ~NixVector ();
  /**
   * \return a copy of this nix-vector
   */
  Ptr<NixVector> Copy (void) const;
  /**
   * \param o the NixVector to copy to a new NixVector
   *          using a constructor
   */
  NixVector (const NixVector &o);
  /**
   * \return a reference to the assignee
   *
   * \param o the NixVector to copy to a new NixVector using the
   *          equals operator
   */
  NixVector &operator = (const NixVector &o);
  /**
   * \param newBits the neighbor-index to be added to the vector
   * \param numberOfBits the number of bits that newBits contains
   *
   * Adds the neighbor index to the vector using a fair amount of
   * bit manipulation to pack everything in efficiently.
   *
   * Note: This function assumes that the number of bits to be added
   * is always less than or equal to 32, ie., you can only span one
   * entry of a nix-vector at a time.  This is reasonable, since 32
   * bits gives you 2^32 possible neighbors.
   */
  void AddNeighborIndex (uint32_t newBits, uint32_t numberOfBits);
  /**
   * \return the neighbor index
   *
   * \param numberOfBits the number of bits to extract from the vector
   *
   * Extracts the number of bits specified from
   * the vector and returns the value extracted
   *
   * Note: This function assumes that the number of bits to be extracted
   * is always less than or equal to 32, ie., you can only span one
   * entry of a nix-vector at a time.  This is reasonable, since 32
   * bits gives you 2^32 possible neighbors.
   */
  uint32_t ExtractNeighborIndex (uint32_t numberOfBits);
  /**
   * \return number of bits remaining in the
   *         nix-vector (ie m_total - m_used)
   */
  uint32_t GetRemainingBits (void) const;
  /**
   * \return the number of bytes required for serialization
   */
  uint32_t GetSerializedSize (void) const;
  /**
   * \return zero if buffer not large enough
   *
   * \param buffer points to serialization buffer
   *
   * \param maxSize max number of bytes to write
   *
   * This nix-vector is serialized into the raw character
   * buffer parameter.
   */
  uint32_t Serialize (uint32_t* buffer, uint32_t maxSize) const;
  /**
   * \return zero if a complete nix-vector is not deserialized
   *
   * \param buffer points to buffer for deserialization
   *
   * \param size number of bytes to deserialize
   *
   * The raw character buffer containing all the nix-vector
   * information is deserialized into this nix-vector.
   */
  uint32_t Deserialize (const uint32_t* buffer, uint32_t size);
  /**
   * \return number of bits of numberOfNeighbors
   *
   * \param numberOfNeighbors the total number of neighbors
   *
   * This function is used to determine the number of bits of
   * numberOfNeighbors so that this value can be passed in to
   * AddNeighborIndex or ExtractNeighborIndex.
   */
  uint32_t BitCount (uint32_t numberOfNeighbors) const;

  /**
   * Set the NixVector Epoch
   * \param epoch the NixVector Epoch
   */
  void SetEpoch (uint32_t epoch);

  /**
   * Get the NixVector Epoch
   * \return the NixVector Epoch
   */
  uint32_t GetEpoch () const;

private:
  /// Typedef: the NixVector bits storage.
  typedef std::vector<uint32_t> NixBits_t;

  /**
   * \brief Print the NixVector.
   *
   * \param os the output stream
   * 
   * \note: this could be greatly simplified by using std::format (but it's C++20).
   */
  void DumpNixVector (std::ostream &os) const;

  /**
   * \brief Stream insertion operator.
   *
   * \param os the stream
   * \param nix the Nixvector
   * \returns a reference to the stream
   */
  friend std::ostream & operator << ( std::ostream &os, const NixVector &nix);

  NixBits_t m_nixVector; //!< the actual nix-vector
  uint32_t m_used; //!< For tracking where we are in the nix-vector

  /**
   * A counter of how total bits are in
   * the nix-vector
   */
  uint32_t m_totalBitSize;

  uint32_t m_epoch; //!< Epoch of the Nix-vector creation

  /**
   * Internal for pretty printing of nix-vector (no fill)
   * \param decimalNum decimal divider
   * \param bitCount bit counter
   * \param os output stream
   */
  void PrintDec2BinNix (uint32_t decimalNum, uint32_t bitCount, std::ostream &os) const;
};
} // namespace ns3

#endif /* NIX_VECTOR_H */
