/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2007 University of Washington
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
 * Author:  Craig Dowell (craigdo@ee.washington.edu)
 */

#ifndef CANDIDATE_QUEUE_H
#define CANDIDATE_QUEUE_H

#include <stdint.h>
#include <list>
#include "ns3/ipv4-address.h"

namespace ns3 {

class SPFVertex;

/**
 * \ingroup globalrouting
 *
 * \brief A Candidate Queue used in routing calculations.
 *
 * The CandidateQueue is used in the OSPF shortest path computations.  It
 * is a priority queue used to store candidates for the shortest path to a
 * given network.
 *
 * The queue holds Shortest Path First Vertex pointers and orders them
 * according to the lowest value of the field m_distanceFromRoot.  Remaining
 * vertices are ordered according to increasing distance.  This implements a
 * priority queue.
 *
 * Although a STL priority_queue almost does what we want, the requirement
 * for a Find () operation, the dynamic nature of the data and the derived
 * requirement for a Reorder () operation led us to implement this simple 
 * enhanced priority queue.
 */
class CandidateQueue
{
public:
/**
 * @brief Create an empty SPF Candidate Queue.
 *
 * @see SPFVertex
 */
  CandidateQueue ();

/**
 * @brief Destroy an SPF Candidate Queue and release any resources held 
 * by the contents.
 *
 * @see SPFVertex
 */
  virtual ~CandidateQueue ();

/**
 * @brief Empty the Candidate Queue and release all of the resources 
 * associated with the Shortest Path First Vertex pointers in the queue.
 *
 * @see SPFVertex
 */
  void Clear (void);

/**
 * @brief Push a Shortest Path First Vertex pointer onto the queue according
 * to the priority scheme.
 * 
 * On completion, the top of the queue will hold the Shortest Path First
 * Vertex pointer that points to a vertex having lowest value of the field
 * m_distanceFromRoot.  Remaining vertices are ordered according to 
 * increasing distance.
 *
 * @see SPFVertex
 * @param vNew The Shortest Path First Vertex to add to the queue.
 */
  void Push (SPFVertex *vNew);

/**
 * @brief Pop the Shortest Path First Vertex pointer at the top of the queue.
 *
 * The caller is given the responsibility for releasing the resources 
 * associated with the vertex.
 *
 * @see SPFVertex
 * @see Top ()
 * @returns The Shortest Path First Vertex pointer at the top of the queue.
 */
  SPFVertex* Pop (void);

/**
 * @brief Return the Shortest Path First Vertex pointer at the top of the 
 * queue.
 *
 * This method does not pop the SPFVertex* off of the queue, it simply 
 * returns the pointer.
 *
 * @see SPFVertex
 * @see Pop ()
 * @returns The Shortest Path First Vertex pointer at the top of the queue.
 */
  SPFVertex* Top (void) const;

/**
 * @brief Test the Candidate Queue to determine if it is empty.
 *
 * @returns True if the queue is empty, false otherwise.
 */
  bool Empty (void) const;

/**
 * @brief Return the number of Shortest Path First Vertex pointers presently
 * stored in the Candidate Queue.
 *
 * @see SPFVertex
 * @returns The number of SPFVertex* pointers in the Candidate Queue.
 */
  uint32_t Size (void) const;

/**
 * @brief Searches the Candidate Queue for a Shortest Path First Vertex 
 * pointer that points to a vertex having the given IP address.
 *
 * @see SPFVertex
 * @param addr The IP address to search for.
 * @returns The SPFVertex* pointer corresponding to the given IP address.
 */
  SPFVertex* Find (const Ipv4Address addr) const;

/**
 * @brief Reorders the Candidate Queue according to the priority scheme.
 * 
 * On completion, the top of the queue will hold the Shortest Path First
 * Vertex pointer that points to a vertex having lowest value of the field
 * m_distanceFromRoot.  Remaining vertices are ordered according to 
 * increasing distance.
 *
 * This method is provided in case the values of m_distanceFromRoot change
 * during the routing calculations.
 *
 * @see SPFVertex
 */
  void Reorder (void);

private:
/**
 * Candidate Queue copy construction is disallowed (not implemented) to 
 * prevent the compiler from slipping in incorrect versions that don't
 * properly deal with deep copies.
 * \param sr object to copy
 */
  CandidateQueue (CandidateQueue& sr);

/**
 * Candidate Queue assignment operator is disallowed (not implemented) to
 * prevent the compiler from slipping in incorrect versions that don't
 * properly deal with deep copies.
 * \param sr object to assign
 * \return copied object
 */
  CandidateQueue& operator= (CandidateQueue& sr);
/**
 * \brief return true if v1 < v2
 *
 * SPFVertexes are added into the queue according to the ordering
 * defined by this method. If v1 should be popped before v2, this 
 * method return true; false otherwise
 *
 * \param v1 first operand
 * \param v2 second operand
 * \return True if v1 should be popped before v2; false otherwise
 */
  static bool CompareSPFVertex (const SPFVertex* v1, const SPFVertex* v2);

  typedef std::list<SPFVertex*> CandidateList_t; //!< container of SPFVertex pointers
  CandidateList_t m_candidates;  //!< SPFVertex candidates

  /**
   * \brief Stream insertion operator.
   *
   * \param os the reference to the output stream
   * \param q the CandidateQueue
   * \returns the reference to the output stream
   */
  friend std::ostream& operator<< (std::ostream& os, const CandidateQueue& q);
};

} // namespace ns3

#endif /* CANDIDATE_QUEUE_H */
