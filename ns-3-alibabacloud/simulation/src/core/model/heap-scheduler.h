/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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

#ifndef HEAP_SCHEDULER_H
#define HEAP_SCHEDULER_H

#include "scheduler.h"
#include <stdint.h>
#include <vector>

/**
 * \file
 * \ingroup scheduler
 * ns3::HeapScheduler declaration.
 */

namespace ns3 {

/**
 * \ingroup scheduler
 * \brief a binary heap event scheduler
 *
 * This code started as a c++ translation of a Java-based code written in 2005
 * to implement a heap sort. So, this binary heap is really a pretty
 * straightforward implementation of the classic data structure,
 * implemented on a `std::vector`.  This implementation does not make use
 * of any of the heap functions from the STL.  Not much to say
 * about it.
 *
 * What is smart about this code ?
 *  - it does not use the index 0 in the array to avoid having to convert
 *    C-style array indexes (which start at zero) and heap-style indexes
 *    (which start at 1). This is why _all_ indexes start at 1, and that
 *    the index of the root is 1.
 *  - It uses a slightly non-standard while loop for top-down heapify
 *    to move one if statement out of the loop.
 *
 * \par Time Complexity
 *
 * Operation    | Amortized %Time | Reason
 * :----------- | :-------------- | :-----
 * Insert()     | Logarithmic     | Heapify
 * IsEmpty()    | Constant        | Explicit queue size
 * PeekNext()   | Constant        | Heap kept sorted
 * Remove()     | Logarithmic     | Search, heapify
 * RemoveNext() | Logarithmic     | Heapify
 *
 * \par Memory Complexity
 *
 * Category  | Memory                           | Reason
 * :-------- | :------------------------------- | :-----
 * Overhead  | 3 x `sizeof (*)`<br/>(24 bytes)  | `std::vector`
 * Per Event | 0                                | Events stored in `std::vector` directly
 */
class HeapScheduler : public Scheduler
{
public:
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /** Constructor. */
  HeapScheduler ();
  /** Destructor. */
  virtual ~HeapScheduler ();

  // Inherited
  virtual void Insert (const Scheduler::Event &ev);
  virtual bool IsEmpty (void) const;
  virtual Scheduler::Event PeekNext (void) const;
  virtual Scheduler::Event RemoveNext (void);
  virtual void Remove (const Scheduler::Event &ev);

private:
  /** Event list type:  vector of Events, managed as a heap. */
  typedef std::vector<Scheduler::Event> BinaryHeap;

  /**
   * Get the parent index of a given entry.
   *
   * \param [in] id The child index.
   * \return The index of the parent of \pname{id}.
   */
  inline std::size_t Parent (std::size_t id) const;
  /**
   * Get the next sibling of a given entry.
   *
   * \param [in] id The starting index.
   * \returns The next sibling of \pname{id}.
   */
  std::size_t Sibling (std::size_t id) const;
  /**
   * Get the left child of a given entry.
   *
   * \param [in] id The parent index.
   * \returns The index of the left (first) child.
   */
  inline std::size_t LeftChild (std::size_t id) const;
  /**
   * Get the right child index of a given entry.
   *
   * \param [in] id The parent index.
   * \returns The index of the right (second) child.
   */
  inline std::size_t RightChild (std::size_t id) const;
  /**
   * Get the root index of the heap.
   *
   * \returns The root index.
   */
  inline std::size_t Root (void) const;
  /**
   * Return the index of the last element.
   * \returns The last index.
   */
  std::size_t Last (void) const;
  /**
   * Test if an index is the root.
   *
   * \param [in] id The index to test.
   * \returns \c true if the \pname{id} is the root.
   */
  inline bool IsRoot (std::size_t id) const;
  /**
   * Test if an index is at the bottom of the heap.
   *
   * \param [in] id The index to test.
   * \returns \c true if the index is at the bottom.
   */
  inline bool IsBottom (std::size_t id) const;
  /**
   * Compare (less than) two items.
   *
   * \param [in] a The first item.
   * \param [in] b The second item.
   * \returns \c true if \c a < \c b
   */
  inline bool IsLessStrictly (std::size_t a, std::size_t b) const;
  /**
   * Minimum of two items.
   *
   * \param [in] a The first item.
   * \param [in] b The second item.
   * \returns The smaller of the two items.
   */
  inline std::size_t Smallest (std::size_t a, std::size_t b) const;
  /**
   * Swap two items.
   *
   * \param [in] a The first item.
   * \param [in] b The second item.
   */
  inline void Exch (std::size_t a, std::size_t b);
  /** Percolate a newly inserted Last item to its proper position. */
  void BottomUp (void);
  /**
   * Percolate a deletion bubble down the heap.
   *
   * \param [in] start Starting entry.
   */
  void TopDown (std::size_t start);

  /** The event list. */
  BinaryHeap m_heap;
};

} // namespace ns3

#endif /* HEAP_SCHEDULER_H */
