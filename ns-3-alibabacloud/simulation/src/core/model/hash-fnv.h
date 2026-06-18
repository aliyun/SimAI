/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Lawrence Livermore National Laboratory
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#ifndef HASH_FNV_H
#define HASH_FNV_H

#include "hash-function.h"

/**
 * \file
 * \ingroup hash
 * \brief ns3::Hash::Function::Fnv1a declaration.
 */

namespace ns3 {

namespace Hash {

namespace Function {

/**
 *  \ingroup hash
 *
 *  \brief Fnv1a hash function implementation
 *
 *  This is the venerable Fowler-Noll-Vo hash, version 1A.  (See the
 *  <a href="http://isthe.com/chongo/tech/comp/fnv/">FNV page</a>.)
 *
 *  The implementation here is taken directly from the published FNV
 *  <a href="http://isthe.com/chongo/tech/comp/fnv/#FNV-reference-source">
 *  reference code</a>,
 *  with minor modifications to wrap into this class.  See the
 *  hash-fnv.cc file for details.
 *
 */
class Fnv1a : public Implementation
{
public:
  /**
   * Constructor
   */
  Fnv1a ();
  /**
   * Compute 32-bit hash of a byte buffer
   *
   * Call clear () between calls to GetHash32() to reset the
   * internal state and hash each buffer separately.
   *
   * If you don't call clear() between calls to GetHash32,
   * you can hash successive buffers.  The final return value
   * will be the cumulative hash across all calls.
   *
   * \param [in] buffer pointer to the beginning of the buffer
   * \param [in] size length of the buffer, in bytes
   * \return 32-bit hash of the buffer
   */
  uint32_t  GetHash32  (const char * buffer, const size_t size);
  /**
   * Compute 64-bit hash of a byte buffer.
   *
   * Call clear () between calls to GetHash64() to reset the
   * internal state and hash each buffer separately.
   *
   * If you don't call clear() between calls to GetHash64,
   * you can hash successive buffers.  The final return value
   * will be the cumulative hash across all calls.
   *
   * \param [in] buffer pointer to the beginning of the buffer
   * \param [in] size length of the buffer, in bytes
   * \return 64-bit hash of the buffer
   */
  uint64_t  GetHash64  (const char * buffer, const size_t size);
  /**
   * Restore initial state
   */
  virtual void clear (void);

private:
  /**
   * Seed value
   */
  enum seed
  {
    SEED = 0x8BADF00D  /**< Ate bad food */
  };
  /** Cache last hash value, for incremental hashing. */
  /**@{*/
  uint32_t m_hash32;
  uint64_t m_hash64;
  /**@}*/

};  // class Fnv1a

}  // namespace Function

}  // namespace Hash

}  // namespace ns3

#endif  /* HASH_FNV_H */
