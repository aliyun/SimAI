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

#ifndef HASHFUNCTION_H
#define HASHFUNCTION_H

#include <cstring>  // memcpy
#include "simple-ref-count.h"

/**
 * \file
 * \ingroup hash
 * \brief ns3::Hash::Implementation, ns3::Hash::Function::Hash32 and
 * ns3::Hash::Function::Hash64 declarations.
 */

namespace ns3 {

/**
 * \ingroup hash
 * Hash function implementations
 */
namespace Hash {

/**
 *  \ingroup hash
 *
 *  \brief Hash function implementation base class.
 */
class Implementation : public SimpleRefCount<Implementation>
{
public:
  /**
   * Compute 32-bit hash of a byte buffer
   *
   * Call clear() between calls to GetHash32() to reset the
   * internal state and hash each buffer separately.
   *
   * If you don't call clear() between calls to GetHash32,
   * you can hash successive buffers.  The final return value
   * will be the cumulative hash across all calls.
   *
   * \param [in] buffer Pointer to the beginning of the buffer.
   * \param [in] size Length of the buffer, in bytes.
   * \return 32-bit hash of the buffer.
   */
  virtual uint32_t  GetHash32  (const char * buffer, const std::size_t size) = 0;
  /**
   * Compute 64-bit hash of a byte buffer.
   *
   * Default implementation returns 32-bit hash, with a warning.
   *
   * Call clear() between calls to GetHash64() to reset the
   * internal state and hash each buffer separately.
   *
   * If you don't call clear() between calls to GetHash64,
   * you can hash successive buffers.  The final return value
   * will be the cumulative hash across all calls.
   *
   * \param [in] buffer Pointer to the beginning of the buffer.
   * \param [in] size Length of the buffer, in bytes.
   * \return 64-bit hash of the buffer.
   */
  virtual uint64_t  GetHash64  (const char * buffer, const std::size_t size);
  /**
   * Restore initial state.
   */
  virtual void clear (void) = 0;
  /**
   * Constructor.
   */
  Implementation ()
  { }
  /**
   * Destructor.
   */
  virtual ~Implementation ()
  { }
};  // Hashfunction


/*--------------------------------------
 *  Hash function implementation
 *  by function pointers and templates
 */

/**
 *
 * \ingroup hash
 *
 * \brief Function pointer signatures for basic hash functions.
 *
 * See Hash::Function::Hash32 or Hash::Function::Hash64
 * @{
 */
typedef uint32_t (*Hash32Function_ptr) (const char *, const std::size_t);
typedef uint64_t (*Hash64Function_ptr) (const char *, const std::size_t);
/**@}*/

/**
 * \ingroup hash
 * Hash functions.
 */
namespace Function {

/**
 * \ingroup hash
 *
 * \brief Template for creating a Hash::Implementation from
 * a 32-bit hash function.
 */
class Hash32 : public Implementation
{
public:
  /**
   * Constructor from a 32-bit hash function pointer.
   *
   * \param [in] hp Function pointer to a 32-bit hash function.
   */
  Hash32 (Hash32Function_ptr hp) : m_fp (hp)
  { }
  uint32_t GetHash32 (const char * buffer, const std::size_t size)
  {
    return (*m_fp)(buffer, size);
  }
  void clear ()
  { }

private:
  Hash32Function_ptr m_fp;  /**< The hash function. */
};  // Hash32

/**
 * \ingroup hash
 *
 * \brief Template for creating a Hash::Implementation from
 * a 64-bit hash function.
 */
class Hash64 : public Implementation
{
public:
  /**
   * Constructor from a 64-bit hash function pointer.
   *
   * \param [in] hp Function pointer to a 64-bit hash function.
   */
  Hash64 (Hash64Function_ptr hp) : m_fp (hp)
  { }
  uint64_t GetHash64 (const char * buffer, const std::size_t size)
  {
    return (*m_fp)(buffer, size);
  }
  uint32_t GetHash32 (const char * buffer, const std::size_t size)
  {
    uint32_t hash32;
    uint64_t hash64 = GetHash64 (buffer, size);

    memcpy (&hash32, &hash64, sizeof (hash32));
    return hash32;
  }
  void clear ()
  { }

private:
  Hash64Function_ptr m_fp;  /**< The hash function. */
};  // Hash64<Hash64Function_ptr>


}  // namespace Function

}  // namespace Hash

}  // namespace ns3

#endif /* HASHFUNCTION_H */

