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

#ifndef HASH_H
#define HASH_H

#include <string>

#include "assert.h"
#include "ptr.h"

#include "hash-function.h"
#include "hash-murmur3.h"
#include "hash-fnv.h"

/**
 * \file
 * \ingroup hash
 * \brief ns3::Hasher, ns3::Hash32() and ns3::Hash64() function declarations.
 */

namespace ns3 {

/**
 * \ingroup core
 * \defgroup hash Hash Functions
 *
 *  \brief Generic Hash function interface.
 *
 * See \ref Hasher for main entry point.
 * See \ref hash-example.cc for example usage.
 */
/**
 *  \ingroup hash
 *
 *  \brief Generic Hash function interface.
 *
 *  This class provides a generic interface for computing hashes
 *  of buffers.  Various getters return hashes of different lengths.
 *
 *  Call clear() between calls to the getter to reset the
 *  internal state and hash each buffer separately.
 *
 *  If you don't call clear() between calls to the getter
 *  you can hash successive buffers.  The final return value
 *  will be the cumulative hash across all calls.
 *
 *  The choice of hash function can be made at construction by
 *    \code
 *    Hasher hasher = Hasher ( Create<Hash::Function::Fnv1a> () );
 *    uint32_t hash = Hasher.GetHash32 (data);
 *    \endcode
 *
 *  The available implementations are documented in \ref hash.
 *  The default implementation is Murmur3.  FNV1a is also available.
 *
 *  In addition to this class interface, global functions are
 *  defined which use the default hash implementation.
 *
 * \internal
 *
 *  Would be nice to offer longer hashes.  \c uint128_t looks doable,
 *  except that our fallback \c int64x64_t implementation doesn't
 *  offer \c unsigned.
 *
 *  Longer hashes require returning a byte buffer of some sort,
 *  but our \ref Buffer class seems a bit overkill for this case.
 *
 */
class Hasher
{
public:
  /**
   * Constructor using the default implementation.
   */
  Hasher ();
  /**
   * Constructor using the supplied implementation.
   *
   * \param [in] hp Ptr<Hash::Implementation> to the desired implementation.
   */
  Hasher (Ptr<Hash::Implementation> hp);
  /**
   * Compute 32-bit hash of a byte buffer.
   *
   * Call clear () between calls to GetHash32() to reset the
   * internal state and hash each buffer separately.
   *
   * If you don't call clear() between calls to GetHash32,
   * you can hash successive buffers.  The final return value
   * will be the cumulative hash across all calls.
   *
   * \param [in] buffer Pointer to the beginning of the buffer.
   * \param [in] size Length of the buffer, in bytes.
   * \return 32-bit hash of the buffer..
   */
  uint32_t  GetHash32  (const char * buffer, const std::size_t size);
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
   * \param [in] buffer Pointer to the beginning of the buffer.
   * \param [in] size Length of the buffer, in bytes.
   * \return 64-bit hash of the buffer.
   */
  uint64_t  GetHash64  (const char * buffer, const std::size_t size);

  /**
   * Compute 32-bit hash of a string.
   *
   * Call clear () between calls to GetHash32() to reset the
   * internal state and hash each string separately.
   *
   * If you don't call clear() between calls to GetHash32,
   * you can hash successive strings.  The final return value
   * will be the cumulative hash across all calls.
   *
   * \param [in] s String to hash.
   * \return 32-bit hash of the string.
   */
  uint32_t  GetHash32  (const std::string s);
  /**
   * Compute 64-bit hash of a string.
   *
   * Call clear () between calls to GetHash64() to reset the
   * internal state and hash each string separately.
   *
   * If you don't call clear() between calls to GetHash64,
   * you can hash successive strings.  The final return value
   * will be the cumulative hash across all calls.
   *
   * \param [in] s String to hash.
   * \return 64-bit hash of the string.
   */
  uint64_t  GetHash64  (const std::string s);
  /**
   * Restore initial state.
   *
   * Returning this Hasher allows code like this:
   *
   * \code
   *   Hasher h;
   *   h.GetHash32 (...);
   *   ...
   *   h.clear ().GetHash64 (...);
   * \endcode
   *
   * \return This hasher.
   */
  Hasher & clear (void);

private:
  Ptr<Hash::Implementation> m_impl;  /**< Hash implementation. */
};  // Hasher


/*************************************************
 **  Global functions declarations
 ************************************************/

/**
 * \ingroup hash
 *
 * Compute 32-bit hash of a byte buffer, using the default hash function.
 *
 * \param [in] buffer Pointer to the beginning of the buffer.
 * \param [in] size Length of the buffer, in bytes.
 * \return 32-bit hash of the buffer.
 */
uint32_t Hash32 (const char * buffer, const std::size_t size);
/**
 * \ingroup hash
 *
 * Compute 64-bit hash of a byte buffer, using the default hash function.
 *
 * \param [in] buffer Pointer to the beginning of the buffer.
 * \param [in] size Length of the buffer, in bytes.
 * \return 64-bit hash of the buffer.
 */
uint64_t Hash64 (const char * buffer, const std::size_t size);

/**
 * \ingroup hash
 *
 * Compute 32-bit hash of a string, using the default hash function.
 *
 * \param [in] s String to hash.
 * \return 32-bit hash of the string.
 */
uint32_t Hash32 (const std::string s);
/**
 * \ingroup hash
 *
 * Compute 64-bit hash of a string, using the default hash function.
 *
 * \param [in] s String to hash.
 * \return 64-bit hash of the string.
 */
uint64_t Hash64 (const std::string s);

}  // namespace ns3


/*************************************************
 **  Inline implementations for rvo
 ************************************************/

namespace ns3 {

/*************************************************
 class Hasher implementation, inlined for rvo
*/

inline
uint32_t
Hasher::GetHash32  (const char * buffer, const std::size_t size)
{
  NS_ASSERT (m_impl != 0);
  return m_impl->GetHash32  (buffer, size);
}

inline
uint64_t
Hasher::GetHash64  (const char * buffer, const std::size_t size)
{
  NS_ASSERT (m_impl != 0);
  return m_impl->GetHash64  (buffer, size);
}

inline
uint32_t
Hasher::GetHash32  (const std::string s)
{
  NS_ASSERT (m_impl != 0);
  return m_impl->GetHash32  (s.c_str (), s.size ());
}

inline
uint64_t
Hasher::GetHash64  (const std::string s)
{
  NS_ASSERT (m_impl != 0);
  return m_impl->GetHash64  (s.c_str (), s.size ());
}


/*************************************************
  Global hash functions, inlined for rvo
*/

/**
  * \brief Get a reference to the static global hasher at g_hasher
  * \return Reference to the static Hasher instance.
  */
Hasher& GetStaticHash (void);

inline
uint32_t
Hash32 (const char * buffer, const std::size_t size)
{
#ifdef NS3_MTP
  return Hasher ().GetHash32 (buffer, size);
#else
  return GetStaticHash ().GetHash32 (buffer, size);
#endif
}

inline
uint64_t
Hash64 (const char * buffer, const std::size_t size)
{
#ifdef NS3_MTP
  return Hasher ().GetHash64 (buffer, size);
#else
  return GetStaticHash ().GetHash64 (buffer, size);
#endif
}

inline
uint32_t
Hash32 (const std::string s)
{
#ifdef NS3_MTP
  return Hasher ().GetHash32 (s);
#else
  return GetStaticHash ().GetHash32 (s);
#endif
}

inline
uint64_t
Hash64 (const std::string s)
{
#ifdef NS3_MTP
  return Hasher ().GetHash64 (s);
#else
  return GetStaticHash ().GetHash64 (s);
#endif
}


}  // namespace ns3

#endif /* HASH_H */
