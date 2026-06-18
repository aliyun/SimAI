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
 *
 * This copyright notice applies strictly to the wrapper material.
 *
 * The murmur3 source code itself is in the public domain.  The murmur3 source
 * code sections are marked by
 *   // Begin <murmur3-file> ---->
 * and
 *   // End <murmur3-file>   ---->
 * comments.
 *
 * Changes from the murmur3 distribution are marked with `//PDB'
 * In addition comment blocks have been converted to Doxygen format.
 * Function arguments for buffer length which were originally
 * "int len" or "int i" have been changed to "std::size_t".
 * In the _x86 versions the main loop used negative indexes, as shown.
 * Other conversions to std::size_t are marked.
 */

#include "log.h"
#include "hash-murmur3.h"

#include <iomanip>

/**
 * \file
 * \ingroup hash
 * \brief ns3::Hash::Function::Murmur3 implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Hash-Murmur3");

namespace Hash {

namespace Function {

/** Murmur3 hash implementation details. */
namespace Murmur3Implementation {

/**
 * \ingroup hash
 * \defgroup hash_murmur3 Murmur3 Hash Implementation
 */
/**@{*/


// Changes from Murmur3 distribution are marked with `//PDB'
//

/*************************************************
 **  class Murmur3HashImplementation
 ************************************************/

// Adapted from http://code.google.com/p/smhasher/

// Begin Murmur3.cpp -------- *NS_CHECK_STYLE_OFF* ---->

//
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.


/**
 * Barrel shift (rotate) left on 32 bits.
 *
 * \param [in] x The initial value.
 * \param [in] r The number of bit positions to rotate.
 * \return The rotated value.
 */
inline uint32_t rotl32 ( uint32_t x, int8_t r )
{
  return (x << r) | (x >> (32 - r));
}

/**
 * Barrel shift (rotate) left on 64 bits.
 *
 * \param [in] x The initial value.
 * \param [in] r The number of bit positions to rotate.
 * \return The rotated value.
 */
inline uint64_t rotl64 ( uint64_t x, int8_t r )
{
  return (x << r) | (x >> (64 - r));
}

/** Unsigned long long constants. */
#define BIG_CONSTANT(x) (x##LLU)

//-----------------------------------------------------------------------------
/**
 * Block read
 *
 * If your platform needs to do endian-swapping or can only
 * handle aligned reads, do the conversion here.
 *
 * \param [in] p Block base address.
 * \param [in] i Index into the block.
 * \returns The \c i'th word from the block.
 */
inline uint32_t getblock ( const uint32_t * p, std::size_t i )
{
  return p[i];
}
/** \copydoc getblock(const uint32_t*,std::size_t) */
inline uint64_t getblock ( const uint64_t * p, std::size_t i )
{
  return p[i];
}

//-----------------------------------------------------------------------------
/**
 * Finalization mix - force all bits of a hash block to avalanche.
 *
 * \param [in] h Final word of the hash block.
 * \returns Fully mixed final word.
 */
inline uint32_t fmix ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//----------
/** \copydoc fmix(uint32_t) */
inline uint64_t fmix ( uint64_t h )
{
  h ^= h >> 33;
  h *= BIG_CONSTANT(0xff51afd7ed558ccd);
  h ^= h >> 33;
  h *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  h ^= h >> 33;

  return h;
}

//-----------------------------------------------------------------------------

//PDB forward
/**
 * Initial and incremental hash.
 *
 * \param [in] key Data to be hashed.
 * \param [in] len Number of words in the \c key.
 * \param [in] seed Initial or current hash state.
 * \param [out] out Output hash value.
 */
void MurmurHash3_x86_32_incr ( const void * key, std::size_t len,
                               uint32_t seed, void * out );
/**
 * Finalize a hash.
 *
 * \param [in] len Total number of words that have gone in to the hash.
 * \param [in] seed Initial or current hash state.
 * \param [out] out Output hash value.
 */
void MurmurHash3_x86_32_fin ( std::size_t len,
                              uint32_t seed, void * out );

//PDB - incremental hashing
/** \copydoc MurmurHash3_x86_32_incr() */
void MurmurHash3_x86_32 ( const void * key, std::size_t len,
                          uint32_t seed, void * out )
{
  uint32_t h1;
  MurmurHash3_x86_32_incr (key, len, seed, &h1);
  MurmurHash3_x86_32_fin (len, h1, out);
}

void MurmurHash3_x86_32_incr ( const void * key, std::size_t len,
                               uint32_t seed, void * out )
{
  const uint8_t * data = (const uint8_t*)key;
  const std::size_t nblocks = len / 4;  //PDB: was const int nblocks

  uint32_t h1 = seed;

  uint32_t c1 = 0xcc9e2d51;
  uint32_t c2 = 0x1b873593;

  //----------
  // body

  //PDB: const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);
  const uint32_t * blocks = (const uint32_t *)(data);

  //PDB: for(int i = -nblocks; i; i++)
  for(std::size_t i = 0; i < nblocks; i++)
  {
    uint32_t k1 = getblock(blocks,i);

    k1 *= c1;
    k1 = rotl32(k1,15);
    k1 *= c2;

    h1 ^= k1;
    h1 = rotl32(h1,13); 
    h1 = h1*5+0xe6546b64;
  }

  //----------
  // tail

  const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

  uint32_t k1 = 0;

  switch(len & 3)
  {
  case 3: k1 ^= tail[2] << 16;
  case 2: k1 ^= tail[1] << 8;
  case 1: k1 ^= tail[0];
          k1 *= c1; k1 = rotl32(k1,15); k1 *= c2; h1 ^= k1;
  };

  *(uint32_t *)out = h1;
}

//PDB - incremental hashing - finalization
void MurmurHash3_x86_32_fin ( std::size_t len,
                              uint32_t seed, void * out )
{
  uint32_t h1 = seed;

  //----------
  // finalization

  h1 ^= len;

  h1 = fmix(h1);

  *(uint32_t *)out = h1;
} 

//-----------------------------------------------------------------------------

//PDB forward
/**
 * Initial and incremental hash.
 *
 * \param [in] key Data to be hashed.
 * \param [in] len Number of words in the \c key.
 * \param [in] seeds Initial or current hash state.
 * \param [out] out Output hash value.
 */
void MurmurHash3_x86_128_incr ( const void * key, const std::size_t len,
                                uint32_t * seeds, void * out );
/**
 * Finalize a hash.
 *
 * \param [in] len Total number of words that have gone in to the hash.
 * \param [in] seeds Initial or current hash state.
 * \param [out] out Output hash value.
 */
void MurmurHash3_x86_128_fin ( const std::size_t len,
                               uint32_t * seeds, void * out );

//PDB - incremental hashing
/**
 * Initial and incremental hash.
 *
 * \param [in] key Data to be hashed.
 * \param [in] len Number of words in the \c key.
 * \param [in] seed Initial or current hash state.
 * \param [out] out Output hash value.
 */
void MurmurHash3_x86_128 ( const void * key, const std::size_t len,
                           uint32_t seed, void * out )
{
  uint32_t seeds[4];
  uint32_t h[4];
  seeds[0] = seeds[1] = seeds[2] = seeds[3] = seed;
  MurmurHash3_x86_128_incr (key, len, seeds, h);
  MurmurHash3_x86_128_fin (len, h, out);
}

void MurmurHash3_x86_128_incr ( const void * key, const std::size_t len,
                                uint32_t * seeds, void * out )
{
  const uint8_t * data = (const uint8_t*)key;
  const std::size_t nblocks = len / 16;  //PDB: was const int nblocks

  uint32_t h1 = seeds[0];
  uint32_t h2 = seeds[1];
  uint32_t h3 = seeds[2];
  uint32_t h4 = seeds[3];

  uint32_t c1 = 0x239b961b; 
  uint32_t c2 = 0xab0e9789;
  uint32_t c3 = 0x38b34ae5; 
  uint32_t c4 = 0xa1e38b93;

  //----------
  // body

  //PDB: const uint32_t * blocks = (const uint32_t *)(data + nblocks*16);
  const uint32_t * blocks = (const uint32_t *)(data);

  //PDB: for(int i = -nblocks; i; i++)
  for(std::size_t i = 0; i < nblocks; i++)
  {
    uint32_t k1 = getblock(blocks,i*4+0);
    uint32_t k2 = getblock(blocks,i*4+1);
    uint32_t k3 = getblock(blocks,i*4+2);
    uint32_t k4 = getblock(blocks,i*4+3);

    k1 *= c1; k1  = rotl32(k1,15); k1 *= c2; h1 ^= k1;

    h1 = rotl32(h1,19); h1 += h2; h1 = h1*5+0x561ccd1b;

    k2 *= c2; k2  = rotl32(k2,16); k2 *= c3; h2 ^= k2;

    h2 = rotl32(h2,17); h2 += h3; h2 = h2*5+0x0bcaa747;

    k3 *= c3; k3  = rotl32(k3,17); k3 *= c4; h3 ^= k3;

    h3 = rotl32(h3,15); h3 += h4; h3 = h3*5+0x96cd1c35;

    k4 *= c4; k4  = rotl32(k4,18); k4 *= c1; h4 ^= k4;

    h4 = rotl32(h4,13); h4 += h1; h4 = h4*5+0x32ac3b17;
  }

  //----------
  // tail

  const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

  uint32_t k1 = 0;
  uint32_t k2 = 0;
  uint32_t k3 = 0;
  uint32_t k4 = 0;

  switch(len & 15)
  {
  case 15: k4 ^= tail[14] << 16;
  case 14: k4 ^= tail[13] << 8;
  case 13: k4 ^= tail[12] << 0;
           k4 *= c4; k4  = rotl32(k4,18); k4 *= c1; h4 ^= k4;

  case 12: k3 ^= tail[11] << 24;
  case 11: k3 ^= tail[10] << 16;
  case 10: k3 ^= tail[ 9] << 8;
  case  9: k3 ^= tail[ 8] << 0;
           k3 *= c3; k3  = rotl32(k3,17); k3 *= c4; h3 ^= k3;

  case  8: k2 ^= tail[ 7] << 24;
  case  7: k2 ^= tail[ 6] << 16;
  case  6: k2 ^= tail[ 5] << 8;
  case  5: k2 ^= tail[ 4] << 0;
           k2 *= c2; k2  = rotl32(k2,16); k2 *= c3; h2 ^= k2;

  case  4: k1 ^= tail[ 3] << 24;
  case  3: k1 ^= tail[ 2] << 16;
  case  2: k1 ^= tail[ 1] << 8;
  case  1: k1 ^= tail[ 0] << 0;
           k1 *= c1; k1  = rotl32(k1,15); k1 *= c2; h1 ^= k1;
  };

  ((uint32_t *)out)[0] = h1;
  ((uint32_t *)out)[1] = h2;
  ((uint32_t *)out)[2] = h3;
  ((uint32_t *)out)[3] = h4;
}

//PDB - incremental hashing - finalization
void MurmurHash3_x86_128_fin ( const std::size_t len,
                               uint32_t * seeds, void * out )
{
  //----------
  // finalization

  uint32_t h1 = seeds[0];
  uint32_t h2 = seeds[1];
  uint32_t h3 = seeds[2];
  uint32_t h4 = seeds[3];

  h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;

  h1 += h2; h1 += h3; h1 += h4;
  h2 += h1; h3 += h1; h4 += h1;

  h1 = fmix(h1);
  h2 = fmix(h2);
  h3 = fmix(h3);
  h4 = fmix(h4);

  h1 += h2; h1 += h3; h1 += h4;
  h2 += h1; h3 += h1; h4 += h1;

  ((uint32_t *)out)[0] = h1;
  ((uint32_t *)out)[1] = h2;
  ((uint32_t *)out)[2] = h3;
  ((uint32_t *)out)[3] = h4;
}

//-----------------------------------------------------------------------------
/** \copydoc MurmurHash3_x86_32() */
void MurmurHash3_x64_128 ( const void * key, const std::size_t len,
                           const uint32_t seed, void * out )
{
  const uint8_t * data = (const uint8_t*)key;
  const std::size_t nblocks = len / 16;  //PDB: was const int nblocks

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  //----------
  // body

  const uint64_t * blocks = (const uint64_t *)(data);

  for(std::size_t i = 0; i < nblocks; i++)  //PDB: was int i
  {
    uint64_t k1 = getblock(blocks,i*2+0);
    uint64_t k2 = getblock(blocks,i*2+1);

    k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;

    h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

    k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

    h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
  }

  //----------
  // tail

  const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch(len & 15)
  {
  case 15: k2 ^= uint64_t(tail[14]) << 48;
  case 14: k2 ^= uint64_t(tail[13]) << 40;
  case 13: k2 ^= uint64_t(tail[12]) << 32;
  case 12: k2 ^= uint64_t(tail[11]) << 24;
  case 11: k2 ^= uint64_t(tail[10]) << 16;
  case 10: k2 ^= uint64_t(tail[ 9]) << 8;
  case  9: k2 ^= uint64_t(tail[ 8]) << 0;
           k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

  case  8: k1 ^= uint64_t(tail[ 7]) << 56;
  case  7: k1 ^= uint64_t(tail[ 6]) << 48;
  case  6: k1 ^= uint64_t(tail[ 5]) << 40;
  case  5: k1 ^= uint64_t(tail[ 4]) << 32;
  case  4: k1 ^= uint64_t(tail[ 3]) << 24;
  case  3: k1 ^= uint64_t(tail[ 2]) << 16;
  case  2: k1 ^= uint64_t(tail[ 1]) << 8;
  case  1: k1 ^= uint64_t(tail[ 0]) << 0;
           k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
  };

  //----------
  // finalization

  h1 ^= len; h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix(h1);
  h2 = fmix(h2);

  h1 += h2;
  h2 += h1;

  ((uint32_t *)out)[0] = static_cast<uint32_t> (h1);  //PDB cast
  ((uint32_t *)out)[1] = static_cast<uint32_t> (h2);  //PDB cast
}


// End Murmur3.cpp ---------- *NS_CHECK_STYLE_ON* ----->

#undef BIG_CONSTANT


//-----------------------------------------------------------------------------


/**@}*/  // \defgroup hash_murmur3

}  // namespace Murmur3Implementation


Murmur3::Murmur3 ()
{
  clear ();
}

uint32_t
Murmur3::GetHash32  (const char * buffer, const std::size_t size)
{
  using namespace Murmur3Implementation;

  MurmurHash3_x86_32_incr (buffer, size,
                           m_hash32, (void *) &m_hash32);
  m_size32 += static_cast<uint32_t> (size);
  uint32_t hash;
  MurmurHash3_x86_32_fin  (m_size32, m_hash32, (void *) &hash);

  return hash;
}

uint64_t
Murmur3::GetHash64  (const char * buffer, const std::size_t size)
{
  using namespace Murmur3Implementation;

  MurmurHash3_x86_128_incr (buffer, static_cast<int> (size),
                            (uint32_t *)(void *)m_hash64, m_hash64);
  m_size64 += size;

  // Simpler would be:
  //
  //   uint64_t hash[2];
  //   MurmurHash3_x86_128_fin (m_size64, m_hash64, hash);
  //   return hash[0];
  //
  // but this triggers an aliasing bug in gcc-4.4 (perhaps related to
  // http://gcc.gnu.org/bugzilla/show_bug.cgi?id=39390).
  // In ns-3, this bug produces incorrect results in static optimized
  // builds only.
  //
  // Using uint32_t here avoids the bug, and continues to works with newer gcc.
  uint32_t hash[4];

  MurmurHash3_x86_128_fin (static_cast<int> (m_size64),
                           (uint32_t *)(void *)m_hash64, hash);
  uint64_t result = hash[1];
  result = (result << 32) | hash[0];
  return result;
}

void
Murmur3::clear (void)
{
  m_hash32 = (uint32_t)SEED;
  m_size32 = 0;
  m_hash64[0] = m_hash64[1] = ((uint64_t)SEED << 32) | (uint32_t)SEED;
  m_size64 = 0;
}

}  // namespace Function

}  // namespace Hash

}  // namespace ns3
