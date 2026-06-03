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

#include <iomanip>
#include <string>

#include "ns3/test.h"
#include "ns3/hash.h"

/**
 * \file
 * \ingroup core-tests
 * \ingroup hash
 * \ingroup hash-tests
 * Hash test suite
 */

/**
 * \ingroup core-tests
 * \ingroup hash
 * \defgroup hash-tests Hash test suite
 */

namespace ns3 {

namespace tests {


/**
 * \ingroup hash-tests
 *  Base class for hash tests
 */
class HashTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param [in] name reference name
   */
  HashTestCase (const std::string name);
  /** Destructor. */
  virtual ~HashTestCase ();

protected:
  /**
   * Check function
   * \param [in] hashName the name of the hash
   * \param [in] hash the hash value
   */
  void Check ( const std::string hashName, const uint32_t hash);
  /**
   * Check function
   * \param [in] hashName the name of the hash
   * \param [in] hash the hash value
   */
  void Check ( const std::string hashName, const uint64_t hash);

  std::string key;           //!< The reference value to hash.
  uint32_t hash32Reference;  //!< The 32-bit hash of the reference.
  uint64_t hash64Reference;  //!< The 64-bit hash of the reference.

private:
  /**
   * Check function
   * \param [in] hashName the name of the hash
   * \param [in] bits the number of bits
   * \param [in] hash the hash value
   */
  void Check ( const std::string hashName, const int bits, const uint64_t hash);
  virtual void DoRun (void);

};  // class HashTestCase

HashTestCase::HashTestCase (const std::string name)
  : TestCase (name),
    key ("The quick brown fox jumped over the lazy dogs.")
{}

HashTestCase::~HashTestCase ()
{}

void
HashTestCase::Check ( const std::string hashName, const uint32_t hash)
{
  Check (hashName, 32, hash);
}

void
HashTestCase::Check ( const std::string hashName, const uint64_t hash)
{
  Check (hashName, 64, hash);
}

void
HashTestCase::Check ( std::string hashName, int bits, uint64_t hash)
{
  int w;
  std::string type;
  uint64_t hashRef;

  if (bits == 32)
    {
      w = 8;
      type = "Hash32";
      hashRef = hash32Reference;
    }
  else
    {
      w = 16;
      type = "Hash64";
      hashRef = hash64Reference;
    }

  std::cout << GetName () << "checking "
            << hashName << " "
            << bits     << "-bit result...";
  NS_TEST_EXPECT_MSG_EQ (hash, hashRef,
                         hashName << " " << type
                                  << " produced "  << std::hex << std::setw (w) << hash
                                  << ", expected " << std::hex << std::setw (w) << hashRef
                                  << std::dec
                         );
  std::cout << std::hex << std::setw (w) << hash << ", ok"
            << std::dec << std::endl;
}

void
HashTestCase::DoRun (void)
{}


/**
 * \ingroup hash-tests
 * Test default hash on fixed string
 */
class DefaultHashTestCase : public HashTestCase
{
public:
  /** Constructor. */
  DefaultHashTestCase ();
  /** Destructor. */
  virtual ~DefaultHashTestCase ();

private:
  virtual void DoRun (void);
};

DefaultHashTestCase::DefaultHashTestCase ()
  : HashTestCase ("DefaultHash: ")
{}

DefaultHashTestCase::~DefaultHashTestCase ()
{}

void
DefaultHashTestCase::DoRun (void)
{
  std::cout << GetName () << "checking with key: \""
            << key << "\"" << std::endl;

  hash32Reference = 0x463d70e2;  // murmur3(key)
  Check ( "default", Hash32 (key));

  hash64Reference = 0xa750412079d53e04ULL;
  Check ( "default", Hash64 (key));
}

/**
 * \ingroup hash-tests
 * FNV hash on fixed string
 */
class Fnv1aTestCase : public HashTestCase
{
public:
  /** Constructor. */
  Fnv1aTestCase ();
  /** Destructor. */
  virtual ~Fnv1aTestCase ();

private:
  virtual void DoRun (void);
};

Fnv1aTestCase::Fnv1aTestCase ()
  : HashTestCase ("Fnv1a: ")
{}

Fnv1aTestCase::~Fnv1aTestCase ()
{}

void
Fnv1aTestCase::DoRun (void)
{
  Hasher hasher = Hasher ( Create<Hash::Function::Fnv1a> () );
  hash32Reference = 0xa3fc0d6d;  // Fnv1a(key)
  Check ("FNV1a", hasher.clear ().GetHash32 (key));

  hash64Reference = 0x88f6cdbe0a31098dULL;
  Check ( "FNV1a", hasher.clear ().GetHash64 (key));
}


/**
 * \ingroup hash-tests
 * Test Murmur3 hash on fixed string
 */
class Murmur3TestCase : public HashTestCase
{
public:
  /** Constructor. */
  Murmur3TestCase ();
  /** Destructor. */
  virtual ~Murmur3TestCase ();

private:
  virtual void DoRun (void);
};

Murmur3TestCase::Murmur3TestCase ()
  : HashTestCase ("Murmur3: ")
{}

Murmur3TestCase::~Murmur3TestCase ()
{}

void
Murmur3TestCase::DoRun (void)
{
  Hasher hasher = Hasher ( Create<Hash::Function::Murmur3> () );
  hash32Reference = 0x463d70e2;  // Murmur3(key)
  Check ( "murmur3", hasher.clear ().GetHash32 (key));

  hash64Reference = 0xa750412079d53e04ULL;
  Check ( "murmur3", hasher.clear ().GetHash64 (key));
}


/**
 * \ingroup hash-tests
 * Simple hash function based on the GNU sum program.
 *
 * 16-bit checksum algorithm.  See
 * http://svnweb.freebsd.org/base/stable/9/usr.bin/cksum/sum1.c?view=markup
 *
 * Used to test Hash32Function_ptr/Hash64Function_ptr
 *
 * \param [in,out] buffer The data to hash.
 * \param [in] size The buffer size.
 * \returns The checksum of the buffer contents.
 */
uint16_t
gnu_sum (const char * buffer, const std::size_t size)
{
  const char * p = buffer;
  const char * const pend = p + size;

  uint16_t checksum = 0;     /* The checksum mod 2^16. */

  while (p != pend)
    {
      checksum = (checksum >> 1) + ((checksum & 1) << 15);  // barrel shift
      checksum += *p++;
    }
  return checksum;
}

/**
 * \ingroup hash-tests
 * A 32-bit hash function, based on gnu_sum().
 * \copydetails gnu_sum()
 */
uint32_t
gnu_sum32 (const char * buffer, const std::size_t size)
{
  uint32_t h = gnu_sum (buffer, size);
  return (uint32_t)( (h << 16) + h);
}

/**
 * \ingroup hash-tests
 * A 64-bit hash function, base on gnu_sum().
 * \copydetails gnu_sum()
 */
uint64_t
gnu_sum64 (const char * buffer, const std::size_t size)
{
  uint64_t h = gnu_sum32 (buffer, size);
  return (uint64_t)( (h << 32) + h);
}

/**
 * \ingroup hash-tests
 * Test 32-bit function pointer
 */
class Hash32FunctionPtrTestCase : public HashTestCase
{
public:
  /** Constructor. */
  Hash32FunctionPtrTestCase ();
  /** Destructor. */
  virtual ~Hash32FunctionPtrTestCase ();

private:
  virtual void DoRun (void);
};

Hash32FunctionPtrTestCase::Hash32FunctionPtrTestCase ()
  : HashTestCase ("Hash32FunctionPtr: ")
{}

Hash32FunctionPtrTestCase::~Hash32FunctionPtrTestCase ()
{}

void
Hash32FunctionPtrTestCase::DoRun (void)
{
  Hasher hasher = Hasher ( Create<Hash::Function::Hash32> (&gnu_sum32) );
  hash32Reference = 0x41264126;  // Hash32FunctionPtr(key)
  Check ( "gnu_sum32", hasher.clear ().GetHash32 (key));
}

/**
 * \ingroup hash-tests
 * Test 64-bit function pointer
 */
class Hash64FunctionPtrTestCase : public HashTestCase
{
public:
  /** Constructor. */
  Hash64FunctionPtrTestCase ();
  /** Destructor. */
  virtual ~Hash64FunctionPtrTestCase ();

private:
  virtual void DoRun (void);
};

Hash64FunctionPtrTestCase::Hash64FunctionPtrTestCase ()
  : HashTestCase ("Hash64FunctionPtr: ")
{}

Hash64FunctionPtrTestCase::~Hash64FunctionPtrTestCase ()
{}

void
Hash64FunctionPtrTestCase::DoRun (void)
{
  Hasher hasher = Hasher ( Create<Hash::Function::Hash64> (&gnu_sum64) );
  hash64Reference = 0x4126412641264126ULL;  // Hash64FunctionPtr(key)
  Check ( "gnu_sum64", hasher.clear ().GetHash64 (key));
}

/**
 * \ingroup hash-tests
 * Test incremental hashing
 */
class IncrementalTestCase : public HashTestCase
{
public:
  /** Constructor. */
  IncrementalTestCase ();
  /** Destructor. */
  virtual ~IncrementalTestCase ();

private:
  virtual void DoRun (void);
  /**
   * Complute the hash test function
   * \param name the hash name
   * \param hasher the hash function
   */
  void DoHash (const std::string name, Hasher hasher);
  std::string key1;  //!< test string
  std::string key2;  //!< test string
  std::string key12; //!< test string
};

IncrementalTestCase::IncrementalTestCase ()
  : HashTestCase ("Incremental: ")
{}

IncrementalTestCase::~IncrementalTestCase ()
{}

void
IncrementalTestCase::DoHash (const std::string name, Hasher hasher)
{
  hash32Reference = hasher.clear ().GetHash32 (key12);
  hasher.clear ().GetHash32 (key1);
  Check ( name, hasher.GetHash32 (key2));

  hash64Reference = hasher.clear ().GetHash64 (key12);
  hasher.clear ().GetHash64 (key1);
  Check ( name, hasher.GetHash64 (key2));
}

void
IncrementalTestCase::DoRun (void)
{
  key1 = "The quick brown ";
  key2 = "Incremental.";
  key12  = key1 + key2;

  std::cout << GetName () << "checking with key: "
            << "\"" << key1  << "\"[" << key1.size ()  << "] + "
            << "\"" << key2  << "\"[" << key2.size ()  << "]" << std::endl;
  std::cout << GetName () << "equivalent to:     "
            << "\"" << key12 << "\"[" << key12.size () << "]" << std::endl;

  DoHash ( "default", Hasher ( ) );
  DoHash ( "murmur3", Hasher ( Create<Hash::Function::Murmur3> () ) );
  DoHash ( "FNV1a",   Hasher ( Create<Hash::Function::Fnv1a> () ) );
}


/**
 * \ingroup hash-tests
 * Hash functions test suite
 */
class HashTestSuite : public TestSuite
{
public:
  /** Constructor. */
  HashTestSuite ();
};

HashTestSuite::HashTestSuite ()
  : TestSuite ("hash")
{
  AddTestCase (new DefaultHashTestCase);
  AddTestCase (new Murmur3TestCase);
  AddTestCase (new Fnv1aTestCase);
  AddTestCase (new IncrementalTestCase);
  AddTestCase (new Hash32FunctionPtrTestCase);
  AddTestCase (new Hash64FunctionPtrTestCase);
}

/**
 * \ingroup hash-tests
 * HashTestSuite instance variable.
 */
static HashTestSuite g_hashTestSuite;


}    // namespace tests

}  // namespace ns3
