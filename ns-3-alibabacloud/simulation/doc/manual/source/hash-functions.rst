.. include:: replace.txt
.. highlight:: cpp

Hash Functions
----------------

|ns3| provides a generic interface to general purpose hash functions.
In the simplest usage, the hash function returns the 32-bit or 64-bit
hash of a data buffer or string.  The default underlying hash function
is murmur3_, chosen because it has good hash function properties and
offers a 64-bit version.  The venerable FNV1a_ hash is also available.

There is a straight-forward mechanism to
add (or provide at run time) alternative hash function implementations.

.. _murmur3: http://code.google.com/p/smhasher/wiki/MurmurHash3
.. _FNV1a:   http://isthe.com/chongo/tech/comp/fnv/

Basic Usage
***********

The simplest way to get a hash value of a data buffer or string is just::

  #include "ns3/hash.h"

  using namespace ns3;

  char * buffer = ...
  size_t buffer_size = ...
  
  uint32_t  buffer_hash = Hash32 ( buffer, buffer_size);

  std::string s;
  uint32_t  string_hash = Hash32 (s);

Equivalent functions are defined for 64-bit hash values.

Incremental Hashing
*******************

In some situations it's useful to compute the hash of multiple buffers,
as if they had been joined together.  (For example, you might want
the hash of a packet stream, but not want to assemble a single buffer
with the combined contents of all the packets.)

This is almost as straight-forward as the first example::

  #include "ns3/hash.h"

  using namespace ns3;

  char * buffer;
  size_t buffer_size;

  Hasher hasher;  // Use default hash function

  for (<every buffer>)
    {
	buffer = get_next_buffer ();
	hasher (buffer, buffer_size);
    }
  uint32_t combined_hash = hasher.GetHash32 ();

By default ``Hasher`` preserves internal state to enable incremental
hashing.  If you want to reuse a ``Hasher`` object (for example
because it's configured with a non-default hash function), but don't
want to add to the previously computed hash, you need to ``clear()``
first::

  hasher.clear ().GetHash32 (buffer, buffer_size);

This reinitializes the internal state before hashing the buffer.


Using an Alternative Hash Function
**********************************

The default hash function is murmur3_.  FNV1a_ is also available.  To specify
the hash function explicitly, use this constructor::

  Hasher hasher = Hasher ( Create<Hash::Function::Fnv1a> () );


Adding New Hash Function Implementations
****************************************

To add the hash function ``foo``, follow the ``hash-murmur3.h``/``.cc`` pattern:

 * Create a class declaration (``.h``) and definition (``.cc``) inheriting
   from ``Hash::Implementation``.
 * ``include`` the declaration in ``hash.h`` (at the point where
   ``hash-murmur3.h`` is included.
 * In your own code, instantiate a ``Hasher`` object via the constructor
   ``Hasher (Ptr<Hash::Function::Foo> ())``
   

If your hash function is a single function, e.g. ``hashf``, you don't
even need to create a new class derived from HashImplementation::

  Hasher hasher =
    Hasher ( Create<Hash::Function::Hash32> (&hashf) );

For this to compile, your ``hashf`` has to match one of the function pointer
signatures::

  typedef uint32_t (*Hash32Function_ptr) (const char *, const size_t);
  typedef uint64_t (*Hash64Function_ptr) (const char *, const size_t);


Sources for Hash Functions
**************************

Sources for other hash function implementations include:

 * Peter Kankowski: http://www.strchr.com
 * Arash Partow:    http://www.partow.net/programming/hashfunctions/index.html
 * SMHasher:        http://code.google.com/p/smhasher/
 * Sanmayce:        http://www.sanmayce.com/Fastest_Hash/index.html
