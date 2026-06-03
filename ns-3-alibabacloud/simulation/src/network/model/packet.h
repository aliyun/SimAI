/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
#ifndef PACKET_H
#define PACKET_H

#include <stdint.h>
#include "buffer.h"
#include "header.h"
#include "trailer.h"
#include "packet-metadata.h"
#include "tag.h"
#include "byte-tag-list.h"
#include "packet-tag-list.h"
#include "nix-vector.h"
#include "ns3/mac48-address.h"
#include "ns3/callback.h"
#include "ns3/assert.h"
#include "ns3/ptr.h"
#include "ns3/deprecated.h"

namespace ns3 {

// Forward declaration
class Address;
  
/**
 * \ingroup network
 * \defgroup packet Packet
 */

/**
 * \ingroup packet
 * \brief Iterator over the set of byte tags in a packet
 *
 * This is a java-style iterator.
 */
class ByteTagIterator
{
public:
  /**
   * \brief Identifies a byte tag and a set of bytes within a packet
   * to which the tag applies.
   */
  class Item
  {
public:
    /**
     * \returns the ns3::TypeId associated to this tag.
     */
    TypeId GetTypeId (void) const;
    /**
     * \returns the index of the first byte tagged by this tag.
     *
     * \brief The index is an offset from the start of the packet.
     */
    uint32_t GetStart (void) const;
    /**
     * \returns the index of the last byte tagged by this tag.
     *
     * \brief The index is an offset from the start of the packet.
     */
    uint32_t GetEnd (void) const;
    /**
     * \brief Read the requested tag and store it in the user-provided tag instance.
     *
     * \param tag the user tag to which the data should be copied.
     *
     * This method will crash if the type of the tag provided
     * by the user does not match the type of the underlying tag.
     */
    void GetTag (Tag &tag) const;
private:
    /// Friend class
    friend class ByteTagIterator;
    /**
     * \brief Constructor
     * \param tid the ns3::TypeId associated to this tag.
     * \param start the index of the first byte tagged by this tag.
     * \param end the index of the last byte tagged by this tag.
     * \param buffer the buffer associated with this tag.
     */
    Item (TypeId tid, uint32_t start, uint32_t end, TagBuffer buffer);

    TypeId m_tid;       //!< the ns3::TypeId associated to this tag.
    uint32_t m_start;   //!< the index of the first byte tagged by this tag.
    uint32_t m_end;     //!< the index of the last byte tagged by this tag.
    TagBuffer m_buffer; //!< the buffer associated with this tag.
  };
  /**
   * \returns true if calling Next is safe, false otherwise.
   */
  bool HasNext (void) const;
  /**
   * \returns the next item found and prepare for the next one.
   */
  Item Next (void);
private:
  /// Friend class
  friend class Packet;
  /**
   * Copy Constructor
   * \param i object to copy
   */
  ByteTagIterator (ByteTagList::Iterator i);
  ByteTagList::Iterator m_current; //!< actual position over the set of byte tags in a packet
};

/**
 * \ingroup packet
 * \brief Iterator over the set of packet tags in a packet
 *
 * This is a java-style iterator.
 */
class PacketTagIterator
{
public:
  /**
   * Identifies a packet tag within a packet.
   */
  class Item 
  {
public:
    /**
     * \returns the ns3::TypeId associated to this tag.
     */
    TypeId GetTypeId (void) const;
    /**
     * Read the requested tag and store it in the user-provided tag instance.
     *
     * \param tag the user tag to which the data should be copied.
     *
     * This method will crash if the type of the tag provided
     * by the user does not match the type of the underlying tag.
     */
    void GetTag (Tag &tag) const;
private:
    /// Friend class
    friend class PacketTagIterator;
    /**
     * Constructor
     * \param data the data to copy.
     */
    Item (const struct PacketTagList::TagData *data);
    const struct PacketTagList::TagData *m_data; //!< the tag data
  };
  /**
   * \returns true if calling Next is safe, false otherwise.
   */
  bool HasNext (void) const;
  /**
   * \returns the next item found and prepare for the next one.
   */
  Item Next (void);
private:
  /// Friend class
  friend class Packet;
  /**
   * Constructor
   * \param head head of the items
   */
  PacketTagIterator (const struct PacketTagList::TagData *head);
  const struct PacketTagList::TagData *m_current;  //!< actual position over the set of tags in a packet
};

/**
 * \ingroup packet
 * \brief network packets
 *
 * Each network packet contains a byte buffer, a set of byte tags, a set of
 * packet tags, and metadata.
 *
 * - The byte buffer stores the serialized content of the headers and trailers 
 * added to a packet. The serialized representation of these headers is expected
 * to match that of real network packets bit for bit (although nothing
 * forces you to do this) which means that the content of a packet buffer
 * is expected to be that of a real packet.
 *
 * - The metadata describes the type of the headers and trailers which
 * were serialized in the byte buffer. The maintenance of metadata is
 * optional and disabled by default. To enable it, you must call
 * Packet::EnablePrinting and this will allow you to get non-empty
 * output from Packet::Print. If you wish to only enable
 * checking of metadata, and do not need any printing capability, you can
 * call Packet::EnableChecking: its runtime cost is lower than
 * Packet::EnablePrinting.
 *
 * - The set of tags contain simulation-specific information which cannot
 * be stored in the packet byte buffer because the protocol headers or trailers
 * have no standard-conformant field for this information. So-called
 * 'byte' tags are used to tag a subset of the bytes in the packet byte buffer
 * while 'packet' tags are used to tag the packet itself. The main difference
 * between these two kinds of tags is what happens when packets are copied,
 * fragmented, and reassembled: 'byte' tags follow bytes while 'packet' tags
 * follow packets. Another important difference between these two kinds of tags
 * is that byte tags cannot be removed and are expected to be written once,
 * and read many times, while packet tags are expected to be written once,
 * read many times, and removed exactly once. An example of a 'byte' 
 * tag is a FlowIdTag which contains a flow id and is set by the application
 * generating traffic. An example of a 'packet' tag is a cross-layer 
 * qos class id set by an application and processed by a lower-level MAC 
 * layer.
 *
 * Implementing a new type of Header or Trailer for a new protocol is 
 * pretty easy and is a matter of creating a subclass of the ns3::Header 
 * or of the ns3::Trailer base class, and implementing the methods
 * described in their respective API documentation.
 *
 * Implementing a new type of Tag requires roughly the same amount of
 * work and this work is described in the ns3::Tag API documentation.
 *
 * The performance aspects copy-on-write semantics of the
 * Packet API are discussed in \ref packetperf
 */
class Packet : public SimpleRefCount<Packet>
{
public:

  /**
   * \brief Create an empty packet with a new uid (as returned
   * by getUid).
   */
  Packet ();
  /**
   * \brief Copy constructor
   * \param o object to copy
   */
  Packet (const Packet &o);
  /**
   * \brief Basic assignment
   * \param o object to copy
   * \return the copied object
   */
  Packet &operator = (const Packet &o);
  /**
   * \brief Create a packet with a zero-filled payload.
   *
   * The memory necessary for the payload is not allocated:
   * it will be allocated at any later point if you attempt
   * to fragment this packet or to access the zero-filled
   * bytes. The packet is allocated with a new uid (as 
   * returned by getUid).
   * 
   * \param size the size of the zero-filled payload
   */
  Packet (uint32_t size);
  /**
   * \brief Create a new packet from the serialized buffer.
   *
   * This new packet
   * is identical to the serialized packet contained in the buffer 
   * and is magically deserialized for you
   * 
   * \param buffer the serialized packet to be created
   * \param size the size of the packet for deserialization
   * \param magic allows packet deserialization; 
   *        asserts when set to false
   */
  Packet (uint8_t const*buffer, uint32_t size, bool magic);
  /**
   * \brief Create a packet with payload filled with the content
   * of this buffer.
   *
   * The input data is copied: the input
   * buffer is untouched.
   *
   * \param buffer the data to store in the packet.
   * \param size the size of the input buffer.
   */
  Packet (uint8_t const*buffer, uint32_t size);
  /**
   * \brief Create a new packet which contains a fragment of the original
   * packet.
   *
   * The returned packet shares the same uid as this packet.
   *
   * \param start offset from start of packet to start of fragment to create
   * \param length length of fragment to create
   * \returns a fragment of the original packet
   */
  Ptr<Packet> CreateFragment (uint32_t start, uint32_t length) const;
  /**
   * \brief Returns the the size in bytes of the packet (including the zero-filled
   * initial payload).
   *
   * \returns the size in bytes of the packet
   */
  inline uint32_t GetSize (void) const;
  /**
   * \brief Add header to this packet.
   *
   * This method invokes the
   * Header::GetSerializedSize and Header::Serialize
   * methods to reserve space in the buffer and request the 
   * header to serialize itself in the packet buffer.
   *
   * \param header a reference to the header to add to this packet.
   */
  void AddHeader (const Header & header);
  /**
   * \brief Deserialize and remove the header from the internal buffer.
   *
   * This method invokes Header::Deserialize (begin) and should be used for
   * fixed-length headers.
   *
   * \param header a reference to the header to remove from the internal buffer.
   * \returns the number of bytes removed from the packet.
   */
  uint32_t RemoveHeader (Header &header);
  /**
   * \brief Deserialize and remove the header from the internal buffer.
   *
   * This method invokes Header::Deserialize (begin, end) and should be 
   * used for variable-length headers (where the size is determined somehow
   * by the caller).
   *
   * \param header a reference to the header to remove from the internal buffer.
   * \param size number of bytes to deserialize
   * \returns the number of bytes removed from the packet.
   */
  uint32_t RemoveHeader (Header &header, uint32_t size);
  /**
   * \brief Deserialize but does _not_ remove the header from the internal buffer.
   * s
   * This method invokes Header::Deserialize.
   *
   * \param header a reference to the header to read from the internal buffer.
   * \returns the number of bytes read from the packet.
   */
  uint32_t PeekHeader (Header &header) const;
  /**
   * \brief Deserialize but does _not_ remove the header from the internal buffer.
   * s
   * This method invokes Header::Deserialize (begin, end) and should be used
   * for variable-length headers (where the size is determined somehow
   * by the caller).
   *
   * \param header a reference to the header to read from the internal buffer.
   * \param size number of bytes to deserialize
   * \returns the number of bytes read from the packet.
   */
  uint32_t PeekHeader (Header &header, uint32_t size) const;
  /**
   * \brief Add trailer to this packet.
   *
   * This method invokes the
   * Trailer::GetSerializedSize and Trailer::Serialize
   * methods to reserve space in the buffer and request the trailer 
   * to serialize itself in the packet buffer.
   *
   * \param trailer a reference to the trailer to add to this packet.
   */
  void AddTrailer (const Trailer &trailer);
  /**
   * \brief Remove a deserialized trailer from the internal buffer.
   *
   * This method invokes the Deserialize method.
   *
   * \param trailer a reference to the trailer to remove from the internal buffer.
   * \returns the number of bytes removed from the end of the packet.
   */
  uint32_t RemoveTrailer (Trailer &trailer);
  /**
   * \brief Deserialize but does _not_ remove a trailer from the internal buffer.
   *
   * This method invokes the Trailer::Deserialize method.
   *
   * \param trailer a reference to the trailer to read from the internal buffer.
   * \returns the number of bytes read from the end of the packet.
   */
  uint32_t PeekTrailer (Trailer &trailer);

  /**
   * \brief Concatenate the input packet at the end of the current
   * packet.
   *
   * This does not alter the uid of either packet.
   *
   * \param packet packet to concatenate
   */
  void AddAtEnd (Ptr<const Packet> packet);
  /**
   * \brief Add a zero-filled padding to the packet.
   *
   * \param size number of padding bytes to add.
   */
  void AddPaddingAtEnd (uint32_t size);
  /** 
   * \brief Remove size bytes from the end of the current packet.
   *
   * It is safe to remove more bytes than are present in
   * the packet.
   *
   * \param size number of bytes from remove
   */
  void RemoveAtEnd (uint32_t size);
  /** 
   * \brief Remove size bytes from the start of the current packet.
   *
   * It is safe to remove more bytes than are present in
   * the packet.
   *
   * \param size number of bytes from remove
   */
  void RemoveAtStart (uint32_t size);

  /**
   * \brief Copy the packet contents to a byte buffer.
   *
   * \param buffer a pointer to a byte buffer where the packet data 
   *        should be copied.
   * \param size the size of the byte buffer. 
   * \returns the number of bytes read from the packet
   *
   * No more than \b size bytes will be copied by this function.
   */
  uint32_t CopyData (uint8_t *buffer, uint32_t size) const;

  /**
   * \brief Copy the packet contents to an output stream.
   *
   * \param os pointer to output stream in which we want
   *        to write the packet data.
   * \param size the maximum number of bytes we want to write
   *        in the output stream.
   */
  void CopyData (std::ostream *os, uint32_t size) const;

  /**
   * \brief performs a COW copy of the packet.
   *
   * \returns a COW copy of the packet.
   *
   * The returns packet will behave like an independent copy of
   * the original packet, even though they both share the
   * same datasets internally.
   */
  Ptr<Packet> Copy (void) const;

  /**
   * \brief Returns the packet's Uid.
   *
   * A packet is allocated a new uid when it is created
   * empty or with zero-filled payload.
   *
   * Note: This uid is an internal uid and cannot be counted on to
   * provide an accurate counter of how many "simulated packets" of a
   * particular protocol are in the system. It is not trivial to make
   * this uid into such a counter, because of questions such as what
   * should the uid be when the packet is sent over broadcast media, or
   * when fragmentation occurs. If a user wants to trace actual packet
   * counts, he or she should look at e.g. the IP ID field or transport
   * sequence numbers, or other packet or frame counters at other
   * protocol layers.
   *
   * \returns an integer identifier which uniquely
   *          identifies this packet.
   */
  uint64_t GetUid (void) const;

  /**
   * \brief Print the packet contents.
   *
   * \param os output stream in which the data should be printed.
   *
   * Iterate over the headers and trailers present in this packet, 
   * from the first header to the last trailer and invoke, for
   * each of them, the user-provided method Header::DoPrint or 
   * Trailer::DoPrint methods.
   */
  void Print (std::ostream &os) const;

  /**
   * \brief Return a string representation of the packet
   *
   * An empty string is returned if you haven't called EnablePrinting ()
   *
   * \return String representation
   */
  std::string ToString (void) const;

  /**
   * \brief Returns an iterator which points to the first 'item'
   * stored in this buffer.
   *
   * Note that this iterator will point
   * to an empty array of items if you don't call EnablePrinting
   * or EnableChecking before.
   *
   * \returns an iterator
   *
   * \sa EnablePrinting EnableChecking
   */
  PacketMetadata::ItemIterator BeginItem (void) const;

  /**
   * \brief Enable printing packets metadata.
   *
   * By default, packets do not keep around enough metadata to
   * perform the operations requested by the Print methods. If you
   * want to be able the Packet::Print method, 
   * you need to invoke this method at least once during the 
   * simulation setup and before any packet is created.
   */
  static void EnablePrinting (void);
  /**
   * \brief Enable packets metadata checking.
   *
   * The packet metadata is also used to perform extensive
   * sanity checks at runtime when performing operations on a 
   * Packet. For example, this metadata is used to verify that
   * when you remove a header from a packet, this same header
   * was actually present at the front of the packet. These
   * errors will be detected and will abort the program.
   */
  static void EnableChecking (void);

  /**
   * \brief Returns number of bytes required for packet
   * serialization.
   *
   * \returns number of bytes required for packet
   * serialization
   *
   * For packet serialization, the total size is checked
   * in order to determine the size of the buffer 
   * required for serialization
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serialize a packet, tags, and metadata into a byte buffer.
   *
   * \param buffer a raw byte buffer to which the packet will be serialized
   * \param maxSize the max size of the buffer for bounds checking
   *
   * \returns one if all data were serialized, zero if buffer size was too small.
   */
  uint32_t Serialize (uint8_t* buffer, uint32_t maxSize) const;

  /**
   * \brief Tag each byte included in this packet with a new byte tag.
   *
   * \param tag the new tag to add to this packet
   *
   * Note that adding a tag is a const operation which is pretty 
   * un-intuitive. The rationale is that the content and behavior of
   * a packet is _not_ changed when a tag is added to a packet: any
   * code which was not aware of the new tag is going to work just
   * the same if the new tag is added. The real reason why adding a
   * tag was made a const operation is to allow a trace sink which gets
   * a packet to tag the packet, even if the packet is const (and most
   * trace sources should use const packets because it would be
   * totally evil to allow a trace sink to modify the content of a
   * packet).
   */
  void AddByteTag (const Tag &tag) const;

  /**
   * \brief Tag the indicated byte range of this packet with a new byte tag.
   *
   * As parameters for this method, we do not use indexes, but byte position.
   * Moreover, as there is no 0-th position, the first position is 1.
   *
   * As example, if you want to tag the first 10 bytes, you have to call
   * the method in this way:
   *
   * \code{.cpp}
       Ptr<Packet> p = ... ;
       SomeTag tag;
       p->AddByteTag (tag, 1, 10);
     \endcode
   *
   * \param tag the new tag to add to this packet
   * \param start the position of the first byte tagged by this tag
   * \param end the position of the last byte tagged by this tag
   */
  void AddByteTag (const Tag &tag, uint32_t start, uint32_t end) const;
  /**
   * \brief Returns an iterator over the set of byte tags included in this packet
   *
   * \returns an iterator over the set of byte tags included in this packet.
   */
  ByteTagIterator GetByteTagIterator (void) const;
  /**
   * \brief Finds the first tag matching the parameter Tag type
   *
   * \param tag the byte tag type to search in this packet
   * \returns true if the requested tag type was found, false otherwise.
   *
   * If the requested tag type is found, it is copied in the user's 
   * provided tag instance.
   */
  bool FindFirstMatchingByteTag (Tag &tag) const;

  /**
   * \brief Remove all byte tags stored in this packet.
   */
  void RemoveAllByteTags (void);

  /**
   * \param os output stream in which the data should be printed.
   *
   * \brief Iterate over the byte tags present in this packet, and
   * invoke the Print method of each tag stored in the packet.
   */
  void PrintByteTags (std::ostream &os) const;

  /**
   * \brief Add a packet tag.
   *
   * \param tag the packet tag type to add.
   *
   * Note that this method is const, that is, it does not
   * modify the state of this packet, which is fairly
   * un-intuitive.  See AddByteTag"()" discussion.
   */
  void AddPacketTag (const Tag &tag) const;
  /**
   * \brief Remove a packet tag.
   *
   * \param tag the packet tag type to remove from this packet.
   *        The tag parameter is set to the value of the tag found.
   * \returns true if the requested tag is found, false
   *          otherwise.
   */
  bool RemovePacketTag (Tag &tag);
  /**
   * \brief Replace the value of a packet tag.
   *
   * \param tag the packet tag type to replace.  To get the old
   *        value of the tag, use PeekPacketTag first.
   * \returns true if the requested tag is found, false otherwise.
   *        If the tag isn't found, Add is performed instead (so
   *        the packet is guaranteed to have the new tag value
   *        either way).
   */
  bool ReplacePacketTag (Tag & tag);
  /**
   * \brief Search a matching tag and call Tag::Deserialize if it is found.
   *
   * \param tag the tag to search in this packet
   * \returns true if the requested tag is found, false
   *          otherwise.
   */
  bool PeekPacketTag (Tag &tag) const;
  /**
   * \brief Remove all packet tags.
   */
  void RemoveAllPacketTags (void);

  /**
   * \brief Print the list of packet tags.
   *
   * \param os the stream on which to print the tags.
   *
   * \sa Packet::AddPacketTag, Packet::RemovePacketTag, Packet::PeekPacketTag,
   *  Packet::RemoveAllPacketTags
   */
  void PrintPacketTags (std::ostream &os) const;

  /**
   * \brief Returns an object which can be used to iterate over the list of
   *  packet tags.
   *
   * \returns an object which can be used to iterate over the list of
   *  packet tags.
   */
  PacketTagIterator GetPacketTagIterator (void) const;

  /**
   * \brief Set the packet nix-vector.
   *
   * \note Note: This function supports a temporary solution
   * to a specific problem in this generic class, i.e. 
   * how to associate something specific like nix-vector 
   * with a packet.  This design methodology 
   * should _not_ be followed, and is only here as an 
   * impetus to fix this general issue.
   *
   * \warning For real this function is not const, as it is the
   * setter for a mutable variable member. The const qualifier
   * is needed to set a private mutable variable of const objects.
   * 
   * \param nixVector the nix vector
   */
  void SetNixVector (Ptr<NixVector> nixVector) const;
  /**
   * \brief Get the packet nix-vector.
   *
   * See the comment on SetNixVector
   *
   * \returns the Nix vector
   */
  Ptr<NixVector> GetNixVector (void) const; 

  /**
   * TracedCallback signature for Ptr<Packet>
   *
   * \param [in] packet The packet.
   */
  typedef void (* TracedCallback) (Ptr<const Packet> packet);
  
  /**
   * TracedCallback signature for packet and Address.
   *
   * \param [in] packet The packet.
   * \param [in] address The address.
   */
  typedef void (* AddressTracedCallback)
    (Ptr<const Packet> packet, const Address &address);
  
   /**
    * TracedCallback signature for packet and source/destination addresses.
    *
    * \param [in] packet The packet.
    * \param [in] srcAddress The source address.
    * \param [in] destAddress The destination address.
    */
  typedef void (* TwoAddressTracedCallback)
    (const Ptr<const Packet> packet, const Address &srcAddress, const Address &destAddress);

  /**
   * TracedCallback signature for packet and Mac48Address.
   *
   * \param [in] packet The packet.
   * \param [in] mac The Mac48Address.
   */
  typedef void (* Mac48AddressTracedCallback)
    (Ptr<const Packet> packet, Mac48Address mac);
  
  /**
   * TracedCallback signature for changes in packet size.
   *
   * \param [in] oldSize The previous packet's size.
   * \param [in] newSize The actual packet's size.
   */
  typedef void (* SizeTracedCallback)
    (uint32_t oldSize, uint32_t newSize);

  /**
   * TracedCallback signature for packet and SINR.
   *
   * \param [in] packet The packet.
   * \param [in] sinr The received SINR.
   */
  typedef void (* SinrTracedCallback)
    (Ptr<const Packet> packet, double sinr);
  
  uint8_t* GetBuffer() const;  
  
private:
  /**
   * \brief Constructor
   * \param buffer the packet buffer
   * \param byteTagList the ByteTag list
   * \param packetTagList the packet's Tag list
   * \param metadata the packet's metadata
   */
  Packet (const Buffer &buffer, const ByteTagList &byteTagList, 
          const PacketTagList &packetTagList, const PacketMetadata &metadata);

  /**
   * \brief Deserializes a packet.
   * \param [in] buffer the input buffer.
   * \param [in] size the buffer size.
   * \returns the number of deserialized bytes.
   */
  uint32_t Deserialize (uint8_t const*buffer, uint32_t size);

  Buffer m_buffer;                //!< the packet buffer (it's actual contents)
  ByteTagList m_byteTagList;      //!< the ByteTag list
  PacketTagList m_packetTagList;  //!< the packet's Tag list
  PacketMetadata m_metadata;      //!< the packet's metadata

  /* Please see comments above about nix-vector */
  mutable Ptr<NixVector> m_nixVector; //!< the packet's Nix vector

  static uint32_t m_globalUid; //!< Global counter of packets Uid
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param packet the packet
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const Packet &packet);

/**
 * \ingroup network
 * \defgroup packetperf Packet Performance
 * The current implementation of the byte buffers and tag list is based
 * on COW (Copy On Write. An introduction to COW can be found in Scott 
 * Meyer's "More Effective C++", items 17 and 29). What this means is that
 * copying packets without modifying them is very cheap (in terms of cpu
 * and memory usage) and modifying them can be also very cheap. What is 
 * key for proper COW implementations is being
 * able to detect when a given modification of the state of a packet triggers
 * a full copy of the data prior to the modification: COW systems need
 * to detect when an operation is "dirty".
 *
 * Dirty operations:
 *   - ns3::Packet::AddHeader
 *   - ns3::Packet::AddTrailer
 *   - both versions of ns3::Packet::AddAtEnd
 *   - ns3::Packet::RemovePacketTag
 *   - ns3::Packet::ReplacePacketTag
 *
 * Non-dirty operations:
 *   - ns3::Packet::AddPacketTag
 *   - ns3::Packet::PeekPacketTag
 *   - ns3::Packet::RemoveAllPacketTags
 *   - ns3::Packet::AddByteTag
 *   - ns3::Packet::FindFirstMatchingByteTag
 *   - ns3::Packet::RemoveAllByteTags
 *   - ns3::Packet::RemoveHeader
 *   - ns3::Packet::RemoveTrailer
 *   - ns3::Packet::CreateFragment
 *   - ns3::Packet::RemoveAtStart
 *   - ns3::Packet::RemoveAtEnd
 *   - ns3::Packet::CopyData
 *
 * Dirty operations will always be slower than non-dirty operations,
 * sometimes by several orders of magnitude. However, even the
 * dirty operations have been optimized for common use-cases which
 * means that most of the time, these operations will not trigger
 * data copies and will thus be still very fast.
 */

} // namespace ns3

/****************************************************
 *  Implementation of inline methods for performance
 ****************************************************/

namespace ns3 {

uint32_t 
Packet::GetSize (void) const
{
  return m_buffer.GetSize ();
}

} // namespace ns3

#endif /* PACKET_H */
