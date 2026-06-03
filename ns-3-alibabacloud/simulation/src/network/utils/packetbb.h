/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* vim: set ts=2 sw=2 sta expandtab ai si cin: */
/* 
 * Copyright (c) 2009 Drexel University
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
 * Author: Tom Wambold <tom5760@gmail.com>
 */
/* These classes implement RFC 5444 - The Generalized Mobile Ad Hoc Network
 * (MANET) Packet/PbbMessage Format
 * See: http://tools.ietf.org/html/rfc5444 for details */

#ifndef PACKETBB_H
#define PACKETBB_H

#include <list>

#include "ns3/ptr.h"
#include "ns3/address.h"
#include "ns3/header.h"
#include "ns3/buffer.h"
#include "ns3/simple-ref-count.h"

namespace ns3 {

/* Forward declare objects */
class PbbMessage;
class PbbAddressBlock;
class PbbTlv;
class PbbAddressTlv;

/** Used in Messages to determine whether it contains IPv4 or IPv6 addresses */
enum PbbAddressLength {
  IPV4 = 3,
  IPV6 = 15,
};

/**
 * \brief A block of packet or message TLVs (PbbTlv).
 *
 * Acts similar to a C++ STL container.  Should not be used for Address TLVs.
 */
class PbbTlvBlock
{
public:
  /// PbbTlv container iterator
  typedef std::list< Ptr<PbbTlv> >::iterator Iterator;
  /// PbbTlv container const iterator
  typedef std::list< Ptr<PbbTlv> >::const_iterator ConstIterator;

  PbbTlvBlock (void);
  ~PbbTlvBlock (void);

  /**
   * \return an iterator to the first TLV in this block.
   */
  Iterator Begin (void);

  /**
   * \return a const iterator to the first TLV in this block.
   */
  ConstIterator Begin (void) const;

  /**
   * \return an iterator to the past-the-end element in this block.
   */
  Iterator End (void);

  /**
   * \return a const iterator to the past-the-end element in this block.
   */
  ConstIterator End (void) const;

  /**
   * \return the number of TLVs in this block.
   */
  int Size (void) const;

  /**
   * \return true if there are no TLVs in this block, false otherwise.
   */
  bool Empty (void) const;

  /**
   * \return a smart pointer to the first TLV in this block.
   */
  Ptr<PbbTlv> Front (void) const;

  /**
   * \return a smart pointer to the last TLV in this block.
   */
  Ptr<PbbTlv> Back (void) const;

  /**
   * \brief Prepends a TLV to the front of this block.
   * \param tlv a smart pointer to the TLV to prepend.
   */
  void PushFront (Ptr<PbbTlv> tlv);

  /**
   * \brief Removes a TLV from the front of this block.
   */
  void PopFront (void);

  /**
   * \brief Appends a TLV to the back of this block.
   * \param tlv a smart pointer to the TLV to append.
   */
  void PushBack (Ptr<PbbTlv> tlv);

  /**
   * \brief Removes a TLV from the back of this block.
   */
  void PopBack (void);

  /**
   * \brief Inserts a TLV at the specified position in this block.
   * \param position an Iterator pointing to the position in this block to
   *        insert the TLV.
   * \param tlv a smart pointer to the TLV to insert.
   * \return An iterator pointing to the newly inserted TLV.
   */
  Iterator Insert (Iterator position, const Ptr<PbbTlv> tlv);

  /**
   * \brief Removes the TLV at the specified position.
   * \param position an Iterator pointing to the TLV to erase.
   * \return an iterator pointing to the next TLV in the block.
   */
  Iterator Erase (Iterator position);

  /**
   * \brief Removes all TLVs from [first, last) (includes first, not includes
   *        last).
   * \param first an Iterator pointing to the first TLV to erase (inclusive).
   * \param last an Iterator pointing to the element past the last TLV to erase.
   * \return an iterator pointing to the next TLV in the block.
   */
  Iterator Erase (Iterator first, Iterator last);

  /**
   * \brief Removes all TLVs from this block.
   */
  void Clear (void);

  /**
   * \return The size (in bytes) needed to serialize this block.
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serializes this block into the specified buffer.
   * \param start a reference to the point in a buffer to begin serializing.
   *
   * Users should not need to call this.  Blocks will be serialized by their
   * containing packet.
   */
  void Serialize (Buffer::Iterator &start) const;

  /**
   * \brief Deserializes a block from the specified buffer.
   * \param start a reference to the point in a buffer to begin deserializing.
   *
   * Users should not need to call this.  Blocks will be deserialized by their
   * containing packet.
   */
  void Deserialize (Buffer::Iterator &start);

  /**
   * \brief Pretty-prints the contents of this block.
   * \param os a stream object to print to.
   */
  void Print (std::ostream &os) const;

  /**
   * \brief Pretty-prints the contents of this block, with specified indentation.
   * \param os a stream object to print to.
   * \param level level of indentation.
   *
   * This probably never needs to be called by users.  This is used when
   * recursively printing sub-objects.
   */
  void Print (std::ostream &os, int level) const;

  /**
   * \brief Equality operator for PbbTlvBlock
   * \param other PbbTlvBlock to compare this one to
   * \returns true if the blocks are equal
   */
  bool operator== (const PbbTlvBlock &other) const;
  /**
   * \brief Inequality operator for PbbTlvBlock
   * \param other PbbTlvBlock to compare this one to
   * \returns true if the blocks are not equal
   */  
  bool operator!= (const PbbTlvBlock &other) const;

private:
  std::list< Ptr<PbbTlv> > m_tlvList; //!< PbbTlv container
};

/**
 * \brief A block of Address TLVs (PbbAddressTlv).
 *
 * Acts similar to a C++ STL container.
 */
class PbbAddressTlvBlock
{
public:
  /// PbbAddressTlv iterator for PbbAddressTlvBlock
  typedef std::list< Ptr<PbbAddressTlv> >::iterator Iterator;
  /// PbbAddressTlv const iterator for PbbAddressTlvBlock
  typedef std::list< Ptr<PbbAddressTlv> >::const_iterator ConstIterator;

  PbbAddressTlvBlock (void);
  ~PbbAddressTlvBlock (void);

  /**
   * \return an iterator to the first Address TLV in this block.
   */
  Iterator Begin (void);

  /**
   * \return a const iterator to the first Address TLV in this block.
   */
  ConstIterator Begin (void) const;

  /**
   * \return an iterator to the past-the-end element in this block.
   */
  Iterator End (void);

  /**
   * \return a const iterator to the past-the-end element in this block.
   */
  ConstIterator End (void) const;

  /**
   * \return the number of Address TLVs in this block.
   */
  int Size (void) const;

  /**
   * \return true if there are no Address TLVs in this block, false otherwise.
   */
  bool Empty (void) const;

  /**
   * \return the first Address TLV in this block.
   */
  Ptr<PbbAddressTlv> Front (void) const;

  /**
   * \return the last AddressTLV in this block.
   */
  Ptr<PbbAddressTlv> Back (void) const;

  /**
   * \brief Prepends an Address TLV to the front of this block.
   * \param tlv a smart pointer to the Address TLV to prepend.
   */
  void PushFront (Ptr<PbbAddressTlv> tlv);

  /**
   * \brief Removes an AddressTLV from the front of this block.
   */
  void PopFront (void);

  /**
   * \brief Appends an Address TLV to the back of this block.
   * \param tlv a smart pointer to the Address TLV to append.
   */
  void PushBack (Ptr<PbbAddressTlv> tlv);

  /**
   * \brief Removes an Address TLV from the back of this block.
   */
  void PopBack (void);

  /**
   * \brief Inserts an Address TLV at the specified position in this block.
   * \param position an Iterator pointing to the position in this block to
   *        insert the Address TLV.
   * \param tlv a smart pointer to the Address TLV to insert.
   * \return An iterator pointing to the newly inserted Address TLV.
   */
  Iterator Insert (Iterator position, const Ptr<PbbAddressTlv> tlv);

  /**
   * \brief Removes the Address TLV at the specified position.
   * \param position an Iterator pointing to the Address TLV to erase.
   * \return an iterator pointing to the next Address TLV in the block.
   */
  Iterator Erase (Iterator position);

  /**
   * \brief Removes all Address TLVs from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first Address TLV to erase
   *        (inclusive).
   * \param last an Iterator pointing to the element past the last Address TLV
   *        to erase.
   * \return an iterator pointing to the next Address TLV in the block.
   */
  Iterator Erase (Iterator first, Iterator last);

  /**
   * \brief Removes all Address TLVs from this block.
   */
  void Clear (void);

  /**
   * \return The size (in bytes) needed to serialize this block.
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serializes this block into the specified buffer.
   * \param start a reference to the point in a buffer to begin serializing.
   *
   * Users should not need to call this.  Blocks will be serialized by their
   * containing packet.
   */
  void Serialize (Buffer::Iterator &start) const;

  /**
   * \brief Deserializes a block from the specified buffer.
   * \param start a reference to the point in a buffer to begin deserializing.
   *
   * Users should not need to call this.  Blocks will be deserialized by their
   * containing packet.
   */
  void Deserialize (Buffer::Iterator &start);

  /**
   * \brief Pretty-prints the contents of this block.
   * \param os a stream object to print to.
   */
  void Print (std::ostream &os) const;

  /**
   * \brief Pretty-prints the contents of this block, with specified indentation.
   * \param os a stream object to print to.
   * \param level level of indentation.
   *
   * This probably never needs to be called by users.  This is used when
   * recursively printing sub-objects.
   */
  void Print (std::ostream &os, int level) const;

  /**
   * \brief Equality operator for PbbAddressTlvBlock
   * \param other PbbAddressTlvBlock to compare to this one
   * \returns true if PbbAddressTlvBlock are equal
   */
  bool operator== (const PbbAddressTlvBlock &other) const;

  /**
   * \brief Inequality operator for PbbAddressTlvBlock
   * \param other PbbAddressTlvBlock to compare to this one
   * \returns true if PbbAddressTlvBlock are not equal
   */  
  bool operator!= (const PbbAddressTlvBlock &other) const;

private:
  std::list< Ptr<PbbAddressTlv> > m_tlvList; //!< PbbAddressTlv container
};

/**
 * \brief Main PacketBB Packet object.
 *
 * A PacketBB packet is made up of zero or more packet TLVs (PbbTlv), and zero
 * or more messages (PbbMessage).
 *
 * See: \RFC{5444} for details.
 */
class PbbPacket : public SimpleRefCount<PbbPacket,Header>
{
public:
  /// PbbTlv iterator for PbbPacket
  typedef std::list< Ptr<PbbTlv> >::iterator TlvIterator;
  /// PbbTlv const iterator for PbbPacket
  typedef std::list< Ptr<PbbTlv> >::const_iterator ConstTlvIterator;
  /// PbbMessage Iterator for PbbPacket
  typedef std::list< Ptr<PbbMessage> >::iterator MessageIterator;
  /// PbbMessage Const Iterator for PbbPacket
  typedef std::list< Ptr<PbbMessage> >::const_iterator ConstMessageIterator;

  PbbPacket (void);
  ~PbbPacket (void);

  /**
   * \return the version of PacketBB that constructed this packet.
   *
   * This will always return 0 for packets constructed using this API.
   */
  uint8_t GetVersion (void) const;

  /**
   * \brief Sets the sequence number of this packet.
   * \param number the sequence number.
   */
  void SetSequenceNumber (uint16_t number);

  /**
   * \return the sequence number of this packet.
   *
   * Calling this while HasSequenceNumber is False is undefined.  Make sure you
   * check it first.  This will be checked by an assert in debug builds.
   */
  uint16_t GetSequenceNumber (void) const;

  /**
   * \brief Tests whether or not this packet has a sequence number.
   * \return true if this packet has a sequence number, false otherwise.
   *
   * This should be called before calling GetSequenceNumber to make sure there
   * actually is one.
   */
  bool HasSequenceNumber (void) const;

  /* Manipulating Packet TLVs */

  /**
   * \return an iterator to the first Packet TLV in this packet.
   */
  TlvIterator TlvBegin (void);

  /**
   * \return a const iterator to the first Packet TLV in this packet.
   */
  ConstTlvIterator TlvBegin (void) const;

  /**
   * \return an iterator to the past-the-end element in this packet TLV block.
   */
  TlvIterator TlvEnd (void);

  /**
   * \return a const iterator to the past-the-end element in this packet TLV
   *         block.
   */
  ConstTlvIterator TlvEnd (void) const;

  /**
   * \return the number of packet TLVs in this packet.
   */
  int TlvSize (void) const;

  /**
   * \return true if there are no packet TLVs in this packet, false otherwise.
   */
  bool TlvEmpty (void) const;

  /**
   * \return a smart pointer to the first packet TLV in this packet.
   */
  Ptr<PbbTlv> TlvFront (void);

  /**
   * \return a const smart pointer to the first packet TLV in this packet.
   */
  const Ptr<PbbTlv> TlvFront (void) const;

  /**
   * \return a smart pointer to the last packet TLV in this packet.
   */
  Ptr<PbbTlv> TlvBack (void);

  /**
   * \return a const smart pointer to the last packet TLV in this packet.
   */
  const Ptr<PbbTlv> TlvBack (void) const;

  /**
   * \brief Prepends a packet TLV to the front of this packet.
   * \param tlv a smart pointer to the packet TLV to prepend.
   */
  void TlvPushFront (Ptr<PbbTlv> tlv);

  /**
   * \brief Removes a packet TLV from the front of this packet.
   */
  void TlvPopFront (void);

  /**
   * \brief Appends a packet TLV to the back of this packet.
   * \param tlv a smart pointer to the packet TLV to append.
   */
  void TlvPushBack (Ptr<PbbTlv> tlv);

  /**
   * \brief Removes a packet TLV from the back of this block.
   */
  void TlvPopBack (void);

  /**
   * \brief Removes the packet TLV at the specified position.
   * \param position an Iterator pointing to the packet TLV to erase.
   * \return an iterator pointing to the next packet TLV in the block.
   */
  TlvIterator Erase (TlvIterator position);

  /**
   * \brief Removes all packet TLVs from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first packet TLV to erase
   *        (inclusive).
   * \param last an Iterator pointing to the element past the last packet TLV
   *        to erase.
   * \return an iterator pointing to the next packet TLV in the block.
   */
  TlvIterator Erase (TlvIterator first, TlvIterator last);

  /**
   * \brief Removes all packet TLVs from this packet.
   */
  void TlvClear (void);

  /* Manipulating Packet Messages */

  /**
   * \return an iterator to the first message in this packet.
   */
  MessageIterator MessageBegin (void);

  /**
   * \return a const iterator to the first message in this packet.
   */
  ConstMessageIterator MessageBegin (void) const;

  /**
   * \return an iterator to the past-the-end element in this message block.
   */
  MessageIterator MessageEnd (void);

  /**
   * \return a const iterator to the past-the-end element in this message
   *         block.
   */
  ConstMessageIterator MessageEnd (void) const;

  /**
   * \return the number of messages in this packet.
   */
  int MessageSize (void) const;

  /**
   * \return true if there are no messages in this packet, false otherwise.
   */
  bool MessageEmpty (void) const;

  /**
   * \return a smart pointer to the first message in this packet.
   */
  Ptr<PbbMessage> MessageFront (void);

  /**
   * \return a const smart pointer to the first message in this packet.
   */
  const Ptr<PbbMessage> MessageFront (void) const;

  /**
   * \return a smart pointer to the last message in this packet.
   */
  Ptr<PbbMessage> MessageBack (void);

  /**
   * \return a const smart pointer to the last message in this packet.
   */
  const Ptr<PbbMessage> MessageBack (void) const;

  /**
   * \brief Prepends a message to the front of this packet.
   * \param message a smart pointer to the message to prepend.
   */
  void MessagePushFront (Ptr<PbbMessage> message);

  /**
   * \brief Removes a message from the front of this packet.
   */
  void MessagePopFront (void);

  /**
   * \brief Appends a message to the back of this packet.
   * \param message a smart pointer to the message to append.
   */
  void MessagePushBack (Ptr<PbbMessage> message);

  /**
   * \brief Removes a message from the back of this packet.
   */
  void MessagePopBack (void);

  /**
   * \brief Removes the message at the specified position.
   * \param position an Iterator pointing to the message to erase.
   * \return an iterator pointing to the next message in the packet.
   */
  MessageIterator Erase (MessageIterator position);

  /**
   * \brief Removes all messages from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first message to erase (inclusive).
   * \param last an Iterator pointing to the element past the last message to erase.
   * \return an iterator pointing to the next message in the block.
   */
  MessageIterator Erase (MessageIterator first, MessageIterator last);

  /**
   * \brief Removes all messages from this packet.
   */
  void MessageClear (void);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * \return The size (in bytes) needed to serialize this packet.
   */
  virtual uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serializes this packet into the specified buffer.
   * \param start a reference to the point in a buffer to begin serializing.
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserializes a packet from the specified buffer.
   * \param start start offset
   * \return the number of bytes deserialized
   *
   * If this returns a number smaller than the total number of bytes in the
   * buffer, there was an error.
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

  /**
   * \brief Pretty-prints the contents of this block.
   * \param os a stream object to print to.
   */
  virtual void Print (std::ostream &os) const;

  /**
   * \brief Equality operator for PbbPacket
   * \param other PbbPacket to compare to this one
   * \returns true if PbbPacket are equal
   */
  bool operator== (const PbbPacket &other) const;
  
  /**
   * \brief Inequality operator for PbbPacket
   * \param other PbbPacket to compare to this one
   * \returns true if PbbPacket are not equal
   */
  bool operator!= (const PbbPacket &other) const;

protected:

private:
  PbbTlvBlock m_tlvList; //!< PbbTlv container
  std::list< Ptr<PbbMessage> > m_messageList; //!< PbbTlvBlock container

  uint8_t m_version; //!< version

  bool m_hasseqnum; //!< Sequence number present
  uint16_t m_seqnum; //!< Sequence number
};

/**
 * \brief A message within a PbbPacket packet.
 *
 * There may be any number of messages in one packet packet.  This is a pure
 * virtual base class, when creating a message, you should instantiate either
 * PbbMessageIpv4 or PbbMessageIpv6.
 */
class PbbMessage : public SimpleRefCount<PbbMessage>
{
public:
  /// PbbTlv iterator
  typedef std::list< Ptr<PbbTlv> >::iterator TlvIterator;
  /// PbbTlv const iterator
  typedef std::list< Ptr<PbbTlv> >::const_iterator ConstTlvIterator;
  /// PbbAddressBlock iterator
  typedef std::list< Ptr<PbbAddressBlock> >::iterator AddressBlockIterator;
  /// PbbAddressBlock const iterator
  typedef std::list< Ptr<PbbAddressBlock> >::const_iterator ConstAddressBlockIterator;

  PbbMessage ();
  virtual ~PbbMessage ();

  /**
   * \brief Sets the type for this message.
   * \param type the type to set.
   */
  void SetType (uint8_t type);

  /**
   * \return the type assigned to this packet
   */
  uint8_t GetType (void) const;

  /**
   * \brief Sets the address for the node that created this packet.
   * \param address the originator address.
   */
  void SetOriginatorAddress (Address address);

  /**
   * \return the address of the node that created this packet.
   *
   * Calling this while HasOriginatorAddress is False is undefined.  Make sure
   * you check it first.  This will be checked by an assert in debug builds.
   */
  Address GetOriginatorAddress (void) const;

  /**
   * \brief Tests whether or not this message has an originator address.
   * \return true if this message has an originator address, false otherwise.
   */
  bool HasOriginatorAddress (void) const;

  /**
   * \brief Sets the maximum number of hops this message should travel
   * \param hoplimit the limit to set
   */
  void SetHopLimit (uint8_t hoplimit);

  /**
   * \return the maximum number of hops this message should travel.
   *
   * Calling this while HasHopLimit is False is undefined.  Make sure you check
   * it first.  This will be checked by an assert in debug builds.
   */
  uint8_t GetHopLimit (void) const;

  /**
   * \brief Tests whether or not this message has a hop limit.
   * \return true if this message has a hop limit, false otherwise.
   *
   * If this is set, messages should not hop further than this limit.
   */
  bool HasHopLimit (void) const;

  /**
   * \brief Sets the current number of hops this message has traveled.
   * \param hopcount the current number of hops
   */
  void SetHopCount (uint8_t hopcount);

  /**
   * \return the current number of hops this message has traveled.
   *
   * Calling this while HasHopCount is False is undefined.  Make sure you check
   * it first.  This will be checked by an assert in debug builds.
   */
  uint8_t GetHopCount (void) const;

  /**
   * \brief Tests whether or not this message has a hop count.
   * \return true if this message has a hop limit, false otherwise.
   */
  bool HasHopCount (void) const;

  /**
   * \brief Sets the sequence number of this message.
   * \param seqnum the sequence number to set.
   */
  void SetSequenceNumber (uint16_t seqnum);

  /**
   * \return the sequence number of this message.
   *
   * Calling this while HasSequenceNumber is False is undefined.  Make sure you
   * check it first.  This will be checked by an assert in debug builds.
   */
  uint16_t GetSequenceNumber (void) const;

  /**
   * \brief Tests whether or not this message has a sequence number.
   * \return true if this message has a sequence number, false otherwise.
   */
  bool HasSequenceNumber (void) const;

  /* Manipulating PbbMessage TLVs */

  /**
   * \return an iterator to the first message TLV in this message.
   */
  TlvIterator TlvBegin ();

  /**
   * \return a const iterator to the first message TLV in this message.
   */
  ConstTlvIterator TlvBegin () const;

  /**
   * \return an iterator to the past-the-end message TLV element in this
   *         message.
   */
  TlvIterator TlvEnd ();

  /**
   * \return a const iterator to the past-the-end message TLV element in this
   *         message.
   */
  ConstTlvIterator TlvEnd () const;

  /**
   * \return the number of message TLVs in this message.
   */
  int TlvSize (void) const;

  /**
   * \return true if there are no message TLVs in this message, false otherwise.
   */
  bool TlvEmpty (void) const;

  /**
   * \return a smart pointer to the first message TLV in this message.
   */
  Ptr<PbbTlv> TlvFront (void);

  /**
   * \return a const smart pointer to the first message TLV in this message.
   */
  const Ptr<PbbTlv> TlvFront (void) const;

  /**
   * \return a smart pointer to the last message TLV in this message.
   */
  Ptr<PbbTlv> TlvBack (void);

  /**
   * \return a const smart pointer to the last message TLV in this message.
   */
  const Ptr<PbbTlv> TlvBack (void) const;

  /**
   * \brief Prepends a message TLV to the front of this message.
   * \param tlv a smart pointer to the message TLV to prepend.
   */
  void TlvPushFront (Ptr<PbbTlv> tlv);

  /**
   * \brief Removes a message TLV from the front of this message.
   */
  void TlvPopFront (void);

  /**
   * \brief Appends a message TLV to the back of this message.
   * \param tlv a smart pointer to the message TLV to append.
   */
  void TlvPushBack (Ptr<PbbTlv> tlv);

  /**
   * \brief Removes a message TLV from the back of this message.
   */
  void TlvPopBack (void);

  /**
   * \brief Removes the message TLV at the specified position.
   * \param position an Iterator pointing to the message TLV to erase.
   * \return an iterator pointing to the next TLV in the block.
   */
  TlvIterator TlvErase (TlvIterator position);

  /**
   * \brief Removes all message TLVs from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first message TLV to erase
   *        (inclusive).
   * \param last an Iterator pointing to the element past the last message TLV
   *        to erase.
   * \return an iterator pointing to the next message TLV in the message.
   */
  TlvIterator TlvErase (TlvIterator first, TlvIterator last);

  /**
   * \brief Removes all message TLVs from this block.
   */
  void TlvClear (void);

  /* Manipulating Address Block and Address TLV pairs */

  /**
   * \return an iterator to the first address block in this message.
   */
  AddressBlockIterator AddressBlockBegin ();

  /**
   * \return a const iterator to the first address block in this message.
   */
  ConstAddressBlockIterator AddressBlockBegin () const;

  /**
   * \return an iterator to the past-the-end address block element in this
   *         message.
   */
  AddressBlockIterator AddressBlockEnd ();

  /**
   * \return a const iterator to the past-the-end address block element in this
   *         message.
   */
  ConstAddressBlockIterator AddressBlockEnd () const;

  /**
   * \return the number of address blocks in this message.
   */
  int AddressBlockSize (void) const;

  /**
   * \return true if there are no address blocks in this message, false
   *         otherwise.
   */
  bool AddressBlockEmpty (void) const;

  /**
   * \return a smart pointer to the first address block in this message.
   */
  Ptr<PbbAddressBlock> AddressBlockFront (void);

  /**
   * \return a const smart pointer to the first address block in this message.
   */
  const Ptr<PbbAddressBlock> AddressBlockFront (void) const;

  /**
   * \return a smart pointer to the last address block in this message.
   */
  Ptr<PbbAddressBlock> AddressBlockBack (void);

  /**
   * \return a const smart pointer to the last address block in this message.
   */
  const Ptr<PbbAddressBlock> AddressBlockBack (void) const;

  /**
   * \brief Prepends an address block to the front of this message.
   * \param block a smart pointer to the address block to prepend.
   */
  void AddressBlockPushFront (Ptr<PbbAddressBlock> block);

  /**
   * \brief Removes an address block from the front of this message.
   */
  void AddressBlockPopFront (void);

  /**
   * \brief Appends an address block to the front of this message.
   * \param block a smart pointer to the address block to append.
   */
  void AddressBlockPushBack (Ptr<PbbAddressBlock> block);

  /**
   * \brief Removes an address block from the back of this message.
   */
  void AddressBlockPopBack (void);

  /**
   * \brief Removes the address block at the specified position.
   * \param position an Iterator pointing to the address block to erase.
   * \return an iterator pointing to the next address block in the message.
   */
  AddressBlockIterator AddressBlockErase (AddressBlockIterator position);

  /**
   * \brief Removes all address blocks from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first address block to erase
   *        (inclusive).
   * \param last an Iterator pointing to the element past the last address
   *        block to erase.
   * \return an iterator pointing to the next address block in the message.
   */
  AddressBlockIterator AddressBlockErase (AddressBlockIterator first,
                                          AddressBlockIterator last);

  /**
   * \brief Removes all address blocks from this message.
   */
  void AddressBlockClear (void);

  /**
   * \brief Deserializes a message, returning the correct object depending on
   *        whether it is an IPv4 message or an IPv6 message.
   * \param start a reference to the point in a buffer to begin deserializing.
   * \return A pointer to the deserialized message, or 0 on error.
   *
   * Users should not need to call this.  Blocks will be deserialized by their
   * containing packet.
   */
  static Ptr<PbbMessage> DeserializeMessage (Buffer::Iterator &start);

  /**
   * \return The size (in bytes) needed to serialize this message.
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serializes this message into the specified buffer.
   * \param start a reference to the point in a buffer to begin serializing.
   *
   * Users should not need to call this.  Blocks will be deserialized by their
   * containing packet.
   */
  void Serialize (Buffer::Iterator &start) const;

  /**
   * \brief Deserializes a message from the specified buffer.
   * \param start a reference to the point in a buffer to begin deserializing.
   *
   * Users should not need to call this.  Blocks will be deserialized by their
   * containing packet.
   */
  void Deserialize (Buffer::Iterator &start);

  /**
   * \brief Pretty-prints the contents of this message.
   * \param os a stream object to print to.
   */
  void Print (std::ostream &os) const;

  /**
   * \brief Pretty-prints the contents of this message, with specified
   *        indentation.
   * \param os a stream object to print to.
   * \param level level of indentation.
   *
   * This probably never needs to be called by users.  This is used when
   * recursively printing sub-objects.
   */
  void Print (std::ostream &os, int level) const;

  /**
   * \brief Equality operator for PbbMessage
   * \param other PbbMessage to compare to this one
   * \returns true if PbbMessages are equal
   */
  bool operator== (const PbbMessage &other) const;
  /**
   * \brief Inequality operator for PbbMessage
   * \param other PbbMessage to compare to this one
   * \returns true if PbbMessages are not equal
   */
  bool operator!= (const PbbMessage &other) const;

protected:  
  /**
   * \brief Returns address length (IPV4 3 or IPV6 15)
   *
   *  Returns message size in bytes - 1
   *  IPv4 = 4 - 1 = 3, IPv6 = 16 - 1 = 15
   *
   * \returns Address length (IPV4 3 or IPV6 15)
   */
  virtual PbbAddressLength GetAddressLength (void) const = 0;

  /**
   * \brief Serialize the originator address
   * \param start the buffer iterator start
   */
  virtual void SerializeOriginatorAddress (Buffer::Iterator &start) const = 0;
  /**
   * \brief Deserialize the originator address
   * \param start the buffer iterator start
   * \returns the deserialized address
   */
  virtual Address DeserializeOriginatorAddress (Buffer::Iterator &start) const = 0;
  /**
   * \brief Print the originator address
   * \param os the output stream
   */
  virtual void PrintOriginatorAddress (std::ostream &os) const = 0;

  /**
   * \brief Deserialize an address block
   * \param start the buffer iterator start
   * \returns the deserialized address block
   */
  virtual Ptr<PbbAddressBlock> AddressBlockDeserialize (Buffer::Iterator &start) const = 0;

private:
  PbbTlvBlock m_tlvList; //!< PbbTlvBlock
  std::list< Ptr<PbbAddressBlock> > m_addressBlockList; //!< PbbAddressBlock container

  uint8_t m_type; //!< the type for this message
  PbbAddressLength m_addrSize; //!< the address size

  bool m_hasOriginatorAddress; //!< Originator address present
  Address m_originatorAddress; //!< originator address

  bool m_hasHopLimit; //!< Hop limit present
  uint8_t m_hopLimit; //!< Hop limit

  bool m_hasHopCount; //!< Hop count present
  uint8_t m_hopCount; //!< Hop count

  bool m_hasSequenceNumber;  //!< Sequence number present
  uint16_t m_sequenceNumber; //!< Sequence number
};

/**
 * \brief Concrete IPv4 specific PbbMessage.
 *
 * This message will only contain IPv4 addresses.
 */
class PbbMessageIpv4 : public PbbMessage {
public:
  PbbMessageIpv4 ();
  virtual ~PbbMessageIpv4 ();

protected:
  /**
   * \brief Returns address length (IPV4 3 or IPV6 15)
   *
   *  Returns message size in bytes - 1
   *  IPv4 = 4 - 1 = 3, IPv6 = 16 - 1 = 15
   *
   * \returns Address length (IPV4 3 or IPV6 15)
   */
  virtual PbbAddressLength GetAddressLength (void) const;

  virtual void SerializeOriginatorAddress (Buffer::Iterator &start) const;
  virtual Address DeserializeOriginatorAddress (Buffer::Iterator &start) const;
  virtual void PrintOriginatorAddress (std::ostream &os) const;

  virtual Ptr<PbbAddressBlock> AddressBlockDeserialize (Buffer::Iterator &start) const;
};

/**
 * \brief Concrete IPv6 specific PbbMessage class.
 *
 * This message will only contain IPv6 addresses.
 */
class PbbMessageIpv6 : public PbbMessage {
public:
  PbbMessageIpv6 ();
  virtual ~PbbMessageIpv6 ();

protected:
  /**
   * \brief Returns address length (IPV4 3 or IPV6 15)
   *
   *  Returns message size in bytes - 1
   *  IPv4 = 4 - 1 = 3, IPv6 = 16 - 1 = 15
   *
   * \returns Address length (IPV4 3 or IPV6 15)
   */
  virtual PbbAddressLength GetAddressLength (void) const;

  virtual void SerializeOriginatorAddress (Buffer::Iterator &start) const;
  virtual Address DeserializeOriginatorAddress (Buffer::Iterator &start) const;
  virtual void PrintOriginatorAddress (std::ostream &os) const;

  virtual Ptr<PbbAddressBlock> AddressBlockDeserialize (Buffer::Iterator &start) const;
};

/**
 * \brief An Address Block and its associated Address TLV Blocks.
 *
 * This is a pure virtual base class, when creating address blocks, you should
 * instantiate either PbbAddressBlockIpv4 or PbbAddressBlockIpv6.
 */
class PbbAddressBlock : public SimpleRefCount<PbbAddressBlock>
{
public:
  /// Address iterator
  typedef std::list< Address >::iterator AddressIterator;
  /// Address const iterator
  typedef std::list< Address >::const_iterator ConstAddressIterator;

  /// Prefix iterator
  typedef std::list<uint8_t>::iterator PrefixIterator;
  /// Prefix const iterator
  typedef std::list<uint8_t>::const_iterator ConstPrefixIterator;

  /// tlvblock iterator
  typedef PbbAddressTlvBlock::Iterator TlvIterator;
  /// tlvblock const iterator
  typedef PbbAddressTlvBlock::ConstIterator ConstTlvIterator;

  PbbAddressBlock ();
  virtual ~PbbAddressBlock ();

  /* Manipulating the address block */

  /**
   * \return an iterator to the first address in this block.
   */
  AddressIterator AddressBegin (void);

  /**
   * \return a const iterator to the first address in this block.
   */
  ConstAddressIterator AddressBegin (void) const;

  /**
   * \return an iterator to the last address in this block.
   */
  AddressIterator AddressEnd (void);

  /**
   * \return a const iterator to the last address in this block.
   */
  ConstAddressIterator AddressEnd (void) const;

  /**
   * \return the number of addresses in this block.
   */
  int AddressSize (void) const;

  /**
   * \return true if there are no addresses in this block, false otherwise.
   */
  bool AddressEmpty (void) const;

  /**
   * \return the first address in this block.
   */
  Address AddressFront (void) const;

  /**
   * \return the last address in this block.
   */
  Address AddressBack (void) const;

  /**
   * \brief Prepends an address to the front of this block.
   * \param address the address to prepend.
   */
  void AddressPushFront (Address address);

  /**
   * \brief Removes an address from the front of this block.
   */
  void AddressPopFront (void);

  /**
   * \brief Appends an address to the back of this block.
   * \param address the address to append.
   */
  void AddressPushBack (Address address);

  /**
   * \brief Removes an address from the back of this block.
   */
  void AddressPopBack (void);

  /**
   * \brief Inserts an address at the specified position in this block.
   * \param position an Iterator pointing to the position in this block to
   *        insert the address.
   * \param value the address to insert.
   * \return An iterator pointing to the newly inserted address.
   */
  AddressIterator AddressInsert (AddressIterator position,
                                 const Address value);

  /**
   * \brief Removes the address at the specified position.
   * \param position an Iterator pointing to the address to erase.
   * \return an iterator pointing to the next address in the block.
   */
  AddressIterator AddressErase (AddressIterator position);

  /**
   * \brief Removes all addresses from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first address to erase
   *        (inclusive).
   * \param last an Iterator pointing to the element past the last address to
   *        erase.
   * \return an iterator pointing to the next address in the block.
   */
  AddressIterator AddressErase (AddressIterator first, AddressIterator last);

  /**
   * \brief Removes all addresses from this block.
   */
  void AddressClear (void);

  /* Prefix methods */

  /**
   * \return an iterator to the first prefix in this block.
   */
  PrefixIterator PrefixBegin (void);

  /**
   * \return a const iterator to the first prefix in this block.
   */
  ConstPrefixIterator PrefixBegin (void) const;

  /**
   * \return an iterator to the last prefix in this block.
   */
  PrefixIterator PrefixEnd (void);

  /**
   * \return a const iterator to the last prefix in this block.
   */
  ConstPrefixIterator PrefixEnd (void) const;

  /**
   * \return the number of prefixes in this block.
   */
  int PrefixSize (void) const;

  /**
   * \return true if there are no prefixes in this block, false otherwise.
   */
  bool PrefixEmpty (void) const;

  /**
   * \return the first prefix in this block.
   */
  uint8_t PrefixFront (void) const;

  /**
   * \return the last prefix in this block.
   */
  uint8_t PrefixBack (void) const;

  /**
   * \brief Prepends a prefix to the front of this block.
   * \param prefix the prefix to prepend.
   */
  void PrefixPushFront (uint8_t prefix);

  /**
   * \brief Removes a prefix from the front of this block.
   */
  void PrefixPopFront (void);

  /**
   * \brief Appends a prefix to the back of this block.
   * \param prefix the prefix to append.
   */
  void PrefixPushBack (uint8_t prefix);

  /**
   * \brief Removes a prefix from the back of this block.
   */
  void PrefixPopBack (void);

  /**
   * \brief Inserts a prefix at the specified position in this block.
   * \param position an Iterator pointing to the position in this block to
   *        insert the prefix.
   * \param value the prefix to insert.
   * \return An iterator pointing to the newly inserted prefix.
   */
  PrefixIterator PrefixInsert (PrefixIterator position, const uint8_t value);

  /**
   * \brief Removes the prefix at the specified position.
   * \param position an Iterator pointing to the prefix to erase.
   * \return an iterator pointing to the next prefix in the block.
   */
  PrefixIterator PrefixErase (PrefixIterator position);

  /**
   * \brief Removes all prefixes from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first prefix to erase
   *        (inclusive).
   * \param last an Iterator pointing to the element past the last prefix to
   *        erase.
   * \return an iterator pointing to the next prefix in the block.
   */
  PrefixIterator PrefixErase (PrefixIterator first, PrefixIterator last);

  /**
   * \brief Removes all prefixes from this block.
   */
  void PrefixClear (void);

  /* Manipulating the TLV block */

  /**
   * \return an iterator to the first address TLV in this block.
   */
  TlvIterator TlvBegin (void);

  /**
   * \return a const iterator to the first address TLV in this block.
   */
  ConstTlvIterator TlvBegin (void) const;

  /**
   * \return an iterator to the last address TLV in this block.
   */
  TlvIterator TlvEnd (void);

  /**
   * \return a const iterator to the last address TLV in this block.
   */
  ConstTlvIterator TlvEnd (void) const;

  /**
   * \return the number of address TLVs in this block.
   */
  int TlvSize (void) const;

  /**
   * \return true if there are no address TLVs in this block, false otherwise.
   */
  bool TlvEmpty (void) const;

  /**
   * \return a smart pointer to the first address TLV in this block.
   */
  Ptr<PbbAddressTlv> TlvFront (void);

  /**
   * \return a const smart pointer to the first address TLV in this message.
   */
  const Ptr<PbbAddressTlv> TlvFront (void) const;

  /**
   * \return a smart pointer to the last address TLV in this message.
   */
  Ptr<PbbAddressTlv> TlvBack (void);

  /**
   * \return a const smart pointer to the last address TLV in this message.
   */
  const Ptr<PbbAddressTlv> TlvBack (void) const;

  /**
   * \brief Prepends an address TLV to the front of this message.
   * \param address a smart pointer to the address TLV to prepend.
   */
  void TlvPushFront (Ptr<PbbAddressTlv> address);

  /**
   * \brief Removes an address TLV from the front of this message.
   */
  void TlvPopFront (void);

  /**
   * \brief Appends an address TLV to the back of this message.
   * \param address a smart pointer to the address TLV to append.
   */
  void TlvPushBack (Ptr<PbbAddressTlv> address);

  /**
   * \brief Removes an address TLV from the back of this message.
   */
  void TlvPopBack (void);

  /**
   * \brief Inserts an address TLV at the specified position in this block.
   * \param position an Iterator pointing to the position in this block to
   *        insert the address TLV.
   * \param value the prefix to insert.
   * \return An iterator pointing to the newly inserted address TLV.
   */
  TlvIterator TlvInsert (TlvIterator position, const Ptr<PbbTlv> value);

  /**
   * \brief Removes the address TLV at the specified position.
   * \param position an Iterator pointing to the address TLV to erase.
   * \return an iterator pointing to the next address TLV in the block.
   */
  TlvIterator TlvErase (TlvIterator position);

  /**
   * \brief Removes all address TLVs from [first, last) (includes first, not
   *        includes last).
   * \param first an Iterator pointing to the first address TLV to erase
   *        (inclusive).
   * \param last an Iterator pointing to the element past the last address TLV
   *        to erase.
   * \return an iterator pointing to the next address TLV in the message.
   */
  TlvIterator TlvErase (TlvIterator first, TlvIterator last);

  /**
   * \brief Removes all address TLVs from this block.
   */
  void TlvClear (void);

  /**
   * \return The size (in bytes) needed to serialize this address block.
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serializes this address block into the specified buffer.
   * \param start a reference to the point in a buffer to begin serializing.
   *
   * Users should not need to call this.  Blocks will be deserialized by their
   * containing packet.
   */
  void Serialize (Buffer::Iterator &start) const;

  /**
   * \brief Deserializes an address block from the specified buffer.
   * \param start a reference to the point in a buffer to begin deserializing.
   *
   * Users should not need to call this.  Blocks will be deserialized by their
   * containing packet.
   */
  void Deserialize (Buffer::Iterator &start);

  /**
   * \brief Pretty-prints the contents of this address block.
   * \param os a stream object to print to.
   */
  void Print (std::ostream &os) const;

  /**
   * \brief Pretty-prints the contents of this address block, with specified
   *        indentation.
   * \param os a stream object to print to.
   * \param level level of indentation.
   *
   * This probably never needs to be called by users.  This is used when
   * recursively printing sub-objects.
   */
  void Print (std::ostream &os, int level) const;

  /**
   * \brief Equality operator for PbbAddressBlock
   * \param other PbbAddressBlock to compare to this one
   * \returns true if PbbMessages are equal
   */
  bool operator== (const PbbAddressBlock &other) const;
  
  /**
   * \brief Inequality operator for PbbAddressBlock
   * \param other PbbAddressBlock to compare to this one
   * \returns true if PbbAddressBlock are not equal
   */  
  bool operator!= (const PbbAddressBlock &other) const;

protected:
  /**
   * \brief Returns address length
   * \returns Address length
   */
  virtual uint8_t GetAddressLength (void) const = 0;
  /**
   * \brief Serialize one or more addresses
   * \param buffer the buffer to serialize to
   * \param iter the iterator to the addresses
   */
  virtual void SerializeAddress (uint8_t *buffer, ConstAddressIterator iter) const = 0;
  /**
   * \brief Deserialize one address
   * \param buffer the buffer to deserialize from
   * \returns the address
   */
  virtual Address DeserializeAddress (uint8_t *buffer) const = 0;
  /**
   * \brief Print one or more addresses
   * \param os the output stream
   * \param iter the iterator to the addresses
   */
  virtual void PrintAddress (std::ostream &os, ConstAddressIterator iter) const = 0;

private:
  /**
   * \brief Get the prefix flags
   * \return the prefix flags
   */
  uint8_t GetPrefixFlags (void) const;
  /**
   * \brief Get head and tail
   * \param head the head
   * \param headlen the head length
   * \param tail the tail
   * \param taillen the tail length
   */
  void GetHeadTail (uint8_t *head, uint8_t &headlen,
                    uint8_t *tail, uint8_t &taillen) const;

  /**
   * \brief Check if the tail is empty
   * \param tail the tail
   * \param taillen the tail length
   * \returns true if the tail is empty
   */
  bool HasZeroTail (const uint8_t *tail, uint8_t taillen) const;

  std::list<Address> m_addressList; //!< Addresses container
  std::list<uint8_t> m_prefixList; //!< Prefixes container
  PbbAddressTlvBlock m_addressTlvList; //!< PbbAddressTlv container
};

/**
 * \brief Concrete IPv4 specific PbbAddressBlock.
 *
 * This address block will only contain IPv4 addresses.
 */
class PbbAddressBlockIpv4 : public PbbAddressBlock
{
public:
  PbbAddressBlockIpv4 ();
  virtual ~PbbAddressBlockIpv4 ();

protected:
  /**
   * \brief Returns address length
   * \returns Address length
   */
  virtual uint8_t GetAddressLength (void) const;
  virtual void SerializeAddress (uint8_t *buffer, ConstAddressIterator iter) const;
  virtual Address DeserializeAddress (uint8_t *buffer) const;
  virtual void PrintAddress (std::ostream &os, ConstAddressIterator iter) const;
};

/**
 * \brief Concrete IPv6 specific PbbAddressBlock.
 *
 * This address block will only contain IPv6 addresses.
 */
class PbbAddressBlockIpv6 : public PbbAddressBlock
{
public:
  PbbAddressBlockIpv6 ();
  virtual ~PbbAddressBlockIpv6 ();

protected:
  /**
   * \brief Returns address length
   * \returns Address length
   */
  virtual uint8_t GetAddressLength (void) const;
  virtual void SerializeAddress (uint8_t *buffer, ConstAddressIterator iter) const;
  virtual Address DeserializeAddress (uint8_t *buffer) const;
  virtual void PrintAddress (std::ostream &os, ConstAddressIterator iter) const;
};

/**
 * \brief A packet or message TLV
 */
class PbbTlv : public SimpleRefCount<PbbTlv>
{
public:
  PbbTlv (void);
  virtual ~PbbTlv (void);

  /**
   * \brief Sets the type of this TLV.
   * \param type the type value to set.
   */
  void SetType (uint8_t type);

  /**
   * \return the type of this TLV.
   */
  uint8_t GetType (void) const;

  /**
   * \brief Sets the type extension of this TLV.
   * \param type the type extension value to set.
   *
   * The type extension is like a sub-type used to further distinguish between
   * TLVs of the same type.
   */
  void SetTypeExt (uint8_t type);

  /**
   * \return the type extension for this TLV.
   *
   * Calling this while HasTypeExt is False is undefined.  Make sure you check
   * it first.  This will be checked by an assert in debug builds.
   */
  uint8_t GetTypeExt (void) const;

  /**
   * \brief Tests whether or not this TLV has a type extension.
   * \return true if this TLV has a type extension, false otherwise.
   *
   * This should be called before calling GetTypeExt to make sure there
   * actually is one.
   */
  bool HasTypeExt (void) const;

  /**
   * \brief Sets the value of this message to the specified buffer.
   * \param start a buffer instance.
   *
   * The buffer is _not_ copied until this TLV is serialized.  You should not
   * change the contents of the buffer you pass in to this function.
   */
  void SetValue (Buffer start);

  /**
   * \brief Sets the value of this message to a buffer with the specified data.
   * \param buffer a pointer to data to put in the TLVs buffer.
   * \param size the size of the buffer.
   *
   * The buffer *is copied* into a *new buffer instance*.  You can free the
   * data in the buffer provided anytime you wish.
   */
  void SetValue (const uint8_t * buffer, uint32_t size);

  /**
   * \return a Buffer pointing to the value of this TLV.
   *
   * Calling this while HasValue is False is undefined.  Make sure you check it
   * first.  This will be checked by an assert in debug builds.
   */
  Buffer GetValue (void) const;

  /**
   * \brief Tests whether or not this TLV has a value.
   * \return true if this tlv has a TLV, false otherwise.
   *
   * This should be called before calling GetTypeExt to make sure there
   * actually is one.
   */
  bool HasValue (void) const;

  /**
   * \return The size (in bytes) needed to serialize this TLV.
   */
  uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serializes this TLV into the specified buffer.
   * \param start a reference to the point in a buffer to begin serializing.
   *
   * Users should not need to call this.  TLVs will be serialized by their
   * containing blocks.
   */
  void Serialize (Buffer::Iterator &start) const;

  /**
   * \brief Deserializes a TLV from the specified buffer.
   * \param start a reference to the point in a buffer to begin deserializing.
   *
   * Users should not need to call this.  TLVs will be deserialized by their
   * containing blocks.
   */
  void Deserialize (Buffer::Iterator &start);

  /**
   * \brief Pretty-prints the contents of this TLV.
   * \param os a stream object to print to.
   */
  void Print (std::ostream &os) const;

  /**
   * \brief Pretty-prints the contents of this TLV, with specified indentation.
   * \param os a stream object to print to.
   * \param level level of indentation.
   *
   * This probably never needs to be called by users.  This is used when
   * recursively printing sub-objects.
   */
  void Print (std::ostream &os, int level) const;

  /**
   * \brief Equality operator for PbbTlv
   * \param other PbbTlv to compare to this one
   * \returns true if PbbTlv are equal
   */
  bool operator== (const PbbTlv &other) const;

  /**
   * \brief Inequality operator for PbbTlv
   * \param other PbbTlv to compare to this one
   * \returns true if PbbTlv are not equal
   */
  bool operator!= (const PbbTlv &other) const;

protected:
  /**
   * \brief Set an index as starting point
   * \param index the starting index
   */
  void SetIndexStart (uint8_t index);
  /**
   * \brief Get the starting point index
   * \returns the starting index
   */
  uint8_t GetIndexStart (void) const;
  /**
   * \brief Checks if there is a starting index
   * \returns true if the start index has been set
   */
  bool HasIndexStart (void) const;

  /**
   * \brief Set an index as stop point
   * \param index the stop index
   */
  void SetIndexStop (uint8_t index);
  /**
   * \brief Get the stop point index
   * \returns the stop index
   */
  uint8_t GetIndexStop (void) const;
  /**
   * \brief Checks if there is a stop index
   * \returns true if the stop index has been set
   */
  bool HasIndexStop (void) const;

  /**
   * \brief Set the multivalue parameter
   * \param isMultivalue the multivalue status
   */
  void SetMultivalue (bool isMultivalue);
  /**
   * \brief Check the multivalue parameter
   * \returns the multivalue status
   */
  bool IsMultivalue (void) const;

private:
  uint8_t m_type; //!< Type of this TLV.

  bool m_hasTypeExt; //!< Extended type present.
  uint8_t m_typeExt; //!< Extended type.

  bool m_hasIndexStart; //!< Start index present.
  uint8_t m_indexStart; //!< Start index.

  bool m_hasIndexStop; //!< Stop index present.
  uint8_t m_indexStop; //!< Stop index.

  bool m_isMultivalue; //!< Is multivalue.
  bool m_hasValue;     //!< Has value.
  Buffer m_value;      //!< Value.
};

/**
 * \brief An Address TLV
 */
class PbbAddressTlv : public PbbTlv
{
public:
  /**
   * \brief Sets the index of the first address in the associated address block
   * that this address TLV applies to.
   * \param index the index of the first address.
   */
  void SetIndexStart (uint8_t index);

  /**
   * \return the first (inclusive) index of the address in the corresponding
   * address block that this TLV applies to.
   *
   * Calling this while HasIndexStart is False is undefined.  Make sure you
   * check it first.  This will be checked by an assert in debug builds.
   */
  uint8_t GetIndexStart (void) const;

  /**
   * \brief Tests whether or not this address TLV has a start index.
   * \return true if this address TLV has a start index, false otherwise.
   *
   * This should be called before calling GetIndexStart to make sure there
   * actually is one.
   */
  bool HasIndexStart (void) const;

  /**
   * \brief Sets the index of the last address in the associated address block
   * that this address TLV applies to.
   * \param index the index of the last address.
   */
  void SetIndexStop (uint8_t index);

  /**
   * \return the last (inclusive) index of the address in the corresponding
   * PbbAddressBlock that this TLV applies to.
   *
   * Calling this while HasIndexStop is False is undefined.  Make sure you
   * check it first.  This will be checked by an assert in debug builds.
   */
  uint8_t GetIndexStop (void) const;

  /**
   * \brief Tests whether or not this address TLV has a stop index.
   * \return true if this address TLV has a stop index, false otherwise.
   *
   * This should be called before calling GetIndexStop to make sure there
   * actually is one.
   */
  bool HasIndexStop (void) const;

  /**
   * \brief Sets whether or not this address TLV is "multivalue"
   * \param isMultivalue whether or not this address TLV should be multivalue.
   *
   * If true, this means the value associated with this TLV should be divided
   * evenly into (GetIndexStop() - GetIndexStart() + 1) values.  Otherwise, the
   * value is one single value that applies to each address in the range.
   */
  void SetMultivalue (bool isMultivalue);

  /**
   * \brief Tests whether or not this address TLV is "multivalue"
   * \return whether this address TLV is multivalue or not.
   */
  bool IsMultivalue (void) const;
};

} /* namespace ns3 */

#endif /* PACKETBB_H */
