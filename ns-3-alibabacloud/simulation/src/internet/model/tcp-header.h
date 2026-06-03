/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Georgia Tech Research Corporation
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
 * Author: Raj Bhattacharjea <raj.b@gatech.edu>
 */

#ifndef TCP_HEADER_H
#define TCP_HEADER_H

#include <stdint.h>
#include "ns3/header.h"
#include "ns3/tcp-option.h"
#include "ns3/buffer.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/sequence-number.h"

namespace ns3 {

/**
 * \ingroup tcp
 * \brief Header for the Transmission Control Protocol
 *
 * This class has fields corresponding to those in a network TCP header
 * (port numbers, sequence and acknowledgement numbers, flags, etc) as well
 * as methods for serialization to and deserialization from a byte buffer.
 */

class TcpHeader : public Header
{
public:
  TcpHeader ();
  virtual ~TcpHeader ();

  typedef std::list< Ptr<const TcpOption> > TcpOptionList; //!< List of TcpOption

  /**
   * \brief Print a TCP header into an output stream
   *
   * \param os output stream
   * \param tc TCP header to print
   * \return The ostream passed as first argument
   */
  friend std::ostream& operator<< (std::ostream& os, TcpHeader const & tc);

  /**
   * \brief Converts an integer into a human readable list of Tcp flags
   *
   * \param flags Bitfield of TCP flags to convert to a readable string
   * \param delimiter String to insert between flags
   *
   * FIN=0x1, SYN=0x2, RST=0x4, PSH=0x8, ACK=0x10, URG=0x20, ECE=0x40, CWR=0x80
   * TcpHeader::FlagsToString (0x1) should return the following string;
   *     "FIN"
   *
   * TcpHeader::FlagsToString (0xff) should return the following string;
   *     "FIN|SYN|RST|PSH|ACK|URG|ECE|CWR";
   *
   * \return the generated string
   **/
  static std::string FlagsToString (uint8_t flags, const std::string& delimiter = "|");

  /**
   * \brief Enable checksum calculation for TCP
   *
   * \todo currently has no effect
   */
  void EnableChecksums (void);

//Setters

/**
 * \brief Set the source port
 * \param port The source port for this TcpHeader
 */
  void SetSourcePort (uint16_t port);

  /**
   * \brief Set the destination port
   * \param port the destination port for this TcpHeader
   */
  void SetDestinationPort (uint16_t port);

  /**
   * \brief Set the sequence Number
   * \param sequenceNumber the sequence number for this TcpHeader
   */
  void SetSequenceNumber (SequenceNumber32 sequenceNumber);

  /**
   * \brief Set the ACK number
   * \param ackNumber the ACK number for this TcpHeader
   */
  void SetAckNumber (SequenceNumber32 ackNumber);

  /**
   * \brief Set flags of the header
   * \param flags the flags for this TcpHeader
   */
  void SetFlags (uint8_t flags);

  /**
   * \brief Set the window size
   * \param windowSize the window size for this TcpHeader
   */
  void SetWindowSize (uint16_t windowSize);

  /**
   * \brief Set the urgent pointer
   * \param urgentPointer the urgent pointer for this TcpHeader
   */
  void SetUrgentPointer (uint16_t urgentPointer);

//Getters

  /**
   * \brief Get the source port
   * \return The source port for this TcpHeader
   */
  uint16_t GetSourcePort () const;

  /**
   * \brief Get the destination port
   * \return the destination port for this TcpHeader
   */
  uint16_t GetDestinationPort () const;

  /**
   * \brief Get the sequence number
   * \return the sequence number for this TcpHeader
   */
  SequenceNumber32 GetSequenceNumber () const;

  /**
   * \brief Get the ACK number
   * \return the ACK number for this TcpHeader
   */
  SequenceNumber32 GetAckNumber () const;

  /**
   * \brief Get the length in words
   *
   * A word is 4 bytes; without Tcp Options, header is 5 words (20 bytes).
   * With options, it can reach up to 15 words (60 bytes).
   *
   * \return the length of this TcpHeader
   */
  uint8_t GetLength () const;

  /**
   * \brief Get the flags
   * \return the flags for this TcpHeader
   */
  uint8_t GetFlags () const;

  /**
   * \brief Get the window size
   * \return the window size for this TcpHeader
   */
  uint16_t GetWindowSize () const;

  /**
   * \brief Get the urgent pointer
   * \return the urgent pointer for this TcpHeader
   */
  uint16_t GetUrgentPointer () const;

  /**
   * \brief Get the option specified
   * \param kind the option to retrieve
   * \return Whether the header contains a specific kind of option, or 0
   */
  Ptr<const TcpOption> GetOption (uint8_t kind) const;

  /**
   * \brief Get the list of option in this header
   * \return a const reference to the option list
   */
  const TcpOptionList& GetOptionList (void) const;

  /**
   * \brief Get the total length of appended options
   * \return the total length of options appended to this TcpHeader
   */
  uint8_t GetOptionLength () const;

  /**
   * \brief Get maximum option length
   * \return the maximum option length
   */
  uint8_t GetMaxOptionLength () const;

  /**
   * \brief Check if the header has the option specified
   * \param kind Option to check for
   * \return true if the header has the option, false otherwise
   */
  bool HasOption (uint8_t kind) const;

  /**
   * \brief Append an option to the TCP header
   * \param option The option to append
   * \return true if option has been appended, false otherwise
   */
  bool AppendOption (Ptr<const TcpOption> option);

  /**
   * \brief Initialize the TCP checksum.
   *
   * If you want to use tcp checksums, you should call this
   * method prior to adding the header to a packet.
   *
   * \param source the IP source to use in the underlying
   *        IP packet.
   * \param destination the IP destination to use in the
   *        underlying IP packet.
   * \param protocol the protocol number to use in the underlying
   *        IP packet.
   *
   */
  void InitializeChecksum (const Ipv4Address &source,
                           const Ipv4Address &destination,
                           uint8_t protocol);

  /**
   * \brief Initialize the TCP checksum.
   *
   * If you want to use tcp checksums, you should call this
   * method prior to adding the header to a packet.
   *
   * \param source the IP source to use in the underlying
   *        IP packet.
   * \param destination the IP destination to use in the
   *        underlying IP packet.
   * \param protocol the protocol number to use in the underlying
   *        IP packet.
   *
   */
  void InitializeChecksum (const Ipv6Address &source,
                           const Ipv6Address &destination,
                           uint8_t protocol);

  /**
   * \brief Initialize the TCP checksum.
   *
   * If you want to use tcp checksums, you should call this
   * method prior to adding the header to a packet.
   *
   * \param source the IP source to use in the underlying
   *        IP packet.
   * \param destination the IP destination to use in the
   *        underlying IP packet.
   * \param protocol the protocol number to use in the underlying
   *        IP packet.
   *
   */
  void InitializeChecksum (const Address &source,
                           const Address &destination,
                           uint8_t protocol);

  /**
   * \brief TCP flag field values
   */
  typedef enum
  {
    NONE = 0,   //!< No flags
    FIN  = 1,   //!< FIN
    SYN  = 2,   //!< SYN
    RST  = 4,   //!< Reset
    PSH  = 8,   //!< Push
    ACK  = 16,  //!< Ack
    URG  = 32,  //!< Urgent
    ECE  = 64,  //!< ECE
    CWR  = 128  //!< CWR
  } Flags_t;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  /**
   * \brief Is the TCP checksum correct ?
   * \returns true if the checksum is correct, false otherwise.
   */
  bool IsChecksumOk (void) const;

  /**
   * Comparison operator
   * \param lhs left operand
   * \param rhs right operand
   * \return true if the operands are equal
   */
  friend bool operator== (const TcpHeader &lhs, const TcpHeader &rhs);

private:
  /**
   * \brief Calculate the header checksum
   * \param size packet size
   * \returns the checksum
   */
  uint16_t CalculateHeaderChecksum (uint16_t size) const;

  /**
   * \brief Calculates the header length (in words)
   *
   * Given the standard size of the header, the method checks for options
   * and calculates the real length (in words).
   *
   * \return header length in 4-byte words
   */
  uint8_t CalculateHeaderLength () const;

  uint16_t m_sourcePort;        //!< Source port
  uint16_t m_destinationPort;   //!< Destination port
  SequenceNumber32 m_sequenceNumber;  //!< Sequence number
  SequenceNumber32 m_ackNumber;       //!< ACK number
  uint8_t m_length;             //!< Length (really a uint4_t) in words.
  uint8_t m_flags;              //!< Flags (really a uint6_t)
  uint16_t m_windowSize;        //!< Window size
  uint16_t m_urgentPointer;     //!< Urgent pointer

  Address m_source;       //!< Source IP address
  Address m_destination;  //!< Destination IP address
  uint8_t m_protocol;     //!< Protocol number

  bool m_calcChecksum;    //!< Flag to calculate checksum
  bool m_goodChecksum;    //!< Flag to indicate that checksum is correct

  static const uint8_t m_maxOptionsLen = 40;         //!< Maximum options length
  TcpOptionList m_options;     //!< TcpOption present in the header
  uint8_t m_optionsLen;        //!< Tcp options length.
};

} // namespace ns3

#endif /* TCP_HEADER */
