/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Adrian Sai-wah Tam
 * Copyright (c) 2015 ResiliNets, ITTC, University of Kansas
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
 * Original Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 * Documentation, test cases: Truc Anh N. Nguyen   <annguyen@ittc.ku.edu>
 *                            ResiliNets Research Group   http://wiki.ittc.ku.edu/resilinets
 *                            The University of Kansas
 *                            James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 */

#ifndef TCP_OPTION_SACK_H
#define TCP_OPTION_SACK_H

#include "ns3/tcp-option.h"
#include "ns3/sequence-number.h"

namespace ns3 {

/**
 * \brief Defines the TCP option of kind 5 (selective acknowledgment option) as
 * in \RFC{2018}
 *
 * TCP SACK Option is used by a receiver to report non-contiguous blocks of data
 * that have been received and queued in the receiving buffer. Using the
 * information conveyed in SACK option sender retransmits only the segments that
 * have actually been lost, allowing the recovery of multiple packet losses per
 * sending window.
 *
 * Each SACK block is defined by two 32-bit unsigned integers specifying the
 * left and the right edge of the block. It means that with the 40-byte TCP
 * option limitation in addition to the presence of TCP Timestamp Option, the
 * maximum number of SACK blocks that can be appended to each segment is 3.
 */
class TcpOptionSack : public TcpOption
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  typedef std::pair<SequenceNumber32, SequenceNumber32> SackBlock; //!< SACK block definition
  typedef std::list<SackBlock> SackList;                           //!< SACK list definition

  TcpOptionSack ();
  virtual ~TcpOptionSack ();

  virtual void Print (std::ostream &os) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  virtual uint8_t GetKind (void) const;
  virtual uint32_t GetSerializedSize (void) const;

  /**
   * \brief Add a SACK block
   * \param s the SACK block to be added
   */
  void AddSackBlock (SackBlock s);

  /**
   * \brief Count the total number of SACK blocks
   * \return the total number of SACK blocks
   */
  uint32_t GetNumSackBlocks (void) const;

  /**
   * \brief Clear the SACK list
   */
  void ClearSackList (void);

  /**
   * \brief Get the SACK list
   * \return the SACK list
   */
  SackList GetSackList (void) const;

  friend std::ostream & operator<< (std::ostream & os, TcpOptionSack const & sackOption);

protected:
  SackList m_sackList; //!< the list of SACK blocks
};

/**
 * \brief Output operator.
 * \param os The output stream.
 * \param sackOption the option to print.
 * \returns The output stream.
 */
std::ostream & operator<< (std::ostream & os,
                           TcpOptionSack const & sackOption);

/**
 * \brief Output operator.
 * \param os The output stream.
 * \param sackBlock the block to print.
 * \returns The output stream.
 */
std::ostream & operator<< (std::ostream & os,
                           TcpOptionSack::SackBlock const & sackBlock);

} // namespace ns3

#endif /* TCP_OPTION_SACK */
