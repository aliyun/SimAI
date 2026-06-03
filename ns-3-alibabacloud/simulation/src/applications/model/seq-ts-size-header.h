/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *   Copyright (c) 2009 INRIA
 *   Copyright (c) 2018 Natale Patriciello <natale.patriciello@gmail.com>
 *                      (added timestamp and size fields)
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License version 2 as
 *   published by the Free Software Foundation;
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef SEQ_TS_SIZE_HEADER_H
#define SEQ_TS_SIZE_HEADER_H

#include <ns3/seq-ts-header.h>

namespace ns3 {
/**
 * \ingroup applications
 * \brief Header with a sequence, a timestamp, and a "size" attribute
 *
 * This header adds a size attribute to the sequence number and timestamp
 * of class \c SeqTsHeader.  The size attribute can be used to track
 * application data units for stream-based sockets such as TCP.
 *
 * \sa ns3::SeqTsHeader
 */
class SeqTsSizeHeader : public SeqTsHeader
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief constructor
   */
  SeqTsSizeHeader ();

  /**
   * \brief Set the size information that the header will carry
   * \param size the size
   */
  void SetSize (uint64_t size);

  /**
   * \brief Get the size information that the header is carrying
   * \return the size
   */
  uint64_t GetSize (void) const;

  // Inherited
  virtual TypeId GetInstanceTypeId (void) const override;
  virtual void Print (std::ostream &os) const override;
  virtual uint32_t GetSerializedSize (void) const override;
  virtual void Serialize (Buffer::Iterator start) const override;
  virtual uint32_t Deserialize (Buffer::Iterator start) override;

private:
  uint64_t m_size {0}; //!< The 'size' information that the header is carrying
};

} // namespace ns3

#endif /* SEQ_TS_SIZE_HEADER */
