/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Adrian Sai-wah Tam
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
 * Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 */
#ifndef TCPOPTIONRFC793_H
#define TCPOPTIONRFC793_H

#include "ns3/tcp-option.h"

namespace ns3 {

/**
 * \ingroup tcp
 *
 * Defines the TCP option of kind 0 (end of option list) as in \RFC{793}
 */
class TcpOptionEnd : public TcpOption
{
public:
  TcpOptionEnd ();
  virtual ~TcpOptionEnd ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  virtual void Print (std::ostream &os) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  virtual uint8_t GetKind (void) const;
  virtual uint32_t GetSerializedSize (void) const;

};

/**
 * Defines the TCP option of kind 1 (no operation) as in \RFC{793}
 */
class TcpOptionNOP : public TcpOption
{
public:
  TcpOptionNOP ();
  virtual ~TcpOptionNOP ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  virtual void Print (std::ostream &os) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  virtual uint8_t GetKind (void) const;
  virtual uint32_t GetSerializedSize (void) const;
};

/**
 * Defines the TCP option of kind 2 (maximum segment size) as in \RFC{793}
 */
class TcpOptionMSS : public TcpOption
{
public:
  TcpOptionMSS ();
  virtual ~TcpOptionMSS ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;

  virtual void Print (std::ostream &os) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  virtual uint8_t GetKind (void) const;
  virtual uint32_t GetSerializedSize (void) const;

  /**
   * \brief Get the Maximum Segment Size stored in the Option
   * \return The Maximum Segment Size
   */
  uint16_t GetMSS (void) const;
  /**
   * \brief Set the Maximum Segment Size stored in the Option
   * \param mss The Maximum Segment Size
   */
  void SetMSS (uint16_t mss);

protected:
  uint16_t m_mss; //!< maximum segment size
};

} // namespace ns3

#endif // TCPOPTIONRFC793_H
