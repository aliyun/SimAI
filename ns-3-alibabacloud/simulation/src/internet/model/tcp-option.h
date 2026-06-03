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

#ifndef TCP_OPTION_H
#define TCP_OPTION_H

#include <stdint.h>
#include "ns3/object.h"
#include "ns3/buffer.h"
#include "ns3/object-factory.h"

namespace ns3 {

/**
 * \ingroup tcp
 *
 * Base class for all kinds of TCP options
 */
class TcpOption : public Object
{
public:
  TcpOption ();
  virtual ~TcpOption ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * The option Kind, as defined in the respective RFCs.
   */
  enum Kind
  {
    // Remember to extend IsKindKnown() with new value, when adding values here
    //
    END = 0,                    //!< END
    NOP = 1,                    //!< NOP
    MSS = 2,                    //!< MSS
    WINSCALE = 3,               //!< WINSCALE
    SACKPERMITTED = 4,          //!< SACKPERMITTED
    SACK = 5,                   //!< SACK
    TS = 8,                     //!< TS
    UNKNOWN = 255               //!< not a standardized value; for unknown recv'd options
  };

  /**
   * \brief Print the Option contents
   * \param os the output stream
   */
  virtual void Print (std::ostream &os) const = 0;
  /**
   * \brief Serialize the Option to a buffer iterator
   * \param start the buffer iterator
   */
  virtual void Serialize (Buffer::Iterator start) const = 0;

  /**
   * \brief Deserialize the Option from a buffer iterator
   * \param start the buffer iterator
   * \returns the number of deserialized bytes
   */
  virtual uint32_t Deserialize (Buffer::Iterator start) = 0;

  /**
   * \brief Get the `kind' (as in \RFC{793}) of this option
   * \return the Option Kind
   */
  virtual uint8_t GetKind (void) const = 0;
  /**
   * \brief Returns number of bytes required for Option
   * serialization.
   *
   * \returns number of bytes required for Option
   * serialization
   */
  virtual uint32_t GetSerializedSize (void) const = 0;

  /**
   * \brief Creates an option
   * \param kind the option kind
   * \return the requested option or an ns3::UnknownOption if the option is not supported
   */
  static Ptr<TcpOption> CreateOption (uint8_t kind);

  /**
   * \brief Check if the option is implemented
   * \param kind the Option kind
   * \return true if the option is known
   */
  static bool IsKindKnown (uint8_t kind);
};

/**
 * \ingroup tcp
 *
 * \brief An unknown TCP option.
 *
 * An unknown option can be deserialized and (only if deserialized previously)
 * serialized again.
 */
class TcpOptionUnknown : public TcpOption
{
public:
  TcpOptionUnknown ();
  virtual ~TcpOptionUnknown ();

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

private:
  uint8_t m_kind; //!< The unknown option kind
  uint32_t m_size; //!< The unknown option size
  uint8_t m_content[40]; //!< The option data

};

} // namespace ns3

#endif /* TCP_OPTION */
