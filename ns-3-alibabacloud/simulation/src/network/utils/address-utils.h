/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
#ifndef ADDRESS_UTILS_H
#define ADDRESS_UTILS_H

#include "ns3/buffer.h"
#include "ipv4-address.h"
#include "ipv6-address.h"
#include "ns3/address.h"
#include "mac64-address.h"
#include "mac48-address.h"
#include "mac16-address.h"

namespace ns3 {

/**
 * \brief Write an Ipv4Address to a Buffer
 * \param i a reference to the buffer to write to
 * \param ad the Ipv4Address
 */
void WriteTo (Buffer::Iterator &i, Ipv4Address ad);

/**
 * \brief Write an Ipv4Address to a Buffer
 * \param i a reference to the buffer to write to
 * \param ad the Ipv6Address
 */
void WriteTo (Buffer::Iterator &i, Ipv6Address ad);

/**
 * \brief Write an Address to a Buffer
 * \param i a reference to the buffer to write to
 * \param ad the Address
 */
void WriteTo (Buffer::Iterator &i, const Address &ad);

/**
 * \brief Write an Mac64Address to a Buffer
 * \param i a reference to the buffer to write to
 * \param ad the Mac64Address
 */
void WriteTo (Buffer::Iterator &i, Mac64Address ad);

/**
 * \brief Write an Mac48Address to a Buffer
 * \param i a reference to the buffer to write to
 * \param ad the Mac48Address
 */
void WriteTo (Buffer::Iterator &i, Mac48Address ad);

/**
 * \brief Write an Mac16Address to a Buffer
 * \param i a reference to the buffer to write to
 * \param ad the Mac16Address
 */
void WriteTo (Buffer::Iterator &i, Mac16Address ad);


/**
 * \brief Read an Ipv4Address from a Buffer
 * \param i a reference to the buffer to read from
 * \param ad a reference to the Ipv4Address to be read
 */
void ReadFrom (Buffer::Iterator &i, Ipv4Address &ad);

/**
 * \brief Read an Ipv6Address from a Buffer
 * \param i a reference to the buffer to read from
 * \param ad a reference to the Ipv6Address to be read
 */
void ReadFrom (Buffer::Iterator &i, Ipv6Address &ad);

/**
 * \brief Read an Address from a Buffer
 * \param i a reference to the buffer to read from
 * \param ad a reference to the Address to be read
 * \param len the length of the Address
 */
void ReadFrom (Buffer::Iterator &i, Address &ad, uint32_t len);

/**
 * \brief Read a Mac64Address from a Buffer
 * \param i a reference to the buffer to read from
 * \param ad a reference to the Mac64Address to be read
 */
void ReadFrom (Buffer::Iterator &i, Mac64Address &ad);

/**
 * \brief Read a Mac48Address from a Buffer
 * \param i a reference to the buffer to read from
 * \param ad a reference to the Mac48Address to be read
 */
void ReadFrom (Buffer::Iterator &i, Mac48Address &ad);

/**
 * \brief Read a Mac16Address from a Buffer
 * \param i a reference to the buffer to read from
 * \param ad a reference to the Mac16Address to be read
 */
void ReadFrom (Buffer::Iterator &i, Mac16Address &ad);

namespace addressUtils {

/**
 * \brief Address family-independent test for a multicast address
 */
bool IsMulticast (const Address &ad);
};

};

#endif /* ADDRESS_UTILS_H */
