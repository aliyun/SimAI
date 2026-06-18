/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 PIOTR JURKIEWICZ
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
 * Author: Piotr Jurkiewicz <piotr.jerzy.jurkiewicz@gmail.com>
 */
#ifndef CRC32_H
#define CRC32_H
#include <stdint.h>

namespace ns3 {

/**
 * Calculates the CRC-32 for a given input
 *
 * \param data buffer to calculate the checksum for
 * \param length the length of the buffer (bytes)
 * \returns the computed crc-32.
 *
 */
uint32_t CRC32Calculate (const uint8_t *data, int length);

} // namespace ns3

#endif
