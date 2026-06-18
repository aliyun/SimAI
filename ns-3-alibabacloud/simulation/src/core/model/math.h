/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 *
 * FreeBSD log2 patch from graphviz port to FreeBSD 9, courtesy of
 * Christoph Moench-Tegeder <cmt@burggraben.net>
 */

// It is recommended to include this header instead of <math.h> or
// <cmath> whenever the log2(x) function is needed.  See bug 1467.

#ifndef MATH_H
#define MATH_H

/**
 * \file
 * \ingroup core
 * log2() macro definition; to deal with \bugid{1467}.
 */

#include <cmath>

#ifdef __FreeBSD__

#if __FreeBSD_version <= 704101                                     \
  || (__FreeBSD_version >= 800000 && __FreeBSD_version < 802502)    \
  || (__FreeBSD_version >= 900000 && __FreeBSD_version < 900027)
#define log2(x) (std::log (x) / M_LN2)
#endif /* __FreeBSD_version */

#endif /* __FreeBSD__ */

#endif /* MATH_H */
