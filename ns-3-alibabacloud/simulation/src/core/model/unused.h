/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Andrey Mazo
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
 * Author: Andrey Mazo <ahippo@yandex.com>
 */

#ifndef UNUSED_H
#define UNUSED_H

#include "ns3/deprecated.h"

/**
 * \file
 * \ingroup core
 * NS_UNUSED and NS_UNUSED_GLOBAL macro definitions.
 */


/**
 * \ingroup core
 * \def NS_UNUSED()
 * Mark a local variable as unused.
 *
 * \deprecated Please use `[[maybe_unused]]` directly. This macro is being
 * kept temporarily to support older code.
 */

// We can't use NS_DEPRECATED_3_36 here because NS_UNUSED
// is used as a statement following a declaration:
//   int x;
//   NS_UNUSED (x);
// NS_DEPRECATED needs to be applied to the declaration
//   NS_DEPRECATED_3_36 ("use double instead") int x;
// Instead we resort to a pragma

#ifndef NS_UNUSED
# define NS_UNUSED(x) \
  _Pragma ("GCC warning \"NS_UNUSED is deprecated, use [[maybe_unused]] directly\"") \
  ((void)(x))
#endif

/**
 * \ingroup core
 * \def NS_UNUSED_GLOBAL()
 * Mark a variable at file scope as unused.
 *
 * \deprecated Please use `[[maybe_unused]]` directly. This macro is being
 * kept temporarily to support older code.
 */
#ifndef NS_UNUSED_GLOBAL
#if defined(__GNUC__)
# define NS_UNUSED_GLOBAL(x) \
  NS_DEPRECATED_3_36 ("NS_UNUSED_GLOBAL is deprecated, use [[maybe_unused]] directly") \
  [[maybe_unused]] x
#elif defined(__LCLINT__)
# define NS_UNUSED_GLOBAL(x) \
  NS_DEPRECATED_3_36 ("NS_UNUSED_GLOBAL is deprecated, use [[maybe_unused]] directly") \
  /*@unused@*/ x
#else
# define NS_UNUSED_GLOBAL(x) \
  NS_DEPRECATED_3_36 ("NS_UNUSED_GLOBAL is deprecated, use [[maybe_unused]] directly") \
  x
#endif
#endif

#endif /* UNUSED_H */
