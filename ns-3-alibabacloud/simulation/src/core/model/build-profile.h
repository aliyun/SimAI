/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 LLNL
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#ifndef NS3_BUILD_PROFILE_H
#define NS3_BUILD_PROFILE_H

/**
 * \file
 * \ingroup debugging
 * NS_BUILD_DEBUG, NS_BUILD_RELEASE, and NS_BUILD_OPTIMIZED
 * macro definitions.
 */

/**
 * \ingroup debugging
 * Build profile no-op macro.
 * \param [in] code The code to skip.
 */
#define NS_BUILD_PROFILE_NOOP(code)            \
  do if (false)                                \
    {                                          \
      code;                                    \
    } while (false)

/**
 * \ingroup debugging
 * Build profile macro to execute a code snippet.
 * \param [in] code The code to execute.
 */
#define NS_BUILD_PROFILE_OP(code) \
  do                                            \
    {                                           \
      code;                                     \
    }  while (false)


#ifdef NS3_BUILD_PROFILE_DEBUG
/**
 * \ingroup debugging
 * Execute a code snippet in debug builds.
 * \param [in] code The code to execute.
 */
#define NS_BUILD_DEBUG(code)     NS_BUILD_PROFILE_OP (code)
#else
#define NS_BUILD_DEBUG(code)     NS_BUILD_PROFILE_NOOP (code)
#endif

#ifdef NS3_BUILD_PROFILE_RELEASE
/**
 * \ingroup debugging
 * Execute a code snippet in release builds.
 * \param [in] code The code to execute.
 */
#define NS_BUILD_RELEASE(code)   NS_BUILD_PROFILE_OP (code)
#else
#define NS_BUILD_RELEASE(code)   NS_BUILD_PROFILE_NOOP (code)
#endif

#ifdef NS3_BUILD_PROFILE_OPTIMIZED
/**
 * \ingroup debugging
 * Execute a code snippet in optimized builds.
 * \param [in] code The code to execute.
 */
#define NS_BUILD_OPTIMIZED(code) NS_BUILD_PROFILE_OP (code)
#else
#define NS_BUILD_OPTIMIZED(code) NS_BUILD_PROFILE_NOOP (code)
#endif




#endif /* NS3_BUILD_PROFILE_H */
