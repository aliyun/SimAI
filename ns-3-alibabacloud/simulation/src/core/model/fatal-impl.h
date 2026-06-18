/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 NICTA
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
 * Author: Quincy Tse <quincy.tse@nicta.com.au>
 */

#ifndef FATAL_IMPL_H
#define FATAL_IMPL_H

#include <ostream>

/**
 * \file
 * \ingroup fatalimpl
 * ns3::FatalImpl::RegisterStream(), ns3::FatalImpl::UnregisterStream(),
 * and ns3::FatalImpl::FlushStreams() declarations.
 */

/**
 * \ingroup fatal
 * \defgroup fatalimpl Fatal Implementation.
 */

namespace ns3 {

/**
 * \ingroup fatalimpl
 * \brief Implementation namespace for fatal error handlers.
 */
namespace FatalImpl {

/**
 * \ingroup fatalimpl
 *
 * \brief Register a stream to be flushed on abnormal exit.
 *
 * If a \c std::terminate() call is encountered after the
 * stream had been registered and before it has been
 * unregistered, \c stream->flush() will be called. Users of
 * this function should ensure the stream remains valid until
 * it had been unregistered.
 *
 * \param [in] stream The stream to be flushed on abnormal exit.
 */
void RegisterStream (std::ostream* stream);

/**
 * \ingroup fatalimpl
 *
 * \brief Unregister a stream for flushing on abnormal exit.
 *
 * After a stream had been unregistered, \c stream->flush()
 * will no longer be called should abnormal termination be
 * encountered.
 *
 * If the stream is not registered, nothing will happen.
 *
 * \param [in] stream The stream to be unregistered.
 */
void UnregisterStream (std::ostream* stream);

/**
 * \ingroup fatalimpl
 *
 * \brief Flush all currently registered streams.
 *
 * This function iterates through each registered stream and
 * unregisters them. The default \c SIGSEGV handler is overridden
 * when this function is being executed, and will be restored
 * when this function returns.
 *
 * If a \c SIGSEGV is encountered (most likely due to a bad \c ostream*
 * being registered, or a registered \c osteam* pointing to an
 * \c ostream that had already been destroyed), this function will
 * skip the bad \c ostream* and continue to flush the next stream.
 * The function will then terminate raising \c SIGIOT (aka \c SIGABRT)
 *
 * DO NOT call this function until the program is ready to crash.
 */
void FlushStreams (void);

} //FatalImpl
} //ns3

#endif
