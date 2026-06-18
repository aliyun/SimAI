/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006,2007 INRIA
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

#ifndef TRACED_CALLBACK_H
#define TRACED_CALLBACK_H

#include <list>
#include "callback.h"

/**
 * \file
 * \ingroup tracing
 * ns3::TracedCallback declaration and template implementation.
 */

namespace ns3 {

/**
 * \ingroup tracing
 * \brief Forward calls to a chain of Callback
 *
 * A TracedCallback has almost exactly the same API as a normal
 * Callback but instead of forwarding calls to a single function
 * (as a Callback normally does), it forwards calls to a chain
 * of Callback.  Connect adds a Callback at the end of the chain
 * of callbacks.  Disconnect removes a Callback from the chain of callbacks.
 *
 * This is a functor: the chain of Callbacks is invoked by
 * calling the \c operator() form with the appropriate
 * number of arguments.
 *
 * \tparam Ts \explicit Types of the functor arguments.
 */
template<typename... Ts>
class TracedCallback
{
public:
  /** Constructor. */
  TracedCallback ();
  /**
   * Append a Callback to the chain (without a context).
   *
   * \param [in] callback Callback to add to chain.
   */
  void ConnectWithoutContext (const CallbackBase & callback);
  /**
   * Append a Callback to the chain with a context.
   *
   * The context string will be provided as the first argument
   * to the Callback.
   *
   * \param [in] callback Callback to add to chain.
   * \param [in] path Context string to provide when invoking the Callback.
   */
  void Connect (const CallbackBase & callback, std::string path);
  /**
   * Remove from the chain a Callback which was connected without a context.
   *
   * \param [in] callback Callback to remove from the chain.
   */
  void DisconnectWithoutContext (const CallbackBase & callback);
  /**
   * Remove from the chain a Callback which was connected with a context.
   *
   * \param [in] callback Callback to remove from the chain.
   * \param [in] path Context path which was used to connect the Callback.
   */
  void Disconnect (const CallbackBase & callback, std::string path);
  /**
   * \brief Functor which invokes the chain of Callbacks.
   * \tparam Ts \deduced Types of the functor arguments.
   * \param [in] args The arguments to the functor
   */
  void operator() (Ts... args) const;
  /**
   * \brief Checks if the Callbacks list is empty.
   * \return true if the Callbacks list is empty.
   */
  bool IsEmpty () const;

  /**
   *  TracedCallback signature for POD.
   *
   * \param [in] value Value of the traced variable.
   * @{
   */
  // Uint32Callback appears to be the only one used at the moment.
  // Feel free to add typedef's for any other POD you need.
  typedef void (* Uint32Callback)(const uint32_t value);
  /**@}*/

private:
  /**
   * Container type for holding the chain of Callbacks.
   *
   * \tparam Ts \deduced Types of the functor arguments.
   */
  typedef std::list<Callback<void,Ts...> > CallbackList;
  /** The chain of Callbacks. */
  CallbackList m_callbackList;
};

} // namespace ns3


/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

namespace ns3 {

template<typename... Ts>
TracedCallback<Ts...>::TracedCallback ()
  : m_callbackList ()
{}
template<typename... Ts>
void
TracedCallback<Ts...>::ConnectWithoutContext (const CallbackBase & callback)
{
  Callback<void,Ts...> cb;
  if (!cb.Assign (callback))
    {
      NS_FATAL_ERROR_NO_MSG ();
    }
  m_callbackList.push_back (cb);
}
template<typename... Ts>
void
TracedCallback<Ts...>::Connect (const CallbackBase & callback, std::string path)
{
  Callback<void,std::string,Ts...> cb;
  if (!cb.Assign (callback))
    {
      NS_FATAL_ERROR ("when connecting to " << path);
    }
  Callback<void,Ts...> realCb = cb.Bind (path);
  m_callbackList.push_back (realCb);
}
template<typename... Ts>
void
TracedCallback<Ts...>::DisconnectWithoutContext (const CallbackBase & callback)
{
  for (typename CallbackList::iterator i = m_callbackList.begin ();
       i != m_callbackList.end (); /* empty */)
    {
      if ((*i).IsEqual (callback))
        {
          i = m_callbackList.erase (i);
        }
      else
        {
          i++;
        }
    }
}
template<typename... Ts>
void
TracedCallback<Ts...>::Disconnect (const CallbackBase & callback, std::string path)
{
  Callback<void,std::string,Ts...> cb;
  if (!cb.Assign (callback))
    {
      NS_FATAL_ERROR ("when disconnecting from " << path);
    }
  Callback<void,Ts...> realCb = cb.Bind (path);
  DisconnectWithoutContext (realCb);
}
template<typename... Ts>
void
TracedCallback<Ts...>::operator() (Ts... args) const
{
  for (typename CallbackList::const_iterator i = m_callbackList.begin ();
       i != m_callbackList.end (); i++)
    {
      (*i)(args...);
    }
}

template <typename... Ts>
bool
TracedCallback<Ts...>::IsEmpty () const
{
  return m_callbackList.empty ();
}

} // namespace ns3

#endif /* TRACED_CALLBACK_H */
