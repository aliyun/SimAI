/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef WIFI_PROTECTION_H
#define WIFI_PROTECTION_H

#include "ns3/nstime.h"
#include "wifi-tx-vector.h"
#include "ctrl-headers.h"
#include <memory>


namespace ns3 {

/**
 * \ingroup wifi
 *
 * WifiProtection is an abstract base struct. Each derived struct defines a protection
 * method and stores the information needed to perform protection according to
 * that method.
 */
struct WifiProtection
{
  /**
   * \enum Method
   * \brief Available protection methods
   */
  enum Method
    {
      NONE = 0,
      RTS_CTS,
      CTS_TO_SELF
    };

  /**
   * Constructor.
   * \param m the protection method for this object
   */
  WifiProtection (Method m);
  virtual ~WifiProtection ();

  /**
   * Clone this object.
   * \return a pointer to the cloned object
   */
  virtual std::unique_ptr<WifiProtection> Copy (void) const = 0;

  /**
   * \brief Print the object contents.
   * \param os output stream in which the data should be printed.
   */
  virtual void Print (std::ostream &os) const = 0;

  const Method method;   //!< protection method
  Time protectionTime;   //!< time required by the protection method
};


/**
 * \ingroup wifi
 *
 * WifiNoProtection specifies that no protection method is used.
 */
struct WifiNoProtection : public WifiProtection
{
  WifiNoProtection ();

  std::unique_ptr<WifiProtection> Copy (void) const override;
  void Print (std::ostream &os) const override;
};


/**
 * \ingroup wifi
 *
 * WifiRtsCtsProtection specifies that RTS/CTS protection method is used.
 */
struct WifiRtsCtsProtection : public WifiProtection
{
  WifiRtsCtsProtection ();

  std::unique_ptr<WifiProtection> Copy (void) const override;
  void Print (std::ostream &os) const override;

  WifiTxVector rtsTxVector;       //!< RTS TXVECTOR
  WifiTxVector ctsTxVector;       //!< CTS TXVECTOR
};


/**
 * \ingroup wifi
 *
 * WifiCtsToSelfProtection specifies that CTS-to-self protection method is used.
 */
struct WifiCtsToSelfProtection : public WifiProtection
{
  WifiCtsToSelfProtection ();

  std::unique_ptr<WifiProtection> Copy (void) const override;
  void Print (std::ostream &os) const override;

  WifiTxVector ctsTxVector;       //!< CTS TXVECTOR
};


/**
 * \brief Stream insertion operator.
 *
 * \param os the output stream
 * \param protection the protection method
 * \returns a reference to the stream
 */
std::ostream& operator<< (std::ostream& os, const WifiProtection* protection);

} //namespace ns3

#endif /* WIFI_PROTECTION_H */
