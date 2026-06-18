/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  This program is free software; you can redistribute it and/or modify
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
 * Authors: Faker Moatamri <faker.moatamri@sophia.inria.fr>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef GTK_CONFIG_STORE_H
#define GTK_CONFIG_STORE_H

namespace ns3 {

/**
 * \ingroup configstore
 * \brief A class that provides a GTK-based front end to ns3::ConfigStore 
 *
 */
class GtkConfigStore
{
public:
  GtkConfigStore ();

  /**
   * Process default values
   */
  void ConfigureDefaults (void);
  /**
   * Process attribute values
   */
  void ConfigureAttributes (void);
};


} // namespace ns3

#endif /* GTK_CONFIG_STORE_H */
