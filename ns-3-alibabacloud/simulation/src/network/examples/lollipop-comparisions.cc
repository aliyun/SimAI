/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/lollipop-counter.h"

using namespace ns3;

int
main (int argc, char *argv[])
{
  LollipopCounter8 counterA;
  LollipopCounter8 counterB;

//  counterA.SetSequenceWindowSize (5);
//  counterB.SetSequenceWindowSize (5);

  for (uint8_t i = 0; i < std::numeric_limits<uint8_t>::max (); i++)
    {
      std::cout << +i << " -- ";
      for (uint8_t j = 0; j < std::numeric_limits<uint8_t>::max (); j++)
        {
          counterA = i;
          counterB = j;
          if (counterA < counterB)
            {
              std::cout << "<";
            }
          else if (counterA == counterB)
            {
              std::cout << "=";
            }
          else if (counterA > counterB)
            {
              std::cout << ">";
            }
          else
            {
              std::cout << ".";
            }
        }
      std::cout << std::endl;
    }
}
