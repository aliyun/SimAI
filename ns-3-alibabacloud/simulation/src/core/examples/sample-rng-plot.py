# -*- Mode:Python; -*-
# /*
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License version 2 as
#  * published by the Free Software Foundation
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#  */

## @file
#  @ingroup core-examples
#  @ingroup randomvariable
#  Demonstrate use of ns-3 as a random number generator integrated with
#  plotting tools.
#  
#  This is adapted from Gustavo Carneiro's ns-3 tutorial


import numpy as np
import matplotlib.pyplot as plt
import ns.core

# mu, var = 100, 225

## Random number generator.
rng = ns.core.NormalRandomVariable()
rng.SetAttribute("Mean", ns.core.DoubleValue(100.0))
rng.SetAttribute("Variance", ns.core.DoubleValue(225.0))
## Random number samples.
x = [rng.GetValue() for t in range(10000)]

# the histogram of the data

## Make a probability density histogram
density = 1
## Plot color
facecolor='g'
## Plot alpha value (transparency)
alpha=0.75

# We don't really need the plot results, we're just going to show it later.
# n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)

plt.title('ns-3 histogram')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
