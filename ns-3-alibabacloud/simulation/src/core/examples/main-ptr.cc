/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
 */

#include "ns3/ptr.h"
#include "ns3/object.h"
#include "ns3/command-line.h"
#include <iostream>

/**
 * \file
 * \ingroup core-examples
 * \ingroup ptr
 * Example program illustrating use of the ns3::Ptr smart pointer.
 */

using namespace ns3;

/**
 * Example class illustrating use of Ptr.
 */
class PtrExample : public Object
{
public:
  /** Constructor. */
  PtrExample ();
  /** Destructor. */
  ~PtrExample ();
  /** Example class method. */
  void Method (void);
};
PtrExample::PtrExample ()
{
  std::cout << "PtrExample constructor" << std::endl;
}
PtrExample::~PtrExample ()
{
  std::cout << "PtrExample destructor" << std::endl;
}
void
PtrExample::Method (void)
{
  std::cout << "PtrExample method" << std::endl;
}


/**
 *  Example Ptr global variable.
 */
static Ptr<PtrExample> g_ptr = 0;

/**
 * Example Ptr manipulations.
 *
 * This function stores it's argument in the global variable \c g_ptr
 * and returns the old value of \c g_ptr.
 * \param [in] p A Ptr.
 * \returns The prior value of \c g_ptr.
 */
static Ptr<PtrExample>
StorePtr (Ptr<PtrExample> p)
{
  Ptr<PtrExample> prev = g_ptr;
  g_ptr = p;
  return prev;
}

/**
 *  Set \c g_ptr to NULL.
 */
static void
ClearPtr (void)
{
  g_ptr = 0;
}



int main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  {
    // Create a new object of type PtrExample, store it in global
    // variable g_ptr
    Ptr<PtrExample> p = CreateObject<PtrExample> ();
    p->Method ();
    Ptr<PtrExample> prev = StorePtr (p);
    NS_ASSERT (prev == 0);
  }

  {
    // Create a new object of type PtrExample, store it in global
    // variable g_ptr, get a hold on the previous PtrExample object.
    Ptr<PtrExample> p = CreateObject<PtrExample> ();
    Ptr<PtrExample> prev = StorePtr (p);
    // call method on object
    prev->Method ();
    // Clear the currently-stored object
    ClearPtr ();
    // get the raw pointer and release it.
    PtrExample *raw = GetPointer (prev);
    prev = 0;
    raw->Method ();
    raw->Unref ();
  }


  return 0;
}
