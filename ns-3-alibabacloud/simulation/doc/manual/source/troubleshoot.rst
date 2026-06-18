.. include:: replace.txt
.. highlight:: bash

Troubleshooting
---------------

This chapter posts some information about possibly common errors in building
or running |ns3| programs.

Please note that the wiki
(`<http://www.nsnam.org/wiki/Troubleshooting>`_) may have contributed
items.

Build errors
************

Run-time errors
***************

Sometimes, errors can occur with a program after a successful build. These are
run-time errors, and can commonly occur when memory is corrupted or pointer
values are unexpectedly null.

Here is an example of what might occur::

    $ ./ns3 run tcp-point-to-point
    Entering directory '/home/tomh/ns-3-nsc/build'
    Compilation finished successfully 
    Command ['/home/tomh/ns-3-nsc/build/debug/examples/tcp-point-to-point'] exited with code -11 

The error message says that the program terminated unsuccessfully, but it is
not clear from this information what might be wrong. To examine more
closely, try running it under the `gdb debugger
<http://sources.redhat.com/gdb/>`_:

.. sourcecode:: bash

    $ ./ns3 run tcp-point-to-point --gdb
    Entering directory '/home/tomh/ns-3-nsc/build'
    Compilation finished successfully 
    GNU gdb Red Hat Linux (6.3.0.0-1.134.fc5rh)
    Copyright 2004 Free Software Foundation, Inc.
    GDB is free software, covered by the GNU General Public License, and you are
    welcome to change it and/or distribute copies of it under certain conditions.
    Type "show copying" to see the conditions.
    There is absolutely no warranty for GDB.  Type "show warranty" for details.
    This GDB was configured as "i386-redhat-linux-gnu"...Using host libthread_db 
    library "/lib/libthread_db.so.1".

    (gdb) run
    Starting program: /home/tomh/ns-3-nsc/build/debug/examples/tcp-point-to-point 
    Reading symbols from shared object read from target memory...done.
    Loaded system supplied DSO at 0xf5c000

    Program received signal SIGSEGV, Segmentation fault.
    0x0804aa12 in main (argc=1, argv=0xbfdfefa4)
        at ../examples/tcp-point-to-point.cc:136
    136       Ptr<Socket> localSocket = socketFactory->CreateSocket ();
    (gdb) p localSocket
    $1 = {m_ptr = 0x3c5d65}
    (gdb) p socketFactory
    $2 = {m_ptr = 0x0}
    (gdb) quit
    The program is running.  Exit anyway? (y or n) y

Note first the way the program was invoked-- pass the command to run as an
argument to the command template "gdb %s".  

This tells us that there was an attempt to dereference a null pointer
socketFactory.

Let's look around line 136 of tcp-point-to-point, as gdb suggests:

.. sourcecode:: cpp

  Ptr<SocketFactory> socketFactory = n2->GetObject<SocketFactory> (Tcp::iid);
  Ptr<Socket> localSocket = socketFactory->CreateSocket ();
  localSocket->Bind ();

The culprit here is that the return value of GetObject is not being checked and
may be null.  

Sometimes you may need to use the `valgrind memory checker
<http://valgrind.org>`_ for more subtle errors. Again, you invoke the use of
valgrind similarly::

    $ ./ns3 run tcp-point-to-point --valgrind
