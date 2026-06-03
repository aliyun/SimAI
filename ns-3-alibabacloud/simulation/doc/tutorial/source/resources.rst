.. include:: replace.txt

Resources
---------

The Web
*******

There are several important resources of which any |ns3| user must be
aware.  The main web site is located at https://www.nsnam.org and 
provides access to basic information about the |ns3| system.  Detailed 
documentation is available through the main web site at
https://www.nsnam.org/documentation/.  You can also find documents 
relating to the system architecture from this page.

There is a Wiki that complements the main |ns3| web site which you will
find at https://www.nsnam.org/wiki/.  You will find user and developer 
FAQs there, as well as troubleshooting guides, third-party contributed code, 
papers, etc. 

The source code may be found and browsed at GitLab.com: 
https://gitlab.com/nsnam/.
There you will find the current development tree in the repository named
``ns-3-dev``. Past releases and experimental repositories of the core
developers may also be found at the project's old Mercurial site at
http://code.nsnam.org.

Git
***

Complex software systems need some way to manage the organization and 
changes to the underlying code and documentation.  There are many ways to
perform this feat, and you may have heard of some of the systems that are
currently used to do this.  Until recently, the |ns3| project used Mercurial
as its source code management system, but in December 2018, switch to
using Git.  Although you do not need to know much about Git in order to
complete this tutorial, we recommend becoming familiar with Git and using it 
to access the source code.  GitLab.com provides resources to get started
at: https://docs.gitlab.com/ee/gitlab-basics/.

CMake
*****

Once you have source code downloaded to your local system, you will need 
to compile that source to produce usable programs.  Just as in the case of
source code management, there are many tools available to perform this 
function.  Probably the most well known of these tools is ``make``.  Along
with being the most well known, ``make`` is probably the most difficult to
use in a very large and highly configurable system.  Because of this, many
alternatives have been developed.

The build system CMake is used on the |ns3| project.

For those interested in the details of CMake, the CMake documents are available
at https://cmake.org/cmake/help/latest/index.html and the current code
at https://gitlab.kitware.com/cmake/cmake.

Development Environment
***********************

As mentioned above, scripting in |ns3| is done in C++ or Python.
Most of the |ns3| API is available in Python, but the 
models are written in C++ in either case.  A working 
knowledge of C++ and object-oriented concepts is assumed in this document.
We will take some time to review some of the more advanced concepts or 
possibly unfamiliar language features, idioms and design patterns as they 
appear.  We don't want this tutorial to devolve into a C++ tutorial, though,
so we do expect a basic command of the language.  There are a wide
number of sources of information on C++ available on the web or
in print.

If you are new to C++, you may want to find a tutorial- or cookbook-based
book or web site and work through at least the basic features of the language
before proceeding.  For instance, `this tutorial
<http://www.cplusplus.com/doc/tutorial/>`_.

On Linux, the |ns3| system uses several components of the GNU "toolchain" 
for development.  A 
software toolchain is the set of programming tools available in the given 
environment. For a quick review of what is included in the GNU toolchain see,
http://en.wikipedia.org/wiki/GNU_toolchain.  |ns3| uses gcc, 
GNU binutils, and gdb.  However, we do not use the GNU build system tools, 
neither make directly.  We use CMake for these functions.

On macOS, the toolchain used is Xcode.  |ns3| users on a Mac are strongly
encouraged to install Xcode and the command-line tools packages from the
Apple App Store, and to look at the |ns3| installation wiki for more
information (https://www.nsnam.org/wiki/Installation).

Typically an |ns3| author will work in Linux or a Unix-like environment.  
For those running under Windows, there do exist environments 
which simulate the Linux environment to various degrees.  The |ns3| 
project has in the past (but not presently) supported development in the Cygwin environment for 
these users.  See http://www.cygwin.com/ 
for details on downloading, and visit the |ns3| wiki for more information
about Cygwin and |ns3|.  MinGW is presently not officially supported.
Another alternative to Cygwin is to install a virtual machine environment
such as VMware server and install a Linux virtual machine.

Socket Programming
******************

We will assume a basic facility with the Berkeley Sockets API in the examples
used in this tutorial.  If you are new to sockets, we recommend reviewing the
API and some common usage cases.  For a good overview of programming TCP/IP
sockets we recommend `TCP/IP Sockets in C, Donahoo and Calvert
<https://www.elsevier.com/books/tcp-ip-sockets-in-c/donahoo/978-0-12-374540-8>`_.

There is an associated web site that includes source for the examples in the
book, which you can find at:
http://cs.baylor.edu/~donahoo/practical/CSockets/.

If you understand the first four chapters of the book (or for those who do
not have access to a copy of the book, the echo clients and servers shown in 
the website above) you will be in good shape to understand the tutorial.
There is a similar book on Multicast Sockets,
`Multicast Sockets, Makofske and Almeroth
<https://www.elsevier.com/books/multicast-sockets/makofske/978-1-55860-846-7>`_.
that covers material you may need to understand if you look at the multicast 
examples in the distribution.
