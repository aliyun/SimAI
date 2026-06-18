.. include:: replace.txt
.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

.. _Coding style:

Coding style
------------

When writing code to be contributed to the |ns3| open source project, we
ask that you follow the coding standards, guidelines, and recommendations
found below.

A lot of the syntactical rules described can be easily enforced with the
help of the ``utils/check-style.py`` script which relies on a working version
of `uncrustify <http://uncrustify.sourceforge.net/>`_.  We recommend to
run this script over your newly introduced C++ files prior to submission
as a Merge Request.
However, we ask that you avoid introducing whitespace changes to any
portions of existing files that do not pertain to your submission (even
though such portions of existing files may not be currently compliant
with the coding style).

To run ``check-style.py`` on a new file, the following command is
suggested:

::

  $ /path/to/utils/check-style.py -i -l 3 -f new-file.cc

The ``-i`` flag tells check-style.py to make modifications in-place.  The
``-l 3`` argument asks to apply the highest level of whitespace compliance
changes to the code.

Code layout
***********

The code layout follows the
`GNU coding standard <https://www.gnu.org/prep/standards/>`_ layout for C and
extends it to C++.  Do not use tabs for indentation.  Indentation spacing is
2 spaces (the default emacs C++ mode) as outlined below:

::

   Foo (void)
   {
     if (test)
       {
         // do stuff here
       }
     else
       {
         // do other stuff here
       }
   
     for (int i = 0; i < 100; i++)
       {
         // do loop
       }
   
     while (test)
       {
         // do while
       }
   
     do
       {
         // do stuff
       } while ();
   }

Each statement should be put on a separate line to increase readability, and
multi-statement blocks following conditional or looping statements are always
delimited by braces (single-statement blocks may be placed on the same line
of an ``if ()`` statement).  Each variable declaration is on a separate line.
Variables should be declared at the point in the code where they are needed,
and should be assigned an initial value at the time of declaration. Except
when used in a switch statement, the open and close braces ``{`` and ``}``
are always on a separate line. Do not use the C++ ``goto`` statement.

The layout of variables declared in a class may either be aligned with
the variable names or unaligned, as long as the file is internally consistent
and no tab characters are included. Examples:

::

  int varOne;
  double varTwo;   // OK (unaligned)
  int      varOne;
  double   varTwo;  // also OK (aligned)

Naming
******

Name encoding
=============

Function, method, and type names should follow the
`CamelCase <https://en.wikipedia.org/wiki/CamelCase>`_ convention: words are
joined without spaces and are capitalized. For example, "my computer" is
transformed into ``MyComputer``.  Do not use all capital letters such as
``MAC`` or ``PHY``, but choose instead ``Mac`` or ``Phy``. Do not use all
capital letters, even for acronyms such as ``EDCA``; use ``Edca`` instead.
This applies also to two-letter acronyms, such as ``IP`` (which becomes
``Ip``). The goal of the CamelCase convention is to ensure that the words
which make up a name can be separated by the eye: the initial Caps
fills that role. Use PascalCasing (CamelCase with first letter capitalized)
for function, property, event, and class names.

Variable names should follow a slight variation on the base CamelCase
convention: camelBack. For example, the variable ``user name`` would be named
``userName``. This variation on the basic naming pattern is used to allow a
reader to distinguish a variable name from its type. For example,
``UserName userName`` would be used to declare a variable named
``userName`` of type ``UserName``.

Global variables should be prefixed with a ``g_`` and member variables
(including static member variables) should be prefixed with a ``m_``. The goal
of that prefix is to give a reader a sense of where a variable of a given
name is declared to allow the reader to locate the variable declaration and
infer the variable type from that declaration. Defined types will start
with an upper case letter, consist of upper and lower case letters, and may
optionally end with a ``_t``. Defined constants (such as static const class
members, or enum constants) will be all uppercase letters or numeric digits,
with an underscore character separating words. Otherwise, the underscore
character should not be used in a variable name. For example, you could
declare in your class header ``my-class.h``:

::

  typedef int NewTypeOfInt_t;
  const uint8_t PORT_NUMBER = 17;

  class MyClass
  {
    void MyMethod (int aVar);
    int m_aVar;
    static int m_anotherVar;
  };

and implement in your class file ``my-class.cc``:

::

  int MyClass::m_anotherVar = 10;
  static int g_aStaticVar = 100;
  int g_aGlobalVar = 1000;
  void
  MyClass::MyMethod (int aVar)
  {
    m_aVar = aVar;
  }

As an exception to the above, the members of structures do not need to be
prefixed with an ``m_``.

Finally, do not use
`Hungarian notation <https://en.wikipedia.org/wiki/Hungarian_notation>`_, and
do not prefix enums, classes, or delegates with any letter.

Choosing names
==============

Variable, function, method, and type names should be based on the English
language, American spelling. Furthermore, always try to choose descriptive
names for them. Types are often english names such as: Packet, Buffer, Mac,
or Phy. Functions and methods are often named based on verbs and adjectives:
GetX, DoDispose, ClearArray, etc.

A long descriptive name which requires a lot of typing is always better than
a short name which is hard to decipher. Do not use abbreviations in names
unless the abbreviation is really unambiguous and obvious to everyone (e.g.,
use "size" over "sz"). Do not use short inappropriate names such as foo,
bar, or baz. The name of an item should always match its purpose. As such,
names such as "tmp" to identify a temporary variable, or such as "i" to
identify a loop index are ok.

If you use predicates (that is, functions, variables or methods which return
a single boolean value), prefix the name with "is" or "has".

File layout and code organization
*********************************

A class named ``MyClass`` should be declared in a header named ``my-class.h``
and implemented in a source file named ``my-class.cc``. The goal of this
naming pattern is to allow a reader to quickly navigate through the |ns3|
codebase to locate the source file relevant to a specific type.

Each ``my-class.h`` header should start with the following comments: the
first line ensures that developers who use the emacs editor will be able to
indent your code correctly. The following lines ensure that your code
is licensed under the GPL, that the copyright holders are properly
identified (typically, you or your employer), and that the actual author
of the code is identified. The latter is purely informational and we use it
to try to track the most appropriate person to review a patch or fix a bug.
Please do not add the "All Rights Reserved" phrase after the copyright
statement.

::

  /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
  /*
   * Copyright (c) YEAR COPYRIGHTHOLDER
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
   * Author: MyName <myemail@example.com>
   */

Below these C-style comments, always include the following which defines a
set of header guards (``MY_CLASS_H``) used to avoid multiple header includes,
which ensures that your code is included in the |ns3| namespace and which
provides a set of Doxygen comments for the public part of your class API.
Detailed information on the set of tags available for doxygen documentation
is described in the `Doxygen website <https://www.doxygen.nl/index.html>`_.

::

  #ifndef MY_CLASS_H
  #define MY_CLASS_H

  namespace n3 {

  /**
   * \brief short one-line description of the purpose of your class
   *
   * A longer description of the purpose of your class after a blank
   * empty line.
   */
  class MyClass
  {
  public:
    MyClass ();
    /**
     * \param firstParam a short description of the purpose of this parameter
     * \returns a short description of what is returned from this function.
     *
     * A detailed description of the purpose of the method.
     */
    int DoSomething (int firstParam);
  private:
    /**
     * Private method doxygen is also recommended
     */
    void MyPrivateMethod (void); 
    int m_myPrivateMemberVariable; ///< Brief description of member variable
  };

  } // namespace ns3

  #endif /* MY_CLASS_H */

The ``my-class.cc`` file is structured similarly:

::

  /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
  /*
   * Copyright (c) YEAR COPYRIGHTHOLDER
   *
   * 3-paragraph GPL blurb
   *
   * Author: MyName <myemail@foo.com>
   */
  #include "my-class.h"

  namespace ns3 {

  MyClass::MyClass ()
  {}

  ...

  } // namespace ns3

Language features
*****************

As of ns-3.36, |ns3| permits the use of C++-17 (or earlier) features
in the implementation files. Because the |ns3| public APIs are scanned by a
a `Python bindings framework <https://github.com/gjcarneiro/pybindgen>`_ that
doesn't support even all of the C++-11 features, the |ns3| project is more
conservative in using newer C++ standards in the APIs, and may ask
to rework some APIs to retain compatibility with pybindgen.

If a developer would like to propose to raise this bar to include more
features than this, please email the developers list. We will move this
language support forward as our minimally supported compiler moves forward.

Comments
********

The project uses `Doxygen <https://www.doxygen.nl/index.html>`_ to document
the interfaces, and uses comments for improving the clarity of the code
internally. All classes, methods, and members should have Doxygen comments. Doxygen comments
should use the C comment (also known as Javadoc) style. For comments that
are intended to not be exposed publicly in the Doxygen output, use the
``@internal`` and ``@endinternal`` tags.
Please use the ``@see`` tag for cross-referencing. All
parameters and return values should be documented.  The |ns3| codebase uses
both the ``@`` or ``\`` characters for tag identification; please make sure
that usage is consistent within a file.

As for comments within the code, comments should be used to describe intention
or algorithmic overview where is it not immediately obvious from reading the
code alone. There are no minimum comment requirements and small routines
probably need no commenting at all, but it is hoped that many larger
routines will have commenting to aid future maintainers. Please write
complete English sentences and capitalize the first word unless a lower-case
identifier begins the sentence. Two spaces after each sentence helps to make
emacs sentence commands work. Sometimes ``NS_LOG_DEBUG`` statements can
be also used in place of comments.

Short one-line comments and long comments can use the C++ comment style;
that is, ``//``, but longer comments may use C-style comments:

::

  /*
   * A longer comment,
   * with multiple lines.
   */

Variable declaration should have a short, one or two line comment describing
the purpose of the variable, unless it is a local variable whose use is
obvious from the context. The short comment should be on the same line as the
variable declaration, unless it is too long, in which case it should be on
the preceding lines.

Casts
*****

Where casts are necessary, use the Google C++ guidance:  "Use C++-style casts
like ``static_cast<float>(double_value)``, or brace initialization for
conversion of arithmetic types like ``int64 y = int64{1} << 42``."

Try to avoid (and remove current instances of) casting of ``uint8_t``
type to larger integers in our logging output by overriding these types
within the logging system itself. Also, the unary ``+`` operator can be used
to print the numeric value of any variable, such as:

::

  uint8_t flags = 5;
  std::cout << "Flags numeric value: " << +flags << std::endl;

Avoid unnecessary casts if minor changes to variable declarations can solve
the issue.

Miscellaneous items
*******************

- The following emacs mode line should be the first line in a file:
  ::

    /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

- ``NS_LOG_COMPONENT_DEFINE("log-component-name");`` statements should be
  placed within namespace ns3 (for module code) and after the
  ``using namespace ns3;``.  In examples. 
  ``NS_OBJECT_ENSURE_REGISTERED()`` should also be placed within namespace ns3.

- Const reference syntax:
  ::

    void MySub (const T&);    // Method 1  (prefer this syntax)
    void MySub (T const&);    // Method 2  (avoid this syntax)

- Use a space between the function name and the parentheses, e.g.:
  ::

    void MySub(const T&);     // avoid this
    void MySub (const T&);    // use this instead
  This spacing rule applies both to function declarations and invocations.

- Do not include inline implementations in header files; put all
  implementation in a .cc file (unless implementation in the header file
  brings demonstrable and significant performance improvement).

- Do not use ``nil`` or ``NULL`` constants; use ``0`` (improves portability)

- Consider whether you want the default constructor, copy constructor, or assignment
  operator in your class, and if not, explicitly mark them as deleted:
  ::

    public:
      // Explain why these are not supported
      ClassName () = delete;
      ClassName (const ClassName&) = delete;
      ClassName& operator= (const ClassName&) = delete;

- Avoid returning a reference to an internal or local member of an object:
  ::

    a_type& foo (void);  // should be avoided, return a pointer or an object.
    const a_type& foo (void); // same as above
  This guidance does not apply to the use of references to implement operators.

- Expose class members through access functions, rather than direct access
  to a public object. The access functions are typically named Get" and 
  "Set". For example, a member m_delayTime might have accessor functions
  ``GetDelayTime ()`` and ``SetDelayTime ()``.

- For standard headers, use the C++ style of inclusion, such as
  ::

    #include <cheader>
  instead of
  ::

    #include <header.h>

- Do not bring the C++ standard library namespace into |ns3| source files by
  using the "using" directive; i.e. avoid ``using namespace std;``.

- When including |ns3| headers in other |ns3| files, use `<>` when you expect
  the header to be found in build/ (or to be installed) and use `""` when
  you know the header is in the same directory as the implementation.

  - inside .h files, always use
    ::

      #include <ns3/header.h>

  - inside .cc files, use
    ::

      #include "header.h"
    if file is in same directory, otherwise use
    ::

      #include <ns3/header.h>

- Compilers will typically issue warnings on unused entities (e.g., variables,
  function parameters).  Use the ``[[maybe_unused]]`` attribute to suppress
  such warnings when the entity may be unused depending on how the code
  is compiled (e.g., if the entity is only used in a logging statement or
  an assert statement).  Example (parameter 'p' is only used for logging):
  ::
    void
    TcpSocketBase::CompleteFork (Ptr<Packet> p [[maybe_unused]], const TcpHeader& h,
                             const Address& fromAddress, const Address& toAddress)
    {
    NS_LOG_FUNCTION (this << p << h << fromAddress << toAddress);
    ...

  In function or method parameters, if the parameter is definitely unused,
  it should be left unnamed.  Example (second parameter is not used):
  ::
    void
    UanMacAloha::RxPacketGood (Ptr<Packet> pkt, double, UanTxMode txMode)
    {
      UanHeaderCommon header;
      pkt->RemoveHeader (header);
      ...

  In this case, the parameter is also not referenced by Doxygen; e.g.:
  ::
    /*
     * Receive packet from lower layer (passed to PHY as callback).
     *
     * \param pkt Packet being received.
     * \param txMode Mode of received packet.
     */
    void RxPacketGood (Ptr<Packet> pkt, double, UanTxMode txMode);

  The omission is preferred to commenting out unused parameters such as:
  ::
    void
    UanMacAloha::RxPacketGood (Ptr<Packet> pkt, double /*sinr*/, UanTxMode txMode)
    {
      UanHeaderCommon header;
      pkt->RemoveHeader (header);
      ...

