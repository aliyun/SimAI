.. include:: replace.txt
.. highlight:: cpp

Creating Documentation
----------------------

|ns3| supplies two kinds of documentation:  expository "user-guide"-style
chapters, and source code API documentation.

The "user-guide" chapters are written by hand in reStructuredText_
format (``.rst``), which is processed by the Python documentation
system Sphinx_ to generate web pages and pdf files.
The API documentation is generated from the source code itself,
using Doxygen_, to generate cross-linked web pages.
Both of these are important:  the Sphinx chapters explain the *why*
and overview of using a model; the API documentation explains the
*how* details.  

This chapter gives a quick overview of these
tools, emphasizing preferred usage and customizations for |ns3|.

To build all the standard documentation:

.. sourcecode:: bash

   $ ./ns3 docs

For more specialized options, read on.

.. _reStructuredText:  http://sphinx-doc.org/rest.html
.. _sphinx:            http://sphinx-doc.org/
.. _doxygen:           http://www.doxygen.org/


Documenting with Sphinx
***********************

We use Sphinx_ to generate expository chapters describing
the design and usage of each module.  Right now you are reading the
:doc:`Documentation <documentation>` Chapter.
If you are reading the html version, the
`Show Source <_sources/documentation.txt>`_ link in the sidebar
will show you the reStructuredText source for this chapter.

Adding New Chapters
===================

Adding a new chapter takes three steps (described in more detail below):

#. Choose `Where?`_ the documentation file(s) will live.
#. `Link`_ from an existing page to the new documentation.
#. Add the new file to the `Makefile`_.

Where?
######

Documentation for a specific module, ``foo``, should normally go in
``src/foo/doc/``.  For example ``src/foo/doc/foo.rst`` would be the
top-level document for the module.  The ``utils/create-module.py`` script
will create this file for you.

Some models require several ``.rst`` files, and figures; these should
all go in the ``src/foo/doc/`` directory.  The docs are actually built
by a Sphinx Makefile.  For especially involved
documentation, it may be helpful to have a local ``Makefile``
in the ``src/foo/doc/`` directory to
simplify building the documentation for this module 
(`Antenna`_ is an example).  Setting this up
is not particularly hard, but is beyond the scope of this chapter.

In some cases, documentation spans multiple models; the
`Network`_ chapter is an example.  In these cases
adding the ``.rst`` files directly to ``doc/models/source/`` might
be appropriate.

.. _antenna:  http://www.nsnam.org/docs/models/html/antenna.html
.. _network:  http://www.nsnam.org/docs/models/html/network.html

Link
####

Sphinx has to know *where* your new chapter should appear.  In most
cases, a new model chapter should appear the in `Models` book.
To add your chapter there, edit ``doc/models/source/index.rst``

.. sourcecode:: rest

   .. toctree::
      :maxdepth: 1
 
     organization
     animation
     antenna
     aodv
     applications
     ...

Add the name of your document (without the ``.rst`` extension) to
this list.  Please keep the Model chapters in alphabetical order,
to ease visual scanning for specific chapters.

Makefile
########

You also have to add your document to the appropriate ``Makefile``,
so ``make`` knows to check it for updates.  The Models book Makefile
is ``doc/models/Makefile``, the Manual book Makefile is
``doc/manual/Makefile``.

.. sourcecode:: make

   # list all model library .rst files that need to be copied to $SOURCETEMP
   SOURCES = \
           source/conf.py \
           source/_static \
           source/index.rst \
           source/replace.txt \
           source/organization.rst \
           ...
           $(SRC)/antenna/doc/source/antenna.rst \
           ...

You add your ``.rst`` files to the ``SOURCES`` variable.  To add figures,
read the comments in the ``Makefile`` to see which variable should contain
your image files.  Again, please keep these in alphabetical order.

Building Sphinx Docs
====================

Building the Sphinx documentation is pretty simple.  
To build all the Sphinx documentation:

.. sourcecode:: bash

   $ ./ns3 sphinx

To build just the Models documentation:

.. sourcecode:: bash

   $ make -C doc/models html

To see the generated documentation point your browser at
``doc/models/build/html``.

As you can see, Sphinx uses Make to guide the process.
The default target builds all enabled output forms, which in
|ns3| are the multi-page ``html``, single-page ``singlehtml``, and pdf
(``latex``).  To build just the multi-page html, you add the ``html`` target:

.. sourcecode:: bash

   $ make -C doc/models html

This can be helpful to reduce the build time (and the size of the
build chatter) as you are writing your chapter.

Before committing your documentation to the repo, please check
that it builds without errors or warnings.  The build process
generates lots of output (mostly normal chatter from LaTeX),
which can make it difficult to see if there are any Sphinx
warnings or errors.  To find important warnings and errors
build just the ``html`` version, then search the build log
for ``warning`` or ``error``.


|ns3| Specifics
===============

The Sphinx documentation_ and tutorial_ are pretty good.  We won't duplicate
the basics here, instead focusing on preferred usage for |ns3|.

.. _documentation:  http://sphinx-doc.org/contents.html
.. _tutorial:       http://sphinx-doc.org/tutorial.html


* Start documents with these two lines:
   
  .. sourcecode:: rest
     
     .. include:: replace.txt
     .. highlight:: cpp
   
  The first line enables some simple replacements.  For example,
  typing ``|ns3|`` renders as |ns3|.
  The second sets the default source code highlighting language explicitly
  for the file, since the parser guess isn't always accurate.
  (It's also possible to set the language explicitly for a single code block,
  see below.)

* Sections:
  
  Sphinx is pretty liberal about marking section headings.  By convention,
  we prefer this hierarchy:
  
  .. sourcecode:: rest
      
     .. heading hierarchy:
        ------------- Chapter
        ************* Section (#.#)
        ============= Subsection (#.#.#)
        ############# Sub-subsection

* Syntax Highlighting:
  
  To use the default syntax highlighter, simply start a sourcecode block:

  +--------------------------------------+------------------------------------+
  | Sphinx Source                        | Rendered Output                    |
  +======================================+====================================+
  | .. sourcecode:: rest                 |                                    |
  |                                      |                                    |
  |   The ``Frobnitz`` is accessed by::  | The ``Frobnitz`` is accessed by::  |
  |                                      |                                    |
  |     Foo::Frobnitz frob;              |   Foo::Frobnitz frob;              |
  |     frob.Set (...);                  |   frob.Set (...);                  |
  +--------------------------------------+------------------------------------+

  To use a specific syntax highlighter, for example, ``bash`` shell commands:

  +--------------------------------------+------------------------------------+
  | Sphinx Source                        | Rendered Output                    |
  +======================================+====================================+
  | .. sourcecode:: rest                 |                                    |
  |                                      |                                    |
  |   .. sourcecode:: bash               | .. sourcecode:: bash               |
  |                                      |                                    |
  |      $ ls                            |    $ ls		              |
  +--------------------------------------+------------------------------------+
  
* Shorthand Notations:

  These shorthands are defined:

  +--------------------------------------+------------------------------------+
  | Sphinx Source                        | Rendered Output                    |
  +======================================+====================================+
  | .. sourcecode:: rest                 |                                    |
  |                                      |                                    |
  |    |ns3|                             |  |ns3|                             |
  +--------------------------------------+------------------------------------+
  | .. sourcecode:: rest                 |                                    |
  |                                      |                                    |
  |    |ns2|                             |  |ns2|                             |
  +--------------------------------------+------------------------------------+
  | .. sourcecode:: rest                 |                                    |
  |                                      |                                    |
  |    |check|                           |  |check|                           |
  +--------------------------------------+------------------------------------+
  | .. sourcecode:: rest                 |                                    |
  |                                      |                                    |
  |    :rfc:`6282`                       |  :rfc:`6282`                       |
  +--------------------------------------+------------------------------------+


Documenting with Doxygen
************************

We use Doxygen_ to generate browsable_ API documentation.  Doxygen
provides a number of useful features:

* Summary table of all class members.
* Graphs of inheritance and collaboration for all classes.
* Links to the source code implementing each function.
* Links to every place a member is used.
* Links to every object used in implementing a function.
* Grouping of related classes, such as all the classes related to
  a specific protocol.

In addition, we use the ``TypeId`` system to add to the documentation
for every class

* The ``Config`` paths by which such objects can be reached.
* Documentation for any ``Attributes``, including ``Attributes``
  defined in parent classes.
* Documentation for any ``Trace`` sources defined by the class.
* The memory footprint for each class.

Doxygen operates by scanning the source code, looking for
specially marked comments.  It also creates a cross reference,
indicating *where* each file, class, method, and variable is used.

.. _browsable: https://www.nsnam.org/docs/doxygen


Preferred Style
===============

The preferred style for Doxygen comments is the JavaDoc style::

  /**
   *  Brief description of this class or method.
   *  Adjacent lines become a single paragraph.
   *
   *  Longer description, with lots of details.
   *
   *  Blank lines separate paragraphs.
   *
   *  Explain what the class or method does, using what algorithm.
   *  Explain the units of arguments and return values.
   *
   *  \note Note any limitations or gotchas.
   *
   *  (For functions with arguments or return valued:)
   *  \param [in] foo Brief noun phrase describing this argument. Note
   *                  that we indicate if the argument is input,
   *                  output, or both.
   *  \param [in,out] bar Note Sentence case, and terminating period.
   *  \param [in] baz Indicate boolean values with \c true or \c false.
   *  \return Brief noun phrase describing the value.
   *
   *  \internal
   *
   *  You can also discuss internal implementation details.
   *  Understanding this material shouldn't be necessary to using
   *  the class or method.
   */
   void ExampleFunction (const int foo, double & bar, const bool baz);

In this style the Doxygen comment block begins with two \`*' characters:
``/**``, and precedes the item being documented.

For items needing only a brief description, either of these short forms
is appropriate::

  /** Destructor implementation. */
  void DoDispose ();

  int m_count;  //!< Count of ...

Note the special form of the end of line comment, ``//!<``, indicating
that it refers to the *preceding* item.

Some items to note:

* Use sentence case, including the initial capital.
* Use punctuation, especially \`.'s at the end of sentences or phrases.
* The ``\brief`` tag is not needed; the first sentence will be
    used as the brief description.

Every class, method, typedef, member variable, function argument
and return value should be documented in all source code files
which form the formal API and implementation for |ns3|, such as
``src/<module>/model/*``, ``src/<module>/helper/*`` and
``src/<module>/utils/*``.  Documentation for items in ``src/<module>/test/*``
and ``src/<module>/examples/*`` is preferred, but not required.

Useful Features
===============

* Inherited members will automatically inherit docs from the parent,
  (but can be replaced by local documentation).

  #. Document the base class.
  #. In the sub class mark inherited functions with an ordinary comment::

       // Inherited methods
       virtual void FooBar (void);
       virtual int BarFoo (double baz);

     Note that the signatures have to match exactly, so include the formal
     argument ``(void)``

     This doesn't work for static functions; see ``GetTypeId``, below, for an
     example.

Building Doxygen Docs
=====================

Building the Doxygen documentation is pretty simple:

.. sourcecode:: bash

   $ ./ns3 doxygen

This builds using the default configuration, which generates
documentation sections for *all* items, even if they do not have
explicit comment documentation blocks.  This has the effect of
suppressing warnings for undocumented items, but makes sure everything
appears in the generated output, which is usually what you want for
general use.  Note that we generate documentation even for modules
which are disabled, to make it easier to see all the features
available in |ns3|.

When writing documentation, it's often more useful to see which items
are generating warnings, typically about missing documentation.  To
see the full warnings list, use the ``doc/doxygen.warnings.report.sh``
script:

.. sourcecode:: bash

    $ doc/doxygen.warnings.report.sh
    
    doxygen.warnings.report.sh:
    Building and running print-introspected-doxygen...done.
    Rebuilding doxygen (v1.8.10) docs with full errors...done.


    Report of Doxygen warnings
    ----------------------------------------

    (All counts are lower bounds.)

    Warnings by module/directory:

    Count Directory
    ----- ----------------------------------
    3414 src/lte/model
    1532 src/wimax/model
     825 src/lte/test
    ....
       1 src/applications/test
     97 additional undocumented parameters.
    ----------------------------------------
     12460 total warnings
       100 directories with warnings


    Warnings by file (alphabetical)

    Count File
    ----- ----------------------------------
      15 examples/routing/manet-routing-compare.cc
      26 examples/stats/wifi-example-apps.h
      12 examples/tutorial/fifth.cc
    ....
      17 utils/python-unit-tests.py
    ----------------------------------------
       771 files with warnings


    Warnings by file (numerical)

    Count File
    ----- ----------------------------------
     273 src/lte/model/lte-rrc-sap.h
     272 src/core/model/simulator.h
     221 src/netanim/model/animation-interface.h
    ....
       1 src/wimax/model/ul-job.cc
    ----------------------------------------
       771 files with warnings


    Doxygen Warnings Summary
    ----------------------------------------
       100 directories
       771 files
     12460 warnings

(This snippet has *a lot* of lines suppressed!)

The script modifies the configuration to show all warnings, and to
shorten the run time.  (It shortens the run time primarily by
disabling creation of diagrams, such as call trees, and doesn't
generate documentation for undocumented items, in order to trigger the
warnings.)  As you can see, at this writing we have *a lot* of
undocumented items.  The report summarizes warnings by module
``src/*/*``, and by file, in alphabetically and numerical order.

The script has a few options to pare things down and make the output more
manageable.  For help, use the ``-h`` option.  Having run it once
to do the Doxygen build and generate the full warnings log,
you can reprocess the log file with various "filters,"
without having to do the full Doxygen build again by
using the ``-s`` option.  You can exclude warnings
from ``*/examples/*`` files (``-e`` option), and/or ``*/test/*`` files
(``-t``).  Just to be clear, all of the filter options do the complete
fast doxygen build; they just filter doxygen log and warnings output.

Perhaps the most useful option when writing documentation comments
is ``-m <module>``, which will limit the report to just files matching
``src/<module>/*``, and follow the report with the actual warning lines.
Combine with ``-et`` and you can focus on the warnings that are most
urgent in a single module:

.. sourcecode:: bash

   $ doc/doxygen.warnings.report.sh -m mesh/helper
   ...
   Doxygen Warnings Summary
   ----------------------------------------
     1 directories
     3 files
   149 warnings


   Filtered Warnings
   ========================================
   src/mesh/helper/dot11s/dot11s-installer.h:72: warning: Member m_root (variable) of class ns3::Dot11sStack is not documented.
   src/mesh/helper/dot11s/dot11s-installer.h:35: warning: return type of member ns3::Dot11sStack::GetTypeId is not documented
   src/mesh/helper/dot11s/dot11s-installer.h:56: warning: return type of member ns3::Dot11sStack::InstallStack is not documented
   src/mesh/helper/flame/lfame-installer.h:40: warning: Member GetTypeId() (function) of class ns3::FlameStack is not documented.
   src/mesh/helper/flame/flame-installer.h:60: warning: return type of member ns3::FlameStack::InstallStack is not documented
   src/mesh/helper/mesh-helper.h:213: warning: Member m_nInterfaces (variable) of class ns3::MeshHelper is not documented.
   src/mesh/helper/mesh-helper.h:214: warning: Member m_spreadChannelPolicy (variable) of class ns3::MeshHelper is not documented.
   src/mesh/helper/mesh-helper.h:215: warning: Member m_stack (variable) of class ns3::MeshHelper is not documented.
   src/mesh/helper/mesh-helper.h:216: warning: Member m_stackFactory (variable) of class ns3::MeshHelper is not documented.
   src/mesh/helper/mesh-helper.h:209: warning: parameters of member ns3::MeshHelper::CreateInterface are not (all) documented
   src/mesh/helper/mesh-helper.h:119: warning: parameters of member ns3::MeshHelper::SetStandard are not (all) documented


Finally, note that undocumented items (classes, methods, functions,
typedefs, *etc.* won't produce documentation when you build with
``doxygen.warnings.report.sh``, and only the outermost item
will produce a warning.  As a result, if you don't see documentation
for a class method in the generated documentation, the class itself
probably needs documentation.

Now it's just a matter of understanding the code, and writing some
docs!



|ns3| Specifics
===============

As for Sphinx, the Doxygen docs_ and reference_ are pretty good.
We won't duplicate the basics here, instead focusing on preferred
usage for |ns3|.

.. _docs:       http://www.stack.nl/~dimitri/doxygen/index.html
.. _reference:  http://www.stack.nl/~dimitri/doxygen/manual/commands.html


* Use Doxygen ``Modules`` to group related items.

  In the main header for a module, create a Doxgyen group::

    /**
     *  \defgroup foo Foo protocol.
     *  Implementation of the Foo protocol.
     */

  The symbol ``foo`` is how other items can add themselves to this group.
  The string following that will be the title for the group.  Any further
  text will be the detailed description for the group page.

* Document each file, assigning it to the relevant group.  In a header file::

    /**
     *  \file
     *  \ingroup foo
     *  Class Foo declaration.
     */

  or in the corresponding ``.cc`` file::
  
    /**
     *  \file
     *  \ingroup foo
     *  Class FooBar implementation.
     */

* Mark each associated class as belonging to the group::

    /**
     *  \ingroup foo
     *
     *  FooBar packet type.
    */
    class FooBar

* Did you know ``typedefs`` can have formal arguments?  This enables
  documentation of function pointer signatures::

    /**
     *  Bar callback function signature.
     *
     *  \param ale The size of a pint of ale, in Imperial ounces.
     */
    typedef void (* BarCallback)(const int ale);
    
* Copy the ``Attribute`` help strings from the ``GetTypeId`` method to use
  as the brief descriptions of associated members.

* ``\bugid{298}`` will create a link to bug 298 in our Bugzilla.

* ``\p foo`` in a description will format ``foo``
  the same as the ``\param foo`` parameter, making it clear that you
  are referring to an actual argument.

* ``\RFC{301}`` will create a link to RFC 301.

* Document the direction of function arguments with ``\param [in]``, *etc*.
  The allowed values of the direction token are ``[in]``, ``[out]``, and
  ``[in,out]`` (note the explicit square brackets), as discussed in the
  Doxygen docs for ``\param``.
  
* Document template arguments with ``\tparam``, just as you use ``\param``
  for function arguments.

* For template arguments, indicate if they will be deduced or must be given
  explicitly::

    /**
     * A templated function.
     * \tparam T \explicit The return type.
     * \tparam U \deduced The argument type.
     * \param [in] a The argument.
     */
    template <typename T, typename U> T Function (U a);

  * Use ``\tparam U \deduced`` because the type ``U`` can be deduced at
    the site where the template is invoked.  Basically deduction can only
    be done for function arguments.

  * Use ``\tparam T \explicit`` because the type ``T`` can't be deduced;
    it must be given explicitly at the invocation site, as in
    ``Create<MyObject> (...)``

* ``\internal`` should be used only to set off a discussion of implementation
  details, not to mark ``private`` functions (they are already marked,
  as ``private``!)

* Don't create classes with trivial names, such as ``class A``,
  even in test suites.  These cause all instances of the class name
  literal \`A' to be rendered as links.

As noted above, static functions don't inherit the documentation
of the same functions in the parent class.  |ns3| uses a few static
functions ubiquitously; the suggested documentation block for these
cases is:

* Default constructor/destructor::

    MyClass ();   //!< Default constructor
    ~MyClass ();  //!< Destructor

* Dummy destructor and DoDispose::

    /** Dummy destructor, see DoDispose. */
    ~MyClass ();

    /** Destructor implementation */
    virtual void DoDispose ();

* GetTypeId::

    /**
     *  Register this type.
     *  \return The object TypeId.
     */
    static TypeId GetTypeId (void);



