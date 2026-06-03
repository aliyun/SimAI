.. include:: replace.txt
.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

.. _Introduction:

Introduction
------------

This document is a guide for those who wish to contribute code to |ns3|
or its related projects in some way.

Changes to |ns3| software are made by
or reviewed by *maintainers*.  |ns3| has a small core team of maintainers,
and also some specialized maintainers who maintain a portion of the software.
The source code branch that is centrally maintained is sometimes
called the *mainline*, a term used further herein.
End users are encouraged to report issues and bugs, or to propose
software or documentation modifications, to be reviewed by and handled by
maintainers.  End users can also help by reviewing code
proposals made by others.  Some end users who contribute high quality
patches or code reviews over time may ask or be invited to become 
a maintainer of software within their areas of expertise.  Finally, some
end users wish to disseminate their |ns3| work to the community, through
the addition of new features or modules to |ns3|.

A question often asked by newcomers is "How can I contribute to |ns3|?" 
or "How do I get started?".
This document summarizes the various ways and processes used to contribute
to |ns3|.  Contribution by users is essential for a project maintained
largely by volunteers.  However, one of the most scarce resources on
an open source project is the available time of maintainers, so we ask
contributors to please become familiar with conventions and processes
used by |ns3| so as to smooth the contribution process.

The very first step is to become familiar with |ns3| by reading the tutorial,
running some example programs, and then reading some of the code and 
documentation to get a feel for things.  From that point, there are a 
number of options to contribute:

* Contributing a small documentation correction
* Reporting or discussing a bug in the software
* Fixing an already reported bug by submitting a patch
* Reviewing code contributed by others
* Submitting new code for inclusion in the mainline
* Alerting users to code that is maintained or archived elsewhere
* Submitting a module or fork for publishing in the ns-3 app store
* Becoming a maintainer of part of the simulator

|ns3| is mainly a C++ project, and this contribution guide focuses on
how to contribute |ns3| code specifically, but the overall open source
project maintains various related codebases written in several other
languages, so if you are interested in contributing outside of |ns3|
C++ code, there are several possibilities: 

* |ns3| provides Python bindings to most of its API, and maintains an
  automated API scanning process that relies on other tools.  We can use
  maintenance help in the C++/Python bindings support.
* Another Python project is the `bake build tool <https://gitlab.com/nsnam/bake/-/issues>`_, which has a number of open issues.
* See also our Python-based `PyViz visualizer <https://www.nsnam.org/wiki/PyViz>`_; extensions and documentation would be welcome.
* The `NetAnim <https://gitlab.com/nsnam/netanim>`_ animator is written in `Qt <https://www.qt.io/>`_ and has lacked a maintainer for several years.
* If you are interested in Linux kernel hacking, or use of applications in |ns3| such as open source routing daemons, we maintain the
  `Direct Code Execution project <https://www.nsnam.org/docs/dce/manual/html/index.html>`_.
* If you are familiar with `Django <https://www.djangoproject.com/>`_, we have work to do on `our app store infrastructure <https://gitlab.com/nsnam/ns-3-AppStore>`_. 
* Our `website <https://gitlab.com/nsnam/nsnam-web>`_ is written in `Jekyll <https://jekyllrb.com/>`_ and is in need of more work.

The remainder of this document is organized as follows.

* Chapter 2 covers general issues surrounding code and documentation
  contributions, including license and copyright;
* Chapter 3 describes approaches for contributing small enhancements to
  the |ns3| mainline, such as a documentation or bug fix;
* Chapter 4 outlines the approach for proposing new models for the mainline;
* Chapter 5 describes how to contribute code that will be stored outside
  of the |ns3| mainline, with emphasis on the |ns3| *AppStore*; and
* Chapter 6 provides the coding style guidelines that are mandatory for the
  |ns3| mainline and strongly suggested for contributed modules.

