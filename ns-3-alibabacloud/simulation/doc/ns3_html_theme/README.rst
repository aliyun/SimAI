.. ns3-theme documentation master file, created by
   sphinx-quickstart on Fri Jun 15 12:53:37 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


The files in this directory customize the Doxygen and Sphinx
layout and rendering to give a common look.


Design
------

The basic color scheme is taken from the ns3 homepage, http://www.nsnam.org/

Since Sphinx provides more flexible layout and configuration control,
the Doxygen layout and images are used as the basis.

The top bar design is derived from Doxygen's default when a logo,
project name and description are given in the Doxygen configuration file.
The files ``layout.html`` and ``ns3_doxy_header.html`` should kept in sync.


Sphinx Files
------------

``layout.html`` (:ref:`static filename <sphinx-static-filenames>`)
    Augments the default page layout, including the top bar.

``theme.conf`` (:ref:`static filename <sphinx-static-filenames>`)
    Configuration parameters for ``layout.html`` and ``default.css_t``.

``static/bc_s.png``
    Angle bracket in Sphinx relbar, taken from the Doxygen html output.

``static/default.css_t`` (:ref:`static filename <sphinx-static-filenames>`)
    Default Sphinx CSS template, copied from the default theme.

``static/nav_f.png``
    Background image for headings, taken from the Doxygen html output.

``static/sidebar.js`` (:ref:`static filename <sphinx-static-filenames>`)
    Sidebar script, copied from the default theme.

``static/tab_b.png``
    Background image for the Sphinx relbar, taken from the Doxygen html output.

.. _sphinx-static-filenames:
.. note:: **Static Filenames**

  Sphinx uses hard-coded file  names (and the ``static`` directory name).
  The files referenced above can't be renamed.


Doxygen Files
-------------

``ns3_doxy_footer.html``
    Defines the page footer.

``ns3_doxy_header.html``
    Defines the page header, including the top bar.

``static/doxygen.css``
    Default Doxygen CSS file, obtained by the ``doxygen -w html ...``
    command.

Shared Files
------------

``README.rst``
    This file, not used by Doxygen or Sphinx.

``static/bar-top.png``
    Background image for the top bar, resized from the ns-3 homepage.

``static/favicon.ico``
    Browser location bar favicon, referenced in ``layout.html`` and
    ``ns3_doxy_header.html``, taken from the ns-3 homepage.

``static/ns-3-inverted-notext-small.png``
    Logo used in the top bar, taken from the ns-3 media kit.

``static/ns3_stylesheet.css``
    Style customizations for both Doxygen and Sphinx.
