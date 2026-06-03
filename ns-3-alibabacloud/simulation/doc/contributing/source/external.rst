.. include:: replace.txt
.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

.. _External:

Submitting externally maintained code
-------------------------------------

This chapter mainly pertains to code that will be maintained in external
repositories (such as a personal or university research group repository,
possibly hosted on GitHub or GitLab.com),
but for which the contributor wishes to keep consistent and compatible 
with the |ns3| mainline.

If the contributor does not want to maintain the external code
but wants to make the community aware that it is available with
no ongoing support, links to the code can be posted on the |ns3|
wiki contributed code page.  A typical example is the graduating
student who wishes to make his or her thesis-related code available
but who does not plan to update it further.
See :ref:`sec-unmaintained-contributed` below.

However, much of the emphasis of this chaper is on hosting |ns3| extensions
in the |ns3| `App Store <https://apps.nsnam.org>`_.

Rationale for the app store
***************************

Historically, |ns3| tried to merge all code contributions (that met the
contribution guidelines) to the mainline.  However, this approach has
reached its limits in terms of scalability:

1. As |ns3| includes more lines of code, it takes longer to compile, while
   users typically only need a subset of the available |ns3| model code.

2. Merging to the mainline requires maintainer participation for code
   reviews (which can be time consuming for maintainers), resulting in some
   contributions not being merged due to lack of code reviews.

The app store federates the maintenance of |ns3| modules, and puts more
control in the hands of the contributor to maintain their code.  The idea
is to provide a central place where users can discover |ns3| extensions
and add them to their |ns3| installation.  One of the drawbacks of this
approach, however, is the potential for a lot of compatibility headaches,
such as a user trying to use two separately developed modules, but the
modules are compatible with different |ns3| mainline versions.  Therefore,
the app store is designed to assist in testing and clarifying which version
of the external module is compatible with which mainline version of |ns3|.

In brief, what it means for |ns3| users is the following.  There is an
empty directory in the |ns3| source code tree called `contrib`.  This directory
behaves the same way (with respect to the build system) as the `src`
directory, but is the place in which externally developed models can
be downloaded and added to |ns3|.

For contributors, the app store provides a means for module developers
to make releases on their own schedule and not be tied to the release
schedule of the mainline, nor to be blocked by mainline code reviews
or style requirements.  It is still possible (and recommended) to obtain
code reviews and follow style guidelines for contributed modules, as
described below.

Example app
===========

For example, consider a user who wants to use a model of the LoRaWAN
sensor networking protocol in |ns3|.  This model is not available in
the mainline |ns3| releases, but the user checks in the
`App Store <https://apps.nsnam.org>`_.  Sensor networking protocols can
be found by browsing the category *Sensor Networks* in the left-hand
menu, upon which the LoRaWAN module is displayed.  This module is maintained
by an external research group.  Clicking on the app icon leads to a page
in which more details about this module are displayed, including
*Release History*, *Installation*, and *Maintenance*.

The installation tab suggests two primary ways to add a `lorawan` module
to an |ns3| installation:

1. Download a source archive from the link provided on the app store and
   unpack it in the `contrib/` directory.

2. Perform a `git clone` operation from within the |ns3| `contrib/` directory.

There is a third way in general, which is to use the Bake build system
to add or remove modules.  Most modules have a `bakeconf.xml` file
associated with them, which can be added to the `contrib/` folder of
Bake.  This permits bake configuration operations such as:

.. code-block:: text

  $ ./bake.py configure -e ns-3.34 -e lorawan-0.3.0
  $ ./bake.py download
  $ ./bake.py build

Users will be concerned about which version of the module applies to the
version of |ns3|.  The *Release History* tab describes the releases and
the minimum (and, optionally, the maximum) |ns3| version that the release
works with.

Finally, information about how the module will continue to be maintained,
and how to submit issue reports or ask questions about it, are typically
posted in the optional *Maintenance* or *Development* tabs.

App types
*********

Two types of "apps" are hosted on the app store:  1) contributed modules,
and 2) forks of |ns3|.  It benefits users the most when contributed code
can be added as modules to the `contrib` directory, because this allows
the modules to be used with different versions of the |ns3| mainline
and to combine multiple such modules.  However, some authors have chosen
to create a full fork of |ns3|, and to publish full |ns3| source trees.
One reason for such forks is if the fork requires intrusive changes to
the |ns3| mainline code, in such a manner that the patch will likely not
be accepted in the mainline.  For an example of a full fork in the
app store, see the `Public Safety Communications <https://apps.nsnam.org/app/publicsafetylte/>`_ app page; this is a full fork because of the intrusive
changes to the mainline LTE module necessary to support the public
safety extensions.

Sometimes a contributed module does require a very small patch to the
mainline despite being largely implemented in a modular way.  In this
case, the best strategy is to try to create a minimal patch to
upstream to the mainline, and work with maintainers to incorporate it.
Even if this is not successful or not attempted, one can maintain
a small patch file as part of the module, and bake build instructions
for the module can be extended to patch the mainline code as the first
step in the build process.  Please consult the app store maintainers
if guidance on this approach is needed.

Submitting to the app store
***************************

If you are interested in adding an app to the app store, contact
``webmaster@nsnam.org`` or one of the app store maintainers.  Before
getting started, browse through the existing apps so that you get
a feel for what type of information will be required.

The main requirements are:

1. Define a module name that will be the name of the subdirectory in the
   `contrib` folder.  Use |ns3| naming conventions for directory names; i.e.,
   all lower case, with separate words separated by dashes.

2. Ensure that your module can be successfully cloned into the contrib folder
   using the proposed module name, that it builds successfully, and that
   all examples and tests run.

3. Make at least one release on your GitHub or GitLab.com, associated
   with the latest |ns3| release if possible.  To do this, you will need
   to decide on a numbering scheme, such as 'v1.0' or '0.1', etc.
   Consult GitHub or GitLab.com documentation on how to make a (tagged)
   release.

4. Come up with a small thumbnail icon for your app.  A general guideline
   is a transparent PNG, size 100x100 pixels.

5. Provide names and email addresses of the person(s) who should have edit
   privileges (accounts) on the app store.

After providing this information, a skeleton page will be set up on the
app store and will be initially invisible to users who visit the front
page.  Contributors can then log in and work with the app store
maintainer to finalize the page, and then make it active (visible) to
other users when ready.

Code review for apps
********************

Even though apps are not added to the |ns3| mainline, it is still possible
to request code reviews for them (in order to improve the code).  There
is a special repository set up for this purpose:
`ns-3-contrib-reviews <https://gitlab.com/nsnam/ns-3-contrib-reviews>`_.  The
purpose of this repository is to provide a fork-able repository against
which to generate Merge Requests, but no eventual merge ever takes place.

The steps to request a code review are:

1. Fork `ns-3-contrib-reviews` into your own namespace
2. Clone the fork to your local machine
3. Create a new feature branch  on what you just checked out for your new code
4. cd to 'contrib' and clone your extension module there
5. remove the .git directory of that module so that git does not treat it as a submodule; e.g., if your module name is `modulename`:

.. code-block:: text

  $ rm -rf modulename/.git

6. Force an add of the module, overcoming the .gitignore file for contrib modules

.. code-block:: text

  $ git add -f modulename

7. Add your files:

.. code-block:: text

  $ git commit -m"Add modulename for review" -a

8.  Push the feature branch to your remote repository

.. code-block:: text

  $ git push -u origin

9. Navigate to your github.com page and generate a Merge Request towards
   the upstream `ns-3-contrib-reviews`

10. Announce on ns-developers mailing list that you would like a review.

Maintaining app store modules
*****************************

A recent `Google Summer of Code project <https://mishal.dev/gsoc19-report>`_
worked on allowing integration of the app store with continuous integration
(CI) testing.  This integration is ongoing, but the eventual goal is that
regressions that are caused by upstream changes to |ns3| will be quickly
caught, and the app owner will be asked to fix the build of their app,
possibly by issuing another module release as well.

Links to related projects
*************************
Some projects choose to maintain their own version of |ns3|, or maintain models
outside of the main tree of the code.  In this case, the way to find out
about these is to look at the Related Projects page on the |ns3| 
`wiki <https://www.nsnam.org/wiki/Related_Projects>`_.

If you know of externally maintained code that the project does not know about,
please email ``webmaster@nsnam.org`` to request that it be added to the
list of external projects.

Going forward, we will tend to prefer to list related projects on the
app store so that there is a single place to discover them.

.. _sec-unmaintained-contributed:

Unmaintained, contributed code
******************************

Anyone who wants to provide access to code that has been developed to 
extend |ns3| but that will not be further maintained may list its
availability on our website. Furthermore, we can provide, in some
circumstances, storage of a compressed archive on a web server if needed.
This type of code contribution will 
not be listed in the app store, although popular extensions might be
adopted by a future contributor.

We ask anyone who wishes to do this to provide at least this information 
on our wiki:

* Authors,
* Name and description of the extension,
* How it is distributed (as a patch or full tarball),
* Location,
* Status (how it is maintained)

Please also make clear in your code the applicable software license.
The contribution will be stored on our `wiki <https://www.nsnam.org/wiki/Contributed_Code>`_. If you need web server 
space for your archive, please contact ``webmaster@nsnam.org``.

