.. include:: replace.txt
.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

.. _General:

General
-------

This section pertains to general topics about licensing, coding style,
and working with Git features, including patch submission.

Licensing
*********
All code submitted must conform to the project licensing framework, which
is `GNU GPLv2 <https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html>`_
compatibility. All new source files should contain a license
statement.  In general, we ask that new source files be provided with a
GNU GPLv2 license, but the author may select another GNU GPLv2-compatible
license if necessary.  GNU GPLv3 is not accepted in the |ns3| mainline.
Note that the Free Software Foundation maintains
`a list <https://www.gnu.org/licenses/license-list.en.html#GPLCompatibleLicenses>`_ of GPLv2-compatible licenses.

If a contribution is based upon or contains copied code that itself
uses GNU GPLv2, then the author should in most cases retain the GPLv2 and
optionally extend the copyright and/or the author (or 'Modified by')
statements.

If a contribution is borrowed from another project under different licensing,
the borrowed code must have a compatible license, and the license must
be copied over as well as the code.  The author may add the GNU GPLv2 if
desired, but in such a case, should clarify which aspects of the code
(i.e., the modifications) are covered by the GPLv2 and not found in the
original.  The Software Freedom Law Center has published
`guidance <http://www.softwarefreedom.org/resources/2007/gpl-non-gpl-collaboration.html>`_ on handling this case.

Note that it is incumbent upon the submitter to make sure that the
licensing attribution is correct and that the code is suitable for |ns3| 
inclusion. Do not include code (even snippets) from sources that have
incompatible licenses.  Even if the code licenses are compatible, do not
copy someone else's code without attribution.

Documentation Licensing
=======================

Licensing for documentation or for material found on |ns3| websites
is covered by the `Creative Commons CC-BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_ license,
and documentation submissions should be submitted under that license.
Please ensure that any documentation submitted does not violate the
terms of another copyright holder, and has correct attribution.  In
particular, copying of substantial portions of an academic journal
paper, or copying or redrawing of figures from such a paper, likely
requires explicit permission from the copyright holder.

Copyright
*********

Copyright is a statement of ownership of code, while licensing describes
the terms by which the owners of the code permit the reuse of the code.

The |ns3| project does not maintain copyright of contributed code.
Copyright remains with the author(s) or their employer.  Because multiple
people or organizations work on the |ns3| code over time, one can think
of the project and even individual files as a "mixed copyright" work.

Copyright can be stated when originating files in |ns3| or when making
"substantial" changes to such files; the definition of substantial is
open to interpretation.  Copyright need not be claimed explicitly by adding
a copyright statement; according to copyright laws in many countries,
copyright rights are automatic upon publishing a work.  Copyright should
not be explicitly listed for *all* changes to a file; for instance, patches
to fix small things or make small adjustments or improvements are not
considered to be subjected to copyright protection.

Use of copyright statements in open source projects, as a means of
author attribution, can be controversial, because having a long list
of copyright statements on every file impairs readability.  Also,
since copyright is automatic, there is little formal legal requirement
to add a copyright statement.  The |ns3| project has decided to follow
the guidance provided by the Software Freedom Law Center in this regard:
https://softwarefreedom.org/resources/2012/ManagingCopyrightInformation.html

Copyright on new files
======================

When originating a new file, the originating author should place his or her
copyright statement at the top in the header, preceding the copy of the
license.  An important exception to this is if the new file is copied from
somewhere else and modified to make the new file; please do not delete
the previous copyrights from the copyright file!  See below for this case.

An example placement of a copyright statement can be found in the file
``src/network/model/packet.h``::

  /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
  /*
   * Copyright (c) 2005,2006 INRIA
   *
   * This program is free software; you can redistribute it and/or modify
   * it under the terms of the GNU General Public License version 2 as
   * published by the Free Software Foundation;

Copyright on existing files
===========================

When providing a substantial feature (maintainers and contributors should
mutually agree on this point) as a patch to an existing file, the
contributor may add a copyright statement that clarifies the new portion
of code that is covered by the new copyright.  An example is the program
``src/lte/model/lte-ue-phy.h``::

  /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
  /*
   * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
   * Copyright (c) 2018 Fraunhofer ESK : RLF extensions
   *
   * This program is free software; you can redistribute it and/or modify
   * it under the terms of the GNU General Public License version 2 as
   * published by the Free Software Foundation;

Here, Fraunhofer ESK added extensions to support radio link failure (RLF),
and the copyright statement clarifies the extension (separated from the
organization by a colon).

Copyright on external code copied from elsewhere
================================================

If a contributor borrows code from somewhere else (such as a snippet of
code to implement an algorithm, or whole files altogether), it is important
to keep the original copyright (and license statement) in the new file.
Contributors who fail to do this may be accused of plagiarism.

Some existing examples of code copied from elsewhere are:

* ``src/network/utils/error-model.h``
* ``src/core/model/valgrind.h``
* ``src/core/model/math.h``

If in doubt about how to handle a specific case, ask a maintainer.

Attribution
***********
Besides copyright, authors also often seek to list attribution, or
even credit for funding, in the headers.  We request that contributors
refrain from aggressively inserting statements of attribution into the
code such as:

::

  // New Hello Timer implementation contributed by John Doe, 2014

especially for small touches to files, because, over time, it clutters
the code.  Git logs are used to track who contributed what over time.

Likewise, if someone contributes a minor enhancement or a bug fix to
an existing file, this is not typically justification to insert an
``Authored by`` or ``Copyright`` statement at the top of the file.  If
everyone who touched a file did this, we would end up with unwieldy
lists of authors on many files.  In general, we recommend to follow
these guidelines:

* if you are authoring a new file or contributing a substantial portion
  of code (such as 30% or more new or changed statements), you can
  list yourself as co-author or add a new copyright to the file header
* if you are modifying less than the above, please refrain from adding
  copyright or author statements as part of your patch
* do not put your name or your organization's name anywhere in the
  main body of the code, for attribution purposes

An example of a substantial modification that led to extension of
the authors section of the header can be found in 
``src/lte/model/lte-ue-phy.h``::

   * Author: Giuseppe Piro  <g.piro@poliba.it>
   * Author: Marco Miozzo <mmiozzo@cttc.es>
   * Modified by:
   *          Vignesh Babu <ns3-dev@esk.fraunhofer.de> (RLF extensions)
   */

Here, there were two original authors, and then a third added a substantial
new feature (RLF extensions).

Please work with maintainers to balance the competing concerns of
obtaining proper attribution and avoiding long headers.

Coding style
************
We ask that all contributors make their code conform to the 
coding standard which is outlined in :ref:`Coding style`.

The project maintains a Python program called ``check-style.py`` found in
the ``utils/`` directory.  This is a wrapper around the ``uncrustify``
utility with configuration set to the conventions used by |ns3|, and
can be used to quickly format new source code files proposed for the
mainline. 

Creating a patch
****************
Patches are preferably submitted as a GitLab.com
`Merge Request <https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html>`_.
Short patches can be attached to an issue report or sent to the mailing-lists,
but a Merge Request is the best way to submit.

The UNIX diff tool is the most common way of producing a patch: a patch is 
a text-based representation of the difference between two text files or two 
directories with text files in them. If you have two files, 
``original.cc``, and, ``modified.cc``, you can generate a patch with the 
command ``diff -u original.cc modified.cc``. If you wish to produce a patch 
between two directories, use the command ``diff -uprN original modified``.

Make sure you specify to the reviewer where the original files came from 
and make sure that the resulting patch file does not contain unexpected 
content by performing a final inspection of the patch file yourself.

Patches such as this are sufficient for simple bug fixes or very simple 
small features.

Git can be used to create a patch of what parts differ from the last 
committed code; try::

  $ git diff

The output of `git diff` can be redirected into a patch file:

::

  $ git diff > proposal.patch

Keep in mind that `git diff` could include unrelated changes made locally to
files in the repository (a common example is `.vscode/launch.json`). In order
to avoid cluttering, please amend your diff file using a normal text editor
before submitting it.

Likewise, if you submit a merge request using GitLab, please add only the
changes to the relevant files to the branch you're using for the merge request.

Maintainers
***********
Maintainers are the set of people who make decisions on code and documentation
changes.  Maintainers are contributors who have demonstrated, over time,
knowledge and good judgment as pertains to contributions to |ns3|, and who
have expressed willingness to maintain some code.  |ns3| is like other 
open source projects in terms of how people gradually become maintainers
as their involvement with the project deepens; maintainers are not newcomers
to the project.

The list of maintainers for each module is found here:
https://www.nsnam.org/developers/maintainers/

Maintainers review code (bug fixes, new models) within scope of their 
maintenance responsibility.  A maintainer of a module should "sign off"
(or approve of) changes to an |ns3| module before it is committed to
the main codebase.  Note that we typically do not formalize the signing
off using Git's sign off feature, but instead, maintainers will indicate
their approval of the merge request using GitLab.com.

Note that some modules do not have active maintainers; these types of 
modules typically receive less maintenance attention than those with
active maintainers (and bugs may linger unattended).

The best way to become a maintainer is to start by submitting patches
that fix open bugs or otherwise improve some part of the simulator, and
to join in on the technical discussions.
People who submit quality patches will catch the attention of the maintainers
and may be asked to become one at some future date.

People who ask to upstream a new module or model so that it is part of the
main |ns3| distribution will typically be asked to maintain it going forward
(or to find new maintainers for it).

