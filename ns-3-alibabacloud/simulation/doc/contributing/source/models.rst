.. include:: replace.txt
.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

.. _Models:

Submitting new models
---------------------

We actively encourage submission of new features to |ns3|. 
Independent submissions are essential for open source projects, and if
accepted into the mainline, you will also be credited as an author of
future versions of |ns3|. However, 
please keep in mind that there is already a large burden on the |ns3|
maintainers to manage the flow of incoming contributions and maintain new 
and existing code. The goal of this chapter is to outline the options for
new models in |ns3|, and how you can help to minimize the burden on maintainers
and thus minimize the average time-to-merge of your code.

Options for new models
**********************

|ns3| is organized into modules, each of which is compiled and linked into
a separate library.  Users can selectively enable and disable the inclusion
of modules (via the `--enable-modules` argument to ns3 configure, or via
the selective inclusion of contributed modules).

When proposing new models to |ns3|, please keep in mind that not all models
will be accepted into the mainline.  However, we aim to provide at least
one or more options for any code contribution.

Because of the long-term maintenance burden, |ns3| is no longer accepting
all new proposals into the mainline.  Features that are of general
interest are more likely to be approved for the mainline, but features that
are more specialized may be recommended for the `ns-3 App Store <https://apps.nsnam.org>`_.  Some modules that have been in the mainline for a long time, but
fall out of use (or lose their maintainers) may also be moved out of the
mainline into the App Store in the future.

The options for publishing new models are:

1. Propose a Merge Request for the |ns3| mainline and follow the guidelines below.

2. Organize your code as a "contributed module" or modules and maintain them in 
   your own public Git repository.  A page on the App Store can be made to
   advertise this to users, and other tooling can be used to ensure that the
   module stays compatible with the mainline.

3. Organize your code as a "public fork" that evolves separately from the
   |ns3| mainline.  This option is sometimes chosen for models that require
   significant intrusive changes to the |ns3| mainline to support.  Some
   recent examples include the Public Safety models and the millimeter-wave
   extensions to |ns3|.  This has the benefit of being self-contained, but
   the drawback of losing compatibility with the mainline over time.  A page
   on the App Store can be made for these forks as well.  For maintenance
   reasons and improved user experience, we prefer to upstream mainline
   changes so that public forks can be avoided.

4. Archive your code somewhere, or publish in a Git repository, and link to
   it from the |ns3| `Contributed Code <https://www.nsnam.org/wiki/Contributed_Code>`_ 
   wiki page.  This option requires the least amount of work from the
   contributor, but visibility of the code to new |ns3| users will likely be
   reduced.  To follow this route, obtain a wiki account from the webmaster,
   and make edits as appropriate to link your code.

The remainder of the chapter concerns option 1 (upstreaming to ns-3-dev); the
other options are described in the next chapter (:ref:`External`).

Upstreaming new models
**********************
The term "upstreaming" refers to the process whereby new code is 
contributed back to an upstream source (the main open source project)
whereby that project takes responsibility for further enhancement and
maintenance.

Making sure that each code submission fulfills as many items 
as possible in the following checklist is the best way to ensure quick 
merging of your code.

In brief, we can summarize the guidelines as follows:

1.  Be licensed appropriately (see :ref:`General`)
2.  Understand how and why |ns3| conducts code reviews before merging
3.  Follow the ns-3 coding style (:ref:`Coding style`) and software
    engineering and consistency feedback that maintainers may provide
4.  Write associated documentation, tests, and example programs

If you do not have the time to follow through the process to include your 
code in the main tree, please see the next chapter (:ref:`External`)
about contributing ns-3 code that is not maintained in the main tree.

The process can take a long time when submissions are large or when
contributors skip some of these steps.  Therefore, best results are found 
when the following guidelines are followed:

* Ask for review of small chunks of code, rather than large patches.  Split
  large submissions into several more manageable pieces.
* Make sure that the code follows the guidelines (coding style,
  documentation, tests, examples) or you may be asked to fix these
  things before maintainers look at it again.

Code reviews
************

Code submissions to ns-3-dev are expected to go through a review process
where one or more maintainers comment on the code.  The basic purpose of the
code reviews is to ensure the long-term maintenance and overall system
integrity of ns-3.  Contributors may be asked
to revise their code or rewrite portions during this process. New features are
also typically only merged during the early stages of a new release cycle,
to avoid destabilizing code before release.

|ns3| code reviews are conducted using GitLab.com merge requests, possibly
supported by discussion on the ns-developers mailing list for reviews that
involve code that cuts across maintainer boundaries or is otherwise
controversial.  Many examples of ongoing reviews can be browsed on
our `GitLab.com site <https://gitlab.com/nsnam/ns-3-dev/-/merge_requests>`_.

Submission structure
********************

For each code submission, include a description of what your code is doing,
and why. Ideally, you should be able to provide a summary description in a
short paragraph at the top of the Merge Request.  If you want to flag
to maintainers that your submission is known to be incomplete (e.g., you
are looking for early feedback), preface the title of the Merge Request
with ``Draft:``.

Coherent and multi-part submissions
===================================

Large code submissions should be split into smaller submissions where possible.
The likelihood of getting maintainers to review your submission is inversely
proportional to its size, because large code reviews require a large block
of time for a maintainer.  For instance, if you are submitting two
routing protocol models and each protocol model can stand on its own,
submit two Merge Requests rather than a single one containing both models.

Each submission should be a coherent whole: if you need to edit ten files to
get a feature to work, then, the submission should contain all the changes
for these ten files.  Of course, if you can split the feature in sub-features,
then, you should do it to decrease the size of the submission as per the
previous paragraph.

For example, if you have made changes to optimize one module, and in the
course of doing so, you fixed a bug in another module, make sure you separate
these two sets of changes in two separate submissions.

When splitting a large submission into seprate submissions, (ideally) these
submissions should be prefaced (in the Merge Request description) by a
detailed explanation of the overall plan
such that code reviewers can review each submission separately but within a
larger context. This kind of submission typically is split in multiple dependent
steps where each step depends on the previous one. If this is the case,
make it very clear in your initial explanation. If you can, minimize
dependencies between each step such that reviewers can merge each step
separately without having to consider the impact of merging one submission
on other submissions.

History rewriting
=================

It is good practice to rebase your feature branch onto the tip of the
(evolving) ns-3-dev, so that the commits appear on top of a recent commit
of ns-3-dev.  For instance, you may have started with an initial Merge
Request, and it has been some time to gather feedback from maintainers,
and now you want to update your Merge Request with that feedback.  A
rebase onto the current ns-3-dev ``master`` branch is good practice
for any Merge Request update.

In addition, in the course of your development, and in responding to
reviewer comments, the git commit log will accumulate lots of small
commits that are not helpful to future maintainers:  they make it more painful
to use the annotate and bisect commands.  For this reason, it is good
practice to "squash" and "rebase" commits into coherent groups.  For example,
assume that the commit history, after you have responded to a reviewer's
comment, looks like this:

::

  commit 4938c01400fb961c37716c3f415f47ba2b04ab5f (HEAD -> master, origin/master, origin/HEAD)
  Author: Some Contributor <some.contributor@example.com>
  Date:   Mon Jun 15 11:19:49 2020 -0700

    Fix typo in Doxygen based on code review comment

  commit 915cf65cb8f464d9398ddd8bea9af167ced64663
  Author: Some Contributor <some.contributor@example.com>
  Date:   Thu Jun 11 16:44:46 2020 -0400

    Add documentation, tests, and examples for clever protocol

  commit a1d51db126fb3d1c7c76427473f02ee792fdfd53
  Author: Some Contributor <some.contributor@example.com>
  Date:   Thu Jun 11 16:35:48 2020 -0400

    Add new models for clever protocol

In the above case, it will be helpful to squash the top commit about the
typo into the first commit where the Doxygen was first written, leaving
you again with two commits.  `Git interactive rebase <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History>`_ is a good tool for this;
ask a maintainer for help in doing this if you need some help to learn
this process.

Documentation, tests, and examples
**********************************

Often we have observed that contributors will provide nice new classes
implementing a new model, but will fail to include supporting test code,
example scripts, and documentation. Therefore, we ask people submitting the
code (who are in the best position to do this type of work) to provide
documentation, test code, and example scripts.

Note that you may want to go through multiple phases of code reviews, and
all of this supporting material may not be needed at the first stage (e.g.
when you want some feedback on public API header declarations only, before 
starting the implementation). However, when it times come to merge your code,
you should be prepared to provide these things, as fits your contribution
(maintainers will provide some guidance here).

Including examples
==================
For many submissions, it will be important to include at least one example that exercises the new code, so the reviewer understands how it is intended to be used. For final submission, please consider to add as many examples as you can that will help new users of your code. The <tt>samples/</tt> and <tt>examples/</tt> directories are places for these files.

Tests
=====

All new models should include automated tests.  These should, where
appropriate, be unit tests of model correctness, validation tests of
stochastic behavior, and overall system tests if the model's interaction with
other components must be tested.  The test code should try to cover as much
model code as possible, and should be checked as much as possible by
regression tests.

The ns-3 manual provides documentation and suggestions for
`how to write tests <https://www.nsnam.org/docs//manual/html/how-to-write-tests.html>`_.

Documentation
=============

If you add a new features, or make changes to existing features, you need to
update existing or write new documentation and example code. Consider the
following checklist:

* Doxygen should be added to header files for all public classes and methods, and should be checked for Doxygen errors
* new features should be described in the ``RELEASE_NOTES.md``
* public API changes (if any) must be documented in ``CHANGES.html``
* new API or changes to existing API must update the inline doxygen documentation in header files
* consider updating or adding a new section to the manual (``doc/manual``) or
  model library (``doc/models``) as appropriate
* update the ``AUTHORS`` file for any new authors

Module dependencies
===================

Since |ns3| is a modular system, each module has (typically) dependencies on
other modules. These must be kept to the minimum necessary, and circular
dependencies must be avoided (they will generate errors).
As an example, the ``lr-wpan`` module depends on ``core``, ``network``,
``mobility``, ``spectrum``, and ``propagation``. Tests for your new module
must not add any other dependency (that was not needed for the model or
helper code itself), but examples can use other modules because their module
dependencies are handled independently. As an example, the ``lr-wpan`` examples
also use the ``stats`` module.

This rule might pose limitations to the authoring of tests. In such a case,
ask a maintainer for suggestions, or check existing tests to find solutions.
Sometimes tests that involve adding additional dependencies are placed into
the ``src/test`` subdirectory.
