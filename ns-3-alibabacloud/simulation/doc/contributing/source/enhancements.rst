.. include:: replace.txt
.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

.. _Enhancements:

Submitting enhancements
-----------------------

This chapter covers how to submit fixes and small patches for the existing
mainline code and its documentation.

Enhancements (new models) can be proposed for the mainline, and
maintainers will make a decision on whether to include it as mainline
or recommend that it be supported in the ns-3 App Store.  Documentation
on hosting code in the ns-3 App Store is provided in the next
chapter (:ref:`External`). This chapter provides guidance on submitting
code for inclusion in the mainline (and much of it applies also as
best practice for app store code).

GitLab.com trackers
*******************

|ns3| uses two trackers to keep track of known issues or submitted code.
Maintainers prefer to list everything on the tracker so that issues do not
slip through the cracks.  Users are encouraged to use these to report
or comment on issues or merge requests; this requires users to obtain
a GitLab.com account.

If a user wants to report an issue with |ns3|, please first search
the `issue tracker <https://gitlab.com/nsnam/ns-3-dev/-/issues>`_ for
something that may be similar, and if nothing is found, please report
the new issue.

If a user wants to submit proposed new code for |ns3|, please 
submit on the `merge request tracker <https://gitlab.com/nsnam/ns-3-dev/-/merge_requests>`_.

More details for each are provided below.  Similarly, users who want to
report issues on other related repositories under the `nsnam` project
(such as the `Bake build system <https://gitlab.com/nsnam/bake>`_)
should follow similar steps there.

Reporting issues
****************

Issues can be reported against the code or the documentation, if you believe
that something is incorrect or could be improved.  The key to reporting
an issue with the code is to try to provide as much information as possible
to allow a maintainer to quickly reproduce the problem.  After you've
determined which module your bug is related to, if it is inside the
official distribution (mainline), then create an issue, label it with the
name of the module, and provide as much information as possible.

First, perform a cursory search on the 
`open issue list <https://gitlab.com/nsnam/ns-3-dev/-/issues>`_ to see if
the problem has already been reported. If it has and the issue is still
open, add a comment to the existing issue instead of opening a new one.

If you are reporting an issue against an older version of |ns3|, please
scan the most recent
`Release Notes <https://gitlab.com/nsnam/ns-3-dev/-/blob/master/RELEASE_NOTES.md>`_ to see if it has been fixed since that release.

If you then decide to list an issue, include details of your environment:

1. Which version of ns-3 are you using?

2. What's the name and version of the OS you're using?

3. Which modules do you have installed?

4. Have you modified ns-3-dev in any way?

Here are some additional guidelines:

1. Use a clear and descriptive title for the issue to identify the problem.

2. Describe the exact steps which reproduce the problem in as many details
   as possible. For example, start by explaining how you coded your script;
   e.g. which functions were called in what order, or else provide an example
   program.  If your program includes modifications to the ns-3 core or a
   module, please list them (or even better, provide the diffs).

3. Provide specific examples to demonstrate the steps. Include links to
   files or projects, or copy/pasteable snippets, which you use in those
   examples. If you're providing snippets in the issue, use Markdown code
   block formatting.

4. Describe the behavior you observed after following the steps and point out
   what exactly is the problem with that behavior.  Explain which behavior you 
   expected to see instead and why.

5.  If you're reporting that ns-3 crashed, include a crash report with a
    stack trace from the operating system. On macOS, the crash report will
    be available in Console.app under 
    `"Diagnostic and usage information" > "User diagnostic reports"`. Include
    the crash report in the issue in a code block, or a file attachment.

6.  If the problem is related to performance or memory, include a CPU
    profile capture with your report.

Some issues have suggested resolutions that are trivial and do not
require submitting a merge request.  For more complicated resolutions,
if you have a patch to propose, either attach it to the issue, or submit a
merge request (described next).

Submitting merge requests
*************************

To submit code proposed for |ns3| as one or more commits, use a merge
request.  The following steps are recommended to smooth the process.

If you are new to public Git repositories, you may want to read
`this overview of merge requests <https://docs.gitlab.com/ee/user/project/merge_requests/getting_started.html>`_.  If you are familiar with GitHub
pull requests, the GitLab.com merge requests are very similar.

In brief, you will want to fork ns-3-dev into your own namespace (i.e.,
fork the repository into your personal GitLab.com account, via the user 
interface on GitLab.com), clone your fork of ns-3-dev to your local machine,
create a new feature branch that is based on the
current tip of ns-3-dev, push that new feature branch up to your fork,
and then from the GitLab.com user interface, generate a Merge Request back
to the |ns3| mainline.  You will want to monitor and respond to any
comments from reviewers, and try to resolve threads.

Remember the documentation
==========================

If you add or change API to the simulator, please include `Doxygen <https://www.doxygen.nl>`_ changes as appropriate.  Please scan the module documentation
(written in `Restructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ in the `docs` directory) to check if an update is needed to align with the patch proposal. 

Commit message format
=====================

Commit messages should be written as follows.  For examples, please look
at the output of `git log` command.

1.  The author string should be formatted such as "John Doe <john.doe@example.com>".  It is a good idea to run
`git config <https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration>`_ on your machine, or hand-edit the `.gitconfig` file in your home directory,
to ensure that your name and email are how you want them to be displayed.

.. raw:: latex

    \newpage

2. The first line of the commit message should be limited to 72 columns if possible.  This is not a hard requirement but a preference.  If you prefer to add
more detail, you can add subsequent message lines below the first one, separated by a blank line.  Example:

.. code-block:: text

  commit e6ca9be6fb5a0592a44967f7885545dce3a6da1a
  Author: Rediet <getachew.redieteab@orange.com>
  Date:   Wed May 19 16:34:01 2021 +0200

    lte: Assign default values
    
    Fixes crashing optimized/release builds with 'may be used uninitialized' error


3. The first line of the commit message should include the relevant module name or names, separated by a colon.  Example: 

.. code-block:: text

  commit 15ab50c03132a5f7686045014c6bedf10ac7d421
  Author: Stefano Avallone <stavallo@unina.it>
  Date:   Wed Jan 27 14:58:54 2021 +0100

    wifi,wave,mesh: Rescan python bindings


4. If the commit fixes an issue in the issue tracker, list it in parentheses
   after the colon (by saying 'fixes #NNN' where NNN is the issue number).
   This reference alerts future readers to an issue where
   more may be discussed about the commit.  Example:

.. code-block:: text

  commit 10ef08140ab2a9f2b550f24d1e881e76ea0873ff
  Author: Tom Henderson <tomh@tomh.org>
  Date:   Fri May 21 11:11:33 2021 -0700

    network: (fixes #404) Use Queue::Dispose() for SimpleNetDevice::DoDispose()


5. If the commit is from a merge request, that may also be added in a similar
   way the same by saying 'merges !NNN'.  The exclamation point differentiates
   merge requests from issues (which use the number sign '#') on GitLab.com. 
   Example:

.. code-block:: text

  commit d4258b2b32d6254b878eca9200271fa3f4ee7174
  Author: Tom Henderson <tomh@tomh.org>
  Date:   Sat Mar 27 09:56:55 2021 -0700

    build: (merges !584) Exit configuration if path whitespace detected


Here is an example making use of both:

.. code-block:: text

  commit a97574779b575af70d975f9e2ca899e2405cf497
  Author: Federico Guerra <federico@guerra-tlc.com>
  Date:   Tue Jan 14 21:14:37 2020 +0100

    uan: (fixes #129, merges !162) EndTx moved to PhyListener


6. Use the present tense ("Add feature", not "Added feature") and the imperative mood ("Move cursor to ...", not "Moves cursor to...").

No extraneous white space
=========================

Do not make whitespace changes to unrelated parts of the code (unrelated
to your merge request).  This can often happen if the `check-style.py`
program is used on existing files.  Use the automated style correction
program `check-style.py` only on your *new* source code files.

Squashing your history
======================

In the course of developing and responding to review comments, you may
add more commits, so what started out as a single commit might grow into
several.  Please consider to squash any such revisions if they do not
need to be preserved as separate commits in the mainline Git history.

If you squash commits, you must force-push your branch back to your fork.
Do not worry about this; GitLab.com will update the Merge Request
automatically.  This `tutorial <https://docs.gitlab.com/ee/topics/git/git_rebase.html>`_ may be helpful to learn about Git rebase, force-push, and merge
conflicts.

Note that GitLab can squash the commits while merging. However, it is often
preferred to keep multiple commit messages, especially when the merge request
contains multiple parts or multiple authors.

It is a good practice to NOT squash commits while the merge request is being
reviewed and updated (this helps the reviewers), and perform a selective
squash before the merge.

Rebasing on ns-3-dev
====================

It is also helpful to maintainers to keep your feature branch up to date
so that the commits are appended to the tip of the mainline code.  This
is not strictly required; maintainers may do such a rebase upon merging
your finalized Merge Request.  This may help catch possible merge conflicts
before the time to merge arrives.

Note that sometimes it is not possible to rebase a merge request through
GitLab's web interface. Hence, it is a good practice to keep your merge
request in line with the mainline (i.e., rebase it periodically and push
the updated branch).

Resolving discussion threads
============================

Any time someone opens a new comment thread on a Merge Request, a counter
of 'Unresolved threads' is incremented (near the top of the Merge Request).
If you are able to successfully resolve the comment thread (either by
changing your code, or convincing the reviewer that no change is needed),
then please mark the thread as resolved.  Maintainers will look at the
count of unresolved threads and make decisions based on this count as to
whether the Merge Request is ready.  Maintainers prefer that all threads
are resolved successfully before moving forward with a merge.

Adding a label
==============
You can use labels to indicate whether the Merge Request is a bug, pertains
to a specific module or modules, is documentation related, etc.  This is
not required; if you do not add a label, a maintainer probably will.

Other metadata
==============

It is not necessary to set other metadata on the Merge Request such as
milestone, reviewers, etc.

Feature requests
****************

Feature requests are tracked as `GitLab.com issues <https://gitlab.com/nsnam/ns-3-dev/-/issues>`_.  If you want to suggest an enhancement, create an issue and provide the following information:

1. Use a clear and descriptive title for the issue to identify the suggestion.

2. Provide a step-by-step description of the suggested enhancement in as many details as possible.

3. Provide specific examples to demonstrate the steps. Include copy/pasteable snippets which you use in those examples.

4. Describe the current behavior and explain which behavior you expected to see instead and why.

5. Explain why this enhancement would be useful to most ns-3 users.

The |ns3| project does not have professional developers available to respond
to feature requests, so your best bet is to try to implement it yourself and
work with maintainers to improve it, but the project does like to hear back
from users about what would be a useful improvement, and you may find
like-minded collaborators in the community willing to work on it with you.

Use the `enhancement` Label on your feature request.
