# Contributing to ns-3

*This file is heavily inspired by Atom's [CONTRIBUTING.md file](https://raw.githubusercontent.com/atom/atom/master/CONTRIBUTING.md).*

The following is a set of guidelines for contributing to ns-3, which are hosted in the [nsnam organization](https://gitlab.com/nsnam) on GitLab.com. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a merge request.

#### Table Of Contents

[I do not want to read this whole thing, I just have a question!!!](#i-do-not-want-to-read-this-whole-thing-i-just-have-a-question)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)
  * [ns-3 Documentation](#ns-3-documentation)
  * [ns-3 modules](#ns-3-modules)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Merge Requests](#merge-requests)

[Styleguides](#styleguides)
  * [Git Commit Messages](#git-commit-messages)
  * [C++ Styleguide](#c-styleguide)
  * [Documentation Styleguide](#documentation-styleguide)

[Additional Notes](#additional-notes)
  * [Issue and Merge Request Labels](#issue-and-merge-request-labels)


## I do not want to read this whole thing I just have a question!!!

> **Note:** Please don't file an issue to ask a question.
            You'll get faster results by using the resources below.

We have an official message board where the community chimes in with helpful advice if you have questions.

* [ns-3-users, the official ns-3 message board](https://groups.google.com/forum/#!forum/ns-3-users)
* [ns-3 manual](https://www.nsnam.org/docs/manual/html/index.html)

If chat is more your speed, you can join the ns-3 Zulip channel:

* [Join the ns-3 Zulip chat](https://ns-3.zulipchat.com/)
    * Even though Zulip is a chat service, sometimes it takes several hours
      for community members to respond &mdash; please be patient!
    * Use the `#general` channel for general questions or discussion about ns-3
    * Use the `#GSoC` channel for questions about GSoC
    * There are many other channels available, check the channel list

## What should I know before I get started?

### ns-3 Documentation

The ns-3 project maintains the documentation in different places, dependings on the need. The documentation that is current with the development tree is the following:

* [Tutorial](https://www.nsnam.org/docs/tutorial/html/index.html): Its purpose is to get you started, with simple examples, into the ns-3 world.
* [Manual](https://www.nsnam.org/docs/manual/html/index.html) is a descriptive documentation of the ns-3 core capabilities, as well as shared development practices.
* [Model library](https://www.nsnam.org/docs/models/html/index.html), contains a list of manuals, one for each module we officially ship.
* [Doxygen](https://www.nsnam.org/docs/doxygen/index.html), contains the programming interface of ns-3, as well as description of classes, methods, and functions that permit the interaction between different modules, as well as with the user code.

### ns-3 modules

ns-3 is an open source project &mdash; it's made up of [49 modules](https://gitlab.com/nsnam/ns-3-dev/tree/master/src). When you initially consider contributing to ns-3, you might be unsure about which of those 49 modules implements the functionality you want to change or report a bug for. This section should help you with that.

ns-3 is intentionally very modular, and it's possible that a feature you need or a technology you may want to use is not coming from the official repository, but rather from a [community maintained module](https://apps.nsnam.org/). Each community module has its own repository too, and it should contain the information on how to integrate that code with ns-3. If you are looking, instead, more information on how to work with ns-3's official modules, please take a look first to the [model documentation](https://www.nsnam.org/docs/models/html/index.html).

### Design Decisions

When we make a significant decision in how we maintain the project and what we can or cannot support, we will document it through the [ns-developers mailing list](https://mailman.isi.edu/mailman/listinfo/ns-developers). If you have a question around how we do things, please subscribe and lurk around to see how we proceed. If you have a development problem, please open a new topic and ask your question.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for ns-3. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

Before creating bug reports, please check [this list](#before-submitting-a-bug-report-to-be-completed) as you might find out that you don't need to create one. When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report).

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report (to be completed)

* **Perform a [cursory search](https://gitlab.com/nsnam/ns-3-dev/issues)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.
* **Check the [release notes](RELEASE_NOTES.md)** for a list of known bugs.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitLab issues](https://docs.gitlab.com/ee/user/project/issues/). After you've determined [which module](#ns-3-modules) your bug is related to, if it is inside the official distribution then create an issue, label it with the name of the module, and provide as much information as possible.

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. For example, start by explaining how you coded your script, e.g. which functions exactly you called. When listing steps, **don't just say what you did, but explain how you did it**. For example, if your program includes modifications to the ns-3 core or a module, please list them (or even better, provide the diffs).
* **Provide specific examples to demonstrate the steps**. Include links to files or projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://docs.gitlab.com/ee/user/markdown.html).
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **If you're reporting that ns-3 crashed**, include a crash report with a stack trace from the operating system. On macOS, the crash report will be available in `Console.app` under "Diagnostic and usage information" > "User diagnostic reports". Include the crash report in the issue in a [code block](https://docs.gitlab.com/ee/user/markdown.html), or a file attachment.
* **If the problem is related to performance or memory**, include a CPU profile capture with your report.

Include details about your configuration and environment:

* **Which version of ns-3 are you using?**
* **What's the name and version of the OS you're using**?
* **Which [modules](#ns-3-modules) do you have installed?**

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for ns-3, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion :pencil: and find related suggestions :mag_right:.

Before creating enhancement suggestions, please check [this list](#before-submitting-an-enhancement-suggestion) as you might find out that you don't need to create one. When you are creating an enhancement suggestion, please [include as many details as possible](#how-do-i-submit-a-good-enhancement-suggestion).

#### Before Submitting An Enhancement Suggestion

* **Check if there's already [a merge request](https://gitlab.com/nsnam/ns-3-dev/merge_requests) which provides that enhancement.**
* **Determine [which module the enhancement should be suggested in](#ns-3-modules).**
* **Perform a [cursory search](https://gitlab.com/nsnam/ns-3-dev/issues)** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [Gitlab issues](https://docs.gitlab.com/ee/user/project/issues/). After you've determined [which module](#ns-3-modules) your enhancement suggestion is related to, create an issue on that repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples.
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Explain why this enhancement would be useful** to most ns-3 users.
* **Specify the name and version of the OS you're using.**

### Your First Code Contribution

Unsure where to begin contributing to ns-3? You can start by looking through these `good first issue` and `project ideas` issues:

* [Good first issue](https://gitlab.com/nsnam/ns-3-dev/-/issues?label_name%5B%5D=good+first+issue) - issues which should only require a few lines of code, and a test or two.
* [Project ideas](https://gitlab.com/nsnam/ns-3-dev/-/issues?label_name%5B%5D=project+ideas) - issues which should be a bit more involved than `good first issue`.

Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.

#### Local development

ns-3 and all packages can be developed locally. For instructions on how to do this, see the following sections in the [ns-3 Manual](https://www.nsnam.org/docs/manual/html/index.html):

* Contributing to ns-3 with git

### Merge Requests

The process described here has several goals:

- Maintain ns-3's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible ns-3
- Enable a sustainable system for ns-3's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow the [styleguides](#styleguides)
2. After you submit your merge request, verify that all status checks are passing <details><summary>What if the status checks are failing?</summary>If a status check is failing, and you believe that the failure is unrelated to your change, please leave a comment on the merge request explaining why you believe the failure is unrelated. A maintainer will re-run the status check for you. If we conclude that the failure was a false positive, then we will open an issue to track that problem with our status check suite.</details>

While the prerequisites above must be satisfied prior to having your merge request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your merge request can be ultimately accepted.

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues in the first line, by prepending `[#issue]`

### C++ Styleguide

We maintain our code style through the webpage [Code Style](https://www.nsnam.org/develop/contributing-code/coding-style/) and as a `clang-format` configure [file](https://gitlab.com/nsnam/ns-3-dev/blob/master/.clang-format)

### Documentation Styleguide

* Use [Doxygen](http://www.doxygen.nl/) for what regards the code.
* Use [ReStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for what regards the model (module) documentation.

## Additional Notes

### Issue and Merge Request Labels

This section lists the labels we use to help us track and manage issues and merge requests.

The labels are loosely grouped by their purpose, but it's not required that every issue have a label from every group or that an issue can't have more than one label from the same group.

Please open an issue on `infrastructure` if you have suggestions for new labels, and if you notice some labels are missing.

#### Type of Issue and Issue State

| Label name | Description |
| --- | --- |
| `enhancement` | Feature requests. |
| `bug` | Confirmed bugs or reports that are very likely to be bugs. |
| `question` | Questions more than bug reports or feature requests (e.g. how do I do X). |
| `feedback` | General feedback more than bug reports or feature requests. |
| `help-wanted` | The ns-3 team would appreciate help from the community in resolving these issues. |
| `beginner` | Less complex issues which would be good first issues to work on for users who want to contribute to ns-3. |
| `more-information-needed` | More information needs to be collected about these problems or feature requests (e.g. steps to reproduce). |
| `needs-reproduction` | Likely bugs, but haven't been reliably reproduced. |
| `blocked` | Issues blocked on other issues. |
| `duplicate` | Issues which are duplicates of other issues, i.e. they have been reported before. |
| `wontfix` | The ns-3 team has decided not to fix these issues for now, either because they're working as intended or for some other reason. |
| `invalid` | Issues which aren't valid (e.g. user errors). |

#### Topic Categories

| Label name | Description |
| --- | --- |
| `windows` | Related to ns-3 running on Windows. |
| `linux` | Related to ns-3 running on Linux. |
| `mac` | Related to ns-3 running on macOS. |
| `documentation` | Related to any type of documentation (doxygen, manual, models) |
| `git` | Related to Git functionality (e.g. problems with gitignore files or with showing the correct file status). |
| `infrastructure` | Related to gitlab infrastructure. |
| `utils` | Related to scripts and various utils for ns-3 repository or code maintenance. |

#### Merge Request Labels

| Label name | Description
| --- | --- |
| `work-in-progress` | Merge requests which are still being worked on, more changes will follow. |
| `needs-review` | Merge requests which need code review, and approval from maintainers or ns-3 core team. |
| `under-review` | Merge requests being reviewed by maintainers or ns-3 core team. |
| `requires-changes` | Merge requests which need to be updated based on review comments and then reviewed again. |
| `needs-testing` | Merge requests which need manual testing. |

### Module Labels

| Label name | Description |
| --- | --- |
| `animator` | Any issue related to this module |
| `antenna` | Any issue related to this module |
| `aodv` | Any issue related to this module |
| `applications` | Any issue related to this module |
| `bridge` | Any issue related to this module |
| `brite` | Any issue related to this module |
| `build system` | Any issue related to this module |
| `buildings` | Any issue related to this module |
| `click` | Any issue related to this module |
| `config-store` | Any issue related to this module |
| `core` | Any issue related to this module |
| `csma` | Any issue related to this module |
| `designer` | Any issue related to this module |
| `devices` | Any issue related to this module |
| `dsdv` | Any issue related to this module |
| `dsr` | Any issue related to this module |
| `emulation` | Any issue related to this module |
| `energy` | Any issue related to this module |
| `examples` | Any issue related to this module |
| `fd-net-device` | Any issue related to this module |
| `flow-monitor` | Any issue related to this module |
| `global-routing` | Any issue related to this module |
| `general` | Any issue related to this module |
| `helpers` | Any issue related to this module |
| `internet` | Any issue related to this module |
| `internet-apps` | Any issue related to this module |
| `ipv6` | Any issue related to this module |
| `lr-wpan` | Any issue related to this module |
| `lwip` | Any issue related to this module |
| `lte` | Any issue related to this module |
| `mesh` | Any issue related to this module |
| `mobility-models` | Any issue related to this module |
| `mpi` | Any issue related to this module |
| `netanim` | Any issue related to this module |
| `network` | Any issue related to this module |
| `nix-vector` | Any issue related to this module |
| `nsc-tcp` | Any issue related to this module |
| `olsr` | Any issue related to this module |
| `openflow` | Any issue related to this module |
| `point-to-point` | Any issue related to this module |
| `python bindings` | Any issue related to this module |
| `propagation` | Any issue related to this module |
| `quagga` | Any issue related to this module |
| `regression` | Any issue related to this module |
| `routing` | Any issue related to this module |
| `sixlowpan` | Any issue related to this module |
| `spectrum` | Any issue related to this module |
| `stats` | Any issue related to this module |
| `tap-bridge` | Any issue related to this module |
| `tcp` | Any issue related to this module |
| `test framework` | Any issue related to this module |
| `topology-read` | Any issue related to this module |
| `traffic-control` | Any issue related to this module |
| `uan` | Any issue related to this module |
| `virtual-net-device` | Any issue related to this module |
| `visualizer` | Any issue related to this module |
| `wave module` | Any issue related to this module |
| `wifi` | Any issue related to this module |
| `wimax` | Any issue related to this module |
