.. include:: replace.txt
.. highlight:: bash


Working with gitlab-ci-local
----------------------------

.. _continuos integration (CI) : https://docs.gitlab.com/ee/ci/
.. _pipelines : https://docs.gitlab.com/ee/ci/introduction/index.html#continuous-integration
.. _daily and weekly pipelines : https://gitlab.com/nsnam/ns-3-dev/-/pipeline_schedules
.. _crypto miners abuse : https://about.gitlab.com/blog/2021/05/17/prevent-crypto-mining-abuse/
.. _GitLab-CI-local : https://github.com/firecow/gitlab-ci-local
.. _GitLab CI : https://docs.gitlab.com/ee/ci/
.. _Docker : https://docs.docker.com/desktop/
.. _rootless mode : https://docs.docker.com/engine/security/rootless/

The ns-3 project repository is currently hosted in GitLab, which includes
`continuos integration (CI)`_ tools to automate build, tests, packaging and 
distribution of software. The CI works based on jobs, that are defined 
on YAML files. 

The ns-3 GitLab CI files are located in ``ns-3-dev/utils/tests/``. 
The main GitLab CI file is ``gitlab-ci.yml``. The different jobs
are used to check if a multitude of compilers and package versions
are compatible with the current ns-3 build, which is why a build is 
usually followed by a test run. Other CI jobs build and warn about 
missing the documentation. 

The GitLab CI jobs are executed based on `pipelines`_ containing a 
sequence of job batches. Jobs within a batch can be executed in parallel.
These `pipelines`_ can be triggered manually, or scheduled to run automatically 
per commit and/or based on a time period 
(ns-3 has `daily and weekly pipelines`_ scheduled). 

The GitLab CI free tier is very slow, taking a lot of time to identify
issues during active merge request development. 

Note: the free tier
now requires a credit card due to `crypto miners abuse`_.

`GitLab-CI-local`_ is a tool that allows an user to use the `GitLab CI`_
configuration files locally, allowing for the debugging of CI settings
and pipelines without requiring pushes to test repositories or main
repositories that fill up the CI job queues with failed jobs due to
script errors. 

GitLab-CI-local relies on `Docker`_ to setup the environment to execute
the jobs. 

Note: Docker is usually setup in root mode, requiring 
frequent use of administrative permissions/sudo. However,
this is highly discouraged. You can configure Docker to run 
in `rootless mode`_. From this point onwards, we assume Docker is configured 
in `rootless mode`_.

After installing both `Docker`_ in `rootless mode`_ and `GitLab-CI-local`_, 
the ns-3 jobs can be listed using the following command:

.. sourcecode:: bash

    ~/ns-3-dev$ gitlab-ci-local --file ./utils/tests/gitlab-ci.yml --list
    parsing and downloads finished in 226 ms
    name                                   description  stage          when        allow_failure  needs
    weekly-build-ubuntu-18.04-debug                     build          on_success  false      
    
    ...  

    weekly-build-clang-11-optimized                     build          on_success  false      
    pybindgen                                           build          on_success  false      
    per-commit-compile-debug                            build          on_success  false      
    per-commit-compile-release                          build          on_success  false      
    per-commit-compile-optimized                        build          on_success  false      
    daily-test-debug                                    test           on_success  false      
    daily-test-release                                  test           on_success  false      
    daily-test-optimized                                test           on_success  false      
    daily-test-optimized-valgrind                       test           on_success  false      
    weekly-test-debug-valgrind                          test           on_success  false      
    weekly-test-release-valgrind                        test           on_success  false      
    weekly-test-optimized-valgrind                      test           on_success  false      
    weekly-test-takes-forever-optimized                 test           on_success  false      
    doxygen                                             documentation  on_success  false      
    manual                                              documentation  on_success  false      
    tutorial                                            documentation  on_success  false      
    models                                              documentation  on_success  false   

To execute the ``per-commit-compile-release`` job, or any of the others listed above, use 
the following command. 

.. sourcecode:: console

    ~/ns-3-dev$ gitlab-ci-local --file ./utils/tests/gitlab-ci.yml per-commit-compile-release

WARNING: if you do not specify the job name, all jobs that can be executed in parallel will be
executed at the same time. You may run out of disk, memory or both.

Some jobs might require a previous job to complete successfully before getting started.
The doxygen job is one of these.

.. sourcecode:: console

    ~/ns-3-dev$ gitlab-ci-local --file ./utils/tests/gitlab-ci.yml doxygen
    Using fallback git user.name
    Using fallback git user.email
    parsing and downloads finished in 202 ms
    doxygen                               starting archlinux:latest (documentation)
    doxygen                               pulled archlinux:latest in 64 ms
    doxygen                               > still running...
    doxygen                               > still running...
    doxygen                               copied to container in 20 s
    doxygen                               imported cache 'ccache-' in 3.67 s
    ~/ns-3-dev/.gitlab-ci-local/artifacts/pybindgen doesn't exist, did you forget --needs

As instructed by the previous command output, you can add the ``--needs`` to build required
jobs before proceeding. However, doing so will run all jobs as the ``doxygen`` is only supposed
to run after weekly jobs are successfully executed.

Another option is to run the specific job that produces the required artifact. In this case
the ``pybindgen`` job.

.. sourcecode:: console

    ~/ns-3-dev$ gitlab-ci-local --file ./utils/tests/gitlab-ci.yml pybindgen
    Using fallback git user.name
    Using fallback git user.email
    parsing and downloads finished in 202 ms
    pybindgen                               starting archlinux:latest (build)

    ...

    pybindgen                             $ git diff src > pybindgen_new.patch
    pybindgen                             exported artifacts in 911 ms
    pybindgen                             copied artifacts to cwd in 56 ms
    pybindgen                             finished in 5.77 min

     PASS  pybindgen

Then run the doxygen job again:

.. sourcecode:: bash

    ~/ns-3-dev$ gitlab-ci-local --file ./utils/tests/gitlab-ci.yml doxygen
    Using fallback git user.name
    Using fallback git user.email
    parsing and downloads finished in 170 ms
    doxygen                               starting archlinux:latest (documentation)

    ...

    doxygen                               >      1 files with warnings
    doxygen                               > Doxygen Warnings Summary
    doxygen                               > ----------------------------------------
    doxygen                               >      1 directories
    doxygen                               >      1 files
    doxygen                               >     23 warnings
    doxygen                               > done.
    doxygen                               exported cache ns-3-ccache-storage/ 'ccache-' in 6.86 s
    doxygen                               exported artifacts in 954 ms
    doxygen                               copied artifacts to cwd in 59 ms
    doxygen                               finished in 15 min

     PASS  doxygen

Artifacts built by the CI jobs will be stored in separate subfolders
based on the job name. 

``~/ns-3-dev/.gitlab-ci-local/artifacts/jobname``
