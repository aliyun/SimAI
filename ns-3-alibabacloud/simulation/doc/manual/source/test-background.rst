.. include:: replace.txt

Background
----------

**This chapter may be skipped by readers familiar with the basics of
software testing.**

Writing defect-free software is a difficult proposition.  There are many
dimensions to the problem and there is much confusion regarding what is 
meant by different terms in different contexts.  We have found it worthwhile
to spend a little time reviewing the subject and defining some terms.

Software testing may be loosely defined as the process of executing a program
with the intent of finding errors.  When one enters a discussion regarding 
software testing, it quickly becomes apparent that there are many distinct 
mind-sets with which one can approach the subject.

For example, one could break the process into broad functional categories 
like ''correctness testing,'' ''performance testing,'' ''robustness testing''
and ''security testing.''  Another way to look at the problem is by life-cycle:
''requirements testing,'' ''design testing,'' ''acceptance testing,'' and 
''maintenance testing.''  Yet another view is by the scope of the tested system.
In this case one may speak of ''unit testing,'' ''component testing,'' 
''integration testing,'' and ''system testing.''  These terms are also not
standardized in any way, and so ''maintenance testing'' and ''regression
testing'' may be heard interchangeably.  Additionally, these terms are often
misused.

There are also a number of different philosophical approaches to software 
testing.  For example, some organizations advocate writing test programs
before actually implementing the desired software, yielding ''test-driven 
development.''  Some organizations advocate testing from a customer perspective
as soon as possible, following a parallel with the agile development process:
''test early and test often.''  This is sometimes called ''agile testing.''  It
seems that there is at least one approach to testing for every development 
methodology.

The |ns3| project is not in the business of advocating for any one of
these processes, but the project as a whole has requirements that help inform
the test process.

Like all major software products, |ns3| has a number of qualities that
must be present for the product to succeed.  From a testing perspective, some
of these qualities that must be addressed are that |ns3| must be 
''correct,'' ''robust,''  ''performant'' and ''maintainable.''  Ideally there
should be metrics for each of these dimensions that are checked by the tests
to identify when the product fails to meet its expectations / requirements.

Correctness
***********

The essential purpose of testing is to determine that a piece of software 
behaves ''correctly.''  For |ns3| this means that if we simulate 
something, the simulation should faithfully represent some physical entity or 
process to a specified accuracy and precision.

It turns out that there are two perspectives from which one can view 
correctness.  Verifying that a particular model is implemented according 
to its specification is generically called *verification*.  The process of 
deciding that the model is correct for its intended use is generically called 
*validation*.

Validation and Verification
***************************

A computer model is a mathematical or logical representation of something. It 
can represent a vehicle, an elephant (see 
`David Harel's talk about modeling an elephant at SIMUTools 2009 <http://simutools.org/2009/>`_, or a networking card.  Models can also represent
processes such as global warming, freeway traffic flow or a specification of a 
networking protocol.  Models can be completely faithful representations of a 
logical process specification, but they necessarily can never completely 
simulate a physical object or process.  In most cases, a number of 
simplifications are made to the model to make simulation computationally 
tractable. 

Every model has a *target system* that it is attempting to simulate.  The 
first step in creating a simulation model is to identify this target system and
the level of detail and accuracy that the simulation is desired to reproduce. 
In the case of a logical process, the target system may be identified as ''TCP 
as defined by RFC 793.''  In this case, it will probably be desirable to create
a model that completely and faithfully reproduces RFC 793.  In the case of a 
physical process this will not be possible. If, for example, you would like to 
simulate a wireless networking card, you may determine that you need,  ''an 
accurate MAC-level implementation of the 802.11 specification and [...] a 
not-so-slow PHY-level model of the 802.11a specification.'' 

Once this is done, one can develop an abstract model of the target system.  This
is typically an exercise in managing the tradeoffs between complexity, resource
requirements and accuracy.  The process of developing an abstract model has been
called *model qualification* in the literature.  In the case of a TCP 
protocol, this process results in a design for a collection of objects, 
interactions and behaviors that will fully implement RFC 793 in |ns3|.
In the case of the wireless card, this process results in a number of tradeoffs
to allow the physical layer to be simulated and the design of a network device 
and channel for ns-3, along with the desired objects, interactions and behaviors. 

This abstract model is then developed into an |ns3| model that 
implements the abstract model as a computer program.  The process of getting the
implementation to agree with the abstract model is called *model 
verification* in the literature. 

The process so far is open loop. What remains is to make a determination that a
given ns-3 model has some connection to some reality -- that a model is an 
accurate representation of a real system, whether a logical process or a physical
entity. 

If one is going to use a simulation model to try and predict how some real 
system is going to behave, there must be some reason to believe your results -- 
i.e., can one trust that an inference made from the model translates into a 
correct prediction for the real system.  The process of getting the ns-3 model
behavior to agree with the desired target system behavior as defined by the model
qualification process is called *model validation* in the literature. In the
case of a TCP implementation, you may want to compare the behavior of your ns-3 
TCP model to some reference implementation in order to validate your model.  In 
the case of a wireless physical layer simulation, you may want to compare the 
behavior of your model to that of real hardware in a controlled setting,

The |ns3| testing environment provides tools to allow for both model
validation and testing, and encourages the publication of validation results.

Robustness
**********

Robustness is the quality of being able to withstand stresses, or changes in 
environments, inputs or calculations, etc.  A system or design is ''robust''
if it can deal with such changes with minimal loss of functionality.

This kind of testing is usually done with a particular focus.  For example, the 
system as a whole can be run on many different system configurations to 
demonstrate that it can perform correctly in a large number of environments.

The system can be also be stressed by operating close to or beyond capacity by 
generating or simulating resource exhaustion of various kinds.  This genre of
testing is called ''stress testing.''

The system and its components may be exposed to so-called ''clean tests'' that
demonstrate a positive result -- that is that the system operates correctly in 
response to a large variation of expected configurations.  

The system and its components may also be exposed to ''dirty tests'' which 
provide inputs outside the expected range.  For example, if a module expects a 
zero-terminated string representation of an integer, a dirty test might provide
an unterminated string of random characters to verify that the system does not
crash as a result of this unexpected input.  Unfortunately, detecting such 
''dirty'' input and taking preventive measures to ensure the system does not
fail catastrophically can require a huge amount of development overhead.  In
order to reduce development time, a decision was taken early on in the project
to minimize the amount of parameter validation and error handling in the 
|ns3| codebase.  For this reason, we do not spend much time on dirty
testing -- it would just uncover the results of the design decision we know
we took.

We do want to demonstrate that |ns3| software does work across some set
of conditions.  We borrow a couple of definitions to narrow this down a bit.  
The *domain of applicability* is a set of prescribed conditions for which
the model has been tested, compared against reality to the extent possible, and 
judged  suitable for use.  The *range of accuracy* is an agreement between 
the computerized model and reality within a domain of applicability. 

The |ns3| testing environment provides tools to allow for setting up 
and running test environments over multiple systems (buildbot) and provides 
classes to encourage clean tests to verify the operation of the system over the
expected ''domain of applicability'' and ''range of accuracy.''

Performant
**********

Okay, ''performant'' isn't a real English word.  It is, however, a very concise 
neologism that is quite often used to describe what we want |ns3| to 
be: powerful and fast enough to get the job done.

This is really about the broad subject of software performance testing.  One of
the key things that is done is to compare two systems to find which performs 
better (cf benchmarks).  This is used to demonstrate that, for example, 
|ns3| can perform a basic kind of simulation at least as fast as a 
competing tool, or can be used to identify parts of the system that perform
badly.

In the |ns3| test framework, we provide support for timing various kinds
of tests.

Maintainability
***************

A software product must be maintainable.  This is, again, a very broad 
statement, but a testing framework can help with the task.  Once a model has
been developed, validated and verified, we can repeatedly execute the suite
of tests for the entire system to ensure that it remains valid and verified
over its lifetime.

When a feature stops functioning as intended after some kind of change to the
system is integrated, it is called generically a *regression*.  
Originally the 
term regression referred to a change that caused a previously fixed bug to
reappear, but the term has evolved to describe any kind of change that breaks
existing functionality.  There are many kinds of regressions that may occur
in practice.

A *local regression* is one in which a change affects the changed component
directly.  For example, if a component is modified to allocate and free memory
but stale pointers are used, the component itself fails.

A *remote regression* is one in which a change to one component breaks
functionality in another component.  This reflects violation of an implied but
possibly unrecognized contract between components.

An *unmasked regression* is one that creates a situation where a previously
existing bug that had no affect is suddenly exposed in the system.  This may
be as simple as exercising a code path for the first time.

A *performance regression* is one that causes the performance requirements
of the system to be violated.  For example, doing some work in a low level 
function that may be repeated large numbers of times may suddenly render the
system unusable from certain perspectives.

The |ns3| testing framework provides tools for automating the process
used to validate and verify the code in nightly test suites to help quickly
identify possible regressions.
