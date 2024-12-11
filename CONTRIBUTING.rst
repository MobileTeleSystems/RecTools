Contributing Guide
==================

Welcome! There are many ways to contribute, including submitting bug
reports, improving documentation, submitting feature requests, reviewing
new submissions, or contributing code that can be incorporated into the
project.

For support questions please use `Telegram Channel <https://t.me/RecTools_Support>`_
or open an issue of type `Question`

Feature Requests
----------------

Please create a new GitHub issue for any significant changes and
enhancements that you wish to make. Provide the feature you would like
to see, why you need it, and how it will work. Discuss your ideas
transparently and get community feedback before proceeding.

Significant changes that you wish to contribute to the project should be
discussed first in a GitHub issue that clearly outlines the changes and
benefits of the feature.

Small Changes can directly be crafted and submitted to the GitHub
Repository as a Pull Request.

Pull Request Process
--------------------

#. Fork RecTools `main repository <https://github.com/MobileTeleSystems/RecTools>`_
   on GitHub. See `this guide <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_ if you have questions.
#. Clone your fork from GitHub to your local disk.
#. Create a virtual environment and install dependencies including all 
   extras and development dependencies.
   
   #. Make sure you have ``python>=3.9`` and ``poetry>=1.5.0`` installed
   #. Deactivate any active virtual environments. Deactivate conda ``base``
      environment if applicable
   #. Run ``make install`` command which will create a virtual env and
      install everything that is needed with poetry. See `poetry usage details <https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock>`_
   
#. Implement your changes. Check the following after you are done:
   
   #. Docstrings. Please ensure that all new public classes and methods
      have docstrings. We use numpy style. Also check that examples are
      provided
   #. Code styling. Autoformat with ``make format``
   #. Linters. Run checks with ``make lint``
   #. Tests. Make sure you've covered new features with tests. Check
      code correctness with ``make test``
   #. Coverage. Check with ``make coverage``
   #. Changelog. Please describe you changes in `CHANGELOG.MD <https://github.com/MobileTeleSystems/RecTools/blob/main/CHANGELOG.md>`_

#. Create a pull request from your fork. See `instructions <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_


You may merge the Pull Request in once you have the approval of one 
of the core developers, or if you do not have permission to do that, you
may request the a reviewer to merge it for you. 

Review Process
--------------

We keep pull requests open for a few days for multiple people to have
the chance to review/comment.

After feedback has been given, we expect responses within two weeks.
After two weeks, we may close the pull request if it isn't showing any
activity.
