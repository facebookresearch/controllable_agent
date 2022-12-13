# Contributing to `controllable_agent`
We want to make contributing to this project as easy and transparent as possible.

## Our Development Process
`controllable_agent` is the open source repository of the Controllable Agent project at FAIR.


## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Create a virtual environment and activate it: `source env.sh install` then `source env.sh activate`
3. If you've added code please add tests.
4. If you've changed APIs, please update the documentation.
5. Make sure typing is correct: `mypy url_benchmark controllable_agent`
6. Run the formatter: `black controllable_agent` (black is not yet used on `url_benchmark`)
7. Ensure the test suite passes: `pytest url_benchmark controllable_agent`

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
We use black coding style with a generous 110 line length for the `controllable_agent` package. `url_benchmark` follows most of pep8 style but is not blacked.

## License
By contributing to `controllable_agent`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
