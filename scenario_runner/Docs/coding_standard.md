<h1>Coding standard</h1>

> _This document is a work in progress and might be incomplete._

General
-------

  * Use spaces, not tabs.
  * Avoid adding trailing whitespace as it creates noise in the diffs.
  * Comments should not exceed 120 columns, code may exceed this limit a bit in
    rare occasions if it results in clearer code.

Python
------

  * All code must be compatible with Python 2.7, 3.5, and 3.6.
  * [Pylint][pylintlink] should not give any error or warning (few exceptions
    apply with external classes like `numpy`, see our `.pylintrc`).
  * Python code follows [PEP8 style guide][pep8link] (use `autopep8` whenever
    possible).

[pylintlink]: https://www.pylint.org/
[pep8link]: https://www.python.org/dev/peps/pep-0008/

C++
---

  * Compilation should not give any error or warning
    (`clang++ -Wall -Wextra -std=C++14 -Wno-missing-braces`).
  * Unreal C++ code (CarlaUE4 and Carla plugin) follow the
    [Unreal Engine's Coding Standard][ue4link] with the exception of using
    spaces instead of tabs.
  * LibCarla uses a variation of [Google's style guide][googlelink].

[ue4link]: https://docs.unrealengine.com/latest/INT/Programming/Development/CodingStandard/
[googlelink]: https://google.github.io/styleguide/cppguide.html
