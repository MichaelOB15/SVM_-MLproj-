# Contributor Guide

This is a short file containing guidelines for contributing to this library, as hopefully many TAs will be using this
and adding/modifying code in future semesters.

## Commits/PRs

TODO

Only commit to master upon an approved PR

TODO master commit guide

no guide for non-master commit messages, we don't need to have sticks up our butts

## Releases

Releases will be branches titled `release/x.x.x`. Make sure students know which version is to be used for a given
assignment.

## Code Style Guide

### Follow PIP style guidelines at all times!
- **Method names do NOT use camelCase in Python! Don't do it!**

### Guidelines for source metadata

From [this page](https://stackoverflow.com/questions/1523427/what-is-the-common-header-format-of-python-files)

> Next should be authorship information. This information should follow this format:

```python
__author__ = "Rob Knight, Gavin Huttley, and Peter Maxwell"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Rob Knight", "Peter Maxwell", "Gavin Huttley",
                    "Matthew Wakefield"]
__license__ = "GPL"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Production"
```
> Status should typically be one of "Prototype", "Development", or "Production". `__maintainer__` should be the person
> who will fix bugs and make improvements if imported. `__credits__` differs from `__author__` in that `__credits__`
> includes people who reported bug fixes, made suggestions, etc. but did not actually write the code.

The `__version__` attribute has been deprecated, do not use it.

### Don't skimp on docstrings

But also don't be too verbose. Docstrings should provide a *concise* description of what a method or class does, without
getting into any implementation details. Provide enough information that someone will be able to use that class/method,
and no more.

Use the Google style for docstrings:

```python
def very_cool_method(arg1: int, arg2: float) -> bool:
    """
    description description description
    
    Example:
        You can use this when you need something cool, like so:
            ``cool_variable = very_cool_method(5, 7.0)``
    
    Args:
        arg1 (int): this is arg1
        arg2 (float): this is arg2
    
    Returns:
        bool: something cool
    """
    ...
```

### Use properties whenever possible

Use the `@property` decorator!

```python
class Foo(object):
    def __init__(self, bar):
        self._bar = bar

    @property
    def bar(self):
        return self._bar
    
    @bar.setter
    def bar(self, value):
        self._bar = value
```

This is the Python equivalent to getters and setters, and all self-respecting languages have a version of this syntax.
(Looking at you, Java!) Remember: a leading underscore `self._attribute_name` denotes protected attributes, a double
leading underscore ("dunderscore") `self.__attribute_name` denotes private attributes. There is hardly ever a time in
which it is appropriate to make an attribute private.

### Use type hints in method signatures

Python 3 supports type hints, a way to annotate variables with their expected type. Python is a dynamically-typed
language so of course this isn't necessary, but it helps provide an implicit sort of documentation and helps with
code completion in IDEs like PyCharm to make things less ambiguous.

To learn more about type hints check out the docs for the [typing](https://docs.python.org/3/library/typing.html) module
in the Python Standard Library.

## Unit Tests

TODO
