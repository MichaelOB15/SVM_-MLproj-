# sting

Library containing boilerplate code for ML applications. Written for CSDS 440

# Grading Rubric

Generally, point deductions will follow these criteria:

- Incomplete implementation/Not following assignment description: up to 100%
- Syntax Errors/Errors causing the code to fail to compile/run:
    - Works with minor fix: 10%
    - Does not work even with minor fixes: 50%
- Inefficient implementation: 10%
    - Algorithm takes too long to run during grading: +10%
- Poor code design: 10%
    - Examples:
        - Hard-to-follow logic
        - Lack of modularity/encapsulation
        - Imperative/ad-hoc/"spaghetti" code
        - Duplicate code
- Poor UI:
    - Bad input (inadequate exception handling, no `--help` flag, etc.): 5%
    - Bad output (overly verbose `print` debugging statements, unclear program output): 5%
- Poor code style: 5%
- Poor documentation: 5%
- Bad commits: 5%
    - Examples:
        - Committing data files
        - Committing non-source files (`.idea` files, `.iml` files, etc.)
    - **Hint:** use your `.gitignore` file!
- Not being able to identify git contributor by their name or case ID: 3% per person

Bonus points may be awarded if you do the following:

- Exceptionally well-documented code
- Exceptionally well-written code
- Exceptionally efficient code (Takes advantage of C/FORTRAN optimizations, using numba, etc.)
    - **Hint:** use pure python (`for` loops, etc.) as minimally as possible!
