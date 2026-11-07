## Unit tests
Do not add unit tests permanently. You may add them transiently e.g. while adding or updating functionality; clean them up after.


## Code format
Use blank lines to separate logical groups within functions.
- Separate nested helper definitions from the function's main body.
- Separate guard clauses and early returns from subsequent work.
- Keep closely related setup and operations together.
- Put a blank line before the final return expression.
- Review spacing manually after running the formatter.


## Readme and documentation prose style
Don't edit the readme unless I explicitly ask you to, or to fix things in it that you changed the behavior of. Any text you write in documentation comments in code (Or the REAMDE after I give you permission) should be terse. The dialect you prefer to write in is generally too verbose and diffulct to read; don't do that.


## Changes you didn't make, and git
Don't comment on uncommitted changes - you are not the sole author of this. Changes outside what you made are normal. Don't offer to commit anything. Git is a tool you can use to track history, but do not conflate uncommitted changes with ones you made.


## Markdown formatting
Don't use explicit line breaks in paragraphs; assume the viewing/editing software handles that.


## If on a machine without Cuda/nvidia graphics card
Use the `--no-default-features` flag to compile without needing CUDA.