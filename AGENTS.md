## Unit tests

Do not add unit tests permanently. You may add them transiently e.g. while adding or updating
functionality; clean them up after.

## Code format

Add appropriate line breaks in code; I have noticed LLMs tend to not use line breaks, which makes it tough for me to read and edit code.

## If on a machine without Cuda/nvidia graphics card

Use the `--no-default-features` flag to compile without needing CUDA.


## Readme and documentation prose style
Don't edit the readme unless I explicitly ask you to, or to fix things in it that you changed the behavior of. Any text you write in documentation comments in code (Or the REAMDE after I give you permission) should be terse. The dialect you prefer to write in is generally too verbose and diffulct to read; don't do that.