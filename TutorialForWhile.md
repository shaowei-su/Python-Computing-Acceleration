#Overview for this topic

This document focuses on the internal mechanism of while loop in Python.

We are going to disassemble one simple python file into bytecode and then illustate its functionality.

Here is the 'while.py' file:

```python
  1 i = 0
  2 while(i<10):
  3     print i
  4     i += 1
  5 
```