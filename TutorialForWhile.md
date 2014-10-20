#Overview for this topic

This document focuses on the internal mechanism of while loop in Python.

We are going to disassemble one simple python file into bytecode and then illustate its functionality.

Here is the 'while.py' file:

```python
i = 0
while(i<10):
    print i
    i += 1
  
```