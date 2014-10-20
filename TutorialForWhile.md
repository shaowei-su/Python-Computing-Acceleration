#Overview for this topic

This document focuses on the internal mechanism of while loop in Python.

We are going to disassemble one simple python file into bytecode and then illustrate its functionality.

Here is the `CSC453/while.py` file:

```python
i = 0
while(i<10):
    print i
    i += 1
  
```

This is the disassembled version:

```
  1           0 LOAD_CONST               0 (0)
              3 STORE_NAME               0 (i)

  2           6 SETUP_LOOP              31 (to 40)
        >>    9 LOAD_NAME                0 (i)
             12 LOAD_CONST               1 (10)
             15 COMPARE_OP               0 (<)
             18 POP_JUMP_IF_FALSE       39

  3          21 LOAD_NAME                0 (i)
             24 PRINT_ITEM          
             25 PRINT_NEWLINE       

  4          26 LOAD_NAME                0 (i)
             29 LOAD_CONST               2 (1)
             32 INPLACE_ADD         
             33 STORE_NAME               0 (i)
             36 JUMP_ABSOLUTE            9
        >>   39 POP_BLOCK           
        >>   40 LOAD_CONST               3 (None)
             43 RETURN_VALUE 
```
Now we will dig into the python bytecode line by line and explain.