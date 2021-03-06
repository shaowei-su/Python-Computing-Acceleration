#Tutorial for while loop in Python

This document focuses on the internal mechanism of while loop in Python.

We are going to disassemble one simple python file into bytecode and then illustrate its functionality.

Here is the `CSC453/while.py` file:

```python
  1 i = 0
  2 while(i<10):
  3     i += 1
  4     if i < 3:
  5         continue
  6     if i > 7:
  7         break
  8     print i
```
The execution result is:

```
3
4
5
6
7
```

Then we use the dis tool:

```
python -m dis while.py
```

This is the disassembled version:

```
  1           0 LOAD_CONST               0 (0)
              3 STORE_NAME               0 (i)

  2           6 SETUP_LOOP              65 (to 74)
        >>    9 LOAD_NAME                0 (i)
             12 LOAD_CONST               1 (10)
             15 COMPARE_OP               0 (<)
             18 POP_JUMP_IF_FALSE       73

  3          21 LOAD_NAME                0 (i)
             24 LOAD_CONST               2 (1)
             27 INPLACE_ADD         
             28 STORE_NAME               0 (i)

  4          31 LOAD_NAME                0 (i)
             34 LOAD_CONST               3 (3)
             37 COMPARE_OP               0 (<)
             40 POP_JUMP_IF_FALSE       49

  5          43 JUMP_ABSOLUTE            9
             46 JUMP_FORWARD             0 (to 49)

  6     >>   49 LOAD_NAME                0 (i)
             52 LOAD_CONST               4 (7)
             55 COMPARE_OP               4 (>)
             58 POP_JUMP_IF_FALSE       65

  7          61 BREAK_LOOP          
             62 JUMP_FORWARD             0 (to 65)

  8     >>   65 LOAD_NAME                0 (i)
             68 PRINT_ITEM          
             69 PRINT_NEWLINE       
             70 JUMP_ABSOLUTE            9
        >>   73 POP_BLOCK           
        >>   74 LOAD_CONST               5 (None)
             77 RETURN_VALUE 
```
Now we will dig into the python bytecode line by line and explain. Almost all of the instructions are handled inside the `ceval.c` file.

##Set up the loop

Instruction `SETUP_LOOP` creates a PyTryBlock to save the current status, which will be used after loop:

```c
  PyFrame_BlockSetup(f, opcode, INSTR_OFFSET() + oparg, STACK_LEVEL());
```
Here `f` stands for the current FrameObject, `INSTR_OFFSET() + oparg` points to the instruction after while loop and `STACK_LEVEL()` equals to size of current value stack.

In this case, INSTR_OFFESTE() returns value 9 and oparg equals to 65, the result points to line number 74, where this program terminates. And opcode is `SETUP_LOOP`.
```c
void
PyFrame_BlockSetup(PyFrameObject *f, int type, int handler, int level)
{
    PyTryBlock *b;
    if (f->f_iblock >= CO_MAXBLOCKS)
        Py_FatalError("XXX block stack overflow");
    b = &f->f_blockstack[f->f_iblock++];
    b->b_type = type; // CSC253: what kind of block it is 
    b->b_level = level; // CSC253: value stack level to pop to 
    b->b_handler = handler; // CSC253: where to jump to find the instruction index after while loop
}
```
 If the number of blocks reaches the maximum then one error message will be printed out. Here the size of `CO_MAXBLOCKS`(max number of static block nested within a function) is defined as 20 in `code.h` file. By increasing the number of `f_iblock` by one, the PyTryblock we created will be stored in the block stack of the PyFrameObject.

##The general procedure of while loop

We start the illustration of loop with a general case that goes through the instructions and jump back to the begining of loop, regardless of `continue` and `break` situations.

First of all, a judgement is made to determine whether the loop should continue or not. In this case, we are going to compare the value of i with integer 10, if the result is `Py_True`, it will increase the value of i by 1 through `INPLACE_ADD` and then push the result to the top of value stack. Finally, the value of i will be printed out and execute the instruction:
```
  70 JUMP_ABSOLUTE            9
```

to jump back to the start of loop.

For i = {0, 1, 2, 3, 4, 5, 6, 7}, the loop will be executed. When i equals to 7, the `if i > 7` statement will be satisfied after the addition on line 3 in file `CSC453/while.py` and then break out of loop. Details could be found in next topic.

##Continue and break

In Python, `continue ` and `break` works the same as in other languages like C and Java.

After the comparison at line 40 to determine whether i is less than 3 or not, if the result is `Py_True` then execute the instruction:
```
 5          43 JUMP_ABSOLUTE            9
```
to jump back to the start of loop, without execution of the remaining instructions.

In our example, when i = {0, 1, 2} the loop will be continued without the following comparison and print. As a result of this, the first number printed by `CSC453/while.py` should be 3.

On the other hand, the comparison at line 55 will determine if the loop will terminate or not. If the result is `Py_True`, then by the instruction:
```c
  case BREAK_LOOP:
     why = WHY_BREAK;
     goto fast_block_end;
```
We break out of the loop through fast_block_end.
```
fast_block_end:
        while (why != WHY_NOT && f->f_iblock > 0) {
            /* Peek at the current block. */
            PyTryBlock *b = &f->f_blockstack[f->f_iblock - 1];

            ...

            /* Now we have to pop the block. */
            f->f_iblock--;

            while (STACK_LEVEL() > b->b_level) {
                v = POP();
                Py_XDECREF(v);
            }
            if (b->b_type == SETUP_LOOP && why == WHY_BREAK) {
                why = WHY_NOT;
                JUMPTO(b->b_handler);
                break;
            }
```
In fast_block_end, the information stored in PyTryBlock is reloaded to return to the status before while loop.
Specifically, the block is poped. Then the stack level is returned to the previous value stored in PyTryBlock by continuously popping the redundant variables in the stack. Besides, `WHY_NOT` is assigned to `why` to indicate that there is no error and continue to execute the next instruction by `JUMPTO` its instruction index:
```
  74 LOAD_CONST               5 (None)
```

As mentioned above, when i becomes 8 the result of comparion `if i > 7` is true and break statement will be taken. By doing so, the last number printed out will be 7.

##The normal end of while loop

If there is no break statement inside the while loop, then it will terminates through the control statement `if i < 10`.

When the comparison at line 15 returns `Py_False` to the value stack, then it will jump to line 74 to terminate. The `PyFrame_BlockPop(f)` will pop and return the PyTryBlock we stored in block stack. Then reload the stack level to previous state. This procedure is quite similar to the one when we break out of loop.




