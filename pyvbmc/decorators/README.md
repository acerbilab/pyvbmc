# decorators subpackage developing notes

These notes are used for keeping track of ToDos and porting information.

## Porting status
- The handle_0D_1D_input decorator should be refactored. Essentially, the decorator aims to ensure that zero-dim and one-dim arguments to functions and methods are casted to two-dim arguments. Furthermore, it can be used to modify the return value of the function to be a scalar. The decorator has been modified several times to provide new functionalities and thus it should be checked, whether it should be used in this way or modified.
