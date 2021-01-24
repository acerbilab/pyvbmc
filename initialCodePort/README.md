# Initial Codeport
This works as a preminarly port to Python to figure out the objects and methods required for the actual port.

## Render class diagrams
The diagrams can be rendered using pyreverse which is a part of [pylint](https://pypi.org/project/pylint/)

```
pyreverse -o png -f ALL entropy/* gaussian_process/* variational_posterior/* vbmc/*
```