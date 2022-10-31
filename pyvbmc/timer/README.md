# Timer subpackage developing notes

The main instance of the PyVBMC ``Timer`` class can be imported into a module with
```python
from pyvbmc.timer import main_timer as timer
```
This allows the timer to be accessed from any module without passing it back and forth explicitly.
