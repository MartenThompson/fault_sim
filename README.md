# fault_sim
Experimental tools to simulate, detect, and classify conductive faults. 

## Overview 

Experiments are conducted within the imagined product scenario diagramed below.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/36a752c6-9f8f-4059-92cc-5e02884ee851" />


[diagram](https://excalidraw.com/#json=ZWxGtjUG8Fc94i9KaMenp,l9A0dQwiD8_GujTcj4NK9g)


`code/echo_simulator.py` used to generate different sampling scenarios. 
```
code/echo_simulator.py -n 30 -o data/baseline.csv -f baseline 
code/echo_simulator.py -n 2 -o data/open_fault.csv -f open
```


## TODO 

- [ ] this readme needs some love.
- [ ] `echo_simulator` seed should vary by default, supply optional arg.
