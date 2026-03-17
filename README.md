# fault_sim
Experimental tools to simulate, detect, and classify conductive faults. 

## Overview 

Experiments are conducted within the imagined product scenario diagramed below.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/36a752c6-9f8f-4059-92cc-5e02884ee851" />


[diagram](https://excalidraw.com/#json=ZWxGtjUG8Fc94i9KaMenp,l9A0dQwiD8_GujTcj4NK9g)


`code/echo_simulator.py` used to generate different sampling scenarios. 
```
python3.12 -m code.echo_simulator -n 3 -o data/short.csv -f short -plot
```


## TODO 

- [ ] this readme needs some love.
- [ ] `echo_simulator` seed should vary by default, supply optional arg.
- [ ] random seed management.
