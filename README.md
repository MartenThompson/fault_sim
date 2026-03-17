# fault_sim
Experimental tools to simulate, detect, and classify conductive faults. 

## Overview 

Experiments are conducted within the imagined product scenario diagramed below.

<img width="600" alt="imagined deployment flowchart" src="https://github.com/user-attachments/assets/a92bb75d-5dbf-46b5-86a9-2bf9535ed868" />


[diagram](https://excalidraw.com/#json=pEvizIRmO61uVgM_lR85l,ilofvimXC81gyA3azqMzJQ)


`code/echo_simulator.py` used to generate different sampling scenarios. 
```
python3.12 -m code.echo_simulator -n 3 -o data/short.csv -f short -plot
```


## TODO 

- [ ] this readme needs some love.
- [ ] `echo_simulator` seed should vary by default, supply optional arg.
- [ ] random seed management.
