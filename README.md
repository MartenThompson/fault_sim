# fault_sim
Sandbox environment to simulate, detect, and classify conductive faults.


`code/echo_simulator.py` used to generate different sampling scenarios. 
```
code/echo_simulator.py -n 30 -o data/baseline.csv -f baseline 
code/echo_simulator.py -n 2 -o data/open_fault.csv -f open
```


## TODO 

- [ ] this readme needs some love.
- [ ] `echo_simulator` seed should vary by default, supply optional arg.