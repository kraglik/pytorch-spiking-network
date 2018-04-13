# Pytorch spiking network simulator

In this simulator was implemented following: 
* Izhikevich model neuronal groups
* STDP synaptic plasticity rule

Simulation of network of spiking networks with plastic synapses
===============================================================

For simulation was created network with 2 groups.

For each group was added recurrent connection with 1ms synaptic delay and connection to other group with 2ms delay.

Network was simulated for 2 seconds of biological time.
After first half of simulation (1 second) synaptic plasticity was disabled and input current for second group was significantly reduced. 

Example of synaptic weights changes by STDP rule
------------------------------------------------

Before simulation:

![Before](https://github.com/kraglik/pytorch-spiking-network/raw/master/plots/g1_to_g1_w_initial.png "Before simulation")

After 1s of simulation with enabled plasticity:

![After](https://github.com/kraglik/pytorch-spiking-network/raw/master/plots/g1_to_g1_w.png "After 1s of simulation with enabled plasticity")

Example of neuronal activity of Izhikevich neuronal groups with recurrent connections
-------------------------------------------------------------------------------------

![Group 1](https://github.com/kraglik/pytorch-spiking-network/raw/master/plots/group_1_activity.png "Group 1")
![Group 2](https://github.com/kraglik/pytorch-spiking-network/raw/master/plots/group_2_activity.png "Group 2")
