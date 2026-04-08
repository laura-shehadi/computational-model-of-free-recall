# Computational Model of the Serial Position Effect

A computational cognitive modeling project developed for COG260: Data, Computation, and the Mind at the University of Toronto. This project simulates the serial position effect in free recall by modeling short-term and long-term memory dynamics, including decay, rehearsal, interference, and probabilistic retrieval.

## Features

- Activation-based representation of memory items in STM and LTM
- Simulation of study, interference, and recall phases
- Modeling of decay, rehearsal, and interference mechanisms
- Competitive and probabilistic retrieval process
- Generation of a simulated serial position curve
- pyClarion-based phase control for memory state transitions

## Technologies Used

- Python 3
- Matplotlib
- pyClarion
- Computational cognitive modeling
- Simulation and probabilistic modeling

## Model Overview

This project models recall as an activation-based process rather than a binary remembered or forgotten state.

During the study phase, items are presented sequentially and enter short-term memory with high activation. Activation decays over time, while rehearsal strengthens active items and supports consolidation into long-term memory. A small primacy boost is applied to earlier list positions.

During the interference phase, short-term memory activation decays further in the absence of rehearsal.

During the recall phase, items compete for retrieval based on their combined STM and LTM activation, with added stochastic noise. Retrieval continues until recall fails probabilistically, producing a serial position curve that captures primacy and recency effects.

## File Structure

- `serial_position_model.py`  
  Main implementation of the pyClarion controller and memory simulation model

## How to Run

1. Make sure Python 3 is installed
2. Install required dependencies, including Matplotlib
3. Install pyClarion from the official repository:

```bash
git clone https://github.com/cmekik/pyClarion
```

## Note

This project depends on the pyClarion framework. The code will not run unless pyClarion is installed.

## Academic Disclaimer

This project was created as part of a university course assignment.
All code is original or adapted from course materials and modeling frameworks where applicable.
Please do not copy, redistribute, or use this code for any other purpose.
