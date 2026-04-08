# Computational Model of the Serial Position Effect

A computational cognitive modeling project developed for COG260: Data, Computation, and the Mind at the University of Toronto. This project simulates the serial position effect in free recall, capturing primacy and recency effects through activation-based memory dynamics.

## Features

* Activation-based memory representation (STM and LTM)
* Modeling of decay, rehearsal, and interference mechanisms
* Competitive retrieval process based on activation levels
* Simulation of recall across multiple trials
* Generation of serial position curves (U-shaped recall pattern)

## Technologies Used

* Python 3
* NumPy
* Matplotlib
* pyClarion (cognitive modeling framework)
* Simulation and probabilistic modeling

## Model Overview

Memory items are represented as activation-based units rather than binary stored values.
Activation evolves over time through decay, rehearsal, and interference.

* Early items benefit from repeated rehearsal and consolidation into long-term memory
* Late items retain high activation in short-term memory
* Middle items are more susceptible to interference and decay

Recall is modeled as a competitive process where items with higher activation have a greater probability of being retrieved. Across simulations, this produces the characteristic U-shaped serial position curve observed in human memory.

## How to Run

1. Make sure Python 3 is installed
2. Install required dependencies
3. Install pyClarion from the official repository:

```id="code1"
https://github.com/cmekik/pyClarion
```

4. Run the project files after installing dependencies

## Note

This project depends on the pyClarion framework. The code will not run unless pyClarion is installed.

## Academic Disclaimer

This project was created as part of a university course assignment.
All code is original or adapted from course materials and modeling frameworks where applicable.
Please do not copy, redistribute, or use this code for any other purpose.
