# Protected probabilistic classification experiments

This repository contains the supporting code to reproduce the experiments
in the paper "Protected Probabilistic Classification Library" submitted to the conference
Conformal and Probabilistic Prediction with Application (COPA 2025) (https://copa-conference.com/)

The experiments were carried out using Python 3.11.8 and make use of the **protected-classification** library
(https://github.com/ip200/protected-classification)
## Instructions

Please install the necessary requirements:

```commandline
pip install -r requirements.txt
```

In order to run the experiments for a given section please run

```commandline
python section_5_1_1.py
```

The results will be stored in the *./results* folder with the corresponding
tex formatted output in the *./results/experiment/tex* folder (e.g *./results/section_5_1_1/tex*)