# Research Project Plan

## Rationale

I'm doing an undergraduate engineering research project based on this repo.
I need to understand what this codebase is doing and how it works in order to adapt it for my own project.

## What I understand

I think the research paper based on this codebase describes a framework for detecting out-of-distribution (OOD) samples and adversarial attacks.
It does this by:

1. Inducing a generative classifier from a pre-trained softmax classifier (which is stored in `../pre_trained/`).
2. Measuring the Mahalanobis distance of the outputs of the induced generative classifier.

If the Mahalanobis distance for a given sample is above a certain threshold, the sample is likely OOD or adversarial.
The program which performs this process relies on a CUDA-enabled version of Pytorch.

## What I don't understand

- What does it mean to "induce" a generative classifier?
- How exactly is Mahalanobis distance used to detect adversarial attacks? What would a working example look like?
- Can this codebase be adapted to work on a CPU-only system (i.e. without using CUDA)?
