# TASKS.md — Detailed Revision Plan (Neurocomputing R1)

## High-Level Objective
Address reviewer comments to elevate the paper from a "proof of concept" to a rigorously validated Continual Learning framework. The core mechanism (P-Factor structural consolidation) is sound, but it lacks the stress-testing required for a high-impact journal.

## The Winning Strategy: The Two-Pronged Attack
Forget the CIFAR-10 Convolutional SNN. Forget Recurrent SNNs. You are going to use your current fully connected architecture to hit their two biggest critiques simultaneously:

### Prong 1: The 5-Task Sequence (Permuted-MNIST / Split-Fashion-MNIST)
**What to do:** Run Permuted-MNIST (where you shuffle the pixels for each new task) or Split-Fashion-MNIST across 5 consecutive tasks.

**Why it works:** This completely satisfies the demand for "testing on longer task sequences, rather than only two tasks." It proves the P-Factor does not run out of capacity after one reset.

### Prong 2: The "Event-Based" Secret Weapon (N-MNIST)
**What to do:** Run a 2-task split on Neuromorphic-MNIST (N-MNIST).

**Why it works:** Reviewer 2 explicitly asked for "other neuromorphic/event-based benchmarks". N-MNIST provides natively event-driven data, which aligns perfectly with the biological plausibility claim of Spiking Neural Networks, proving that the architecture handles real-world temporal dynamics, not just rate-coded static images.

## Next Steps
- [ ] **Generate Fashion MNIST dataset:** Download and convert Fashion MNIST to spike trains.
- [ ] **Run Split-Fashion-MNIST:** Integrate and run the experiment pipeline on Split-Fashion-MNIST to begin testing Prong 1.
- [ ] **Multi-Task Pipeline:** Implement multi-task training logic (scripts/run_multitask.py) capable of tracking P-factors across 5 consecutive tasks.
- [ ] **N-MNIST Integration:** Integrate N-MNIST dataset and adapt input layers for Prong 2.