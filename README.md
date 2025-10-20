# Monopoly AI — MARL, Self-Play with PPO

This repository contains a multi-agent, self-play reinforcement learning setup for Monopoly using Ray RLlib (PPO) and a custom Gymnasium environment.

What’s included
- monopolyEnv.py: Custom Monopoly environment used for training and inference.
- train.py: Trains PPO agents via self-play; saves checkpoints to the training/ folder.
- singleAction.py: Runs a small HTTP server that returns the agent’s next action given an observation (deterministic inference).
- singleActionNonDeterministic.py: Non-deterministic inference variant.
- singleRandomAction.py: random action server.
- test.py / testEnv.py: Simple test/diagnostic scripts.
- training/: Saved checkpoints and artifacts (some examples included).
- requirements.txt and install.py: Dependencies and optional setup helper.

Quick start
1) Install dependencies
   - Python 3.10+ recommended
   - python install.py
2) Train
   - python train.py
   - Checkpoints are written under training/
3) Run the AI
   - python singleAction.py

Notes
- The included training/ subfolders contain example checkpoints.
- Ray may try to use a GPU if available; adjust config in singleAction.py or train.py as needed.


