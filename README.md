# Continuous Action Deep Reinforcement Learning for Residential Home Energy Management Systems

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of the paper **"Continuous Action Deep Reinforcement Learning for Residential Home Energy Management Systems"** . The code provides a complete framework for training and evaluating continuous-action DRL algorithms (SAC, TD3, ACKTR) for home energy management systems (HEMS).

## 📋 Overview

This framework formulates the HEMS problem as a continuous-state, continuous-action Markov decision process (MDP) that coordinates:

- Battery storage (charge/discharge)
- Grid power exchange (import/export)
- Photovoltaic (PV) utilization
- Controllable loads
- HVAC thermal comfort control

The multi-objective reward function minimizes electricity cost while balancing thermal comfort, battery degradation, unmet demand, and PV self-consumption.

## 📁 Repository Structure
