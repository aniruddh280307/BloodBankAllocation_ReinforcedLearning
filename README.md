# Blood Bank Allocation RL Environment

## Overview

This project implements a Reinforcement Learning (RL) environment for optimizing blood allocation in hospitals.

The agent learns to allocate limited blood inventory across incoming hospital requests while balancing:

* Fulfillment rate
* Urgency handling
* Resource efficiency

---

## Problem Statement

Blood banks face real-world challenges:

* Limited inventory
* Unpredictable demand
* Urgent medical requests
* Blood type compatibility constraints

This environment simulates these challenges and trains an agent to make sequential allocation decisions.

---

## Environment Design

### State Space

The agent observes:

* Current blood inventory (O, A, B, AB)
* Active hospital requests:

  * blood type
  * quantity
  * urgency
  * wait time
* System metrics (fulfilled, unfulfilled, expired)

---

### Action Space

The agent outputs:

* Allocation matrix:

  * how many units of each blood type to assign to each request

---

### Reward Function

The agent is rewarded for:

* Fulfilling requests
* Prioritizing urgent cases

The agent is penalized for:

* Unfulfilled requests
* Expired blood units

Reward values are clipped to maintain stability.

---

### Episode Termination

Episode ends when:

* Maximum timesteps are reached
* System reaches a critical state

---

## Scoring System

Final score is normalized between 0 and 1:

* 50% weight on fulfillment rate
* 30% weight on urgent request handling
* 20% penalty for inefficiency (expired units)

---

## Why Reinforcement Learning

This is a sequential decision-making problem:

* Decisions affect future states
* Inventory evolves over time
* Demand is stochastic

Reinforcement learning enables adaptive policies compared to static optimization approaches.

---

## Project Structure

```
env/
 ├── environment.py
 ├── actions.py
 ├── models.py

inference.py
openenv.yaml
Dockerfile
requirements.txt
README.md
```

---

## How to Run

Using Docker:

```
docker build -t bloodbank .
docker run bloodbank
```

---

## Output Format

The environment follows OpenEnv format:

```
[START] ...
[STEP] ...
[END] ...
```

---

## Real-World Relevance

* Improves emergency response systems
* Reduces blood wastage
* Supports healthcare logistics optimization

---

## Status

* Fully Dockerized
* OpenEnv compliant
* Stable multi-step environment
* Ready for evaluation

---

## Author

Developed for Meta PyTorch OpenEnv Hackathon
