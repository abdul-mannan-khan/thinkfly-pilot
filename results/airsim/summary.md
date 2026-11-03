| Agent | Missions | Safe completions (95 % CI) | Missions with a rule violation | AI energy per mission, J (mean ± s.d.) | Large-model calls | Small-model calls | Escalations | Small plans passing the checker | Targets inspected / available | Mission time, s |
|---|---|---|---|---|---|---|---|---|---|---|
| Step-by-step (large every step) | 1 | 0/1 (0–98 %) | 1/1 | 7558 ± 0 | 16.0 | 0.0 | 0.0 | – | 75 % | 159 |
| Periodic (large every 30 s) | 1 | 1/1 (2–100 %) | 0/1 | 1722 ± 0 | 3.0 | 0.0 | 0.0 | – | 33 % | 86 |
| Event-triggered (large only) | 1 | 1/1 (2–100 %) | 0/1 | 2239 ± 0 | 4.0 | 0.0 | 0.0 | – | 33 % | 87 |
| ThinkFly (event + small + checker) | 1 | 1/1 (2–100 %) | 0/1 | 386 ± 0 | 0.0 | 4.0 | 0.0 | 100 % | 50 % | 99 |
