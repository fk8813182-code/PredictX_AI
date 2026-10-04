# Dataset insights (auto-generated from the data; associations, not causation)

- 10000 rows, 339 failures (3.39%). Missing: 0, duplicates: 0.
- Air temperature [K]: mean 299.97 (normal) vs 300.89 (failed) in this dataset.
- Process temperature [K]: mean 310.0 (normal) vs 310.29 (failed) in this dataset.
- Rotational speed [rpm]: mean 1540.26 (normal) vs 1496.49 (failed) in this dataset.
- Torque [Nm]: mean 39.63 (normal) vs 50.17 (failed) in this dataset.
- Tool wear [min]: mean 106.69 (normal) vs 143.78 (failed) in this dataset.
- Temp diff [K]: mean 10.02 (normal) vs 9.4 (failed) in this dataset.
- Power [W]: mean 6244.55 (normal) vs 7282.82 (failed) in this dataset.
- Failure rate by Type (%): {'H': 2.09, 'L': 3.92, 'M': 2.77}
- Failure rate by tool-wear quartile (%): {'(-0.001, 53.0]': 2.17, '(53.0, 108.0]': 2.36, '(108.0, 162.0]': 2.1, '(162.0, 253.0]': 6.93}
- Failure-mode counts: {'TWF': 46, 'HDF': 115, 'PWF': 95, 'OSF': 98, 'RNF': 19}
- Correlation with failure: {'Air temperature [K]': 0.083, 'Process temperature [K]': 0.036, 'Rotational speed [rpm]': -0.044, 'Torque [Nm]': 0.191, 'Tool wear [min]': 0.105, 'Temp diff [K]': -0.112, 'Power [W]': 0.176}
