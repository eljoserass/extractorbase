
"""

factory for loading metrics compatation
each metric has
- input format (eg, text + schemas)
- output format (eg, the entities or whatever, the grading scheme )
- metric computation inside
    - metric reporting
    - metric aggregation, from the data split passed as parameter (not the slices but the percentage)


try to be a thin wrapper of other libraries
"""