# Positive samples predicted successfully at threshold 0.5

This report uses the same fixed 19 Stage-1 actions as
`cascade_metrics.json`. Here, **predicted successfully** means a true positive:
the physical label is 1 and the Stage-2 prediction is also 1. It does not mean
all samples predicted as 1.

| Stage-2 input | TP / actual positive | FN | Mean TP probability | Min | Max |
|---|---:|---:|---:|---:|---:|
| Touch Only | 11/17 | 6 | 0.686448 | 0.535938 | 0.999999 |
| RGB+Action13+Touch | 16/17 | 1 | 0.987529 | 0.944289 | 0.999939 |
| RGB+Action13+ContactGS+CorridorGS+Touch | 17/17 | 0 | 0.997062 | 0.978564 | 0.999978 |

## Touch Only true-positive probabilities

`obj_11=0.606777`, `obj_22=0.756280`, `obj_32=0.690738`,
`obj_34=0.562838`, `obj_36=0.625034`, `obj_42=0.601371`,
`obj_60=0.999999`, `obj_64=0.659952`, `obj_77=0.681974`,
`obj_78=0.535938`, `obj_94=0.830024`.

Missed positives (FN): `obj_33=0.000000`, `obj_41=0.073739`,
`obj_76=0.182220`, `obj_84=0.489740`, `obj_93=0.001045`,
`obj_95=0.489927`.

## RGB+Action13+Touch true-positive probabilities

`obj_11=0.999436`, `obj_22=0.993796`, `obj_32=0.944289`,
`obj_34=0.996858`, `obj_36=0.998892`, `obj_41=0.999876`,
`obj_42=0.999939`, `obj_60=0.957753`, `obj_64=0.998634`,
`obj_76=0.990548`, `obj_77=0.982543`, `obj_78=0.995616`,
`obj_84=0.972536`, `obj_93=0.999194`, `obj_94=0.976031`,
`obj_95=0.994519`.

Missed positive (FN): `obj_33=0.000000`.

## RGB+Action13+ContactGS+CorridorGS+Touch true-positive probabilities

`obj_11=0.999972`, `obj_22=0.999293`, `obj_32=0.998686`,
`obj_33=0.999596`, `obj_34=0.999974`, `obj_36=0.999797`,
`obj_41=0.999957`, `obj_42=0.999978`, `obj_60=0.981998`,
`obj_64=0.999738`, `obj_76=0.995964`, `obj_77=0.978564`,
`obj_78=0.998055`, `obj_84=0.999857`, `obj_93=0.999555`,
`obj_94=0.999389`, `obj_95=0.999678`.

Missed positives (FN): none.
