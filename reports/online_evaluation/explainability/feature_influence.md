# Feature Influence Report

Sensitivity estimated via perturbation analysis on the fine-tuned model.

| Rank | Feature | Normalised Importance | Raw Flip Rate |
|------|---------|----------------------|---------------|
| 1 | `distance` | ████████████████████ 1.0000 | 0.0220 |
| 2 | `speed` | █ 0.0000 | 0.0000 |
| 3 | `lane` | █ 0.0000 | 0.0000 |
| 4 | `destination` | █ 0.0000 | 0.0000 |
| 5 | `vehicle_order` | █ 0.0000 | 0.0000 |

## Interpretation

- **speed** and **distance** are the dominant features because they directly determine
  arrival time (t = dist / speed_ms), which is the core conflict condition.
- **lane** matters because it encodes the vehicle's turning intention,
  determining whether paths physically cross.
- **destination** has moderate influence since it specifies the target quadrant,
  affecting crossing geometry.
- **vehicle_order** sensitivity measures prompt-order bias in the model.

## Conflict Detection Conditions

A conflict exists when **all** of:
1. Paths cross (opposing or perpendicular directions with crossing trajectories)
2. Both vehicles arrive within 5 seconds of each other
3. Vehicles come from different directions