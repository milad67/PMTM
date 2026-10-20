# Training objective

`train.py` supports three core loss components (segmentation, auxiliary, and
semantic state supervision) and three optional regularizers (consistency,
cross-stage identity, and slot diversity). Each optional regularizer can be
disabled independently by setting its coefficient to `0`.

The supplied configurations enable all three optional regularizers. With
their current non-zero coefficients, the objective is:

```text
L = Lseg + lambda_aux(epoch) * Laux
    + 0.05 * Lstate
    + 0.40 * Lconsistency
    + 0.02 * Lidentity
    + 0.005 * Ldiversity
```

- `Lseg` combines the shared decoder segmentation terms.
- `Laux` averages auxiliary segmentation losses over the available modalities.
- `Lstate` supervises modality-specific and mean assignment maps. The recorded
  configurations average the four encoder stages with equal weights.
- `Lconsistency` compares states and mean assignment maps from incomplete and
  complete inputs. The complete-input branch is detached for this comparison.
- `Lidentity` compares corresponding slot identities between consecutive stages.
- `Ldiversity` penalizes off-diagonal slot cosine similarities above its margin.

## Optional regularizers

| Regularizer | Coefficient | Value in the supplied configurations | Disabled value |
|---|---|---:|---:|
| Consistency | `lambda_consistency` | 0.40 | 0 |
| Cross-stage identity | `lambda_identity` | 0.02 | 0 |
| Slot diversity | `lambda_diversity` | 0.005 | 0 |

To disable all three when using the supplied launcher, set these fields in
the selected `configs/brats*.json` file:

```json
{
  "lambda_consistency": 0.0,
  "lambda_identity": 0.0,
  "lambda_diversity": 0.0
}
```

These are the fields to update in the existing configuration, rather than
a replacement for the full configuration file. For a direct `train.py`
invocation, the corresponding command-line arguments are:

```text
--lambda_consistency 0 --lambda_identity 0 --lambda_diversity 0
```

With all three coefficients set to zero, the training objective contains
only the segmentation, auxiliary, and semantic state terms. The supplied
configuration files retain their enabled settings. Optional denotes that
these terms are configurable; it does not establish their use or absence
in any particular historical experiment.

## Schedules

The auxiliary multiplier is 0.5 through the first 30 percent of training,
decreases linearly to 0.2 at 70 percent, and then to 0.1 at the end.
The identity and diversity terms start when the zero-based epoch index reaches
20; the consistency term starts at index 40. The identity temperature is 0.1,
the diversity margin is 0.85, and the consistency map multiplier is 0.5.

These are recorded training settings. This release does not establish the
separate effect of each regularization term on the reported Dice scores.
The loss implementations are in `utils/losses.py`.
