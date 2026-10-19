# Training objective

The supplied configurations retain all six terms used by `train.py`:

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

The auxiliary multiplier is 0.5 through the first 30 percent of training,
decreases linearly to 0.2 at 70 percent, and then to 0.1 at the end.
The identity and diversity terms start when the zero-based epoch index reaches
20; the consistency term starts at index 40. The identity temperature is 0.1,
the diversity margin is 0.85, and the consistency map multiplier is 0.5.

These are recorded training settings. This release does not establish the
separate effect of each regularization term on the reported Dice scores.
The loss implementations are in `utils/losses.py`.
