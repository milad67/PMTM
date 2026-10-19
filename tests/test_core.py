import unittest

import numpy as np
import torch

from data.datasets import one_hot_numpy, validate_and_remap_label
from data.transforms import Compose, RandomRotation3D
from models import TSMNet
from evaluation import binary_dice
from utils.losses import (
    cross_stage_identity_loss,
    multistage_semantic_state_loss,
    slot_diversity_loss,
    state_health_metrics,
    build_optimizer_parameter_groups,
)


class CoreTests(unittest.TestCase):
    def test_label_validation_rejects_negative(self):
        with self.assertRaises(ValueError):
            validate_and_remap_label(np.asarray([[[0, -1]]], dtype=np.int64))

    def test_rotation_keeps_label_domain(self):
        x = np.zeros((1, 32, 32, 32, 4), dtype=np.float32)
        y = np.zeros((1, 32, 32, 32), dtype=np.int64)
        y[:, 8:24, 8:24, 8:24] = 3
        transform = Compose([RandomRotation3D(10, 1.0)])
        for _ in range(10):
            _, y_aug = transform([x, y])
            self.assertTrue(set(np.unique(y_aug)).issubset({0, 1, 2, 3}))


    def test_binary_dice_empty_convention(self):
        empty = torch.zeros(2, 4, 4, 4)
        nonempty = empty.clone()
        nonempty[0, 0, 0, 0] = 1
        scores = binary_dice(empty, nonempty)
        self.assertEqual(float(scores[0]), 0.0)
        self.assertEqual(float(scores[1]), 1.0)

    def test_optimizer_groups_protect_assignment_endpoints(self):
        model = TSMNet(basic_dims=4, state_dim=8, num_states=4, variant="stage234")
        groups = build_optimizer_parameter_groups(model, weight_decay=1e-4)
        decay_ids = {id(parameter) for parameter in groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
        for stage in (2, 3, 4):
            endpoint = model.state_heads[f"s{stage}"].assign_branch[-1].weight
            self.assertIn(id(endpoint), no_decay_ids)
            self.assertNotIn(id(endpoint), decay_ids)
        self.assertIn(id(model.memory.T0), no_decay_ids)
        self.assertIn(id(model.memory.eta_logit), no_decay_ids)

    def test_forward_and_corrected_losses(self):
        torch.manual_seed(1)
        model = TSMNet(basic_dims=4, state_dim=8, num_states=4, variant="stage234")
        x = torch.randn(1, 4, 32, 32, 32)
        mask = torch.tensor([[True, False, True, True]])
        label = torch.randint(0, 4, (1, 32, 32, 32))
        target = torch.nn.functional.one_hot(label, 4).permute(0, 4, 1, 2, 3).float()
        output = model(x, mask, return_details=True, compute_auxiliary=False)
        self.assertEqual(tuple(output["prediction"].shape), (1, 4, 32, 32, 32))
        self.assertEqual(sorted(output["stage_details"]), [2, 3, 4])
        state_loss, per_stage = multistage_semantic_state_loss(
            output["stage_details"], target, mask
        )
        self.assertEqual(sorted(per_stage), [2, 3, 4])
        identity = cross_stage_identity_loss(output["stage_details"])
        diversity = slot_diversity_loss(output["stage_details"])
        total = output["prediction"].mean() + 0.05 * state_loss + 0.02 * identity + 0.005 * diversity
        total.backward()
        for stage in (2, 3, 4):
            parameter = model.state_heads[f"s{stage}"].assign_branch[-1].weight
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        health = state_health_metrics(output["stage_details"])
        self.assertEqual(sorted(health), [2, 3, 4])

    def test_all_stage_progressive_forward_and_losses(self):
        torch.manual_seed(2)
        model = TSMNet(
            basic_dims=4,
            state_dim=8,
            num_states=4,
            variant="stage1234",
            token_mixer_scope="active",
        )
        x = torch.randn(1, 4, 32, 32, 32)
        mask = torch.tensor([[True, True, False, True]])
        label = torch.randint(0, 4, (1, 32, 32, 32))
        target = torch.nn.functional.one_hot(label, 4).permute(0, 4, 1, 2, 3).float()
        output = model(x, mask, return_details=True, compute_auxiliary=False)
        self.assertEqual(tuple(output["prediction"].shape), (1, 4, 32, 32, 32))
        self.assertEqual(sorted(output["stage_details"]), [1, 2, 3, 4])
        self.assertTrue(all(model.sgfs[f"s{s}"].use_token_mixer for s in (1, 2, 3, 4)))

        stage_weights = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        state_loss, per_stage = multistage_semantic_state_loss(
            output["stage_details"], target, mask, stage_weights=stage_weights
        )
        self.assertEqual(sorted(per_stage), [1, 2, 3, 4])
        identity = cross_stage_identity_loss(output["stage_details"])
        diversity = slot_diversity_loss(output["stage_details"])
        total = (
            output["prediction"].mean()
            + 0.05 * state_loss
            + 0.02 * identity
            + 0.005 * diversity
        )
        total.backward()
        for stage in (1, 2, 3, 4):
            parameter = model.state_heads[f"s{stage}"].assign_branch[-1].weight
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_all_stage_local_resets_state_at_every_stage(self):
        model = TSMNet(
            basic_dims=4,
            state_dim=8,
            num_states=4,
            variant="stage1234_local",
            token_mixer_scope="active",
        )
        x = torch.randn(1, 4, 32, 32, 32)
        mask = torch.tensor([[True, True, True, True]])
        output = model(x, mask, return_details=True, compute_auxiliary=False)
        self.assertEqual(sorted(output["stage_details"]), [1, 2, 3, 4])
        for stage in (1, 2, 3, 4):
            expected = model.memory.init_state(
                batch_size=1,
                device=output["stage_details"][stage]["state_input"].device,
                dtype=output["stage_details"][stage]["state_input"].dtype,
            )
            self.assertTrue(
                torch.allclose(output["stage_details"][stage]["state_input"], expected)
            )


if __name__ == "__main__":
    unittest.main()
