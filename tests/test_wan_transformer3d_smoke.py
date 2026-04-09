import unittest

import torch

from videox_fun.models.wan_transformer3d import WanTransformer3DModel


class WanTransformer3DSmokeTest(unittest.TestCase):
    def test_import_init_and_forward(self):
        model = WanTransformer3DModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=4,
            in_dim=16,
            dim=64,
            ffn_dim=128,
            freq_dim=32,
            text_dim=32,
            out_dim=16,
            num_heads=8,
            num_layers=2,
            qk_norm=True,
            cross_attn_norm=True,
        ).eval()

        x = [torch.randn(16, 1, 2, 2)]
        t = torch.tensor([1])
        context = [torch.randn(4, 32)]

        with torch.no_grad():
            output = model(x=x, t=t, context=context, seq_len=1)

        self.assertEqual(output.shape, (1, 16, 1, 2, 2))
        self.assertEqual(output.dtype, x[0].dtype)

    def test_simulation_branch_forward(self):
        model = WanTransformer3DModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=4,
            dim=32,
            ffn_dim=64,
            freq_dim=16,
            text_dim=24,
            out_dim=4,
            num_heads=4,
            num_layers=6,
            add_simulation_branch=True,
            simulation_state_dim=6,
            simulation_cond_dim=3,
            simulation_out_dim=6,
            simulation_num_layers=3,
            simulation_max_seq_len=128,
        ).eval()

        x = torch.randn(2, 4, 2, 4, 4)
        context = torch.randn(2, 5, 24)
        t = torch.tensor([10, 20])
        sim_state = torch.randn(2, 3, 5, 6)
        sim_cond = torch.randn(2, 3, 5, 3)

        with torch.no_grad():
            video, sim = model(
                x=x,
                t=t,
                context=context,
                seq_len=8,
                simulation_states=sim_state,
                simulation_cond=sim_cond,
                return_simulation=True,
            )

        self.assertEqual(video.shape, (2, 4, 2, 4, 4))
        self.assertEqual(sim.shape, (2, 3, 5, 6))
        self.assertEqual(model.simulation_pairing, [0, 2, 5])


if __name__ == "__main__":
    unittest.main()
