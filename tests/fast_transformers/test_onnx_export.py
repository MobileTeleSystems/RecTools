#  Copyright 2026 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for ONNX export of UniSRecNet network and UniSRecModel.export_to_onnx."""

# pylint: disable=redefined-outer-name,protected-access,import-outside-toplevel

import typing as tp
from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from rectools.fast_transformers.unisrec.model import UniSRecModel  # noqa: E402  # pylint: disable=wrong-import-position
from rectools.fast_transformers.unisrec.net import UniSRecNet  # noqa: E402  # pylint: disable=wrong-import-position


@pytest.fixture()
def net() -> UniSRecNet:
    torch.manual_seed(0)
    pretrained = torch.randn(11, 32)
    pretrained[0] = 0.0
    model = UniSRecNet(
        n_items=10,
        pretrained_embeddings=pretrained,
        n_factors=16,
        projection_hidden=32,
        n_blocks=1,
        n_heads=2,
        session_max_len=8,
        dropout=0.0,
        adaptor_dropout=0.0,
    )
    model.eval()
    return model


def _export_and_load(net: torch.nn.Module, args: tp.Any, tmp_path: Path, **kwargs: tp.Any) -> tp.Any:
    path = str(tmp_path / "model.onnx")
    torch.onnx.export(net, args, path, opset_version=18, **kwargs)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return ort.InferenceSession(path)


class TestUniSRecNetOnnxExport:
    def test_export_succeeds(self, net: UniSRecNet, tmp_path: Path) -> None:
        dummy = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long)
        path = str(tmp_path / "model.onnx")
        torch.onnx.export(
            net,
            (dummy,),
            path,
            input_names=["input_ids"],
            output_names=["hidden"],
            opset_version=18,
        )
        model = onnx.load(path)
        onnx.checker.check_model(model)

    def test_forward_roundtrip(self, net: UniSRecNet, tmp_path: Path) -> None:
        dummy = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long)
        sess = _export_and_load(
            net,
            (dummy,),
            tmp_path,
            input_names=["input_ids"],
            output_names=["hidden"],
        )
        with torch.no_grad():
            expected = net(dummy).numpy()
        result = sess.run(None, {"input_ids": dummy.numpy()})[0]
        np.testing.assert_allclose(result, expected, atol=1e-5)

    @pytest.mark.xfail(reason="dynamic_shapes requires dynamo=True which is not used here")
    def test_dynamic_batch(self, net: UniSRecNet, tmp_path: Path) -> None:
        dummy = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long)
        batch = torch.export.Dim("batch", min=1)
        sess = _export_and_load(
            net,
            (dummy,),
            tmp_path,
            input_names=["input_ids"],
            output_names=["hidden"],
            dynamic_shapes=({0: batch},),
        )
        batch_input = torch.tensor(
            [[0, 0, 1, 2, 3], [0, 1, 4, 5, 6], [0, 0, 0, 7, 8]],
            dtype=torch.long,
        )
        with torch.no_grad():
            expected = net(batch_input).numpy()
        result = sess.run(None, {"input_ids": batch_input.numpy()})[0]
        assert result.shape[0] == 3
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_different_sequence_lengths(self, net: UniSRecNet, tmp_path: Path) -> None:
        dummy = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long)
        batch = torch.export.Dim("batch", min=1)
        seq_len = torch.export.Dim("seq_len", min=1, max=8)
        sess = _export_and_load(
            net,
            (dummy,),
            tmp_path,
            input_names=["input_ids"],
            output_names=["hidden"],
            dynamic_shapes=({0: batch, 1: seq_len},),
        )
        short = torch.tensor([[0, 1, 2]], dtype=torch.long)
        with torch.no_grad():
            expected = net(short).numpy()
        result = sess.run(None, {"input_ids": short.numpy()})[0]
        assert result.shape == (1, 3, 16)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_padding_only_input(self, net: UniSRecNet, tmp_path: Path) -> None:
        dummy = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long)
        sess = _export_and_load(
            net,
            (dummy,),
            tmp_path,
            input_names=["input_ids"],
            output_names=["hidden"],
        )
        all_pad = torch.zeros(1, 5, dtype=torch.long)
        with torch.no_grad():
            expected = net(all_pad).numpy()
        result = sess.run(None, {"input_ids": all_pad.numpy()})[0]
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_output_shape(self, net: UniSRecNet, tmp_path: Path) -> None:
        dummy = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long)
        sess = _export_and_load(
            net,
            (dummy,),
            tmp_path,
            input_names=["input_ids"],
            output_names=["hidden"],
        )
        result = sess.run(None, {"input_ids": dummy.numpy()})[0]
        assert result.shape == (1, 5, 16)

    def test_project_all_roundtrip(self, net: UniSRecNet, tmp_path: Path) -> None:
        class _ProjectAll(torch.nn.Module):
            def __init__(self, inner: UniSRecNet):
                super().__init__()
                self.inner = inner

            def forward(self) -> torch.Tensor:
                return self.inner.project_all()

        wrapper = _ProjectAll(net)
        wrapper.eval()
        path = str(tmp_path / "project_all.onnx")
        torch.onnx.export(
            wrapper,
            (),
            path,
            input_names=[],
            output_names=["item_embs"],
            opset_version=18,
        )
        model = onnx.load(path)
        onnx.checker.check_model(model)
        sess = ort.InferenceSession(path)
        with torch.no_grad():
            expected = net.project_all().numpy()
        result = sess.run(None, {})[0]
        assert result.shape == (11, 16)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestUniSRecModelExport:
    """Tests for UniSRecModel.export_to_onnx."""

    @pytest.fixture()
    def model(self) -> UniSRecModel:
        torch.manual_seed(0)
        pretrained = torch.randn(11, 32)
        pretrained[0] = 0.0
        m = UniSRecModel(
            pretrained_item_embeddings=pretrained,
            n_factors=16,
            projection_hidden=32,
            n_blocks=1,
            n_heads=2,
            session_max_len=8,
            epochs=0,
        )
        from rectools.fast_transformers.preprocessing.sequence_data import align_embeddings

        unique_items = torch.arange(1, 11)
        aligned = align_embeddings(pretrained, unique_items, 10)
        net = UniSRecNet(
            n_items=10,
            pretrained_embeddings=aligned,
            n_factors=16,
            projection_hidden=32,
            n_blocks=1,
            n_heads=2,
            session_max_len=8,
            dropout=0.0,
            adaptor_dropout=0.0,
        )
        net.eval()
        m._net = net
        m._unique_items = unique_items
        m._unique_users = torch.arange(5)
        m.is_fitted = True
        return m

    def test_export_encoder(self, model: UniSRecModel, tmp_path: Path) -> None:
        path = tmp_path / "encoder.onnx"
        model.export_to_onnx(str(path))
        loaded = onnx.load(str(path))
        onnx.checker.check_model(loaded)

    def test_export_encoder_roundtrip(self, model: UniSRecModel, tmp_path: Path) -> None:
        path = tmp_path / "encoder.onnx"
        model.export_to_onnx(str(path))
        sess = ort.InferenceSession(str(path))
        dummy = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            expected = model.net(dummy).numpy()
        result = sess.run(None, {"input_ids": dummy.numpy()})[0]
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_export_encoder_and_items(self, model: UniSRecModel, tmp_path: Path) -> None:
        enc_path = tmp_path / "encoder.onnx"
        items_path = tmp_path / "items.onnx"
        model.export_to_onnx(str(enc_path), items_path=str(items_path))

        loaded_enc = onnx.load(str(enc_path))
        onnx.checker.check_model(loaded_enc)
        loaded_items = onnx.load(str(items_path))
        onnx.checker.check_model(loaded_items)

    def test_items_roundtrip(self, model: UniSRecModel, tmp_path: Path) -> None:
        items_path = tmp_path / "items.onnx"
        model.export_to_onnx(str(tmp_path / "enc.onnx"), items_path=str(items_path))
        sess = ort.InferenceSession(str(items_path))
        with torch.no_grad():
            expected = model.net.project_all().numpy()
        result = sess.run(None, {})[0]
        assert result.shape == (11, 16)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_unfitted_model_raises(self, tmp_path: Path) -> None:
        pretrained = torch.randn(5, 8)
        m = UniSRecModel(pretrained_item_embeddings=pretrained, n_factors=8)
        with pytest.raises(AssertionError):
            m.export_to_onnx(str(tmp_path / "model.onnx"))
