#  Copyright 2025 MTS (Mobile Telesystems)
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

import io
import typing as tp
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import typing_extensions as tpe
from pydantic import BeforeValidator, PlainSerializer
from pytorch_lightning import Trainer

from rectools import ExternalIds
from rectools.dataset.dataset import Dataset, DatasetSchema, DatasetSchemaDict, IdMap
from rectools.models.base import ErrorBehaviour, InternalRecoTriplet, ModelBase, ModelConfig
from rectools.types import InternalIdsArray
from rectools.utils.misc import get_class_or_function_full_path, import_object

from ..item_net import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    ItemNetConstructorBase,
    SumOfEmbeddingsConstructor,
)
from .data_preparator import TransformerDataPreparatorBase
from .lightning import TransformerLightningModule, TransformerLightningModuleBase
from .negative_sampler import CatalogUniformSampler, TransformerNegativeSamplerBase
from .net_blocks import (
    LearnableInversePositionalEncoding,
    PositionalEncodingBase,
    PreLNTransformerLayers,
    TransformerLayersBase,
)
from .similarity import DistanceSimilarityModule, SimilarityModuleBase
from .torch_backbone import TransformerBackboneBase, TransformerTorchBackbone

InitKwargs = tp.Dict[str, tp.Any]

# ####  --------------  Transformer Model Config  --------------  #### #


def _get_class_obj(spec: tp.Any) -> tp.Any:
    if not isinstance(spec, str):
        return spec
    return import_object(spec)


def _get_class_obj_sequence(spec: tp.Sequence[tp.Any]) -> tp.Tuple[tp.Any, ...]:
    return tuple(map(_get_class_obj, spec))


def _serialize_type_sequence(obj: tp.Sequence[tp.Type]) -> tp.Tuple[str, ...]:
    return tuple(map(get_class_or_function_full_path, obj))


PositionalEncodingType = tpe.Annotated[
    tp.Type[PositionalEncodingBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

TransformerLayersType = tpe.Annotated[
    tp.Type[TransformerLayersBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

TransformerLightningModuleType = tpe.Annotated[
    tp.Type[TransformerLightningModuleBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

SimilarityModuleType = tpe.Annotated[
    tp.Type[SimilarityModuleBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

TransformerBackboneType = tpe.Annotated[
    tp.Type[TransformerBackboneBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

TransformerDataPreparatorType = tpe.Annotated[
    tp.Type[TransformerDataPreparatorBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

TransformerNegativeSamplerType = tpe.Annotated[
    tp.Type[TransformerNegativeSamplerBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]


ItemNetConstructorType = tpe.Annotated[
    tp.Type[ItemNetConstructorBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

ItemNetBlockTypes = tpe.Annotated[
    tp.Sequence[tp.Type[ItemNetBase]],
    BeforeValidator(_get_class_obj_sequence),
    PlainSerializer(
        func=_serialize_type_sequence,
        return_type=str,
        when_used="json",
    ),
]


ValMaskCallable = Callable[[], np.ndarray]

ValMaskCallableSerialized = tpe.Annotated[
    ValMaskCallable,
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]

TrainerCallable = Callable[[], Trainer]

TrainerCallableSerialized = tpe.Annotated[
    TrainerCallable,
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]


class TransformerModelConfig(ModelConfig):
    """Transformer model base config."""

    data_preparator_type: TransformerDataPreparatorType
    n_blocks: int = 2
    n_heads: int = 4
    n_factors: int = 256
    use_pos_emb: bool = True
    use_causal_attn: bool = False
    use_key_padding_mask: bool = False
    dropout_rate: float = 0.2
    session_max_len: int = 100
    dataloader_num_workers: int = 0
    batch_size: int = 128
    loss: str = "softmax"
    n_negatives: int = 1
    gbce_t: float = 0.2
    lr: float = 0.001
    epochs: int = 3
    verbose: int = 0
    deterministic: bool = False
    recommend_batch_size: int = 256
    recommend_torch_device: tp.Optional[str] = None
    train_min_user_interactions: int = 2
    item_net_block_types: ItemNetBlockTypes = (IdEmbeddingsItemNet, CatFeaturesItemNet)
    item_net_constructor_type: ItemNetConstructorType = SumOfEmbeddingsConstructor
    pos_encoding_type: PositionalEncodingType = LearnableInversePositionalEncoding
    transformer_layers_type: TransformerLayersType = PreLNTransformerLayers
    lightning_module_type: TransformerLightningModuleType = TransformerLightningModule
    negative_sampler_type: TransformerNegativeSamplerType = CatalogUniformSampler
    similarity_module_type: SimilarityModuleType = DistanceSimilarityModule
    backbone_type: TransformerBackboneType = TransformerTorchBackbone
    get_val_mask_func: tp.Optional[ValMaskCallableSerialized] = None
    get_trainer_func: tp.Optional[TrainerCallableSerialized] = None
    data_preparator_kwargs: tp.Optional[InitKwargs] = None
    transformer_layers_kwargs: tp.Optional[InitKwargs] = None
    item_net_constructor_kwargs: tp.Optional[InitKwargs] = None
    pos_encoding_kwargs: tp.Optional[InitKwargs] = None
    lightning_module_kwargs: tp.Optional[InitKwargs] = None
    negative_sampler_kwargs: tp.Optional[InitKwargs] = None
    similarity_module_kwargs: tp.Optional[InitKwargs] = None
    backbone_kwargs: tp.Optional[InitKwargs] = None


TransformerModelConfig_T = tp.TypeVar("TransformerModelConfig_T", bound=TransformerModelConfig)


# ####  --------------  Transformer Model Base  --------------  #### #


class TransformerModelBase(ModelBase[TransformerModelConfig_T]):  # pylint: disable=too-many-instance-attributes
    """
    Base model for all recommender algorithms that work on transformer architecture (e.g. SASRec, Bert4Rec).
    To create a custom transformer model it is necessary to inherit from this class
    and write self.data_preparator initialization logic.
    """

    config_class: tp.Type[TransformerModelConfig_T]
    train_loss_name: str = "train_loss"
    val_loss_name: str = "val_loss"

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        data_preparator_type: tp.Type[TransformerDataPreparatorBase],
        transformer_layers_type: tp.Type[TransformerLayersBase] = PreLNTransformerLayers,
        n_blocks: int = 2,
        n_heads: int = 4,
        n_factors: int = 256,
        use_pos_emb: bool = True,
        use_causal_attn: bool = False,
        use_key_padding_mask: bool = False,
        dropout_rate: float = 0.2,
        session_max_len: int = 100,
        dataloader_num_workers: int = 0,
        batch_size: int = 128,
        loss: str = "softmax",
        n_negatives: int = 1,
        gbce_t: float = 0.2,
        lr: float = 0.001,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        recommend_batch_size: int = 256,
        recommend_torch_device: tp.Optional[str] = None,
        train_min_user_interactions: int = 2,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        item_net_constructor_type: tp.Type[ItemNetConstructorBase] = SumOfEmbeddingsConstructor,
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        lightning_module_type: tp.Type[TransformerLightningModuleBase] = TransformerLightningModule,
        negative_sampler_type: tp.Type[TransformerNegativeSamplerBase] = CatalogUniformSampler,
        similarity_module_type: tp.Type[SimilarityModuleBase] = DistanceSimilarityModule,
        backbone_type: tp.Type[TransformerBackboneBase] = TransformerTorchBackbone,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        get_trainer_func: tp.Optional[TrainerCallable] = None,
        data_preparator_kwargs: tp.Optional[InitKwargs] = None,
        transformer_layers_kwargs: tp.Optional[InitKwargs] = None,
        item_net_constructor_kwargs: tp.Optional[InitKwargs] = None,
        pos_encoding_kwargs: tp.Optional[InitKwargs] = None,
        lightning_module_kwargs: tp.Optional[InitKwargs] = None,
        negative_sampler_kwargs: tp.Optional[InitKwargs] = None,
        similarity_module_kwargs: tp.Optional[InitKwargs] = None,
        backbone_kwargs: tp.Optional[InitKwargs] = None,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(verbose=verbose)
        self.transformer_layers_type = transformer_layers_type
        self.data_preparator_type = data_preparator_type
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_factors = n_factors
        self.use_pos_emb = use_pos_emb
        self.use_causal_attn = use_causal_attn
        self.use_key_padding_mask = use_key_padding_mask
        self.dropout_rate = dropout_rate
        self.session_max_len = session_max_len
        self.dataloader_num_workers = dataloader_num_workers
        self.batch_size = batch_size
        self.loss = loss
        self.n_negatives = n_negatives
        self.gbce_t = gbce_t
        self.lr = lr
        self.epochs = epochs
        self.deterministic = deterministic
        self.recommend_batch_size = recommend_batch_size
        self.recommend_torch_device = recommend_torch_device
        self.train_min_user_interactions = train_min_user_interactions
        self.similarity_module_type = similarity_module_type
        self.item_net_block_types = item_net_block_types
        self.item_net_constructor_type = item_net_constructor_type
        self.pos_encoding_type = pos_encoding_type
        self.lightning_module_type = lightning_module_type
        self.negative_sampler_type = negative_sampler_type
        self.backbone_type = backbone_type
        self.get_val_mask_func = get_val_mask_func
        self.get_trainer_func = get_trainer_func
        self.data_preparator_kwargs = data_preparator_kwargs
        self.transformer_layers_kwargs = transformer_layers_kwargs
        self.item_net_constructor_kwargs = item_net_constructor_kwargs
        self.pos_encoding_kwargs = pos_encoding_kwargs
        self.lightning_module_kwargs = lightning_module_kwargs
        self.negative_sampler_kwargs = negative_sampler_kwargs
        self.similarity_module_kwargs = similarity_module_kwargs
        self.backbone_kwargs = backbone_kwargs

        self._init_data_preparator()
        self._init_trainer()

        self.lightning_model: TransformerLightningModuleBase
        self.data_preparator: TransformerDataPreparatorBase
        self.fit_trainer: tp.Optional[Trainer] = None

    @staticmethod
    def _get_kwargs(actual_kwargs: tp.Optional[InitKwargs]) -> InitKwargs:
        kwargs = {}
        if actual_kwargs is not None:
            kwargs = actual_kwargs
        return kwargs

    def _init_data_preparator(self) -> None:
        requires_negatives = self.lightning_module_type.requires_negatives(self.loss)
        self.data_preparator = self.data_preparator_type(
            session_max_len=self.session_max_len,
            batch_size=self.batch_size,
            dataloader_num_workers=self.dataloader_num_workers,
            train_min_user_interactions=self.train_min_user_interactions,
            negative_sampler=self._init_negative_sampler() if requires_negatives else None,
            n_negatives=self.n_negatives if requires_negatives else None,
            get_val_mask_func=self.get_val_mask_func,
            shuffle_train=True,
            **self._get_kwargs(self.data_preparator_kwargs),
        )

    def _init_trainer(self) -> None:
        if self.get_trainer_func is None:
            self._trainer = Trainer(
                max_epochs=self.epochs,
                min_epochs=self.epochs,
                deterministic=self.deterministic,
                enable_progress_bar=self.verbose > 0,
                enable_model_summary=self.verbose > 0,
                logger=self.verbose > 0,
                enable_checkpointing=False,
                devices=1,
            )
        else:
            self._trainer = self.get_trainer_func()

    def _init_negative_sampler(self) -> TransformerNegativeSamplerBase:
        return self.negative_sampler_type(
            n_negatives=self.n_negatives,
            **self._get_kwargs(self.negative_sampler_kwargs),
        )

    def _construct_item_net(self, dataset: Dataset) -> ItemNetBase:
        return self.item_net_constructor_type.from_dataset(
            dataset,
            self.n_factors,
            self.dropout_rate,
            self.item_net_block_types,
            **self._get_kwargs(self.item_net_constructor_kwargs),
        )

    def _construct_item_net_from_dataset_schema(self, dataset_schema: DatasetSchema) -> ItemNetBase:
        return self.item_net_constructor_type.from_dataset_schema(
            dataset_schema,
            self.n_factors,
            self.dropout_rate,
            self.item_net_block_types,
            **self._get_kwargs(self.item_net_constructor_kwargs),
        )

    def _init_pos_encoding_layer(self) -> PositionalEncodingBase:
        return self.pos_encoding_type(
            self.use_pos_emb,
            self.session_max_len,
            self.n_factors,
            **self._get_kwargs(self.pos_encoding_kwargs),
        )

    def _init_transformer_layers(self) -> TransformerLayersBase:
        return self.transformer_layers_type(
            n_blocks=self.n_blocks,
            n_factors=self.n_factors,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            **self._get_kwargs(self.transformer_layers_kwargs),
        )

    def _init_similarity_module(self) -> SimilarityModuleBase:
        return self.similarity_module_type(**self._get_kwargs(self.similarity_module_kwargs))

    def _init_torch_model(self, item_model: ItemNetBase) -> TransformerBackboneBase:
        pos_encoding_layer = self._init_pos_encoding_layer()
        transformer_layers = self._init_transformer_layers()
        similarity_module = self._init_similarity_module()
        return self.backbone_type(
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            item_model=item_model,
            pos_encoding_layer=pos_encoding_layer,
            transformer_layers=transformer_layers,
            similarity_module=similarity_module,
            use_causal_attn=self.use_causal_attn,
            use_key_padding_mask=self.use_key_padding_mask,
            **self._get_kwargs(self.backbone_kwargs),
        )

    def _init_lightning_model(
        self,
        torch_model: TransformerBackboneBase,
        dataset_schema: DatasetSchemaDict,
        item_external_ids: ExternalIds,
        model_config: tp.Dict[str, tp.Any],
    ) -> None:
        self.lightning_model = self.lightning_module_type(
            torch_model=torch_model,
            dataset_schema=dataset_schema,
            item_external_ids=item_external_ids,
            item_extra_tokens=self.data_preparator.item_extra_tokens,
            data_preparator=self.data_preparator,
            model_config=model_config,
            lr=self.lr,
            loss=self.loss,
            gbce_t=self.gbce_t,
            verbose=self.verbose,
            train_loss_name=self.train_loss_name,
            val_loss_name=self.val_loss_name,
            adam_betas=(0.9, 0.98),
            **self._get_kwargs(self.lightning_module_kwargs),
        )

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        self.data_preparator.process_dataset_train(dataset)
        train_dataloader = self.data_preparator.get_dataloader_train()
        val_dataloader = self.data_preparator.get_dataloader_val()

        item_model = self._construct_item_net(self.data_preparator.train_dataset)
        torch_model = self._init_torch_model(item_model)

        dataset_schema = self.data_preparator.train_dataset.get_schema()
        item_external_ids = self.data_preparator.train_dataset.item_id_map.external_ids
        model_config = self.get_config(simple_types=True)
        self._init_lightning_model(
            torch_model=torch_model,
            dataset_schema=dataset_schema,
            item_external_ids=item_external_ids,
            model_config=model_config,
        )

        self.fit_trainer = deepcopy(self._trainer)
        self.fit_trainer.fit(self.lightning_model, train_dataloader, val_dataloader)

    def _custom_transform_dataset_u2i(
        self, dataset: Dataset, users: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        return self.data_preparator.transform_dataset_u2i(dataset, users)

    def _custom_transform_dataset_i2i(
        self, dataset: Dataset, target_items: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        return self.data_preparator.transform_dataset_i2i(dataset)

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,  # [n_rec_users x n_items + n_item_extra_tokens]
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],  # model_internal
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()  # model internal

        recommend_dataloader = self.data_preparator.get_dataloader_recommend(dataset, self.recommend_batch_size)
        return self.lightning_model._recommend_u2i(  # pylint: disable=protected-access
            user_ids=user_ids,
            recommend_dataloader=recommend_dataloader,
            sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
            k=k,
            filter_viewed=filter_viewed,
            dataset=dataset,
            torch_device=self.recommend_torch_device,
        )

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,  # model internal
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()

        return self.lightning_model._recommend_i2i(  # pylint: disable=protected-access
            target_ids=target_ids,
            sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
            k=k,
            torch_device=self.recommend_torch_device,
        )

    @property
    def torch_model(self) -> TransformerBackboneBase:
        """Pytorch model."""
        return self.lightning_model.torch_model

    @classmethod
    def _from_config(cls, config: TransformerModelConfig_T) -> tpe.Self:
        params = config.model_dump()
        params.pop("cls")
        return cls(**params)

    def _get_config(self) -> TransformerModelConfig_T:
        attrs = self.config_class.model_json_schema(mode="serialization")["properties"].keys()
        params = {attr: getattr(self, attr) for attr in attrs if attr != "cls"}
        params["cls"] = self.__class__
        return self.config_class(**params)

    @classmethod
    def _model_from_checkpoint(cls, checkpoint: tp.Dict[str, tp.Any]) -> tpe.Self:
        """Create model from loaded Lightning checkpoint."""
        model_config = checkpoint["hyper_parameters"]["model_config"]
        loaded = cls.from_config(model_config)
        loaded.is_fitted = True
        dataset_schema = checkpoint["hyper_parameters"]["dataset_schema"]
        dataset_schema = DatasetSchema.model_validate(dataset_schema)

        # Update data preparator
        item_external_ids = checkpoint["hyper_parameters"]["item_external_ids"]
        loaded.data_preparator.item_id_map = IdMap(item_external_ids)
        loaded.data_preparator._init_extra_token_ids()  # pylint: disable=protected-access

        # Init and update torch model and lightning model
        item_model = loaded._construct_item_net_from_dataset_schema(dataset_schema)
        torch_model = loaded._init_torch_model(item_model)
        loaded._init_lightning_model(
            torch_model=torch_model,
            dataset_schema=dataset_schema,
            item_external_ids=item_external_ids,
            model_config=model_config,
        )
        loaded.lightning_model.load_state_dict(checkpoint["state_dict"])

        return loaded

    def __getstate__(self) -> object:
        if self.is_fitted:
            if self.fit_trainer is None:
                explanation = """
                Model is fitted but has no `fit_trainer`. Most likely it was just loaded from the
                checkpoint. Model that was loaded from checkpoint cannot be saved without being
                fitted again.
                """
                raise RuntimeError(explanation)
            with NamedTemporaryFile() as f:
                self.fit_trainer.save_checkpoint(f.name)
                checkpoint = Path(f.name).read_bytes()
            state: tp.Dict[str, tp.Any] = {"fitted_checkpoint": checkpoint}
            return state
        state = {"model_config": self.get_config(simple_types=True)}
        return state

    def __setstate__(self, state: tp.Dict[str, tp.Any]) -> None:
        if "fitted_checkpoint" in state:
            checkpoint = torch.load(io.BytesIO(state["fitted_checkpoint"]), weights_only=False)
            loaded = self._model_from_checkpoint(checkpoint)
        else:
            loaded = self.from_config(state["model_config"])

        self.__dict__.update(loaded.__dict__)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: tp.Union[str, Path]) -> tpe.Self:
        """
        Load model from Lightning checkpoint path.

        Parameters
        ----------
        checkpoint_path: Union[str, Path]
            Path to checkpoint location.

        Returns
        -------
        Model instance.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        loaded = cls._model_from_checkpoint(checkpoint)
        return loaded

    def load_weights_from_checkpoint(self, checkpoint_path: tp.Union[str, Path]) -> None:
        """
        Load model weights from Lightning checkpoint path.

        Parameters
        ----------
        checkpoint_path: Union[str, Path]
            Path to checkpoint location.
        """
        if self.fit_trainer is None:
            raise RuntimeError("Model weights cannot be loaded from checkpoint into unfitted model")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.lightning_model.load_state_dict(checkpoint["state_dict"])
