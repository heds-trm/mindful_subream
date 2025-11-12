import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Sequence, Union

from mindful_core.data.subset_id import SubsetID
from mindful_core.models.model_output import ClassifierOutput, PrototypeOutput
from mindful_core.models.classification.abstract_classifier import AbstractClassifier
from mindful_core.models.classification.dense_classifier import DenseClassifier
from mindful_core.models.representation.encoders.factory import FusionTransformer, MultiModalEncoder, make_encoders
from mindful_core.models.loss_aggregator import LossAggregator
from mindful_core.models.attention_interface import AttentionInterface
from mindful_core.utils.metrics import batched_corrcoef


class TemporalSteps(MultimodalSieve):
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "temporal_steps"
    
    def encode_modalities(self,
                        inputs: Sequence[torch.Tensor],
                        ) -> torch.Tensor:
        encoded = super(TemporalSteps, self).encode_modalities(inputs)
        encoded = self.random_branch_detached(encoded)
        return encoded

    # endregion

    def __init__(self,
                 encoders_config: dict[str, dict[str, Any]],
                 sieve_config: dict[str, Any],
                 classifier_config: dict[str, Any],
                 class_count: int,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 label_smoothing=0.0,

                 yield_confidence: bool = False,
                 confidence_lambda=1e-1,
                 confidence_corr_lambda=1e-1,
                 confidence_budget=None,

                 prototype_model_config: dict | None = None,
                 prototype_lambda=1e-1,

                 positive_class=-1,
                 use_focal_loss=False,
                 sieve_mutual_lambda=1e-1,
                 sieve_exclusive_lambda=1e-1,
                 **kwargs):
        hidden_size = self._get_hidden_size(encoders_config)
        super(TemporalSteps, self).__init__(class_count=class_count,
                                              optimizer_config=optimizer_config,
                                              label_smoothing=label_smoothing,
                                              confidence_lambda=confidence_lambda,
                                              confidence_corr_lambda=confidence_corr_lambda,
                                              confidence_budget=confidence_budget,
                                              hidden_size=hidden_size,
                                              prototype_model_config=prototype_model_config,
                                              prototype_lambda=prototype_lambda,
                                              positive_class=positive_class,
                                              use_focal_loss=use_focal_loss,
                                              **kwargs)

        encoders = make_encoders(encoders_config)
        for modality, encoder in encoders.items():
            self.register_module("{}_encoder".format(modality), encoder)
        self.modalities = list(encoders.keys())
        self.modality_encoders = list(encoders.values())

        self.sieve = FusionTransformer(input_size=self.hidden_size, pooling=None,
                                       project_output=False, add_cls_token=True,
                                       modality_count=self.modality_count,
                                       **sieve_config)

        self._yield_confidence = yield_confidence
        if yield_confidence:
            class_count += 1
        self.representation_aggregator = FusionTransformer(input_size=self.hidden_size, pooling="cls",
                                                           project_output=False, add_cls_token=True,
                                                           modality_count=self.modality_count + 1,
                                                           pre_activation=None,
                                                           **classifier_config)
        representation_size = self.prototype_model.prototype_count if self.train_prototype_model else self.hidden_size
        self.final_classifier = DenseClassifier(input_dimension=representation_size,
                                                features=[],
                                                class_count=class_count,
                                                yield_confidence=yield_confidence)

        self.sieve_mutual_lambda = sieve_mutual_lambda
        self.sieve_exclusive_lambda = sieve_exclusive_lambda

    def forward(self,
                inputs,
                *args,
                compute_sieve_loss: bool = False,
                **kwargs
                ) -> ClassifierOutput:
        inputs, mask = MultiModalEncoder.unpack_mask(inputs, self.modality_count)

        early_representations = self.encode_modalities(inputs)
        sieve_outputs = self.sieve(early_representations, mask=mask)
        representations = self.representation_aggregator(sieve_outputs)

        if self.train_prototype_model:
            prototype_outputs: PrototypeOutput = self.prototype_model(representations)
            classifier_outputs: ClassifierOutput = self.final_classifier(prototype_outputs.similarities)
            classifier_outputs.prototype_outputs = prototype_outputs
        else:
            classifier_outputs = self.final_classifier(representations)
        classifier_outputs.confidence_threshold = self.confidence_threshold

        if compute_sieve_loss:
            self.compute_sieve_information_loss(early_representations, sieve_outputs)

        return classifier_outputs

    # def encode_modalities(self,
                          # inputs: Sequence[torch.Tensor],
                          # ) -> torch.Tensor:
        # inputs = [encoder(modality)
                  # for encoder, modality
                  # in zip(self.modality_encoders, inputs)]
        # return torch.stack(inputs, dim=1)

    def base_step(self, batch, subset_id: SubsetID, **model_kwargs) -> STEP_OUTPUT:
        return super(TemporalSteps, self).base_step(batch, subset_id, compute_sieve_loss=True, **model_kwargs)

    def compute_sieve_information_loss(self,
                                       early_representations: torch.Tensor,
                                       sieve_outputs: torch.Tensor
                                       ) -> LossAggregator:
        # early_representations:    [batch_size, modality_count,     hidden_size]
        # sieve_outputs:            [batch_size, modality_count + 1, hidden_size]

        # todo: add a hparam that enables/disables detach() on tensor (on by default)
        early_representations = early_representations.detach()

        mutual_token, exclusive_tokens = sieve_outputs[:, :1], sieve_outputs[:, 1:]
        # mutual_corrcoef = batched_corrcoef(torch.concat([mutual_token, early_representations], dim=1))
        # exclusive_corrcoef = batched_corrcoef(torch.concat([mutual_token.detach(), exclusive_tokens], dim=1))

        mutual_corrcoef = batched_corrcoef(mutual_token, early_representations, absolute=True)
        exclusive_corrcoef = batched_corrcoef(mutual_token.detach(), exclusive_tokens, absolute=True)

        compare_count = ((self.modality_count + 1) * self.modality_count) // 2
        mutual_information_loss = 1.0 - mutual_corrcoef[:, 0, 1:].mean()
        exclusive_information_loss = (exclusive_corrcoef[..., 1:].sum(dim=1) / compare_count).mean()

        # mutual_information_loss = torch.abs(mutual_information_loss)
        # exclusive_information_loss = torch.abs(exclusive_information_loss)

        self.loss_aggregator.add_loss(mutual_information_loss, "mutual_information_loss", self.sieve_mutual_lambda)
        self.loss_aggregator.add_loss(exclusive_information_loss, "exclusive_information_loss",
                                      self.sieve_exclusive_lambda)

        return self.loss_aggregator

    @property
    def modality_count(self) -> int:
        return len(self.modalities)

    @staticmethod
    def _get_hidden_size(encoders_config: dict[str, dict[str, Any]]) -> int:
        hidden_sizes = [encoder_config["output_dimension"] for encoder_config in encoders_config.values()]
        hidden_size = hidden_sizes[0]

        if not isinstance(hidden_size, int):
            raise ValueError("Encoder output dimension must be an integer, got {}({})"
                             .format(hidden_size, type(hidden_size)))
        all_same_size = all([encoder_hidden_size == hidden_size for encoder_hidden_size in hidden_sizes[1:]])

        if not all_same_size:
            raise ValueError("All encoder output dimensions must match "
                             "for a MultimodalSieve, got {}".format(hidden_sizes))

        return hidden_size

    @property
    def yield_confidence(self) -> bool:
        return self._yield_confidence

    # region AttentionInterface
    def get_attention_modules(self) -> Union["AttentionInterface", list["AttentionInterface"]]:
        return [self.sieve, self.representation_aggregator]

    def get_attention_layers(self) -> list[nn.Module]:
        raise RuntimeError("Use `get_attention_layers` from `self.sieve` and `self.representation_aggregator` instead.")

    def get_attention_recordings(self, inputs: Any, outputs: Any) -> torch.Tensor | list[torch.Tensor]:
        raise RuntimeError("Use `get_attention_recordings` from `self.sieve`"
                           " and `self.representation_aggregator` instead.")

    @property
    def pooling_method(self) -> int | str:
        return self.sieve.pooling_method

    @property
    def attention_rank(self) -> int:
        return 1

    def reduce_1d_pooling_attention(self, attention_maps: torch.Tensor) -> torch.Tensor:
        return self.sieve.reduce_1d_pooling_attention(attention_maps)

    # endregion
