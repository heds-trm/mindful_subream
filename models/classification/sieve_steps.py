import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Sequence, Union

from mindful_core.data.subset_id import SubsetID
from mindful_core.models.model_output import ClassifierOutput, PrototypeOutput
from mindful_core.models.classification.abstract_classifier import AbstractClassifier
from mindful_core.models.classification.dense_classifier import DenseClassifier
from mindful_core.models.representation.encoders.factory import (
    FusionTransformer, 
    MultiModalEncoder, 
    make_encoders, 
    LSTMFusion
)
from mindful_core.models.loss_aggregator import LossAggregator
from mindful_core.models.attention_interface import AttentionInterface
from mindful_core.utils.metrics import batched_corrcoef
from mindful_subream.models.classification.sieve import MultimodalSieve


class TemporalSteps(MultimodalSieve):
    """
    Modèle ViT-LSTM avec détachement stochastique des gradients
    pour économiser la mémoire lors du traitement des phases
    """
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "temporal_steps"
    
#    def encode_modalities(self,
#                        inputs: Sequence[torch.Tensor],
#                        ) -> torch.Tensor:
#        encoded = super(TemporalSteps, self).encode_modalities(inputs)
#        encoded = self.random_branch_detached(encoded)
#        return encoded

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
                
                # nouveaux paramètres pour le stochastic detach
                 stochastic_detach_prob: float = 0.0,  # Probabilité de détacher chaque phase
                 stochastic_detach_mode: str = "random",  # "random" = chaque phase a une proba d'être detach, "keep_first_last" = garde toujours la 1ère phase et dernière phase et detach randomly le milieu, "keep_every_n"= garde 1 phase sur N (si N=3 --> 0, 3, 6, 9, 12)
                 keep_every_n: int = 2,  # Si mode "keep_every_n", garder 1 phase sur N
                 min_phases_with_grad: int = 2,  # Nombre minimum de phases avec gradient
                 
                 **kwargs):
        super(TemporalSteps, self).__init__(encoders_config=encoders_config,
                                            sieve_config=sieve_config,
                                            classifier_config=classifier_config,
                                            
                                            class_count=class_count,
                                            optimizer_config=optimizer_config,
                                            label_smoothing=label_smoothing,
                                            
                                            yield_confidence = yield_confidence,
                                            confidence_lambda=confidence_lambda,
                                            confidence_corr_lambda=confidence_corr_lambda,
                                            confidence_budget=confidence_budget,
                                            
                                            prototype_model_config=prototype_model_config,
                                            prototype_lambda=prototype_lambda,
                                            
                                            positive_class=positive_class,
                                            use_focal_loss=use_focal_loss,
                                            sieve_mutual_lambda=sieve_mutual_lambda,
                                            sieve_exclusive_lambda=sieve_exclusive_lambda,
                                            **kwargs)
        
        # Paramètres stochastic detach
        self.stochastic_detach_prob = stochastic_detach_prob
        self.stochastic_detach_mode = stochastic_detach_mode
        self.keep_every_n = keep_every_n
        self.min_phases_with_grad = min_phases_with_grad
        
    # region nouvelle partie 
    def make_sieve(self, **sieve_config):
        return make_sieve_steps_fusion_module(input_size=self.hidden_size, pooling=None,
                                              add_cls_token=True,
                                              modality_count=self.modality_count,
                                              **sieve_config)
        
    def make_representation_aggregator(self, **classifier_config):
        return make_sieve_steps_fusion_module(input_size=self.hidden_size, pooling="cls", 
                                              add_cls_token=True,
                                              modality_count=self.modality_count + 1,
                                              pre_activation=None,
                                              **classifier_config)
    
    def random_branch_detached(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Applique un détachement stochastique des gradients sur les phases temporelles.
        
        Pour un batch de 12 phases IRM encodées par ViT, détache aléatoirement
        certaines phases pour économiser la mémoire GPU lors du backprop.
        
        Args:
            encoded: Tensor [batch_size, num_phases, hidden_size]
                    où num_phases
        
        Returns:
            Tensor avec certaines phases détachées selon la stratégie choisie
        """
        if not self.training:
            # En mode évaluation, pas de détachement
            return encoded
        
        batch_size, num_phases, hidden_size = encoded.shape
        
        # Déterminer quelles phases garder avec gradient
        keep_grad_mask = self._get_keep_gradient_mask(num_phases)
        if all(keep_grad_mask):
            raise RuntimeError("tout est bloque")
        
        # Appliquer le détachement phase par phase
        detached_encoded = []
        for phase_idx in range(num_phases):
            encoded_phase = encoded[:, phase_idx:phase_idx+1, :]
            if keep_grad_mask[phase_idx]:
                # Garder le gradient pour cette phase
                detached_encoded.append(encoded_phase)
            else:
                # Détacher le gradient pour cette phase
                detached_encoded.append(encoded_phase.detach())
            print(phase_idx, keep_grad_mask[phase_idx])
        
        return torch.cat(detached_encoded, dim=1)
    
    def _get_keep_gradient_mask(self, num_phases: int) -> list[bool]:
        """
        Génère un masque indiquant quelles phases garder avec gradient.
        
        Args:
            num_phases: Nombre total de phases
        
        Returns:
            Liste de booléens [True, False, True, ...] de longueur num_phases
        """
        if self.stochastic_detach_mode in ["random", "keep_first_last"]:
            # Mode aléatoire : chaque phase a une probabilité p d'être gardée
            rand = torch.rand([num_phases])
            keep = rand < (1.0 - self.stochastic_detach_prob)
            
            if self.stochastic_detach_mode == "keep_first_last":
                keep[0] = keep[-1] = True
                
            total_kept = keep.to(torch.int32).sum()
            
            # S'assurer d'avoir au moins min_phases_with_grad phases avec gradient
            if total_kept < self.min_phases_with_grad:
                order = torch.argsort(rand)
                keep_count = self.min_phases_with_grad
                detach_count = num_phases - self.min_phases_with_grad
                keep = torch.as_tensor([True] * keep_count + [False] * detach_count)
                keep = keep[order]
        
        elif self.stochastic_detach_mode == "keep_every_n":
            # Garder 1 phase sur N de façon déterministe
            keep = [i % self.keep_every_n == 0 for i in range(num_phases)]
        else:
            # Mode désactivé : tout garder
            keep = [True] * num_phases
            
        return keep
    
    def encode_modalities(self,
                        inputs: Sequence[torch.Tensor],
                        ) -> torch.Tensor:
        """
        Encode chaque modalité (phase temporelle IRM) avec le ViT
        puis applique le détachement stochastique.
        """
        # Appel à la méthode parente pour encoder
        encoded = super(TemporalSteps, self).encode_modalities(inputs)
        
        # Application du détachement stochastique
        encoded = self.random_branch_detached(encoded)
        
        return encoded
    # endregion    

    # region AttentionInterface
    def get_attention_modules(self) -> Union["AttentionInterface", list["AttentionInterface"]]:
        raise NotImplementedError("Need first to check if sieve and representation_aggregator are transformers")
        # return [self.sieve, self.representation_aggregator]

    # endregion

def make_sieve_steps_fusion_module(input_size: int,
                                   pooling,
                                   add_cls_token: bool,
                                   modality_count: int,
                                   pre_activation: str | None = "ReLU",
                                   project_output: bool = False,
                                   use_lstm: bool = False,
                                   **kwargs
                                   ) -> FusionTransformer | LSTMFusion:
    if use_lstm:
        proj_size = input_size if project_output else 0
        bidirectional = kwargs.get("bidirectional", False)
        if bidirectional:
            proj_size = proj_size // 2
        return LSTMFusion(input_size=input_size, 
                          pooling=pooling, 
                          use_cls_token=add_cls_token,
                          proj_size=proj_size,
                          **kwargs)
    else:
        return FusionTransformer(input_size=input_size, 
                                 pooling=pooling,
                                 add_cls_token=add_cls_token,
                                 modality_count=modality_count,
                                 pre_activation=pre_activation,
                                 project_output=project_output,
                                 **kwargs)
    