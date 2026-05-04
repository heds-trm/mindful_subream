import torch
from typing import Sequence

from mindful_core.models.representation.encoders.multimodal.multimodal_encoder import MultiModalEncoder



class MultiModalEncoderSteps(MultiModalEncoder):
    def __init__(self,
                 encoders: list[nn.Module],
                 fusion_module: FusionModule,
                 output_module: Optional[nn.Module],
                 modalities: list[str] = None,
                 intermediate_sizes: list[int] = None,
                 modality_dropout: float = None,
                 batch_dropout: bool = False,
                 
                # nouveaux paramètres pour le stochastic detach
                 stochastic_detach_prob: float = 0.0,  # Probabilité de détacher chaque phase
                 stochastic_detach_mode: str = "random",  # "random" = chaque phase a une proba d'être detach, "keep_first_last" = garde toujours la 1ère phase et dernière phase et detach randomly le milieu, "keep_every_n"= garde 1 phase sur N (si N=3 --> 0, 3, 6, 9, 12)
                 keep_every_n: int = 2,  # Si mode "keep_every_n", garder 1 phase sur N
                 min_phases_with_grad: int = 2,  # Nombre minimum de phases avec gradient
                 *args, **kwargs):
        super().__init__(
                        encoders=encoders,
                        fusion_module=fusion_module,
                        output_module=output_module,
                        intermediate_sizes=intermediate_sizes,
                        modalities=modalities,
                        intermediate_sizes=intermediate_sizes,
                        modality_dropout=modality_dropout,
                        batch_dropout=batch_dropout,
                        *args, **kwargs)
    
        self.stochastic_detach_prob = stochastic_detach_prob
        self.stochastic_detach_mode = stochastic_detach_mode
        self.keep_every_n = keep_every_n
        self.min_phases_with_grad = min_phases_with_grad  
            

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
        
        return torch.cat(detached_encoded, dim=1)
    
    def _get_keep_gradient_mask(self, num_phases: int) -> torch.Tensor:
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
            
        keep = torch.as_tensor(keep)
        return keep
    
    def encode_modalities(self,
                        inputs: Sequence[torch.Tensor],
                        ) -> torch.Tensor:
        """
        Encode chaque modalité (phase temporelle IRM) avec le ViT
        puis applique le détachement stochastique.
        """
        # Appel à la méthode parente pour encoder
        encoded = super().encode_modalities(inputs)
        
        # Application du détachement stochastique
        encoded = self.random_branch_detached(encoded)
        
        return encoded
    # endregion


        
