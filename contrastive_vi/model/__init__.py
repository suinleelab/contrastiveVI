"""scvi-tools Model classes for contrastive-VI."""
from .contrastive_vi import ContrastiveVIModel as ContrastiveVI
from .total_contrastive_vi import TotalContrastiveVIModel as TotalContrastiveVI

__all__ = ["ContrastiveVI", "TotalContrastiveVI"]
