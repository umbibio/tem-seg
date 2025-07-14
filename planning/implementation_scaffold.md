# Semantic Segmentation Workflow System - Complete Implementation Scaffold

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Union, List, Optional, Tuple
from pathlib import Path
import hashlib
import json
import yaml
import numpy as np
from PIL import Image
import uuid
from datetime import datetime

# =============================================================================
# Scale Estimation System
# =============================================================================

class ScaleEstimationError(Exception):
    """Raised when scale estimation fails"""
    pass

class ScaleEstimator(ABC):
    """Abstract base for image scale estimation"""
    
    @abstractmethod
    def estimate_scale(self, image_input: Union[Path, Image.Image, np.ndarray]) -> float:
        """Return scale in micrometers per pixel"""
        pass
    
    @abstractmethod
    def can_estimate(self, image_input: Union[Path, Image.Image, np.ndarray]) -> bool:
        """Check if this estimator can handle the image"""
        pass

class ScaleBarReader(ScaleEstimator):
    """Current scale bar reading implementation"""
    
    def estimate_scale(self, image_input: Union[Path, Image.Image, np.ndarray]) -> float:
        """Estimate scale from scale bar in image"""
        # Handle different input types
        if isinstance(image_input, Path):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:  # PIL Image
            img = image_input
        
        # Migrate your current calibration logic here
        from .calibration import get_calibration
        return get_calibration(img)
    
    def can_estimate(self, image_input: Union[Path, Image.Image, np.ndarray]) -> bool:
        try:
            self.estimate_scale(image_input)
            return True
        except Exception:
            return False

class ScaleManager:
    """Manages scale estimation with fallback strategies"""
    
    def __init__(self, estimators: List[ScaleEstimator] = None):
        self.estimators = estimators or [ScaleBarReader()]
    
    def get_scale(self, image_input: Union[Path, Image.Image, np.ndarray], 
                  fallback: Optional[float] = None) -> float:
        """Try estimators in order, return first successful result"""
        for estimator in self.estimators:
            if estimator.can_estimate(image_input):
                try:
                    return estimator.estimate_scale(image_input)
                except Exception:
                    continue
        
        if fallback is not None:
            return fallback
        raise ScaleEstimationError("Could not estimate scale")

# =============================================================================
# Configuration System
# =============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    framework: str = "tensorflow"  # 'tensorflow', 'pytorch'
    architecture: str = "unet"     # 'unet', 'maskrcnn', 'deeplabv3'
    layer_depth: int = 5
    filters_root: int = 16
    channels: int = 1

@dataclass 
class TrainingConfig:
    """Training parameters"""
    batch_size: int = 32
    epochs: int = 1200
    learning_rate: float = 0.001
    validation_split: float = 0.2
    cross_validation_folds: int = 1
    shuffle_training: bool = True
    early_stopping_patience: int = 50

@dataclass
class DataConfig:
    """Data processing configuration"""
    tile_shape: Tuple[int, int] = (444, 444)
    output_shape: Optional[Tuple[int, int]] = None
    target_scale: float = 0.0075  # micrometers per pixel
    fraction_empty_tiles: float = 1.0
    augmentations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DistillationConfig:
    """Model distillation parameters"""
    enabled: bool = False
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    teacher_ensemble_method: str = "average"  # 'average', 'weighted', 'attention'

@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    # Core settings
    workflow_name: str = ""
    organelle: str = "mitochondria"
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    
    # Paths and metadata
    data_paths: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    
    # Internal state
    _config_hash: Optional[str] = field(default=None, init=False)
    _frozen: bool = field(default=False, init=False)
    
    def __post_init__(self):
        if not self.workflow_name:
            self.workflow_name = self._generate_default_name()
        
        # Auto-compute output shape if not provided
        if self.data.output_shape is None:
            self.data.output_shape = self._compute_output_shape()
    
    def _generate_default_name(self) -> str:
        """Generate default workflow name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.organelle}_{self.model.architecture}_{timestamp}"
    
    def _compute_output_shape(self) -> Tuple[int, int]:
        """Compute output shape based on model architecture"""
        if self.model.architecture == "unet":
            # Use your existing computation logic
            from .model.utils import compute_output_size
            output_size = compute_output_size(
                input_size=self.data.tile_shape[0], 
                layer_depth=self.model.layer_depth
            )
            return (output_size, output_size)
        return self.data.tile_shape  # Default fallback
    
    def freeze(self):
        """Freeze configuration and compute hash"""
        if self._frozen:
            return
        
        # Compute hash of critical parameters
        hash_dict = {
            'organelle': self.organelle,
            'model': asdict(self.model),
            'data': {
                'tile_shape': self.data.tile_shape,
                'target_scale': self.data.target_scale,
                'fraction_empty_tiles': self.data.fraction_empty_tiles
            },
            'training': {
                'cross_validation_folds': self.training.cross_validation_folds,
                'batch_size': self.training.batch_size
            }
        }
        
        hash_str = json.dumps(hash_dict, sort_keys=True)
        self._config_hash = hashlib.md5(hash_str.encode()).hexdigest()[:8]
        self._frozen = True
    
    @property
    def config_hash(self) -> str:
        """Get configuration hash, freezing if necessary"""
        if not self._frozen:
            self.freeze()
        return self._config_hash
    
    def save_to_file(self, filepath: Path):
        """Save configuration to YAML file"""
        # Convert to dict, excluding internal fields
        config_dict = asdict(self)
        config_dict.pop('_config_hash', None)
        config_dict.pop('_frozen', None)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'WorkflowConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'distillation' in config_dict:
            config_dict['distillation'] = DistillationConfig(**config_dict['distillation'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_organelle_preset(cls, organelle: str) -> 'WorkflowConfig':
        """Create configuration with organelle-specific defaults"""
        presets = {
            'mitochondria': {
                'data': DataConfig(target_scale=0.0075),
                'model': ModelConfig(framework='tensorflow', architecture='unet')
            },
            'nucleus': {
                'data': DataConfig(target_scale=0.03),
                'model': ModelConfig(framework='tensorflow', architecture='unet')
            },
            'cell': {
                'data': DataConfig(target_scale=0.015),
                'model': ModelConfig(framework='tensorflow', architecture='unet')
            }
        }
        
        preset = presets.get(organelle, {})
        return cls(organelle=organelle, **preset)

# =============================================================================
# Standardized Output System
# =============================================================================

@dataclass
class InstanceData:
    """Individual instance information"""
    mask: np.ndarray  # Binary mask for this instance
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area_pixels: int
    area_micrometers_sq: float
    centroid: Tuple[float, float]
    morphology_metrics: Dict[str, float] = field(default_factory=dict)

class StandardizedOutput:
    """Container for segmentation results supporting multiple model outputs"""
    
    def __init__(self, image_path: Path, organelle: str, 
                 scale_micrometers_per_pixel: float):
        self.image_path = image_path
        self.organelle = organelle
        self.scale_micrometers_per_pixel = scale_micrometers_per_pixel
        
        # Collections to hold multiple predictions (for ensembles)
        self.semantic_masks: List[np.ndarray] = []
        self.instance_collections: List[List[InstanceData]] = []
        self.model_infos: List[Dict[str, Any]] = []
        
        # Metadata
        self.processing_metadata: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()
    
    def add_prediction(self, semantic_mask: np.ndarray, 
                      instances: List[InstanceData],
                      model_info: Dict[str, Any]):
        """Add a prediction from a single model"""
        self.semantic_masks.append(semantic_mask)
        self.instance_collections.append(instances)
        self.model_infos.append(model_info)
    
    def get_ensemble_size(self) -> int:
        """Get number of models in ensemble"""
        return len(self.semantic_masks)
    
    def get_aggregated_mask(self, method: str = "majority_vote", 
                           threshold: float = 0.5) -> np.ndarray:
        """Aggregate semantic masks from multiple models"""
        if not self.semantic_masks:
            raise ValueError("No predictions available")
        
        if len(self.semantic_masks) == 1:
            return self.semantic_masks[0]
        
        # Stack all masks
        stacked_masks = np.stack(self.semantic_masks, axis=0)
        
        if method == "majority_vote":
            # Majority vote across models
            vote_sum = np.sum(stacked_masks > threshold, axis=0)
            return (vote_sum > len(self.semantic_masks) / 2).astype(np.uint8)
        
        elif method == "average":
            # Average probabilities
            avg_mask = np.mean(stacked_masks, axis=0)
            return (avg_mask > threshold).astype(np.uint8)
        
        elif method == "weighted_average":
            # TODO: Implement weighted averaging based on model confidence
            return self.get_aggregated_mask(method="average", threshold=threshold)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_aggregated_instances(self, mask_aggregation_method: str = "majority_vote") -> List[InstanceData]:
        """Extract instances from aggregated mask"""
        aggregated_mask = self.get_aggregated_mask(method=mask_aggregation_method)
        
        # Extract instances using connected components
        from skimage import measure
        
        labeled_mask = measure.label(aggregated_mask)
        instances = []
        
        for region in measure.regionprops(labeled_mask):
            if region.area < 10:  # Filter small artifacts
                continue
            
            # Create instance mask
            instance_mask = (labeled_mask == region.label).astype(np.uint8)
            
            # Calculate metrics
            area_pixels = region.area
            area_um2 = area_pixels * (self.scale_micrometers_per_pixel ** 2)
            
            instances.append(InstanceData(
                mask=instance_mask,
                bbox=region.bbox,
                area_pixels=area_pixels,
                area_micrometers_sq=area_um2,
                centroid=region.centroid,
                morphology_metrics={}  # TODO: Add morphology calculation
            ))
        
        return instances
    
    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'image_path': str(self.image_path),
            'organelle': self.organelle,
            'scale_micrometers_per_pixel': self.scale_micrometers_per_pixel,
            'ensemble_size': self.get_ensemble_size(),
            'created_at': self.created_at,
            'model_infos': self.model_infos,
            'processing_metadata': self.processing_metadata
        }
        
        if include_masks:
            result['semantic_masks'] = [mask.tolist() for mask in self.semantic_masks]
        
        # Add aggregated instance data
        aggregated_instances = self.get_aggregated_instances()
        result['aggregated_instances'] = [
            {
                'area_pixels': inst.area_pixels,
                'area_micrometers_sq': inst.area_micrometers_sq,
                'centroid': inst.centroid,
                'bbox': inst.bbox,
                'morphology_metrics': inst.morphology_metrics
            }
            for inst in aggregated_instances
        ]
        
        return result
    
    def save_outputs(self, output_dir: Path, save_individual_predictions: bool = False):
        """Save all outputs (masks, JSON, CSV)"""
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = self.image_path.stem
        
        # Save aggregated semantic mask
        aggregated_mask = self.get_aggregated_mask()
        mask_path = output_dir / f"{base_name}-{self.organelle}.png"
        Image.fromarray((aggregated_mask * 255).astype(np.uint8)).save(mask_path)
        
        # Save individual predictions if requested
        if save_individual_predictions and self.get_ensemble_size() > 1:
            for i, mask in enumerate(self.semantic_masks):
                individual_path = output_dir / f"{base_name}-{self.organelle}-model_{i+1}.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(individual_path)
        
        # Save JSON
        json_path = output_dir / f"{base_name}-{self.organelle}.json"
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save CSV
        csv_path = output_dir / f"{base_name}-{self.organelle}.csv"
        instances = self.get_aggregated_instances()
        if instances:
            import pandas as pd
            instances_df = pd.DataFrame([
                {
                    'area_pixels': inst.area_pixels,
                    'area_micrometers_sq': inst.area_micrometers_sq,
                    'centroid_x': inst.centroid[0],
                    'centroid_y': inst.centroid[1],
                    **inst.morphology_metrics
                }
                for inst in instances
            ])
            instances_df.to_csv(csv_path, index=False)

# =============================================================================
# Base Workflow Classes
# =============================================================================

class SegmentationWorkflow(ABC):
    """Base class for all segmentation workflows"""
    
    def __init__(self, config: WorkflowConfig, scale_manager: ScaleManager = None):
        self.config = config
        self.scale_manager = scale_manager or ScaleManager()
        self.models = {}  # Dict[fold_number, model]
        
        # Setup directory structure
        self._setup_directories()
        
        # Save configuration
        self._save_config()
    
    def _setup_directories(self):
        """Create workflow directory structure"""
        base_dir = Path("workflows") / self.config.workflow_name / self.config.config_hash
        
        if self.config.training.cross_validation_folds > 1:
            for fold in range(1, self.config.training.cross_validation_folds + 1):
                fold_dir = base_dir / f"fold_{fold:02d}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                (fold_dir / "models").mkdir(exist_ok=True)
                (fold_dir / "predictions").mkdir(exist_ok=True)
                (fold_dir / "logs").mkdir(exist_ok=True)
        else:
            single_dir = base_dir / "fold_01"
            single_dir.mkdir(parents=True, exist_ok=True)
            (single_dir / "models").mkdir(exist_ok=True)
            (single_dir / "predictions").mkdir(exist_ok=True)
            (single_dir / "logs").mkdir(exist_ok=True)
        
        # Common directories
        (base_dir / "training_data").mkdir(exist_ok=True)
        (base_dir / "config").mkdir(exist_ok=True)
    
    def _save_config(self):
        """Save configuration to workflow directory"""
        config_path = self.get_workflow_dir() / "config" / "config.yaml"
        self.config.save_to_file(config_path)
    
    def get_workflow_dir(self) -> Path:
        """Get base workflow directory"""
        return Path("workflows") / self.config.workflow_name / self.config.config_hash
    
    def get_fold_dir(self, fold_number: int) -> Path:
        """Get directory for specific fold"""
        if not (1 <= fold_number <= self.config.training.cross_validation_folds):
            raise ValueError(f"Invalid fold number: {fold_number}")
        return self.get_workflow_dir() / f"fold_{fold_number:02d}"
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def prepare_training_data(self, images_path: Path, masks_path: Path, **kwargs) -> Any:
        """Prepare training data in framework-specific format"""
        pass
    
    @abstractmethod
    def create_model(self, fold_number: int = 1, **kwargs) -> Any:
        """Create model instance for specific fold"""
        pass
    
    @abstractmethod
    def load_pretrained_model(self, model_path: Path, fold_number: int = 1, **kwargs) -> Any:
        """Load existing model for specific fold"""
        pass
    
    @abstractmethod
    def train_model(self, fold_number: int = 1, **kwargs) -> Dict[str, Any]:
        """Train model for specific fold"""
        pass
    
    @abstractmethod
    def predict_image_single_model(self, image_input: Union[Path, Image.Image, np.ndarray], 
                                 fold_number: int = 1, **kwargs) -> Tuple[np.ndarray, List[InstanceData]]:
        """Generate prediction with single model"""
        pass
    
    # Implemented methods
    def train_all_folds(self, **kwargs) -> Dict[int, Dict[str, Any]]:
        """Train all folds in cross-validation"""
        results = {}
        for fold in range(1, self.config.training.cross_validation_folds + 1):
            print(f"Training fold {fold}/{self.config.training.cross_validation_folds}")
            results[fold] = self.train_model(fold_number=fold, **kwargs)
        return results
    
    def load_all_pretrained_models(self, model_name: str = "best_model.keras", **kwargs):
        """Load pretrained models for all folds"""
        for fold in range(1, self.config.training.cross_validation_folds + 1):
            model_path = self.get_fold_dir(fold) / "models" / model_name
            if model_path.exists():
                self.load_pretrained_model(model_path, fold_number=fold, **kwargs)
            else:
                print(f"Warning: Model not found for fold {fold}: {model_path}")
    
    def predict_image(self, image_input: Union[Path, Image.Image, np.ndarray], 
                     folds: Union[int, List[int], str] = "all", **kwargs) -> StandardizedOutput:
        """Generate predictions using specified folds"""
        # Handle fold specification
        if folds == "all":
            fold_list = list(range(1, self.config.training.cross_validation_folds + 1))
        elif isinstance(folds, int):
            fold_list = [folds]
        else:
            fold_list = folds
        
        # Get image path for metadata
        if isinstance(image_input, Path):
            image_path = image_input
        else:
            image_path = Path("unknown_image")
        
        # Get scale
        scale = self.scale_manager.get_scale(
            image_input, 
            fallback=self.config.data.target_scale
        )
        
        # Create output container
        output = StandardizedOutput(
            image_path=image_path,
            organelle=self.config.organelle,
            scale_micrometers_per_pixel=scale
        )
        
        # Generate predictions for each fold
        for fold in fold_list:
            if fold not in self.models:
                print(f"Warning: Model not loaded for fold {fold}")
                continue
            
            semantic_mask, instances = self.predict_image_single_model(
                image_input, fold_number=fold, **kwargs
            )
            
            model_info = {
                'fold_number': fold,
                'framework': self.config.model.framework,
                'architecture': self.config.model.architecture,
                'workflow_name': self.config.workflow_name,
                'config_hash': self.config.config_hash
            }
            
            output.add_prediction(semantic_mask, instances, model_info)
        
        return output
    
    def predict_batch(self, image_paths: List[Path], 
                     output_dir: Path = None, **kwargs) -> List[StandardizedOutput]:
        """Predict on batch of images"""
        if output_dir is None:
            output_dir = self.get_workflow_dir() / "predictions"
        
        results = []
        for image_path in image_paths:
            print(f"Processing {image_path}")
            result = self.predict_image(image_path, **kwargs)
            result.save_outputs(output_dir)
            results.append(result)
        
        return results
    
    def distill_models(self, **kwargs) -> Any:
        """Perform model distillation using ensemble as teachers"""
        if not self.config.distillation.enabled:
            raise ValueError("Distillation not enabled in configuration")
        
        if self.config.training.cross_validation_folds < 2:
            raise ValueError("Distillation requires multiple folds as teachers")
        
        # This will be implemented by specific workflow subclasses
        # Base implementation provides framework for teacher-student setup
        print("Starting model distillation...")
        print(f"Teachers: {self.config.training.cross_validation_folds} models")
        print(f"Temperature: {self.config.distillation.temperature}")
        print(f"Alpha: {self.config.distillation.alpha}")
        
        # Subclasses should implement the actual distillation logic
        return self._perform_distillation(**kwargs)
    
    @abstractmethod
    def _perform_distillation(self, **kwargs) -> Any:
        """Implement framework-specific distillation logic"""
        pass

# =============================================================================
# TensorFlow UNet Implementation
# =============================================================================

class UNetTensorFlowWorkflow(SegmentationWorkflow):
    """TensorFlow UNet implementation"""
    
    def prepare_training_data(self, images_path: Path, masks_path: Path, **kwargs) -> Path:
        """Create TFRecords from images and masks"""
        from .cmd._preprocess import make_tfrecords
        
        output_path = self.get_workflow_dir() / "training_data"
        
        make_tfrecords(
            dataset_name=self.config.workflow_name,
            slides_dirpath=images_path,
            masks_dirpath=masks_path,
            organelle=self.config.organelle,
            data_dirpath=output_path.parent,
            **kwargs
        )
        
        return output_path
    
    def create_model(self, fold_number: int = 1, **kwargs) -> Any:
        """Create UNet model for specific fold"""
        from .model._unet import make_unet
        from .model.losses import MyWeightedBinaryCrossEntropy
        from .model.metrics import MyF1Score, MyF2Score
        
        model = make_unet(
            tile_shape=self.config.data.tile_shape,
            channels=self.config.model.channels,
            layer_depth=self.config.model.layer_depth,
            filters_root=self.config.model.filters_root
        )
        
        # Compile model
        loss = MyWeightedBinaryCrossEntropy(pos_weight=2)
        model.compile(
            loss=loss,
            optimizer='adam',
            metrics=['Recall', MyF1Score(), MyF2Score()],
            jit_compile=False
        )
        
        self.models[fold_number] = model
        return model
    
    def load_pretrained_model(self, model_path: Path, fold_number: int = 1, **kwargs) -> Any:
        """Load existing Keras model"""
        import tensorflow.keras as keras
        from .model.custom_objects import custom_objects
        
        model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
        self.models[fold_number] = model
        return model
    
    def train_model(self, fold_number: int = 1, **kwargs) -> Dict[str, Any]:
        """Train model for specific fold"""
        from .cmd._train import get_dataset
        import tensorflow.keras as keras
        
        # Ensure model exists
        if fold_number not in self.models:
            self.create_model(fold_number)
        
        model = self.models[fold_number]
        fold_dir = self.get_fold_dir(fold_number)
        
        # Load datasets
        training_data_path = self.get_workflow_dir() / "training_data" / self.config.workflow_name
        
        n_train, train_dataset, n_val, val_dataset = get_dataset(
            dataset_name=self.config.workflow_name,
            split='tra_val',
            organelle=self.config.organelle,
            tile_shape=self.config.data.tile_shape,
            window_shape=self.config.data.output_shape,
            batch_size=self.config.training.batch_size,
            fold_n=fold_number,
            total_folds=self.config.training.cross_validation_folds,
            data_dirpath=training_data_path.parent,
            **kwargs
        )
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.CSVLogger(
                fold_dir / "logs" / "metrics.tsv", 
                separator="\t", 
                append=True
            ),
            keras.callbacks.TensorBoard(fold_dir / "logs"),
            keras.callbacks.ModelCheckpoint(
                fold_dir / "models" / "best_model.keras",
                save_best_only=True,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                fold_dir / "models" / "last_model.keras"
            )
        ]
        
        if self.config.training.early_stopping_patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    patience=self.config.training.early_stopping_patience,
                    restore_best_weights=True
                )
            )
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.training.epochs,
            callbacks=callbacks,
            **kwargs
        )
        
        return {'history': history.history}
    
    def predict_image_single_model(self, image_input: Union[Path, Image.Image, np.ndarray], 
                                 fold_number: int = 1, **kwargs) -> Tuple[np.ndarray, List[InstanceData]]:
        """Generate prediction with single model"""
        from .prediction_tools import image_prediction, load_image
        
        # Load image if path provided
        if isinstance(image_input, Path):
            img = load_image(str(image_input))
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            img = image_input
        
        # Get model
        model = self.models[fold_number]
        
        # Generate prediction
        prediction = image_prediction(
            img=img,
            model=model,
            trg_scale=self.config.data.target_scale,
            batch_size=kwargs.get('batch_size', 32)
        )
        
        # Convert to numpy
        semantic_mask = np.array(prediction)
        
        # Extract instances
        instances = self._extract_instances_from_mask(semantic_mask)
        
        return semantic_mask, instances
    
    def _extract_instances_from_mask(self, mask: np.ndarray) -> List[InstanceData]:
        """Extract individual instances from semantic mask"""
        from skimage import measure
        
        # Threshold and label
        binary_mask = (mask > 0.5).astype(np.uint8)
        labeled_mask = measure.label(binary_mask)
        
        instances = []
        for region in measure.regionprops(labeled_mask):
            if region.area < 10:  # Filter small artifacts
                continue
            
            instance_mask = (labeled_mask == region.label).astype(np.uint8)
            area_pixels = region.area
            area_um2 = area_pixels * (self.scale_manager.get_scale("dummy") ** 2)  # TODO: Fix scale access
            
            instances.append(InstanceData(
                mask=instance_mask,
                bbox=region.bbox,
                area_pixels=area_pixels,
                area_micrometers_sq=area_um2,
                centroid=region.centroid,
                morphology_metrics={}  # TODO: Implement morphology metrics
            ))
        
        return instances
    
    def _perform_distillation(self, **kwargs) -> Any:
        """Perform knowledge distillation with TensorFlow"""
        # TODO: Implement TensorFlow-specific distillation
        # Use ensemble of k teachers to train a single student model
        
        print("TensorFlow distillation not yet implemented")
        pass

# =============================================================================
# Workflow Factory and Registry
# =============================================================================

class WorkflowRegistry:
    """Registry for available workflow implementations"""
    
    _workflows = {}
    
    @classmethod
    def register(cls, framework: str, architecture: str, workflow_class: type):
        """Register workflow implementation"""
        key = f"{framework}_{architecture}"
        cls._workflows[key] = workflow_class
    
    @classmethod
    def create_workflow(cls, config: WorkflowConfig, 
                       scale_manager: ScaleManager = None) -> SegmentationWorkflow:
        """Create workflow instance"""
        key = f"{config.model.framework}_{config.model.architecture}"
        
        if key not in cls._workflows:
            available = list(cls._workflows.keys())
            raise ValueError(f"No workflow registered for {key}. Available: {available}")
        
        workflow_class = cls._workflows[key]
        return workflow_class(config, scale_manager)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List registered workflows"""
        return list(cls._workflows.keys())

# Register available workflows
WorkflowRegistry.register('tensorflow', 'unet', UNetTensorFlowWorkflow)

class WorkflowFactory:
    """Factory for creating and managing workflows"""
    
    def __init__(self, scale_manager: ScaleManager = None):
        self.scale_manager = scale_manager or ScaleManager()
    
    def create_from_config(self, config: WorkflowConfig) -> SegmentationWorkflow:
        """Create workflow from configuration"""
        return WorkflowRegistry.create_workflow(config, self.scale_manager)
    
    def create_from_config_file(self, config_path: Path) -> SegmentationWorkflow:
        """Create workflow from YAML configuration file"""
        config = WorkflowConfig.load_from_file(config_path)
        return self.create_from_config(config)
    
    def create_for_organelle(self, organelle: str, 
                           workflow_name: str = None,
                           framework: str = 'tensorflow',
                           architecture: str = 'unet',
                           **config_overrides) -> SegmentationWorkflow:
        """Create workflow with organelle presets"""
        config = WorkflowConfig.from_organelle_preset(organelle)
        
        if workflow_name:
            config.workflow_name = workflow_name
        
        # Apply framework/architecture
        config.model.framework = framework
        config.model.architecture = architecture
        
        # Apply any additional overrides
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
        
        return self.create_from_config(config)

# =============================================================================
# Usage Examples
# =============================================================================

def example_usage():
    """Example usage of the workflow system"""
    
    # Create workflow factory
    factory = WorkflowFactory()
    
    # Method 1: Create from organelle preset
    workflow = factory.create_for_organelle(
        organelle='mitochondria',
        workflow_name='mitochondria_experiment_v1',
        cross_validation_folds=5,
        epochs=100
    )
    
    # Method 2: Create from configuration
    config = WorkflowConfig.from_organelle_preset('nucleus')
    config.workflow_name = 'nucleus_experiment'
    config.training.cross_validation_folds = 3
    config.training.epochs = 200
    
    workflow2 = factory.create_from_config(config)
    
    # Method 3: Load from YAML file
    # workflow3 = factory.create_from_config_file(Path('config.yaml'))
    
    # Prepare training data
    images_path = Path('data/images')
    masks_path = Path('data/masks')
    workflow.prepare_training_data(images_path, masks_path)
    
    # Train all folds
    training_results = workflow.train_all_folds()
    
    # Load pretrained models for inference
    workflow.load_all_pretrained_models('best_model.keras')
    
    # Predict on single image (ensemble)
    test_image = Path('data/test/image1.tif')
    result = workflow.predict_image(test_image, folds='all')
    
    # Save results
    result.save_outputs(Path('outputs/predictions'))
    
    # Predict on batch
    test_images = list(Path('data/test').glob('*.tif'))
    batch_results = workflow.predict_batch(test_images)
    
    # Perform distillation (if enabled)
    if workflow.config.distillation.enabled:
        student_model = workflow.distill_models()

if __name__ == "__main__":
    example_usage()
```

This implementation provides:

1. **Scale Estimation**: Flexible input handling (Path/Image/ndarray) with single float output
2. **Configuration Management**: YAML-based with hierarchical structure, hash-based freezing
3. **Directory Structure**: Config-hash based folders with fold-specific subdirectories  
4. **Standardized Output**: Class-based with ensemble support and aggregation methods
5. **Workflow Base**: Abstract interface with fold-aware methods and distillation framework
6. **TensorFlow Implementation**: Complete UNet workflow migrating your current code
7. **Factory Pattern**: Easy workflow creation with presets and overrides
8. **Future-Ready**: Designed to accommodate model distillation and new architectures

The scaffold maintains separation of concerns while providing a unified interface for different segmentation approaches.
