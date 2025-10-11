import json
import pickle
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import cv2

class DataLoader:
    """
    Data loader utility for MAITRI system
    Handles loading of models, datasets, and configuration files
    """
    
    def __init__(self, data_path: str = "data/", models_path: str = "models/"):
        self.logger = logging.getLogger(__name__)
        self.data_path = data_path
        self.models_path = models_path
        
        # Create directories if they don't exist
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(os.path.join(data_path, "audio"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "datasets"), exist_ok=True)
        
        self.logger.info("Data loader initialized")

    def load_model(self, model_name: str, model_type: str = "keras") -> Any:
        """
        Load a pre-trained model
        
        Args:
            model_name: Name of the model file
            model_type: Type of model ('keras', 'pkl', 'pytorch')
            
        Returns:
            Loaded model object
        """
        model_path = os.path.join(self.models_path, model_name)
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            return None
            
        try:
            if model_type == "keras":
                from tensorflow import keras
                model = keras.models.load_model(model_path)
                self.logger.info(f"Keras model loaded: {model_name}")
                return model
                
            elif model_type == "pkl":
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.logger.info(f"Pickle model loaded: {model_name}")
                return model
                
            elif model_type == "pytorch":
                import torch
                model = torch.load(model_path)
                self.logger.info(f"PyTorch model loaded: {model_name}")
                return model
                
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration file
        
        Args:
            config_name: Name of configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = os.path.join(self.data_path, "config", f"{config_name}.json")
        
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Config loaded: {config_name}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config {config_name}: {e}")
            return {}

    def load_emotion_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load emotion recognition dataset
        
        Args:
            dataset_name: Name of the dataset ('FER2013', 'RAVDESS', 'TESS', etc.)
            
        Returns:
            DataFrame with dataset or None if error
        """
        dataset_path = os.path.join(self.data_path, "datasets", f"{dataset_name}.csv")
        
        if not os.path.exists(dataset_path):
            self.logger.warning(f"Dataset not found: {dataset_path}")
            return None
            
        try:
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Dataset loaded: {dataset_name} with {len(df)} samples")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {e}")
            return None

    def load_audio_file(self, audio_path: str) -> Optional[tuple]:
        """
        Load audio file for processing
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate) or None if error
        """
        if not os.path.exists(audio_path):
            self.logger.warning(f"Audio file not found: {audio_path}")
            return None
            
        try:
            import librosa
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            self.logger.info(f"Audio file loaded: {audio_path} (duration: {len(audio_data)/sample_rate:.2f}s)")
            return audio_data, sample_rate
            
        except Exception as e:
            self.logger.error(f"Error loading audio file {audio_path}: {e}")
            return None

    def load_image(self, image_path: str, target_size: tuple = None) -> Optional[np.ndarray]:
        """
        Load image for processing
        
        Args:
            image_path: Path to image file
            target_size: Optional target size (width, height)
            
        Returns:
            Image array or None if error
        """
        if not os.path.exists(image_path):
            self.logger.warning(f"Image file not found: {image_path}")
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
                
            if target_size:
                image = cv2.resize(image, target_size)
                
            self.logger.info(f"Image loaded: {image_path} (shape: {image.shape})")
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None

    def load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if error
        """
        session_path = os.path.join(self.data_path, "sessions", f"{session_id}.json")
        
        if not os.path.exists(session_path):
            self.logger.warning(f"Session data not found: {session_path}")
            return None
            
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            self.logger.info(f"Session data loaded: {session_id}")
            return session_data
            
        except Exception as e:
            self.logger.error(f"Error loading session data {session_id}: {e}")
            return None

    def save_session_data(self, session_data: Dict[str, Any], session_id: str = None) -> bool:
        """
        Save session data
        
        Args:
            session_data: Data to save
            session_id: Session identifier (generated if None)
            
        Returns:
            True if successful, False otherwise
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        session_path = os.path.join(self.data_path, "sessions", f"{session_id}.json")
        
        try:
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Session data saved: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session data {session_id}: {e}")
            return False

    def load_knowledge_base(self, kb_name: str) -> Dict[str, Any]:
        """
        Load knowledge base for dialogue system
        
        Args:
            kb_name: Knowledge base name
            
        Returns:
            Knowledge base dictionary
        """
        kb_path = os.path.join(self.data_path, "knowledge_bases", f"{kb_name}.json")
        
        if not os.path.exists(kb_path):
            self.logger.warning(f"Knowledge base not found: {kb_path}")
            return self._get_default_knowledge_base()
            
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            self.logger.info(f"Knowledge base loaded: {kb_name}")
            return knowledge_base
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base {kb_name}: {e}")
            return self._get_default_knowledge_base()

    def _get_default_knowledge_base(self) -> Dict[str, Any]:
        """Get default knowledge base when file is not available"""
        return {
            "psychological_interventions": {
                "deep_breathing": "Practice 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s",
                "grounding": "Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste",
                "mindfulness": "Focus on present moment without judgment"
            },
            "space_facts": {
                "sleep": "Astronauts need 8 hours but often get less due to workload",
                "exercise": "2 hours daily prevents muscle and bone loss",
                "isolation": "Regular contact reduces feelings of isolation"
            }
        }

    def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load user profile
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile dictionary or None if error
        """
        profile_path = os.path.join(self.data_path, "profiles", f"{user_id}.json")
        
        if not os.path.exists(profile_path):
            self.logger.info(f"User profile not found, creating new: {user_id}")
            return self._create_default_user_profile(user_id)
            
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            self.logger.info(f"User profile loaded: {user_id}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error loading user profile {user_id}: {e}")
            return self._create_default_user_profile(user_id)

    def save_user_profile(self, profile_data: Dict[str, Any], user_id: str) -> bool:
        """
        Save user profile
        
        Args:
            profile_data: Profile data to save
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        profile_path = os.path.join(self.data_path, "profiles", f"{user_id}.json")
        
        try:
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"User profile saved: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving user profile {user_id}: {e}")
            return False

    def _create_default_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Create default user profile"""
        return {
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "preferences": {
                "intervention_types": ["psychological", "physical"],
                "max_intervention_duration": 10,  # minutes
                "communication_style": "supportive"
            },
            "history": {
                "sessions_count": 0,
                "total_interventions": 0,
                "preferred_interventions": []
            }
        }

    def load_emotional_vocabulary(self) -> Dict[str, List[str]]:
        """
        Load emotional vocabulary for text analysis
        
        Returns:
            Dictionary of emotion words and phrases
        """
        vocab_path = os.path.join(self.data_path, "vocabulary", "emotional_vocabulary.json")
        
        if not os.path.exists(vocab_path):
            self.logger.warning("Emotional vocabulary not found, using default")
            return self._get_default_emotional_vocabulary()
            
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)
            self.logger.info("Emotional vocabulary loaded")
            return vocabulary
            
        except Exception as e:
            self.logger.error(f"Error loading emotional vocabulary: {e}")
            return self._get_default_emotional_vocabulary()

    def _get_default_emotional_vocabulary(self) -> Dict[str, List[str]]:
        """Get default emotional vocabulary"""
        return {
            "stress": ["stress", "pressure", "overwhelm", "anxious", "worried", "nervous"],
            "fatigue": ["tired", "exhaust", "sleep", "fatigue", "drain", "burnout"],
            "sadness": ["sad", "depress", "unhappy", "down", "hopeless", "grief"],
            "anger": ["angry", "frustrat", "annoy", "irritat", "mad", "furious"],
            "fear": ["scared", "afraid", "fear", "terrified", "anxiety", "panic"],
            "happiness": ["happy", "joy", "excite", "good", "great", "wonderful"],
            "isolation": ["lonely", "alone", "isolat", "miss", "homesick", "separat"]
        }

    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List of model filenames
        """
        if not os.path.exists(self.models_path):
            return []
            
        models = []
        for file in os.listdir(self.models_path):
            if file.endswith(('.h5', '.pkl', '.pth', '.pt')):
                models.append(file)
                
        return models

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored data
        
        Returns:
            Data statistics dictionary
        """
        stats = {
            "total_size_bytes": 0,
            "file_counts": {},
            "last_updated": None
        }
        
        # Calculate total size and file counts
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                file_path = os.path.join(root, file)
                stats["total_size_bytes"] += os.path.getsize(file_path)
                
                # Count by file type
                file_ext = os.path.splitext(file)[1].lower()
                stats["file_counts"][file_ext] = stats["file_counts"].get(file_ext, 0) + 1
                
                # Get last update time
                mtime = os.path.getmtime(file_path)
                if stats["last_updated"] is None or mtime > stats["last_updated"]:
                    stats["last_updated"] = mtime
        
        # Convert last updated to readable format
        if stats["last_updated"]:
            stats["last_updated"] = datetime.fromtimestamp(stats["last_updated"]).isoformat()
            
        # Add human-readable size
        stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)
        
        return stats

    def cleanup_old_data(self, max_age_days: int = 30) -> int:
        """
        Clean up old data files
        
        Args:
            max_age_days: Maximum age of files in days
            
        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0
        
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip important files
                if any(important in file_path for important in ['models', 'config', 'profiles']):
                    continue
                    
                # Check file age
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        self.logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error cleaning up file {file_path}: {e}")
        
        self.logger.info(f"Cleanup completed: {deleted_count} files deleted")
        return deleted_count