"""
Distributed ML Pipeline - Feature Store Module

Handles online and offline feature serving with Redis caching.
"""

from typing import Dict, List, Any, Optional
import redis
import json
import hashlib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStore:
    """Online and offline feature store with Redis caching."""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        offline_store_path: str = "s3://features/offline",
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.offline_path = offline_store_path
        self.ttl = 3600  # 1 hour cache TTL
    
    def get_online(
        self,
        entity_ids: List[str],
        feature_set: str = "default",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get features for online inference.
        
        Args:
            entity_ids: List of entity IDs
            feature_set: Name of the feature set
        
        Returns:
            Dict mapping entity_id -> features
        """
        result = {}
        
        # Check Redis cache first
        for entity_id in entity_ids:
            key = self._make_key(feature_set, entity_id)
            cached = self.redis_client.get(key)
            
            if cached:
                result[entity_id] = json.loads(cached)
            else:
                # Fetch from offline store and cache
                features = self._fetch_from_offline(feature_set, entity_id)
                if features:
                    self._cache_feature(feature_set, entity_id, features)
                    result[entity_id] = features
                else:
                    result[entity_id] = {}
        
        return result
    
    def write_online(
        self,
        entity_id: str,
        features: Dict[str, Any],
        feature_set: str = "default",
    ):
        """
        Write features to online store.
        
        Args:
            entity_id: Entity identifier
            features: Feature dictionary
            feature_set: Feature set name
        """
        key = self._make_key(feature_set, entity_id)
        self.redis_client.setex(key, self.ttl, json.dumps(features))
        logger.info(f"Cached features for {entity_id}")
    
    def _make_key(self, feature_set: str, entity_id: str) -> str:
        """Generate Redis key."""
        return f"feature:{feature_set}:{entity_id}"
    
    def _fetch_from_offline(
        self,
        feature_set: str,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch features from offline store (S3/Parquet).
        
        In production, this would query:
        - Spark/Flink jobs
        - Delta Lake tables
        - BigQuery/Redshift
        
        For now, returns mock data.
        """
        # Mock implementation
        return {
            "entity_id": entity_id,
            "feature_set": feature_set,
            "created_at": datetime.utcnow().isoformat(),
            "values": {
                "feature_1": 0.5,
                "feature_2": 0.3,
                "feature_3": 0.8,
            },
        }
    
    def _cache_feature(
        self,
        feature_set: str,
        entity_id: str,
        features: Dict[str, Any],
    ):
        """Cache feature in Redis."""
        key = self._make_key(feature_set, entity_id)
        self.redis_client.setex(key, self.ttl, json.dumps(features))
    
    def register_feature_set(
        self,
        name: str,
        features: List[Dict[str, Any]],
        source: str,
        description: str = "",
    ):
        """
        Register a new feature set.
        
        Args:
            name: Feature set name
            features: List of feature definitions
            source: Data source (table, stream, etc.)
            description: Feature set description
        """
        metadata = {
            "name": name,
            "features": features,
            "source": source,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        key = f"feature_set:{name}"
        self.redis_client.set(key, json.dumps(metadata))
        logger.info(f"Registered feature set: {name}")
    
    def get_feature_set(self, name: str) -> Optional[Dict[str, Any]]:
        """Get feature set metadata."""
        key = f"feature_set:{name}"
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None


# Feature definition DSL
class Feature:
    """Feature definition."""
    
    def __init__(
        self,
        name: str,
        dtype: str,
        source: str,
        transformation: Optional[str] = None,
        description: str = "",
    ):
        self.name = name
        self.dtype = dtype
        self.source = source
        self.transformation = transformation
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "source": self.source,
            "transformation": self.transformation,
            "description": self.description,
        }


# Aggregation functions for windowed features
class Aggregations:
    """Common aggregation functions for features."""
    
    @staticmethod
    def count(events: List[Dict]) -> int:
        return len(events)
    
    @staticmethod
    def sum(field: str, events: List[Dict]) -> float:
        return sum(e.get(field, 0) for e in events)
    
    @staticmethod
    def avg(field: str, events: List[Dict]) -> float:
        values = [e.get(field, 0) for e in events]
        return sum(values) / len(values) if values else 0
    
    @staticmethod
    def min(field: str, events: List[Dict]) -> float:
        values = [e.get(field, 0) for e in events]
        return min(values) if values else 0
    
    @staticmethod
    def max(field: str, events: List[Dict]) -> float:
        values = [e.get(field, 0) for e in events]
        return max(values) if values else 0
    
    @staticmethod
    def distinct_count(field: str, events: List[Dict]) -> int:
        return len(set(e.get(field) for e in events if e.get(field)))
