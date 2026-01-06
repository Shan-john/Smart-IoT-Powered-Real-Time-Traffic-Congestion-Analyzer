"""
Congestion Analyzer - Advanced congestion detection and reason classification.

Combines:
- Optical flow for motion analysis
- Rule-based classification for common scenarios
- CLIP model with 500+ reason database for detailed classification
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from PIL import Image
from enum import Enum
import random


class CongestionLevel(Enum):
    """Traffic congestion severity levels."""
    NORMAL = "Normal"
    LIGHT = "Light Congestion"
    MODERATE = "Moderate Congestion"
    HEAVY = "Heavy Congestion"
    SEVERE = "Severe Congestion"


# Import reasons database
try:
    from reasons_database import CONGESTION_REASONS, ALL_REASONS, get_reasons_by_category
    REASONS_AVAILABLE = True
except ImportError:
    REASONS_AVAILABLE = False
    ALL_REASONS = [
        "normal traffic flow",
        "heavy traffic volume", 
        "vehicles moving slowly",
        "vehicles stopped",
        "wrong parking",
        "road blocked",
        "traffic signal not working",
        "possible accident",
        "frequent braking"
    ]


class OpticalFlowAnalyzer:
    """Optical flow analysis for motion detection."""
    
    def __init__(self):
        self.prev_gray = None
        self.flow_history: List[float] = []
        self.history_size = 10
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """
        Analyze frame motion using optical flow.
        
        Args:
            frame: BGR frame from video
            
        Returns:
            Dict with flow metrics
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        result = {
            'magnitude': 0.0,
            'avg_magnitude': 0.0,
            'motion_detected': False,
            'flow_direction': None,
            'flow_variance': 0.0
        }
        
        if self.prev_gray is not None:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Get average magnitude (motion intensity)
            avg_mag = np.mean(mag)
            max_mag = np.max(mag)
            
            # Update history
            self.flow_history.append(avg_mag)
            if len(self.flow_history) > self.history_size:
                self.flow_history.pop(0)
            
            # Calculate smoothed average
            smoothed_avg = np.mean(self.flow_history)
            
            # Dominant motion direction
            dominant_angle = np.mean(ang[mag > np.percentile(mag, 90)]) if np.any(mag > 0) else 0
            
            result = {
                'magnitude': float(avg_mag),
                'max_magnitude': float(max_mag),
                'avg_magnitude': float(smoothed_avg),
                'motion_detected': avg_mag > 0.5,
                'flow_direction': float(dominant_angle),
                'flow_variance': float(np.std(mag))
            }
        
        self.prev_gray = gray
        return result


class CongestionAnalyzer:
    """
    Main congestion analysis class combining multiple detection methods.
    Uses 500+ reasons database for detailed classification.
    """
    
    def __init__(self, use_clip: bool = True, use_detailed_reasons: bool = True):
        """
        Initialize analyzer.
        
        Args:
            use_clip: Whether to use CLIP model for visual classification
            use_detailed_reasons: Whether to use the 500+ reasons database
        """
        self.optical_flow = OpticalFlowAnalyzer()
        self.use_clip = use_clip
        self.use_detailed_reasons = use_detailed_reasons and REASONS_AVAILABLE
        self.clip_model = None
        self.clip_processor = None
        
        # Thresholds (tunable)
        self.thresholds = {
            'stuck_ratio_severe': 0.7,
            'stuck_ratio_heavy': 0.5,
            'stuck_ratio_moderate': 0.3,
            'speed_slow': 10.0,
            'speed_very_slow': 3.0,
            'density_high': 0.05,
            'density_moderate': 0.03,
            'flow_low': 1.0,
        }
        
        # Category mapping based on metrics
        self.category_map = {
            'signal_issue': 'signal_infrastructure',
            'road_blocked': 'accidents_incidents',
            'heavy_traffic': 'vehicle_flow',
            'stopped_traffic': 'vehicle_flow',
            'frequent_braking': 'driver_behavior',
            'slow_moving': 'vehicle_flow',
            'normal': 'vehicle_flow',
            'parking_issue': 'parking_issues',
            'accident': 'accidents_incidents',
            'weather': 'weather_conditions',
            'construction': 'construction_roadwork',
        }
        
        # Load CLIP lazily if needed
        if use_clip:
            self._load_clip()
    
    def _load_clip(self):
        """Lazily load CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("[CongestionAnalyzer] CLIP model loaded successfully")
        except Exception as e:
            print(f"[CongestionAnalyzer] CLIP loading failed: {e}")
            self.use_clip = False
    
    def analyze(self, frame: np.ndarray, traffic_metrics: Dict) -> Dict:
        """
        Analyze frame for congestion.
        
        Args:
            frame: BGR frame from video
            traffic_metrics: Metrics from EnhancedTracker.get_traffic_metrics()
            
        Returns:
            Dict with congestion analysis results
        """
        # Get optical flow metrics
        flow_metrics = self.optical_flow.analyze(frame)
        
        # Extract traffic metrics
        vehicle_count = traffic_metrics.get('vehicle_count', 0)
        avg_speed = traffic_metrics.get('average_speed', 0)
        stuck_ratio = traffic_metrics.get('stuck_ratio', 0)
        density = traffic_metrics.get('density', 0)
        stuck_count = traffic_metrics.get('stuck_count', 0)
        
        # Determine congestion level
        level = self._determine_level(stuck_ratio, avg_speed, density, flow_metrics)
        
        # Determine base category using rules
        base_category = self._determine_category(
            stuck_ratio, avg_speed, density, 
            flow_metrics, vehicle_count, stuck_count
        )
        
        # Get detailed reason
        if self.use_clip and level not in [CongestionLevel.NORMAL, CongestionLevel.LIGHT]:
            reason = self._get_detailed_reason_with_clip(frame, base_category)
        else:
            reason = self._get_reason_from_category(base_category)
        
        # Build result
        is_congested = level not in [CongestionLevel.NORMAL, CongestionLevel.LIGHT]
        
        return {
            'is_congested': is_congested,
            'level': level.value,
            'reason': reason,
            'category': base_category,
            'confidence': self._calculate_confidence(stuck_ratio, avg_speed, flow_metrics),
            'metrics': {
                'vehicle_count': vehicle_count,
                'average_speed': avg_speed,
                'stuck_ratio': stuck_ratio,
                'stuck_count': stuck_count,
                'density': density,
                'optical_flow': flow_metrics['avg_magnitude']
            }
        }
    
    def _determine_level(self, stuck_ratio: float, avg_speed: float, 
                         density: float, flow_metrics: Dict) -> CongestionLevel:
        """Determine congestion level from metrics."""
        flow_mag = flow_metrics.get('avg_magnitude', 0)
        
        if stuck_ratio >= self.thresholds['stuck_ratio_severe'] and flow_mag < self.thresholds['flow_low']:
            return CongestionLevel.SEVERE
        
        if stuck_ratio >= self.thresholds['stuck_ratio_heavy']:
            return CongestionLevel.HEAVY
        
        if stuck_ratio >= self.thresholds['stuck_ratio_moderate'] or avg_speed < self.thresholds['speed_slow']:
            return CongestionLevel.MODERATE
        
        if stuck_ratio > 0.1 or avg_speed < self.thresholds['speed_slow'] * 2:
            return CongestionLevel.LIGHT
        
        return CongestionLevel.NORMAL
    
    def _determine_category(self, stuck_ratio: float, avg_speed: float, density: float,
                            flow_metrics: Dict, vehicle_count: int, stuck_count: int) -> str:
        """Determine the likely category of congestion."""
        flow_mag = flow_metrics.get('avg_magnitude', 0)
        flow_variance = flow_metrics.get('flow_variance', 0)
        
        # Almost all vehicles stopped
        if stuck_ratio >= 0.8 and flow_mag < 0.5:
            if flow_variance > 1.0:
                return 'signal_issue'
            return 'road_blocked'
        
        # Many vehicles stopped
        if stuck_ratio >= 0.6:
            if density > self.thresholds['density_high']:
                return 'heavy_traffic'
            return 'stopped_traffic'
        
        # Moderate stopping with high flow variance
        if stuck_ratio >= 0.3 and flow_variance > 2.0:
            return 'frequent_braking'
        
        # Slow but moving
        if avg_speed < self.thresholds['speed_very_slow'] and stuck_ratio < 0.5:
            return 'slow_moving'
        
        # High density
        if density > self.thresholds['density_moderate'] and vehicle_count > 5:
            return 'heavy_traffic'
        
        # Slow movement
        if avg_speed < self.thresholds['speed_slow']:
            return 'slow_moving'
        
        return 'normal'
    
    def _get_reason_from_category(self, category: str) -> str:
        """Get a reason string from the category."""
        if not self.use_detailed_reasons:
            # Fallback to basic reasons
            basic_reasons = {
                'signal_issue': 'traffic signal not working',
                'road_blocked': 'road blocked',
                'heavy_traffic': 'heavy traffic volume',
                'stopped_traffic': 'vehicles stopped',
                'frequent_braking': 'frequent braking',
                'slow_moving': 'vehicles moving slowly',
                'normal': 'normal traffic flow',
                'parking_issue': 'wrong parking',
                'accident': 'possible accident',
            }
            return basic_reasons.get(category, 'traffic congestion detected')
        
        # Get reasons from the category
        db_category = self.category_map.get(category, 'vehicle_flow')
        category_reasons = get_reasons_by_category(db_category)
        
        if category_reasons:
            # Return a contextually appropriate reason
            return random.choice(category_reasons[:10])  # Top 10 most common
        
        return 'traffic congestion detected'
    
    def _get_detailed_reason_with_clip(self, frame: np.ndarray, base_category: str) -> str:
        """Use CLIP to match frame with detailed reasons from database."""
        if not self.clip_model or not self.clip_processor:
            return self._get_reason_from_category(base_category)
        
        try:
            import torch
            
            # Convert frame to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            
            # Get candidate reasons from the relevant category
            db_category = self.category_map.get(base_category, 'vehicle_flow')
            
            if self.use_detailed_reasons:
                category_reasons = get_reasons_by_category(db_category)
                # Limit to top reasons for performance (CLIP can handle ~77 tokens per text)
                candidate_reasons = category_reasons[:30] if category_reasons else ALL_REASONS[:30]
            else:
                candidate_reasons = ALL_REASONS[:30]
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=candidate_reasons, 
                images=image_pil, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                best_idx = probs.argmax().item()
                confidence = probs[0][best_idx].item()
            
            # Return the matched reason if confidence is reasonable
            if confidence > 0.15:  # Lower threshold since we have many candidates
                return candidate_reasons[best_idx]
            
        except Exception as e:
            print(f"[CongestionAnalyzer] CLIP detailed analysis failed: {e}")
        
        return self._get_reason_from_category(base_category)
    
    def _calculate_confidence(self, stuck_ratio: float, avg_speed: float, 
                              flow_metrics: Dict) -> float:
        """Calculate confidence score for the analysis (0-1)."""
        flow_factor = 1.0 - min(flow_metrics.get('avg_magnitude', 0) / 5.0, 1.0)
        stuck_factor = stuck_ratio
        confidence = (stuck_factor * 0.6 + flow_factor * 0.4)
        return round(min(max(confidence, 0.0), 1.0), 2)
    
    def get_all_reasons(self) -> List[str]:
        """Get all available reasons from the database."""
        return ALL_REASONS
    
    def get_reason_count(self) -> int:
        """Get total number of available reasons."""
        return len(ALL_REASONS)


# Convenience function
def analyze_congestion(frame: np.ndarray, traffic_metrics: Dict, 
                       analyzer: Optional[CongestionAnalyzer] = None) -> Dict:
    """
    Convenience function to analyze congestion.
    
    Args:
        frame: BGR frame from video
        traffic_metrics: Metrics from EnhancedTracker
        analyzer: Optional pre-initialized analyzer
        
    Returns:
        Congestion analysis results
    """
    if analyzer is None:
        analyzer = CongestionAnalyzer(use_clip=False, use_detailed_reasons=True)
    return analyzer.analyze(frame, traffic_metrics)
