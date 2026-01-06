import math
import time
from collections import deque
from typing import List, Tuple, Dict, Optional


class EnhancedTracker:
    """
    Enhanced vehicle tracker with speed, direction, and flow metrics.
    
    Improvements over SimpleTracker:
    - Tracks actual vehicle speeds (pixels/second)
    - Calculates traffic flow direction
    - Maintains position history for trajectory analysis
    - Provides comprehensive traffic metrics
    """
    
    def __init__(self, distance_threshold: int = 50, stuck_speed_threshold: float = 5.0, 
                 stuck_seconds: float = 3.0, history_size: int = 10):
        """
        Args:
            distance_threshold: Max pixels between frames to consider same vehicle
            stuck_speed_threshold: Speed (px/s) below which vehicle is "stuck"
            stuck_seconds: Seconds at low speed to consider stuck
            history_size: Number of positions to keep for trajectory analysis
        """
        self.tracked: Dict[int, Dict] = {}
        self.next_id = 0
        self.threshold = distance_threshold
        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_seconds = stuck_seconds
        self.history_size = history_size
        self.last_update_time = time.time()
        
        # Frame-level statistics
        self._frame_speeds: List[float] = []
        self._total_vehicles = 0
    
    def update(self, detections: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int]]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x, y) center points for detected vehicles
            
        Returns:
            List of (id, (x, y)) for tracked vehicles
        """
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        updated = []
        used_detections = set()
        
        # Match existing tracks to new detections
        for obj_id, data in list(self.tracked.items()):
            best_match = None
            best_dist = float('inf')
            
            for i, (x, y) in enumerate(detections):
                if i in used_detections:
                    continue
                    
                prev_x, prev_y = data['pos']
                dist = math.hypot(x - prev_x, y - prev_y)
                
                if dist < self.threshold and dist < best_dist:
                    best_match = i
                    best_dist = dist
            
            if best_match is not None:
                x, y = detections[best_match]
                used_detections.add(best_match)
                
                # Calculate speed
                prev_x, prev_y = data['pos']
                distance = math.hypot(x - prev_x, y - prev_y)
                speed = distance / dt if dt > 0 else 0
                
                # Calculate direction (angle in radians)
                direction = math.atan2(y - prev_y, x - prev_x)
                
                # Update history
                data['history'].append((x, y, now))
                if len(data['history']) > self.history_size:
                    data['history'].popleft()
                
                # Update speed history
                data['speed_history'].append(speed)
                if len(data['speed_history']) > self.history_size:
                    data['speed_history'].popleft()
                
                # Calculate average speed
                avg_speed = sum(data['speed_history']) / len(data['speed_history'])
                
                # Update stuck status
                if avg_speed < self.stuck_speed_threshold:
                    if data['slow_since'] is None:
                        data['slow_since'] = now
                else:
                    data['slow_since'] = None
                
                # Update data
                data['pos'] = (x, y)
                data['speed'] = speed
                data['avg_speed'] = avg_speed
                data['direction'] = direction
                data['last_seen'] = now
                
                updated.append((obj_id, (x, y)))
            
        # Create new tracks for unmatched detections
        for i, (x, y) in enumerate(detections):
            if i not in used_detections:
                self.tracked[self.next_id] = {
                    'pos': (x, y),
                    'last_seen': now,
                    'slow_since': None,
                    'speed': 0,
                    'avg_speed': 0,
                    'direction': 0,
                    'history': deque([(x, y, now)], maxlen=self.history_size),
                    'speed_history': deque([0], maxlen=self.history_size)
                }
                updated.append((self.next_id, (x, y)))
                self.next_id += 1
        
        # Remove stale tracks (not seen for 3 seconds)
        stale_ids = [i for i, d in self.tracked.items() if now - d['last_seen'] > 3.0]
        for i in stale_ids:
            del self.tracked[i]
        
        # Update frame statistics
        self._frame_speeds = [d['avg_speed'] for d in self.tracked.values()]
        self._total_vehicles = len(self.tracked)
        
        return updated
    
    def get_stuck_vehicles(self) -> List[Tuple[int, Tuple[int, int]]]:
        """Get vehicles that have been stuck (low speed) for stuck_seconds."""
        now = time.time()
        stuck = []
        for obj_id, data in self.tracked.items():
            if data['slow_since'] is not None:
                if now - data['slow_since'] >= self.stuck_seconds:
                    stuck.append((obj_id, data['pos']))
        return stuck
    
    def get_average_speed(self) -> float:
        """Get average speed across all tracked vehicles (pixels/second)."""
        if not self._frame_speeds:
            return 0.0
        return sum(self._frame_speeds) / len(self._frame_speeds)
    
    def get_max_speed(self) -> float:
        """Get maximum speed among tracked vehicles."""
        if not self._frame_speeds:
            return 0.0
        return max(self._frame_speeds)
    
    def get_min_speed(self) -> float:
        """Get minimum speed among tracked vehicles."""
        if not self._frame_speeds:
            return 0.0
        return min(self._frame_speeds)
    
    def get_vehicle_count(self) -> int:
        """Get current number of tracked vehicles."""
        return self._total_vehicles
    
    def get_stuck_ratio(self) -> float:
        """Get ratio of stuck vehicles to total vehicles (0.0 to 1.0)."""
        if self._total_vehicles == 0:
            return 0.0
        stuck_count = len(self.get_stuck_vehicles())
        return stuck_count / self._total_vehicles
    
    def get_vehicle_density(self, frame_width: int = 640, frame_height: int = 480) -> float:
        """
        Get vehicle density (vehicles per 10000 pixels).
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Density value (higher = more congested)
        """
        area = frame_width * frame_height
        return (self._total_vehicles / area) * 10000
    
    def get_traffic_metrics(self, frame_width: int = 640, frame_height: int = 480) -> Dict:
        """
        Get comprehensive traffic metrics in one call.
        
        Returns:
            Dictionary with all traffic metrics
        """
        stuck_vehicles = self.get_stuck_vehicles()
        return {
            'vehicle_count': self._total_vehicles,
            'average_speed': round(self.get_average_speed(), 2),
            'max_speed': round(self.get_max_speed(), 2),
            'min_speed': round(self.get_min_speed(), 2),
            'stuck_count': len(stuck_vehicles),
            'stuck_ratio': round(self.get_stuck_ratio(), 3),
            'density': round(self.get_vehicle_density(frame_width, frame_height), 4),
            'stuck_vehicles': stuck_vehicles
        }
    
    def get_flow_direction(self) -> Optional[float]:
        """
        Get dominant traffic flow direction (radians).
        
        Returns:
            Angle in radians, or None if no movement detected
        """
        if not self.tracked:
            return None
        
        # Weighted average of directions by speed
        weighted_x = 0
        weighted_y = 0
        total_weight = 0
        
        for data in self.tracked.values():
            if data['speed'] > 1:  # Ignore stationary vehicles
                weight = data['speed']
                weighted_x += math.cos(data['direction']) * weight
                weighted_y += math.sin(data['direction']) * weight
                total_weight += weight
        
        if total_weight == 0:
            return None
        
        return math.atan2(weighted_y, weighted_x)


# Backward compatibility alias
SimpleTracker = EnhancedTracker
