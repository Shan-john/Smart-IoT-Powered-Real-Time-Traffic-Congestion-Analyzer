# AI-Powered Suggestion Generator for Traffic Congestion Reasons
# This module provides intelligent, context-specific suggestions for each congestion reason

from typing import List, Dict
import re

# Comprehensive suggestion database organized by category keywords
# These are targeted remediation actions to REDUCE/ELIMINATE each specific congestion cause
SUGGESTION_DATABASE: Dict[str, List[str]] = {
    # Traffic Flow Issues
    "heavy traffic": [
        "Widen road or add extra lane to increase vehicle capacity",
        "Implement smart traffic signals to reduce waiting time at intersections",
        "Promote public transport usage to reduce number of private vehicles"
    ],
    "bumper to bumper": [
        "Build flyover or underpass to separate crossing traffic",
        "Introduce staggered office timings to distribute vehicle load",
        "Add alternative bypass route to divert through-traffic"
    ],
    "slow moving": [
        "Remove bottleneck causing the slowdown (parked vehicles, narrow section)",
        "Synchronize traffic signals to create continuous green wave flow",
        "Ban slow-moving heavy vehicles during peak hours"
    ],
    "high density": [
        "Create one-way traffic system to improve flow direction",
        "Restrict private vehicles and encourage public transport",
        "Develop parallel roads to distribute traffic density"
    ],
    "stop and go": [
        "Install adaptive signal control to reduce red light waiting",
        "Remove unnecessary traffic signals causing frequent stops",
        "Create dedicated lanes for through-traffic without stops"
    ],
    "rush hour": [
        "Enforce staggered work timings for offices in the area",
        "Increase bus/metro frequency to reduce private vehicle usage",
        "Implement work-from-home policies for nearby businesses"
    ],
    "vehicles stopped": [
        "Deploy tow trucks to immediately remove stopped vehicles",
        "Create emergency breakdown bays to clear main carriageway",
        "Install vehicle detection sensors for instant incident alerts"
    ],
    "intermittent": [
        "Identify and fix the cause of irregular traffic flow",
        "Install real-time monitoring to detect flow interruptions",
        "Deploy traffic marshals during peak hours for smooth flow"
    ],
    
    # Parking Issues
    "parking": [
        "Remove illegally parked vehicles blocking traffic flow",
        "Build dedicated parking lot nearby to free up road space",
        "Enforce strict no-parking with heavy fines and towing"
    ],
    "double parking": [
        "Deploy tow trucks to immediately clear double-parked vehicles",
        "Install CCTV with auto-detection for instant penalty issuance",
        "Create loading zones to prevent parking on main road"
    ],
    "delivery": [
        "Restrict delivery vehicles to off-peak hours only",
        "Create designated delivery bays away from traffic lanes",
        "Switch to night delivery policy for commercial areas"
    ],
    "illegal parking": [
        "Tow away all illegally parked vehicles immediately",
        "Install bollards to physically prevent parking in no-parking zones",
        "Increase fine amount significantly to deter violations"
    ],
    
    # Road Conditions
    "road blocked": [
        "Clear the blockage immediately with emergency response team",
        "Set up diversion route to redirect traffic around blockage",
        "Remove the cause of blockage (debris, broken vehicle, obstruction)"
    ],
    "construction": [
        "Move construction work to night hours when traffic is low",
        "Fast-track completion with strict deadline enforcement",
        "Provide well-marked alternate route during construction"
    ],
    "pothole": [
        "Fill and repair the pothole within 24 hours",
        "Resurface the entire road section to prevent future damage",
        "Install proper drainage to prevent road surface deterioration"
    ],
    "road damage": [
        "Repair damaged road section immediately on priority basis",
        "Use quality materials to ensure long-lasting repair",
        "Address root cause (drainage, heavy vehicles) to prevent recurrence"
    ],
    "narrow": [
        "Widen the narrow section or acquire adjacent land for expansion",
        "Convert to one-way traffic to maximize narrow road capacity",
        "Restrict large vehicles from using narrow road"
    ],
    
    # Traffic Signals
    "signal": [
        "Fix or replace malfunctioning traffic signal immediately",
        "Optimize signal timing to reduce unnecessary waiting",
        "Install backup power to prevent signal failures"
    ],
    "traffic light": [
        "Adjust signal timing to match actual traffic flow patterns",
        "Implement intelligent signal that adapts to real-time traffic",
        "Synchronize signals for green wave on main corridor"
    ],
    "junction": [
        "Redesign junction for smoother traffic flow",
        "Install roundabout to eliminate signal-related congestion",
        "Add dedicated turn lanes to separate turning traffic"
    ],
    
    # Accidents & Incidents
    "accident": [
        "Clear accident site quickly and move vehicles to roadside",
        "Install safety barriers at this spot to prevent future accidents",
        "Deploy incident response team for faster clearance"
    ],
    "collision": [
        "Install speed cameras to reduce speeding at this location",
        "Add lane markings and signage to prevent lane confusion",
        "Install rumble strips to alert drivers of dangerous zone"
    ],
    "breakdown": [
        "Arrange quick towing service to clear broken-down vehicle",
        "Create breakdown lane for vehicles to pull over safely",
        "Set up roadside assistance patrol for fast response"
    ],
    
    # Weather Conditions
    "rain": [
        "Improve drainage system to prevent water logging on road",
        "Apply anti-skid treatment on road surface for wet conditions",
        "Install proper road markings visible in rain"
    ],
    "fog": [
        "Install fog lights and reflectors along the road",
        "Deploy warning signs for drivers during low visibility",
        "Enforce reduced speed limit during foggy conditions"
    ],
    "flood": [
        "Build elevated road section in flood-prone area",
        "Install flood barrier and proper drainage systems",
        "Create alternate route that bypasses flood-prone zone"
    ],
    "waterlogging": [
        "Clean and unclog drainage system immediately",
        "Install additional storm water drains",
        "Raise road level in waterlogging-prone sections"
    ],
    
    # Driver Behavior
    "speeding": [
        "Install speed cameras with automatic fine system",
        "Add speed breakers or chicanes to force speed reduction",
        "Increase police patrolling in speeding-prone areas"
    ],
    "aggressive": [
        "Increase visible traffic police presence",
        "Install surveillance cameras to catch traffic violations",
        "Launch strict enforcement drive against aggressive driving"
    ],
    "wrong way": [
        "Install clear one-way signage with no-entry boards",
        "Add physical barriers to prevent wrong-way entry",
        "Install wrong-way detection system with instant alerts"
    ],
    
    # Special Events
    "event": [
        "Create separate traffic management plan for event",
        "Arrange shuttle services to reduce private vehicles",
        "Deploy extra traffic police for event-day management"
    ],
    "school": [
        "Create separate drop-off zone away from main road",
        "Stagger school timings to reduce traffic peak",
        "Install pedestrian crossing with traffic signal"
    ],
    "market": [
        "Relocate market to area with better access and parking",
        "Pedestrianize market area and divert vehicles around it",
        "Schedule market hours to avoid peak traffic times"
    ],
    
    # Infrastructure Issues
    "bottleneck": [
        "Widen the bottleneck section to match road capacity",
        "Build bypass road to divert traffic around bottleneck",
        "Remove the physical constraint causing the bottleneck"
    ],
    
    # Default/Generic - for monitoring
    "normal": [
        "No immediate action required - traffic flowing normally",
        "Continue monitoring for any emerging issues",
        "Maintain road infrastructure in good condition"
    ]
}

# Fallback suggestions for unmatched reasons
DEFAULT_SUGGESTIONS = [
    "Investigate root cause of congestion through on-site assessment",
    "Deploy traffic monitoring to identify specific problem areas",
    "Coordinate with engineering team to design appropriate solution"
]


def get_suggestions_for_reason(reason: str) -> List[str]:
    """
    Generate AI-powered suggestions based on the congestion reason.
    Uses keyword matching to find relevant suggestions from the database.
    
    Args:
        reason: The congestion reason string
        
    Returns:
        List of 3 relevant suggestions
    """
    reason_lower = reason.lower()
    
    # Try to find matching keywords
    for keyword, suggestions in SUGGESTION_DATABASE.items():
        if keyword in reason_lower:
            return suggestions
    
    # Try partial matching for compound reasons
    for keyword, suggestions in SUGGESTION_DATABASE.items():
        # Split keyword and check if all parts are in reason
        keywords = keyword.split()
        if all(kw in reason_lower for kw in keywords):
            return suggestions
    
    # Category-based fallback
    if any(word in reason_lower for word in ['traffic', 'vehicle', 'car', 'congestion']):
        return SUGGESTION_DATABASE.get('heavy traffic', DEFAULT_SUGGESTIONS)
    
    if any(word in reason_lower for word in ['park', 'block', 'obstruct']):
        return SUGGESTION_DATABASE.get('parking', DEFAULT_SUGGESTIONS)
    
    if any(word in reason_lower for word in ['weather', 'wet', 'storm']):
        return SUGGESTION_DATABASE.get('rain', DEFAULT_SUGGESTIONS)
    
    if any(word in reason_lower for word in ['crash', 'incident', 'emergency']):
        return SUGGESTION_DATABASE.get('accident', DEFAULT_SUGGESTIONS)
    
    return DEFAULT_SUGGESTIONS


def get_all_suggestion_categories() -> List[str]:
    """Get all available suggestion categories."""
    return list(SUGGESTION_DATABASE.keys())


# Example usage
if __name__ == "__main__":
    test_reasons = [
        "heavy traffic volume",
        "bumper to bumper traffic",
        "vehicles moving slowly",
        "wrong parking blocking lane",
        "road blocked by construction",
        "traffic signal not working",
        "possible accident ahead",
        "heavy rain reducing visibility",
        "stop and go traffic pattern",
        "some unknown reason"
    ]
    
    print("AI Suggestion Generator Test\n" + "="*50)
    for reason in test_reasons:
        suggestions = get_suggestions_for_reason(reason)
        print(f"\n📍 {reason}")
        for i, s in enumerate(suggestions, 1):
            print(f"   {i}. {s}")
