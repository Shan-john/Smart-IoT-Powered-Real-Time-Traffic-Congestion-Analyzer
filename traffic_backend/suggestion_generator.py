# AI-Powered Suggestion Generator for Traffic Congestion Reasons
# This module provides intelligent, context-specific suggestions for each congestion reason

from typing import List, Dict
import re

# Comprehensive suggestion database organized by category keywords
SUGGESTION_DATABASE: Dict[str, List[str]] = {
    # Traffic Flow Issues
    "heavy traffic": [
        "Use real-time navigation apps to find less congested routes",
        "Consider traveling during off-peak hours (before 7 AM or after 8 PM)",
        "Carpool to reduce the number of vehicles on the road"
    ],
    "bumper to bumper": [
        "Maintain safe following distance to prevent chain-reaction accidents",
        "Use cruise control where possible to maintain steady speed",
        "Avoid sudden lane changes which worsen congestion"
    ],
    "slow moving": [
        "Be patient and avoid aggressive driving behaviors",
        "Use the time to listen to audiobooks or podcasts",
        "Check if there's an incident ahead using traffic apps"
    ],
    "stop and go": [
        "Anticipate braking to reduce fuel consumption",
        "Keep engine in good condition for frequent stopping",
        "Consider hybrid or electric vehicles for better efficiency"
    ],
    "rush hour": [
        "Adjust work schedule if flexible working is available",
        "Use public transportation during peak hours",
        "Plan meetings to avoid peak travel times"
    ],
    "vehicles stopped": [
        "Turn off engine if stopped for more than 2 minutes",
        "Stay alert for emergency vehicles that may need passage",
        "Use parking brake on inclines to reduce fatigue"
    ],
    
    # Parking Issues
    "parking": [
        "Report illegal parking to traffic authorities",
        "Use designated parking areas only",
        "Look for parking apps to find available spots"
    ],
    "double parking": [
        "Report to traffic enforcement immediately",
        "Take photos for evidence if blocking your vehicle",
        "Use horn briefly to alert the driver if present"
    ],
    "delivery": [
        "Be patient with essential delivery services",
        "Advocate for designated loading zones in your area",
        "Shop during non-delivery peak hours"
    ],
    
    # Road Conditions
    "road blocked": [
        "Use GPS to find alternative routes immediately",
        "Report the blockage to local traffic authorities",
        "Share information with other drivers via apps"
    ],
    "construction": [
        "Follow temporary signs and speed limits carefully",
        "Be extra cautious around construction workers",
        "Check road work schedules before planning trips"
    ],
    "pothole": [
        "Report potholes to local maintenance department",
        "Reduce speed when approaching damaged road sections",
        "Avoid sudden swerving which can cause accidents"
    ],
    
    # Traffic Signals
    "signal": [
        "Report malfunctioning signals to traffic department",
        "Treat broken signals as 4-way stop intersections",
        "Exercise extra caution and make eye contact with other drivers"
    ],
    "traffic light": [
        "Wait patiently for signal changes",
        "Avoid running amber lights which cause accidents",
        "Report timing issues to improve flow"
    ],
    
    # Accidents & Incidents
    "accident": [
        "Give space to emergency responders",
        "Don't slow down excessively to observe (rubbernecking)",
        "If you witnessed it, provide information to authorities"
    ],
    "collision": [
        "Call emergency services if injuries are suspected",
        "Move to a safe location if possible and involved",
        "Document the scene for insurance purposes"
    ],
    "breakdown": [
        "Turn on hazard lights if you break down",
        "Move to the shoulder if safe to do so",
        "Call roadside assistance immediately"
    ],
    "fire": [
        "Evacuate the area immediately if safe",
        "Call emergency services (fire department)",
        "Do not attempt to fight vehicle fires yourself"
    ],
    
    # Weather Conditions
    "rain": [
        "Reduce speed by 20-30% in wet conditions",
        "Increase following distance to 4-5 seconds",
        "Turn on headlights for visibility"
    ],
    "fog": [
        "Use low-beam headlights, not high-beams",
        "Reduce speed significantly",
        "Use road markings as a guide"
    ],
    "snow": [
        "Fit winter tires or chains if required",
        "Accelerate and brake gently",
        "Keep emergency supplies in vehicle"
    ],
    "ice": [
        "Avoid sudden movements (steering, braking)",
        "Stay home if conditions are severe",
        "Allow extra time for all journeys"
    ],
    "flood": [
        "Never drive through standing water",
        "Turn around if road is flooded",
        "Know your vehicle's water fording depth"
    ],
    
    # Driver Behavior
    "braking": [
        "Maintain steady speed to reduce need for braking",
        "Keep eyes ahead to anticipate slowdowns",
        "Avoid tailgating to prevent sudden braking"
    ],
    "speeding": [
        "Follow posted speed limits at all times",
        "Remember speed fines can be expensive",
        "High speed increases accident severity"
    ],
    "aggressive": [
        "Stay calm and don't engage with aggressive drivers",
        "Give way to aggressive drivers for safety",
        "Report dangerous driving to authorities"
    ],
    
    # Special Events
    "event": [
        "Plan alternative routes before event days",
        "Use public transportation to event venues",
        "Leave early or wait until crowds disperse"
    ],
    "parade": [
        "Check event schedules before traveling",
        "Use designated pedestrian crossing points",
        "Be patient and enjoy the festivities"
    ],
    "stadium": [
        "Arrive early to avoid post-event rush",
        "Book parking in advance if available",
        "Consider rideshare or public transit"
    ],
    
    # Default/Generic
    "normal": [
        "Continue driving safely",
        "Stay alert for changing conditions",
        "Follow all traffic rules"
    ]
}

# Fallback suggestions for unmatched reasons
DEFAULT_SUGGESTIONS = [
    "Monitor the traffic situation and adjust route if needed",
    "Stay patient and avoid aggressive driving behaviors",
    "Use real-time navigation for alternative route suggestions"
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
