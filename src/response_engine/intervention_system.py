import logging
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum

class InterventionLevel(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRISIS = "crisis"

class InterventionType(Enum):
    PSYCHOLOGICAL = "psychological"
    PHYSICAL = "physical"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    CRISIS = "crisis"

class InterventionSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.intervention_history = []
        self.user_responses = {}
        
        # Evidence-based intervention database
        self.intervention_database = {
            InterventionType.PSYCHOLOGICAL: {
                InterventionLevel.MILD: [
                    {
                        "name": "Deep Breathing Exercise",
                        "description": "Practice 4-7-8 breathing technique to calm the nervous system",
                        "steps": [
                            "Inhale quietly through your nose for 4 seconds",
                            "Hold your breath for 7 seconds", 
                            "Exhale completely through your mouth for 8 seconds",
                            "Repeat this cycle 4 times"
                        ],
                        "duration": "2-3 minutes",
                        "effectiveness": 0.7
                    },
                    {
                        "name": "Mindful Observation",
                        "description": "Ground yourself in the present moment through observation",
                        "steps": [
                            "Identify 5 things you can see in your environment",
                            "Identify 4 things you can touch",
                            "Identify 3 things you can hear",
                            "Identify 2 things you can smell", 
                            "Identify 1 thing you can taste"
                        ],
                        "duration": "1-2 minutes",
                        "effectiveness": 0.6
                    }
                ],
                InterventionLevel.MODERATE: [
                    {
                        "name": "Progressive Muscle Relaxation",
                        "description": "Systematically tense and relax muscle groups to release physical tension",
                        "steps": [
                            "Start with your feet, tense muscles for 5 seconds",
                            "Release and notice the relaxation for 10 seconds",
                            "Move upward to calves, thighs, abdomen, etc.",
                            "Finish with facial muscles and scalp",
                            "Take deep breaths throughout the process"
                        ],
                        "duration": "5-10 minutes",
                        "effectiveness": 0.75
                    },
                    {
                        "name": "Cognitive Reframing",
                        "description": "Challenge and change negative thought patterns",
                        "steps": [
                            "Identify the stressful thought or situation",
                            "Examine the evidence for and against this thought",
                            "Consider alternative explanations or perspectives",
                            "Develop a more balanced, realistic thought",
                            "Practice this new perspective"
                        ],
                        "duration": "5-7 minutes", 
                        "effectiveness": 0.8
                    }
                ],
                InterventionLevel.SEVERE: [
                    {
                        "name": "Emergency Grounding Protocol",
                        "description": "Immediate grounding technique for severe distress",
                        "steps": [
                            "Sit or stand firmly, feeling your connection to the surface",
                            "Name objects around you aloud",
                            "Describe their colors, shapes, and textures",
                            "Recite your name, mission, and current location",
                            "Focus on your breathing pattern"
                        ],
                        "duration": "3-5 minutes",
                        "effectiveness": 0.85
                    }
                ],
                InterventionLevel.CRISIS: [
                    {
                        "name": "Crisis Stabilization Protocol",
                        "description": "Immediate crisis intervention with ground control alert",
                        "steps": [
                            "MAITRI ALERT: Activating emergency protocols",
                            "Ground control has been notified of your situation",
                            "Focus on maintaining regular breathing patterns",
                            "I am here with you - you are not alone",
                            "Help is available and support is on the way"
                        ],
                        "duration": "Until support arrives",
                        "effectiveness": 0.95,
                        "emergency_contact": True
                    }
                ]
            },
            
            InterventionType.PHYSICAL: {
                InterventionLevel.MILD: [
                    {
                        "name": "Microgravity Stretching",
                        "description": "Gentle stretches adapted for space environment",
                        "steps": [
                            "Slowly extend arms overhead, reaching for 10 seconds",
                            "Gently rotate shoulders forward and backward",
                            "Carefully stretch neck from side to side",
                            "Perform ankle rotations while secured",
                            "Finish with deep breathing"
                        ],
                        "duration": "3-5 minutes",
                        "effectiveness": 0.65
                    }
                ],
                InterventionLevel.MODERATE: [
                    {
                        "name": "Space-Adapted Yoga",
                        "description": "Modified yoga poses for microgravity",
                        "steps": [
                            "Secure yourself in a stable position",
                            "Perform modified child's pose using handholds",
                            "Gentle spinal twists while anchored",
                            "Modified warrior poses with support",
                            "Finish with floating savasana using restraints"
                        ],
                        "duration": "7-10 minutes",
                        "effectiveness": 0.7
                    }
                ]
            },
            
            InterventionType.ENVIRONMENTAL: {
                InterventionLevel.MILD: [
                    {
                        "name": "Lighting Adjustment",
                        "description": "Optimize lighting for mood and circadian rhythm",
                        "steps": [
                            "Increasing blue-light exposure for alertness",
                            "Adjusting brightness to comfortable levels",
                            "Setting lighting to match mission schedule",
                            "Creating visual comfort in your workspace"
                        ],
                        "duration": "Immediate",
                        "effectiveness": 0.6
                    }
                ],
                InterventionLevel.MODERATE: [
                    {
                        "name": "Environmental Enrichment",
                        "description": "Enhance living space for psychological comfort",
                        "steps": [
                            "Displaying Earth images or personal photos",
                            "Playing ambient space sounds or favorite music",
                            "Adjusting temperature to optimal comfort",
                            "Organizing workspace for efficiency and calm"
                        ],
                        "duration": "5-10 minutes",
                        "effectiveness": 0.65
                    }
                ]
            },
            
            InterventionType.SOCIAL: {
                InterventionLevel.MILD: [
                    {
                        "name": "Virtual Social Connection",
                        "description": "Facilitate connection with support network",
                        "steps": [
                            "Scheduling video call with family/friends",
                            "Preparing message for next data transmission",
                            "Viewing crew photos or mission updates",
                            "Participating in virtual team activities"
                        ],
                        "duration": "Varies",
                        "effectiveness": 0.7
                    }
                ]
            }
        }
        
        self.logger.info("Intervention system initialized")

    def assess_situation(self, emotional_state: Dict, physical_state: Dict = None, 
                        text_analysis: Dict = None) -> Tuple[InterventionLevel, List[InterventionType]]:
        """Assess situation and determine appropriate intervention level and types"""
        
        # Extract key metrics
        dominant_emotion, confidence = self._get_dominant_emotion(emotional_state)
        wellbeing_score = emotional_state.get('wellbeing_score', 50)
        risk_level = emotional_state.get('risk_level', 'normal')
        
        # Determine intervention level
        if risk_level == 'high' or wellbeing_score < 30:
            intervention_level = InterventionLevel.CRISIS
        elif risk_level == 'medium' or wellbeing_score < 50:
            intervention_level = InterventionLevel.SEVERE
        elif risk_level == 'low' or wellbeing_score < 70:
            intervention_level = InterventionLevel.MODERATE
        else:
            intervention_level = InterventionLevel.MILD
        
        # Determine intervention types needed
        intervention_types = [InterventionType.PSYCHOLOGICAL]
        
        # Add physical interventions if physical issues detected
        if physical_state:
            if physical_state.get('fatigue') == 'high' or physical_state.get('posture') == 'poor':
                intervention_types.append(InterventionType.PHYSICAL)
        
        # Add environmental interventions for stress or isolation
        if dominant_emotion in ['sad', 'fear'] or wellbeing_score < 60:
            intervention_types.append(InterventionType.ENVIRONMENTAL)
        
        # Add social interventions for isolation indicators
        if text_analysis and text_analysis.get('isolation_mentioned', False):
            intervention_types.append(InterventionType.SOCIAL)
        
        return intervention_level, intervention_types

    def generate_intervention(self, emotional_state: Dict, physical_state: Dict = None,
                            text_analysis: Dict = None, user_preferences: Dict = None) -> Dict:
        """Generate appropriate intervention based on assessment"""
        
        intervention_level, intervention_types = self.assess_situation(
            emotional_state, physical_state, text_analysis
        )
        
        # Select interventions from database
        selected_interventions = []
        
        for intervention_type in intervention_types:
            if (intervention_type in self.intervention_database and 
                intervention_level in self.intervention_database[intervention_type]):
                
                available_interventions = self.intervention_database[intervention_type][intervention_level]
                
                # Filter based on user preferences if available
                filtered_interventions = self._filter_by_preferences(
                    available_interventions, user_preferences
                )
                
                if filtered_interventions:
                    # Select most effective intervention
                    selected = max(filtered_interventions, key=lambda x: x['effectiveness'])
                    selected_interventions.append(selected)
        
        # Create intervention plan
        intervention_plan = {
            'level': intervention_level.value,
            'types': [t.value for t in intervention_types],
            'interventions': selected_interventions,
            'timestamp': datetime.now().isoformat(),
            'situation_assessment': {
                'dominant_emotion': self._get_dominant_emotion(emotional_state)[0],
                'wellbeing_score': emotional_state.get('wellbeing_score', 0),
                'risk_level': emotional_state.get('risk_level', 'normal')
            }
        }
        
        # Log intervention
        self._log_intervention(intervention_plan)
        
        return intervention_plan

    def _filter_by_preferences(self, interventions: List[Dict], preferences: Dict) -> List[Dict]:
        """Filter interventions based on user preferences"""
        if not preferences:
            return interventions
        
        filtered = []
        for intervention in interventions:
            # Check duration preferences
            if 'max_duration' in preferences:
                intervention_duration = self._parse_duration(intervention.get('duration', '5 minutes'))
                if intervention_duration > preferences['max_duration']:
                    continue
            
            # Check type preferences
            if 'preferred_types' in preferences:
                intervention_name = intervention.get('name', '').lower()
                type_match = any(pref.lower() in intervention_name 
                               for pref in preferences['preferred_types'])
                if not type_match and preferences.get('strict_matching', False):
                    continue
            
            filtered.append(intervention)
        
        return filtered if filtered else interventions

    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to minutes"""
        try:
            if 'minute' in duration_str:
                return int(duration_str.split()[0])
            elif 'hour' in duration_str:
                return int(duration_str.split()[0]) * 60
            else:
                return 5  # Default
        except:
            return 5

    def format_intervention_for_delivery(self, intervention_plan: Dict, 
                                       delivery_mode: str = "voice") -> str:
        """Format intervention for delivery to user"""
        if not intervention_plan['interventions']:
            return "I'm here to support you. How can I help you right now?"
        
        intervention = intervention_plan['interventions'][0]  # Use first intervention
        
        if delivery_mode == "voice":
            return self._format_for_voice(intervention, intervention_plan)
        else:
            return self._format_for_text(intervention, intervention_plan)

    def _format_for_voice(self, intervention: Dict, plan: Dict) -> str:
        """Format intervention for voice delivery"""
        level = plan['level'].upper()
        
        if level == "CRISIS":
            message = f"🚨 CRISIS INTERVENTION ACTIVATED. {intervention['description']}. "
        else:
            message = f"I have a {intervention['name']} suggestion for you. {intervention['description']}. "
        
        # Add steps
        message += "Let me guide you through this. "
        for i, step in enumerate(intervention['steps'], 1):
            message += f"Step {i}: {step}. "
        
        message += f"This should take about {intervention['duration']}. Ready to begin?"
        
        return message

    def _format_for_text(self, intervention: Dict, plan: Dict) -> str:
        """Format intervention for text delivery"""
        level_indicator = {
            "mild": "💡 Suggestion",
            "moderate": "📋 Recommendation", 
            "severe": "⚠️ Important Intervention",
            "crisis": "🚨 CRISIS PROTOCOL"
        }
        
        header = level_indicator.get(plan['level'], "📋 Recommendation")
        
        message = f"{header}: {intervention['name']}\n\n"
        message += f"{intervention['description']}\n\n"
        message += "Steps:\n"
        
        for i, step in enumerate(intervention['steps'], 1):
            message += f"{i}. {step}\n"
        
        message += f"\nDuration: {intervention['duration']}\n"
        message += f"Effectiveness: {intervention['effectiveness']*100:.0f}% based on research"
        
        return message

    def track_intervention_effectiveness(self, intervention_id: str, user_feedback: Dict):
        """Track effectiveness of delivered interventions"""
        # Find intervention in history
        for intervention in self.intervention_history:
            if intervention.get('intervention_id') == intervention_id:
                intervention['user_feedback'] = user_feedback
                intervention['feedback_timestamp'] = datetime.now().isoformat()
                
                # Calculate effectiveness score
                effectiveness = self._calculate_effectiveness_score(user_feedback)
                intervention['measured_effectiveness'] = effectiveness
                
                self.logger.info(f"Intervention {intervention_id} effectiveness: {effectiveness}")
                break

    def _calculate_effectiveness_score(self, feedback: Dict) -> float:
        """Calculate effectiveness score from user feedback"""
        score = 0.5  # Default neutral score
        
        if 'mood_improvement' in feedback:
            score += feedback['mood_improvement'] * 0.3
        
        if 'stress_reduction' in feedback:
            score += feedback['stress_reduction'] * 0.3
        
        if 'willingness_repeat' in feedback:
            score += feedback['willingness_repeat'] * 0.2
        
        if 'overall_satisfaction' in feedback:
            score += feedback['overall_satisfaction'] * 0.2
        
        return max(0.0, min(1.0, score))

    def get_personalized_recommendations(self, user_id: str) -> List[Dict]:
        """Get personalized intervention recommendations based on history"""
        user_history = [i for i in self.intervention_history 
                       if i.get('user_id') == user_id]
        
        if not user_history:
            return []
        
        # Analyze successful interventions
        successful_interventions = []
        for intervention in user_history[-10:]:  # Last 10 interventions
            effectiveness = intervention.get('measured_effectiveness', 0)
            if effectiveness > 0.7:  # Successful interventions
                successful_interventions.append(intervention)
        
        # Group by emotion and situation
        recommendations = []
        for intervention in successful_interventions:
            emotion = intervention.get('situation_assessment', {}).get('dominant_emotion')
            if emotion:
                recommendations.append({
                    'emotion': emotion,
                    'intervention': intervention['interventions'][0] if intervention['interventions'] else {},
                    'success_rate': intervention.get('measured_effectiveness', 0)
                })
        
        return recommendations

    def _get_dominant_emotion(self, emotional_state: Dict) -> Tuple[str, float]:
        """Get dominant emotion and confidence"""
        if not emotional_state or 'fused_emotions' not in emotional_state:
            return "neutral", 1.0
        
        emotions = emotional_state['fused_emotions']
        if not emotions:
            return "neutral", 1.0
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        return dominant_emotion

    def _log_intervention(self, intervention_plan: Dict):
        """Log intervention for monitoring and analysis"""
        intervention_entry = {
            'intervention_id': f"intv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            'plan': intervention_plan,
            'timestamp': datetime.now().isoformat()
        }
        
        self.intervention_history.append(intervention_entry)
        
        # Keep only last 100 interventions
        if len(self.intervention_history) > 100:
            self.intervention_history = self.intervention_history[-100:]
        
        self.logger.info(f"Intervention generated: {intervention_plan['level']} level")

    def get_intervention_statistics(self) -> Dict:
        """Get statistics about interventions delivered"""
        if not self.intervention_history:
            return {"total_interventions": 0}
        
        recent_interventions = self.intervention_history[-30:]  # Last 30 days equivalent
        
        stats = {
            "total_interventions": len(self.intervention_history),
            "recent_interventions": len(recent_interventions),
            "level_distribution": {},
            "type_distribution": {},
            "average_effectiveness": 0.0
        }
        
        # Calculate level distribution
        for intervention in recent_interventions:
            level = intervention['plan']['level']
            stats['level_distribution'][level] = stats['level_distribution'].get(level, 0) + 1
            
            # Calculate type distribution
            for intv_type in intervention['plan']['types']:
                stats['type_distribution'][intv_type] = stats['type_distribution'].get(intv_type, 0) + 1
        
        # Calculate average effectiveness
        effectiveness_scores = [i.get('measured_effectiveness', 0) 
                               for i in recent_interventions 
                               if 'measured_effectiveness' in i]
        if effectiveness_scores:
            stats['average_effectiveness'] = sum(effectiveness_scores) / len(effectiveness_scores)
        
        return stats

    def generate_crisis_protocol(self, crisis_type: str) -> Dict:
        """Generate specific crisis protocol for emergency situations"""
        crisis_protocols = {
            "panic_attack": {
                "name": "Panic Attack Response Protocol",
                "steps": [
                    "MAITRI: I'm here with you. You're having a panic attack, but you're safe.",
                    "Focus on your breathing with me: Inhale... 2... 3... 4... Exhale... 2... 3... 4...",
                    "Name three things you can see in the module.",
                    "Feel the surface beneath you. You're secure and supported.",
                    "Medical team has been alerted. Continue breathing with me."
                ],
                "emergency_actions": [
                    "Alert ground control medical team",
                    "Monitor vital signs",
                    "Maintain verbal contact",
                    "Prepare emergency medical kit"
                ]
            },
            "severe_depression": {
                "name": "Severe Depression Response Protocol", 
                "steps": [
                    "MAITRI: I understand this is incredibly difficult. You're not alone.",
                    "Your ground support team cares deeply about you.",
                    "Let's focus on getting through the next few minutes together.",
                    "Would you like me to contact a specific support person?",
                    "Emergency psychological support has been activated."
                ],
                "emergency_actions": [
                    "Notify ground control psychology team",
                    "Initiate emergency communication protocol", 
                    "Schedule immediate video consultation",
                    "Activate crew support network"
                ]
            }
        }
        
        return crisis_protocols.get(crisis_type, crisis_protocols["panic_attack"])