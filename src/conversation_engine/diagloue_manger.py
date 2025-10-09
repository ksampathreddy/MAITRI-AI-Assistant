from typing import Dict, List, Tuple
import random
import logging
from datetime import datetime

class DialogueManager:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.conversation_history = []
        self.user_state = {
            "current_emotion": "neutral",
            "emotional_trend": [],
            "last_intervention": None,
            "intervention_count": 0
        }
    
    def generate_response(self, emotional_summary: Dict, user_input: str = "") -> str:
        """Generate appropriate response based on emotional state"""
        
        dominant_emotion = emotional_summary["dominant_emotion"]
        confidence = emotional_summary["confidence"]
        requires_intervention = emotional_summary["requires_intervention"]
        
        # Update user state
        self._update_user_state(emotional_summary)
        
        # Generate appropriate response
        if requires_intervention:
            response = self._generate_intervention_response(emotional_summary)
        elif user_input:
            response = self._generate_conversational_response(user_input, emotional_summary)
        else:
            response = self._generate_proactive_response(emotional_summary)
        
        # Log conversation
        self._log_conversation(user_input, response, emotional_summary)
        
        return response
    
    def _generate_intervention_response(self, emotional_summary: Dict) -> str:
        """Generate intervention response for critical emotional states"""
        
        critical_emotion = emotional_summary["critical_emotion"]
        templates = self.config.RESPONSE_TEMPLATES.get(critical_emotion, [])
        
        if templates:
            response = random.choice(templates)
        else:
            response = f"I notice you're feeling {critical_emotion}. Would you like to talk about it?"
        
        # Update intervention count
        self.user_state["intervention_count"] += 1
        self.user_state["last_intervention"] = datetime.now()
        
        return response
    
    def _generate_conversational_response(self, user_input: str, emotional_summary: Dict) -> str:
        """Generate response to user input"""
        
        # Simple rule-based response generation
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['stress', 'stressed', 'pressure']):
            return "Stress is common in space missions. Remember your training and take deep breaths."
        
        elif any(word in input_lower for word in ['tired', 'fatigue', 'sleep']):
            return "Maintaining good sleep hygiene is crucial. Would you like me to adjust the environment?"
        
        elif any(word in input_lower for word in ['help', 'assist', 'support']):
            return "I'm here to help. You can talk to me about anything that's on your mind."
        
        elif any(word in input_lower for word in ['home', 'earth', 'family']):
            return "Missing Earth is natural. Remember you're doing important work for humanity."
        
        else:
            return self._generate_empathetic_response(emotional_summary)
    
    def _generate_proactive_response(self, emotional_summary: Dict) -> str:
        """Generate proactive response based on emotional state"""
        
        emotion = emotional_summary["dominant_emotion"]
        wellbeing = emotional_summary["wellbeing_score"]
        
        if wellbeing > 80:
            responses = [
                "You seem to be in good spirits today! How can I assist you?",
                "Great to see you doing well. Is there anything you'd like to talk about?",
                "Your positive energy is noticeable. Keep up the good work!"
            ]
        elif wellbeing > 60:
            responses = [
                "How are you feeling today?",
                "Is there anything on your mind you'd like to discuss?",
                "I'm here if you need someone to talk to."
            ]
        else:
            responses = [
                "I notice you might be having a tough time. Would you like to talk?",
                "Remember, it's okay to not feel okay sometimes. I'm here for you.",
                "How about we try a quick relaxation exercise together?"
            ]
        
        return random.choice(responses)
    
    def _generate_empathetic_response(self, emotional_summary: Dict) -> str:
        """Generate empathetic response based on emotional state"""
        
        emotion = emotional_summary["dominant_emotion"]
        
        empathetic_responses = {
            'sad': "I'm sorry you're feeling this way. Remember that your work is incredibly valuable.",
            'angry': "I understand this might be frustrating. Let's work through this together.",
            'fear': "It's natural to feel afraid sometimes. You're in a controlled environment and help is available.",
            'happy': "It's great to see you happy! Your positive mood benefits the entire mission.",
            'neutral': "I'm here whenever you need someone to talk to.",
            'surprise': "Unexpected situations can be challenging. How can I help you process this?"
        }
        
        return empathetic_responses.get(emotion, "I'm here to support you.")
    
    def _update_user_state(self, emotional_summary: Dict):
        """Update user emotional state history"""
        
        self.user_state["current_emotion"] = emotional_summary["dominant_emotion"]
        self.user_state["emotional_trend"].append({
            "emotion": emotional_summary["dominant_emotion"],
            "confidence": emotional_summary["confidence"],
            "timestamp": datetime.now(),
            "wellbeing": emotional_summary["wellbeing_score"]
        })
        
        # Keep only last 10 entries
        if len(self.user_state["emotional_trend"]) > 10:
            self.user_state["emotional_trend"] = self.user_state["emotional_trend"][-10:]
    
    def _log_conversation(self, user_input: str, response: str, emotional_summary: Dict):
        """Log conversation for monitoring and improvement"""
        
        conversation_entry = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "assistant_response": response,
            "emotional_state": emotional_summary,
            "intervention_required": emotional_summary["requires_intervention"]
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Keep log manageable
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]