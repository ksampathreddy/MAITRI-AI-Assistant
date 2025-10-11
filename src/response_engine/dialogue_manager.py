import random
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class DialogueManager:
    def __init__(self, knowledge_base_path: str = "data/knowledge_base.json"):
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
        self.user_profile = {}
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        # Response templates for different scenarios
        self.response_templates = {
            "greeting": [
                "Hello! I'm MAITRI, your AI assistant. How can I support you today?",
                "Hi there! I'm here to help with psychological and physical wellbeing. What's on your mind?",
                "Welcome! I'm MAITRI, ready to assist you with mission support and personal wellbeing."
            ],
            "stress": [
                "I notice you're experiencing stress. Let's practice some deep breathing together.",
                "Stress is common in space missions. Remember your training - you're prepared for this.",
                "How about we break this challenge into smaller, manageable steps?"
            ],
            "fatigue": [
                "I'm detecting signs of fatigue. Proper rest is crucial for mission success.",
                "Fatigue can impact performance. Consider taking a scheduled break.",
                "Remember the importance of sleep hygiene in microgravity environments."
            ],
            "isolation": [
                "Feelings of isolation are normal when you're far from Earth. Your team is here with you.",
                "You're doing groundbreaking work for humanity. Your efforts are deeply appreciated.",
                "Would you like to schedule a video call with your support network back home?"
            ],
            "physical": [
                "I notice some physical discomfort. Let me suggest some microgravity exercises.",
                "Proper ergonomics are important. Would you like posture adjustment tips?",
                "Consider using the ARED for some resistance exercise to relieve tension."
            ],
            "general_support": [
                "I'm here to listen and support you through this.",
                "Your wellbeing is my priority. Please share what you're comfortable with.",
                "I'm trained to provide evidence-based psychological support."
            ]
        }
        
        self.logger.info("Dialogue manager initialized")
    
    def _load_knowledge_base(self, path: str) -> Dict:
        """Load psychological and space mission knowledge base"""
        base_knowledge = {
            "psychological_interventions": {
                "deep_breathing": "Practice 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s",
                "grounding": "Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste",
                "progressive_relaxation": "Tense and relax muscle groups from toes to head",
                "mindfulness": "Focus on present moment without judgment for 5 minutes"
            },
            "space_mission_facts": {
                "sleep": "Astronauts need 8 hours sleep but often get less due to workload",
                "exercise": "2 hours daily exercise prevents muscle and bone loss",
                "nutrition": "Balanced diet with supplements maintains health in space",
                "communication": "Regular contact with ground control and family reduces isolation"
            },
            "crisis_protocols": {
                "panic_attack": "Focus on breathing, use grounding techniques, contact medical officer",
                "severe_anxiety": "Practice relaxation, use cognitive restructuring, seek support",
                "depression_signs": "Persistent sadness, loss of interest, changes in sleep/appetite",
                "emergency": "Immediately contact ground control medical team"
            }
        }
        
        try:
            with open(path, 'r') as f:
                custom_knowledge = json.load(f)
                base_knowledge.update(custom_knowledge)
        except FileNotFoundError:
            self.logger.info("No custom knowledge base found, using default")
        
        return base_knowledge
    
    def generate_response(self, user_input: str, current_state: Dict, history: List) -> str:
        """Generate contextual response based on user input and current state"""
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "state": current_state,
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze emotional context
        emotional_context = self._analyze_emotional_context(current_state)
        
        # Generate appropriate response
        if self._is_greeting(user_input):
            response = random.choice(self.response_templates["greeting"])
        
        elif self._requires_intervention(current_state):
            response = self._generate_intervention_response(current_state)
        
        elif self._is_crisis_situation(user_input, current_state):
            response = self._generate_crisis_response(user_input, current_state)
        
        else:
            response = self._generate_conversational_response(user_input, emotional_context)
        
        # Add to history
        self.conversation_history[-1]["assistant"] = response
        
        return response
    
    def _analyze_emotional_context(self, state: Dict) -> str:
        """Analyze current emotional state for context"""
        emotion = state.get("emotion", {})
        physical = state.get("physical", {})
        
        dominant_emotion, confidence = max(emotion.items(), key=lambda x: x[1]) if emotion else ("neutral", 1.0)
        
        if confidence > 0.7:
            if dominant_emotion in ["sad", "angry", "fear"]:
                return "negative_high"
            elif dominant_emotion == "happy":
                return "positive_high"
        
        if physical.get("fatigue") == "high":
            return "fatigue_high"
        
        return "neutral"
    
    def _is_greeting(self, text: str) -> bool:
        """Check if input is a greeting"""
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        return any(greeting in text.lower() for greeting in greetings)
    
    def _requires_intervention(self, state: Dict) -> bool:
        """Check if current state requires psychological intervention"""
        emotion = state.get("emotion", {})
        
        # High negative emotions
        if any(score > 0.7 for emotion, score in emotion.items() if emotion in ["sad", "angry", "fear"]):
            return True
        
        # Physical distress
        physical = state.get("physical", {})
        if physical.get("fatigue") == "high" and physical.get("posture") == "poor":
            return True
        
        return False
    
    def _is_crisis_situation(self, user_input: str, state: Dict) -> bool:
        """Check for crisis indicators in user input"""
        crisis_keywords = [
            "emergency", "help me", "can't cope", "overwhelmed", 
            "panic", "anxiety attack", "suicide", "hurt myself"
        ]
        
        input_lower = user_input.lower()
        if any(keyword in input_lower for keyword in crisis_keywords):
            return True
        
        # Check emotional state
        emotion = state.get("emotion", {})
        if emotion.get("fear", 0) > 0.8 or emotion.get("sad", 0) > 0.8:
            return True
        
        return False
    
    def _generate_intervention_response(self, state: Dict) -> str:
        """Generate evidence-based psychological intervention"""
        emotion = state.get("emotion", {})
        physical = state.get("physical", {})
        
        # Determine intervention type
        if any(score > 0.7 for emotion, score in emotion.items() if emotion in ["sad", "angry", "fear"]):
            intervention = random.choice(self.knowledge_base["psychological_interventions"].values())
            return f"I notice you might benefit from some support. {intervention}"
        
        elif physical.get("fatigue") == "high":
            return ("I'm detecting signs of fatigue. Remember that proper rest is essential for "
                   "mission performance. Consider taking a scheduled break and practicing "
                   "the sleep hygiene protocols we've discussed.")
        
        else:
            return random.choice(self.response_templates["general_support"])
    
    def _generate_crisis_response(self, user_input: str, state: Dict) -> str:
        """Generate response for crisis situations"""
        crisis_protocol = (
            "I'm detecting this is a crisis situation. I'm immediately alerting the "
            "ground control medical team. Please stay on the line. Help is coming. "
            "In the meantime, let's practice some grounding techniques together. "
            "Can you name five things you can see in your immediate environment?"
        )
        
        # Log crisis for ground control
        self.logger.critical(
            f"CRISIS DETECTED: User input: '{user_input}', "
            f"Emotional state: {state.get('emotion', {})}"
        )
        
        return crisis_protocol
    
    def _generate_conversational_response(self, user_input: str, context: str) -> str:
        """Generate conversational response based on input and context"""
        input_lower = user_input.lower()
        
        # Topic-based responses
        if any(word in input_lower for word in ['stress', 'pressure', 'overwhelm']):
            return random.choice(self.response_templates["stress"])
        
        elif any(word in input_lower for word in ['tired', 'exhaust', 'sleep', 'fatigue']):
            return random.choice(self.response_templates["fatigue"])
        
        elif any(word in input_lower for word in ['lonely', 'isolat', 'miss', 'home', 'earth']):
            return random.choice(self.response_templates["isolation"])
        
        elif any(word in input_lower for word in ['pain', 'discomfort', 'ache', 'physical']):
            return random.choice(self.response_templates["physical"])
        
        elif any(word in input_lower for word in ['help', 'support', 'advice']):
            return ("I'm here to provide psychological first aid, stress management techniques, "
                   "and general support. You can talk to me about anything from mission stress "
                   "to personal concerns.")
        
        else:
            # Generative fallback response
            return self._generate_contextual_fallback(input_lower, context)
    
    def _generate_contextual_fallback(self, user_input: str, context: str) -> str:
        """Generate contextual fallback response"""
        if context == "negative_high":
            responses = [
                "I sense you're going through something difficult. I'm here to listen.",
                "Whatever you're experiencing, you don't have to face it alone.",
                "Your feelings are valid. Would you like to talk more about what's happening?"
            ]
        elif context == "fatigue_high":
            responses = [
                "I notice you might be tired. Remember that rest is a crucial part of mission readiness.",
                "Fatigue can make everything feel more challenging. How about we focus on one thing at a time?",
                "Your body is working hard in this unique environment. Be kind to yourself."
            ]
        else:
            responses = [
                "I'm here to support you. What would you like to talk about?",
                "How can I assist you with your wellbeing today?",
                "I'm listening. Please share what's on your mind."
            ]
        
        return random.choice(responses)
    
    def generate_intervention(self, emotional_summary: Dict) -> str:
        """Generate specific intervention based on emotional summary"""
        if emotional_summary.get("requires_intervention", False):
            critical_emotion = emotional_summary.get("critical_emotion", "")
            
            if critical_emotion == "sad":
                return ("I'm detecting significant sadness. This is a normal human response, "
                       "especially in isolated environments. Let's practice some mood-lifting "
                       "techniques together.")
            elif critical_emotion == "angry":
                return ("I notice strong feelings of anger. Anger often masks other emotions. "
                       "Would you like to explore what might be underneath this feeling?")
            elif critical_emotion == "fear":
                return ("I'm detecting high levels of fear or anxiety. Remember the safety "
                       "protocols and your extensive training. You are prepared for this situation.")
        
        return "I'm here to support you through whatever you're experiencing."