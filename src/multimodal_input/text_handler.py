import logging
import re
import string
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

class TextHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_context = []
        self.user_profile = {}
        
        # Keywords for different emotional states
        self.emotion_keywords = {
            'stress': ['stress', 'pressure', 'overwhelm', 'anxious', 'worried', 'nervous'],
            'fatigue': ['tired', 'exhaust', 'sleep', 'fatigue', 'drain', 'burnout'],
            'sadness': ['sad', 'depress', 'unhappy', 'down', 'hopeless', 'grief'],
            'anger': ['angry', 'frustrat', 'annoy', 'irritat', 'mad', 'furious'],
            'fear': ['scared', 'afraid', 'fear', 'terrified', 'anxiety', 'panic'],
            'happiness': ['happy', 'joy', 'excite', 'good', 'great', 'wonderful'],
            'isolation': ['lonely', 'alone', 'isolat', 'miss', 'homesick', 'separat']
        }
        
        # Urgency indicators
        self.urgency_indicators = ['help', 'emergency', 'urgent', 'now', 'immediately', 'crisis']
        
        # Physical discomfort indicators
        self.physical_indicators = ['pain', 'ache', 'hurt', 'sore', 'discomfort', 'nausea']
        
        self.logger.info("Text handler initialized")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def extract_emotional_content(self, text: str) -> Dict[str, float]:
        """Extract emotional content from text"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        emotion_scores = {}
        total_keywords = 0
        
        # Count emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            count = 0
            for keyword in keywords:
                # Check for word matches
                for word in words:
                    if keyword in word:
                        count += 1
                        total_keywords += 1
            
            emotion_scores[emotion] = count
        
        # Normalize scores
        if total_keywords > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_keywords
        else:
            # Default to neutral if no keywords found
            emotion_scores['neutral'] = 1.0
        
        return emotion_scores

    def detect_urgency(self, text: str) -> Tuple[bool, float]:
        """Detect urgency level in text"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        urgency_count = 0
        for indicator in self.urgency_indicators:
            for word in words:
                if indicator in word:
                    urgency_count += 1
        
        urgency_level = min(1.0, urgency_count / 3.0)  # Normalize to 0-1
        is_urgent = urgency_level > 0.3
        
        return is_urgent, urgency_level

    def detect_physical_issues(self, text: str) -> Dict[str, float]:
        """Detect physical discomfort mentions"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        physical_scores = {}
        total_mentions = 0
        
        # Common physical issues in space
        physical_issues = {
            'headache': ['headache', 'migraine', 'head pain'],
            'nausea': ['nausea', 'sick', 'vomit', 'dizzy'],
            'fatigue': ['tired', 'exhaust', 'fatigue', 'weak'],
            'sleep': ['insomnia', 'sleep', 'rest', 'awake'],
            'muscle': ['muscle', 'back pain', 'joint', 'cramp']
        }
        
        for issue, keywords in physical_issues.items():
            count = 0
            for keyword in keywords:
                for word in words:
                    if keyword in word:
                        count += 1
                        total_mentions += 1
            
            physical_scores[issue] = count
        
        # Normalize
        if total_mentions > 0:
            for issue in physical_scores:
                physical_scores[issue] /= total_mentions
        
        return physical_scores

    def analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment (-1 to 1 scale)"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Simple sentiment lexicon
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'joy',
            'pleased', 'satisfied', 'comfortable', 'better', 'improve', 'helpful'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'frustrated',
            'annoyed', 'pain', 'hurt', 'tired', 'exhausted', 'stress', 'anxious'
        }
        
        # Intensity modifiers
        intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely'
        }
        
        positive_count = 0
        negative_count = 0
        intensity = 1.0
        
        for i, word in enumerate(words):
            if word in positive_words:
                positive_count += intensity
                intensity = 1.0  # Reset intensity
            elif word in negative_words:
                negative_count += intensity
                intensity = 1.0  # Reset intensity
            elif word in intensifiers:
                intensity = 1.5  # Increase intensity for next word
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment))

    def extract_context(self, text: str) -> Dict[str, any]:
        """Extract contextual information from text"""
        context = {
            'emotional_state': self.extract_emotional_content(text),
            'sentiment': self.analyze_sentiment(text),
            'urgency': self.detect_urgency(text),
            'physical_issues': self.detect_physical_issues(text),
            'word_count': len(text.split()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update conversation context
        self._update_conversation_context(text, context)
        
        return context

    def _update_conversation_context(self, text: str, context: Dict):
        """Update conversation context with new input"""
        context_entry = {
            'text': text,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_context.append(context_entry)
        
        # Keep only last 20 conversations
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]
        
        # Update user profile based on conversation history
        self._update_user_profile()

    def _update_user_profile(self):
        """Update user profile based on conversation history"""
        if not self.conversation_context:
            return
        
        # Analyze emotional patterns
        emotional_patterns = {}
        for entry in self.conversation_context[-10:]:  # Last 10 entries
            emotions = entry['context']['emotional_state']
            for emotion, score in emotions.items():
                if emotion not in emotional_patterns:
                    emotional_patterns[emotion] = []
                emotional_patterns[emotion].append(score)
        
        # Calculate average emotional tendencies
        emotional_tendencies = {}
        for emotion, scores in emotional_patterns.items():
            if scores:
                emotional_tendencies[emotion] = np.mean(scores)
        
        # Update user profile
        self.user_profile.update({
            'emotional_tendencies': emotional_tendencies,
            'conversation_count': len(self.conversation_context),
            'last_updated': datetime.now().isoformat()
        })

    def get_emotional_trend(self) -> Dict[str, float]:
        """Get emotional trend from recent conversations"""
        if len(self.conversation_context) < 3:
            return {}
        
        recent_contexts = self.conversation_context[-5:]
        
        trend = {}
        all_emotions = set()
        
        # Collect all emotions mentioned
        for context in recent_contexts:
            all_emotions.update(context['context']['emotional_state'].keys())
        
        # Calculate average for each emotion
        for emotion in all_emotions:
            scores = [ctx['context']['emotional_state'].get(emotion, 0) 
                     for ctx in recent_contexts]
            trend[emotion] = np.mean(scores)
        
        return trend

    def detect_crisis_indicators(self, text: str) -> Tuple[bool, List[str]]:
        """Detect crisis indicators in text"""
        processed_text = self.preprocess_text(text)
        crisis_indicators = []
        
        # Severe emotional distress indicators
        severe_indicators = [
            'suicide', 'kill myself', 'end it all', 'can\'t go on',
            'hopeless', 'worthless', 'no point', 'give up'
        ]
        
        # Emergency situation indicators
        emergency_indicators = [
            'emergency', 'help me', 'now', 'immediately', 'urgent',
            'danger', 'unsafe', 'scared for my life'
        ]
        
        # Check for severe indicators
        for indicator in severe_indicators:
            if indicator in processed_text:
                crisis_indicators.append(f"severe: {indicator}")
        
        # Check for emergency indicators
        for indicator in emergency_indicators:
            if indicator in processed_text:
                crisis_indicators.append(f"emergency: {indicator}")
        
        is_crisis = len(crisis_indicators) > 0
        return is_crisis, crisis_indicators

    def generate_text_analysis_report(self, text: str) -> Dict:
        """Generate comprehensive text analysis report"""
        context = self.extract_context(text)
        is_urgent, urgency_level = context['urgency']
        is_crisis, crisis_indicators = self.detect_crisis_indicators(text)
        
        report = {
            'text_analysis': {
                'original_text': text,
                'processed_text': self.preprocess_text(text),
                'word_count': context['word_count'],
                'sentiment_score': context['sentiment'],
                'urgency_detected': is_urgent,
                'urgency_level': urgency_level,
                'crisis_detected': is_crisis,
                'crisis_indicators': crisis_indicators
            },
            'emotional_analysis': context['emotional_state'],
            'physical_analysis': context['physical_issues'],
            'contextual_analysis': {
                'conversation_history_count': len(self.conversation_context),
                'emotional_trend': self.get_emotional_trend(),
                'user_profile_updated': self.user_profile.get('last_updated', 'never')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report

    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversation entries"""
        current_time = datetime.now()
        self.conversation_context = [
            entry for entry in self.conversation_context
            if (current_time - datetime.fromisoformat(entry['timestamp'])).total_seconds() / 3600 < max_age_hours
        ]

    def get_handler_status(self) -> Dict:
        """Get text handler status report"""
        return {
            'conversation_count': len(self.conversation_context),
            'user_profile_exists': bool(self.user_profile),
            'last_activity': self.conversation_context[-1]['timestamp'] if self.conversation_context else 'never',
            'emotional_trend_available': len(self.conversation_context) >= 3,
            'status': 'active'
        }