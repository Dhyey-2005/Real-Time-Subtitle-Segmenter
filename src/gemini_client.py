"""
Gemini Client Module for Phase 1.

Provides optional, controlled semantic support using the official Google GenAI SDK.
The core deterministic NLP pipeline operates completely independently of this module.

SECURITY & SAFETY GUARANTEES:
1. GEMINI_API_KEY is read strictly from environment (.env).
2. The API key is NEVER printed, logged, or hardcoded.
3. All operations fail gracefully on missing key, network failure, quota exceeded, or malformed responses.
4. Core NLP segmentation, extraction, and filtering do not depend on this module.
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Set up clean logger
logger = logging.getLogger("gemini_client")

# Load environment variables from .env if present
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)


class GeminiSemanticClient:
    """
    Controlled semantic-support client using the official Google GenAI SDK.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_env: bool = True):
        """
        Initialize the Gemini client safely.
        
        Args:
            api_key: Optional explicit API key.
            use_env: Whether to fallback to GEMINI_API_KEY environment variable if api_key is None.
        """
        if api_key is not None:
            raw_key = api_key
        elif use_env:
            raw_key = os.getenv("GEMINI_API_KEY")
        else:
            raw_key = None
            
        # Sanitize placeholder values
        if raw_key and raw_key.strip() and raw_key.strip() != "your_gemini_api_key_here":
            self._api_key = raw_key.strip()
        else:
            self._api_key = None
            
        self._client = None
        if self._api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                logger.warning("Failed to initialize Google GenAI client: %s", type(e).__name__)
                self._client = None

    def is_available(self) -> bool:
        """Check if Gemini client is initialized and ready for calls."""
        return self._client is not None

    def categorize_phrases(self, phrases: List[str]) -> Dict[str, str]:
        """
        Optional semantic support: categorizes a small list of candidate phrases
        into high-level topic domains (e.g., Economy, Energy, Technology).
        
        If the API key is absent, network fails, or an error occurs, this function
        gracefully falls back without raising exceptions or disrupting the pipeline.
        
        Args:
            phrases: List of candidate phrase strings (e.g. top 5-10 filtered phrases).
            
        Returns:
            Dictionary mapping phrase to category string.
        """
        if not phrases:
            return {}
            
        if not self.is_available():
            # Graceful local fallback
            return {p: "Local Deterministic Extraction" for p in phrases}
            
        try:
            # Prepare minimal prompt with candidate phrases only (never full transcript)
            phrase_list_str = "\n".join(f"- {p}" for p in phrases[:10])
            prompt = (
                "You are an NLP assistant. For each of the following recurring key phrases, "
                "provide a concise 1-2 word topical domain label (e.g., Economy, Energy, Technology, Governance). "
                "Respond strictly in the format: Phrase: Domain\n\n"
                f"{phrase_list_str}"
            )
            
            response = self._client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            
            results = {}
            if response and response.text:
                for line in response.text.strip().splitlines():
                    if ":" in line:
                        parts = line.split(":", 1)
                        phrase = parts[0].strip().lstrip("-* ").lower()
                        category = parts[1].strip()
                        results[phrase] = category
                        
            # Fill in any missing phrases with fallback
            for p in phrases:
                if p.lower() not in results and p not in results:
                    results[p] = "General Concept"
                    
            return results
            
        except Exception as e:
            # Log error type safely WITHOUT printing any potential secrets, keys, or sensitive payload data
            logger.warning("Gemini semantic query skipped due to: %s", type(e).__name__)
            return {p: "Local Deterministic Extraction" for p in phrases}
