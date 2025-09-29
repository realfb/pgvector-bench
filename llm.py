import os
from typing import Optional, Any, Type, TypeVar, List
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """Simple client for Google Gemini API with structured output support"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini client

        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            model: Gemini model to use
        """
        # Load from .env if not provided
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or .env file")

        # Configure Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        max_output_tokens: int = 1000,
    ) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: Input prompt
            temperature: Controls randomness (0.0-1.0)
            top_k: Limits vocabulary for each step
            top_p: Cumulative probability for vocabulary
            max_output_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        # Generate content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            ),
        )

        return response.text

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        max_output_tokens: int = 1000,
    ) -> T | None:
        """
        Generate structured output using Pydantic schema

        Args:
            prompt: Input prompt
            response_schema: Pydantic model for response structure
            temperature: Controls randomness
            top_k: Limits vocabulary for each step
            top_p: Cumulative probability for vocabulary
            max_output_tokens: Maximum tokens to generate

        Returns:
            Parsed Pydantic model instance or None if parsing fails
        """
        # Generate content with structured output
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        # Return the parsed response
        try:
            return response.parsed
        except Exception as e:
            print(f"Warning: Failed to parse structured response: {e}")
            return None

    def generate_enum(self, prompt: str, enum_class: Type, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate enum value from a set of options

        Args:
            prompt: Input prompt
            enum_class: Enum class with options
            temperature: Controls randomness
            **kwargs: Additional generation parameters

        Returns:
            Selected enum value as string
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                response_mime_type="text/x.enum",
                response_schema=enum_class,
                **kwargs,
            ),
        )

        return response.text


# Example Pydantic models for testing
class SearchQuery(BaseModel):
    query_text: str
    difficulty: str  # easy, medium, hard
    query_type: str  # keyword, question, conceptual


class SearchQueries(BaseModel):
    queries: List[SearchQuery]


def test_structured_output():
    """Test structured output with Pydantic models"""
    try:
        client = GeminiClient()

        # Test single structured response
        print("Testing structured query generation...")
        query = client.generate_structured(
            prompt="Generate a challenging search query about quantum physics",
            response_schema=SearchQuery,
            temperature=0.8,
        )

        if query:
            print(f"Generated query: '{query.query_text}'")
            print(f"Difficulty: {query.difficulty}")
            print(f"Type: {query.query_type}")

        # Test batch structured response
        print("\nTesting batch query generation...")
        queries_response = client.generate_structured(
            prompt="Generate 3 different search queries about artificial intelligence",
            response_schema=SearchQueries,
            temperature=0.8,
        )

        if queries_response and hasattr(queries_response, "queries"):
            for i, q in enumerate(queries_response.queries, 1):
                print(f"{i}. '{q.query_text}' ({q.difficulty}, {q.query_type})")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    test_structured_output()
