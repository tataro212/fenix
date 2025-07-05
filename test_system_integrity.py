import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from config_manager import ConfigManager, AppSettings, GeminiSettings, YOLOv8Settings
from gemini_service import GeminiService
from tenacity import RetryError

class TestSystemIntegrity(unittest.TestCase):
    """
    Test suite for validating key architectural improvements:
    1. Pydantic-based configuration loading.
    2. Tenacity-based retry mechanism in GeminiService.
    """

    def test_pydantic_config_loading(self):
        """
        Verify that the ConfigManager correctly loads settings
        from config.ini into Pydantic models.
        """
        print("\n--- Running Test: Pydantic Config Loading ---")
        
        # 1. Initialize ConfigManager
        config_manager = ConfigManager()
        
        # 2. Assert that the main settings object is an instance of AppSettings
        self.assertIsInstance(config_manager.settings, AppSettings)
        
        # 3. Assert that nested settings are the correct Pydantic models
        self.assertIsInstance(config_manager.settings.gemini, GeminiSettings)
        self.assertIsInstance(config_manager.settings.yolov8, YOLOv8Settings)
        
        # 4. Assert that a sample value is loaded correctly
        self.assertIsNotNone(config_manager.settings.gemini.api_key)
        
        print("✅ Pydantic models are being loaded correctly.")

    @patch('gemini_service.genai.GenerativeModel')
    def test_gemini_retry_mechanism(self, mock_generative_model):
        """
        Verify that the tenacity retry mechanism in GeminiService is triggered
        during simulated API failures.
        """
        print("\n--- Running Test: Gemini Service Retry Mechanism ---")

        # 1. Mock the async method that makes the API call
        mock_response = MagicMock()
        mock_response.text = "Mocked successful translation"
        
        mock_api_call = AsyncMock(
            side_effect=[
                asyncio.TimeoutError("Simulated timeout"),
                ConnectionError("Simulated connection error"),
                mock_response  # Success on the 3rd attempt
            ]
        )
        mock_generative_model.return_value.generate_content_async = mock_api_call
        
        # 2. Initialize GeminiService
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            gemini_service = GeminiService()

        # 3. Run the translation call that should trigger retries
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            gemini_service._make_gemini_api_call("Hello world", 10)
        )

        # 4. Assertions
        # Verify it was called 3 times (1 initial + 2 retries)
        self.assertEqual(mock_api_call.call_count, 3)
        print(f"✅ API call was retried {mock_api_call.call_count - 1} times as expected.")
        
        # Verify that the final result is the successful one
        self.assertEqual(result, "Mocked successful translation")
        print("✅ Service returned the successful response after retries.")

    @patch('gemini_service.genai.GenerativeModel')
    def test_gemini_retry_failure(self, mock_generative_model):
        """
        Verify that the service fails gracefully after all retries are exhausted.
        """
        print("\n--- Running Test: Gemini Service Retry Failure ---")

        # 1. Mock the async method to always fail
        mock_api_call = AsyncMock(side_effect=ConnectionError("Simulated persistent connection error"))
        mock_generative_model.return_value.generate_content_async = mock_api_call

        # 2. Initialize GeminiService
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            gemini_service = GeminiService()

        # 3. Run the translation and assert that it raises the final exception
        with self.assertRaises(RetryError) as context:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                gemini_service._make_gemini_api_call("Hello", 10)
            )

        # 4. Assertions
        self.assertEqual(mock_api_call.call_count, 3) # Max retries
        print(f"✅ API call failed {mock_api_call.call_count} times as expected.")
        self.assertIsInstance(context.exception.__cause__, ConnectionError)
        print("✅ Correctly raised RetryError after exhausting all attempts.")


if __name__ == '__main__':
    # Fix for compatibility issue with asyncio and unittest on Windows
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    unittest.main() 