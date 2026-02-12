
import unittest
from simpleai.exceptions import ProviderError

# Mock Exception with headers
class MockOpenAIError(Exception):
    def __init__(self, message, headers=None):
        super().__init__(message)
        self.headers = headers

class TestErrorFormatting(unittest.TestCase):
    def test_formatting(self):
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "59",
            "other-header": "ignore-me"
        }
        exc = MockOpenAIError("Error code: 429", headers)

        msg = f"OpenAI adapter failed: {exc}"

        # Logic from the adapter
        ext_headers = getattr(exc, "headers", None)
        if ext_headers:
            relevant_headers = [
                "x-ratelimit-limit-requests",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-requests",
            ]
            details = [
                f"{key}: {ext_headers.get(key)}"
                for key in relevant_headers
                if ext_headers.get(key) is not None
            ]
            if details:
                msg += "\n\nRate limit headers:\n" + "\n".join(details)

        print("\nGenerated Message:")
        print(msg)

        self.assertIn("x-ratelimit-limit-requests: 60", msg)
        self.assertIn("x-ratelimit-remaining-requests: 59", msg)
        self.assertNotIn("other-header", msg)

if __name__ == "__main__":
    unittest.main()

