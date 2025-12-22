"""VLM-specific exception hierarchy.

This module defines custom exceptions for VLM operations, providing
clear error categorization for different failure modes.
"""


class VLMException(Exception):
    """Base exception for all VLM-related errors.

    All VLM exceptions inherit from this base class, allowing
    catch-all error handling when needed.
    """

    def __init__(self, message: str, original_error: Exception = None):
        """Initialize VLM exception.

        Args:
            message: Human-readable error description
            original_error: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error


class VLMAuthError(VLMException):
    """Raised when API authentication fails.

    This includes:
    - Invalid API keys
    - Expired credentials
    - Insufficient permissions
    """
    pass


class VLMRateLimitError(VLMException):
    """Raised when API rate limits are exceeded.

    This exception includes retry information to help
    implement exponential backoff strategies.
    """

    def __init__(self, message: str, retry_after: int = None, original_error: Exception = None):
        """Initialize rate limit error.

        Args:
            message: Human-readable error description
            retry_after: Seconds to wait before retrying (if provided by API)
            original_error: Optional underlying exception
        """
        super().__init__(message, original_error)
        self.retry_after = retry_after


class VLMResponseError(VLMException):
    """Raised when VLM returns invalid or malformed responses.

    This includes:
    - Invalid JSON format
    - Missing required fields
    - Unexpected response structure
    """

    def __init__(self, message: str, response_content: str = None, original_error: Exception = None):
        """Initialize response error.

        Args:
            message: Human-readable error description
            response_content: Raw response content (for debugging)
            original_error: Optional underlying exception
        """
        super().__init__(message, original_error)
        self.response_content = response_content


class VLMNetworkError(VLMException):
    """Raised when network connectivity issues occur.

    This includes:
    - Connection timeouts
    - DNS resolution failures
    - Network unreachable errors
    """
    pass
