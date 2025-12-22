# VLM Module Usage Guide

The VLM (Vision-Language Model) module provides a standalone, testable interface for interacting with Vision-Language Model APIs (OpenAI GPT-4o, GPT-4o-mini, etc.) with built-in error handling, rate limiting, and cost tracking.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Single Queries](#single-queries)
- [Batch Queries](#batch-queries)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Cost Tracking](#cost-tracking)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

---

## Installation

The VLM module is included in the `src/vlm/` directory. No additional installation is required beyond the main project dependencies:

```bash
pip install openai pydantic pillow
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

---

## Quick Start

Here's a minimal example to get started:

```python
import os
from pathlib import Path
from src.vlm import VLMClient, VLMConfig, VLMRequest

# Configure the client
config = VLMConfig(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="gpt-4o-mini"  # or "gpt-4o"
)
client = VLMClient(config)

# Create a request
request = VLMRequest(
    image_path=Path("path/to/image.jpg"),
    prompt="What object is shown in this image?",
    model="gpt-4o-mini"
)

# Query the VLM
response = client.query_region(request)

# Access results
print(f"Label: {response.label}")
print(f"Confidence: {response.confidence}")
print(f"Cost: ${response.cost:.4f}")
print(f"Latency: {response.latency_ms:.0f}ms")
```

---

## Configuration

### VLMConfig

The `VLMConfig` class configures the VLM client with validation and sensible defaults:

```python
from src.vlm import VLMConfig

config = VLMConfig(
    api_key="sk-...",                    # Required: OpenAI API key
    model_name="gpt-4o",                 # Default model for requests
    timeout=30,                          # Request timeout (1-300 seconds)
    max_retries=5,                       # Retry attempts (0-10)
    retry_base_delay=1.0,               # Exponential backoff base (0.1-10.0)
    retry_max_delay=60.0,               # Max retry delay (1.0-300.0)
    confidence_threshold=0.7,           # Threshold for VLM_UNCERTAIN (0.0-1.0)
    rate_limiter_config=None            # Optional rate limiter (see below)
)
```

All fields except `api_key` have sensible defaults. The config is **immutable** (frozen) after creation.

### Model Selection

Choose a model based on your accuracy/cost tradeoffs:

| Model | Input Cost | Output Cost | Use Case |
|-------|-----------|------------|----------|
| `gpt-4o-mini` | $0.15/1M tokens | $0.60/1M tokens | High-volume, cost-sensitive |
| `gpt-4o` | $2.50/1M tokens | $10.00/1M tokens | High-accuracy requirements |
| `gpt-4-turbo` | $10.00/1M tokens | $30.00/1M tokens | Legacy, highest accuracy |

---

## Single Queries

### Basic Single Query

```python
from src.vlm import VLMRequest, VLMStatus

request = VLMRequest(
    image_path=Path("excavator.jpg"),
    prompt="Identify the type of vehicle in this image",
    model="gpt-4o-mini",
    max_tokens=500,        # Max tokens in response
    temperature=0.2        # Lower = more deterministic
)

response = client.query_region(request)

# Check status
if response.status == VLMStatus.SUCCESS:
    print(f"✅ Success: {response.label}")
    print(f"   Confidence: {response.confidence:.2%}")
    print(f"   Reasoning: {response.reasoning}")
elif response.status == VLMStatus.VLM_UNCERTAIN:
    print(f"⚠️  Uncertain: {response.label} (low confidence)")
else:
    print(f"❌ Failed: {response.error_message}")
```

### Response Fields

```python
response.request_id          # UUID of the request
response.status              # VLMStatus: SUCCESS, VLM_UNCERTAIN, or FAILED
response.label               # Semantic label (e.g., "excavator")
response.confidence          # Confidence score (0.0-1.0)
response.reasoning           # Explanation from VLM
response.raw_response        # Full JSON response
response.cost                # Cost in USD
response.tokens_used         # Total tokens
response.latency_ms          # Query latency in milliseconds
response.queried_at          # Timestamp of query
response.responded_at        # Timestamp of response
response.error_message       # Error message (if failed)
```

---

## Batch Queries

For multiple images, use batch queries for parallel processing:

```python
from src.vlm import VLMBatchRequest

# Create multiple requests
requests = [
    VLMRequest(
        image_path=Path(f"image_{i}.jpg"),
        prompt="What is this object?",
        model="gpt-4o-mini"
    )
    for i in range(10)
]

# Create batch request
batch_request = VLMBatchRequest(
    requests=requests,
    max_workers=5,         # Parallel workers (1-50)
    fail_fast=False        # Continue on errors
)

# Progress callback (optional)
def on_progress(completed, total):
    print(f"Progress: {completed}/{total}")

# Execute batch
batch_response = client.query_batch(
    batch_request,
    progress_callback=on_progress
)

# Aggregate results
print(f"Total requests: {batch_response.total_requests}")
print(f"Successful: {batch_response.successful}")
print(f"Uncertain: {batch_response.uncertain}")
print(f"Failed: {batch_response.failed}")
print(f"Total cost: ${batch_response.total_cost:.4f}")
print(f"Total time: {batch_response.total_latency_ms:.0f}ms")

# Process individual responses
for response in batch_response.responses:
    if response.status == VLMStatus.SUCCESS:
        print(f"✅ {response.request_id}: {response.label}")
    else:
        print(f"❌ {response.request_id}: {response.error_message}")
```

### Fail-Fast Mode

Set `fail_fast=True` to stop batch processing on first error:

```python
batch_request = VLMBatchRequest(
    requests=requests,
    fail_fast=True  # Stop on first error
)
```

---

## Error Handling

The module provides a hierarchy of exceptions for granular error handling:

### Exception Hierarchy

```
VLMException (base)
├── VLMAuthError          # Authentication failures
├── VLMRateLimitError     # Rate limit exceeded
├── VLMResponseError      # Malformed/invalid responses
└── VLMNetworkError       # Network/API errors
```

### Handling Errors

```python
from src.vlm import (
    VLMAuthError,
    VLMRateLimitError,
    VLMResponseError,
    VLMNetworkError,
    VLMException
)

try:
    response = client.query_region(request)
except VLMAuthError as e:
    print(f"❌ Authentication failed: {e}")
    print(f"   Check your API key")
except VLMRateLimitError as e:
    print(f"⏱️  Rate limit exceeded: {e}")
    print(f"   Retry after: {e.retry_after} seconds")
    time.sleep(e.retry_after or 60)
except VLMResponseError as e:
    print(f"⚠️  Invalid response: {e}")
    print(f"   Response content: {e.response_content}")
except VLMNetworkError as e:
    print(f"🌐 Network error: {e}")
    print(f"   Original error: {e.original_error}")
except VLMException as e:
    print(f"💥 Unexpected error: {e}")
```

### Automatic Retry

The client automatically retries on rate limit errors with exponential backoff:

```python
config = VLMConfig(
    api_key="sk-...",
    max_retries=5,              # Retry up to 5 times
    retry_base_delay=1.0,       # Start with 1 second
    retry_max_delay=60.0        # Cap at 60 seconds
)
```

**Retry delays:**
- Attempt 1: 1s ± 25% jitter
- Attempt 2: 2s ± 25% jitter
- Attempt 3: 4s ± 25% jitter
- Attempt 4: 8s ± 25% jitter
- Attempt 5: 16s ± 25% jitter
- After 5 retries: raises `VLMRateLimitError`

---

## Rate Limiting

Prevent exceeding API rate limits with the built-in rate limiter:

### Enabling Rate Limiting

```python
from src.vlm import VLMConfig, RateLimiterConfig

# Configure rate limiter
rate_limiter_config = RateLimiterConfig(
    requests_per_second=10.0,   # Max 10 requests/second
    burst_capacity=20,          # Allow bursts up to 20
    enabled=True
)

# Apply to client
config = VLMConfig(
    api_key="sk-...",
    rate_limiter_config=rate_limiter_config
)
client = VLMClient(config)
```

### How It Works

The rate limiter uses a **token bucket algorithm**:

1. **Bucket capacity**: Max burst size (e.g., 20 tokens)
2. **Refill rate**: Tokens added per second (e.g., 10/sec)
3. **Token consumption**: Each API call consumes 1 token
4. **Blocking**: Waits if no tokens available

**Example with 10 req/sec, burst capacity 20:**

```python
# First 20 requests: immediate (burst)
for i in range(20):
    client.query_region(request)  # Fast

# Next 10 requests: rate limited (1 per 0.1s)
for i in range(10):
    client.query_region(request)  # Waits ~0.1s each
```

### Rate Limiter Configuration

```python
RateLimiterConfig(
    requests_per_second=10.0,   # Refill rate (0.1-100.0)
    burst_capacity=20,          # Bucket size (1-200)
    enabled=True                # Toggle on/off
)
```

### Disabling Rate Limiting

```python
rate_limiter_config = RateLimiterConfig(enabled=False)
# or
config = VLMConfig(
    api_key="sk-...",
    rate_limiter_config=None  # No rate limiting
)
```

---

## Cost Tracking

Every response includes cost information:

### Per-Request Cost

```python
response = client.query_region(request)

print(f"Cost: ${response.cost:.6f}")
print(f"Tokens: {response.tokens_used}")
print(f"Model: {request.model}")
```

### Batch Cost

```python
batch_response = client.query_batch(batch_request)

print(f"Total cost: ${batch_response.total_cost:.4f}")
print(f"Total tokens: {batch_response.total_tokens}")
print(f"Avg cost/request: ${batch_response.total_cost / len(requests):.6f}")
```

### Cost Estimation

Estimate costs before making requests:

| Model | Cost/Request (typical) |
|-------|----------------------|
| `gpt-4o-mini` | $0.0005 - $0.002 |
| `gpt-4o` | $0.003 - $0.010 |
| `gpt-4-turbo` | $0.015 - $0.050 |

**Note:** Costs vary based on:
- Image resolution (more pixels = more tokens)
- Prompt length
- Response length (max_tokens)

---

## Best Practices

### 1. Use Batch Queries for Multiple Images

```python
# ❌ Slow: Sequential queries
for image_path in image_paths:
    response = client.query_region(VLMRequest(image_path=image_path, ...))

# ✅ Fast: Batch query
requests = [VLMRequest(image_path=p, ...) for p in image_paths]
batch_response = client.query_batch(VLMBatchRequest(requests=requests))
```

### 2. Choose the Right Model

```python
# ❌ Expensive: Always use gpt-4o
config = VLMConfig(api_key="...", model_name="gpt-4o")

# ✅ Cost-effective: Use gpt-4o-mini for simple tasks
config = VLMConfig(api_key="...", model_name="gpt-4o-mini")
```

### 3. Handle Low Confidence

```python
response = client.query_region(request)

if response.status == VLMStatus.VLM_UNCERTAIN:
    # Option 1: Retry with more specific prompt
    refined_request = VLMRequest(
        image_path=request.image_path,
        prompt="Look more carefully at the object. What is it?",
        model="gpt-4o"  # Use better model
    )
    response = client.query_region(refined_request)

    # Option 2: Flag for manual review
    uncertain_regions.append(response.request_id)
```

### 4. Use Rate Limiting in Production

```python
# ✅ Production: Prevent rate limit errors
config = VLMConfig(
    api_key="...",
    rate_limiter_config=RateLimiterConfig(
        requests_per_second=5.0,  # Conservative limit
        burst_capacity=10
    )
)
```

### 5. Monitor Costs

```python
total_cost = 0.0

for request in requests:
    response = client.query_region(request)
    total_cost += response.cost

    if total_cost > 10.0:  # $10 budget
        print("⚠️  Budget limit reached")
        break
```

### 6. Optimize Prompts

```python
# ❌ Vague: "What is this?"
# ✅ Specific: "Identify the type of construction vehicle"

# ❌ Long: "Please carefully examine this image and tell me..."
# ✅ Concise: "Identify the object"
```

### 7. Handle Errors Gracefully

```python
try:
    response = client.query_region(request)
except VLMRateLimitError as e:
    time.sleep(e.retry_after or 60)
    response = client.query_region(request)  # Retry
except VLMException as e:
    logger.error(f"VLM query failed: {e}")
    response = None  # Continue processing
```

---

## API Reference

### VLMClient

```python
class VLMClient:
    def __init__(self, config: VLMConfig)

    def query_region(self, request: VLMRequest) -> VLMResponse:
        """Query VLM for a single region."""

    def query_batch(
        self,
        batch_request: VLMBatchRequest,
        progress_callback=None
    ) -> VLMBatchResponse:
        """Query VLM for multiple regions in parallel."""
```

### VLMConfig

```python
class VLMConfig(BaseModel):
    api_key: str
    model_name: str = "gpt-4o"
    timeout: int = 30
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    confidence_threshold: float = 0.7
    rate_limiter_config: Optional[RateLimiterConfig] = None
```

### VLMRequest

```python
class VLMRequest(BaseModel):
    region_id: UUID = Field(default_factory=uuid4)
    image_path: Path
    prompt: str
    model: str = "gpt-4o"
    max_tokens: int = 500
    temperature: float = 0.2
```

### VLMResponse

```python
class VLMResponse(BaseModel):
    request_id: UUID
    status: VLMStatus
    label: Optional[str]
    confidence: Optional[float]
    reasoning: Optional[str]
    raw_response: Optional[Dict[str, Any]]
    cost: float
    tokens_used: int
    latency_ms: float
    queried_at: datetime
    responded_at: Optional[datetime]
    error_message: Optional[str]
```

### VLMBatchRequest

```python
class VLMBatchRequest(BaseModel):
    requests: List[VLMRequest]
    max_workers: int = 5
    fail_fast: bool = False
```

### VLMBatchResponse

```python
class VLMBatchResponse(BaseModel):
    responses: List[VLMResponse]
    total_requests: int
    successful: int
    failed: int
    uncertain: int
    total_cost: float
    total_tokens: int
    total_latency_ms: float
    started_at: datetime
    completed_at: datetime
```

### RateLimiterConfig

```python
class RateLimiterConfig(BaseModel):
    requests_per_second: float = 10.0
    burst_capacity: int = 20
    enabled: bool = True
```

### VLMStatus

```python
class VLMStatus(str, Enum):
    SUCCESS = "success"
    VLM_UNCERTAIN = "vlm_uncertain"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
```

---

## Examples

### Example 1: Simple Object Detection

```python
from pathlib import Path
from src.vlm import VLMClient, VLMConfig, VLMRequest

config = VLMConfig(api_key="sk-...")
client = VLMClient(config)

request = VLMRequest(
    image_path=Path("car.jpg"),
    prompt="What type of vehicle is this?",
    model="gpt-4o-mini"
)

response = client.query_region(request)
print(f"Vehicle type: {response.label}")
```

### Example 2: Batch Processing with Progress

```python
from src.vlm import VLMBatchRequest
import tqdm

requests = [
    VLMRequest(image_path=Path(f"frame_{i}.jpg"), prompt="Object?")
    for i in range(100)
]

batch_request = VLMBatchRequest(requests=requests, max_workers=10)

pbar = tqdm.tqdm(total=len(requests))
def update_progress(completed, total):
    pbar.update(completed - pbar.n)

batch_response = client.query_batch(batch_request, progress_callback=update_progress)
pbar.close()
```

### Example 3: Cost-Aware Processing

```python
MAX_BUDGET = 5.0  # $5 budget
total_cost = 0.0

for image_path in image_paths:
    if total_cost >= MAX_BUDGET:
        print("Budget limit reached")
        break

    request = VLMRequest(image_path=image_path, prompt="Identify object")
    response = client.query_region(request)
    total_cost += response.cost

    print(f"Processed {image_path} (cost: ${response.cost:.4f}, total: ${total_cost:.4f})")
```

### Example 4: Error Recovery

```python
import time

def query_with_retry(client, request, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return client.query_region(request)
        except VLMRateLimitError as e:
            if attempt == max_attempts - 1:
                raise
            wait_time = e.retry_after or (2 ** attempt)
            print(f"Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
        except VLMException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)

    raise VLMException("All retry attempts failed")

response = query_with_retry(client, request)
```

---

## Troubleshooting

### "Authentication failed" error

**Solution:** Check that your API key is valid and has permissions:

```bash
export OPENAI_API_KEY="sk-your-real-api-key"
```

### "Rate limit exceeded" errors

**Solution:** Enable rate limiting or reduce request rate:

```python
config = VLMConfig(
    api_key="...",
    rate_limiter_config=RateLimiterConfig(requests_per_second=5.0)
)
```

### High costs

**Solution:** Use `gpt-4o-mini` and reduce `max_tokens`:

```python
request = VLMRequest(
    image_path=image_path,
    prompt="Object?",
    model="gpt-4o-mini",  # Cheaper
    max_tokens=100        # Limit response length
)
```

### Low confidence scores

**Solution:** Try a more specific prompt or better model:

```python
# More specific prompt
request = VLMRequest(
    image_path=image_path,
    prompt="Identify the specific type of construction equipment",
    model="gpt-4o"  # Better model
)
```

---

## Support

For issues or questions:
- Check the [integration tests](../tests/integration/vlm/test_vlm_integration.py) for examples
- Review the [source code](../src/vlm/) for implementation details
- File an issue in the project repository

---

**Version:** 1.0.0
**Last Updated:** 2025-12-22
