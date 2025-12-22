# Implementation Tasks: Independent VLM Module

**Feature**: Independent VLM Client Module for Semantic Labeling
**Branch**: `002-semantic-labeling-all-regions`
**Date**: 2025-12-22
**Context**: Extract and modularize VLM calling logic for testability and reusability

---

## Overview

This task list focuses on creating an **independent, modular VLM client module** that can be tested and used across different parts of the semantic labeling system. The module will encapsulate all VLM API interactions, error handling, rate limiting, and response parsing in a standalone, testable component.

**User Input Context**: "An independent module for calling VLM models need to be implemented and tested for modularity."

**Total Tasks**: 28 tasks organized in 5 phases

---

## Phase 1: Setup & Foundation (4 tasks)

**Goal**: Establish project structure and testing infrastructure for independent VLM module

- [ ] T001 Create independent VLM module structure in `src/vlm/` directory with `__init__.py`, `client.py`, `models.py`, `exceptions.py`
- [ ] T002 Define VLM-specific exception hierarchy in `src/vlm/exceptions.py` (`VLMException`, `VLMAuthError`, `VLMRateLimitError`, `VLMResponseError`, `VLMNetworkError`)
- [ ] T003 Create Pydantic models for VLM requests/responses in `src/vlm/models.py` (`VLMRequest`, `VLMResponse`, `VLMBatchRequest`, `VLMBatchResponse`)
- [ ] T004 Set up pytest fixtures for VLM module testing in `tests/unit/vlm/conftest.py` (mock API responses, test images, auth fixtures)

---

## Phase 2: Core VLM Client Implementation (8 tasks)

**Goal**: Implement modular VLM client with authentication, request handling, and error management

**Independent Test**: Module can make authenticated API calls, handle errors gracefully, and return structured responses

### Authentication & Configuration

- [ ] T005 [P] Implement VLM client configuration model in `src/vlm/client.py` (`VLMConfig` with api_key, model_name, timeout, retry_config)
- [ ] T006 [P] Implement VLM client initialization in `src/vlm/client.py` (`VLMClient.__init__()` with config validation and OpenAI client setup)
- [ ] T007 Write unit tests for VLM client configuration in `tests/unit/vlm/test_client_config.py` (valid/invalid configs, environment variables)

### Request Handling

- [ ] T008 [P] Implement single region query method in `src/vlm/client.py` (`VLMClient.query_region()` with image encoding, prompt formatting)
- [ ] T009 [P] Implement batch region query method in `src/vlm/client.py` (`VLMClient.query_batch()` with parallel execution using ThreadPoolExecutor)
- [ ] T010 [P] Implement image encoding utilities in `src/vlm/utils.py` (`encode_image_base64()`, `validate_image_format()`)
- [ ] T011 Write unit tests for single region queries in `tests/unit/vlm/test_client_queries.py` (successful queries, invalid images, malformed responses)
- [ ] T012 Write unit tests for batch queries in `tests/unit/vlm/test_client_batch.py` (parallel execution, partial failures, batch size limits)

---

## Phase 3: Error Handling & Resilience (6 tasks)

**Goal**: Implement comprehensive error handling, retry logic, and rate limiting

**Independent Test**: Module handles API errors, rate limits, and network issues without crashing

### Error Handling

- [ ] T013 [P] Implement error classification in `src/vlm/client.py` (`_classify_error()` maps OpenAI exceptions to VLM exceptions)
- [ ] T014 [P] Implement retry logic with exponential backoff in `src/vlm/client.py` (`_retry_with_backoff()` with jitter, max_retries=5)
- [ ] T015 Write unit tests for error handling in `tests/unit/vlm/test_error_handling.py` (auth errors, network errors, invalid responses)

### Rate Limiting

- [ ] T016 [P] Implement proactive rate limiter in `src/vlm/rate_limiter.py` (`RateLimiter` class with token bucket algorithm)
- [ ] T017 [P] Integrate rate limiter with VLM client in `src/vlm/client.py` (`_wait_for_rate_limit()` before API calls)
- [ ] T018 Write unit tests for rate limiting in `tests/unit/vlm/test_rate_limiter.py` (rate limit enforcement, burst handling, backpressure)

---

## Phase 4: Response Processing & Validation (5 tasks)

**Goal**: Implement response parsing, validation, and confidence evaluation

**Independent Test**: Module correctly parses VLM responses, validates JSON, and evaluates confidence scores

- [ ] T019 [P] Implement response parser in `src/vlm/parser.py` (`parse_vlm_response()` with JSON extraction, field validation)
- [ ] T020 [P] Implement confidence evaluator in `src/vlm/parser.py` (`evaluate_confidence()` maps VLM confidence to status: SUCCESS/VLM_UNCERTAIN)
- [ ] T021 [P] Implement response validator in `src/vlm/parser.py` (`validate_response_schema()` checks required fields: label, confidence, reasoning)
- [ ] T022 Write unit tests for response parsing in `tests/unit/vlm/test_parser.py` (valid JSON, malformed JSON, missing fields)
- [ ] T023 Write unit tests for confidence evaluation in `tests/unit/vlm/test_confidence.py` (high confidence, uncertain, edge cases)

---

## Phase 5: Integration & Cost Tracking (5 tasks)

**Goal**: Add cost tracking, usage metrics, and integration testing

**Independent Test**: Module tracks API costs accurately and provides usage statistics

### Cost Tracking

- [ ] T024 [P] Implement cost calculator in `src/vlm/cost.py` (`CostCalculator` with model pricing, token counting)
- [ ] T025 [P] Integrate cost tracking with VLM client in `src/vlm/client.py` (`track_query_cost()` updates usage metrics)
- [ ] T026 Write unit tests for cost calculation in `tests/unit/vlm/test_cost.py` (per-model pricing, batch costs, currency formatting)

### Integration Testing

- [ ] T027 Write integration tests for VLM module in `tests/integration/vlm/test_vlm_integration.py` (real API calls with test images, error scenarios, rate limiting)
- [ ] T028 Create module usage examples in `docs/vlm_module_usage.md` (initialization, single queries, batch queries, error handling)

---

## Dependencies & Execution Order

### Critical Path
```
T001-T004 (Setup)
    ↓
T005-T010 (Core Client)
    ↓
T013-T018 (Error Handling & Rate Limiting)
    ↓
T019-T023 (Response Processing)
    ↓
T024-T028 (Cost Tracking & Integration)
```

### Parallel Opportunities (per phase)

**Phase 2 (Core Client)**:
- Parallel group 1: T005, T006, T010 (Configuration & utilities - different files)
- Sequential: T007 (tests depend on T005-T006)
- Parallel group 2: T008, T009 (Different query methods)
- Sequential: T011-T012 (tests depend on T008-T009)

**Phase 3 (Error Handling)**:
- Parallel group 1: T013, T014, T016 (Different error handling components)
- Sequential: T015 (tests depend on T013-T014)
- Sequential: T017 (integration depends on T016)
- Sequential: T018 (tests depend on T016-T017)

**Phase 4 (Response Processing)**:
- Parallel group 1: T019, T020, T021 (Different parser functions)
- Parallel group 2: T022, T023 (Independent test files)

**Phase 5 (Cost Tracking)**:
- Parallel group 1: T024, T025 (Cost calculator & integration)
- Sequential: T026 (tests depend on T024-T025)
- Sequential: T027-T028 (Integration tests & docs)

---

## Module Architecture

```
src/vlm/
├── __init__.py          # Public API exports
├── client.py            # VLMClient class (main interface)
├── models.py            # Pydantic request/response models
├── exceptions.py        # VLM-specific exceptions
├── parser.py            # Response parsing & validation
├── rate_limiter.py      # Rate limiting logic
├── cost.py              # Cost tracking & calculation
└── utils.py             # Image encoding utilities

tests/unit/vlm/
├── conftest.py          # Test fixtures
├── test_client_config.py
├── test_client_queries.py
├── test_client_batch.py
├── test_error_handling.py
├── test_rate_limiter.py
├── test_parser.py
├── test_confidence.py
└── test_cost.py

tests/integration/vlm/
└── test_vlm_integration.py
```

---

## Public API Interface

```python
# Recommended usage pattern
from src.vlm import VLMClient, VLMConfig, VLMRequest

# Initialize client
config = VLMConfig(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",
    timeout=30,
    max_retries=5
)
client = VLMClient(config)

# Single query
request = VLMRequest(
    image_path=Path("frame.jpg"),
    prompt="Identify the object",
    region_id=uuid4()
)
response = client.query_region(request)

# Batch query
requests = [VLMRequest(...) for _ in range(10)]
responses = client.query_batch(requests, max_workers=5)

# Access results
print(f"Label: {response.label}")
print(f"Confidence: {response.confidence}")
print(f"Cost: {response.cost}")
```

---

## Success Criteria

### Module Independence
- ✅ Zero dependencies on existing semantic labeling services
- ✅ Can be imported and used standalone: `from src.vlm import VLMClient`
- ✅ All VLM API logic encapsulated in `src/vlm/` directory

### Testability
- ✅ All components have unit tests (>90% coverage)
- ✅ Integration tests verify real API interaction
- ✅ Mock fixtures enable testing without API calls

### Modularity
- ✅ Clear separation of concerns (client, parser, rate limiter, cost)
- ✅ Pluggable configuration (can swap models, adjust timeouts)
- ✅ Extensible for future VLM providers (not just OpenAI)

### Error Resilience
- ✅ Handles all OpenAI API error types gracefully
- ✅ Automatic retry with exponential backoff
- ✅ Rate limiting prevents API quota exhaustion

### Documentation
- ✅ Usage examples in `docs/vlm_module_usage.md`
- ✅ API documentation in docstrings
- ✅ Clear error messages for common issues

---

## Implementation Strategy

### MVP Scope (Phase 1-2)
Focus on core functionality first:
1. Basic client configuration
2. Single region queries
3. Simple error handling
4. **Validate**: Can make successful VLM API calls

### Incremental Delivery
1. **Phase 1**: Setup (T001-T004) - Establish structure
2. **Phase 2**: Core Client (T005-T012) - Basic functionality working
3. **Phase 3**: Resilience (T013-T018) - Production-ready error handling
4. **Phase 4**: Processing (T019-T023) - Robust response handling
5. **Phase 5**: Integration (T024-T028) - Complete with cost tracking

### Testing Approach
- **Unit tests first**: Write tests before implementation (TDD)
- **Mock external dependencies**: Use pytest fixtures for OpenAI API
- **Integration tests last**: Verify real API behavior with test account

---

## Notes

- **No existing VLMService dependencies**: This module should be completely independent
- **OpenAI SDK version**: Ensure compatibility with latest `openai` package
- **Environment variables**: Support `OPENAI_API_KEY`, `VLM_MODEL`, `VLM_TIMEOUT` env vars
- **Async support**: Consider adding async methods (e.g., `async def query_region_async()`) in future iteration
- **Multi-provider support**: Architecture should allow adding Claude, Gemini providers later

---

## Validation Checklist

Before marking complete, verify:
- [ ] All 28 tasks have clear file paths
- [ ] Module can be imported independently: `from src.vlm import VLMClient`
- [ ] Unit tests achieve >90% code coverage
- [ ] Integration tests pass with real API
- [ ] Documentation includes usage examples
- [ ] No circular dependencies with existing services
- [ ] Error handling covers all OpenAI exception types
- [ ] Rate limiting prevents quota exhaustion
- [ ] Cost tracking matches actual API charges
