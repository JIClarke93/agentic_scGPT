"""Shared retry policies for Temporal activities.

Common retry policies that can be used across different agents.
Agents can import these or define their own specialized policies.
"""

from datetime import timedelta

from temporalio.common import RetryPolicy


# Standard retry policy for GPU operations
# Use for activities that run on GPU and may fail due to OOM or transient errors
GPU_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=2),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=[
        "InvalidDataError",
        "ModelNotFoundError",
        "FileNotFoundError",
        "ValueError",
    ],
)

# Standard retry policy for API calls
# Use for activities that call external APIs
API_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    backoff_coefficient=2.0,
    maximum_attempts=5,
    non_retryable_error_types=[
        "AuthenticationError",
        "InvalidRequestError",
    ],
)

# Standard retry policy for file I/O operations
# Use for activities that read/write files
IO_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
    backoff_coefficient=1.5,
    maximum_attempts=3,
    non_retryable_error_types=[
        "FileNotFoundError",
        "PermissionError",
    ],
)

# No retry policy - for activities that should not be retried
NO_RETRY_POLICY = RetryPolicy(
    maximum_attempts=1,
)
