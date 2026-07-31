# Nimbus Cloud — API Rate Limits and Quotas

The Nimbus REST API is rate limited per API key. The Starter plan allows 60 requests per minute, the Team plan allows 600 requests per minute, and the Enterprise plan allows 6,000 requests per minute.

When a key exceeds its limit the API returns HTTP 429 with a Retry-After header telling the client how many seconds to wait. Rate-limit counters reset on a rolling 60-second window.

A single API request body may be at most 10 MB. Individual objects placed in storage may be at most 5 GB each. There is no limit on the total number of objects beyond the plan's overall storage quota.

Webhook deliveries are retried with exponential backoff for up to 24 hours before the event is dropped, and every webhook payload is signed with an HMAC-SHA256 signature the receiver can verify.
