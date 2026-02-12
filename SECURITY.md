# Security policy

## Scope and threat model

Brad is designed for local-only operation:

- Audio and transcript processing happen on-device.
- No runtime cloud inference.
- No telemetry or analytics.
- No automatic upload behavior.

Primary security boundary:

- Local filesystem contents under the configured Brad data directory.

Threats considered:

- Unauthorized local access to transcript data.
- Accidental data leakage through misconfiguration.
- Supply-chain risk from third-party Python dependencies.

## Safe defaults

- Offline-safe runtime behavior (no model auto-download).
- Explicit local model paths are required.
- Gradio is launched with `share=False`.
- Data is stored in local SQLite only.

## What Brad does not do

- It does not encrypt local data at rest by default.
- It does not protect against malware or privileged users on the same machine.
- It does not provide remote multi-user authentication.

## Recommended hardening for production-like use

1. Use full-disk encryption on the host.
2. Restrict OS account permissions for Brad data directories.
3. Pin dependencies and run vulnerability scanning.
4. Keep local models and ffmpeg binaries from trusted sources only.

## Reporting vulnerabilities

Please open a private security report if possible, or create an issue with:

- Affected version/commit
- Repro steps
- Impact estimate
- Suggested mitigation

Do not publish sensitive transcript content in reports.
