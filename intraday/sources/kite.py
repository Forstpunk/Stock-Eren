"""Kite Connect adapter - NOT IMPLEMENTED. Placeholder documenting the contract.

When built, ``KiteSource`` must satisfy ``intraday.sources.BarSource`` and handle:

- instrument_token lookup: map an NSE trading symbol (e.g. ``RELIANCE``) to the
  numeric instrument_token from the daily instruments dump (``kite.instruments("NSE")``);
  cache the dump per day; raise if a symbol is absent rather than guessing.
- per-request day caps: the historical API serves at most 100 days per request for
  ``5minute`` and 60 days for ``minute``. Chunk the requested range accordingly
  and concatenate; assert no duplicate timestamps across chunk boundaries.
- rate limit: 3 requests/second on the historical endpoint. Throttle internally;
  callers pass a date range and get bars back.
- access-token expiry: the daily access token expires at 06:00 IST. Detect an
  expired/invalid token (TokenException) and raise ``DataUnavailableError`` with a
  reason naming the expiry - never silently re-login inside a fetch.
- interval naming: Kite uses ``minute``, ``5minute``, ``15minute``, ``day``;
  translate from the pipeline's ``1m``/``5m``/``15m``/``1d`` at the boundary.
- timestamps arrive tz-naive in IST; localise to ``Asia/Kolkata`` explicitly.
- ``max_history_days`` is roughly 10 years for ``5minute`` (subscription dependent).
"""
from __future__ import annotations
