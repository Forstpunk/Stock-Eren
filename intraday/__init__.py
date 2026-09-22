"""Intraday research pipeline for NSE equities.

A measurement instrument. It answers one question - what distinguishes an
opening-range breakout that sustains from one that fails - and emits verdicts
(edge detected / no edge / insufficient sample). It never emits buy signals,
ratings or price predictions.
"""
from __future__ import annotations
