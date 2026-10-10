#!/usr/bin/env python3
"""第04・05章: 時系列データと系列モデルのデモ。

    uv run python scripts/f03_timeseries.py
"""

from ml_lern.foundations import sequence_models, timeseries

if __name__ == "__main__":
    timeseries.demo()
    print()
    sequence_models.demo()
