'''
plots.py
--------
Visualisation functions for the futures pipeline.
All functions accept a results_dir parameter — callers control where output is saved.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from data.config import START_DATE, END_DATE, INSTRUMENTS, FX_INSTRUMENTS
from data.data_loader import load_ohlcv, get_all_sessions, get_front_month_daily
from data.data_processor import (
    build_continuous_series, get_ohlcv_resampled, compute_bar_returns,
    get_profile_volume, get_volume_acceleration_profile, get_profile_summary,
    get_session_returns, build_instrument_data,
)

_BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _plot_profile(summary: pd.DataFrame, title: str, path: str) -> None:
    '''Shared helper: save avg/cum return (top) and avg/cum volume (bottom) chart.'''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    bars1 = ax1.bar(range(len(summary)), summary['avg_return'] * 100,
                    color=['#e74c3c' if x < 0 else '#2ecc71' for x in summary['avg_return']],
                    alpha=0.8, label='Avg Return (%)')
    ax1_twin = ax1.twinx()
    line1, = ax1_twin.plot(range(len(summary)), summary['cum_return'] * 100,
                           color='#2c3e50', linewidth=2, marker='o', markersize=4, label='Cum Return (%)')
    ax1.set_ylabel('Avg Return (%)')
    ax1_twin.set_ylabel('Cumulative Return (%)')
    ax1.set_title(title)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.legend(handles=[bars1, line1], loc='upper left')

    bars2 = ax2.bar(range(len(summary)), summary['avg_volume'],
                    color='#3498db', alpha=0.8, label='Avg Volume')
    ax2_twin = ax2.twinx()
    line2, = ax2_twin.plot(range(len(summary)), summary['cum_volume'],
                           color='#2c3e50', linewidth=2, marker='o', markersize=4, label='Cum Volume')
    ax2.set_ylabel('Avg Volume')
    ax2_twin.set_ylabel('Cumulative Volume')
    ax2.set_xlabel('Time (NY)')
    ax2.legend(handles=[bars2, line2], loc='upper left')

    hour_ticks = [i for i, t in enumerate(summary.index) if t.endswith(':00')]
    plt.xticks(hour_ticks, [summary.index[i] for i in hour_ticks], rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def _plot_profile_vol_accel(summary: pd.DataFrame, vol_accel: pd.Series, title: str, path: str) -> None:
    '''Shared helper: avg/cum return (top) and avg volume acceleration % (bottom).'''
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    bars1 = ax1.bar(range(len(summary)), summary['avg_return'] * 100,
                    color=['#e74c3c' if x < 0 else '#2ecc71' for x in summary['avg_return']],
                    alpha=0.8, label='Avg Return (%)')
    ax1_twin = ax1.twinx()
    line1, = ax1_twin.plot(range(len(summary)), summary['cum_return'] * 100,
                           color='#2c3e50', linewidth=2, marker='o', markersize=4, label='Cum Return (%)')
    ax1.set_ylabel('Avg Return (%)')
    ax1_twin.set_ylabel('Cumulative Return (%)')
    ax1.set_title(title)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.legend(handles=[bars1, line1], loc='upper left')

    aligned = vol_accel.reindex(summary.index)
    bars2 = ax2.bar(range(len(aligned)), aligned * 100,
                    color=['#e74c3c' if x < 0 else '#3498db' for x in aligned.fillna(0)],
                    alpha=0.8, label='Avg Vol Accel (%)')
    ax2.set_ylabel('Avg Volume Acceleration (%)')
    ax2.set_xlabel('Time (NY)')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.legend(handles=[bars2], loc='upper left')

    hour_ticks = [i for i, t in enumerate(summary.index) if t.endswith(':00')]
    plt.xticks(hour_ticks, [summary.index[i] for i in hour_ticks], rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_intraday_profile_vol_accel(
    summary: pd.DataFrame,
    vol_accel: pd.Series,
    bars: pd.DataFrame,
    results_dir: str = _BASE_RESULTS_DIR,
    instrument: str = '',
) -> None:
    '''Save avg/cum return (top) and avg volume acceleration (bottom) chart.'''
    os.makedirs(results_dir, exist_ok=True)
    dt     = bars.index.get_level_values('dt_ny') if isinstance(bars.index, pd.MultiIndex) else bars.index
    start  = dt.min().strftime('%Y-%m-%d')
    end    = dt.max().strftime('%Y-%m-%d')
    prefix = f'{instrument} | ' if instrument else ''
    title  = f'{prefix}Return and Volume Acceleration Profile (NY Time) | {start} → {end}'
    _plot_profile_vol_accel(summary, vol_accel, title, os.path.join(results_dir, 'intraday_profile_vol_accel.png'))


def plot_intraday_profile(summary: pd.DataFrame, bars: pd.DataFrame, results_dir: str = _BASE_RESULTS_DIR, instrument: str = '') -> None:
    '''
    Save a bar+line chart of avg/cumulative return and volume by time of day.
    Accepts any resampling frequency.
    '''
    os.makedirs(results_dir, exist_ok=True)

    dt     = bars.index.get_level_values('dt_ny') if isinstance(bars.index, pd.MultiIndex) else bars.index
    start  = dt.min().strftime('%Y-%m-%d')
    end    = dt.max().strftime('%Y-%m-%d')
    prefix = f'{instrument} | ' if instrument else ''
    title  = f'{prefix}Return and Volume Profile (NY Time) | {start} → {end}'

    _plot_profile(summary, title, os.path.join(results_dir, 'intraday_profile.png'))
    summary.to_csv(os.path.join(results_dir, 'intraday_profile.csv'))


def plot_session_profiles_vol_accel(
    summary: pd.DataFrame,
    vol_accel: pd.Series,
    results_dir: str = _BASE_RESULTS_DIR,
    instrument: str = '',
) -> None:
    '''Save return + volume acceleration profile charts split by intraday and overnight session.'''
    os.makedirs(results_dir, exist_ok=True)
    intraday_mask = (summary.index >= '09:30') & (summary.index < '16:00')
    prefix        = f'{instrument} | ' if instrument else ''

    for mask, label, filename in [
        (intraday_mask,  f'{prefix}Intraday Session — Return & Vol Accel (09:30–16:00)', 'session_profile_vol_accel_intraday'),
        (~intraday_mask, f'{prefix}Overnight Session — Return & Vol Accel',              'session_profile_vol_accel_overnight'),
    ]:
        _plot_profile_vol_accel(
            summary[mask],
            vol_accel[vol_accel.index.isin(summary[mask].index)],
            label,
            os.path.join(results_dir, filename + '.png'),
        )


def plot_session_profiles(summary: pd.DataFrame, results_dir: str = _BASE_RESULTS_DIR, instrument: str = '') -> None:
    '''Save separate return+volume profile charts for intraday and overnight sessions.'''
    os.makedirs(results_dir, exist_ok=True)
    intraday_mask = (summary.index >= '09:30') & (summary.index < '16:00')
    prefix        = f'{instrument} | ' if instrument else ''

    for mask, label, filename in [
        (intraday_mask,  f'{prefix}Intraday Session Profile (09:30–16:00)',  'session_profile_intraday'),
        (~intraday_mask, f'{prefix}Overnight Session Profile',               'session_profile_overnight'),
    ]:
        session_summary = summary[mask]
        _plot_profile(session_summary, label, os.path.join(results_dir, filename + '.png'))
        session_summary.to_csv(os.path.join(results_dir, filename + '.csv'))


def plot_session_return_distributions(
    intraday_returns: pd.DataFrame,
    overnight_returns: pd.DataFrame,
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    '''
    Save a histogram of raw session returns.
    Intraday on top, overnight on bottom, x-axis shared.
    '''
    os.makedirs(results_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.hist(intraday_returns['return'].dropna() * 100, bins=80, color='#2ecc71', alpha=0.8, edgecolor='white')
    ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Intraday Session Return Distribution')

    ax2.hist(overnight_returns['return'].dropna() * 100, bins=80, color='#3498db', alpha=0.8, edgecolor='white')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Return (%)')
    ax2.set_title('Overnight Session Return Distribution')

    plt.tight_layout()

    path = os.path.join(results_dir, 'session_return_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_cumulative_returns(
    intraday_returns: pd.DataFrame,
    overnight_returns: pd.DataFrame,
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    intraday_cum  = (1 + intraday_returns["return"].dropna()).cumprod()
    overnight_cum = (1 + overnight_returns["return"].dropna()).cumprod()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(intraday_cum.index,  intraday_cum,  color="#2ecc71", linewidth=1.5, label="Intraday")
    ax.plot(overnight_cum.index, overnight_cum, color="#3498db", linewidth=1.5, label="Overnight")
    ax.set_yscale('log')
    all_vals = pd.concat([intraday_cum, overnight_cum]).dropna()
    lo, hi   = all_vals.min(), all_vals.max()
    ticks    = [t for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] if lo * 0.9 <= t <= hi * 1.1]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}x'))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("Cumulative Return (log scale)")
    ax.set_xlabel("Date")
    ax.set_title("Cumulative Returns: Intraday vs Overnight")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(results_dir, "cumulative_returns.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_cumulative_returns_comparison(
    overnight_by_instrument: dict,
    intraday_by_instrument: dict,
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    '''Save two charts comparing overnight and intraday cumulative returns across instruments.'''
    os.makedirs(results_dir, exist_ok=True)
    colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for session_label, data_by_instrument in [('Overnight', overnight_by_instrument), ('Intraday', intraday_by_instrument)]:
        cum_series = {}
        _, ax = plt.subplots(figsize=(14, 6))
        for (instrument, returns), color in zip(data_by_instrument.items(), colors):
            cum = (1 + returns['return'].dropna()).cumprod()
            cum_series[instrument] = cum
            ax.plot(cum.index, cum, linewidth=1.5, label=instrument, color=color)
        ax.set_yscale('log')

        all_vals = pd.concat(cum_series.values()).dropna()
        lo, hi   = all_vals.min(), all_vals.max()
        ticks    = [t for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0] if lo * 0.9 <= t <= hi * 1.1]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}x'))
        ax.yaxis.set_minor_locator(plt.NullLocator())

        ax.set_ylabel('Cumulative Return (log scale)')
        ax.set_xlabel('Date')
        ax.set_title(f'{session_label} Cumulative Returns by Instrument')
        ax.legend()
        plt.tight_layout()
        filename = f'{session_label.lower()}_cumulative_returns_comparison'
        plt.savefig(os.path.join(results_dir, filename + '.png'), dpi=150)
        plt.close()
        print(f'Saved: {os.path.join(results_dir, filename + ".png")}')

        csv_path = os.path.join(results_dir, filename + '.csv')
        pd.DataFrame(cum_series).to_csv(csv_path)
        print(f'Saved: {csv_path}')


def plot_intraday_profile_comparison(
    summaries_by_instrument: dict,
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    '''
    Save two charts comparing all instruments at each 30-min interval:
    one for avg volume, one for cumulative return.
    '''
    os.makedirs(results_dir, exist_ok=True)
    colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    instruments = list(summaries_by_instrument.keys())
    index       = list(summaries_by_instrument[instruments[0]].index)
    n           = len(index)
    bar_width   = 0.8 / len(instruments)
    x           = range(n)

    for metric, ylabel, filename, save_csv in [
        ('avg_volume',          'Avg Volume (contracts)',       'volume_profile_comparison',            False),
        ('avg_volume_pct',      'Avg Volume (% of daily total)','volume_pct_profile_comparison',        True),
        ('cum_return',          'Cumulative Return (%)',        'cum_return_profile_comparison',        True),
        ('avg_return',          'Avg Return (%)',               'avg_return_profile_comparison',        True),
    ]:
        _, ax = plt.subplots(figsize=(18, 6))
        csv_data = {}
        for i, (instrument, color) in enumerate(zip(instruments, colors)):
            summary = summaries_by_instrument[instrument]
            if metric == 'avg_volume_pct':
                values = summary['avg_volume'] / summary['avg_volume'].sum() * 100
            else:
                values = summary[metric]
            if metric in ('cum_return', 'avg_return'):
                values = values * 100
            csv_data[instrument] = values
            offsets = [xi + i * bar_width - (len(instruments) - 1) * bar_width / 2 for xi in x]
            ax.bar(offsets, values, width=bar_width, label=instrument, color=color, alpha=0.8)
        hour_ticks = [i for i, t in enumerate(index) if t.endswith(':00')]
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels([index[i] for i in hour_ticks], rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (NY)')
        ax.set_title(f'{ylabel} by 30-Min Interval')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename + '.png'), dpi=150)
        plt.close()
        print(f'Saved: {os.path.join(results_dir, filename + ".png")}')

        if save_csv:
            csv_path = os.path.join(results_dir, filename + '.csv')
            pd.DataFrame(csv_data, index=index).to_csv(csv_path)
            print(f'Saved: {csv_path}')


def _split_session_volume(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    '''Split avg_volume into intraday (09:30–16:00) and overnight buckets, each as % of its session total.'''
    intraday_mask = (summary.index >= '09:30') & (summary.index < '16:00')
    intraday_vol  = summary.loc[intraday_mask,  'avg_volume']
    overnight_vol = summary.loc[~intraday_mask, 'avg_volume']
    return intraday_vol / intraday_vol.sum() * 100, overnight_vol / overnight_vol.sum() * 100


def plot_session_volume_profiles(
    summary: pd.DataFrame,
    instrument: str,
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    '''Save intraday and overnight volume % charts for a single instrument.'''
    os.makedirs(results_dir, exist_ok=True)
    intraday_vol, overnight_vol = _split_session_volume(summary)

    for vol, label, filename in [
        (intraday_vol,  f'{instrument} Intraday Volume (% of session)',  'volume_intraday_pct'),
        (overnight_vol, f'{instrument} Overnight Volume (% of session)', 'volume_overnight_pct'),
    ]:
        _, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(vol)), vol.values, color='#3498db', alpha=0.8)
        ax.set_xticks(range(len(vol)))
        ax.set_xticklabels(vol.index, rotation=45)
        ax.set_ylabel('% of Session Volume')
        ax.set_xlabel('Time (NY)')
        ax.set_title(label)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename + '.png'), dpi=150)
        plt.close()
        print(f'Saved: {os.path.join(results_dir, filename + ".png")}')


def plot_session_volume_profiles_comparison(
    summaries_by_instrument: dict,
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    '''Save intraday and overnight volume % comparison charts across all instruments.'''
    os.makedirs(results_dir, exist_ok=True)
    colors      = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    instruments = list(summaries_by_instrument.keys())
    bar_width   = 0.8 / len(instruments)

    for session, label, filename in [
        ('intraday',  'Intraday Volume (% of session) by Instrument',  'volume_intraday_pct_comparison'),
        ('overnight', 'Overnight Volume (% of session) by Instrument', 'volume_overnight_pct_comparison'),
    ]:
        vols = {}
        for instrument in instruments:
            intraday_vol, overnight_vol = _split_session_volume(summaries_by_instrument[instrument])
            vols[instrument] = intraday_vol if session == 'intraday' else overnight_vol

        index = list(vols[instruments[0]].index)
        x     = range(len(index))
        _, ax = plt.subplots(figsize=(18, 5))
        for i, (instrument, color) in enumerate(zip(instruments, colors)):
            offsets = [xi + i * bar_width - (len(instruments) - 1) * bar_width / 2 for xi in x]
            ax.bar(offsets, vols[instrument].values, width=bar_width, label=instrument, color=color, alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(index, rotation=45)
        ax.set_ylabel('% of Session Volume')
        ax.set_xlabel('Time (NY)')
        ax.set_title(label)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename + '.png'), dpi=150)
        plt.close()
        print(f'Saved: {os.path.join(results_dir, filename + ".png")}')


def plot_session_profile_comparison(
    summaries_by_instrument: dict,
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    '''
    Save three 2-panel charts comparing all instruments:
    full day, intraday session (09:30–16:00), and overnight.
    Top panel: grouped avg return bars + cumulative return lines (twin axis).
    Bottom panel: grouped avg volume % bars + cumulative volume % lines (twin axis).
    '''
    os.makedirs(results_dir, exist_ok=True)
    colors      = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    instruments = list(summaries_by_instrument.keys())
    bar_width   = 0.8 / len(instruments)

    sessions = [
        ('Full Day (24h)',                 None,        'session_profile_comparison_fullday'),
        ('Intraday Session (09:30–16:00)', 'intraday',  'session_profile_comparison_intraday'),
        ('Overnight Session',              'overnight', 'session_profile_comparison_overnight'),
    ]

    for session_label, session_key, filename in sessions:
        slices = {}
        for instrument in instruments:
            summary = summaries_by_instrument[instrument]
            if session_key is None:
                slices[instrument] = summary
            elif session_key == 'intraday':
                mask = (summary.index >= '09:30') & (summary.index < '16:00')
                slices[instrument] = summary[mask]
            else:
                mask = ~((summary.index >= '09:30') & (summary.index < '16:00'))
                slices[instrument] = summary[mask]

        sess_index    = list(slices[instruments[0]].index)
        n             = len(sess_index)
        x             = range(n)
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

        for i, (instrument, color) in enumerate(zip(instruments, colors)):
            sess    = slices[instrument]
            offsets = [xi + i * bar_width - (len(instruments) - 1) * bar_width / 2 for xi in x]
            cum_ret = sess['cum_return'] * 100
            vol_pct = sess['avg_volume'] / sess['avg_volume'].sum() * 100

            ax1.bar(offsets, cum_ret.values, width=bar_width, color=color, alpha=0.75, label=instrument)
            ax2.bar(offsets, vol_pct.values, width=bar_width, color=color, alpha=0.75, label=instrument)

        hour_ticks = [i for i, t in enumerate(sess_index) if t.endswith(':00')]
        tick_labels  = [sess_index[i] for i in hour_ticks]

        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.set_title(f'{session_label} Profile — All Instruments')
        ax1.legend(loc='upper left')
        ax1.set_xticks(hour_ticks)
        ax1.set_xticklabels(tick_labels, rotation=45)

        ax2.set_ylabel('% Cumulative Volume')
        ax2.set_xlabel('Time (NY)')
        ax2.legend(loc='upper left')
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels(tick_labels, rotation=45)

        plt.tight_layout()
        path = os.path.join(results_dir, filename + '.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f'Saved: {path}')


def plot_session_profiles_for(
    instruments: list[str],
    start_date: str,
    end_date: str,
    freq: str = '30min',
    results_dir: str = _BASE_RESULTS_DIR,
) -> None:
    '''
    Convenience function for use in notebooks.
    Builds instrument data and plots the session profile comparison chart.
    '''
    data = build_instrument_data(instruments, start_date, end_date, freq)
    plot_session_profile_comparison({k: v['summary'] for k, v in data.items()}, results_dir)


def main() -> None:

    overnight_by_instrument = {}
    intraday_by_instrument  = {}
    summaries_by_instrument = {}

    for instrument in INSTRUMENTS + FX_INSTRUMENTS:
        print(f'\n── {instrument} ──────────────────────────────────────')
        results_dir = os.path.join(_BASE_RESULTS_DIR, instrument)

        ohlcv        = load_ohlcv(START_DATE, END_DATE, [instrument], verbose=True)
        front_months = get_front_month_daily(ohlcv, verbose=True)
        continuous   = build_continuous_series(ohlcv, front_months)

        bars      = get_ohlcv_resampled(continuous, freq='30min')
        bars      = compute_bar_returns(bars)
        vol       = get_profile_volume(continuous, freq='30min')
        summary   = get_profile_summary(bars, vol)
        vol_accel = get_volume_acceleration_profile(continuous, freq='30min')
        plot_intraday_profile(summary, bars, results_dir, instrument)
        plot_intraday_profile_vol_accel(summary, vol_accel, bars, results_dir, instrument)
        plot_session_profiles_vol_accel(summary, vol_accel, results_dir, instrument)
        plot_session_profiles(summary, results_dir, instrument)
        plot_session_volume_profiles(summary, instrument, results_dir)

        intraday_sessions, _ = get_all_sessions(continuous)
        intraday_ret, overnight_ret = get_session_returns(intraday_sessions)
        plot_session_return_distributions(intraday_ret, overnight_ret, results_dir)
        plot_cumulative_returns(intraday_ret, overnight_ret, results_dir)

        overnight_by_instrument[instrument] = overnight_ret
        intraday_by_instrument[instrument]  = intraday_ret
        summaries_by_instrument[instrument] = summary

    print('\n── Comparison charts ─────────────────────────────────')
    plot_cumulative_returns_comparison(overnight_by_instrument, intraday_by_instrument)
    plot_intraday_profile_comparison(summaries_by_instrument)
    plot_session_volume_profiles_comparison(summaries_by_instrument)
    plot_session_profile_comparison(summaries_by_instrument)


if __name__ == '__main__':
    main()
