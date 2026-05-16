#!/usr/bin/env python3
"""
MIND-X Portfolio Analyzer
Weekly AI stock picker — Claude Haiku + yfinance (prices + news + fundamentals)
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import anthropic
import yfinance as yf

PORTFOLIO_PATH = os.path.join(os.path.dirname(__file__), "..", "portfolio.json")
BANGKOK = ZoneInfo("Asia/Bangkok")

CANDIDATE_UNIVERSE = {
    "growth": [
        "NVDA", "META", "GOOGL", "AMZN", "TSLA",
        "CRM", "SNOW", "PLTR", "NET", "DDOG",
        "CRWD", "ZS", "MDB", "BILL", "TTD"
    ],
    "quality": [
        "AAPL", "MSFT", "V", "MA", "UNH",
        "LLY", "ABBV", "JPM", "BRK-B", "PG",
        "JNJ", "HD", "COST", "TMO", "AVGO"
    ],
    "value": [
        "BAC", "WFC", "C", "USB", "CVX",
        "XOM", "GM", "F", "VZ", "T",
        "MO", "PM", "KO", "PEP", "BEN"
    ]
}

# Top candidates per category to fetch news+fundamentals for (saves API time)
TOP_PER_CATEGORY = 6


def load_portfolio() -> dict:
    with open(PORTFOLIO_PATH, "r") as f:
        return json.load(f)


def save_portfolio(portfolio: dict):
    portfolio["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2)


def get_prices(tickers: list[str]) -> dict:
    if not tickers:
        return {}
    data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
    prices = {}
    close = data["Close"] if "Close" in data else data
    for t in tickers:
        try:
            if len(tickers) == 1:
                prices[t] = round(float(close.dropna().iloc[-1]), 2)
            else:
                prices[t] = round(float(close[t].dropna().iloc[-1]), 2)
        except Exception:
            prices[t] = None
    return prices


def get_sp500_price() -> float | None:
    for symbol in ("^GSPC", "SPY"):
        try:
            data = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
            if not data.empty:
                return round(float(data["Close"].dropna().iloc[-1]), 2)
        except Exception:
            continue
    return None


def get_news(tickers: list[str], max_per_ticker: int = 3) -> dict[str, list[str]]:
    """Fetch recent news headlines for each ticker via yfinance."""
    result = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            news_items = ticker_obj.news or []
            headlines = []
            for item in news_items[:max_per_ticker]:
                title = item.get("title", "")
                if title:
                    headlines.append(title)
            result[t] = headlines
        except Exception:
            result[t] = []
    return result


def get_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """Fetch key fundamentals: PE, revenue growth, EPS, 52w range."""
    result = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            result[t] = {
                "pe":             info.get("forwardPE") or info.get("trailingPE"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "eps":            info.get("trailingEps"),
                "52w_high":       info.get("fiftyTwoWeekHigh"),
                "52w_low":        info.get("fiftyTwoWeekLow"),
                "sector":         info.get("sector", ""),
            }
        except Exception:
            result[t] = {}
    return result


def fmt_fundamentals(ticker: str, fund: dict, price: float | None) -> str:
    parts = []
    if fund.get("pe"):
        parts.append(f"PE={fund['pe']:.1f}")
    if fund.get("revenue_growth") is not None:
        parts.append(f"RevGrowth={fund['revenue_growth']*100:+.1f}%")
    if fund.get("earnings_growth") is not None:
        parts.append(f"EPS_growth={fund['earnings_growth']*100:+.1f}%")
    if fund.get("52w_high") and fund.get("52w_low") and price:
        pct_from_high = (price - fund["52w_high"]) / fund["52w_high"] * 100
        parts.append(f"52w_from_high={pct_from_high:+.1f}%")
    return " | ".join(parts) if parts else "N/A"


def compute_total_value(portfolio: dict, current_prices: dict) -> float:
    total = portfolio["cash"]
    for pos in portfolio["positions"]:
        price = current_prices.get(pos["ticker"])
        if price:
            total += price * pos["shares"]
        else:
            total += pos["avg_cost"] * pos["shares"]
    return round(total, 2)


def build_prompt(portfolio: dict, prices: dict,
                 news: dict[str, list[str]],
                 fundamentals: dict[str, dict]) -> str:
    allocations = portfolio["allocations"]
    total_value = compute_total_value(portfolio, prices)
    target_growth  = round(total_value * allocations["growth"], 2)
    target_quality = round(total_value * allocations["quality"], 2)
    target_value   = round(total_value * allocations["value"], 2)
    target_cash    = round(total_value * allocations["cash"], 2)

    # Current positions
    positions_text = ""
    if portfolio["positions"]:
        for pos in portfolio["positions"]:
            cp = prices.get(pos["ticker"], pos["avg_cost"])
            pnl = round((cp - pos["avg_cost"]) / pos["avg_cost"] * 100, 1) if cp else 0
            fund_str = fmt_fundamentals(pos["ticker"], fundamentals.get(pos["ticker"], {}), cp)
            positions_text += (
                f"  {pos['ticker']} ({pos['category']}): "
                f"{pos['shares']} shares @ avg ${pos['avg_cost']} | "
                f"current ${cp} | P&L {pnl:+.1f}% | {fund_str}\n"
            )
            headlines = news.get(pos["ticker"], [])
            for h in headlines:
                positions_text += f"    • {h}\n"
    else:
        positions_text = "  (empty — first run)"

    # Candidates with price + fundamentals + news
    candidates_text = ""
    for cat, tickers in CANDIDATE_UNIVERSE.items():
        candidates_text += f"\n{cat.upper()} candidates:\n"
        for t in tickers[:TOP_PER_CATEGORY]:
            p = prices.get(t)
            price_str = f"${p}" if p else "N/A"
            fund_str = fmt_fundamentals(t, fundamentals.get(t, {}), p)
            candidates_text += f"  {t}: {price_str} | {fund_str}\n"
            for h in news.get(t, [])[:2]:
                candidates_text += f"    • {h}\n"

    return f"""You are an AI portfolio manager for a paper trading simulation.
Portfolio value: ${total_value:,.2f} | Cash: ${portfolio['cash']:,.2f}
Max positions: {portfolio['max_positions']}

TARGET ALLOCATIONS:
- Growth 50% → ${target_growth:,.2f}
- Quality 30% → ${target_quality:,.2f}
- Value 10%   → ${target_value:,.2f}
- Cash 10%    → ${target_cash:,.2f}

CURRENT POSITIONS (with fundamentals + recent news):
{positions_text}
CANDIDATE STOCKS (price | fundamentals | recent headlines):
{candidates_text}

TASK: Decide this week's portfolio actions based on fundamentals, news, and target allocations.
Rules:
1. Max {portfolio['max_positions']} total positions
2. Buy whole shares only (no fractional)
3. Keep cash ≥ ${target_cash:,.2f}
4. Rebalance if any position drifts >15% from target weight
5. Favor stocks with positive news catalysts and strong fundamentals

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
  "actions": [
    {{
      "action": "BUY" | "SELL" | "HOLD",
      "ticker": "SYMBOL",
      "shares": <integer>,
      "category": "growth" | "quality" | "value",
      "price": <float>,
      "reasoning": "<1-2 sentences citing news or fundamentals>"
    }}
  ],
  "weekly_outlook": "<2-3 sentences market view based on news>",
  "cash_after": <float>
}}"""


def apply_actions(portfolio: dict, actions: list[dict], prices: dict) -> list[dict]:
    executed = []
    for act in actions:
        ticker   = act.get("ticker", "").upper()
        action   = act.get("action", "").upper()
        shares   = int(act.get("shares", 0))
        category = act.get("category", "growth")
        price    = prices.get(ticker) or act.get("price", 0)
        reasoning = act.get("reasoning", "")

        if action == "BUY" and shares > 0 and price:
            cost = round(price * shares, 2)
            if cost > portfolio["cash"]:
                shares = int(portfolio["cash"] // price)
                cost = round(price * shares, 2)
            if shares <= 0:
                continue

            existing = next((p for p in portfolio["positions"] if p["ticker"] == ticker), None)
            if existing:
                total_shares = existing["shares"] + shares
                existing["avg_cost"] = round(
                    (existing["avg_cost"] * existing["shares"] + price * shares) / total_shares, 2
                )
                existing["shares"] = total_shares
            else:
                if len(portfolio["positions"]) >= portfolio["max_positions"]:
                    continue
                portfolio["positions"].append({
                    "ticker": ticker,
                    "shares": shares,
                    "avg_cost": round(price, 2),
                    "category": category,
                    "bought_at": datetime.now(BANGKOK).strftime("%Y-%m-%d"),
                    "reasoning": reasoning
                })
            portfolio["cash"] = round(portfolio["cash"] - cost, 2)
            executed.append({**act, "executed_price": price, "executed_shares": shares})

        elif action == "SELL" and shares > 0:
            existing = next((p for p in portfolio["positions"] if p["ticker"] == ticker), None)
            if not existing:
                continue
            sell_shares = min(shares, existing["shares"])
            proceeds = round(price * sell_shares, 2)
            existing["shares"] -= sell_shares
            if existing["shares"] == 0:
                portfolio["positions"] = [p for p in portfolio["positions"] if p["ticker"] != ticker]
            portfolio["cash"] = round(portfolio["cash"] + proceeds, 2)
            executed.append({**act, "executed_price": price, "executed_shares": sell_shares})

        elif action == "HOLD":
            executed.append({**act, "executed_price": price, "executed_shares": 0})

    return executed


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print("Loading portfolio...")
    portfolio = load_portfolio()

    # Tickers to fetch: current positions + top candidates
    position_tickers = [pos["ticker"] for pos in portfolio["positions"]]
    candidate_tickers = []
    for tickers in CANDIDATE_UNIVERSE.values():
        candidate_tickers.extend(tickers)
    all_tickers = list(set(position_tickers + candidate_tickers))

    print(f"Fetching prices for {len(all_tickers)} tickers...")
    prices = get_prices(all_tickers)
    sp500_price = get_sp500_price()
    print(f"S&P500: ${sp500_price}")

    # Fetch news + fundamentals for positions + top candidates only
    news_tickers = list(set(
        position_tickers +
        [t for tickers in CANDIDATE_UNIVERSE.values() for t in tickers[:TOP_PER_CATEGORY]]
    ))
    print(f"Fetching news for {len(news_tickers)} tickers...")
    news = get_news(news_tickers, max_per_ticker=3)

    print(f"Fetching fundamentals for {len(news_tickers)} tickers...")
    fundamentals = get_fundamentals(news_tickers)

    print("Asking Claude Haiku for analysis...")
    prompt = build_prompt(portfolio, prices, news, fundamentals)
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    print(f"Claude response:\n{raw}\n")

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            decision = json.loads(match.group())
        else:
            print("ERROR: Could not parse Claude response as JSON")
            sys.exit(1)

    actions  = decision.get("actions", [])
    outlook  = decision.get("weekly_outlook", "")

    print(f"Executing {len(actions)} actions...")
    executed = apply_actions(portfolio, actions, prices)

    total_value = compute_total_value(portfolio, prices)

    history_entry = {
        "date": datetime.now(BANGKOK).strftime("%Y-%m-%d"),
        "total_value": total_value,
        "sp500_price": sp500_price,
        "cash": portfolio["cash"],
        "positions_count": len(portfolio["positions"]),
        "weekly_outlook": outlook,
        "actions": executed
    }

    first_sp500 = next((h["sp500_price"] for h in portfolio["history"] if h.get("sp500_price")), None)
    if sp500_price and first_sp500:
        history_entry["sp500_baseline"] = round(10000 * sp500_price / first_sp500, 2)
    else:
        history_entry["sp500_baseline"] = 10000.0

    portfolio["history"].append(history_entry)
    save_portfolio(portfolio)

    print(f"\nDone! Portfolio value: ${total_value:,.2f} | Cash: ${portfolio['cash']:,.2f}")
    print(f"Positions: {len(portfolio['positions'])}")
    for pos in portfolio["positions"]:
        p = prices.get(pos["ticker"], pos["avg_cost"])
        pnl = round((p - pos["avg_cost"]) / pos["avg_cost"] * 100, 1)
        print(f"  {pos['ticker']} ({pos['category']}): {pos['shares']} shares | P&L {pnl:+.1f}%")


if __name__ == "__main__":
    main()
