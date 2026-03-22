"""
Layered pipeline eval — practical version.

Strategy:
  - Layer 0+1 (pre_filter + MiniLM): run all 100 queries (~5ms each, ~1s total)
  - Layer 3 (FunctionGemma): spot-test 1 query per tool (12 queries) since CPU is ~30s each

This gives complete intent accuracy data + representative tool selection data.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path("d:/vs code/assistant")))

TEST_CASES = [
    # ── CASUAL (20) ─────────────────────────────────────────────────────
    ("hi", "casual", None),
    ("hello there", "casual", None),
    ("hey", "casual", None),
    ("good morning", "casual", None),
    ("good night", "casual", None),
    ("how are you", "casual", None),
    ("what's up", "casual", None),
    ("thanks", "casual", None),
    ("thank you so much", "casual", None),
    ("ok cool", "casual", None),
    ("sure", "casual", None),
    ("yeah alright", "casual", None),
    ("bye", "casual", None),
    ("see you later", "casual", None),
    ("lol", "casual", None),
    ("you're awesome", "casual", None),
    ("nice", "casual", None),
    ("perfect", "casual", None),
    ("no thanks", "casual", None),
    ("hmm ok", "casual", None),

    # ── PRODUCTIVITY (20) ────────────────────────────────────────────────
    ("set a timer for 5 minutes", "productivity", "set_timer"),
    ("start a 10 minute countdown", "productivity", "set_timer"),
    ("set my timer for 30 seconds", "productivity", "set_timer"),
    ("wake me up at 7am", "productivity", "set_alarm"),
    ("set an alarm for 6:30 AM", "productivity", "set_alarm"),
    ("set an alarm for 8pm tonight", "productivity", "set_alarm"),
    ("set a 45 minute study timer", "productivity", "set_timer"),
    ("add buy milk to my task list", "productivity", "add_task"),
    ("add finish report to my todos", "productivity", "add_task"),
    ("add walk the dog to my list", "productivity", "add_task"),
    ("add review emails to my tasks", "productivity", "add_task"),
    ("what tasks do I have today", "productivity", "get_tasks"),
    ("show me my to-do list", "productivity", "get_tasks"),
    ("what's on my schedule today", "productivity", "get_tasks"),
    ("schedule a meeting tomorrow at 3pm", "productivity", "create_calendar_event"),
    ("create a calendar event for Monday 9am", "productivity", "create_calendar_event"),
    ("book a dentist appointment next Friday", "productivity", "create_calendar_event"),
    ("remind me to call mom at 5pm", "productivity", "set_alarm"),
    ("create a meeting for next Tuesday at 2pm", "productivity", "create_calendar_event"),
    ("remind me in 2 hours", "productivity", "set_alarm"),

    # ── SYSTEM (20) ─────────────────────────────────────────────────────
    ("open chrome", "system", "open_app"),
    ("launch spotify", "system", "open_app"),
    ("open VS Code", "system", "open_app"),
    ("start discord", "system", "open_app"),
    ("open file explorer", "system", "open_app"),
    ("open notepad", "system", "open_app"),
    ("launch calculator", "system", "open_app"),
    ("open the browser", "system", "open_app"),
    ("open task manager", "system", "open_app"),
    ("start vlc", "system", "open_app"),
    ("run ipconfig", "system", "run_command"),
    ("run the command dir", "system", "run_command"),
    ("run ls in terminal", "system", "run_command"),
    ("what's my current directory", "system", "run_command"),
    ("run python --version", "system", "run_command"),
    ("what time is it right now", "system", "get_system_info"),
    ("what's today's date", "system", "get_system_info"),
    ("show my system status", "system", "get_system_info"),
    ("what is my IP address", "system", "run_command"),
    ("check disk space", "system", "run_command"),

    # ── RESEARCH (20) ───────────────────────────────────────────────────
    ("search for machine learning tutorials", "research", "web_search"),
    ("what is quantum computing", "research", "web_search"),
    ("who is Elon Musk", "research", "web_search"),
    ("latest news in AI", "research", "web_search"),
    ("weather in New Delhi today", "research", "web_search"),
    ("current bitcoin price", "research", "web_search"),
    ("who won the IPL 2024", "research", "web_search"),
    ("what is the capital of France", "research", "web_search"),
    ("latest news on OpenAI", "research", "web_search"),
    ("temperature in London right now", "research", "web_search"),
    ("find papers on diffusion models", "research", "search_arxiv"),
    ("search arxiv for transformer architecture", "research", "search_arxiv"),
    ("find research papers on LLMs", "research", "search_arxiv"),
    ("find papers about attention mechanisms", "research", "search_arxiv"),
    ("find recent papers on computer vision", "research", "search_arxiv"),
    ("how does BERT work", "research", "search_arxiv"),
    ("how to reverse a list in python", "research", "search_stackoverflow"),
    ("search stackoverflow for async await javascript", "research", "search_stackoverflow"),
    ("search stackoverflow for cuda out of memory error", "research", "search_stackoverflow"),
    ("search for Python best practices on stackoverflow", "research", "search_stackoverflow"),

    # ── COMMUNICATION (20) ──────────────────────────────────────────────
    ("send an email to my boss", "communication", "send_email"),
    ("send email with subject meeting notes", "communication", "send_email"),
    ("email the invoice to the client", "communication", "send_email"),
    ("compose an email about tomorrow's meeting", "communication", "send_email"),
    ("send a message to John about the project", "communication", "send_email"),
    ("send follow-up email to the team", "communication", "send_email"),
    ("send an email saying I'll be late", "communication", "send_email"),
    ("forward the report via email", "communication", "send_email"),
    ("send an email with the agenda", "communication", "send_email"),
    ("email my professor about the assignment", "communication", "send_email"),
    ("write an email to HR", "communication", "send_email"),
    ("send a thank you email", "communication", "send_email"),
    ("check my unread emails", "communication", "read_emails"),
    ("read my inbox", "communication", "read_emails"),
    ("what emails did I get today", "communication", "read_emails"),
    ("do I have any new emails", "communication", "read_emails"),
    ("read my latest emails", "communication", "read_emails"),
    ("check if there's a reply from Sarah", "communication", "read_emails"),
    ("show me my recent emails", "communication", "read_emails"),
    ("read my unread messages", "communication", "read_emails"),
]

# FunctionGemma spot-test: 1 representative query per expected tool (12 tools)
FG_SPOT_TEST = [
    ("set a timer for 5 minutes", "productivity", "set_timer"),
    ("wake me up at 7am", "productivity", "set_alarm"),
    ("add buy milk to my list", "productivity", "add_task"),
    ("show me my to-do list", "productivity", "get_tasks"),
    ("schedule a meeting tomorrow at 3pm", "productivity", "create_calendar_event"),
    ("open chrome", "system", "open_app"),
    ("run ipconfig", "system", "run_command"),
    ("what time is it", "system", "get_system_info"),
    ("search for latest AI news", "research", "web_search"),
    ("find papers on diffusion models", "research", "search_arxiv"),
    ("how to fix a null pointer exception", "research", "search_stackoverflow"),
    ("send an email to my boss", "communication", "send_email"),
    ("check my unread emails", "communication", "read_emails"),
]


def run_layer01():
    """Run all 100 test cases through pre_filter + MiniLM (fast)."""
    from src.tools.pre_filter import is_casual_query
    from src.tools.intent_router import predict_intent

    results = []
    print("\n── Layer 0 + 1: Pre-filter + MiniLM (100 queries) ──\n")

    for i, (query, exp_intent, exp_tool) in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        caught = is_casual_query(query)
        if caught:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            results.append({
                "id": i, "query": query,
                "expected_intent": exp_intent, "expected_tool": exp_tool,
                "prefilter_caught": True,
                "minilm_label": "casual", "minilm_confidence": 1.0,
                "intent_correct": (exp_intent == "casual"),
                "latency_ms": ms,
            })
            continue

        pred = predict_intent(query)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        results.append({
            "id": i, "query": query,
            "expected_intent": exp_intent, "expected_tool": exp_tool,
            "prefilter_caught": False,
            "minilm_label": pred["label"],
            "minilm_confidence": pred["confidence"],
            "intent_correct": (pred["label"] == exp_intent),
            "latency_ms": ms,
        })

    return results


def run_layer3_spot():
    """Spot-test FunctionGemma on 13 representative queries."""
    from src.tools.tools_by_category import get_tool_schemas
    from src.tools.tool_router import predict_tool

    print("\n── Layer 3: FunctionGemma spot-test (13 queries) ──\n")
    spot_results = []

    for query, category, exp_tool in FG_SPOT_TEST:
        schemas = get_tool_schemas(category)
        print(f"  Testing: {query!r} (expected: {exp_tool})... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = predict_tool(query, schemas)
            ms = round((time.perf_counter() - t0) * 1000, 2)
            got = result["tool"] if result else None
            correct = (got == exp_tool)
            print(f"got={got or 'None'}  {'✓' if correct else '✗'}  ({ms}ms)")
            spot_results.append({
                "query": query,
                "category": category,
                "expected_tool": exp_tool,
                "got_tool": got,
                "args": result.get("args") if result else None,
                "correct": correct,
                "latency_ms": ms,
            })
        except Exception as e:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            print(f"ERROR: {e}")
            spot_results.append({
                "query": query, "category": category,
                "expected_tool": exp_tool, "got_tool": f"ERROR:{str(e)[:60]}",
                "correct": False, "latency_ms": ms,
            })

    return spot_results


def build_summary(l01_results, fg_results):
    total = len(l01_results)
    intent_correct = sum(1 for r in l01_results if r["intent_correct"])
    prefilter_caught = sum(1 for r in l01_results if r["prefilter_caught"])

    by_cat = {}
    for r in l01_results:
        c = r["expected_intent"]
        if c not in by_cat:
            by_cat[c] = {"total": 0, "correct": 0, "confs": [], "latencies": [], "failures": []}
        by_cat[c]["total"] += 1
        by_cat[c]["latencies"].append(r["latency_ms"])
        if r["intent_correct"]:
            by_cat[c]["correct"] += 1
        else:
            by_cat[c]["failures"].append({
                "query": r["query"],
                "predicted": r["minilm_label"],
                "confidence": r["minilm_confidence"],
            })
        if not r["prefilter_caught"]:
            by_cat[c]["confs"].append(r["minilm_confidence"])

    category_stats = {}
    for cat, s in by_cat.items():
        category_stats[cat] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy_pct": round(s["correct"] / s["total"] * 100, 1),
            "avg_confidence": round(sum(s["confs"]) / len(s["confs"]), 4) if s["confs"] else 1.0,
            "min_confidence": round(min(s["confs"]), 4) if s["confs"] else 1.0,
            "max_confidence": round(max(s["confs"]), 4) if s["confs"] else 1.0,
            "avg_latency_ms": round(sum(s["latencies"]) / len(s["latencies"]), 2),
            "failures": s["failures"],
        }

    fg_correct = sum(1 for r in fg_results if r["correct"])

    return {
        "layer01": {
            "total": total,
            "intent_accuracy_pct": round(intent_correct / total * 100, 1),
            "prefilter_caught": prefilter_caught,
            "avg_latency_ms": round(sum(r["latency_ms"] for r in l01_results) / total, 2),
            "by_category": category_stats,
        },
        "layer3_spot": {
            "total": len(fg_results),
            "correct": fg_correct,
            "accuracy_pct": round(fg_correct / len(fg_results) * 100, 1) if fg_results else 0,
            "results": fg_results,
        }
    }


if __name__ == "__main__":
    l01 = run_layer01()
    fg = run_layer3_spot()
    summary = build_summary(l01, fg)

    out = {"summary": summary, "l01_results": l01}
    with open("d:/vs code/assistant/pipeline_eval.json", "w") as f:
        json.dump(out, f, indent=2)

    s = summary["layer01"]
    print(f"\n{'='*60}")
    print(f"  Layer 0+1 Intent accuracy: {s['intent_accuracy_pct']}%  ({sum(v['correct'] for v in s['by_category'].values())}/{s['total']})")
    print(f"  Pre-filter caught:         {s['prefilter_caught']} casual queries")
    for cat, cs in s["by_category"].items():
        print(f"  {cat:>15}: {cs['accuracy_pct']:5.1f}%  avg_conf={cs['avg_confidence']:.3f}")
    fg_s = summary["layer3_spot"]
    print(f"\n  FunctionGemma spot:        {fg_s['correct']}/{fg_s['total']} correct ({fg_s['accuracy_pct']}%)")
    print(f"\n  Results saved → pipeline_eval.json")
