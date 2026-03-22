"""
Fix functiongemma_dataset.jsonl to match actual Pydantic tool schemas.

Mismatches found:
  - send_email:  trained on (recipient, subject?, message?) → actual is (subject, body)
  - read_emails: trained on ()                             → actual is (count?, filter_type?)
  - create_calendar_event: missing duration optional
  - search_arxiv:          missing max_results optional

This script rewrites the developer content (tool schema lines) and
remaps send_email assistant-output arg names.
"""

import json

INPUT  = "functiongemma_dataset.jsonl"
OUTPUT = "functiongemma_dataset_fixed.jsonl"

# ── Correct schemas (must match get_tool_schemas() output) ──────────────

CORRECT_SCHEMAS = {
    "productivity": (
        "set_alarm(time required, label optional)\n"
        "set_timer(duration required, label optional)\n"
        "add_task(text required)\n"
        "get_tasks()\n"
        "create_calendar_event(title required, date optional, time optional, duration optional)"
    ),
    "system": (
        "open_app(app_name required)\n"
        "run_command(command required)\n"
        "get_system_info()"
    ),
    "research": (
        "web_search(query required)\n"
        "search_stackoverflow(query required)\n"
        "search_arxiv(query required, max_results optional)"
    ),
    "communication": (
        "send_email(subject required, body required)\n"
        "read_emails(count optional, filter_type optional)"
    ),
}

DEVELOPER_PREFIX = (
    "You are a function-calling model. "
    "Select exactly one tool from the available tools and return only valid JSON.\n\n"
    "Available tools:\n"
)

# ── Helpers ─────────────────────────────────────────────────────────────

def detect_category(developer_content: str) -> str:
    """Detect which category this example belongs to."""
    if "set_alarm" in developer_content:
        return "productivity"
    if "open_app" in developer_content:
        return "system"
    if "web_search" in developer_content:
        return "research"
    if "send_email" in developer_content:
        return "communication"
    return "unknown"


def fix_developer_content(category: str) -> str:
    """Return corrected developer content for this category."""
    return DEVELOPER_PREFIX + CORRECT_SCHEMAS[category]


def fix_send_email_args(old_args: dict) -> dict:
    """Remap send_email args: (recipient, subject?, message?) → (subject, body)."""
    new_args = {}

    # subject: prefer old "subject", fall back to old "recipient"
    if "subject" in old_args and old_args["subject"]:
        new_args["subject"] = old_args["subject"]
    elif "recipient" in old_args and old_args["recipient"]:
        new_args["subject"] = old_args["recipient"]

    # body: use old "message" if present
    if "message" in old_args and old_args["message"]:
        new_args["body"] = old_args["message"]

    return new_args


# ── Main ────────────────────────────────────────────────────────────────

def main():
    fixed_lines = []
    stats = {"total": 0, "developer_fixed": 0, "args_fixed": 0}

    with open(INPUT, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            msgs = obj["messages"]
            stats["total"] += 1

            # ── Fix developer content ──
            dev_msg = msgs[0]
            category = detect_category(dev_msg["content"])

            if category == "unknown":
                fixed_lines.append(json.dumps(obj, ensure_ascii=False))
                continue

            new_dev = fix_developer_content(category)
            if dev_msg["content"] != new_dev:
                dev_msg["content"] = new_dev
                stats["developer_fixed"] += 1

            # ── Fix send_email assistant args ──
            assistant_msg = msgs[-1]
            parsed = json.loads(assistant_msg["content"])

            if parsed["tool"] == "send_email":
                old_args = parsed.get("args", {})
                new_args = fix_send_email_args(old_args)
                if old_args != new_args:
                    parsed["args"] = new_args
                    assistant_msg["content"] = json.dumps(parsed, ensure_ascii=False)
                    stats["args_fixed"] += 1

            fixed_lines.append(json.dumps(obj, ensure_ascii=False))

    with open(OUTPUT, "w", encoding="utf-8") as fh:
        for line in fixed_lines:
            fh.write(line + "\n")

    print(f"Done.  {stats['total']} examples processed.")
    print(f"  Developer content fixed: {stats['developer_fixed']}")
    print(f"  send_email args remapped: {stats['args_fixed']}")
    print(f"  Output: {OUTPUT}")


if __name__ == "__main__":
    main()
