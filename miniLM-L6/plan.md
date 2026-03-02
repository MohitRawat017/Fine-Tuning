You are a dataset generator for intent classification. Your job is to generate diverse, realistic training examples for a personal voice assistant named Tsuzi.

The assistant supports these intent categories:

- casual:
  greetings, small talk, short replies, expressions
  Examples: "hello", "thanks", "yea", "ok", "good morning", "how are you", "never mind"

- productivity:
  alarms, timers, tasks, calendar events
  Examples: "set alarm", "start timer", "add task", "show my tasks", "create meeting"

  Supported actions:
  - set_timer
  - set_alarm
  - create_calendar_event
  - add_task
  - get_tasks

- system:
  opening applications, terminal commands, system info
  Examples: "open vscode", "run this command", "execute pip install", "system info", "cpu usage"

  Supported actions:
  - open_app
  - run_command
  - get_system_info

- research:
  searching the web, programming help, academic papers
  Examples: "search for", "find papers on", "how do I fix this error", "stackoverflow results"

  Supported actions:
  - web_search
  - search_stackoverflow
  - search_arxiv

- communication:
  sending or reading emails
  Examples: "send email to", "check my emails", "read latest mail", "reply to"

  Supported actions:
  - send_email
  - read_emails


Generate exactly 80 examples per category (400 total).

STRICT RULES:

1. Vary phrasing heavily — same intent must have 80 DIFFERENT wordings

2. Include typos and informal language
   Examples:
   - "set alram 7"
   - "open vs code"
   - "send mail fast"

3. Include short inputs when realistic
   Examples:
   - "hello"
   - "thanks"
   - "tasks?"

4. Include ambiguous-sounding queries that still clearly belong to one category

5. Include Indian English phrasing
   Examples:
   - "kindly send email"
   - "do the needful"
   - "prepone meeting"
   - "revert back"

6. Use natural assistant-style language — not robotic commands.

7. Do NOT include tool names in queries
   ❌ "set_alarm"
   ❌ "run_command"

8. Queries must match supported capabilities only.
   Do NOT include:
   - weather
   - music
   - habits
   - reminders
   - screenshots
   - volume control
   - file management

9. Avoid duplicates or near-duplicates.

10. Return ONLY a JSON array.
No explanation.
No markdown fences.


Format — each item must have exactly these fields:

{
  "text": "the user query",
  "label": "category_name",
  "label_id": 0
}


Label IDs:

casual = 0
productivity = 1
system = 2
research = 3
communication = 4


Generate all 400 examples now.