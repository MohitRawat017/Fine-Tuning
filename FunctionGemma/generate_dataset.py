"""
FunctionGemma Fine-tuning Dataset Generator
Generates 520 examples (40 per tool) for function calling training.
"""

import json
import random

def generate_dataset(seed=42):
    random.seed(seed)
    
    TOOLS = {
        "productivity": {
            "set_alarm": {"time": "required", "label": "optional"},
            "set_timer": {"duration": "required", "label": "optional"},
            "add_task": {"text": "required"},
            "get_tasks": {},
            "create_calendar_event": {"title": "required", "date": "optional", "time": "optional"}
        },
        "system": {
            "open_app": {"app_name": "required"},
            "run_command": {"command": "required"},
            "get_system_info": {}
        },
        "research": {
            "web_search": {"query": "required"},
            "search_stackoverflow": {"query": "required"},
            "search_arxiv": {"query": "required"}
        },
        "communication": {
            "send_email": {"recipient": "required", "subject": "optional", "message": "optional"},
            "read_emails": {}
        }
    }
    
    def create_example(category, user_query, tool_name, args):
        available_tools = TOOLS[category]
        tool_descriptions = []
        for t_name, params in available_tools.items():
            if params:
                param_str = ", ".join([f"{k} {'optional' if v == 'optional' else 'required'}" for k, v in params.items()])
                tool_descriptions.append(f"{t_name}({param_str})")
            else:
                tool_descriptions.append(f"{t_name}()")
        
        tools_str = "\n".join(tool_descriptions)
        
        return {
            "messages": [
                {
                    "role": "developer",
                    "content": f"You are a function-calling model. Select exactly one tool from the available tools and return only valid JSON.\n\nAvailable tools:\n{tools_str}"
                },
                {
                    "role": "user",
                    "content": user_query
                },
                {
                    "role": "assistant",
                    "content": json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)
                }
            ]
        }
    
    dataset = []
    
    # ===== PRODUCTIVITY (200 examples) =====
    
    # set_alarm - 40 examples
    set_alarm_data = [
        ("wake me up at 7am", {"time": "7am"}),
        ("set an alarm for 6:30 tomorrow morning", {"time": "6:30", "label": "tomorrow morning"}),
        ("alarm at 8 please", {"time": "8"}),
        ("I need to wake up at 5:30", {"time": "5:30"}),
        ("set alarm for 9am with label work", {"time": "9am", "label": "work"}),
        ("morning alarm at 7", {"time": "7", "label": "morning"}),
        ("hey um set alarm for 6am", {"time": "6am"}),
        ("can you wake me at 8:15", {"time": "8:15"}),
        ("alarm me at noon", {"time": "noon"}),
        ("set my alarm to 7:45", {"time": "7:45"}),
        ("need alarm for 5am sharp", {"time": "5am"}),
        ("set alram 7:30", {"time": "7:30"}),
        ("wake me up at quarter past 8", {"time": "quarter past 8"}),
        ("alarm at 9:30 for meeting", {"time": "9:30", "label": "meeting"}),
        ("I have to get up by 6", {"time": "6"}),
        ("set alarm seven thirty am", {"time": "seven thirty am"}),
        ("early morning alarm at 5", {"time": "5", "label": "early morning"}),
        ("alarm for 10pm tonight", {"time": "10pm", "label": "tonight"}),
        ("wake me at 7am tomorrow", {"time": "7am", "label": "tomorrow"}),
        ("set alarm 8 o clock", {"time": "8 o clock"}),
        ("alarm at half past 6", {"time": "half past 6"}),
        ("need to wake up at 4:30", {"time": "4:30"}),
        ("set alarm for 11am labeled lunch", {"time": "11am", "label": "lunch"}),
        ("wake me 7 sharp", {"time": "7"}),
        ("alarm 6am please", {"time": "6am"}),
        ("set an alarm at 8:45 for doctor", {"time": "8:45", "label": "doctor"}),
        ("I want alarm at 9", {"time": "9"}),
        ("alarm for early tomorrow 5am", {"time": "5am", "label": "early tomorrow"}),
        ("set alarm nine fifteen", {"time": "nine fifteen"}),
        ("wake me up 6:30", {"time": "6:30"}),
        ("alarm at 7am work", {"time": "7am", "label": "work"}),
        ("set alarm 5:45 am", {"time": "5:45 am"}),
        ("need alarm 8:30", {"time": "8:30"}),
        ("alarm me at 10 for gym", {"time": "10", "label": "gym"}),
        ("set morning alarm 7", {"time": "7", "label": "morning"}),
        ("wake up call at 6am", {"time": "6am"}),
        ("alarm set for 9:00", {"time": "9:00"}),
        ("set alarm 7:15 please", {"time": "7:15"}),
        ("alarm at 5:30 labeled run", {"time": "5:30", "label": "run"}),
        ("kindly set alarm for 8am", {"time": "8am"}),
    ]
    for query, args in set_alarm_data:
        dataset.append(create_example("productivity", query, "set_alarm", args))
    
    # set_timer - 40 examples
    set_timer_data = [
        ("set a timer for 5 minutes", {"duration": "5 minutes"}),
        ("start 10 minute timer", {"duration": "10 minute"}),
        ("timer for 15 minutes labeled pasta", {"duration": "15 minutes", "label": "pasta"}),
        ("countdown 20 minutes", {"duration": "20 minutes"}),
        ("start timer 25 minutes", {"duration": "25 minutes"}),
        ("I need a 30 minute timer", {"duration": "30 minute"}),
        ("timer 45 minutes", {"duration": "45 minutes"}),
        ("set timer for 1 hour", {"duration": "1 hour"}),
        ("2 minute timer please", {"duration": "2 minute"}),
        ("start timing 3 minutes", {"duration": "3 minutes"}),
        ("timer for 10 mins", {"duration": "10 mins"}),
        ("count me down 5 minutes", {"duration": "5 minutes"}),
        ("begin 15 minute countdown", {"duration": "15 minute"}),
        ("set timer to 20 minutes labeled study", {"duration": "20 minutes", "label": "study"}),
        ("timer 30 min", {"duration": "30 min"}),
        ("start a timer for 40 minutes", {"duration": "40 minutes"}),
        ("timer 1 hour 30 minutes", {"duration": "1 hour 30 minutes"}),
        ("set timmer 10 minutes", {"duration": "10 minutes"}),
        ("timer for cooking 20 min", {"duration": "20 min", "label": "cooking"}),
        ("countdown from 45 minutes", {"duration": "45 minutes"}),
        ("start 5 min timer", {"duration": "5 min"}),
        ("set timer 12 minutes", {"duration": "12 minutes"}),
        ("timer for 8 minutes labeled break", {"duration": "8 minutes", "label": "break"}),
        ("begin timing 50 minutes", {"duration": "50 minutes"}),
        ("timer uh 15 minutes", {"duration": "15 minutes"}),
        ("set a 25 minute pomodoro", {"duration": "25 minute", "label": "pomodoro"}),
        ("start timer for 35 minutes", {"duration": "35 minutes"}),
        ("timer 6 minutes please", {"duration": "6 minutes"}),
        ("countdown timer 18 minutes", {"duration": "18 minutes"}),
        ("set timer 2 hours", {"duration": "2 hours"}),
        ("timer for eggs 7 minutes", {"duration": "7 minutes", "label": "eggs"}),
        ("start 4 minute timer", {"duration": "4 minute"}),
        ("set timer 90 minutes", {"duration": "90 minutes"}),
        ("timer for workout 30 min", {"duration": "30 min", "label": "workout"}),
        ("begin 10 minute timer", {"duration": "10 minute"}),
        ("set timer 1 hour labeled meeting", {"duration": "1 hour", "label": "meeting"}),
        ("timer 3 minutes", {"duration": "3 minutes"}),
        ("start countdown 22 minutes", {"duration": "22 minutes"}),
        ("set timer for 14 minutes", {"duration": "14 minutes"}),
        ("kindly set timer 10 minutes", {"duration": "10 minutes"}),
    ]
    for query, args in set_timer_data:
        dataset.append(create_example("productivity", query, "set_timer", args))
    
    # add_task - 40 examples
    add_task_data = [
        ("add task buy groceries", {"text": "buy groceries"}),
        ("new task call mom", {"text": "call mom"}),
        ("remind me to finish the report", {"text": "finish the report"}),
        ("task review the code", {"text": "review the code"}),
        ("add to my tasks email the client", {"text": "email the client"}),
        ("create task submit assignment", {"text": "submit assignment"}),
        ("I need to do laundry today", {"text": "do laundry today"}),
        ("task pay electricity bill", {"text": "pay electricity bill"}),
        ("add a task called workout", {"text": "workout"}),
        ("put this on my list book flight", {"text": "book flight"}),
        ("new todo buy milk", {"text": "buy milk"}),
        ("add task fix the bug in login", {"text": "fix the bug in login"}),
        ("task for me read documentation", {"text": "read documentation"}),
        ("I have to call the dentist", {"text": "call the dentist"}),
        ("add this task write blog post", {"text": "write blog post"}),
        ("create todo water the plants", {"text": "water the plants"}),
        ("add taks clean room", {"text": "clean room"}),
        ("new task schedule doctor appointment", {"text": "schedule doctor appointment"}),
        ("remind me to buy birthday gift", {"text": "buy birthday gift"}),
        ("task renew car insurance", {"text": "renew car insurance"}),
        ("add task prepare presentation", {"text": "prepare presentation"}),
        ("todo pick up dry cleaning", {"text": "pick up dry cleaning"}),
        ("new task update resume", {"text": "update resume"}),
        ("add to list organize closet", {"text": "organize closet"}),
        ("task send invoice to client", {"text": "send invoice to client"}),
        ("remind me to call bank", {"text": "call bank"}),
        ("add task research new laptop", {"text": "research new laptop"}),
        ("new todo plan weekend trip", {"text": "plan weekend trip"}),
        ("task backup important files", {"text": "backup important files"}),
        ("add this return library books", {"text": "return library books"}),
        ("create task meal prep for week", {"text": "meal prep for week"}),
        ("todo cancel subscription", {"text": "cancel subscription"}),
        ("add task reply to john's email", {"text": "reply to john's email"}),
        ("new task practice guitar", {"text": "practice guitar"}),
        ("remind me to get oil changed", {"text": "get oil changed"}),
        ("task update software", {"text": "update software"}),
        ("add to my list buy protein", {"text": "buy protein"}),
        ("create todo meditate 10 minutes", {"text": "meditate 10 minutes"}),
        ("new task review pull request", {"text": "review pull request"}),
        ("kindly add task prepare dinner", {"text": "prepare dinner"}),
    ]
    for query, args in add_task_data:
        dataset.append(create_example("productivity", query, "add_task", args))
    
    # get_tasks - 40 examples (no args)
    get_tasks_data = [
        "show my tasks", "what are my todos", "list my tasks please",
        "show me what I need to do", "what's on my task list", "display my todos",
        "tell me my pending tasks", "what tasks do I have", "show pending items",
        "list all my todos", "what's my task list looking like", "view my tasks",
        "check my todo list", "what do I have to do today", "show me my list",
        "any pending tasks", "show my taks", "wat are my todos", "sho tasks pls",
        "my task list", "display tasks", "list pending todos", "show today's tasks",
        "what's pending", "check tasks", "view todo list", "show all tasks",
        "list my pending items", "what needs to be done", "show incomplete tasks",
        "tasks please", "my todos", "pending task list", "show what I have to do",
        "list things to do", "what's on my plate", "show task list", "view all todos",
        "check my list", "what tasks are left", "kindly show my tasks",
    ]
    for query in get_tasks_data:
        dataset.append(create_example("productivity", query, "get_tasks", {}))
    
    # create_calendar_event - 40 examples
    create_calendar_event_data = [
        ("schedule a meeting with Rahul tomorrow at 3pm", {"title": "meeting with Rahul", "date": "tomorrow", "time": "3pm"}),
        ("create event team standup at 9am", {"title": "team standup", "time": "9am"}),
        ("book meeting for friday 2pm", {"title": "meeting", "date": "friday", "time": "2pm"}),
        ("add dentist appointment on monday", {"title": "dentist appointment", "date": "monday"}),
        ("schedule call with client", {"title": "call with client"}),
        ("create lunch meeting tomorrow 12pm", {"title": "lunch meeting", "date": "tomorrow", "time": "12pm"}),
        ("book interview next wednesday at 10am", {"title": "interview", "date": "next wednesday", "time": "10am"}),
        ("add event project review", {"title": "project review"}),
        ("schedule sync with team 4pm today", {"title": "sync with team", "date": "today", "time": "4pm"}),
        ("create meeting family dinner sunday", {"title": "family dinner", "date": "sunday"}),
        ("book doctor appointment march 15", {"title": "doctor appointment", "date": "march 15"}),
        ("add event gym session 6pm", {"title": "gym session", "time": "6pm"}),
        ("schedule presentation next monday", {"title": "presentation", "date": "next monday"}),
        ("create calendar entry team outing", {"title": "team outing"}),
        ("book meeting with manager friday 11am", {"title": "meeting with manager", "date": "friday", "time": "11am"}),
        ("add event birthday party saturday 7pm", {"title": "birthday party", "date": "saturday", "time": "7pm"}),
        ("schedule webinar tomorrow 2pm", {"title": "webinar", "date": "tomorrow", "time": "2pm"}),
        ("create event code review", {"title": "code review"}),
        ("book flight checkin tuesday 8am", {"title": "flight checkin", "date": "tuesday", "time": "8am"}),
        ("add meeting quarterly review", {"title": "quarterly review"}),
        ("schedule coffee with sarah thursday", {"title": "coffee with sarah", "date": "thursday"}),
        ("create event parent teacher meeting", {"title": "parent teacher meeting"}),
        ("book dentist next week", {"title": "dentist", "date": "next week"}),
        ("add event yoga class 7am", {"title": "yoga class", "time": "7am"}),
        ("schedule one on one with boss", {"title": "one on one with boss"}),
        ("create meeting product launch", {"title": "product launch"}),
        ("book therapy session wednesday 5pm", {"title": "therapy session", "date": "wednesday", "time": "5pm"}),
        ("add event grocery shopping saturday", {"title": "grocery shopping", "date": "saturday"}),
        ("schedule team lunch tomorrow 1pm", {"title": "team lunch", "date": "tomorrow", "time": "1pm"}),
        ("create event performance review", {"title": "performance review"}),
        ("book car service appointment", {"title": "car service appointment"}),
        ("add meeting sprint planning monday", {"title": "sprint planning", "date": "monday"}),
        ("schedule date night friday 8pm", {"title": "date night", "date": "friday", "time": "8pm"}),
        ("create event conference call", {"title": "conference call"}),
        ("book haircut appointment", {"title": "haircut appointment"}),
        ("add event study session 3pm", {"title": "study session", "time": "3pm"}),
        ("schedule board meeting next thursday", {"title": "board meeting", "date": "next thursday"}),
        ("create event anniversary dinner", {"title": "anniversary dinner"}),
        ("book vet appointment tuesday", {"title": "vet appointment", "date": "tuesday"}),
        ("kindly schedule team sync 10am", {"title": "team sync", "time": "10am"}),
    ]
    for query, args in create_calendar_event_data:
        dataset.append(create_example("productivity", query, "create_calendar_event", args))
    
    # ===== SYSTEM TOOLS (120 examples) =====
    
    # open_app - 40 examples
    open_app_data = [
        ("open vscode", {"app_name": "vscode"}),
        ("launch chrome", {"app_name": "chrome"}),
        ("start spotify", {"app_name": "spotify"}),
        ("open terminal", {"app_name": "terminal"}),
        ("launch firefox", {"app_name": "firefox"}),
        ("start slack", {"app_name": "slack"}),
        ("open discord", {"app_name": "discord"}),
        ("launch notepad", {"app_name": "notepad"}),
        ("start calculator", {"app_name": "calculator"}),
        ("open file manager", {"app_name": "file manager"}),
        ("launch settings", {"app_name": "settings"}),
        ("start browser", {"app_name": "browser"}),
        ("open email client", {"app_name": "email client"}),
        ("launch photoshop", {"app_name": "photoshop"}),
        ("start word", {"app_name": "word"}),
        ("open excel", {"app_name": "excel"}),
        ("launch powerpoint", {"app_name": "powerpoint"}),
        ("start teams", {"app_name": "teams"}),
        ("open zoom", {"app_name": "zoom"}),
        ("launch obs", {"app_name": "obs"}),
        ("start steam", {"app_name": "steam"}),
        ("open edge", {"app_name": "edge"}),
        ("launch safari", {"app_name": "safari"}),
        ("start finder", {"app_name": "finder"}),
        ("open control panel", {"app_name": "control panel"}),
        ("launch task manager", {"app_name": "task manager"}),
        ("start command prompt", {"app_name": "command prompt"}),
        ("open vs code", {"app_name": "vs code"}),
        ("launch visual studio code", {"app_name": "visual studio code"}),
        ("start code editor", {"app_name": "code editor"}),
        ("open chrome browser", {"app_name": "chrome browser"}),
        ("launch web browser", {"app_name": "web browser"}),
        ("start internet browser", {"app_name": "internet browser"}),
        ("open text editor", {"app_name": "text editor"}),
        ("launch music app", {"app_name": "music app"}),
        ("start video player", {"app_name": "video player"}),
        ("open image viewer", {"app_name": "image viewer"}),
        ("launch pdf reader", {"app_name": "pdf reader"}),
        ("start notes app", {"app_name": "notes app"}),
        ("kindly open terminal", {"app_name": "terminal"}),
    ]
    for query, args in open_app_data:
        dataset.append(create_example("system", query, "open_app", args))
    
    # run_command - 40 examples
    run_command_data = [
        ("run pip install requests", {"command": "pip install requests"}),
        ("execute npm start", {"command": "npm start"}),
        ("run git status", {"command": "git status"}),
        ("execute python script.py", {"command": "python script.py"}),
        ("run ls -la", {"command": "ls -la"}),
        ("execute docker compose up", {"command": "docker compose up"}),
        ("run make build", {"command": "make build"}),
        ("execute cargo build", {"command": "cargo build"}),
        ("run apt update", {"command": "apt update"}),
        ("execute node server.js", {"command": "node server.js"}),
        ("run yarn install", {"command": "yarn install"}),
        ("execute pytest", {"command": "pytest"}),
        ("run go run main.go", {"command": "go run main.go"}),
        ("execute cargo test", {"command": "cargo test"}),
        ("run gradle build", {"command": "gradle build"}),
        ("execute maven clean install", {"command": "maven clean install"}),
        ("run flutter run", {"command": "flutter run"}),
        ("execute terraform apply", {"command": "terraform apply"}),
        ("run kubectl get pods", {"command": "kubectl get pods"}),
        ("execute bash script.sh", {"command": "bash script.sh"}),
        ("run pip install numpy pandas", {"command": "pip install numpy pandas"}),
        ("execute git pull origin main", {"command": "git pull origin main"}),
        ("run npm run build", {"command": "npm run build"}),
        ("execute python manage.py migrate", {"command": "python manage.py migrate"}),
        ("run docker ps", {"command": "docker ps"}),
        ("execute cargo run", {"command": "cargo run"}),
        ("run javac Main.java", {"command": "javac Main.java"}),
        ("execute bundle install", {"command": "bundle install"}),
        ("run composer install", {"command": "composer install"}),
        ("execute mix phx.server", {"command": "mix phx.server"}),
        ("run go build", {"command": "go build"}),
        ("execute rake db:migrate", {"command": "rake db:migrate"}),
        ("run pip list", {"command": "pip list"}),
        ("execute git commit -m update", {"command": "git commit -m update"}),
        ("run npm test", {"command": "npm test"}),
        ("execute python -m pytest", {"command": "python -m pytest"}),
        ("run docker build -t myapp", {"command": "docker build -t myapp"}),
        ("execute cargo check", {"command": "cargo check"}),
        ("run flutter build apk", {"command": "flutter build apk"}),
        ("kindly run pip install torch", {"command": "pip install torch"}),
    ]
    for query, args in run_command_data:
        dataset.append(create_example("system", query, "run_command", args))
    
    # get_system_info - 40 examples (no args)
    get_system_info_data = [
        "show system info", "what's my cpu usage", "show memory usage",
        "check disk space", "battery status", "what's my ip address",
        "system information", "cpu temperature", "ram usage",
        "storage info", "network status", "show system details",
        "how much memory left", "cpu load", "system details",
        "hardware info", "show cpu info", "memory stats",
        "disk usage", "battery percentage", "what's my ip",
        "system status", "computer info", "show specs",
        "what are my specs", "check system", "systm info",
        "cpu usge", "show mem usage", "disk spce",
        "batery status", "whats my ip", "system infrmation",
        "cpu temp", "ram usge", "storge info",
        "show hardware details", "check computer specs", "display system stats",
        "kindly show system info",
    ]
    for query in get_system_info_data:
        dataset.append(create_example("system", query, "get_system_info", {}))
    
    # ===== RESEARCH TOOLS (120 examples) =====
    
    # web_search - 40 examples
    web_search_data = [
        ("search for python tutorials", {"query": "python tutorials"}),
        ("search how to make pasta", {"query": "how to make pasta"}),
        ("find information about machine learning", {"query": "machine learning"}),
        ("look up weather in New York", {"query": "weather in New York"}),
        ("search for best laptops 2024", {"query": "best laptops 2024"}),
        ("find recipes for dinner", {"query": "recipes for dinner"}),
        ("search what is quantum computing", {"query": "what is quantum computing"}),
        ("look up how to learn guitar", {"query": "how to learn guitar"}),
        ("find information about climate change", {"query": "climate change"}),
        ("search for travel destinations", {"query": "travel destinations"}),
        ("look up javascript array methods", {"query": "javascript array methods"}),
        ("search how to fix a flat tire", {"query": "how to fix a flat tire"}),
        ("find docs on react hooks", {"query": "react hooks"}),
        ("look up who won the game yesterday", {"query": "who won the game yesterday"}),
        ("search for healthy breakfast ideas", {"query": "healthy breakfast ideas"}),
        ("find information about space exploration", {"query": "space exploration"}),
        ("search what time is it in London", {"query": "what time is it in London"}),
        ("look up nearest coffee shops", {"query": "nearest coffee shops"}),
        ("find a good restaurant nearby", {"query": "good restaurant nearby"}),
        ("search for movie recommendations", {"query": "movie recommendations"}),
        ("look up docker commands", {"query": "docker commands"}),
        ("search how to meditate", {"query": "how to meditate"}),
        ("find study tips", {"query": "study tips"}),
        ("look up history of computers", {"query": "history of computers"}),
        ("search for coding interview questions", {"query": "coding interview questions"}),
        ("find information about ai", {"query": "ai"}),
        ("look up typescript documentation", {"query": "typescript documentation"}),
        ("search for keto diet plan", {"query": "keto diet plan"}),
        ("find yoga poses for beginners", {"query": "yoga poses for beginners"}),
        ("look up current gold price", {"query": "current gold price"}),
        ("search for remote jobs", {"query": "remote jobs"}),
        ("find best budget smartphones", {"query": "best budget smartphones"}),
        ("look up flu symptoms", {"query": "flu symptoms"}),
        ("search for online courses", {"query": "online courses"}),
        ("find hiking trails near me", {"query": "hiking trails near me"}),
        ("look up stock market today", {"query": "stock market today"}),
        ("search for vegan recipes", {"query": "vegan recipes"}),
        ("find productivity apps", {"query": "productivity apps"}),
        ("look up visa requirements for japan", {"query": "visa requirements for japan"}),
        ("kindly search for python documentation", {"query": "python documentation"}),
    ]
    for query, args in web_search_data:
        dataset.append(create_example("research", query, "web_search", args))
    
    # search_stackoverflow - 40 examples
    search_stackoverflow_data = [
        ("search stackoverflow for null pointer exception", {"query": "null pointer exception"}),
        ("find stackoverflow results for react error", {"query": "react error"}),
        ("look on stackoverflow for python dict error", {"query": "python dict error"}),
        ("check stackoverflow for segmentation fault", {"query": "segmentation fault"}),
        ("search stackoverflow for async await", {"query": "async await"}),
        ("find stackoverflow solution for memory leak", {"query": "memory leak"}),
        ("look up this error on stackoverflow", {"query": "error"}),
        ("search stackoverflow for cors issue", {"query": "cors issue"}),
        ("find stackoverflow for import error", {"query": "import error"}),
        ("check stackoverflow for typeerror", {"query": "typeerror"}),
        ("search stackoverflow for git merge conflict", {"query": "git merge conflict"}),
        ("find stackoverflow for docker compose error", {"query": "docker compose error"}),
        ("look up stackoverflow react useeffect issue", {"query": "react useeffect issue"}),
        ("search stackoverflow for numpy error", {"query": "numpy error"}),
        ("find stackoverflow for pandas dataframe error", {"query": "pandas dataframe error"}),
        ("check stackoverflow for spring boot error", {"query": "spring boot error"}),
        ("search stackoverflow for 404 error", {"query": "404 error"}),
        ("find stackoverflow for database connection error", {"query": "database connection error"}),
        ("look up stackoverflow for keyerror python", {"query": "keyerror python"}),
        ("search stackoverflow for npm install error", {"query": "npm install error"}),
        ("find stackoverflow for attribute error", {"query": "attribute error"}),
        ("check stackoverflow for index out of bounds", {"query": "index out of bounds"}),
        ("search stackoverflow for undefined variable", {"query": "undefined variable"}),
        ("find stackoverflow for timeout error", {"query": "timeout error"}),
        ("look up stackoverflow for ssl certificate error", {"query": "ssl certificate error"}),
        ("search stackoverflow for slow query", {"query": "slow query"}),
        ("find stackoverflow for reference error", {"query": "reference error"}),
        ("check stackoverflow for syntax error", {"query": "syntax error"}),
        ("search stackoverflow for connection refused", {"query": "connection refused"}),
        ("find stackoverflow for promise rejection", {"query": "promise rejection"}),
        ("look up stackoverflow for module not found", {"query": "module not found"}),
        ("search stackoverflow for permission denied", {"query": "permission denied"}),
        ("find stackoverflow for null reference", {"query": "null reference"}),
        ("check stackoverflow for array index error", {"query": "array index error"}),
        ("search stackoverflow for file not found", {"query": "file not found"}),
        ("find stackoverflow for out of memory", {"query": "out of memory"}),
        ("look up stackoverflow for thread error", {"query": "thread error"}),
        ("search stackoverflow for json parse error", {"query": "json parse error"}),
        ("find stackoverflow for network error", {"query": "network error"}),
        ("kindly search stackoverflow for python exception", {"query": "python exception"}),
    ]
    for query, args in search_stackoverflow_data:
        dataset.append(create_example("research", query, "search_stackoverflow", args))
    
    # search_arxiv - 40 examples
    search_arxiv_data = [
        ("find papers on neural networks", {"query": "neural networks"}),
        ("search arxiv for transformer models", {"query": "transformer models"}),
        ("find research papers about computer vision", {"query": "computer vision"}),
        ("look up papers on natural language processing", {"query": "natural language processing"}),
        ("find arxiv papers about gpt", {"query": "gpt"}),
        ("search arxiv for reinforcement learning", {"query": "reinforcement learning"}),
        ("find academic papers on deep learning", {"query": "deep learning"}),
        ("look up research on quantum computing", {"query": "quantum computing"}),
        ("search for papers about attention mechanism", {"query": "attention mechanism"}),
        ("find arxiv papers on generative ai", {"query": "generative ai"}),
        ("search arxiv for diffusion models", {"query": "diffusion models"}),
        ("find papers about large language models", {"query": "large language models"}),
        ("look up research on machine translation", {"query": "machine translation"}),
        ("search arxiv for robotics papers", {"query": "robotics"}),
        ("find academic research on sentiment analysis", {"query": "sentiment analysis"}),
        ("look up papers on graph neural networks", {"query": "graph neural networks"}),
        ("search for research on federated learning", {"query": "federated learning"}),
        ("find arxiv papers about optimization", {"query": "optimization"}),
        ("look up papers on computer graphics", {"query": "computer graphics"}),
        ("find research on speech recognition", {"query": "speech recognition"}),
        ("search arxiv for multimodal learning", {"query": "multimodal learning"}),
        ("look up papers about knowledge graphs", {"query": "knowledge graphs"}),
        ("find academic papers on time series", {"query": "time series"}),
        ("search for research on recommendation systems", {"query": "recommendation systems"}),
        ("find papers about self supervised learning", {"query": "self supervised learning"}),
        ("look up arxiv for vision transformers", {"query": "vision transformers"}),
        ("search arxiv for bert papers", {"query": "bert"}),
        ("find papers on convolutional neural networks", {"query": "convolutional neural networks"}),
        ("look up research on object detection", {"query": "object detection"}),
        ("search arxiv for gan papers", {"query": "gan"}),
        ("find academic papers on meta learning", {"query": "meta learning"}),
        ("look up papers on few shot learning", {"query": "few shot learning"}),
        ("search for research on transfer learning", {"query": "transfer learning"}),
        ("find arxiv papers on neural architecture search", {"query": "neural architecture search"}),
        ("look up papers on explainable ai", {"query": "explainable ai"}),
        ("search arxiv for zero shot learning", {"query": "zero shot learning"}),
        ("find papers on natural language understanding", {"query": "natural language understanding"}),
        ("look up research on text summarization", {"query": "text summarization"}),
        ("search arxiv for question answering", {"query": "question answering"}),
        ("kindly find papers on machine learning", {"query": "machine learning"}),
    ]
    for query, args in search_arxiv_data:
        dataset.append(create_example("research", query, "search_arxiv", args))
    
    # ===== COMMUNICATION TOOLS (80 examples) =====
    
    # send_email - 40 examples
    send_email_data = [
        ("send email to john", {"recipient": "john"}),
        ("email sarah about the meeting", {"recipient": "sarah", "subject": "about the meeting"}),
        ("send a mail to my boss", {"recipient": "my boss"}),
        ("compose an email to the team", {"recipient": "the team"}),
        ("send email to mom", {"recipient": "mom"}),
        ("email the client", {"recipient": "the client"}),
        ("send a message to david", {"recipient": "david"}),
        ("compose mail to hr", {"recipient": "hr"}),
        ("send mail to support", {"recipient": "support"}),
        ("email my professor", {"recipient": "my professor"}),
        ("send an email to the manager", {"recipient": "the manager"}),
        ("mail the invoice to accounting", {"recipient": "accounting", "subject": "the invoice"}),
        ("email dad about dinner", {"recipient": "dad", "subject": "about dinner"}),
        ("send message to friend", {"recipient": "friend"}),
        ("compose email to customer service", {"recipient": "customer service"}),
        ("mail the report", {"recipient": "", "subject": "the report"}),
        ("email the document to mike", {"recipient": "mike", "subject": "the document"}),
        ("send a quick email to lisa", {"recipient": "lisa"}),
        ("mail the team about the update", {"recipient": "the team", "subject": "about the update"}),
        ("email my doctor for appointment", {"recipient": "my doctor", "subject": "for appointment"}),
        ("send mail to school", {"recipient": "school"}),
        ("compose message to colleague", {"recipient": "colleague"}),
        ("email the vendor", {"recipient": "the vendor"}),
        ("send an email fast", {"recipient": ""}),
        ("mail the subscription cancellation", {"recipient": "", "subject": "the subscription cancellation"}),
        ("email regarding the job application", {"recipient": "", "subject": "regarding the job application"}),
        ("send mail to insurance company", {"recipient": "insurance company"}),
        ("compose email to landlord", {"recipient": "landlord"}),
        ("email the restaurant for reservation", {"recipient": "the restaurant", "subject": "for reservation"}),
        ("send message to group", {"recipient": "group"}),
        ("mail the payment confirmation", {"recipient": "", "subject": "the payment confirmation"}),
        ("email about the refund", {"recipient": "", "subject": "about the refund"}),
        ("send an email to tech support", {"recipient": "tech support"}),
        ("mail the meeting minutes", {"recipient": "", "subject": "the meeting minutes"}),
        ("email my sister", {"recipient": "my sister"}),
        ("send mail to bank", {"recipient": "bank"}),
        ("compose email to recruiter", {"recipient": "recruiter"}),
        ("email the hotel for booking", {"recipient": "the hotel", "subject": "for booking"}),
        ("send message to coworker", {"recipient": "coworker"}),
        ("mail the feedback", {"recipient": "", "subject": "the feedback"}),
        ("kindly send email to team", {"recipient": "team"}),
    ]
    for query, args in send_email_data:
        dataset.append(create_example("communication", query, "send_email", args))
    
    # read_emails - 40 examples (no args)
    read_emails_data = [
        "check my emails", "read my latest mail", "show my inbox",
        "read unread messages", "show new emails", "check inbox",
        "read latest email", "show my mail", "read emails from boss",
        "check for new messages", "show unread emails", "read morning mail",
        "check email from john", "show emails from today", "read the latest message",
        "check work emails", "show important emails", "read mail from sarah",
        "check for any new mail", "show me my inbox", "read recent emails",
        "check email notifications", "show emails from yesterday", "read the urgent mail",
        "check personal emails", "show all new messages", "read emails from team",
        "check mail from support", "show starred emails", "read the thread",
        "check for replies", "show sent emails", "read draft emails",
        "check spam folder", "show trash emails", "read archived mail",
        "check all folders", "show email count", "view my emails",
        "display inbox", "kindly check my emails",
    ]
    for query in read_emails_data:
        dataset.append(create_example("communication", query, "read_emails", {}))
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    return dataset


if __name__ == "__main__":
    dataset = generate_dataset()
    
    # Save as JSON
    with open("functiongemma_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print("Saved: functiongemma_dataset.json")
    
    # Save as JSONL
    with open("functiongemma_dataset.jsonl", "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print("Saved: functiongemma_dataset.jsonl")
    
    # Print summary
    tool_counts = {}
    for example in dataset:
        tool = json.loads(example["messages"][2]["content"])["tool"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")
    print(f"\n  Total: {len(dataset)} examples")
