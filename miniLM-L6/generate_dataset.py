"""
Tsuzi Intent Classification Dataset Generator
Generates 400 diverse, realistic training examples for voice assistant intent classification.
"""

import json
import random

def generate_dataset(seed=42):
    random.seed(seed)
    dataset = []

    # ===== CASUAL (80 examples) - label_id: 0 =====
    casual_voice = [
        # Greetings
        "hey tsuzi", "hi there", "hello", "hey", "yo", "hiya", "howdy",
        "good morning", "good afternoon", "good evening", "morning", "evening",
        "what's up", "sup", "greetings", "um hey", "uh hi", "hey um",
        "oh hey there", "ah good morning", "well hello", "so hey",
        
        # Small talk
        "how are you doing today", "how's it going", "how you been",
        "what's going on", "how's your day been", "how have you been",
        "what's new with you", "how's everything going", "you doing okay",
        "how's life treating you", "what's happening", "how are things",
        "how you doing", "how's your morning going", "how's work going",
        "long time no talk", "nice to see you again", "good to hear you",
        "missed talking to you", "hey it's been a while", "so uh how are you",
        "um how's it going", "well how have you been", "oh what's new",
        
        # Short replies
        "ok", "okay", "k", "yes", "yeah", "yea", "yep", "yup",
        "nah", "nope", "sure", "alright", "aight", "cool", "nice",
        "great", "awesome", "perfect", "got it", "roger that",
        "uh huh", "hmm", "oh okay", "wow", "oops", "ouch", "phew",
        "brb", "ttyl", "np", "ty", "that's fine", "sounds good",
        "um ok", "uh sure", "yeah um", "oh alright", "well sure",
        
        # Thanks
        "thanks", "thank you", "thank you so much", "thanks a lot",
        "thanks a bunch", "much appreciated", "I appreciate it",
        "you're the best", "cheers", "thx", "um thanks", "thank u",
        
        # Dismissals/Closings
        "never mind", "forget it", "it's fine really", "don't worry about it",
        "all good", "no worries", "it's okay", "just leave it",
        "doesn't matter", "skip it", "bye", "goodbye", "see you",
        "see ya", "talk to you later", "catch you later", "peace out",
        "take care", "have a good one", "later", "um never mind",
        "uh forget it", "oh bye", "well anyway", "so yeah bye"
    ]
    
    casual_all = casual_voice[:80]
    for text in casual_all:
        dataset.append({"text": text, "label": "casual", "label_id": 0})

    # ===== PRODUCTIVITY (80 examples) - label_id: 1 =====
    productivity_voice = [
        # set_timer - conversational with hesitations
        "can you set a timer for like 5 minutes", "hey um start a timer for 10 minutes please",
        "I need a timer for about 15 minutes", "could you start timing 20 minutes for me",
        "um set a timer for 25 minutes", "start a 30 minute timer would you",
        "hey can you do a 45 minute timer", "I want a timer for 1 hour",
        "timer for 2 minutes please", "can I get a 3 minute timer",
        "um time me for 10 minutes", "start counting down from 5 minutes",
        "begin a 15 minute countdown", "set timer to 20 minutes",
        "please time 30 minutes", "let's do a timer for 45 minutes",
        "timer uh 5 minutes", "start timmer for 20 min", "countdown 15 min please",
        
        # set_alarm - conversational
        "can you wake me up at 7 tomorrow", "hey um set an alarm for 6:30 in the morning",
        "I need an alarm at 8 am please", "could you set my alarm for 5:30",
        "um alarm for 7am tomorrow", "set a morning alarm for 9",
        "hey alarm me at noon would you", "I have to wake up at 6:15",
        "wake me up at 7:30 sharp", "can you set alarm for 8:45",
        "alarm at 9:30 please", "I need to get up by 5 am",
        "set alarm seven thirty", "wake me at 6am",
        "alarm for quarter to 9", "need alarm for early tomorrow",
        "um set alram for 7am", "uh wake me up at 6", "wak me up at 7",
        
        # create_calendar_event - conversational
        "can you create a meeting at 3 today", "hey um schedule a call with the team",
        "I need to add an event to my calendar", "could you book a meeting for tomorrow",
        "um set up a meeting at 2pm", "I have a meeting at 4 please add it",
        "schedule my dentist appointment", "add lunch meeting to calendar",
        "book a time slot for 11am", "create an event called team standup",
        "put this meeting on my calendar", "schedule an interview at 10",
        "add daily standup at 9:30", "book time for a 1pm meeting",
        "schedule a sync for Friday", "create calendar entry for 3pm",
        "creat meeting at 3pm", "schedle call with team", "book meting tomorrow",
        "prepone meeting to 2pm", "prepone my call to 3",
        
        # add_task - conversational
        "can you add a task to buy groceries", "hey um new task call my mom",
        "I need to remember to finish the report", "could you add task review the code",
        "um add to my tasks email the client", "create a task for tomorrow",
        "I have to do laundry today", "task submit my assignment please",
        "add a task called workout", "put this on my list pay the bills",
        "new todo buy milk", "add task fix that bug",
        "task for me read the docs", "I have to call the dentist",
        "add this task write a blog post", "create todo water the plants",
        "add taks buy groceries", "new tsk call dad", "um task review code",
        "kindly add task", "do the needful and set timer",
        
        # get_tasks - conversational
        "can you show me my tasks", "hey um what are my todos",
        "I need to see my task list", "could you list my tasks please",
        "um show me what I need to do", "what's on my todo list",
        "display my tasks for me", "tell me my pending tasks",
        "what tasks do I have today", "show my pending items",
        "list all my todos please", "what's my task list looking like",
        "view my tasks", "check my todo list",
        "what do I have to do", "show me my list",
        "show my taks please", "wat are my todos", "my task list",
        "revert back with my tasks"
    ]
    
    for text in productivity_voice[:80]:
        dataset.append({"text": text, "label": "productivity", "label_id": 1})

    # ===== SYSTEM (80 examples) - label_id: 2 =====
    system_voice = [
        # open_app - conversational
        "can you open vscode for me", "hey um launch chrome",
        "I need you to start spotify", "could you open the terminal",
        "um launch firefox please", "start slack for me",
        "hey open discord would you", "launch notepad",
        "start the calculator", "open file manager",
        "launch settings app", "start my browser",
        "open the email client", "launch photoshop",
        "start microsoft word", "open excel please",
        "launch powerpoint", "start microsoft teams",
        "open zoom for me", "launch obs studio",
        "start steam", "open microsoft edge",
        "launch safari", "start finder",
        "open control panel", "launch task manager",
        "start command prompt", "um open vs code",
        "uh launch chrome", "hey open terminal",
        "can you um open spotify", "open code editor please",
        
        # run_command - conversational
        "can you run pip install requests", "hey um execute npm start",
        "I need you to run this command", "could you execute git status",
        "um run python script.py", "execute ls -la please",
        "run docker compose up", "execute make build for me",
        "run cargo build would you", "execute apt update",
        "run this for me please", "execute the command",
        "run a shell command", "execute this bash script",
        "run node server.js", "execute yarn install",
        "run pytest please", "execute go run main.go",
        "run cargo test", "execute gradle build",
        "run maven clean install", "execute flutter run",
        "run terraform apply", "execute kubectl get pods",
        "run this shell command", "execute the script for me",
        "run that command please", "run pip instal requests",
        "excecute npm start", "uh run this command",
        "excute git status", "run pythn script",
        "do the needful and run", "kindly execute this",
        
        # get_system_info - conversational
        "can you show system info", "hey um what's my cpu usage",
        "I need to see memory usage", "could you check disk space",
        "um what's my battery status", "what's my ip address",
        "show me system information", "what's my cpu temperature",
        "show ram usage please", "check storage info",
        "what's my network status", "show me system details",
        "how much memory is left", "what's my cpu load",
        "show system details", "give me hardware info",
        "show cpu information", "memory stats please",
        "disk usage check", "what's my battery percentage",
        "what's my ip", "check system status",
        "computer information", "show my specs",
        "what are my computer specs", "check the system",
        "systm info please", "cpu usge", "show mem usage",
        "disk spce check", "batery status", "whats my ip",
        "kindly show system info", "revert with cpu usage"
    ]
    
    for text in system_voice[:80]:
        dataset.append({"text": text, "label": "system", "label_id": 2})

    # ===== RESEARCH (80 examples) - label_id: 3 =====
    research_voice = [
        # web_search
        "can you search for python tutorials", "hey um search how to make pasta",
        "I need you to find information about machine learning",
        "could you look up the weather in New York",
        "um search for best laptops 2024", "find me recipes for dinner",
        "search what is quantum computing", "look up how to learn guitar",
        "find information about climate change", "search for travel destinations",
        "can you look up javascript array methods", "search how to fix a flat tire",
        "find me docs on react hooks", "look up who won the game yesterday",
        "search for healthy breakfast ideas", "find information about space exploration",
        "search what time is it in London", "look up nearest coffee shops",
        "find me a good restaurant nearby", "search for movie recommendations",
        "can you look up docker commands", "search how to meditate",
        "find me study tips", "look up history of computers",
        "search for coding interview questions", "find information about ai",
        "look up typescript documentation", "um search for python",
        "uh look up javascript", "hey find info on",
        
        # search_stackoverflow
        "can you search stackoverflow for null pointer exception",
        "hey um find stackoverflow results for react error",
        "I need help with this error look on stackoverflow",
        "could you check stackoverflow for python dict error",
        "um search stackoverflow segmentation fault",
        "find stackoverflow answers for async await",
        "look up this error on stackoverflow", "search stackoverflow for cors issue",
        "find stackoverflow solution for memory leak", "check stackoverflow for import error",
        "can you search stackoverflow for typeerror", "find stackoverflow help with git merge conflict",
        "look up stackoverflow for docker compose error", "search stackoverflow react useeffect issue",
        "find stackoverflow answer for numpy error", "check stackoverflow for pandas dataframe error",
        "search stackoverflow for spring boot error", "find stackoverflow solution for 404 error",
        "look up stackoverflow for database connection error", "search stackoverflow for keyerror python",
        "find stackoverflow help with npm install error", "check stackoverflow for attribute error",
        "search stackoverflow for index out of bounds", "find stackoverflow for undefined variable",
        "look up stackoverflow for timeout error", "search stackoverflow for ssl certificate error",
        "find stackoverflow solution for slow query", "can you um search stackoverflow",
        "search stackover flow for error", "find stackoverflow for null error",
        "look up stack overflow", "search stakoverflow for bug",
        
        # search_arxiv
        "can you find papers on neural networks", "hey um search arxiv for transformer models",
        "I need research papers about computer vision",
        "could you look up papers on natural language processing",
        "um find arxiv papers about gpt", "search arxiv for reinforcement learning",
        "find academic papers on deep learning", "look up research on quantum computing",
        "search for papers about attention mechanism", "find arxiv papers on generative ai",
        "can you search arxiv for diffusion models", "find papers about large language models",
        "look up research on machine translation", "search arxiv for robotics papers",
        "find academic research on sentiment analysis", "look up papers on graph neural networks",
        "search for research on federated learning", "find arxiv papers about optimization",
        "can you look up papers on computer graphics", "find research on speech recognition",
        "search arxiv for multimodal learning", "look up papers about knowledge graphs",
        "find academic papers on time series", "search for research on recommendation systems",
        "find papers about self supervised learning", "look up arxiv for vision transformers",
        "search arxiv for papers", "find arxiv papers on", "look up arxiv",
        "find papers on arxiv", "kindly search for this",
        "do the needful and search", "revert with search results"
    ]
    
    for text in research_voice[:80]:
        dataset.append({"text": text, "label": "research", "label_id": 3})

    # ===== COMMUNICATION (80 examples) - label_id: 4 =====
    communication_voice = [
        # send_email
        "can you send an email to john", "hey um email sarah about the meeting",
        "I need to send a mail to my boss", "could you compose an email to the team",
        "um send email to mom", "email the client please",
        "send a message to david", "compose mail to hr",
        "can you send mail to support", "email my professor",
        "send an email to the manager", "mail the invoice to accounting",
        "email dad about dinner", "send message to friend",
        "compose email to customer service", "can you mail the report",
        "email the document to mike", "send a quick email to lisa",
        "mail the team about the update", "email my doctor for appointment",
        "send mail to school", "compose message to colleague",
        "email the vendor", "send an email fast",
        "mail the subscription cancellation", "email regarding the job application",
        "send mail to insurance company", "compose email to landlord",
        "email the restaurant for reservation", "send message to group",
        "mail the payment confirmation", "email about the refund",
        "send an email to tech support", "mail the meeting minutes",
        "email my sister", "send mail to bank",
        "compose email to recruiter", "email the hotel for booking",
        "send message to coworker", "mail the feedback",
        "um send email to john", "uh mail sarah", "hey email boss",
        "can you um send mail", "send mail fast", "email na",
        "kindly send email", "do the needful and send mail",
        
        # read_emails
        "can you check my emails", "hey um read my latest mail",
        "I need to see my inbox", "could you show my emails",
        "um check my email please", "read my unread messages",
        "show me new emails", "check inbox for me",
        "can you read latest email", "show my mail",
        "read emails from boss", "check for new messages",
        "show unread emails", "read my morning mail",
        "check email from john", "show emails from today",
        "read the latest message", "check my work emails",
        "show important emails", "read mail from sarah",
        "check for any new mail", "show me my inbox",
        "read recent emails", "check email notifications",
        "show emails from yesterday", "read the urgent mail",
        "check personal emails", "show all new messages",
        "read emails from team", "check mail from support",
        "show starred emails", "read the thread",
        "check for replies", "show sent emails",
        "read draft emails", "check spam folder",
        "show trash emails", "read archived mail",
        "check all folders", "show email count",
        "um check my emails", "uh read my mail", "show my mails",
        "check email pls", "read latest mail", "show inbox",
        "kindly check emails", "revert back via email",
        "prepone and inform via mail"
    ]
    
    for text in communication_voice[:80]:
        dataset.append({"text": text, "label": "communication", "label_id": 4})

    # Shuffle the entire dataset
    random.shuffle(dataset)
    
    return dataset


if __name__ == "__main__":
    dataset = generate_dataset()
    
    # Save as JSON
    with open("tsuzi_intent_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print("Saved: tsuzi_intent_dataset.json")
    
    # Save as JSONL (recommended for ML training)
    with open("tsuzi_intent_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved: tsuzi_intent_dataset.jsonl")
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    for label in ["casual", "productivity", "system", "research", "communication"]:
        count = len([d for d in dataset if d["label"] == label])
        print(f"{label}: {count} examples")
    print(f"\nTotal: {len(dataset)} examples")
