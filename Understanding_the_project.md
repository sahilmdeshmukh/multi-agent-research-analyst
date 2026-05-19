Multi-Agent Research Analyst — 
The big picture (what you're building)
Imagine you ask a question like "How does ASML's monopoly affect chip prices?" — instead of one AI giving you a rushed answer, you have a team of three AI agents that work together like a small research firm:

Researcher → does the Googling and takes notes
Critic → reads the notes and says "these sources are weak, go find more"
Synthesizer → writes the final report only after the critic approves
The "secret sauce" is that critique loop — the researcher can be sent back to do more work, up to 3 times, before the report is written. Most AI demos skip this. That's what makes this project portfolio-worthy.

The user types a topic into a web page → watches the agents argue in real time → gets a cited memo at the end.

How the agents talk to each other (architecture)

START → Researcher → Critic → [approve?] → Synthesizer → END
                       ↑              ↓
                       └── no, revise─┘  (max 3 loops)
This flow is called a graph — nodes (agents) connected by edges (rules for what happens next). The decision "go to synthesizer or back to researcher?" is called a conditional edge.

All three agents share one big object called AgentState — a Python data structure holding the query, the notes collected so far, the critiques, the round number, and the final memo. Think of it as a shared whiteboard.

The tech stack — what each tool does
Tool	What it actually is	Why it's here
Python 3.12	The programming language	Standard for AI work
uv	A package manager (like npm for Python, but faster)	Modern, fast, replacing pip
LangGraph	A framework for building agent workflows as graphs	This is THE library for multi-agent systems in 2026
Groq	A cloud service that runs open-source LLMs very fast	Free + sub-second responses, vs OpenAI which costs money
Llama 3.3 70B	The actual AI model (Meta's open-weight model)	Smart enough, runs on Groq for free
langchain-groq	Glue code that lets LangGraph talk to Groq	Adapter layer
Tavily	A search engine API built for AI agents	Returns clean text instead of raw HTML — perfect for LLMs
Pydantic	A library that enforces strict data shapes	Forces the AI to output structured data (claim + URL + confidence) instead of messy text
Streamlit	Turns Python scripts into web apps with ~10 lines	Fastest way to build a demo UI
pytest	Testing framework	Standard for Python
Hugging Face Spaces	Free hosting for AI demos	Where the live demo will live publicly
A few concepts worth knowing as a beginner
LLM (Large Language Model) — the AI brain. Here it's Llama 3.3 70B running on Groq.
Structured output — instead of letting the AI ramble, Pydantic forces it to fill in a specific shape (like a form). E.g., ResearchNote(claim="X", source_url="Y", confidence=0.8).
Streaming — instead of waiting 30 seconds for an answer, the UI shows updates as each agent finishes ("Researcher searching…", "Critic found 2 gaps…").
Eval harness — code that runs the system on 10 test queries and compares it against a "dumb" single-agent baseline. This generates the real numbers that go in the README. No fake numbers allowed.