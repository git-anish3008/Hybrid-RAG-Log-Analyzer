🛠️ Intune Local RAG Agent

If you have ever stared at an IntuneManagementExtension.log or Autopilot diagnostic file for three hours trying to figure out exactly why a Win32 app failed, you know the pain.

Recently, I started learning about LLMs and RAG (Retrieval-Augmented Generation) to see if I could use AI to automate this troubleshooting. But I immediately hit a massive roadblock that every enterprise IT engineer faces: Data Privacy.

You simply cannot copy-paste corporate Intune logs into ChatGPT or cloud-based AI tools. Those files are packed with sensitive device identifiers, internal network configurations, and sometimes user data. If that telemetry leaves your environment, your security and compliance teams will have a heart attack (and rightly so!).

I needed a solution of my own—an AI that actually understands Intune architecture, but runs entirely on my local machine where the data never leaves the device.

So, I went down the AI rabbit hole and built this. It is a fully local, privacy-first AI diagnostic tool that cross-references Intune logs against IT runbooks to figure out what broke. I am still relatively new to building AI architectures, so the code might not be 100% perfect, but it solves a massive real-world problem for platform engineers who need AI without compromising security.


🚀 What This Actually Does

Zero Cloud Telemetry: Everything runs locally on your machine. You aren't uploading sensitive corporate device logs to OpenAI.

Hybrid Log Search: I learned that standard AI semantic search is terrible at finding exact IDs. This tool uses a Hybrid Engine (Vector Search + BM25 Exact Match) to ensure it never misses a specific error code.

Smart Bypass (LangGraph): I set up a state machine. If you ask the AI about a specific App ID (like b5e3c7...), it completely bypasses the heavy AI math and runs a hyper-fast "smart-grep" to pull the exact timeline of that app's deployment from the bottom up.

Chronological Context: When it finds an error in the logs, it automatically pulls the next few lines of the file so the LLM can read the "aftermath" of the failure.


🏗️ The Tech Stack

I built this to run on a standard engineering laptop, which meant avoiding super heavy GPU requirements.

Databases: Qdrant (for vector meaning) & BM25 (for exact keyword tracking)

Embeddings: FastEmbed using BAAI/bge-small-en-v1.5 (Runs on CPU so you don't need a massive graphics card)

LLM: Ollama running qwen2.5:3b 


🤝 Contributing

I built this to scratch my own itch and learn how LLMs actually work under the hood. If you are an Intune admin who wants to test this on some nasty autopilot logs, or an AI dev who wants to clean up my Python code, I would love the feedback. Feel free to open an issue or a pull request!
