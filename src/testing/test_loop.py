def run_chat():
    print("🗨️ Start chatting (type 'exit' to quit):")
    while True:
        q = input("\n🧠 You: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("💡 Imagine this would search the web and answer:", q)


if __name__ == "__main__":
    run_chat()
