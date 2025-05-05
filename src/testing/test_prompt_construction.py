def build_prompt(questions, snippets):
    context = "\n\n".join(snippets)
    return f"""You are a helpful assistant with access to recent news.
Context:
{context}
Based on this context, answer the question.
{questions}
"""


if __name__ == "__main__":
    fake_snippets = [
        "Chandrayaan-3 successfully landed on the Moon's south pole.",
        "ISRO's mission was praised globally for its low-cost achievement."
    ]
    question = "What happened to Chandrayaan-3 on August 23, 2023?"
    prompt = build_prompt(question, fake_snippets)
    print(prompt)
