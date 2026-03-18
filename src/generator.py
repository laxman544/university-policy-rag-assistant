from transformers import pipeline


class AnswerGenerator:
    """
    Generates answers using retrieved context and a text-to-text model.
    """

    def __init__(self, model_name: str):
        self.generator = pipeline("text2text-generation", model=model_name)

    def build_prompt(self, query: str, retrieved_chunks: list) -> str:
        """
        Create a grounded prompt using retrieved chunks.
        """
        context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

        prompt = f"""
You are a helpful university handbook assistant.
Answer the user's question only from the provided context.
If the answer is not in the context, say:
"I could not find that information in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt.strip()

    def generate_answer(self, query: str, retrieved_chunks: list) -> str:
        """
        Generate final answer.
        """
        prompt = self.build_prompt(query, retrieved_chunks)

        output = self.generator(
            prompt,
            max_length=256,
            do_sample=False
        )

        return output[0]["generated_text"]
