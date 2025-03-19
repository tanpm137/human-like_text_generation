from groq import Groq
import concurrent.futures

llm_guideline = """
Generate only the result.
The response must be less than words—strictly no more, no less.
Ensure the output is naturally concise while fully covering the topic.
"""

def generate_essay(topic, api_key):
    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": llm_guideline
            },
            {
                "role": "user",
                "content": f"Generate short essay for topic: {topic})"
            }
        ],

        model="llama-3.2-3b-preview",
        temperature=2,
        max_completion_tokens=512,
        top_p=1,
        stop=None,
        stream=False,
    )

    return chat_completion.choices[0].message.content

def generate_n_essays(topic, num_of_results, api_key):    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda _: generate_essay(topic=topic, api_key=api_key), range(num_of_results)))
    return results
