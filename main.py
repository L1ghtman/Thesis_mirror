import asyncio
from components import helpers, custom_llm

def run_custom_llm():
    llm = custom_llm.localLlama()
    answer = llm.invoke("For testing purposes, write a 10 sentence story.")
    print(answer)

async def stream_custom_llm():
    llm = custom_llm.localLlama(n=5)
    output = ""
    async for token in llm.astream("For testing purposes, write a 10 sentence story."):
        print(token, end="", flush=True)

def main():
    # run_custom_llm()
    asyncio.run(stream_custom_llm()) 

if __name__ == "__main__":
    main()

