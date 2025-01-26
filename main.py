from components import helpers, custom_llm

def run_custom_llm():
    llm = custom_llm.localLlama()
    answer = llm.invoke("Hi, how are you?")
    print(answer)

def main():
    run_custom_llm()

if __name__ == "__main__":
    main()

