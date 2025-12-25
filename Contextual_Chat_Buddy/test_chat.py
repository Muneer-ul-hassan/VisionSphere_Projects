from gemini_client import ask_gemini

def main():
    print("Start chatting with Gemini! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = ask_gemini(user_input)
        print("Gemini:", response)

if __name__ == "__main__":
    main()
