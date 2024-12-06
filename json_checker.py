import json

file_name = 'chapter2.json'

try:
    with open(file_name, 'r', encoding='utf-8') as file:
        json_data = file.read()
        parsed = json.loads(json_data) 
        print("JSON is valid!")
except json.JSONDecodeError as e:
    print("JSON is invalid:", e)
except FileNotFoundError:
    print(f"The file '{file_name}' does not exist. Please check the file path.")
