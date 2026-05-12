import pandas as pd
import sys
import re

if len(sys.argv) < 2:
    print("Usage: python parse.py <path_to_file>")
    sys.exit(1)

file_path = sys.argv[1]

conversations = []
intents = []

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split blocks by intent pattern (e.g., PAY_NOW,"...")
    blocks = re.split(r'\n(?=[A-Z_]+,")', content)

    for block in blocks:
        lines = block.strip().split('\n')
        
        # Last line contains intent
        last_line = lines[-1]
        match = re.match(r'([A-Z_]+),', last_line)

        if match:
            intent = match.group(1)
            conversation = "\n".join(lines[:-1]).strip()

            if conversation:
                conversations.append(conversation)
                intents.append(intent)

    df = pd.DataFrame({
        "conversation_text": conversations,
        "intent": intents
    })

    print("\n✅ Parsed dataset preview:\n")
    print(df.head())

    print("\n📊 Intent distribution:\n")
    print(df['intent'].value_counts())

except Exception as e:
    print(f"❌ Error: {e}")
