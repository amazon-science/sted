import json
import os

def extract_sharegpt_conversations(input_file, output_dir="extracted_data"):
    """
    Extract conversations from the ShareGPT structured output JSON dataset
    and save them in a more usable format.
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save the extracted data
    """
    print(f"Processing {input_file}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # The data is in a columnar format, convert it to a list of examples
    num_examples = len(data['contributor'])
    examples = []
    
    for i in range(num_examples):
        example = {
            'contributor': data['contributor'][i],
            'timestamp': data['timestamp'][i],
            'chat_format': data['chat_format'][i],
            'conversations': []
        }
        
        # Extract conversations
        for j in range(len(data['conversations'][i])):
            conversation = {
                'from': data['conversations'][i][j]['from'],
                'value': data['conversations'][i][j]['value']
            }
            example['conversations'].append(conversation)
        
        examples.append(example)
    
    # Save each example as a separate JSON file
    for i, example in enumerate(examples):
        output_file = os.path.join(output_dir, f"conversation_{i+1}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(example, f, ensure_ascii=False, indent=2)
    
    # Also save all examples in a single file
    all_examples_file = os.path.join(output_dir, "all_conversations.json")
    with open(all_examples_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(examples)} conversations to {output_dir}")
    
    # Print some statistics
    system_prompts = []
    human_messages = []
    gpt_responses = []
    
    for example in examples:
        for conv in example['conversations']:
            if conv['from'] == 'system':
                system_prompts.append(conv['value'])
            elif conv['from'] == 'human':
                human_messages.append(conv['value'])
            elif conv['from'] == 'gpt':
                gpt_responses.append(conv['value'])
    
    print(f"\nStatistics:")
    print(f"Total conversations: {len(examples)}")
    print(f"System prompts: {len(system_prompts)}")
    print(f"Human messages: {len(human_messages)}")
    print(f"GPT responses: {len(gpt_responses)}")
    
    # Print a sample conversation
    if examples:
        print("\nSample conversation:")
        sample = examples[0]
        for i, conv in enumerate(sample['conversations']):
            print(f"\n[{conv['from']}]")
            # Print just the first 100 characters if the message is long
            if len(conv['value']) > 100:
                print(f"{conv['value'][:100]}...")
            else:
                print(conv['value'])

if __name__ == "__main__":
    extract_sharegpt_conversations("sharegpt_data/train.json", "extracted_sharegpt_data")
