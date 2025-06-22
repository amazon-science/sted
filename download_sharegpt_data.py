from datasets import load_dataset
import os
import json

def download_sharegpt_structured_data(output_dir="sharegpt_data"):
    """
    Download the Arun63/sharegpt-structured-output-json dataset from Hugging Face
    and save it to the specified output directory.
    
    Args:
        output_dir (str): Directory to save the downloaded data
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("Arun63/sharegpt-structured-output-json")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save each split
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        
        # Convert to list of dictionaries for JSON serialization
        data_to_save = split_data.to_dict()
        
        # Save the full split as JSON
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(split_data)} examples to {output_file}")
        
        # Print some information about the dataset
        print(f"\nDataset split: {split_name}")
        print(f"Number of examples: {len(split_data)}")
        print(f"Features: {split_data.features}")
        
        # Show a sample example
        if len(split_data) > 0:
            print("\nSample example:")
            sample = split_data[0]
            sample_dict = {key: sample[key] for key in sample.keys()}
            print(json.dumps(sample_dict, indent=2))

if __name__ == "__main__":
    download_sharegpt_structured_data()
    print("\nDownload complete!")
