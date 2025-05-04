
#MIT License

#Copyright (c) 2025 Amirprx3, GameDevRichtofen-G

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


#-> Importing necessary libraries. argparse is for handling command-line arguments, os is for interacting with the operating system (like creating directories), torch and torch.nn are the core PyTorch libraries for building and training neural networks, torch.nn.functional has useful functions, DataLoader helps manage batches of data, transformers is from Hugging Face for pre-trained tokenizers and learning rate schedulers, datasets is also from Hugging Face for loading data, tqdm is for making nice progress bars, and NNmodel is where our custom Transformer model is defined (imported from a local file). json is for reading the dataset file.<-#
import argparse
import os
import torch
from torch import nn #-> Alias nn for torch.nn, common practice.<-#
from torch.utils.data import DataLoader #-> Helps create batches and shuffle data.<-#
from transformers import AutoTokenizer, get_linear_schedule_with_warmup #-> AutoTokenizer finds the right tokenizer based on a model name, get_linear_schedule_with_warmup is a learning rate scheduler.<-#
import torch.nn.functional as F #-> Often used for loss functions like cross_entropy.<-#
from datasets import load_dataset #-> A library to easily load and process datasets.<-#
from tqdm import tqdm #-> For showing cool progress bars during loops.<-#
from NNmodel import TRANSFORMER #-> Importing our custom Transformer model class from a separate file.<-#
import json #-> To handle the JSON dataset file.<-#

#-> This function sets up how we can run the script with different options from the command line, like changing the dataset path or model size.<-#
def parse_args():
    #-> Create an ArgumentParser object to handle command-line arguments.<-#
    parser = argparse.ArgumentParser(description="Train a custom Transformer on a JSON dataset")

    #-> Add arguments we want to be able to specify when running the script.<-#
    parser.add_argument("--dataset_path", type=str, default="EXAMPLE_CODE_DATASET.json",
                        help="Path to the local JSON dataset file in Code Alpaca format (list of objects)") #-> Path to the data file.<-#
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints and logs") #-> Where to save the trained model.<-#
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs") #-> How many times to go through the whole dataset.<-#
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training") #-> How many examples to process at once.<-#
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate") #-> How big steps the optimizer takes.<-#
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization and model") #-> Max length of input text we'll feed to the model. Longer texts get cut, shorter ones get padded.<-#
    parser.add_argument("--hidden_size", type=int, default=512, help="Model dimension (d_model)") #-> The size of the vectors used inside the transformer.<-#
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer decoder layers") #-> How many decoder blocks to stack.<-#
    parser.add_argument("--num_heads", type아트t=8, help="Number of attention heads") #-> How many attention heads in each multi-head attention layer.<-#
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate") #-> The probability of dropping out a neuron during training.<-#
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on (cuda or cpu)") #-> Automatically pick GPU if available, otherwise use CPU.<-#

    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate during testing") #-> How many tokens to generate when we test the model.<-#

    #-> Parse the actual arguments from the command line.<-#
    return parser.parse_args()

#-> This is the main function where the training process happens.<-#
def main():
    #-> Get all the arguments we defined.<-#
    args = parse_args()
    #-> Create the output directory if it doesn't exist.<-#
    os.makedirs(args.output_dir, exist_ok=True)
    #-> Set the device (GPU or CPU) based on the arguments.<-#
    device = torch.device(args.device)

    #-> Print which device we're using.<-#
    print(f"Using device: {device}")

    #-> Load a pre-trained tokenizer, specifically the one for gpt2. Tokenizers convert text into numbers (token IDs) that the model understands.<-#
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #-> GPT-2 tokenizer doesn't have a standard padding token. We set it to be the end-of-sequence (eos) token. This is important for batching sequences of different lengths.<-#
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token #-> Make sure we have a padding token.<-#

    #-> Get the size of the vocabulary, which is the number of unique tokens the tokenizer knows. This is needed for the model's output layer.<-#
    VOCAB_SIZE = len(tokenizer)
    print(f"Tokenizer vocabulary size: {VOCAB_SIZE}")

    #-> Load the dataset from the local JSON file using the datasets library.<-#
    print(f"Loading dataset from local file: {args.dataset_path}...")
    try:
        #-> load_dataset function can handle various formats, here 'json' for a JSON file.<-#
        dataset = load_dataset("json", data_files=args.dataset_path)

        #-> The Code Alpaca format typically results in a 'train' split. We check if it exists.<-#
        if 'train' not in dataset:
             raise ValueError(f"Dataset JSON file must result in a 'train' split after loading with datasets.load_dataset('json'). Ensure the file is a list of objects.")

        #-> Get the training split of the dataset.<-#
        train_dataset = dataset['train']
        print(f"Found {len(train_dataset)} training examples.")

    #-> Handle potential errors during dataset loading.<-#
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset_path}")
        print("Please ensure the dataset file is in the correct location.")
        return #-> Stop the script if the file isn't found.<-#
    except Exception as e:
        print(f"Error loading dataset from {args.dataset_path}: {e}")
        print("Please ensure the JSON file is correctly formatted (e.g., a list of objects, where each object has 'instruction', 'input', 'output').")
        return #-> Stop the script if loading fails for other reasons.<-#

    #-> This function defines how to process each example in the dataset (or a batch of examples because batched=True in .map).<-#
    def tokenize_function(examples):
        #-> Get 'instruction', 'input', and 'output' fields from the examples. Provide empty lists as default if keys are missing, matching the length of the batch.<-#
        instructions = examples.get('instruction', [''] * len(next(iter(examples.values()))))
        model_inputs = examples.get('input', [''] * len(instructions))
        responses = examples.get('output', [''] * len(instructions))

        #-> Combine the different parts of the data into a single text string for each example, following a specific format.<-#
        texts = []
        for instruction, model_input, response in zip(instructions, model_inputs, responses):
            #-> Start with the prompt format.<-#
            text = f"Prompt: {instruction}"
            #-> Add the input only if it exists.<-#
            if model_input:
                 text += f"\nInput: {model_input}"
            #-> Add the response and the end-of-sequence token. The model will learn to generate up to the EOS token.<-#
            text += f"\nResponse: {response}{tokenizer.eos_token}"

            texts.append(text)

        #-> Tokenize the list of text strings.<-#
        tokens = tokenizer(
            texts,
            padding=False, #-> Don't pad here; we'll pad later in the DataLoader.<-#
            truncation=True, #-> Cut off sequences longer than max_length.<-#
            max_length=args.max_length, #-> The maximum length to truncate to.<-#
            add_special_tokens=True #-> Add special tokens like BOS/EOS/PAD (tokenizer handles this based on config).<-#
        )

        #-> Return only the 'input_ids'. The datasets library expects a dictionary.<-#
        return {'input_ids': tokens['input_ids']}

    #-> Apply the tokenization function to the entire dataset. batched=True processes multiple examples at once, num_proc uses multiple processes for speed, and remove_columns cleans up the original text columns.<-#
    print("Tokenizing dataset...")

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True, #-> Apply the function to batches of examples.<-#
        num_proc=os.cpu_count(), #-> Use as many CPU cores as available for tokenization.<-#
        remove_columns=train_dataset.column_names #-> Remove the original text columns to save memory.<-#
    )
    print(f"Tokenization complete. Dataset size before filtering: {len(tokenized_dataset)}")

    #-> Filter out examples that ended up empty after tokenization (e.g., empty strings before formatting).<-#
    original_size = len(tokenized_dataset)
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example['input_ids']) > 0)
    filtered_size = len(tokenized_dataset)

    print(f"Filtered out {original_size - filtered_size} examples with empty token sequences.")
    print(f"Dataset size after filtering: {filtered_size}")
    #-> If all examples were filtered, something is wrong, so we stop.<-#
    if filtered_size == 0:
        print("Error: All examples were filtered out. Check your data, tokenizer, or max_length.")
        return

    #-> This function takes a batch of tokenized examples (which might have different lengths) and prepares them for the model.<-#
    def collate_fn(batch):
        #-> Extract the 'input_ids' list from the batch.<-#
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]

        #-> Pad the sequences to the maximum length in the batch. batch_first=True means the batch dimension is the first one (batch_size, seq_len).<-#
        padded_input_ids = nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id #-> Use the padding token ID for padding.<-#
        )

        #-> Create a padding mask: True where the token is a padding token, False otherwise. This mask tells the model to ignore padding tokens in attention.<-#
        padding_mask = (padded_input_ids == tokenizer.pad_token_id)

        #-> Move the tensors to the specified device (GPU or CPU).<-#
        padded_input_ids = padded_input_ids.to(device)
        padding_mask = padding_mask.to(device)

        #-> Return a dictionary containing the padded input IDs and the padding mask.<-#
        return {"input_ids": padded_input_ids, "padding_mask": padding_mask}

    #-> Create the DataLoader, which will handle batching and applying the collate_fn. Shuffle=True is important for training stability.<-#
    print(f"Creating DataLoader with batch size {args.batch_size}...")

    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print("DataLoader created.")

    #-> Initialize our custom Transformer model with the specified parameters.<-#
    print(f"Initializing model with vocab_size={VOCAB_SIZE}, hidden_size={args.hidden_size}, num_layers={args.num_layers}, num_heads={args.num_heads}, max_len={args.max_length}...")

    model = TRANSFORMER(
        vocab_size=VOCAB_SIZE,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_len=args.max_length, #-> Pass max_length to the model for learned positional embeddings.<-#
        dropout=args.dropout
    ).to(device) #-> Move the entire model to the specified device.<-#
    print("Model initialized.")

    #-> Set up the optimizer. AdamW is a common choice for training transformers.<-#
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #-> Calculate the total number of training steps (batches per epoch * number of epochs). This is needed for the scheduler.<-#
    num_training_steps = len(dataloader) * args.epochs

    #-> Calculate the number of warmup steps for the learning rate scheduler (usually a small percentage of total steps).<-#
    num_warmup_steps = int(num_training_steps * 0.05) #-> 5% of steps for warming up.<-#

    #-> Set up the learning rate scheduler. It gradually increases the learning rate from 0 during warmup and then linearly decreases it.<-#
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    #-> Start the training loop. Set the model to training mode.<-#
    model.train()
    print(f"Starting training for {args.epochs} epochs on {len(tokenized_dataset)} examples...")
    for epoch in range(args.epochs):
        total_loss = 0 #-> Keep track of the loss for the current epoch.<-#

        #-> Wrap the dataloader with tqdm to get a progress bar.<-#
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        #-> Iterate over batches in the dataloader.<-#
        for batch in progress_bar:
            #-> Get input IDs and padding mask from the collate_fn output.<-#
            input_ids = batch['input_ids']
            padding_mask = batch['padding_mask']

            #-> Zero out gradients from the previous step.<-#
            optimizer.zero_grad()

            #-> Perform the forward pass: feed the input IDs and padding mask to the model to get the output logits.<-#
            logits = model(input_ids, padding_mask=padding_mask)

            #-> Prepare labels for the loss calculation. For language modeling, we predict the *next* token. So, the labels are the input IDs shifted by one position.<-#
            labels = input_ids[:, 1:].contiguous() #-> Take tokens from index 1 onwards. contiguous() might be needed for view() later.<-#
            #-> Prepare predictions. The model's output for position `i` predicts the token at position `i+1`. So, we take logits up to the second-to-last position.<-#
            predictions = logits[:, :-1, :].contiguous() #-> Take logits up to the second-to-last token position.<-#

            #-> Reshape the predictions and labels to calculate loss easily. We flatten everything except the vocabulary dimension for predictions.<-#
            predictions = predictions.view(-1, predictions.size(-1)) #-> Shape: (batch_size * (seq_len-1), vocab_size) <-#
            labels = labels.view(-1) #-> Shape: (batch_size * (seq_len-1)) <-#

            #-> Calculate the Cross-Entropy loss. This is standard for classification tasks like predicting the next token. ignore_index tells the loss function to ignore padding tokens in the labels.<-#
            loss = F.cross_entropy(predictions, labels, ignore_index=tokenizer.pad_token_id)

            #-> Perform the backward pass: calculate gradients.<-#
            loss.backward()

            #-> Clip gradients to prevent them from becoming too large (exploding gradients), which can destabilize training.<-#
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #-> Update model weights using the optimizer based on the calculated gradients.<-#
            optimizer.step()

            #-> Update the learning rate according to the scheduler.<-#
            scheduler.step()

            #-> Add the current batch's loss to the total loss for the epoch.<-#
            total_loss += loss.item()
            #-> Update the progress bar description with the current loss and learning rate.<-#
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

        #-> Calculate and print the average loss for the epoch.<-#
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{args.epochs} finished. Average Loss: {avg_loss:.4f}")

    #-> After training finishes, save the model's state dictionary (the learned parameters) to a file.<-#
    model_path = os.path.join(args.output_dir, "transformer_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

   
    #-> Now, let's test the trained model's ability to generate text.<-#
    print("\n--- Testing Generation ---")

    #-> This function handles the text generation process.<-#
    def generate_sample(model, tokenizer, prompt_text, max_new_tokens=100,
                        temperature=0.8, top_k=50, device="cpu", max_model_length=512):
        #-> Set the model to evaluation mode. This turns off dropout and batch normalization updates (though we don't have BN here).<-#
        model.eval()

        #-> Format the prompt the same way we formatted the training data.<-#
        input_text = f"Prompt: {prompt_text}\nResponse:"

        #-> Tokenize the initial prompt.<-#
        input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True).to(device) #-> Convert text to tensor and move to device.<-#

        #-> Keep track of the original prompt length.<-#
        initial_prompt_len = input_ids.shape[1]

        #-> Check if the initial prompt is already too long for the model.<-#
        if initial_prompt_len >= max_model_length:
             print(f"Warning: Initial prompt length ({initial_prompt_len}) exceeds max_length ({max_model_length}). Cannot generate.")
             #-> Decode and return the truncated prompt if it was too long.<-#
             decoded_prompt = tokenizer.decode(input_ids[0][:max_model_length], skip_special_tokens=False)
             print(f"Prompt that was too long (truncated for display): {decoded_prompt}")
             return decoded_prompt
        #-> The tensor `generated_ids` will hold the sequence as it's being generated. Start with the prompt.<-#
        generated_ids = input_ids

        print("Generating text...")

        #-> Loop to generate one token at a time up to max_new_tokens.<-#
        for _ in range(max_new_tokens):
            #-> Get the current length of the sequence.<-#
            current_seq_len = generated_ids.shape[1]

            #-> Stop if the generated sequence plus the prompt reaches the model's maximum capacity.<-#
            if current_seq_len >= max_model_length:
                print(f"Reached maximum sequence length ({max_model_length}) during generation. Stopping.")
                break

            #-> Disable gradient calculation during inference (generation) to save memory and speed up.<-#
            with torch.no_grad():
                #-> Feed the *entire current sequence* (prompt + generated tokens so far) into the model. The model predicts the *next* token based on this sequence.<-#
                logits = model(generated_ids)

            #-> Get the logits for the very last token in the sequence. These are the probabilities/scores for the *next* token.<-#
            next_token_logits = logits[:, -1, :] #-> Shape: (batch_size, vocab_size) - but we have batch_size 1 for generation.<-#

            #-> Decide how to pick the next token (greedy or sampling).<-#
            if temperature == 0:
                 #-> Greedy decoding: Pick the token with the highest probability.<-#
                 next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) #-> argmax gets the index, unsqueeze adds batch dim.<-#
            else:
                 #-> Sampling: Apply temperature to the logits to make the distribution sharper (low temp) or flatter (high temp).<-#
                 next_token_logits = next_token_logits / temperature

                 #-> Apply Top-K sampling: only consider the top_k most likely tokens before sampling.<-#
                 if top_k is not None and top_k > 0:
                     #-> Get the values and indices of the top_k tokens.<-#
                     topk_vals, topk_idx = torch.topk(next_token_logits, top_k, dim=-1)
                     #-> Set all other logits to -inf so their probability becomes zero after softmax.<-#
                     next_token_logits = torch.full_like(next_token_logits, float('-inf'), dtype=next_token_logits.dtype)
                     next_token_logits.scatter_(-1, topk_idx, topk_vals) #-> Put the top_k logits back.<-#

                 #-> Convert logits to probabilities using softmax.<-#
                 probs = F.softmax(next_token_logits, dim=-1)
                 #-> Sample the next token from the probability distribution.<-#
                 next_token = torch.multinomial(probs, num_samples=1) #-> Samples one token index.<-#

            #-> Append the newly generated token to the sequence.<-#
            generated_ids = torch.cat([generated_ids, next_token], dim=1) #-> Concatenate along the sequence dimension.<-#

            #-> Stop generation if the end-of-sequence token is generated.<-#
            if next_token.item() == tokenizer.eos_token_id:
                print("Generated EOS token. Stopping generation.")
                break #-> Exit the generation loop.<-#

        #-> Decode the final tensor of token IDs back into a human-readable string.<-#
        output_text_with_prompt = tokenizer.decode(generated_ids[0], skip_special_tokens=False) #-> Decode the first sequence in the batch (since batch_size is 1).<-#

        #-> Return the generated text.<-#
        return output_text_with_prompt

    #-> Define a few test prompts to see how the model performs.<-#
    test_prompts = [
        "Write Python code to print the classic 'Hello, world!' message.",
        "Create a function in Python that calculates the factorial of a number.",
        "Define a list with elements 10, 20, 30.",
        "Write a comment in Python.",
        "Formulate an equation to calculate the height of a triangle given the angle, side lengths and opposite side length.", #-> Might struggle with non-code prompts depending on training data.<-#
        "Write a Python class called 'Book' with attributes title and author.",
        "How to open and read a file in Python?",
        "Generate a list comprehension for squares of numbers from 1 to 5.",
        "Write an if-else statement in Python checking if a number is positive.",
        "Define a dictionary in Python representing a person with name and age."
    ]

    #-> Loop through the test prompts and generate text for each one.<-#
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test Prompt {i+1} ---")
        print(f"Prompt: {prompt}")

        #-> Call the generation function with the current prompt and specified parameters.<-#
        generated_text = generate_sample(model, tokenizer, prompt,
                                         max_new_tokens=args.max_new_tokens,
                                         device=device,
                                         max_model_length=args.max_length)

        #-> Print the generated text.<-#
        print(f"Generated Text:\n{generated_text}")

    print("\n--- Testing Complete ---")


#-> This is the standard Python entry point. It ensures that the main function is called only when the script is executed directly (not when imported as a module).<-#
if __name__ == "__main__":
    main()