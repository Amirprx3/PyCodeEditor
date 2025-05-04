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

#-> We need these libraries again: torch for the model, AutoTokenizer from huggingface to load the tokenizer, os for path handling, and the TRANSFORMER class from our local file.<-#
import torch
from transformers import AutoTokenizer #-> Need the tokenizer to encode the prompt and decode the output.<-#
#-> We need to import the model definition from where we saved it. Make sure NNmodel.py is in the same directory or path.<-#
from NNmodel import TRANSFORMER #-> Import our custom model class definition.<-#
import torch.nn.functional as F #-> Needed for softmax during sampling.<-#
#-> We might need math if our positional encoding in NNmodel was the sine/cosine type, but the final TRANSFORMER class used learned embeddings, so probably not needed directly here.<-#
# import math #-> Might not be strictly necessary depending on the model definition.<-#

#->This is function we usse in our editor.py to ask ai<-#
#-> This is the main function we'll call to generate text based on a prompt.<-#
def predict(model_path, tokenizer_name="gpt2", prompt_text="Write a Python function.",
                        max_new_tokens=100, temperature=0.8, top_k=50,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        model_params=None):
    #-> This function needs the path to the saved model weights, the tokenizer name, the text to start generating from, and some optional parameters for how generation works.<-#

    #-> Set the device to use (GPU if available, otherwise CPU).<-#
    device = torch.device(device)
    print(f"Using device: {device}")

    #-> Load the tokenizer we used during training.<-#
    print(f"Loading tokenizer '{tokenizer_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        #-> Make sure the padding token is set, just like in training.<-#
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer {tokenizer_name}: {e}")
        return "Error loading tokenizer."

    #-> Define the model hyperparameters. These MUST match the ones used to train the saved model.<-#
    #-> We'll use the default values from the training script's argparse as a starting point.<-#
    if model_params is None:
        #-> If no params are provided, use these defaults. In a real app, load these from a saved config file from training!<-#
        print("Using default model parameters. Ensure these match your trained model!")
        model_params = {
            "vocab_size": len(tokenizer), #-> Get vocab size from the loaded tokenizer.<-#
            "hidden_size": 512,
            "num_layers": 4,
            "num_heads": 8,
            "max_len": 512, #-> Max sequence length used during training.<-#
            "dropout": 0.05
        }
    else:
         #-> Update vocab size from tokenizer, as it's the source of truth for prediction.<-#
         model_params["vocab_size"] = len(tokenizer)
         print(f"Using provided model parameters: {model_params}")


    #-> Instantiate the model architecture using the parameters.<-#
    print(f"Initializing model architecture...")
    try:
        model = TRANSFORMER(**model_params).to(device) #-> Create the model instance and move it to the device.<-#
        print("Model architecture initialized.")
    except Exception as e:
         print(f"Error initializing model architecture: {e}")
         return "Error initializing model architecture."


    #-> Load the saved weights into the model.<-#
    print(f"Loading model weights from '{model_path}'...")
    try:
        #-> Load the state dictionary. map_location ensures it's loaded onto the correct device.<-#
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict) #-> Put the weights into the model instance.<-#
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_path}")
        print("Please ensure the model file exists in the specified path.")
        return "Error: Model file not found."
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return "Error loading model weights."


    #-> Set the model to evaluation mode. This is important for consistent results during inference (like turning off dropout).<-#
    model.eval()

    #-> Format the input prompt exactly how it was formatted in the training data for the 'Response' part.<-#
    #-> The training format was "Prompt: {instruction}\nInput: {model_input}\nResponse: {response}<|endoftext|>".<-#
    #-> We are giving it "Prompt: {prompt_text}\nResponse:", expecting it to generate the "{response}<|endoftext|>" part.<-#
    input_text = f"Prompt: {prompt_text}\nResponse:"
    print(f"Inputting formatted text:\n---\n{input_text}\n---")

    #-> Encode the input text into token IDs.<-#
    #-> Add special tokens like BOS/EOS if the tokenizer uses them (GPT-2 uses EOS).<-#
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True).to(device) #-> Encode text, get a PyTorch tensor, and move it to the device.<-#

    #-> Keep track of the initial length of the prompt.<-#
    initial_prompt_len = input_ids.shape[1]
    max_model_length = model_params["max_len"] #-> Get max length from params.<-#


    #-> Check if the prompt itself is already too long.<-#
    if initial_prompt_len >= max_model_length:
         print(f"Warning: Initial prompt length ({initial_prompt_len}) exceeds max_length ({max_model_length}). Cannot generate.")
         decoded_prompt = tokenizer.decode(input_ids[0][:max_model_length], skip_special_tokens=False)
         print(f"Truncated Prompt for display: {decoded_prompt}")
         #-> Return the prompt itself or a message indicating it was too long.<-#
         return f"Prompt too long ({initial_prompt_len} tokens). Max allowed is {max_model_length}. Original prompt: '{prompt_text}'"

    #-> The tensor `generated_ids` will hold the sequence as it's being built. Start with the encoded prompt.<-#
    generated_ids = input_ids

    print(f"Generating up to {max_new_tokens} new tokens...")

    #-> Loop to generate one token at a time.<-#
    for _ in range(max_new_tokens):
        #-> Get the current length of the sequence being generated.<-#
        current_seq_len = generated_ids.shape[1]

        #-> If the sequence reaches the model's maximum length, stop generating to avoid errors or unexpected behavior.<-#
        if current_seq_len >= max_model_length:
            print(f"Reached maximum sequence length ({max_model_length}) during generation. Stopping.")
            break #-> Exit the generation loop.<-#

        #-> Use torch.no_grad() because we are only doing inference, not training, so we don't need to calculate gradients.<-#
        with torch.no_grad():
            #-> Feed the current sequence (prompt + generated tokens so far) into the model. The model predicts the next token.<-#
            #-> Our Transformer model takes input_ids and padding_mask. Since we're generating one sequence, padding isn't strictly needed inside the model for a single sequence, but the function expects it. A simple way is to pass None or a tensor of False. Let's pass None for simplicity here, assuming the DecoderBlock can handle it (or rely on batch_first=True in the second DecoderBlock definition). If the padding mask *is* required and None causes errors, we'd need to create a mask of all False for the single sequence. Let's stick with None for now as the model's causal mask is the important one during generation.<-#
            logits = model(generated_ids, padding_mask=None) #-> Forward pass to get output logits.<-#

        #-> Get the logits (scores) for the *last* token in the sequence. These predict the *next* token.<-#
        next_token_logits = logits[:, -1, :] #-> Shape: (batch_size, vocab_size). Batch size is 1 here.<-#

        #-> Apply temperature for sampling if temperature > 0. Temperature makes the probability distribution sharper (low temp) or smoother (high temp).<-#
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

            #-> Apply Top-K sampling if top_k is set. Only consider the top K most likely next tokens.<-#
            if top_k is not None and top_k > 0:
                #-> Get the top K logits and their indices.<-#
                topk_vals, topk_idx = torch.topk(next_token_logits, top_k, dim=-1)
                #-> Set the logits of tokens *not* in the top K to a very small number so they are effectively ignored after softmax.<-#
                next_token_logits = torch.full_like(next_token_logits, float('-inf'), dtype=next_token_logits.dtype)
                #-> Put the top K logits back.<-#
                next_token_logits.scatter_(-1, topk_idx, topk_vals)

            #-> Convert logits to probabilities using softmax.<-#
            probs = F.softmax(next_token_logits, dim=-1)
            #-> Sample the next token index from the probability distribution.<-#
            next_token = torch.multinomial(probs, num_samples=1) #-> Samples one token ID.<-#
        else:
            #-> If temperature is 0, use greedy decoding: just pick the token with the highest logit/probability.<-#
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) #-> Get index of max value, add batch dimension.<-#


        #-> Append the newly generated token ID to our sequence tensor.<-#
        generated_ids = torch.cat([generated_ids, next_token], dim=1) #-> Concatenate along the sequence length dimension.<-#

        #-> Check if the generated token is the end-of-sequence token. If so, stop generating.<-#
        if next_token.item() == tokenizer.eos_token_id:
            print("Generated EOS token. Stopping generation.")
            break #-> Exit the loop.<-#

    #-> Decode the final tensor of token IDs (which includes the prompt and the generated text) back into a string.<-#
    #-> skip_special_tokens=False means we keep tokens like <|endoftext|>, which is useful for knowing where generation stopped.<-#
    output_text_with_prompt = tokenizer.decode(generated_ids[0], skip_special_tokens=False) #-> Decode the first (and only) sequence in the batch.<-#

    #-> Find the start of the generated response text by looking for the "Response:" part of the prompt.<-#
    #-> We add len("Response:") to get the index *after* the colon and space.<-#
    response_start_index = output_text_with_prompt.find("\nResponse:")
    if response_start_index != -1:
        #-> Add the length of "\nResponse:" to get the actual start index of the generated part.<-#
        generated_response_text = output_text_with_prompt[response_start_index + len("\nResponse:"):].strip() #-> Get the text after "Response:". strip() removes leading/trailing whitespace.<-#
    else:
        #-> If "Response:" wasn't found (shouldn't happen if prompt format is consistent), return the whole thing or an error.<-#
        print("Warning: 'Response:' marker not found in generated text. Returning full output.")
        generated_response_text = output_text_with_prompt


    #-> Return the generated text (ideally just the response part).<-#
    return generated_response_text

#-> This block runs only when the script is executed directly.<-#
if __name__ == "__main__":
    #-> Define the path where the trained model was saved.<-#
    SAVED_MODEL_PATH = "./output/transformer_model.pt"
    #-> Define the name of the tokenizer used.<-#
    TOKENIZER_NAME = "gpt2"

    #-> Define the hyperparameters used for the trained model. <-#
    #-> !! IMPORTANT !! These MUST exactly match the parameters used when training the model. <-#
    #-> In a real application, you would save these parameters during training (e.g., to a JSON file) and load them here.<-#
    #-> Using defaults from the training script:<-#
    MODEL_PARAMS = {
        # vocab_size will be set by the function based on the tokenizer
        "hidden_size": 512,
        "num_layers": 4,
        "num_heads": 8,
        "max_len": 512, #-> Max sequence length is important!<-#
        "dropout": 0.05
    }

    #-> Define generation parameters.<-#
    GEN_PARAMS = {
        "max_new_tokens": 200, #-> Generate up to 200 new tokens.<-#
        "temperature": 0.8, #-> Controls randomness (0.0 is greedy).<-#
        "top_k": 50, #-> Consider only the top 50 most likely tokens at each step.<-#
        "device": "cuda" if torch.cuda.is_available() else "cpu" #-> Use GPU if available.<-#
    }


    #-> Example usage: Define a prompt you want the model to complete.<-#
    input_prompt = "Write a Python function that reverses a string."

    print(f"Attempting to load model from {SAVED_MODEL_PATH} and predict for prompt:\n'{input_prompt}'")

    # #-> Call the function to generate text.<-#
    # generated_output = predict(
    #     model_path=SAVED_MODEL_PATH,
    #     tokenizer_name=TOKENIZER_NAME,
    #     prompt_text=input_prompt,
    #     max_new_tokens=GEN_PARAMS["max_new_tokens"],
    #     temperature=GEN_PARAMS["temperature"],
    #     top_k=GEN_PARAMS["top_k"],
    #     device=GEN_PARAMS["device"],
    #     model_params=MODEL_PARAMS #-> Pass the stored model parameters.<-#
    # )

    # #-> Print the result.<-#
    # print("\n--- Generated Response ---")
    # print(generated_output)
    # print("--------------------------")