def convert_single_abstract_to_embedding(tokenizer, model, in_text, MAX_LEN = 510):
    
    input_ids = tokenizer.encode(
                        in_text, 
                        add_special_tokens = True, 
                        max_length = MAX_LEN,                           
                   )    

    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", 
                              truncating="post", padding="post")
    
    # Remove the outer list.
    input_ids = results[0]

    # Create attention masks    
    attention_mask = [int(i>0) for i in input_ids]
    
    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    # Add an extra dimension for the "batch" (even though there is only one 
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():        
        logits, encoded_layers = model(
                                    input_ids = input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]
        
    # Extract the embedding.
    embedding = encoded_layers[layer_i][batch_i][token_i]

    # Move to the CPU and convert to numpy ndarray.
    embedding = embedding.detach().cpu().numpy()

    return(embedding)


def convert_overall_text_to_embedding(df):
    
    # The list of all the embeddings
    embeddings = []
    
    # Get overall text data
    overall_text_data = data.abstract.values
    
    # Loop over all the comment and get the embeddings
    for abstract in tqdm(overall_text_data):
        
        # Get the embedding 
        embedding = convert_single_abstract_to_embedding(sciBERT_tokenizer, model, abstract)
        
        #add it to the list
        embeddings.append(embedding)
        
    print("Conversion Done!")
    
    return embeddings