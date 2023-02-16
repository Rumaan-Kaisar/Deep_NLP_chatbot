# Building a ChatBot with Deep-NLP


# Impoting the libraries
import numpy as np
import tensorflow as tf
import re
import time


# ==============================     Part 1 : Data Preprocessing     ==============================
# Importing the dataset
lines_in_cnvrstn = open("../data_set/movie_lines.txt", encoding='utf-8', errors='ignore').read().split('\n')
cnvrstn = open("../data_set/movie_conversations.txt", encoding='utf-8', errors='ignore').read().split('\n')


# Creating a dictionary that maps each linea and its ID
id_2_line = {};
for lyn in lines_in_cnvrstn:
    _line = lyn.split(" +++$+++ ")
    if len(_line) == 5:
        id_2_line[_line[0]] = _line[4] # creates the dicttionary
        # _line[0] is id "key in dictionary" and _line[4] is the line "as value of the key"


# Creating the list of all of the conversations
cnvrstn_ids = []
for cvstn in cnvrstn[:-1]:
    _cnvrstn =  cvstn.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    cnvrstn_ids.append(_cnvrstn.split(","))


# Getting seperately the questions and answers
raw_questn = []
raw_ans = []
for cvstn in cnvrstn_ids:
    for i in range(len(cvstn) - 1):
        raw_questn.append(id_2_line[cvstn[i]])  # using the "id_2_line" dictionary, by id-key
        raw_ans.append(id_2_line[cvstn[i+1]]) 
        # range(len(cvstn) - 1) is used because of cvstn[i+1]
        # notice we are using both "cnvrstn_ids" and "id_2_line"


# Doing the first cleaning of the texts
def clean_text(text):
    text = text.lower();
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text) 
    text = re.sub(r"she's", "she is", text) 
    text = re.sub(r"that's", "that is", text) 
    text = re.sub(r"what's", "what is", text) 
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text) 
    text = re.sub(r"\'ve", " have", text) 
    text = re.sub(r"\'re", " are", text) 
    text = re.sub(r"\'d", " would", text) 
    text = re.sub(r"won't", "will not", text) 
    text = re.sub(r"can't", "cannot", text) 
    text = re.sub(r"[-()#/@;:<>{}+=~|.?,]", "", text)
    return text


# Cleaning the questions 
clean_questn = []
for qes in raw_questn:
    clean_questn.append(clean_text(qes))

# Cleaning the answers
clean_ans = []
for aNs in raw_ans:
    clean_ans.append(clean_text(aNs))


# create a dictionary that maps each word to its number of occurrences.
word2count = {}
for questn in clean_questn:
    for word in questn.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for ans in clean_ans:
    for word in ans.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


# Creating two dictionaries that map the questions words and the answers words to a unique integers
threshold = 20
word_number = 0
questnWrd2Int = {}
# 'word' is "key" and 'count' is "value"
for word, count in word2count.items():
    if count >= threshold:
        questnWrd2Int[word] = word_number
        word_number += 1

word_number = 0
ansWrd2Int = {}
for word, count in word2count.items():
    if count >= threshold:
        ansWrd2Int[word] = word_number
        word_number += 1


# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for tkn in tokens:
    # adding 'token' as "key" and 'unique int' as "value"
    questnWrd2Int[tkn] = len(questnWrd2Int) + 1

for tkn in tokens:
    # adding 'token' as "key" and 'unique int' as "value"
    ansWrd2Int[tkn] = len(ansWrd2Int) + 1


# Creating inverse dictionary of the ansWrd2Int, notice ':' is used to crete dictionary
ans_Int_2_Wrd = {w_i: w for w, w_i in ansWrd2Int.items()}

# Adding EOS tokens to the end of every answers
for i in range(len(clean_ans)):
    clean_ans[i] += " <EOS>" # notice a space is added to seperate <EOS>


# Translating all the Questions and the Answers into integers
# and replacing all the words that were filtered out by our token <OUT>
questions_to_int = []
for question in clean_questn:
    ints = []
    for word in question.split():
        if word not in questnWrd2Int:
            # Checking filtered word and adding <OUT>'s token
            ints.append(questnWrd2Int["<OUT>"])
        else:
            # Adding word's corresponding token
            ints.append(questnWrd2Int[word])
    # Finally "ints" is containing a tokenized question sentence, 
    # we append this to "questions_to_int"
    questions_to_int.append(ints)

# We do same for the Answers
answers_to_int = []
for answer in clean_ans:
    ints = []
    for word in answer.split():
        if word not in ansWrd2Int:
            ints.append(ansWrd2Int["<OUT>"])
        else:
            ints.append(ansWrd2Int[word])
    answers_to_int.append(ints)


# Sorting the both questions and answers by the length of the questions
max_line_length = 25
sorted_clean_questions = []
sorted_clean_answers = []
# print(enumerate(questions_to_int));
enumarated = list(enumerate(questions_to_int))
for length in range(1, max_line_length+1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])





# ==============================     Part 2 : building SEQ2SEQ model     ==============================

# creating TF placeholder for inputs and target
def model_inputs():
    inpUts = tf.placeholder(tf.int32, [None, None], name="input")
    tarGets = tf.placeholder(tf.int32, [None, None], name="target")
    lr = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return inpUts, tarGets, lr, keep_prob

# preprocessiong the tergets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int["<SOS>"])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# ------------------ Creating the ENCODER RNN layer ----------------
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    # encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                        cell_bw = encoder_cell, 
                                                        sequence_length = sequence_length, 
                                                        inputs=rnn_inputs, 
                                                        dtype = tf.float32)
    return encoder_state

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
                                                                                                                    attention_states, 
                                                                                                                    attention_option = "bahdanau", 
                                                                                                                    num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test/validatin set
def decode_test_set(encoder_state, decoder_cell, decoder_embedding_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
                                                                                                                    attention_states, 
                                                                                                                    attention_option = "bahdanau", 
                                                                                                                    num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
                                                                                output_function, 
                                                                                encoder_state[0],
                                                                                attention_keys,
                                                                                attention_values,
                                                                                attention_score_function,
                                                                                attention_construct_function,
                                                                                decoder_embedding_matrix, 
                                                                                sos_id, 
                                                                                eos_id, 
                                                                                maximum_length, 
                                                                                num_words,
                                                                                name = "attn_dec_inf")
    test_prediction, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder( decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_prediction


# ------------------ Creating the DECODER RNN layer ----------------
def decoder_rnn(decoder_embedded_input, decoder_embedding_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int,  keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)   # define a layer
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)     # Apply dropout on the layer
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)   # Creating stack of the layer
        # Intialize weight
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(  x,
                                                                        num_words,
                                                                        None,
                                                                        scope = decoding_scope,
                                                                        weights_initializer = weights,
                                                                        biases_initializer = biases)
        # Training Predictions
        training_predictions = decode_training_set( encoder_state, 
                                                    decoder_cell, 
                                                    decoder_embedded_input, 
                                                    sequence_length, 
                                                    decoding_scope, 
                                                    output_function, 
                                                    keep_prob, 
                                                    batch_size)
        # Test Predictions
        decoding_scope.reuse_variables()
        test_prediciton = decode_test_set(  encoder_state, 
                                            decoder_cell, 
                                            decoder_embedding_matrix, 
                                            word2int['<SOS>'], 
                                            word2int['<EOS>'], 
                                            sequence_length -1 , 
                                            num_words, 
                                            decoding_scope, 
                                            output_function, 
                                            keep_prob, 
                                            batch_size)
    return training_predictions, test_prediciton


''' 
--- Python Lambda ---
    A lambda function is a small anonymous function. A lambda function can take any number of arguments, but can only have one expression.

Syntax
        lambda arguments : expression
    
    The expression is executed and the result is returned

Example:
    Add 10 to argument a, and return the result:

    x = lambda a : a + 10
    print(x(5))  

'''

# Building the seq2seq model : Brain of our chatbot
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questnWrd2Int):
    # assemblage = putting togather
    encoder_embedded_input = tf.contrib.layers.embed_sequence(  inputs,
                                                                answers_num_words+1,
                                                                encoder_embedding_size,
                                                                initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questnWrd2Int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    traing_pred, test_pred = decoder_rnn(   decoder_embedded_input,
                                            decoder_embeddings_matrix, 
                                            encoder_state, 
                                            questions_num_words,
                                            sequence_length,
                                            rnn_size,
                                            num_layers,
                                            questnWrd2Int,
                                            keep_prob,
                                            batch_size)
    return traing_pred, test_pred



# python prctc_ctbbt.py

