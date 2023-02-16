# Building a ChatBot with Deep-NLP
''' 
NOTE: Need Specific environment. Install following:
    conda create --name py354 python=3.5.4

    conda activate py354

    pip install protobuf==3.19.4

    pip install tensorflow==1.0.0

    conda install ipykernel

    conda install spyder
 '''

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


# ==============================     Part 3 : Trainig SEQ2SEQ model     ==============================

# Setting the HyperPartameters: We choosed these names for TensorFlow matching
epochs = 100    # or 50
batch_size = 64    # or 128
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5  # optimal according to Hinton

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loding the model Inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the Sequence Length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training predictions and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(ansWrd2Int),
                                                       len(questnWrd2Int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questnWrd2Int)

# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequences with the <PAD> token
    # Question: ["Who", "are", "you"]
    # Answer: [<SOS>, "I", "am", "a", "bot",".", <EOS>]

    # After we apply the padding this question and answer will become this:
        # Question: ["Who", "are", "you", <PAD>, <PAD>, <PAD>, <PAD>]
        # Answer: [<SOS>, "I", "am", "a", "bot",".", <EOS>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questnWrd2Int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, ansWrd2Int))
        yield padded_questions_in_batch, padded_answers_in_batch

# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000      # we choose 1000 to reach the last epoch i.e. 100-th epoch. Generally we set early_stopping 100
checkpoint = "./chatbot_weights.ckpt" # For Windows users, replace "chatbot_weights.ckpt" line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        
        # validation loss-error
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            
            # decay to the learning rates and early stopping
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")





# python prctc_ctbbt.py

