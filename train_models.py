import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_all_labels(file_path="true_labels.csv"):
    ''' Reads csv file where true labels are stored. Each row contains: filename, label  '''

    annotations = {}
    annotations_file = open(file_path, "r")
    for line in annotations_file:
        tokens = line.replace("\n", "").split(",")
        annotations[tokens[0]] = tokens[1]
    return annotations


def get_subset_labels(filenames):
    all_labels = get_all_labels()
    labels = np.zeros(filenames.shape)

    for i in range(filenames.shape[0]):
        labels[i] = all_labels[filenames[i].split("_NW_")[0] + ".wav"]

    return labels


def create_padded_rep(dict_rep, max_len, mask_val):
    ''' Creates representation with shape [examples, max_len segments, embedding].
        If an audio file is shorter than max_len, it pads using mask_val
    '''

    # TODO: Find better way to concatenate all audio embeddings
    all_embeddings = None
    for i, filename in enumerate(dict_rep):
        example = dict_rep[filename]
        if all_embeddings is None:
            all_embeddings = example
        else:
            all_embeddings = np.concatenate((all_embeddings, example), axis=0)

    # Get first example in dict to determine emb_dim
    example = list(dict_rep.values())[0]
    n_examples = len(dict_rep)
    emb_dim = example.shape[-1]

    filenames = []
    dataset = np.full((n_examples, max_len, emb_dim), mask_val, dtype=example.dtype)

    for i, filename in enumerate(dict_rep):
        example = dict_rep[filename]

        rep = np.full((max_len, emb_dim), mask_val, dtype=example.dtype)

        if example.shape[0] < max_len:
            rep[:example.shape[0], :] = example
        else:
            rep[:, :] = example[:max_len, :]

        dataset[i] = rep
        filenames.append(filename)

    # Sort audio files and corresponding representations to ensure all experiments use same splits
    filenames = np.array(filenames)
    idx = np.argsort(filenames)
    filenames, dataset = filenames[idx], dataset[idx]

    return filenames, dataset


def create_avg_rep(dict_rep):
    ''' Averages audio segment embeddings to create fixed-sized audio representations '''
    example = list(dict_rep.values())[0]

    n_examples = len(dict_rep)
    n_feats = example.shape[-1]

    filenames = []
    dataset = np.full((n_examples, n_feats), 0.0, dtype=example.dtype)

    for i, filename in enumerate(dict_rep):
        example = dict_rep[filename]
        avg_rep = example

        # Compute mean when embeddings are stored as [segments, embeddings]
        if example.ndim == 2:
            avg_rep = np.mean(example, axis=0)

        dataset[i] = avg_rep
        filenames.append(filename)

    # Sort audio files and corresponding representations to ensure all experiments use same splits
    filenames = np.array(filenames)
    idx = np.argsort(filenames)
    filenames, dataset = filenames[idx], dataset[idx]

    return filenames, dataset


def get_rnn_model(input_shape, learning_rate, mask_val=-99.0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=mask_val, input_shape=input_shape))
    model.add(tf.keras.layers.GaussianNoise(stddev=0.2))
    model.add(tf.keras.layers.GRU(32, return_sequences=True, dropout=0.25, kernel_regularizer=tf.keras.regularizers.l2(0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.GRU(32, dropout=0.25, kernel_regularizer=tf.keras.regularizers.l2(0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def get_linear_model(input_shape, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.GaussianNoise(stddev=0.2))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def run_experiment(x_all, filenames, build_model, results_path):
    learning_rate = 1e-3
    batch_size = 64
    epochs = 500

    np.random.seed(0)  # For reproducibility purposes

    # Shuffle dataset
    indices = np.arange(x_all.shape[0])
    shuffled_indices = np.random.permutation(indices)

    x_all = x_all[shuffled_indices]
    filenames = filenames[shuffled_indices]

    # Evaluate using 20-fold cross-validation
    k = 20

    # Use 10% of data as validation
    val_size = 0.10

    # Recalculate val_size to account for 95-5 split
    val_size = val_size / (1 - 1/k)
    test_idx_lst = np.array_split(np.arange(x_all.shape[0]), k)

    # Create results csv file where each row contains: filename, predicted label, probability
    open(results_path, "w")
    final_test_acc = 0

    for i, test_idx in enumerate(test_idx_lst):
        print()
        print("---------------")
        print("Fold #", (i + 1), "/", k)
        print("---------------")
        print()

        # Test set for the current fold
        x_test = x_all[test_idx]
        test_utterance_names = filenames[test_idx]
        y_test = get_subset_labels(test_utterance_names)

        mask_test = np.ones(x_all.shape[0], bool)
        mask_test[test_idx] = False

        x_train_val = x_all[mask_test]
        train_utterance_names = filenames[mask_test]
        y_train_val = get_subset_labels(train_utterance_names)

        # Divide into train and val sets multiple times while keeping track of best performing model
        best_acc, best_model = 0, None

        for j in range(k):
            x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=val_size,
                                                                  shuffle=True, random_state=j)

            model = build_model(x_all.shape[1:], learning_rate)

            # Use early stopping to keep track of best performing model via restore_best_weights=True
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=epochs,
                mode='max',
                restore_best_weights=True
            )

            # Train model
            model.fit(
                x=x_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_valid, y_valid),
                verbose=0,
                callbacks=[early_stopping]
            )

            eval_results = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)

            if eval_results[1] > best_acc:
                best_acc = eval_results[1]
                best_model = model

        y_prob = best_model.predict(x_test)
        y_pred = tf.math.round(y_prob).numpy()

        test_acc = accuracy_score(y_test, y_pred)
        final_test_acc += test_acc

        print("Fold Test accuracy", accuracy_score(y_test, y_pred))

        results_file = open(results_path, "a")
        for file_id in range(test_utterance_names.shape[0]):
            utt_name = test_utterance_names[file_id]
            pred_label = y_pred[file_id]
            results_file.write(utt_name + "," + str(pred_label) + "," + str(y_prob[file_id, 0]) + "\n")
        results_file.close()

    final_test_acc /= k
    print()
    print("Final Test Accuracy:", final_test_acc)


def run_rnn_experiment(path, max_secs, frame_dur_sec, results_path, mask_val=-99.0):

    with open(path, "rb") as input_file:
        reps = pickle.load(input_file)

    max_len = round(max_secs / frame_dur_sec)
    filenames, padded_reps = create_padded_rep(reps, max_len, mask_val)
    x_all = padded_reps

    build_model = get_rnn_model

    run_experiment(x_all, filenames, build_model, results_path)


def run_linear_experiment(path, results_path):
    # Read embeddings dictionary - key=filename, value=extracted embedding(s)
    with open(path, "rb") as input_file:
        reps = pickle.load(input_file)

    filenames, avg_reps = create_avg_rep(reps)
    x_all = avg_reps

    build_model = get_linear_model

    run_experiment(x_all, filenames, build_model, results_path)


def main():
    # Run logistic regression experiments
    saved_embeddings_paths = ["t19_reps.pkl", "ts5_full_reps.pkl", "ts5_960_reps.pkl", "cola_reps.pkl"]
    lr_result_paths = ["t19_lr_results.csv", "ts5_full_lr_results.csv", "ts5_960_lr_results.csv", "cola_lr_results.csv"]

    for emb_path, results_path in zip(saved_embeddings_paths, lr_result_paths):
        run_linear_experiment(emb_path, results_path)

    # Run rnn experiments
    saved_embeddings_paths = ["t19_reps.pkl", "ts5_960_reps.pkl", "cola_reps.pkl"]
    rnn_result_paths = ["t19_rnn_results.csv", "ts5_960_rnn_results.csv", "cola_rnn_results.csv"]

    # Truncate audio files longer than 10s - keep first 10s
    max_secs = 10.0

    # Segments are 960ms long
    frame_dur_sec = 0.96

    for emb_path, results_path in zip(saved_embeddings_paths, rnn_result_paths):
        run_rnn_experiment(emb_path, max_secs, frame_dur_sec, results_path)


if __name__ == "__main__":
    main()
