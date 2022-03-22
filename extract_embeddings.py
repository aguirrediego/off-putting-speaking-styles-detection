import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_hub as hub
import numpy as np
import tensorflow_datasets as tfds
import pickle
import math


def read_dataset(dataset_name="asd_dataset"):
    """Reads dataset from TFDS"""

    dataset = tfds.load(
        dataset_name,
        split=["train"],
        shuffle_files=True,
        as_supervised=False,
        with_info=False
    )

    def _process_example(feats):
        # Represent audio as floats in [-1, 1] range
        audio = tf.cast(feats['audio'], tf.float32) / float(tf.int16.max)
        return {"audio": audio, "label": feats["label"], "filename": feats["filename"]}

    dataset = dataset[0].map(_process_example)

    return dataset


def extract_embeddings(asd_ds, trill, cola, trillsson5):
    """ Extracts embeddings from audio files and stores them in dictionaries
        where the key=audio filename and value=embedding(s) extracted from audio
    """

    # Store TRILL19 embeddings in dict with key=filename, value=numpy tensor [segments, embeddings]
    t19_reps = {}

    # Store COLA embeddings in dict with key=filename, value=numpy tensor [segments, embeddings]
    cola_reps = {}

    # Store TRILLsson5 embeddings using entire audio in dict with key=filename, value=embedding
    ts5_full_reps = {}

    # Store TRILLsson5 embeddings dividing audio in 960ms segments in dict with key=filename, value=embedding
    ts5_960_reps = {}

    sample_rate = 16000
    trillsson5_emb_dim = 1024

    for i, example in enumerate(asd_ds):
        audio = example["audio"]
        filename = str(example["filename"].numpy()).split("'")[1]

        t19_reps[filename] = trill(audio, sample_rate=16000)['layer19'].numpy()
        ts5_full_reps[filename] = trillsson5(audio.numpy().reshape((1, -1)))['embedding'].numpy()
        cola_reps[filename] = cola(audio.numpy()).numpy()

        # Compute number of 960ms segments in current audio example
        segment_size = int(sample_rate * 0.96)
        num_segments = math.ceil(audio.shape[0] / segment_size)

        # Extract embeddings from each 0.96 segment
        audio_rep = np.full((num_segments, trillsson5_emb_dim), 0.0)

        for seg_id in range(num_segments):
            start_idx = seg_id * segment_size
            segment = audio[start_idx: start_idx + segment_size]

            segment_rep = trillsson5(segment.numpy().reshape((1, -1)))['embedding'].numpy().squeeze()
            audio_rep[seg_id] = segment_rep

        ts5_960_reps[filename] = audio_rep

    return t19_reps, cola_reps, ts5_full_reps, ts5_960_reps


def save_embeddings_dict(emb_dict, save_path):
    """ Saves embeddings dict at the specified path """
    with open(save_path, 'wb') as f:
        pickle.dump(emb_dict, f)


def main():
    # Load pretrained TRILL19 model. Download URL:https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3
    trill_path = './nonsemantic-speech-benchmark_trill_3'
    trill = hub.load(trill_path)

    # Load pretrained COLA model. Pretrained on CREMA-D using COLA's autfor code:
    # https://github.com/google-research/google-research/tree/master/cola
    cola_path = './cola_pretrained_cremad.h5'
    cola = tf.keras.models.load_model(cola_path)

    # Load pretrained TRILLsson5 model. Download URL:https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson5/1)
    trillsson5_path = './nonsemantic-speech-benchmark_trillsson5_1'
    trillsson5 = hub.KerasLayer(trillsson5_path)

    asd_ds = read_dataset(dataset_name="asd_dataset")

    t19_reps, cola_reps, ts5_full_reps, ts5_960_reps = extract_embeddings(asd_ds, trill, cola, trillsson5)

    save_embeddings_dict(t19_reps, "t19_reps.pkl")
    save_embeddings_dict(ts5_full_reps, "ts5_full_reps.pkl")
    save_embeddings_dict(ts5_960_reps, "ts5_960_reps.pkl")
    save_embeddings_dict(cola_reps, "ts5_960_reps.pkl")


if __name__ == "__main__":
    print("Extracting Embeddings...")
    main()
    print("Done")

