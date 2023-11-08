import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

MIMIC_DIR = 'data'
EMBEDDING_DIR = (
    'generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files'
)


def main() -> None:
    # Load image embedding
    tf_files = glob.glob(EMBEDDING_DIR + '/**/*.tfrecord', recursive=True)
    tf_dataset = tf.data.TFRecordDataset(tf_files)

    # Extract embeddings
    embedding_array = np.zeros(shape=(len(tf_files), 1376), dtype=np.float32)
    arr_keys = []
    dcm_keys = []

    example = tf.train.Example()
    for i, record in enumerate(tqdm(tf_dataset, total=len(tf_files))):
        example.ParseFromString(record.numpy())
        embedding_vector = np.asarray(
            example.features.feature['embedding'].float_list.value
        )
        dcm_id = (
            os.path.basename(example.features.feature['image/id'].bytes_list.value[0])
            .decode('utf8')
            .replace('.dcm', '')
        )

        embedding_array[i] = embedding_vector
        arr_keys.append(i)
        dcm_keys.append(dcm_id)

    # Construct dataframe
    df = pd.DataFrame.from_dict({'idx': arr_keys, 'dicom_id': dcm_keys})
    df['dicom_id'] = df['dicom_id'].astype('string')

    # Load MIMIC metadata file and merge into embedding df
    df_meta = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic-cxr-2.0.0-metadata.csv'))
    df_meta = df_meta.filter(['dicom_id', 'subject_id', 'study_id', 'ViewPosition'])
    df_meta['dicom_id'] = df_meta['dicom_id'].astype('string')
    df = df.merge(df_meta, on='dicom_id', how='left')

    # Add gender and age from patient meta
    df_pat = pd.read_csv(os.path.join(MIMIC_DIR, 'patients.csv'))
    df_pat = df_pat.filter(['subject_id', 'gender', 'anchor_age'])
    df_pat['anchor_age'] = df_pat['anchor_age'].astype(int)
    df = df.merge(df_pat, on='subject_id', how='left')

    # Add race from admission meta
    df_adm = pd.read_csv(os.path.join(MIMIC_DIR, 'admissions.csv'))
    df_adm = df_adm.filter(['subject_id', 'race'])
    df_adm = df_adm.drop_duplicates(subset=['subject_id'])
    df = df.merge(df_adm, on='subject_id', how='left')

    # Add official data splits
    df_split = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic-cxr-2.0.0-split.csv'))
    df_split['dicom_id'] = df_split['dicom_id'].astype('string')
    df_split = df_split.filter(['dicom_id', 'split'])
    df = df.merge(df_split, on='dicom_id', how='inner')

    # Add labels
    df_label = pd.read_csv(os.path.join(MIMIC_DIR, 'mimic-cxr-2.0.0-chexpert.csv'))
    df = df.merge(df_label, on=['subject_id', 'study_id'], how='left')

    # Construct array splits
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validate'].copy()
    test_df = df[df['split'] == 'test'].copy()

    train_emb = embedding_array[train_df['idx']]
    val_emb = embedding_array[val_df['idx']]
    test_emb = embedding_array[test_df['idx']]

    # Join df and arrays back together
    train_df['idx'] = range(len(train_df))
    val_df['idx'] = range(len(train_df), len(train_df) + len(val_df))
    test_df['idx'] = range(
        len(train_df) + len(val_df), len(train_df) + len(val_df) + len(test_df)
    )
    df = pd.concat([train_df, val_df, test_df])
    embedding_array = np.concatenate([train_emb, val_emb, test_emb])

    # Save fully joined meta data and embeddings
    df.to_csv(os.path.join(MIMIC_DIR, 'mimic_meta.csv'), index=False)
    np.save(file=os.path.join(MIMIC_DIR, 'mimic_cfm.npy'), arr=embedding_array)


if __name__ == '__main__':
    main()
