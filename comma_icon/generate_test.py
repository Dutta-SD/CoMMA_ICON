# generate test predictions
import config
import tensorflow.keras as keras
import dataloader
import my_tokenizer
import numpy as np
import pandas as pd
import os
import custom_metrics

LANG_NAME = os.environ.get("LANG")

# --------------------------------------
_lang_dict = config.INPUT_ALL_DIR[LANG_NAME]
test_dl = dataloader.DataLoader("test", _lang_dict)
test_prep = my_tokenizer.KerasPreProcessor(test_dl.get_text())
#  --------------------------------------


def get_inverse_map():
    task_a_inv_mp = {value: key for key, value in config.TASK_A_MAP.items()}
    task_b_inv_mp = {value: key for key, value in config.TASK_B_MAP.items()}
    task_c_inv_mp = {value: key for key, value in config.TASK_C_MAP.items()}
    return task_a_inv_mp, task_b_inv_mp, task_c_inv_mp


def get_test_predictions(model):
    X = test_prep.get_padded_seqs()
    a_pred, b_pred, c_pred = model(X, training=False)
    a_pred = np.argmax(a_pred.numpy(), axis=-1)
    b_pred = np.argmax(b_pred.numpy(), axis=-1)
    c_pred = np.argmax(c_pred.numpy(), axis=-1)
    return a_pred, b_pred, c_pred


def inverse_map(preds, mp):
    preds = [mp[i] for i in preds]
    return preds


if __name__ == "__main__":
    config.set_seed()
    mdl = keras.models.load_model(
        config.MODEL_DIR / f"{_lang_dict['name']}_final.h5",
        custom_objects={"get_f1" : custom_metrics.get_f1}
    )

    a_mp, b_mp, c_mp = get_inverse_map()
    a_pred, b_pred, c_pred = get_test_predictions(mdl)

    a_pred = inverse_map(a_pred, a_mp)
    b_pred = inverse_map(b_pred, b_mp)
    c_pred = inverse_map(c_pred, c_mp)

    final_preds = [f"({i},{j},{k})" for i, j, k in zip(a_pred, b_pred, c_pred)]

    df = pd.DataFrame()
    df["ID"] = test_dl.get_id()
    df["Labels"] = final_preds

    # Save the predicitons
    df.to_csv(
        config.SUBMIT_DIR / f"pred_{_lang_dict['name']}.tsv",
        sep="\t",
        index=None,
    )
