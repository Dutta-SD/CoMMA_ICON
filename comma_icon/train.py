# training file
import my_tokenizer
import dataloader
import my_models
import tensorflow.keras as keras
import config
import os
import custom_metrics

LANG_NAME = os.environ.get("LANG")

_lang_dict = config.INPUT_ALL_DIR[LANG_NAME]


if __name__ == "__main__":
    config.set_seed()

    train_dl = dataloader.DataLoader("train", _lang_dict)
    dev_dl = dataloader.DataLoader("dev", _lang_dict)

    # Preprocessor
    preproc_train = my_tokenizer.KerasPreProcessor(
        train_dl.get_text(), train_dl.get_labels()
    )
    preproc_dev = my_tokenizer.KerasPreProcessor(dev_dl.get_text(), dev_dl.get_labels())

    model = my_models.build_model_1()


    # Compile model
    model.compile(
        optimizer="adam",
        loss={
            "taskA_op": keras.losses.SparseCategoricalCrossentropy(),
            "taskB_op": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "taskC_op": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        },
        metrics={
            "taskA_op": custom_metrics.get_f1,
            "taskB_op": custom_metrics.get_f1,
            "taskC_op": custom_metrics.get_f1,
        },
    )

    # Parameters
    X_train = preproc_train.get_padded_seqs()
    y_train = preproc_train.get_targets()
    X_val = preproc_dev.get_padded_seqs()
    y_val = preproc_dev.get_targets()

    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ReduceLROnPlateau(patience=3),
    ]

    model_name = f"{_lang_dict['name']}_final.h5"

    if  not os.path.exists(config.MODEL_DIR / model_name):

        history = model.fit(
            x=X_train,
            y=[y_train["taskA_op"], y_train["taskB_op"], y_train["taskC_op"]],
            batch_size=config.BATCH_SIZE,
            epochs=config.NUM_TRAINING_EPOCHS,
            validation_data=(
                X_val,
                [y_val["taskA_op"], y_val["taskB_op"], y_val["taskC_op"]],
            ),
            verbose=2,
            callbacks = callbacks,
        )

        # Save
        model.save(config.MODEL_DIR / model_name)