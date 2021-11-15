# Preprocessor class
import textCleaner
import config
import pandas as pd


class DataLoader:
    """
    Preprocess the data
    """

    def __init__(self, mode, lang_dict):
        """
        mode : str - 'train' or 'dev' or 'test'
        """
        self._txt_clnr = textCleaner.TextCleaner()
        self.lang_dict = lang_dict
        self.mode = mode
        self._need_labels = mode is not "test"
        config.set_seed()
        self.load_data()

    def load_data(self):
        # no index col as ID is separate and different
        data = pd.read_csv(
            self.lang_dict[self.mode],
            sep="\t",
        )
        # Save in different columns
        self._id_col = data["ID"]
        self._txt_col = self.clean_txt_col(data["Text"])

        if self._need_labels:
            self._label_col = data["Labels"]
            self.handle_label_col()

    def clean_txt_col(self, txt_col: pd.Series):
        txt_col = txt_col.astype("str").apply(self._txt_clnr.single_text_cleaner)
        return txt_col

    def handle_label_col(self):

        self._label_col = self._label_col.apply(lambda x: x.strip("()").split(","))
        temp_df = pd.DataFrame()
        temp_df[["taskA", "taskB", "taskC"]] = self._label_col.to_list()
        # make 3 columns as list
        self._taskA, self._taskB, self._taskC = (
            temp_df["taskA"],
            temp_df["taskB"],
            temp_df["taskC"],
        )
        # print(temp_df)

        # Apply Label Mapping
        self._taskA = self._taskA.map(config.TASK_A_MAP)
        self._taskB = self._taskB.map(config.TASK_B_MAP)
        self._taskC = self._taskC.map(config.TASK_C_MAP)

    def get_text(self):
        return self._txt_col

    def get_id(self):
        return self._id_col

    def get_labels(self):
        targets = {
            "taskA_op": self._taskA,
            "taskB_op": self._taskB,
            "taskC_op": self._taskC,
        }
        # print(targets)
        return targets
