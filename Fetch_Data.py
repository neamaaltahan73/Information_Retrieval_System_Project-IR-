from Clean_Text import CleanText


class PrepareData:
    def __init__(self, collection):
        self.collection = collection

    def prepare_data(self, file_path):
        processed_texts = []
        with open(file_path, "rt", encoding="utf8") as fin:
            for i, line in enumerate(fin):
                if i < 5000:
                    cleaner = CleanText()
                    processed_line = cleaner.clean_text(line.strip())
                    processed_texts.append(processed_line)
                else:
                    break
        return processed_texts
