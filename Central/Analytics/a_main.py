class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._read_file()
        self.operations = {
            'word_count': self.word_count,
            'search_keyword': self.search_keyword
        }