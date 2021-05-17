class StemWords:
    suffixCollection = {
        2: [
            "कर",
            "ाओ",
            "िए",
            "ाई",
            "ाए",
            "नी",
            "ना",
            "ते",
            "ती",
            "ाँ",
            "ां",
            "ों",
            "ें",
        ],
        3: [
            "ाकर",
            "ाइए",
            "ाईं",
            "ाया",
            "ेगी",
            "ेगा",
            "ोगी",
            "ोगे",
            "ाने",
            "ाना",
            "ाते",
            "ाती",
            "ाता",
            "तीं",
            "ाओं",
            "ाएं",
            "ुओं",
            "ुएं",
            "ुआं",
        ],
        4: [
            "ाएगी",
            "ाएगा",
            "ाओगी",
            "ाओगे",
            "एंगी",
            "ेंगी",
            "एंगे",
            "ेंगे",
            "ूंगी",
            "ूंगा",
            "ातीं",
            "नाओं",
            "नाएं",
            "ताओं",
            "ताएं",
            "ियाँ",
            "ियों",
            "ियां",
        ],
        5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
    }

    @classmethod
    def generate(cls, word):
        for suffixLen, suffixes in cls.suffixCollection.items():
            if len(word) > suffixLen + 1:
                for suffix in suffixes:
                    if word.endswith(suffix):
                        return word[:-suffixLen]
        return word
