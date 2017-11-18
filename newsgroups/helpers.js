average = function (arr) {
    return arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;
}

// Transform string class labels into a one-hot binary vector
class BinaryEncoder {

    constructor() {
        this.label_dims = -1
        this.label_to_i = {}
    }

    fit (Y) {
        Y.map((label) => {
            if (!this.label_to_i[label]) {
                this.label_to_i[label] = this.label_dims + 1
                this.label_dims += 1
            }
        })
        console.log("Built binary encoder with " + this.label_dims + " labels.")
    }

    transform(Y) {
        let newY = []
        Y.map((label, i) => {
            let y = deeplearn.Array1D.zeros([this.label_dims])
            y.set(1.0, this.label_to_i[label])
            newY.push(y)
        })
        return newY
    }

}

// Transform strings of text into a binary bag-of-words after dropping both common and rare words
class BagOfWordsEmbedding {

    constructor() {
        this.word_dims = -1
        this.word_to_i = {}
    }

    fit (X) {
        let word_to_count = {}
        X.map((text) => {
            this.extractWords(text).map((word) => {
                if (!word_to_count[word])
                    word_to_count[word] = 0
                word_to_count[word] += 1.0
            })
        })

        let minCount = this.getPercentile(0.2, Object.values(word_to_count))
        let maxCount = this.getPercentile(0.8, Object.values(word_to_count))
        Object.keys(word_to_count).map((word) => {
            if (minCount < word_to_count[word] && word_to_count[word] < maxCount) {
                if (!this.word_to_i[word]) {
                    this.word_to_i[word] = this.word_dims + 1
                    this.word_dims += 1
                }
            }
        })

        console.log("Built word embedding with " + this.word_dims + " unique words.")
    }

    transform(X) {
        let newX = []
        X.map((text, i) => {
            let x = deeplearn.Array1D.zeros([this.word_dims])
            this.extractWords(text).map((word) => {
                if (this.word_to_i.hasOwnProperty(word)) {
                    x.set(1.0, this.word_to_i[word])
                }
            })
            newX.push(x)
        })
        return newX
    }

    extractWords(text) {
        text = text.toLowerCase()
        text = text.replace(/[^\w\s]/gi, ' ')
        return text.split(" ")
    }

    getPercentile(p, arr) {
        arr = arr.sort((a, b) => (a - b))
        return arr[parseInt(arr.length * p)]
    }

}
