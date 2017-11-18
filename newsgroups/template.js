$.get("dataset.json", (dataset) => {
    // Load the dataset...
    bowe = new BagOfWordsEmbedding()
    bowe.fit(dataset.trainX)
    dataset.trainX = bowe.transform(dataset.trainX)
    dataset.testX = bowe.transform(dataset.testX)

    be = new BinaryEncoder()
    be.fit(dataset.trainY)
    dataset.trainY = be.transform(dataset.trainY)
    dataset.testY = be.transform(dataset.testY)

    // Use deeplearnjs's data providers
    const testSamples = dataset.testY.length
    const trainSamples = dataset.trainY.length

    const trainProvider = new deeplearn.InCPUMemoryShuffledInputProviderBuilder([dataset.trainX, dataset.trainY]).getInputProviders()
    dataset.trainX = trainProvider[0];
    dataset.trainY = trainProvider[1];

    const testProvider = new deeplearn.InCPUMemoryShuffledInputProviderBuilder([dataset.testX, dataset.testY])
    dataset.testX = testProvider.getInputProviders()[0];
    dataset.testY = testProvider.getInputProviders()[1];

    // construct the computation graph
    const g = new deeplearn.Graph();
    const inputShape = [bowe.word_dims];
    const inputTensor = g.placeholder('input', inputShape);

    const labelShape = [be.label_dims];
    const labelTensor = g.placeholder('label', labelShape);

    const outputTensor = g.layers.dense('dense_layer', inputTensor, be.label_dims, undefined, true);

    const costTensor = g.softmaxCrossEntropyCost(outputTensor, labelTensor);
    const accTensor = g.argmaxEquals(outputTensor, labelTensor)

    // set up a new session
    const math = new deeplearn.NDArrayMathGPU();
    const session = new deeplearn.Session(g, math);
    const optimizer = new deeplearn.SGDOptimizer(0.01); // Replace this with AdamOptimizer and bad things (TM) start to happen...

    // every epoch runs train and test and prints the loss/acc to console
    function run_epoch(epoch) {
        let batch_size = 32;
        math.scope(() => {
            testAcc = () => {
                const accs = [];
                const losses = [];
                for (let i = 0; i < testSamples; i++) {
                    let res = session.evalAll([accTensor, costTensor], [
                        {tensor: inputTensor, data: dataset.testX},
                        {tensor: labelTensor, data: dataset.testY}
                    ]);
                    accs.push(res[0].get())
                    losses.push(res[1].get())
                }
                return JSON.stringify({
                    "acc": average(accs),
                    "loss": average(losses)
                })
            }
            
            trainLoss = () => {
                let cost = []
                for (let batch = 0; batch < trainSamples / batch_size; batch++)
                    cost.push(session.train(costTensor, [
                        {tensor: inputTensor, data: dataset.trainX},
                        {tensor: labelTensor, data: dataset.trainY}
                    ], batch_size, optimizer, deeplearn.CostReduction.MEAN).get());
                return average(cost)
            }

            console.log("Epoch " + epoch + ", Train Loss: " + trainLoss() + ", Test Loss: " + testAcc())
        })
    }

    // each epoch takes ~10 seconds
    for (let i = 0; i < 5; i++)
        run_epoch(i)
})