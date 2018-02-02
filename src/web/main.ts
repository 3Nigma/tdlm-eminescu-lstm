import * as dl from 'deeplearn';

const v = {"0": "\t", "1": "</br>", "2": " ", "3": "!", "4": "\"", "5": "(", "6": ")", "7": "*", "8": ",", "9": "-", "10": ".", "11": "1", "12": "2", "13": "3", "14": "5", "15": "6", "16": "8", "17": "9", "18": ":", "19": ";", "20": "?", "21": "A", "22": "B", "23": "C", "24": "D", "25": "E", "26": "F", "27": "G", "28": "H", "29": "I", "30": "J", "31": "K", "32": "L", "33": "M", "34": "N", "35": "O", "36": "P", "37": "Q", "38": "R", "39": "S", "40": "T", "41": "U", "42": "V", "43": "W", "44": "X", "45": "Y", "46": "Z", "47": "[", "48": "]", "49": "a", "50": "b", "51": "c", "52": "d", "53": "e", "54": "f", "55": "g", "56": "h", "57": "i", "58": "j", "59": "k", "60": "l", "61": "m", "62": "n", "63": "o", "64": "p", "65": "q", "66": "r", "67": "s", "68": "t", "69": "u", "70": "v", "71": "w", "72": "x", "73": "y", "74": "z"};

// manifest.json lives in the same directory as the final bundle
const reader = new dl.CheckpointLoader('.');
reader.getAllVariables().then(async vars => {
  const primerData = Math.floor((Math.random() * 75));
  const math = dl.ENV.math;

  const lstmKernel1 =
      vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'] as dl.Array2D;
  const lstmBias1 =
      vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'] as dl.Array1D;

  const lstmKernel2 =
      vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'] as dl.Array2D;
  const lstmBias2 =
      vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'] as dl.Array1D;

  const softmaxLayer1 = vars['softmax/Variable'] as dl.Array2D;
  const softmaxLayer2 = vars['softmax/Variable_1'] as dl.Array1D;

  const results: number[] = [];
    const forgetBias = dl.Scalar.new(1.0);
    const lstm1 = (data: dl.Array2D, c: dl.Array2D, h: dl.Array2D) =>
        dl.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: dl.Array2D, c: dl.Array2D, h: dl.Array2D) =>
        dl.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);

    let c: dl.Array2D[] = [
      dl.zeros([1, lstmBias1.shape[0] / 4]),
      dl.zeros([1, lstmBias2.shape[0] / 4])
    ];
    let h: dl.Array2D[] = [
      dl.zeros([1, lstmBias1.shape[0] / 4]),
      dl.zeros([1, lstmBias2.shape[0] / 4])
    ];

    let input = primerData;
    setInterval(async () => {
      let strChunk = "";
    for (let i = 0; i < 512; i++) {
      const onehot = dl.oneHot(dl.Array1D.new([input]), 75);

      const output = dl.multiRNNCell([lstm1, lstm2], onehot, c, h);

      c = output[0];
      h = output[1];

      const outputH = h[1];
      const logits = outputH.matMul(softmaxLayer1).add(softmaxLayer2);
      
      const smaxVals = dl.softmax(logits);
      const result = await dl.argMax(smaxVals).val();
      strChunk += v[result];
      input = result;
    }
      document.getElementById('results').innerHTML += strChunk;
    }, 10000);
});
