import "babel-polyfill";
import * as tf from "@tensorflow/tfjs";

(() => {
  let height = 320;
  let width = 0;
  let streaming = false;
  let video = document.getElementById("video");
  let canvas = document.getElementById("canvas");
  let uprightbutton = document.getElementById("uprightbutton");
  let slouchbutton = document.getElementById("slouchbutton");
  let trainbutton = document.getElementById("trainbutton");
  let predictbutton = document.getElementById("predictbutton");
  let clearbutton = document.getElementById("clearbutton");
  let switchcamerabutton = document.getElementById("switchcamerabutton");
  let model = null;

  let trainingValue = 0;

  let trainingDataInput = [];
  let trainingDataOutput = [];

  let accuracylabel = document.getElementById("accuracylabel");
  let epochlabel = document.getElementById("epochlabel");
  let traininglabel = document.getElementById("traininglabel");
  let sampleslabel = document.getElementById("sampleslabel");
  let predictionlabel = document.getElementById("predictionlabel");

  let audioUrl = require("./music.mp3");
  let audio;

  let state = {
    isTraining: false,
    epoch: null,
    accuracy: null,
    isTrained: false,
    samples: 0,
    isPredicting: false,
    prediction: "",
    audioPlaying: false,
    frontCamera: true,
    canSwitchCamera: false,
  };

  let predictIntervalId;

  async function startup() {
    refreshState();

    audio = new Audio(audioUrl);

    let devices = await navigator.mediaDevices.enumerateDevices();
    state.canSwitchCamera =
      devices.filter((device) => device.kind === "videoinput").length > 1;

    refreshState();

    canvas.width = 50;
    canvas.height = 50;

    startVideoStream();

    video.addEventListener("canplay", (ev) => {
      if (!streaming) {
        width = video.videoWidth / (video.videoHeight / height);
        video.setAttribute("width", width);
        video.setAttribute("height", height);
        canvas.setAttribute("width", width);
        canvas.setAttribute("height", height);
        streaming = true;
      }
    });

    uprightbutton.addEventListener(
      "click",
      (ev) => {
        trainingValue = 0;
        takePicture();
        ev.preventDefault();
      },
      false
    );

    slouchbutton.addEventListener(
      "click",
      (ev) => {
        trainingValue = 1;
        takePicture();
        ev.preventDefault();
      },
      false
    );

    trainbutton.addEventListener("click", (ev) => {
      trainModel();
    });

    predictbutton.addEventListener("click", (ev) => {
      if (state.isPredicting) {
        clearInterval(predictIntervalId);
        state.isPredicting = false;
        state.prediction = false;
        refreshState();
      } else {
        state.isPredicting = true;
        refreshState();
        predictIntervalId = setInterval(() => {
          predictImage();
        }, 500);
      }
    });

    clearbutton.addEventListener("click", (ev) => {
      trainingDataInput = [];
      trainingDataOutput = [];
      refreshState();
    });

    switchcamerabutton.addEventListener("click", (ev) => {
      state.frontCamera = !state.frontCamera;
      startVideoStream();
    });
  }

  async function startVideoStream() {
    var constraints = {
      audio: false,
      video: {
        facingMode: state.frontCamera ? "user" : "environment",
      },
    };

    if (video.srcObject != null) {
      video.pause();
      try {
        video.srcObject.getTracks().forEach(async (t) => await t.stop());
      } catch (e) {
        predictionlabel.innerHTML += e;
      }
      video.srcObject = null;
    }

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = (e) => {
          video.play();
        };
      })
      .catch((e) => {
        console.error(e);
        predictionlabel.innerHTML += e;
      });
  }

  function takePicture() {
    let intervalId;

    let n = 0;

    intervalId = setInterval(() => {
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, 50, 50);
      const input = tf.browser
        .fromPixels(context.getImageData(0, 0, 50, 50))
        .cast("float32");
      trainingDataInput.push(input);
      trainingDataOutput.push(trainingValue);

      refreshState();

      n += 1;
      if (n >= 100) {
        clearInterval(intervalId);
      }
    }, 60);
  }

  async function predictImage() {
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, 50, 50);
    const input = tf.browser
      .fromPixels(context.getImageData(0, 0, 50, 50))
      .cast("float32")
      .expandDims(0);

    const prediction = tf.reshape(model.predict(input, { batchSize: 1 }), [2]);
    const data = await prediction.data();
    const index = data.indexOf(Math.max(...data));
    if (index == 0) {
      state.prediction = "upright";
      if (state.audioPlaying) {
        state.audioPlaying = false;
        audio.pause();
        audio.currentTime = 0;
      }
    } else if (index == 1) {
      state.prediction = "slouched";
      if (!state.audioPlaying) {
        state.audioPlaying = true;
        audio.play();
      }
    }
    refreshState();
  }

  function createModel(batch_size) {
    model = tf.sequential();
    // First conv layer
    model.add(
      tf.layers.conv2d({
        inputShape: [50, 50, 3],
        kernelSize: 3,
        filters: 24,
        activation: "relu",
      })
    );

    // First pooling layer
    model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
      })
    );

    // Second conv layer
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 8,
        activation: "relu",
      })
    );

    // Second pooling layer
    model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
      })
    );

    // Flatten
    model.add(tf.layers.flatten());

    model.add(
      tf.layers.dense({
        units: 2,
        activation: "softmax",
      })
    );

    // Compile it
    model.compile({
      optimizer: tf.train.sgd(0.00001),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }

  async function trainModel() {
    createModel();

    const { input, output } = await (async () => {
      tf.util.shuffleCombo(trainingDataInput, trainingDataOutput);

      let input = tf.stack(trainingDataInput);
      let output = tf.oneHot(tf.tensor1d(trainingDataOutput, "int32"), 2);

      return { input, output };
    })();

    const batch_size = 10;
    const epochs = 15;
    model
      .fit(input, output, {
        // batchSize: batch_size,
        epochs: epochs,
        callbacks: {
          onTrainBegin(logs) {
            state.isTraining = true;
            refreshState();
          },
          onEpochEnd(epoch, logs) {
            state.epoch = epoch;
            state.accuracy = logs.acc;
            refreshState();
          },
        },
      })
      .then((res) => {
        state.isTraining = false;
        state.isTrained = true;
        refreshState();
      })
      .catch((e) => {
        console.log(e);
      });
  }

  function refreshState() {
    state.samples = trainingDataInput.length;
    traininglabel.innerHTML = state.isTraining ? "Yes" : "No";
    epochlabel.innerHTML = state.epoch ?? "0";
    accuracylabel.innerHTML = state.accuracy ?? "0";
    trainbutton.hidden = state.isTraining;
    predictbutton.hidden = !state.isTrained;
    predictbutton.innerHTML = state.isPredicting
      ? "Stop Predicting"
      : "Start Predicting";
    sampleslabel.innerHTML = state.samples;
    predictionlabel.innerHTML = state.prediction;
    clearbutton.hidden = state.samples == 0;
    switchcamerabutton.hidden = !state.canSwitchCamera;
  }

  startup();
})();
