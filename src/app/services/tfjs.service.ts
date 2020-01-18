import { Injectable } from "@angular/core";
import * as tf from "@tensorflow/tfjs";
import { categoricalCrossentropy } from "@tensorflow/tfjs-layers/dist/exports_metrics";
import { DataProcessingService } from "./data-processing.service";
//-------------------------------------------------------------
// defines 'TrainingMetrics' interface to store training metrics
//-------------------------------------------------------------
export interface TrainingMetrics {
  acc: number; // training accuracy value
  ce: number; // cross entropy value
  loss: number; // loss function value
}

@Injectable({
  providedIn: "root"
})
export class TfjsService {
  public traningMetrics: TrainingMetrics[] = [];
  model: any;
  constructor(private dataProcess: DataProcessingService) {}

  getBackend() {
    return tf.getBackend();
  }
  //-------------------------------------------------------------
  // modifies the pre-trained mobilenet to detect malaria infected
  // cells, freezes layers to train only the last couple of layers
  //-------------------------------------------------------------
  async getModifiedMobilenet() {
    const trainableLayers = [
      "denseModified",
      "conv_pw_13_bn",
      "conv_pw_13",
      "conv_dw_13_bn",
      "conv_dw_13",
      "conv_preds"
    ];
    const mobilenet: tf.LayersModel = await tf.loadLayersModel(
      "http://localhost:2222/static/mobilenet/mobile-net.json"
    );
    mobilenet.summary();
    console.log("Mobilenet model is loaded");
    console.log("backend: ", tf.getBackend());

    const x = mobilenet.getLayer("global_average_pooling2d_1");
    const predictions = <tf.SymbolicTensor>(
      tf.layers
        .dense({ units: 2, activation: "softmax", name: "denseModified" })
        .apply(x.output)
    );
    let mobilenetModified = tf.model({
      inputs: mobilenet.input,
      outputs: predictions,
      name: "modelModified"
    });
    console.log("Mobilenet model is modified");
    mobilenetModified = this.freezeModelLayers(
      trainableLayers,
      mobilenetModified
    );
    console.log("ModifiedMobilenet model layers are freezed");
    mobilenetModified.compile({
      loss: categoricalCrossentropy,
      optimizer: tf.train.adam(1e-3),
      metrics: ["accuracy", "crossentropy"]
    });
    //mobilenet.dispose();
    // x.dispose();

    return mobilenetModified;
  }

  //-------------------------------------------------------------
  // freezes mobilenet layers to make them untrainable
  // just keeps final layers trainable with argument trainableLayers
  //-------------------------------------------------------------
  freezeModelLayers(trainableLayers, mobilenetModified) {
    for (const layer of mobilenetModified.layers) {
      layer.trainable = false;
      for (const tobeTrained of trainableLayers) {
        if (layer.name.indexOf(tobeTrained) === 0) {
          layer.trainable = true;
          break;
        }
      }
    }
    return mobilenetModified;
  }

  async getUpdatedModel() {
    this.model = await this.loadTruncatedMobileNet();
  }

  // Loads mobilenet and returns a model that returns the internal activation
  // we'll use as input to our classifier model.
  async loadTruncatedMobileNet() {
    const mobilenet = await tf.loadLayersModel(
      "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer("global_average_pooling2d_1");
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  }

  //-------------------------------------------------------------
  // calls generateData() to prepare the training dataset
  // calls getModfiedMobilenet() to prepare the model for training
  // calls fineTuneModifiedModel() to finetune the model
  //-------------------------------------------------------------
  async train(csvContent: any): Promise<boolean> {
    const { images, targets } = this.dataProcess.generateData(csvContent, 120);
    // this.ProgressBarValue = 35;
    // this.openSnackBar("Images are loaded into the memory as tensor !", "Close");

    const mobilenetModified = await this.getModifiedMobilenet();
    // this.ProgressBarValue = 50;
    // this.openSnackBar("Modefiled Mobilenet AI Model is loaded !", "Close");

    const data = await this.fineTuneModifiedModel(
      mobilenetModified,
      images,
      targets
    );
    // this.openSnackBar("Model training is completed !", "Close");
    //this.ProgressBarValue = 100;
    const saveResult = await mobilenetModified.save("downloads://my-model");

    return true;
  }

  //-------------------------------------------------------------
  // finetunes the modified mobilenet model in 5 training batches
  // takes model, images and targets as arguments
  //-------------------------------------------------------------
  async fineTuneModifiedModel(model, images, targets) {
    function onBatchEnd(batch, logs) {
      console.log("Accuracy", logs.acc);
      console.log("CrossEntropy", logs.ce);
      console.log("All", logs);
    }
    console.log("Finetuning the model...");

    return await model.fit(images, targets, {
      epochs: 5,
      batchSize: 24,
      validationSplit: 0.2,
      callbacks: { onBatchEnd }
    });
  }
  async loadModelByURL(url) {
    return await tf.loadLayersModel(url);
  }

  async infer(imgIds: any[]): Promise<any> {
    let myModel: tf.LayersModel;

    let model = await this.loadModelByURL(
      "http://localhost:2222/static/mobilenet/my-model.json"
    );

    myModel = model;
    let print = myModel.predict(
      this.dataProcess.getImageTensorsToInference(imgIds)
    ) as any;
    let data = await print.data();
    return data ? data : "Failed to infer";
    // .catch(err => {
    //   console.log("model doesnt exists", err);
    // });
  }
}
