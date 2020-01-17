import { Injectable } from "@angular/core";
import * as tf from "@tensorflow/tfjs";
//-------------------------------------------------------------
// defines 'TrainingImageList' interface to store training dataset
//-------------------------------------------------------------
export interface TrainingImageList {
  ImageSrc: string; // this is to get the
  LabelX1: number; //
  LabelX2: number;
  Class: string;
}

@Injectable({
  providedIn: "root"
})
export class DataProcessingService {
  private isImagesListed: Boolean;
  private isImagesListPerformed: Boolean;

  private picture: HTMLImageElement;

  public tableRows: TrainingImageList[] = []; //instance of TrainingImageList
  //public dataSource = new MatTableDataSource<TrainingImageList>(this.tableRows);  //datasourse as

  public displayedColumns: string[] = [
    "ImageSrc",
    "Class",
    "Label_X1",
    "Label_X2"
  ];

  private csvContent: any;

  private label_x1: number[] = [];
  private label_x2: number[] = [];

  public ProgressBarValue: number;

  constructor() {}

  //-------------------------------------------------------------
  // calls parseImages() to populate imageSrc and targets as a list
  //
  //-------------------------------------------------------------
  async loadCSV(csvData: any) {
    this.csvContent = csvData;
    const data = this.parseImages(120);
    return data;
  }
  reset() {}

  //-------------------------------------------------------------
  // stores Image Src and Class info in CSV file
  // populates the MatTable rows and paginator
  // populates the targets as [1,0] uninfected, [0,1] parasitized
  //-------------------------------------------------------------
  parseImages(batchSize): TrainingImageList[] {
    if (this.isImagesListed) {
      this.isImagesListPerformed = false;
      return;
    }

    let allTextLines = this.csvContent.split(/\r|\n|\r/);

    const csvSeparator = ",";
    const csvSeparator_2 = ".";

    for (let i = 0; i < batchSize; i++) {
      // split content based on comma
      const cols: string[] = allTextLines[i].split(csvSeparator);

      this.tableRows.push({ ImageSrc: "", LabelX1: 0, LabelX2: 0, Class: "" });

      if (cols[0].split(csvSeparator_2)[1] == "png") {
        if (cols[1] == "Uninfected") {
          this.label_x1.push(Number("1"));
          this.label_x2.push(Number("0"));

          this.tableRows[i].ImageSrc = "assets/" + cols[0];
          this.tableRows[i].LabelX1 = 1;
          this.tableRows[i].LabelX2 = 0;
          this.tableRows[i].Class = "Uninfected";
        }

        if (cols[1] == "Parasitized") {
          this.label_x1.push(Number("0"));
          this.label_x2.push(Number("1"));

          this.tableRows[i].ImageSrc = "assets/" + cols[0];
          this.tableRows[i].LabelX1 = 0;
          this.tableRows[i].LabelX2 = 1;
          this.tableRows[i].Class = "Parasitized";
        }
      }
    }
    // this.table.renderRows();
    //this.dataSource.paginator = this.paginator;

    this.isImagesListed = true;
    this.isImagesListPerformed = true;
    return [...this.tableRows];
  }

  //-------------------------------------------------------------
  // this function generate input and target tensors for the training
  // input tensor is produced from 224x224x3 image in HTMLImageElement
  // target tensor shape2 is produced from the class definition
  //-------------------------------------------------------------
  generateData(trainData, batchSize) {
    const imageTensors = [];
    const targetTensors = [];

    let allTextLines = this.csvContent.split(/\r|\n|\r/);

    const csvSeparator = ",";
    const csvSeparator_2 = ".";

    for (let i = 0; i < batchSize; i++) {
      // split content based on comma
      const cols: string[] = allTextLines[i].split(csvSeparator);
      console.log(cols[0].split(csvSeparator_2)[0]);

      if (cols[0].split(csvSeparator_2)[1] == "png") {
        console.log(i);
        const imageTensor = this.capture(i);
        let targetTensor = tf.tensor1d([this.label_x1[i], this.label_x2[i]]);

        targetTensor.print();
        imageTensors.push(imageTensor);
        targetTensors.push(targetTensor);

        imageTensor.print(true);
      }
    }
    const images = tf.stack(imageTensors);
    const targets = tf.stack(targetTensors);

    return { images, targets };
  }

  //-------------------------------------------------------------
  // converts images in HTMLImageElement into the tensors
  // takes Image In in HTML as argument
  //-------------------------------------------------------------
  capture(imgId) {
    // Reads the image as a Tensor from the <image> element.
    this.picture = <HTMLImageElement>document.getElementById(imgId);
    const trainImage = tf.browser.fromPixels(this.picture);

    // Normalize the image between -1 and 1. The image comes in between 0-255,
    // so we divide by 127 and subtract 1.
    const trainim = trainImage
      .toFloat()
      .div(tf.scalar(127))
      .sub(tf.scalar(1));
    5;

    return trainim;
  }

  getImageTensorsToInference(imgIds: number[] = []) {
    let imageTensors = [];
    imgIds.forEach(imgId => {
      imageTensors.push(this.capture(imgId));
    });
    return tf.stack(imageTensors);
  }
}
