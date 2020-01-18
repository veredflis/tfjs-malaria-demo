import { Component, OnInit, Input } from "@angular/core";
import { TfjsService } from "src/app/services/tfjs.service";
import { DataProcessingService } from "src/app/services/data-processing.service";
import { AppMode } from "src/app/dtos/app-mode.dto";

@Component({
  selector: "app-main",
  templateUrl: "./main.component.html",
  styleUrls: ["./main.component.scss"]
})
export class MainComponent implements OnInit {
  @Input() appMode: AppMode;
  private csvContent: any;
  tableRows: any;
  inferImages: string[] = [];
  modelFinishedTraining: boolean;
  inferenceResults: Float32Array[] = [];
  constructor(
    private tfjs: TfjsService,
    private dataProcessing: DataProcessingService
  ) {
    //  tfjs.getModifiedMobilenet2();
  }

  ngOnInit() {
    console.log("tfjs is running on: ", this.tfjs.getBackend());
  }

  ngOnChanges() {
    if (this.appMode === AppMode.Infer) {
      if (this.inferImages.length < 1) {
        for (let i = 0; i < 71; i++) {
          this.inferImages.push(`assets/inference/${i}.png`);
        }
        console.log(this.inferImages);
      }
    }
  }
  ngOnDestroy() {
    this.inferImages = [];
  }
  //-------------------------------------------------------------
  // onFileLoad and onFileSelect functions opens file browser
  // and stores the selected CVS file content in csvContent variable
  //-------------------------------------------------------------
  // onFileLoad(fileLoadedEvent) {
  //   const textFromFileLoaded = fileLoadedEvent.target.result;
  //   this.csvContent = textFromFileLoaded;
  // }
  onFileSelect(input: HTMLInputElement) {
    const files = input.files;

    if (files && files.length) {
      const fileToRead = files[0];

      const fileReader: FileReader = new FileReader();
      fileReader.onload = (event: Event) => {
        const textFromFileLoaded = fileReader.result;
        this.loadCSV(textFromFileLoaded);
      };

      fileReader.readAsText(fileToRead, "UTF-8");

      console.log("Filename: " + files[0].name);
      console.log("Type: " + files[0].type);
      console.log("Size: " + files[0].size + " bytes");
    }
  }

  async loadCSV(csvContent) {
    this.csvContent = csvContent;
    const data = await this.dataProcessing.loadCSV(csvContent);
    console.log(data);
    this.tableRows = data;
  }

  train() {
    this.tfjs.train(this.csvContent).then(finished => {
      this.modelFinishedTraining = finished;
    });
  }

  infer() {
    this.tfjs.infer([2, 3, 4, 5, 9]);
    this.tfjs.infer([54, 55, 56, 58, 59]);
    this.tfjs.infer([6, 7, 8, 10, 11]);
  }

  isTrainMode() {
    return this.appMode === AppMode.Train;
  }
  isInferMode() {
    return this.appMode === AppMode.Infer;
  }

  inferImage(imageId) {
    this.tfjs.infer([this.getInferenceImgId(imageId)]).then(data => {
      console.log(data);
      this.inferenceResults.push(data[0], data[1]);
      alert(
        `Uninfected: ${Math.round(data[0] * 100)}% , Infected: ${Math.round(
          data[1] * 100
        )}%`
      );
    });
  }
  getInferenceImgId(index) {
    return `infer-${index}`;
  }
}
