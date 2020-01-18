import { Component, OnInit } from "@angular/core";
import { TfjsService } from "src/app/services/tfjs.service";
import { DataProcessingService } from "src/app/services/data-processing.service";

@Component({
  selector: "app-main",
  templateUrl: "./main.component.html",
  styleUrls: ["./main.component.scss"]
})
export class MainComponent implements OnInit {
  private csvContent: any;
  tableRows: any;
  modelFinishedTraining: boolean;
  constructor(
    private tfjs: TfjsService,
    private dataProcessing: DataProcessingService
  ) {
    //  tfjs.getModifiedMobilenet2();
  }

  ngOnInit() {
    console.log("tfjs is running on: ", this.tfjs.getBackend());
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
    this.tfjs.infer([54, 55, 56, 58]);
    this.tfjs.infer([6, 7, 8, 10]);
  }
}
