import { Component } from "@angular/core";
import { AppMode } from "./dtos/app-mode.dto";

@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrls: ["./app.component.scss"]
})
export class AppComponent {
  title = "tfjs-demo";
  appMode: AppMode = AppMode.Infer;
  isTrainClicked: boolean = false;
  train() {
    this.isTrainClicked = true;
    this.appMode = AppMode.Train;
  }
  infer() {
    this.isTrainClicked = false;
    this.appMode = AppMode.Infer;
  }
}
