import React, { Component } from "react";
const { ipcRenderer } = window.require("electron");
const log = window.require("electron-log");

class ImageView extends Component {
  constructor(props) {
    super(props);

    this.state = { filepath: "N/A" };
  }

  componentDidMount() {
    log.info("setting event handler filepathArugment");
    ipcRenderer.on("filepathArgument", (event, filepath) => {
      log.info("received filepathargument", filepath);
      this.setState({ filepath: filepath });
      document.title = filepath;
    });
  }

  render() {
    // if (this.state.filepath !== "N/A") {
    //   img = <img src={this.state.filepath} />;
    // }
    let imgdiv = (
      <div
        style={{
          display: "table",
          marginLeft: "auto",
          width: 800,
          height: 600,
          marginRight: "auto",
          backgroundImage: `url("file:///${this.state.filepath}")`,
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center',
          backgroundSize: 'cover',
        }}
      >
        {this.state.filepath}
      </div>
    );
    return imgdiv;
  }
}
export default ImageView;
