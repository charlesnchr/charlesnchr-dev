import React, { Component } from "react";
import "./App.css";
import Button from "@material-ui/core/Button";
import ButtonGroup from "@material-ui/core/ButtonGroup";

import Box from "@material-ui/core/Box";
import Grid from "@material-ui/core/Grid";
import LinearProgress from "@material-ui/core/LinearProgress";
import CheckCircleIcon from "@material-ui/icons/CheckCircle";
import RefreshIcon from "@material-ui/icons/Refresh";
import PlayArrowIcon from "@material-ui/icons/PlayArrow";
import CheckCircleOutlinedIcon from "@material-ui/icons/CheckCircleOutlined";
import ImageSearchIcon from "@material-ui/icons/ImageSearch";
import blue from "@material-ui/core/colors/blue";
import grey from "@material-ui/core/colors/grey";
import Slider from "@material-ui/core/Slider";
import ImageIcon from "@material-ui/icons/Image";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import FormControl from "@material-ui/core/FormControl";
import Switch from "@material-ui/core/Switch";
import AddCircleIcon from "@material-ui/icons/AddCircle";
import RemoveCircleIcon from "@material-ui/icons/RemoveCircle";
import FolderOpenTwoToneIcon from "@material-ui/icons/FolderOpenTwoTone";
import {
  Button as BPButton,
  // ButtonGroup as BPButtonGroup,
  Icon,
  InputGroup,
  Intent,
  Tooltip,
} from "@blueprintjs/core";
// import { Colors } from "@blueprintjs/core";
import SettingsMenu from "./SettingsMenu";
import AppUpdate from "./AppUpdate";
import AboutMenu from "./AboutMenu";
import TextField from "@material-ui/core/TextField";
import Chip from "@material-ui/core/Chip";
import CollectionsIcon from "@material-ui/icons/Collections";
import logo from "./logo.png";
import loadinggif from "./loading.gif";
import Divider from "@material-ui/core/Divider";
import Select from "@material-ui/core/Select";
import InputLabel from "@material-ui/core/InputLabel";
import MenuItem from "@material-ui/core/MenuItem";
import Backdrop from "@material-ui/core/Backdrop";
import CircularProgress from "@material-ui/core/CircularProgress";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

const theme = createMuiTheme({
  palette: {},
  typography: {
    fontFamily: "Roboto",
    body1: {
      fontFamily: "Roboto",
      fontSize: 14,
    },
    body2: {
      fontFamily: "Roboto",
      fontSize: 10,
    },
    div: {
      fontFamily: "Roboto",
    },
  },
});

const darktheme = createMuiTheme({
  palette: {
    type: "dark",
  },
  typography: {
    fontFamily: "Roboto",
    body1: {
      fontFamily: "Roboto",
      fontSize: 10,
    },
    body2: {
      fontFamily: "Roboto",
      fontSize: 10,
    },
    div: {
      fontFamily: "Roboto",
    },
    ul: {
      fontSize: 10,
    },
  },
});

const mytheme = createMuiTheme({
  palette: {
    type: "dark",
  },
  overrides: {
    MuiInputLabel: {
      root: {
        "&$focused": {
          color: "#90caf9",
          fontWeight: "bold",
        },
      },
      focused: {},
    },
    MuiFormControl: {
      root: {
        marginTop: 10,
      },
    },
    MuiSelect: {
      root: {
        fontSize: 12,
      },
    },
    MuiMenuItem: {
      root: {
        fontSize: 12,
      },
    },
  },
});

const { ipcRenderer } = window.require("electron");
const shell = window.require("electron").shell;
const remote = window.require("electron").remote;

const Menu = remote.Menu;
const app = remote.app;
const process = window.require("process");

const Store = window.require("electron-store");
const log = window.require("electron-log");
const store = new Store();

var fs = window.require("fs");
var path = window.require("path");
var crypto = window.require("crypto");
const isDev = window.require("electron-is-dev");

let sess = {};

let baseurl = "https://ML-SIM.com";
if (isDev && false) baseurl = "http://localhost:5000"; // dev of authentication etc.

sess.imgsize = 250; // default width
sess.headerHeight = 70;
sess.topOffset = 100;
sess.sidePanelWidth = 230;
sess.pageYOffset = 0; // stored scroll position (triggered when clearing results)
sess.width = window.innerWidth;
sess.height = window.innerHeight;

sess.desktop_fingerprint = "-1";
sess.serverActive = false;

// functions
sess.updateGeometry = null;
sess.updateSidePanel = null;
sess.removeResults = null;
sess.readFolders = null;
sess.removeSelection = null;
sess.isCalcFeaturesFinished = null;

sess.displayedFolders = null;
sess.dirsizes = null;

sess.filepaths = null;
sess.filepaths_hash = "";
sess.showingResult = false;
sess.resultsDepth = 0; // number of sequential searches
sess.resultsBuffer = [];
sess.thumbJobs = [];
sess.thumbQueue = [];

window.updateblocks = true;

// function GenerateThumbnails(dirpath) {
// var files = readFilesSync([dirpath],['.jpg','.JPG','.jpeg','.JPEG','.PNG','.png'])
// files.forEach(function(file,index) {
//     (async () => {
//       // Get the MD5 hash of an image
//       const hash = await hasha.fromFile(file, {algorithm: 'md5'});
//       sharp(file).resize(512,512).toFile('C:/de/noderesize/' + hash + '.jpg').then(info => {
// log.info('finished',file,hash)
//       })
//       //=> '1abcb33beeb811dca15f0ac3e47b88d9'
//     })();
// })
// }

/*************************************************************
 * Menu
 *************************************************************/
function RenderMenu(profilename) {
  let file_submenu = [];
  if (sess.settingsHandler != null) {
    file_submenu.push({
      label: "Settings",
      accelerator: "CmdOrCtrl+P",
      click: () => {
        sess.settingsHandler();
      },
    });
  }
  file_submenu.push(
    process.platform === "darwin" ? { role: "close" } : { role: "quit" }
  );

  const template = [
    // { role: 'appMenu' }
    ...(process.platform === "darwin"
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about" },
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide" },
              { role: "hideothers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit" },
            ],
          },
        ]
      : []),
    // { role: 'fileMenu' }
    {
      label: "File",
      submenu: file_submenu,
    },
    // { role: 'viewMenu' }
    {
      label: "View",
      submenu: [
        { role: "reload" },
        // { role: 'toggledevtools' },
        ...(isDev
          ? [
              { role: "toggledevtools" },
              {
                label: "Relaunch",
                accelerator: "CmdOrCtrl+Q",
                click: () => {
                  app.relaunch();
                  app.exit(0);
                },
              },
            ]
          : []),
        { type: "separator" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    // { role: 'windowMenu' }
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        ...(process.platform === "darwin"
          ? [
              { type: "separator" },
              { role: "front" },
              { type: "separator" },
              { role: "window" },
            ]
          : [{ role: "close" }]),
      ],
    },
    {
      role: "help",
      submenu: [
        {
          label: "Support",
          click: async () => {
            await shell.openExternal(baseurl + "/support");
          },
        },
        {
          label: "About",
          accelerator: "CmdOrCtrl+H",
          click: () => {
            sess.aboutHandler();
          },
        },
      ],
    },
  ];

  if (profilename) {
    template.push({
      label: profilename,
      submenu: [
        {
          label: "Manage subscription",
          click: async () => {
            await shell.openExternal(baseurl + "/profile");
          },
        },
        {
          label: "Logout",
          click: () => {
            sess.logout();
          },
        },
      ],
    });
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

function calculateGeometry(filepaths, imgsize) {
  var nimgs = filepaths.length;

  var screenheight = window.innerHeight - sess.topOffset;
  var screenwidth = window.innerWidth - sess.sidePanelWidth;
  var Nrow = parseInt(screenheight / (imgsize + 2 * 5));
  var Ncol = parseInt(screenwidth / (imgsize + 2 * 5));
  var Nblock = Nrow * Ncol;
  var blockheight = (imgsize + 2 * 5) * Nrow;
  var baseoffset = sess.topOffset;
  let gridheight;
  if (nimgs === 0) {
    gridheight = 500;
  } else {
    gridheight = parseInt(nimgs / Nblock) * blockheight;
  }
  log.info(
    "calculated geometry",
    nimgs,
    sess.topOffset,
    imgsize,
    Nrow,
    Ncol,
    Nblock,
    window.innerWidth,
    window.innerHeight
  );
  return {
    Nblock: Nblock,
    blockheight: blockheight,
    baseoffset: baseoffset,
    gridheight: gridheight,
  };
}

ipcRenderer.on("selected-file", (event, filepath) => {
  ipcRenderer.send("sendToPython", {
    cmd: "Reconstruct",
    arg: remote.getGlobal("settings").exportdir + "\n" + filepath.join("\n"),
  });
});

function reconstructImages(args) {
  if (args.filepath.length < 1) {
    ipcRenderer.send("open-singlefile-dialog");
  } else {
    ipcRenderer.send("sendToPython", {
      cmd: "Reconstruct",
      arg:
        remote.getGlobal("settings").exportdir +
        "\n" +
        args.filepath.join("\n"),
    });
  }
}

function readFilesRecursively(input_dirs, exts, sortBy, callBack) {
  // Recursive function
  var walk = function(root, dir, done) {
    var results = [];

    fs.readdir(dir, function(err, list) {
      if (err) return done(err);
      var pending = list.length;
      if (!pending) return done(null, results);
      list.forEach(function(file) {
        const name = path.parse(file).name;
        const ext = path.parse(file).ext;
        const filepath = path.resolve(dir, file);
        const stat = fs.statSync(filepath);
        if (stat.isDirectory()) {
          walk(root, filepath, function(err, res) {
            results = results.concat(res);
            if (!--pending) done(null, results);
          });
        } else {
          if (stat.isFile() && exts.indexOf(ext) >= 0)
            results.push({ filepath, name, ext, stat, root });
          if (!--pending) done(null, results);
        }
      });
    });
  };

  // Outer loop
  var pending = input_dirs.length;
  var files_dict = {};
  var nfiles = Array(input_dirs.length);
  let dirs = input_dirs.slice(); // copy by value
  let basedirs = input_dirs.map((dir) => path.basename(dir));

  dirs.forEach(function(dir) {
    walk(dir, dir, function(err, res) {
      files_dict[dir] = res;
      nfiles[dirs.indexOf(dir)] = res.length;
      if (!--pending) {
        var files = [];
        dirs.forEach(function(dir) {
          files = files.concat(files_dict[dir]);
        });
        if (files.length === 0) callBack({ filepaths: [], dirsizes: [] });

        if (sortBy && sortBy === "folderOrder") {
          log.info("sorting by folderorder");
          files.sort((a, b) => {
            let aidx = basedirs.indexOf(path.basename(a.root));
            let bidx = basedirs.indexOf(path.basename(b.root));
            if (aidx == bidx) {
              return a.filepath.localeCompare(b.filepath, undefined, {
                numeric: true,
                sensitivity: "base",
              });
            } else {
              return aidx > bidx ? 1 : -1;
            }
          });
          log.info("sorted now", files[0]);
        } else if (sortBy && sortBy === "reverseFolderOrder") {
          log.info("sorting by reverse folderorder");
          files.sort((a, b) => {
            let aidx = basedirs.indexOf(path.basename(a.root));
            let bidx = basedirs.indexOf(path.basename(b.root));
            if (aidx == bidx) {
              return a.filepath.localeCompare(b.filepath, undefined, {
                numeric: true,
                sensitivity: "base",
              });
            } else {
              return bidx > aidx ? 1 : -1;
            }
          });
          log.info("sorted now", files[0]);
        } else if (sortBy && sortBy === "dateCreated") {
          log.info("sorting by datecreated folderorder");
          files.sort((a, b) => {
            return a.dir < b.dir;
          });
        } else {
          files.sort((a, b) => {
            // natural sort alphanumeric strings
            // https://stackoverflow.com/a/38641281
            return a.filepath.localeCompare(b.filepath, undefined, {
              numeric: true,
              sensitivity: "base",
            });
          });
        }

        callBack({
          filepaths: files.map(function(file) {
            return file.filepath;
          }),
          dirsizes: nfiles,
        });
      }
    });
  });
}

function readFilesSync(input_dirs, exts, sortBy) {
  const files = [];
  const nfiles = [];
  let dirs = input_dirs.slice(); // copy by value

  if (sortBy && sortBy === "reverseFolderOrder") {
    dirs.reverse();
  }

  dirs.forEach((dir) => {
    var count = 0;
    try {
      fs.readdirSync(dir).forEach((filename) => {
        try {
          const name = path.parse(filename).name;
          const ext = path.parse(filename).ext;
          const filepath = path.resolve(dir, filename);
          const stat = fs.statSync(filepath);
          const isFile = stat.isFile();
          if (isFile && exts.indexOf(ext) >= 0)
            files.push({ filepath, name, ext, stat });
          count++;
        } catch (err) {
          log.info("Error reading", filename, "  -  Error message:", err);
        }
      });
      nfiles.push(count);
    } catch (err) {
      log.info("Could not read", dir);
    }
  });

  if (files.length === 0) return { filepaths: [], dirsizes: [] };

  if (sortBy && sortBy === "folderOrder") {
  } else if (sortBy && sortBy === "reverseFolderOrder") {
  } else if (sortBy && sortBy === "dateCreated") {
    files.sort((a, b) => {
      return a.stat.birthtime > b.stat.birthtime;
    });
  } else {
    files.sort((a, b) => {
      // natural sort alphanumeric strings
      // https://stackoverflow.com/a/38641281
      return a.name.localeCompare(b.name, undefined, {
        numeric: true,
        sensitivity: "base",
      });
    });
  }

  return {
    filepaths: files.map(function(file) {
      return file.filepath;
    }),
    dirsizes: nfiles,
  };
}

function getHashOfArray(inputArray) {
  var tmp_arr = inputArray.slice(0, inputArray.length);
  tmp_arr.sort();
  var inputArray_str = tmp_arr.join("");
  var filepaths_hash = crypto
    .createHash("md5")
    .update(inputArray_str)
    .digest("hex");
  return filepaths_hash;
}

function checkIfArrayChangedAndUpdateHash(inputArray) {
  var hash = getHashOfArray(inputArray);
  if (hash !== sess.filepaths_hash) {
    sess.filepaths_hash = hash;
    return true;
  } else {
    return false;
  }
}

/*************************************************************
 * Components
 *************************************************************/

class CalcStatus extends Component {
  constructor(props) {
    super(props);
    this.state = { status: null, progress: null };
  }

  isCalcFeaturesFinished() {
    if (this.state.status == null && this.state.progress == null) return true;
    else return false;
  }

  componentDidMount() {
    sess.isCalcFeaturesFinished = this.isCalcFeaturesFinished.bind(this);

    ipcRenderer.on("status", (event, cmd, msg) => {
      let p, s;

      if (cmd === "i") {
        // indexing
        msg = msg.split(",");
        var n1 = Number.parseInt(msg[0]);
        var n2 = Number.parseInt(msg[1]);
        p =
          Number.parseInt(
            (100.0 * Number.parseFloat(n1)) / Number.parseFloat(n2)
          ) + 1;
        s = (
          <Grid container>
            <Grid style={{ textAlign: "left" }} item xs>
              {"Reconstructing"}
            </Grid>
            <Grid item xs>
              {n1 + "/" + n2}
            </Grid>
          </Grid>
        );
        if (n1 % 10 === 0) log.info("indexing status", n1, "/", n2);
      } else if (cmd === "l") {
        // loading
        p = Number.parseInt(msg);
        s = (
          <Grid container>
            <Grid style={{ textAlign: "left" }} item xs>
              {"Loading:"}
            </Grid>
            <Grid item xs>
              {p + " %"}
            </Grid>
          </Grid>
        );
        if (p % 50 === 0) log.info("loading status", p);
      } else if (cmd === "h") {
        // hashing
        p = Number.parseInt(msg);
        s = (
          <Grid container>
            <Grid style={{ textAlign: "left" }} item xs>
              {"Scanning:"}
            </Grid>
            <Grid item xs>
              {p + " %"}
            </Grid>
          </Grid>
        );
        if (p % 50 === 0) log.info("scanning status", p);
      } else if (cmd === "c") {
        // computing search tree
        s = (
          <Grid container>
            <Grid style={{ textAlign: "left" }} item xs>
              {"Computing search tree"}
            </Grid>
          </Grid>
        );
        p = -1;
      } else if (cmd === "m") {
        // loading model
        s = (
          <Grid container>
            <Grid style={{ textAlign: "left" }} item xs>
              {"Loading model"}
            </Grid>
          </Grid>
        );
        p = -1;
      } else if (cmd === "d") {
        s = null;
        p = null;
      }

      this.setState({ status: s, progress: p });
    });
  }

  render() {
    var progressBar = this.state.progress ? (
      <LinearProgress variant={"indeterminate"} value={this.state.progress} />
    ) : (
      ""
    );
    return (
      <span>
        {progressBar}
        {this.state.status}
      </span>
    );
  }
}

class ImgsizeSlider extends Component {
  constructor(props) {
    super(props);
    this.state = {
      imgsize: props.imgsize,
      handler: props.handler,
      showSelect: true,
    };
    sess.showHideSelect = this.showHideSelect.bind(this);
  }

  imgsizeSlider(event, value) {
    this.setState({ imgsize: value });
    this.state.handler(this.state.imgsize);
  }

  handleSelect(e) {
    // log.info('Sorting by', e.target.value);
    sess.resort = true;
    sess.readFolders(sess.displayedFolders, e.target.value);
  }

  showHideSelect(showSelect) {
    this.setState({ showSelect: showSelect });
  }

  render() {
    return (
      <Grid container direction="row" justify="flex-end" alignItems="center">
        <Grid item>
          <Grid container direction="row" justify="flex-end">
            <Grid item></Grid>
            <Grid item>
              <ThemeProvider theme={darktheme}>
                <FormControl disabled={!this.state.showSelect}>
                  <Select
                    inputProps={{ className: "selectInput" }}
                    defaultValue="filename"
                    onChange={this.handleSelect.bind(this)}
                  >
                    <MenuItem style={{ fontSize: "14px" }} value="filename">
                      Filename
                    </MenuItem>
                    <MenuItem style={{ fontSize: "14px" }} value="dateCreated">
                      Date created
                    </MenuItem>
                    <MenuItem style={{ fontSize: "14px" }} value="folderOrder">
                      Top folder first
                    </MenuItem>
                    <MenuItem
                      style={{ fontSize: "14px" }}
                      value="reverseFolderOrder"
                    >
                      Bottom folder first
                    </MenuItem>
                  </Select>
                </FormControl>
              </ThemeProvider>
            </Grid>
          </Grid>
        </Grid>
        <Divider
          orientation="vertical"
          style={{
            margin: "0 20px 0 10px",
            height: 25,
            background: "rgba(50,50,50)",
          }}
        />
        <Grid className="nondraggable" item>
          <ImageIcon style={{ fontSize: 18 }} />
          &nbsp;&nbsp;&nbsp;
        </Grid>
        <Grid className="nondraggable" item>
          <Slider
            value={this.state.imgsize}
            min={200}
            max={500}
            onChange={this.imgsizeSlider.bind(this)}
            step={50}
            style={{ width: "150px" }}
            aria-labelledby="continuous-slider"
            disabled={!this.state.showSelect}
          />
        </Grid>
        <Grid className="nondraggable" item>
          &nbsp;&nbsp;&nbsp;
          <ImageIcon />
        </Grid>
      </Grid>
    );
  }
}

class SidePanel extends Component {
  constructor(props) {
    super(props);

    this.state = {
      selectFile: props.selectFile,
      selectedFilepaths: [],
      resetResultImages: props.resetResultImages,
      useCloud: props.useCloud,
      useCloud_set: props.useCloud_set,
      folderEntry: props.folderEntry,
      selectedDir: null,
    };
  }

  componentDidMount() {
    sess.updateSidePanel = this.updateSidePanel.bind(this);
  }

  selectedDir_set(key) {
    if (this.state.selectedDir === key) {
      this.setState({ selectedDir: null });
      sess.selectedDir = null;
    } else {
      this.setState({ selectedDir: key });
      sess.selectedDir = key;
    }
  }

  updateSidePanel(selectedFilepaths) {
    this.setState({ selectedFilepaths: selectedFilepaths });
  }

  addDir() {
    ipcRenderer.send("open-file-dialog");
  }

  removeDir() {
    this.setState({ selectedDir: null });
    sess.removeDir();
  }

  removeResults() {
    if (sess.removeResults) {
      // log.info("trying removeresults");
      sess.removeResults();
    }
  }

  folderEntry(key, dirpath, dirsize) {
    let label = path.basename(dirpath);
    if (label.length > 18) label = label.substr(0, 18) + "...";

    return (
      <Grid
        container
        key={key}
        justify="flex-start"
        alignItems="center"
        className={
          "folderEntry " +
          (this.state.selectedDir === key ? "selectedFolderEntry" : "")
        }
        onClick={this.selectedDir_set.bind(this, key)}
      >
        <Grid item>
          <FolderOpenTwoToneIcon
            style={{
              color: "#222222",
              fontSize: 18,
              marginRight: "10px",
              paddingTop: "2px",
            }}
          />
        </Grid>
        <Grid item style={{ width: "80%" }}>
          <Grid container justify="space-between">
            <Grid item>{label}</Grid>
            <Grid
              item
              style={{ fontStyle: "italic", color: blue[600], fontSize: 10 }}
            >
              {dirsize}
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    );
  }

  exportToFolder() {
    // log.info(this.state.selectedFilepaths);
    let selectedFilepaths = this.state.selectedFilepaths;
    ipcRenderer.send("sendToPython", {
      cmd: "exportToFolder",
      arg:
        remote.getGlobal("settings").exportdir +
        "\n" +
        selectedFilepaths.join("\n"),
    });
  }

  exportToZip() {
    // log.info(this.state.selectedFilepaths);
    let selectedFilepaths = this.state.selectedFilepaths;
    ipcRenderer.send("sendToPython", {
      cmd: "exportToZip",
      arg:
        remote.getGlobal("settings").exportdir +
        "\n" +
        selectedFilepaths.join("\n"),
    });
  }

  openInFolder() {
    // log.info(this.state.selectedFilepaths);
    let filename = this.state.selectedFilepaths[0];
    // let dir = path.dirname(filename)
    // const name = path.parse(filename).name;
    // log.info("running", 'explorer.exe /select, "' + filename + '"');
    window
      .require("child_process")
      .exec('explorer.exe /select,"' + filename + '"');
  }

  useCloud_handler(e) {
    this.setState({ useCloud: e.target.checked });
    this.state.useCloud_set(e.target.checked);
  }

  rescanFolders() {
    sess.thumbJobs = {};
    sess.readFolders(sess.displayedFolders);
  }

  render() {
    // log.info("RERENDER sidepanel");

    const resetBtn = (
      <Button
        size="small"
        variant="outlined"
        color="secondary"
        id="reloadbtn"
        style={{ width: "100%", fontSize: 11 }}
        disabled={sess.showingResult ? false : true}
        onClick={this.removeResults.bind(this)}
      >
        Clear results
      </Button>
    );
    const cloudChk = (
      <FormControlLabel
        style={{ marginLeft: "10px", width: "100%", color: "#D9D9D9" }}
        label="Cloud compute"
        control={
          <Switch
            checked={this.state.useCloud}
            onChange={this.useCloud_handler.bind(this)}
            value="useCloud"
            color="primary"
            size="small"
          />
        }
      />
    );

    let folderEntries;

    if (sess.displayedFolders.length > 0) {
      folderEntries = sess.displayedFolders.map((dirpath, idx) => {
        let dirsize =
          sess.dirsizes && sess.dirsizes.length > idx
            ? sess.dirsizes[idx]
            : "N/A";
        return this.folderEntry(idx, dirpath, dirsize);
      });
    } else {
      folderEntries = "";
    }

    let conditionalRemoveCircleIcon =
      this.state.selectedDir !== null ? (
        <RemoveCircleIcon
          style={{ fontSize: 18, cursor: "pointer" }}
          onClick={this.removeDir.bind(this)}
        />
      ) : (
        []
      );

    return (
      <Grid container style={{ backgroundColor: "#292929" }}>
        {/* <Grid container>
            <Grid item>
              {grid}
            </Grid>
          </Grid> */}
        <Grid
          item
          style={{
            width: "100%",
            visibility:
              this.state.selectedFilepaths.length === 0 ? "hidden" : "visible",
          }}
        >
          <Chip
            icon={<CollectionsIcon />}
            label={this.state.selectedFilepaths.length + " selected"}
            color="secondary"
            onClick={sess.removeSelection}
            onDelete={sess.removeSelection}
            style={{ width: "100%" }}
          />
        </Grid>

        {/* FOLDERS */}

        <Grid item className="sidepanelSection">
          <Grid container justify="space-between">
            <Grid item>Folders</Grid>
            <Grid item>
              {conditionalRemoveCircleIcon}
              <AddCircleIcon
                style={{ fontSize: 18, cursor: "pointer" }}
                onClick={this.addDir.bind(this)}
              />
            </Grid>
          </Grid>
        </Grid>

        <Grid
          container
          direction="column"
          style={{
            padding: "20px 5px 20px 20px",
            backgroundColor: "#474747",
            color: "#D9D9D9",
            border: "1px solid black",
          }}
        >
          {folderEntries}
          <Grid item>
            <Button
              size="small"
              style={{
                fontSize: 11,
                marginLeft: 10,
                marginTop: 10,
                backgroundColor: grey[700],
                color: grey[300],
              }}
              onClick={this.rescanFolders.bind(this)}
              variant="contained"
            >
              <RefreshIcon style={{ fontSize: 15 }} />
              &nbsp; Rescan folders
            </Button>
          </Grid>
        </Grid>

        {/* INDEXING */}

        <Grid item className="sidepanelSection">
          <Grid container justify="space-between">
            <Grid item style={{ paddingBottom: 5 }}>
              Processing
            </Grid>
          </Grid>
        </Grid>

        <Grid
          container
          direction="column"
          style={{
            padding: "20px 20px 20px 20px",
            backgroundColor: "#474747",
            color: "#D9D9D9",
            border: "1px solid black",
          }}
        >
          <Grid
            key={"grid"}
            direction="column"
            container
            spacing={1}
            justify="flex-start"
            style={{ padding: "0 6px 0 3px" }}
          >
            <Grid item className="calcStatus">
              <CalcStatus />
            </Grid>

            <Grid item>
              <Button
                size="large"
                style={{
                  fontSize: 11,
                  marginLeft: 10,
                  backgroundColor: grey[700],
                  color: grey[300],
                }}
                onClick={reconstructImages.bind(this, {
                  filepath: this.state.selectedFilepaths,
                })}
                variant="contained"
              >
                <PlayArrowIcon style={{ fontSize: 15 }} />
                &nbsp; Run ML-SIM
              </Button>
            </Grid>
            <Grid item>
              <ThemeProvider theme={mytheme}>
                <FormControl className="modelSelect">
                  <InputLabel>Selected model</InputLabel>
                  <Select defaultValue="mdl1">
                    <MenuItem value="mdl1">Randomised RCAN - 3x3.pth</MenuItem>
                    <MenuItem value="mdl2">Randomised RCAN - 3x5.pth</MenuItem>
                    <MenuItem value="mdl3">
                      My optimised model - 3x5.pth
                    </MenuItem>
                  </Select>
                </FormControl>
              </ThemeProvider>
            </Grid>
          </Grid>
        </Grid>

        {/* EXPORT */}

        <Grid item className="sidepanelSection">
          <Grid container justify="space-between">
            <Grid item style={{ paddingBottom: 5 }}>
              Share and export
            </Grid>
            <Grid item></Grid>
          </Grid>
        </Grid>

        <Grid
          container
          direction="column"
          style={{
            padding: "20px 20px 20px 20px",
            backgroundColor: "#474747",
            color: "#D9D9D9",
            border: "1px solid black",
          }}
        >
          <ButtonGroup
            orientation="vertical"
            variant="contained"
            aria-label="vertical outlined primary button group"
          >
            <Button
              style={{
                backgroundColor: grey[700],
                color: grey[300],
                fontSize: 12,
              }}
              size="small"
              onClick={this.exportToFolder.bind(this)}
            >
              Export to folder
            </Button>
            <Button
              style={{
                backgroundColor: grey[700],
                color: grey[300],
                fontSize: 12,
              }}
              size="small"
              onClick={this.exportToZip.bind(this)}
            >
              Export to zip
            </Button>
            <Button
              style={{
                backgroundColor: grey[700],
                color: grey[300],
                fontSize: 12,
              }}
              size="small"
              onClick={this.openInFolder.bind(this)}
            >
              Show in folder
            </Button>
          </ButtonGroup>
        </Grid>
      </Grid>
    );
  }
}

class ImgContainer extends Component {
  constructor(props) {
    super(props);

    var startidx = 0;
    var g = calculateGeometry(sess.filepaths, sess.imgsize);
    props.gridheight_set(g.gridheight);

    this.state = {
      startidx: startidx,
      gridheight_set: props.gridheight_set,
      gridheight: 0,
      Nblock: g.Nblock,
      blockheight: g.blockheight,
      baseoffset: g.baseoffset,
      resultImages: null,
      selectFile: props.selectFile,
      selectedFilepaths: [],
      storedGeometry: {},
      thumbdict: {},
    };

    // log.info("NOW SETTING UPDATE");
    sess.updateGeometry = this.updateGeometry.bind(this);
    sess.removeResults = this.removeResults.bind(this);
    sess.revertResults = this.revertResults.bind(this);
    sess.recoverResults = this.recoverResults.bind(this);
    sess.imgsizeRerender = this.imgsizeRerender.bind(this);
    sess.removeSelection = this.removeSelection.bind(this);
  }

  getThumb(filepath) {
    if (
      !sess.thumbJobs.includes(filepath) &&
      !sess.thumbQueue.includes(filepath)
    ) {
      if (sess.thumbJobs.length > 1) sess.thumbQueue.push(filepath);
      else {
        setTimeout(() => {
          ipcRenderer.send("sendToPython", {
            cmd: "GetThumb",
            arg: filepath,
          });
        }, 200);
        sess.thumbJobs.push(filepath);
        log.info("starting a new job", filepath, sess.thumbJobs.length);
      }
    }
  }

  checkForServerActive() {
    if (!sess.serverActive) {
      ipcRenderer.send("isServerActive");
      setTimeout(
        function() {
          this.checkForServerActive();
        }.bind(this),
        500
      );
    }
  }

  componentDidMount() {
    /******************
     * EVENT HANDLERS *
     ******************/
    ipcRenderer.on("serverActive", (event) => {
      log.info("serverActive now active")
      sess.serverActive = true;
      sess.readFolders(sess.displayedFolders);
    });

    this.checkForServerActive();

    ipcRenderer.on("thumb", (event, filepath, thumbpath, dim) => {
      // remove completed job
      let idx = sess.thumbJobs.indexOf(filepath);
      if (idx > -1) sess.thumbJobs.splice(idx, 1);

      // start new job from queue
      if (sess.thumbQueue.length > 0) {
        log.info(
          "starting a new job from queue",
          filepath,
          sess.thumbJobs.length
        );
        let job_filepath = sess.thumbQueue.splice(0, 1)[0];
        ipcRenderer.send("sendToPython", {
          cmd: "GetThumb",
          arg: job_filepath,
        });
        sess.thumbJobs.push(job_filepath);
      }

      // render if job was a success
      if (thumbpath !== "0") {
        let thumbdict = this.state.thumbdict;
        thumbdict[filepath] = { src: thumbpath, dim: dim };
        this.setState({ thumbdict: thumbdict });
      }
    });

    ipcRenderer.on("ReceivedSimilarImages", (event, result) => {
      if (result.length === 0) {
        log.info("empty result");
        ipcRenderer.send("emptyResultBox");
        return;
      }

      let filepaths = result.split("\n");

      sess.showingResult = true;
      if (sess.resultsDepth === 0) sess.resultsBuffer = [];
      sess.resultsDepth++;
      if (sess.resultsBuffer.length < sess.resultsDepth - 1)
        sess.resultsBuffer.push(filepaths);
      else {
        if (sess.resultsBuffer.length > sess.resultsDepth)
          sess.resultsBuffer = sess.resultsBuffer.slice(0, sess.resultsDepth);
        sess.resultsBuffer[sess.resultsDepth - 1] = filepaths;
      }

      // log.info("UPDATED resultsbuffer", sess.resultsBuffer);
      this.updateGeometry(filepaths);
      this.setState({ selectedFilepaths: [], resultImages: filepaths });
    });

    window.addEventListener("scroll", () => {
      if (!window.updateblocks) return;
      var blocksAboveLimit = parseInt(
        (window.pageYOffset - (window.innerHeight - sess.topOffset) / 3) /
          this.state.blockheight
      );
      if (blocksAboveLimit < 0) blocksAboveLimit = 0;
      var baseoffset =
        blocksAboveLimit * this.state.blockheight + sess.topOffset;
      var startidx = blocksAboveLimit * this.state.Nblock;
      this.setState({ baseoffset: baseoffset, startidx: startidx });
    });

    window.addEventListener("resize", () => {
      let g = calculateGeometry(sess.filepaths, sess.imgsize);
      var blocksAboveLimit = parseInt(
        (window.pageYOffset - (window.innerHeight - sess.topOffset) / 3) /
          g.blockheight
      );
      if (blocksAboveLimit < 0) blocksAboveLimit = 0;
      var baseoffset = blocksAboveLimit * g.blockheight + sess.topOffset;
      var startidx = blocksAboveLimit * g.Nblock;

      this.state.gridheight_set(g.gridheight);
      this.setState({
        baseoffset: baseoffset,
        startidx: startidx,
        Nblock: g.Nblock,
        blockheight: g.blockheight,
        gridheight: g.gridheight,
      });
    });
  }

  imgsizeRerender() {
    let g = calculateGeometry(sess.filepaths, sess.imgsize);
    var blocksAboveLimit = parseInt(
      (window.pageYOffset - (window.innerHeight - sess.topOffset) / 3) /
        g.blockheight
    );
    if (blocksAboveLimit < 0) blocksAboveLimit = 0;
    var baseoffset = blocksAboveLimit * g.blockheight + sess.topOffset;
    var startidx = blocksAboveLimit * g.Nblock;

    this.state.gridheight_set(g.gridheight);
    this.setState({
      baseoffset: baseoffset,
      startidx: startidx,
      Nblock: g.Nblock,
      blockheight: g.blockheight,
      gridheight: g.gridheight,
    });
  }

  updateGeometry(filepaths) {
    let g,
      storedGeometry = this.state.storedGeometry;

    // if displaying results for the first time, save display geometry
    if (sess.showingResult) {
      if (Object.keys(this.state.storedGeometry).length === 0) {
        storedGeometry["Nblock"] = this.state.Nblock;
        storedGeometry["blockheight"] = this.state.blockheight;
        storedGeometry["gridheight"] = this.state.gridheight;
        storedGeometry["startidx"] = this.state.startidx;
        storedGeometry["baseoffset"] = this.state.baseoffset;
        storedGeometry["pageYOffset"] = window.pageYOffset;
        storedGeometry["filepaths"] = sess.filepaths;
      }
      sess.showHideSelect(false);
      sess.updateSidePanel([]);
      sess.hideBackdrop();
    }

    // new geometry
    if (filepaths) {
      g = calculateGeometry(filepaths, sess.imgsize);
    } else {
      g = calculateGeometry(sess.filepaths, sess.imgsize);
    }

    sess.filepaths = filepaths;

    // log.info("updating geometry", sess.imgsize, g);
    this.state.gridheight_set(g.gridheight);
    this.setState({
      Nblock: g.Nblock,
      blockheight: g.blockheight,
      gridheight: g.gridheight,
      storedGeometry: storedGeometry,
    });
    return g;
  }

  removeResults() {
    // sess.readFolders(sess.displayedFolders);
    // this.resetGeometry()
    sess.showingResult = false;
    sess.showHideSelect(true);
    sess.resultsDepth = 0;
    let storedGeometry = this.state.storedGeometry;
    this.setState({
      Nblock: storedGeometry["Nblock"],
      blockheight: storedGeometry["blockheight"],
      gridheight: storedGeometry["gridheight"],
      startidx: storedGeometry["startidx"],
      baseoffset: storedGeometry["baseoffset"],
      storedGeometry: {},
      selectedFilepaths: [],
      resultImages: null,
    });

    sess.pageYOffset = storedGeometry["pageYOffset"];
    this.state.gridheight_set(storedGeometry["gridheight"]);
    // window.scrollTo(0,storedGeometry['pageYOffset'])
    sess.filepaths = storedGeometry["filepaths"];
    sess.updateSidePanel([]);

    // log.info("now changing imgcontainer state");
  }

  revertResults() {
    log.info(
      "now here TRYING to revert with",
      sess.resultsDepth,
      sess.resultsBuffer.length
    );
    if (sess.resultsDepth < 2) {
      this.removeResults();
    } else {
      sess.resultsDepth--;
      var filepaths = sess.resultsBuffer[sess.resultsDepth - 1];
      this.updateGeometry(filepaths);
      this.setState({ selectedFilepaths: [], resultImages: filepaths });
    }
  }

  recoverResults() {
    if (sess.resultsBuffer.length > sess.resultsDepth) {
      log.info(
        "now here TRYING to recover with",
        sess.resultsDepth,
        sess.resultsBuffer.length
      );
      sess.showingResult = true;
      var filepaths = sess.resultsBuffer[sess.resultsDepth];
      sess.resultsDepth++;
      this.updateGeometry(filepaths);
      this.setState({ selectedFilepaths: [], resultImages: filepaths });
    }
  }

  removeSelection() {
    sess.updateSidePanel([]);
    this.setState({ selectedFilepaths: [] });
  }

  selectImage(filepath, component) {
    let filepaths = this.state.selectedFilepaths;

    let idx = filepaths.indexOf(filepath);
    if (idx > -1) {
      filepaths.splice(idx, 1);
    } else {
      filepaths.push(filepath);
    }

    this.setState({ selectedFilepaths: filepaths });
    sess.updateSidePanel(filepaths);
  }

  imgGrid(filepaths, startidx, Nblock, offset) {
    let filepaths_displayed = filepaths.slice(startidx, startidx + Nblock);
    let selectedFilepaths = this.state.selectedFilepaths;
    const imgdivs = filepaths_displayed.map((filepath, idx) => {
      const style =
        selectedFilepaths.indexOf(filepath) > -1
          ? {
              border: "12px solid #e8f0fe",
              boxSizing: "border-box",
              transition: "border-width 0.13s linear",
            }
          : {
              border: "1px solid black",
              boxSizing: "border-box",
              transition: "border-width 0.3s linear, border-color 1s linear",
            };
      const check =
        selectedFilepaths.indexOf(filepath) > -1
          ? [
              <CheckCircleOutlinedIcon
                key={filepath + "-check"}
                style={{
                  color: "white",
                  position: "absolute",
                  top: "3px",
                  left: "3px",
                }}
              />,
              <CheckCircleIcon
                key={filepath + "-checkbg"}
                style={{
                  color: "#1A73E8",
                  position: "absolute",
                  top: "3px",
                  left: "3px",
                }}
              />,
            ]
          : [];

      let dim, img;

      if (path.extname(filepath) == ".png") {
        // showing results
        dim = "2D image";
        img = (
          <img
            alt="Collection"
            style={style}
            key={filepath + "-img"}
            onClick={this.selectImage.bind(this, filepath)}
            width={sess.imgsize}
            height={sess.imgsize}
            src={filepath}
          />
        );
      } else {
        let thumb = this.state.thumbdict[filepath];

        if (!thumb) {
          this.getThumb(filepath);
          dim = "N/A";
          img = (
            <div
              style={{
                ...style,
                ...{
                  width: sess.imgsize,
                  height: sess.imgsize,
                  backgroundColor: "rgb(1,1,1,0.2)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: 0,
                },
              }}
              onClick={this.selectImage.bind(this, filepath)}
            >
              <img src={loadinggif} width={50} height={50} />
            </div>
          );
        } else {
          dim = thumb.dim;
          img = (
            <img
              alt="Collection"
              style={style}
              key={filepath + "-img"}
              onClick={this.selectImage.bind(this, filepath)}
              width={sess.imgsize}
              height={sess.imgsize}
              src={thumb.src}
            />
          );
        }
      }

      const caption = (
        <span
          onClick={this.selectImage.bind(this, filepath)}
          style={{
            position: "absolute",
            left: 0,
            bottom: 0,
            width: sess.imgsize,
            overflow: "hidden",
            whiteSpace: "nowrap",
            backgroundColor: "rgb(30,30,30,0.8)",
            color: "rgb(200,200,200)",
            paddingLeft: 5,
            paddingRight: 5,
            paddingTop: 2,
            paddingBottom: 2,
          }}
        >
          <span style={{ marginRight: 10, fontStyle: "italic" }}>
            {path.basename(filepath)}
          </span>
          <br />
          Stack info: <span style={{ fontWeight: "bold" }}>{dim}</span>
        </span>
      );

      return (
        <div key={filepath + "-div"} style={{ position: "relative" }}>
          {img}
          {check}
          {caption}
        </div>
      );
    });
    return (
      <Box
        style={{ position: "absolute", top: offset }}
        key={"imgGrid-" + startidx}
        id="imgcontainer"
      >
        {imgdivs}
      </Box>
    );
  }

  downloadImages() {
    shell.openExternal("https://ML-SIM.com/images");
  }

  render() {
    let filepaths,
      imgs = <div></div>;

    if (this.state.resultImages) {
      // log.info("RERENDER results");
      filepaths = this.state.resultImages;
      imgs = this.imgGrid(filepaths, 0, 20, sess.topOffset);
    } else if (sess.displayedFolders.length > 0) {
      // log.info("SHOWING IMAGES ", sess.filepaths.length);
      filepaths = sess.filepaths;
      var startidx = this.state.startidx;
      var Nblock = this.state.Nblock;
      var blockheight = this.state.blockheight;
      var baseoffset = this.state.baseoffset;
      // log.info(
      //   "RERENDER Imgcontainer: ",
      //   startidx,
      //   Nblock,
      //   blockheight,
      //   baseoffset,
      //   sess.showingResult,
      //   sess.resultImages
      // );
      sess.thumbQueue = [];

      imgs = [
        this.imgGrid(filepaths, startidx, Nblock, baseoffset),
        this.imgGrid(
          filepaths,
          startidx + Nblock,
          Nblock,
          baseoffset + blockheight
        ),
        this.imgGrid(
          filepaths,
          startidx + 2 * Nblock,
          Nblock,
          baseoffset + 2 * blockheight
        ),
      ];
    } else {
      return (
        <div
          style={{
            marginLeft: 50,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 25,
            width: 400,
            fontFamily: "Roboto",
          }}
        >
          {" "}
          <p>Choose a folder to begin.</p>
          <p style={{ fontSize: 15 }}>
            {" "}
            You can also{" "}
            <span
              style={{ cursor: "pointer", color: blue[800] }}
              onClick={this.downloadImages.bind(this)}
            >
              try our official library
            </span>{" "}
            of test images to see how everything works.{" "}
          </p>{" "}
        </div>
      );
    }
    return <div justify="flex-start">{imgs}</div>;
  }
}

class App extends Component {
  constructor(props) {
    super(props);

    let displayedFolders = store.get("displayedFolders");
    if (!displayedFolders) displayedFolders = [];
    this.state = {
      imgsize: sess.imgsize,
      resultImages: null,
      displayedFolders: displayedFolders,
      selectedDir: null,
      useCloud: store.get("useCloud"),
      gridheight: 500,
      width: null,
      height: null,
      render_status: 0,
      appWidth: window.innerWidth,
      settingsOpen: false,
      aboutOpen: false,
      desktop_fingerprint: sess.desktop_fingerprint,
      sortBy: null,
      showBackdrop: false,
      engineStatus: "unstarted",
    };

    sess.displayedFolders = displayedFolders;
    sess.dirsize = store.get("dirsizes");
    sess.selectedDir = null;
    sess.filepaths = [];
    sess.dirsizes = [];
    sess.readFolders = this.readFolders.bind(this);
    sess.showBackdrop = this.showBackdrop.bind(this);
    sess.hideBackdrop = this.hideBackdrop.bind(this);
    if (!isDev) AppUpdate(); // update service
  }

  showBackdrop() {
    this.setState({ showBackdrop: true });
  }

  hideBackdrop() {
    this.setState({ showBackdrop: false });
  }

  selectFile() {
    ipcRenderer.send("open-singlefile-dialog");
  }

  imgsizeHandler(value) {
    if (value !== this.state.imgsize) {
      sess.imgsize = value;
      this.setState({ imgsize: value });
      if (sess.imgsizeRerender) sess.imgsizeRerender();
    }
  }

  loggedIn_set(loggedIn, firstname, lastname, emailaddr, token) {
    this.setState({
      loggedIn: loggedIn,
      firstname: firstname,
      lastname: lastname,
      emailaddr: emailaddr,
    });
  }

  setWidth() {
    if (this.state.appWidth !== window.innerWidth) {
      this.setState({ appWidth: window.innerWidth });
    }
  }

  logout() {
    this.setState({ loggedIn: false });
    store.set("loggedIn", false);
    store.set("loggedIn_firstname", null);
    store.set("loggedIn_lastname", null);
    store.set("loggedIn_emailaddr", null);
    RenderMenu();
  }

  componentDidMount() {
    ipcRenderer.on("EngineStatus", (event, res) => {
      if (res === "a") {
        if (this.state.engineStatus === "unstarted") {
          this.setState({ engineStatus: "active" });
        } else {
          // if installation has just finished
          this.setState({ engineStatus: "recently-active" });
          setTimeout(() => {
            this.setState({ engineStatus: "active" });
          }, 5000);
        }
        log.info("Engine connection from renderer");
        if (sess.filepaths != null && sess.filepaths_hash !== "") {
          log.info("Filepaths ready - can send to engine");
          // ipcRenderer.send("sendToPython", {
          //   cmd: "calcFeatures",
          //   filepaths_hash: sess.filepaths_hash,
          //   arg: sess.filepaths,
          //   postIndexOp: "none",
          // });
        } else {
          // readfolders will initiate loading
          log.info("Filepaths not ready yet");
        }
      } else {
        // engine not ready, re-query
        if (res === "e") {
          this.setState({ engineStatus: "Installing engine.." });
          log.info("Renderer: extraction ongoing");
        } else if (res[0] === "d") {
          let dl_status = res.split(",")[1];
          let downloadMsg = "Downloading required files.. " + dl_status + " %";
          this.setState({ engineStatus: downloadMsg });
          log.info("Renderer: Dowloading, status", dl_status);
        } else {
          // log.info("Renderer: Something else is stalling");
          if (this.state.engineStatus !== "unstarted") {
            this.setState({ engineStatus: "Preparing.." });
          }
        }
        setTimeout(() => {
          ipcRenderer.send("EngineStatus");
        }, 100);
      }
    });

    ipcRenderer.send("EngineStatus");

    // EVENT HANDLERS

    ipcRenderer.on("selected-directory", (event, dirpath) => {
      // Add directory
      let displayedFolders = this.state.displayedFolders;
      displayedFolders.push(dirpath);

      this.readFolders(displayedFolders);
    });

    window.addEventListener("resize", this.setWidth.bind(this));

    // BINDINGS

    sess.logout = this.logout.bind(this);
    sess.removeDir = this.removeDir.bind(this);
    sess.loggedIn_set = this.loggedIn_set.bind(this);
    window.setRenderState = function(val) {
      if (this.state.render_status !== val)
        this.setState({ render_status: val });
    }.bind(this);

    sess.settingsHandler = this.settingsOpen.bind(this);
    sess.aboutHandler = this.aboutOpen.bind(this);
    if (
      this.state.loggedIn === true &&
      this.state.firstname &&
      this.state.lastname
    ) {
      // log.info("setting loggedin", this.state.firstname, this.state.lastname);
      // ipcRenderer.send('renderMenu',this.state.firstname + ' ' + this.state.lastname, sess.settingsHandler)
      RenderMenu(this.state.firstname + " " + this.state.lastname);
    } else {
      // ipcRenderer.send('renderMenu')
      RenderMenu();
    }
  }

  componentDidUpdate() {
    if (sess.pageYOffset > 0 && !sess.showingResult) {
      // log.info("first part");
      window.scrollTo(0, sess.pageYOffset);
      sess.pageYOffset = 0;
    } else if (sess.showingResult) {
      window.scrollTo(0, 0);
    }
  }

  readFolders(displayedFolders, sortBy) {
    log.info("Reading folders", displayedFolders);
    if (sortBy == null) sortBy = this.state.sortBy;
    this.setState({ displayedFolders: displayedFolders, sortBy: sortBy });
    if (true) {
      readFilesRecursively(
        displayedFolders,
        [".tif", ".TIF", ".tiff", ".TIFF"],
        sortBy,
        this.readFoldersCallback.bind(this)
      );
    } else {
      var filedata = readFilesSync(
        displayedFolders,
        [".tif", ".TIF", ".tiff", ".TIFF"],
        sortBy
      );
      this.readFoldersCallback(filedata);
    }

    sess.displayedFolders = displayedFolders;
    store.set("displayedFolders", displayedFolders);
  }

  readFoldersCallback(filedata) {
    let g;
    if (checkIfArrayChangedAndUpdateHash(filedata.filepaths)) {
      log.info("Filepaths change detected", filedata.dirsizes);

      sess.filepaths = filedata.filepaths;
      sess.dirsizes = filedata.dirsizes;
      g = sess.updateGeometry(sess.filepaths);
      this.setState({ gridheight: g.gridheight });
    } else if (sess.resort) {
      sess.filepaths = filedata.filepaths;
      sess.dirsizes = filedata.dirsizes;
      g = sess.updateGeometry(sess.filepaths);
      this.setState({ gridheight: g.gridheight });
      sess.resort = false;
    }
  }

  removeDir() {
    if (sess.selectedDir !== null) {
      let displayedFolders = this.state.displayedFolders;
      if (displayedFolders.length === 0) return;
      displayedFolders = displayedFolders.filter((dirpath, idx) => {
        if (idx === sess.selectedDir) return false;
        return true;
      });
      sess.selectedDir = null;

      this.readFolders(displayedFolders);
    }
  }

  renewSubscription() {
    shell.openExternal(baseurl + "/profile");
  }

  // state handling
  displayedFolders_set(displayedFolders) {
    this.setState({ displayedFolders: displayedFolders });
  }
  gridheight_set(gridheight) {
    // log.info("SETTING gridheight", gridheight);
    this.setState({ gridheight: gridheight });
  }

  useCloud_get() {
    return this.state.useCloud;
  }
  useCloud_set(val) {
    store.set("useCloud", val);
    const checked = val ? 1 : 0;
    ipcRenderer.send("sendToPython", { cmd: "SetUseCloud", arg: checked });
    this.setState({ useCloud: val });
  }

  settingsOpen() {
    this.setState({ settingsOpen: true });
  }
  settingsClose() {
    this.setState({ settingsOpen: false });
  }
  aboutOpen() {
    this.setState({ aboutOpen: true });
  }
  aboutClose() {
    this.setState({ aboutOpen: false });
  }

  engineStatus() {
    let estat = this.state.engineStatus;
    log.info("inside engine status", estat);
    if (estat === "active" || estat === "unstarted") {
      return "";
    } else if (estat === "recently-active") {
      return (
        <div
          style={{
            width: "100%",
            height: "auto",
            position: "fixed",
            backgroundColor: "green",
            zIndex: 1000,
            bottom: 0,
            padding: 10,
            fontWeight: "bold",
            marginLeft: sess.sidePanelWidth,
          }}
        >
          The search engine is now installed! &nbsp;
          <span style={{ fontWeight: "normal" }}>
            Indexing of your images can begin.
          </span>
        </div>
      );
    } else {
      var progress = (
        <CircularProgress
          style={{ marginLeft: 15, marginBottom: -3 }}
          size={18}
        />
      );
      return (
        <div
          style={{
            width: "100%",
            height: "auto",
            position: "fixed",
            backgroundColor: "orange",
            zIndex: 1000,
            bottom: 0,
            padding: 10,
            fontWeight: "bold",
            marginLeft: sess.sidePanelWidth,
          }}
        >
          Engine not ready:{" "}
          <span style={{ fontWeight: "normal" }}>{estat}</span>
          {estat === "Installing engine.." ? progress : ""}
        </div>
      );
    }
  }

  render() {
    const appHeader = (
      <div
        className="App-header draggable"
        style={{
          position: "fixed",
          width: this.state.appWidth,
          zIndex: 100,
          height: sess.headerHeight,
        }}
      >
        <Grid container alignItems="center" justify="space-between" spacing={2}>
          <Grid item style={{ width: sess.sidePanelWidth }}>
            <img src={logo} alt="logo" height={50} />
          </Grid>
          <Grid item style={{ marginBottom: 5 }}>
            <BPButton
              large={true}
              disabled={!sess.showingResult}
              minimal={true}
              intent={Intent.PRIMARY}
              onClick={sess.revertResults}
              style={{ marginRight: 5 }}
            >
              <Icon icon="arrow-left" iconSize={20} />
            </BPButton>
            <BPButton
              large={true}
              disabled={sess.resultsBuffer.length <= sess.resultsDepth}
              minimal={true}
              intent={Intent.PRIMARY}
              onClick={sess.recoverResults}
            >
              <Icon icon="arrow-right" iconSize={20} />
            </BPButton>
          </Grid>
          {/* <Grid item >
                            <UpdateStatus/>
                          </Grid> */}

          <Grid item xs style={{ paddingRight: 50, paddingBottom: 10 }}>
            <ImgsizeSlider
              imgsize={this.state.imgsize}
              handler={this.imgsizeHandler.bind(this)}
            />
          </Grid>
        </Grid>
      </div>
    );

    return (
      <ThemeProvider theme={theme}>
        <div className="App">
          <SettingsMenu
            open={this.state.settingsOpen}
            onClose={this.settingsClose.bind(this)}
            exportdir={remote.getGlobal("settings").exportdir}
            cachedir={remote.getGlobal("settings").cachedir}
          />
          <AboutMenu
            open={this.state.aboutOpen}
            onClose={this.aboutClose.bind(this)}
            version={app.getVersion()}
          />
          {appHeader}
          <Backdrop
            style={{ zIndex: 100, color: "#fff" }}
            open={this.state.showBackdrop}
            onClick={sess.hideBackdrop}
          >
            <CircularProgress color="inherit" />
          </Backdrop>

          <Grid container justify="flex-start">
            <Grid
              item
              style={{
                overflow: "auto",
                flex: "0 0 " + sess.sidePanelWidth,
                backgroundColor: "#292929",
                maxWidth: sess.sidePanelWidth,
                position: "fixed",
                height: window.innerHeight,
                paddingTop: 70,
              }}
            >
              <SidePanel
                selectFile={this.selectFile}
                resetResultImages={this.resetResultImages}
                useCloud={this.state.useCloud}
                useCloud_set={this.useCloud_set.bind(this)}
              />
            </Grid>
            <Grid item xs>
              <div
                style={{
                  display: "flex",
                  height: this.state.gridheight,
                  paddingLeft: sess.sidePanelWidth + 15,
                }}
              >
                <ImgContainer
                  resultImages={this.state.resultImages}
                  gridheight_set={this.gridheight_set.bind(this)}
                  selectFile={this.selectFile}
                />
                {/* {this.imgContainer()} */}
              </div>
            </Grid>
          </Grid>
          {/* <div style={{width:'230px',height:'700px',position:'fixed',left:0,top:0,backgroundColor:'black'}}></div> */}
          {this.engineStatus()}
        </div>
      </ThemeProvider>
    );
  }
}

export default App;
