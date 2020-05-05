const electron = require("electron");
const app = electron.app;
// const Tray = electron.Tray;
// const Menu = electron.Menu;
const BrowserWindow = electron.BrowserWindow;

const fs = require("fs");
const rimraf = require("rimraf");
const path = require("path");
const isDev = require("electron-is-dev");

const process = require("process");

const { autoUpdater } = require("electron-updater");
var quitAndInstallReady = false;
var updateDetected = false;
const Store = require("electron-store");
const store = new Store();

var net = require("net");
const log = require("electron-log");

const stream = require("stream");
const { promisify } = require("util");
const got = require("got");
const unzipper = require("unzipper");
const pipeline = promisify(stream.pipeline);

const appData = app.getPath("appData");

let exportdir = store.get("exportdir");
if (exportdir == null) {
  exportdir = path.join(app.getPath("documents"), "ML-SIM");
  store.set("exportdir", exportdir);
}
let cachedir = store.get("cachedir");
if (cachedir == null) {
  cachedir = path.join(appData, "ML-SIM-Library", "2");
  store.set("cachedir", cachedir);
}

let baseurl = "https://ml-sim.s3.eu-west-2.amazonaws.com/pdist";
if (process.env.pdist_server) {
  baseurl = process.env.pdist_server;
}

global.settings = {
  handler: null,
  cachedir: cachedir,
  exportdir: exportdir,
};

/*************************************************************
 * Nodejs socket client
 *************************************************************/
var serverActive = false;
var filepaths = [];
var path_send_count = 0; // elements transmitted

const sendToPython = (event, json) => {
  if (!serverActive) {
    log.warn("Engine not ready -", json["cmd"], "ignored");
    return;
  }
  // if (json["cmd"] == "calcFeatures" && path_send_count > 0) return; // finishing transmitting current batch

  var client = new net.Socket();
  client.connect(5002, "127.0.0.1", function() {
    if (json["cmd"] == "calcFeatures") {
      // start chunked sending of filepaths
      filepaths = json["arg"];
      var msg =
        store.get("loggedIn_emailaddr") +
        "\n" +
        "calcFeatures" +
        "\n" +
        json["filepaths_hash"] +
        "\n" +
        filepaths.length +
        "\n" +
        json["postIndexOp"]; // prepare to receive paths
      log.info("To socket", msg);
      client.write(msg);
    } else {
      var msg =
        store.get("loggedIn_emailaddr") +
        "\n" +
        json["cmd"] +
        "\n" +
        json["arg"];
      log.info("To socket", msg);
      client.write(msg);
    }
  });

  client.on("data", function(data) {
    data = data.toString("utf-8");
    if (data[0] == "2") {
      // received result paths
      log.info("Received results: ", data.substr(0, 20), "...");
      if (event) {
        log.info("now sending back");
        event.sender.send("ReceivedSimilarImages", data.substr(1, data.length));
      }
      client.write("0"); // send more
    } else if (data[0] == "s") {
      log.info("Received: ", data);
      // processing status
      event.sender.send("status", data[1], data.substr(2, data.length));
      client.write("0"); // send more
    } else if (data[0] == "t") {
      log.info("Received thumb: ", data);
      // processing status
      let res = data.split('\n')
      let filepath = res[1]
      let thumbpath = res[2]
      let dim = res[3]
      event.sender.send("thumb", filepath, thumbpath, dim);
      client.write("0"); // send more
    } else if (data[0] == "e") {
      log.info("Received: ", data);
      let newdir = data.substr(1, data.length);
      require("child_process").exec('explorer.exe "' + newdir + '"');
      client.write("0"); // send more
    } else if (data[0] == "z") {
      log.info("Received: ", data);
      let newfile = data.substr(1, data.length);
      require("child_process").exec('explorer.exe /select,"' + newfile + '"');
      client.write("0"); // send more
    } else if (data[0] == "p") {
      // request to send filepaths
      if (path_send_count < filepaths.length) {
        var res_chunk = filepaths.slice(path_send_count, path_send_count + 7);
        path_send_count += 7;
        client.write(res_chunk.join("\n"));
      } else {
        log.info("finished sending all file paths");
        client.write("x"); // tell server there are no more paths to send
        path_send_count = 0;
      }
    } else if (data[0] == "1") {
      // kill signal, communication completed
      client.write("1");
      client.destroy(); // kill client after server's response
    }
  });

  client.on("close", function() {
    if (json["cmd"] == "calcFeatures") {
      path_send_count = 0;
    }
    log.info("Connection closed");
  });

  client.on("error", (error) => {
    log.warn("Error in socket communication", error);
    log.info("Will perform integrity check program start");
    store.set("skip_integrity_check", null);

    const options = {
      type: "warning",
      buttons: ["Relaunch now", "Dismiss"],
      defaultId: 0,
      title: "An error occurred",
      message: "Please relaunch the program to try to recover. ",
      detail: "File a bug on the ML-SIM GitHub Issues tracker if this keeps happening.",
    };

    dialog.showMessageBox(null, options, (response) => {
      if (response == 0) {
        app.relaunch();
        app.exit(0);
      }
    });
  });
};

const waitForPython = () => {
  const client = new net.Socket();
  const tryConnection = () =>
    client.connect({ port: 5002 }, () => {
      client.write("nothing");
      client.end();
      log.info("server is now active");
      if (!store.get("skip_integrity_check"))
        store.set("skip_integrity_check", true);
      serverActive = true;
    });
  tryConnection();
  client.on("error", (error) => {
    log.info("retrying connection");
    if (!serverActive) setTimeout(tryConnection, 100);
  });
};

/*************************************************************
 * start Python socket server
 *************************************************************/

//* Parameters used in the following
var use_pysrc = isDev && !process.env.use_pdist;
let setup_pdist = false;
let setup_models = [];
let total_downloaded = 0;
let total_to_download = 0;
let downloads_in_progress = 0;
let extraction_in_progress = false;
let pythonProgramPath;
let enginedir = path.join(appData, "ML-SIM-Engine");
let enginebin = "engine.exe"; //! Windows specific
var modeldir = path.join(enginedir, "models");
let archivefullpath = path.join(enginedir, "engine.mlsim");
let skip_integrity_check = store.get("skip_integrity_check");
let pdist_json = null;

//*
//* Subprocess function
//*
let pyProc = null;
const createPyProc = (use_pysrc, script) => {
  let port = "5002";
  let useCloud = store.get("useCloud") ? "1" : "0";
  let cachedir = global.settings.cachedir;

  if (!use_pysrc) {
    log.info("using engine", script);
    pyProc = require("child_process").execFile(
      script,
      [port, useCloud, cachedir],
      { cwd: path.dirname(script) }
    );
  } else {
    log.info(
      "spawning python server",
      + "pipenv run python",
      [script, port, useCloud, cachedir],
      { cwd: path.dirname(script) }
    );
    pyProc = require("child_process").spawn(
      "pipenv",
      ["run","python",script, port, useCloud, cachedir],
      { cwd: path.dirname(script) }
    );
  }

  if (pyProc != null) {
    //log.info(pyProc)
    log.info("child process success on port " + port);
    log.info("now waiting for Python server");
    waitForPython();
  }
};

const exitEngine = () => {
  log.info("exited Python socket server");
  pyProc.kill();
  pyProc = null;
};

//*
//* Other functions
//*

const get_latest_json = async () => {
  const response = await got(baseurl + "/latest.json");
  log.info("latest.json", response.body);
  return JSON.parse(response.body);
};

var checkHash = (filepath, callback) => {
  var fd = fs.createReadStream(filepath);
  var hash = crypto.createHash("sha1");
  hash.setEncoding("hex");
  fd.on("end", function() {
    hash.end();
    let hashval = hash.read();
    callback(hashval);
  });

  fd.pipe(hash);
};

var extract = (hashval) => {
  extraction_in_progress = true;

  fs.createReadStream(archivefullpath)
    .pipe(
      unzipper.Extract({
        path: path.join(enginedir, hashval),
      })
    )
    .on("close", () => {
      log.info("extraction complete");
      extraction_in_progress = false;
      store.set("engine_hash", hashval);
      if (downloads_in_progress == 0) {
        log.info("Engine setup complete");
        store.set("pdist_version", pdist_json.version);
        createPyProc(use_pysrc, pythonProgramPath);
        return;
      }
    });
};

var showHashMismatchDialog = () => {
  const options = {
    type: "warning",
    buttons: ["Relaunch now", "Dismiss"],
    defaultId: 0,
    title: "An error occurred during installation of the engine",
    message:
      "The files that were downloaded do not match the signatures of the ones on our server. Please try again by relaunching.",
    detail: "File a bug on the ML-SIM GitHub Issues tracker if this keeps happening.",
  };

  dialog.showMessageBox(null, options, (response) => {
    if (response == 0) {
      app.relaunch();
      app.exit(0);
    }
  });
};

function print_progress() {
  if (downloads_in_progress > 0) {
    log.info("Downloading status", total_downloaded / total_to_download);
    setTimeout(() => {
      print_progress();
    }, 20000);
  } else {
    log.info("nothing to download");
  }
}

function download(url, dest, post_func) {
  log.info("download", url, dest);
  let t0 = new Date().getTime();
  let added_to_total = false;
  let last_downloaded = 0;
  downloads_in_progress++;

  (async () => {
    try {
      await pipeline(
        got.stream(url).on("downloadProgress", (progress) => {
          if (!added_to_total) {
            added_to_total = true;
            total_to_download += progress.total;
            total_downloaded += progress.transferred;
            last_downloaded = progress.transferred;
          } else {
            total_downloaded += progress.transferred - last_downloaded;
            last_downloaded = progress.transferred;
          }
        }),
        fs.createWriteStream(dest)
      );
      downloads_in_progress--;
      if (post_func) {
        log.info("calling post function");
        post_func();
      }
    } catch (error) {
      log.info("error:", error);
    }
  })();
}

var download_checkHash_extract = (url, dest) => {
  download(url, dest, () => {
    log.info("downloaded latest archive");
    checkHash(dest, (hashval) => {
      if (hashval === engine_hash) {
        log.info("hash of new archive is valid");
        extract(hashval); // will overwrite if exists
      } else {
        log.warn("hash of new archive is invalid - ask user to relaunch");
        showHashMismatchDialog();
      }
    });
  });
};

const config_push_model = (modelname) => {
  let engine_models = store.get("engine_models");
  if (engine_models == null || engine_models.length === 0) {
    engine_models = [modelname];
  } else {
    let model_exists = false;
    engine_models.forEach((engine_model) => {
      if (engine_model === modelname) model_exists = true;
    });
    if (!model_exists) engine_models.push(modelname);
    else log.info("model", modelname, "already in config");
  }
  store.set("engine_models", engine_models);
};

const setupEngine = async () => {
  //* Download, extract, and launch

  log.info("baseurl", baseurl);

  if (!fs.existsSync(enginedir)) fs.mkdirSync(enginedir);

  (async () => {
    try {
      log.info("Attempting to download json", baseurl + "/latest.json");
      if (pdist_json == null) pdist_json = await get_latest_json();
      engine_hash = pdist_json.engine.hash;
      pythonProgramPath = path.join(enginedir, engine_hash, enginebin);

      if (setup_pdist) {
        //*  clean up
        log.info("Starting async clean up task");
        (async () => {
          try {
            fs.readdirSync(enginedir).forEach((pathname) => {
              if (pathname !== "engine.mlsim" && pathname !== "models") {
                const stat = fs.statSync(path.join(enginedir, pathname));
                if (stat.isFile()) {
                  log.info("Removing file", pathname);
                  fs.unlink(path.join(enginedir, pathname), (err) => {
                    if (err) throw err;
                    console.log("Deleted", pathname);
                  });
                } else {
                  log.info("Removing folder", path.join(enginedir, pathname));
                  rimraf(path.join(enginedir, pathname), function() {
                    log.info("Deleted", pathname);
                  });
                }
              }
            });
          } catch (err) {
            log.info("Could not read", enginedir, err);
          }
        })();

        //* download pdist
        log.info("downloading pdist");
        if (fs.existsSync(archivefullpath)) {
          checkHash(archivefullpath, (hashval) => {
            if (hashval === engine_hash) {
              log.info(
                "Latest archive found present",
                hashval,
                "- starting extraction"
              );
              extract(pdist_json.engine.hash);
            } else {
              log.info(
                "Hash of present archive does not match, new vs old",
                engine_hash,
                hashval,
                "- downloading latest"
              );
              fs.unlinkSync(archivefullpath);
              download_checkHash_extract(
                baseurl + "/" + pdist_json.engine.url,
                archivefullpath
              );
            }
          });
        } else {
          log.info("archive does not exist, dowloading afresh");
          download_checkHash_extract(
            baseurl + "/" + pdist_json.engine.url,
            archivefullpath
          );
        }
      }

      //* check if models are present
      if (!fs.existsSync(modeldir)) fs.mkdirSync(modeldir);
      log.info("looping over models", pdist_json.models);

      //* model download loop
      setup_models.forEach((model) => {
        let modelpath = path.join(modeldir, model.url);
        download(baseurl + "/models/" + model.url, modelpath, () => {
          checkHash(modelpath, (hashval) => {
            if (model.hash === hashval) {
              //* downloaded model has valid hash
              log.info("Downloaded", model.url, "hash valid");
              config_push_model(model.url);

              //* should subprocess be started
              if (downloads_in_progress === 0) {
                // downloading of models has finished
                if (!extraction_in_progress) {
                  log.info("Engine setup complete");
                  store.set("pdist_version", pdist_json.version);
                  createPyProc(use_pysrc, pythonProgramPath);
                  return;
                }
              }
            } else {
              //* downloaded model has invalid hash
              log.warn("Downloaded", model.url, "invalid - relaunch?");
              showHashMismatchDialog();
            }
          });
        });
      });

      //* start progress viewing
      setTimeout(() => {
        print_progress();
      }, 1000);
    } catch (error) {
      log.info("error:", error);
    }
  })();
};

const initEngine = async () => {
  // attach listener (app ready event inside initEngine)
  ipcMain.on("EngineStatus", (event, json) => {
    if (serverActive) {
      event.sender.send("EngineStatus", "a");
    } else if (downloads_in_progress > 0) {
      let frac = parseFloat(total_downloaded) / parseFloat(total_to_download);
      event.sender.send("EngineStatus", "d," + parseInt(100 * frac));
    } else if (extraction_in_progress) {
      event.sender.send("EngineStatus", "e");
    } else {
      event.sender.send("EngineStatus", "w"); // something else, just wait
    }
  });

  //* run src or compiled
  if (use_pysrc) {
    pythonProgramPath = path.join(
      app.getAppPath(),
      "pyengine",
      "pysrc",
      "engine.py"
    );
    createPyProc(use_pysrc, pythonProgramPath);
    return;
  }

  const runOrSetupEngine = () => {
    if (setup_pdist === false && setup_models.length === 0) {
      log.info("nodownloadsneeded-startingnow");

      createPyProc(use_pysrc, pythonProgramPath);
      return;
    } else {
      setupEngine();
    }
  };

  //* downloading latest json
  pdist_json = await get_latest_json();
  let str_v = store.get("pdist_version");

  if (!skip_integrity_check || pdist_json.version !== str_v) {
    //* integrity check - will be skipped for faster launching
    log.info("Performing integrity check");
    engine_hash = pdist_json.engine.hash;
    pythonProgramPath = path.join(enginedir, engine_hash, enginebin);
    let chk_mdls = 0;
    let chk_engine = false;
    let mdls = pdist_json.models;
    setup_pdist = true; // force re-extraction (for robustness)

    //* engine
    // check existence
    if (!fs.existsSync(archivefullpath)) {
      log.info("engine archive does not exist - will download");
      chk_engine = true;
      if (chk_mdls === mdls.length) runOrSetupEngine();
    } else {
      // check engine hash
      checkHash(archivefullpath, (hashval) => {
        if (pdist_json.engine.hash !== hashval) {
          log.info(
            "new vs old archive hash mismatch - will download",
            pdist_json.engine.hash,
            hashval
          );
          log.info("deleting old archive");
          fs.unlinkSync(archivefullpath);
          store.set("engine_hash", null);
        } else {
          log.info("archive hash match, but will reinstall", hashval);
          store.set("engine_hash", null);
        }
        chk_engine = true;
        if (chk_mdls === mdls.length) runOrSetupEngine();
      });
    }

    //* model integrity
    mdls.forEach((model) => {
      let modelpath = path.join(modeldir, model.url);
      if (!fs.existsSync(modelpath)) {
        log.info("model", model, "does not exist - will download");
        setup_models.push(model);
        if (++chk_mdls === mdls.length && chk_engine) runOrSetupEngine();
      } else {
        // check model hash
        checkHash(modelpath, (hashval) => {
          if (model.hash !== hashval) {
            log.warn(model, "hash mismatch - will download");
            setup_models.push(model);
          } else {
            log.info(model.url, "hash match");
            config_push_model(model.url);
          }
          if (++chk_mdls === mdls.length && chk_engine) runOrSetupEngine();
        });
      }
    });
  } else {
    //* quick check - look at stored hash vs online
    //* should pdist be downloaded
    let engine_hash = store.get("engine_hash");
    if (!engine_hash) {
      setup_pdist = true;
      log.info("nohash,willdownload");
    } else if (engine_hash !== pdist_json.engine.hash) {
      setup_pdist = true;
      log.info("stored hash mismatch,willdownload", engine_hash);
    } else {
      log.info("Stored engine_hash", engine_hash);
      pythonProgramPath = path.join(enginedir, engine_hash, enginebin);
      if (!fs.existsSync(pythonProgramPath)) {
        log.info("engine folder does not exist");
        setup_pdist = true;
      }
    }
    
    //* should models be downloaded
    let pdist_models = pdist_json.models;
    let engine_models = store.get("engine_models");
    if (!engine_models) {
      setup_models = pdist_models;
      log.info("downloadallmodels",setup_models);
    } else {
      pdist_models.forEach((model) => {
        let modelpath = path.join(enginedir, "models", model.url);
        if (!fs.existsSync(modelpath)) {
          setup_models.push(model);
        }
      });
    }

    runOrSetupEngine();
  }
};

app.on("ready", initEngine);
app.on("will-quit", exitEngine);

/*************************************************************
 * window management
 *************************************************************/
let mainWindow;
const { ipcMain, dialog } = require("electron");
const crypto = require("crypto");

// file dialog handler
// ipcMain.on('renderMenu', (event, profilename, settingsHandler) => {
//   rendermenu(profilename,settingsHandler,event)
// })

ipcMain.on("sendToPython", (event, json) => {
  log.info(json["cmd"]);
  sendToPython(event, json);
});

ipcMain.on("open-file-dialog", (event) => {
  dialog
    .showOpenDialog({
      properties: ["openFile", "openDirectory"],
    })
    .then((data) => {
      log.info(data.filePaths);
      if (data.filePaths.length > 0) {
        event.sender.send("selected-directory", data.filePaths[0]);
      }
    });
});

ipcMain.on("open-singlefile-dialog", (event) => {
  dialog
    .showOpenDialog({
      properties: ["openFile","multiSelections"],
      filters: [{ name: "Images", extensions: ["tif", "tiff"] }],
    })
    .then((data) => {
      if (data.filePaths.length > 0) {
        event.sender.send("selected-file", data.filePaths);
      }
    });
});

let desktop_fingerprint = "-1";
crypto.randomBytes(16, function(err, buffer) {
  desktop_fingerprint = buffer.toString("hex");
});

ipcMain.on("getFingerprint", (event) => {
  event.sender.send("getFingerprint", desktop_fingerprint);
});

ipcMain.on("isServerActive", (event) => {
  if(serverActive)
    event.sender.send("serverActive");
});

//  message box
ipcMain.on("messagebox", (event) => {
  const options = {
    type: "info",
    buttons: [],
    defaultId: 0,
    title: "Info",
    message:
      "Search queries using more than one image as reference is not yet implemented. ",
    detail:
      "Put an issue on GitHub if this bothers you.",
  };

  dialog.showMessageBox(null, options, (response) => {
    log.info(response);
  });
});

//  message box
ipcMain.on("waitUntilCalcFeaturesFinishedbox", (event) => {
  const options = {
    type: "info",
    buttons: [],
    defaultId: 0,
    title: "Indexing in progress..",
    message: "You can search the library once the indexing has finished. ",
    detail:
      "In the meantime you can add or remove folders. If the indexing is taking a long time, try out the cloud compute.",
  };

  dialog.showMessageBox(null, options, (response) => {
    log.info(response);
  });
});

//  message box
ipcMain.on("emptyResultBox", (event) => {
  const options = {
    type: "info",
    buttons: [],
    defaultId: 0,
    title: "Missing results for search query",
    message: "No valid results were found for your search query.",
    detail: "If you did not expect this, file a bug on the ML-SIM Github Issues tracker.",
  };

  dialog.showMessageBox(null, options, (response) => {
    log.info(response);
  });
});

function createWindow() {
  let webPreferences = {
    devTools: false,
    nodeIntegration: true,
    webSecurity: false, // is there another way to get images to load?
  };
  if (isDev) webPreferences.devTools = true;

  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    minWidth: 700,
    minHeight: 500,
    webPreferences: webPreferences,
    titleBarStyle: "hidden",
  });

  mainWindow.loadURL(
    isDev && !process.env.RunReactCompile
      ? "http://localhost:3000"
      : `file://${path.join(__dirname, "../build/index.html")}`
  );
  mainWindow.on("closed", () => (mainWindow = null));
  mainWindow.setBackgroundColor("#444444");

  global.mainWindow = mainWindow;
  // BrowserWindow.addDevToolsExtension(path.join('C:/Users/charl/AppData/Local/Google/Chrome/User Data/Default/Extensions/fmkadmapgofadopljbjfkapdkoienihi/4.4.0_0'));
  // BrowserWindow.removeDevToolsExtension(name)
  // name given by: BrowserWindow.getDevToolsExtensions
}

app.on("ready", createWindow);

app.on("window-all-closed", () => {
  if (quitAndInstallReady) {
    autoUpdater.quitAndInstall(false, true); // isSilent, isForceRunAfter (ignored when isSilent is false)
  } else {
    if (!serverActive) {
      store.set("skip_integrity_check", false); // there may be a problem
    }

    if (process.platform !== "darwin") {
      app.quit();
    }
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

/*************************************************************
 * App update
 *************************************************************/

ipcMain.on("startUpdateService", (event) => {
  autoUpdater.on("checking-for-update", () => {});

  autoUpdater.on("update-available", (info) => {
    log.info("Update available", info);
    event.sender.send("update-available", info);
    updateDetected = true;
  });
  autoUpdater.on("update-not-available", (info) => {
    log.info("Update not available.");
  });
  autoUpdater.on("error", (err) => {
    log.info("Error in auto-updater. " + err);
  });
  autoUpdater.on("download-progress", (progressObj) => {
    event.sender.send("download-progress", progressObj);
    let log_message = "Download speed: " + progressObj.bytesPerSecond;
    log_message = log_message + " - Downloaded " + progressObj.percent + "%";
    log_message =
      log_message +
      " (" +
      progressObj.transferred +
      "/" +
      progressObj.total +
      ")";
    log.info(log_message);
  });

  autoUpdater.on("update-downloaded", (info) => {
    log.info("Update downloaded");
    event.sender.send("update-downloaded", info);
    quitAndInstallReady = true;
    store.set("skip_integrity_check", false);

    const options = {
      type: "info",
      buttons: ["Relaunch now", "Later"],
      defaultId: 0,
      title: "ML-SIM update ready to install",
      message: "Do you want to begin installation of the update?",
      detail:
        "File a bug on the ML-SIM GitHub Issues tracker if you have any trouble updating.",
    };

    dialog.showMessageBox(null, options, (response) => {
      if (response == 0) {
        autoUpdater.quitAndInstall(false, true); // isSilent, isForceRunAfter (ignored when isSilent is false)
      }
    });
  });

  function checkUpdate() {
    if (!updateDetected) {
      autoUpdater.checkForUpdates();
      setTimeout(function() {
        checkUpdate();
      }, 10000);
    }
  }

  autoUpdater.logger = log;
  autoUpdater.logger.transports.file.level = "info";
  log.info("Starting update service");
  checkUpdate();
});
