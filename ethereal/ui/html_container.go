package ethui

import (
	"errors"
	"github.com/ethereum/eth-go/ethchain"
	"github.com/ethereum/eth-go/ethpub"
	"github.com/ethereum/eth-go/ethutil"
	"github.com/go-qml/qml"
	"github.com/howeyc/fsnotify"
	"io/ioutil"
	"log"
	"net/url"
	"os"
	"path"
	"path/filepath"
)

type HtmlApplication struct {
	win     *qml.Window
	webView qml.Object
	engine  *qml.Engine
	lib     *UiLib
	path    string
	watcher *fsnotify.Watcher
}

func NewHtmlApplication(path string, lib *UiLib) *HtmlApplication {
	engine := qml.NewEngine()

	return &HtmlApplication{engine: engine, lib: lib, path: path}

}

func (app *HtmlApplication) Create() error {
	component, err := app.engine.LoadFile(app.lib.AssetPath("qml/webapp.qml"))
	if err != nil {
		return err
	}

	if filepath.Ext(app.path) == "eth" {
		return errors.New("Ethereum package not yet supported")

		// TODO
		ethutil.OpenPackage(app.path)
	}

	win := component.CreateWindow(nil)
	win.Set("url", app.path)
	webView := win.ObjectByName("webView")

	app.win = win
	app.webView = webView

	return nil
}

func (app *HtmlApplication) RootFolder() string {
	folder, err := url.Parse(app.path)
	if err != nil {
		return ""
	}
	return path.Dir(folder.RequestURI())
}
func (app *HtmlApplication) RecursiveFolders() []os.FileInfo {
	files, _ := ioutil.ReadDir(app.RootFolder())
	var folders []os.FileInfo
	for _, file := range files {
		if file.IsDir() {
			folders = append(folders, file)
		}
	}
	return folders
}

func (app *HtmlApplication) NewWatcher(quitChan chan bool) {
	var err error

	app.watcher, err = fsnotify.NewWatcher()
	if err != nil {
		return
	}
	err = app.watcher.Watch(app.RootFolder())
	if err != nil {
		log.Fatal(err)
	}
	for _, folder := range app.RecursiveFolders() {
		fullPath := app.RootFolder() + "/" + folder.Name()
		app.watcher.Watch(fullPath)
	}

	go func() {
	out:
		for {
			select {
			case <-quitChan:
				app.watcher.Close()
				break out
			case <-app.watcher.Event:
				//ethutil.Config.Log.Debugln("Got event:", ev)
				app.webView.Call("reload")
			case err := <-app.watcher.Error:
				// TODO: Do something here
				ethutil.Config.Log.Infoln("Watcher error:", err)
			}
		}
	}()

}

func (app *HtmlApplication) Engine() *qml.Engine {
	return app.engine
}

func (app *HtmlApplication) Window() *qml.Window {
	return app.win
}

func (app *HtmlApplication) NewBlock(block *ethchain.Block) {
	b := &ethpub.PBlock{Number: int(block.BlockInfo().Number), Hash: ethutil.Hex(block.Hash())}
	app.webView.Call("onNewBlockCb", b)
}

func (app *HtmlApplication) ObjectChanged(stateObject *ethchain.StateObject) {
	app.webView.Call("onObjectChangeCb", ethpub.NewPStateObject(stateObject))
}

func (app *HtmlApplication) StorageChanged(storageObject *ethchain.StorageState) {
	app.webView.Call("onStorageChangeCb", ethpub.NewPStorageState(storageObject))
}

func (app *HtmlApplication) Destroy() {
	app.engine.Destroy()
}
