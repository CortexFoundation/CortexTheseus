import QtQuick 2.0
import QtQuick.Controls 1.0;
import QtQuick.Layouts 1.0;
import QtQuick.Dialogs 1.0;
import QtQuick.Window 2.1;
import QtQuick.Controls.Styles 1.1
import Ethereum 1.0

ApplicationWindow {
    id: win
    visible: false
    title: "IceCREAM"
    minimumWidth: 1280
    minimumHeight: 700
    width: 1290
    height: 750

    property alias codeText: codeEditor.text
    property alias dataText: rawDataField.text

    onClosing: {
        //dbg.Stop()
    }

    menuBar: MenuBar {
        Menu {
            title: "Edit"
            MenuItem {
                text: "Focus code"
                shortcut: "Ctrl+1"
                onTriggered: {
                    codeEditor.focus = true
                }
            }
            MenuItem {
                text: "Focus data"
                shortcut: "Ctrl+2"
                onTriggered: {
                    rawDataField.focus = true
                }
            }

            MenuItem {
                text: "Command"
                shortcut: "Ctrl+l"
                onTriggered: {
                    dbgCommand.focus = true
                }
            }
        }

        Menu {
            title: "Debugger"
            MenuItem {
                text: "Run"
                shortcut: "Ctrl+r"
                onTriggered: debugCurrent()
            }

            MenuItem {
                text: "Stop"
                onTriggered: dbp.stop()
            }

            MenuSeparator {}

            MenuItem {
                text: "Next"
                shortcut: "Ctrl+n"
                onTriggered: dbg.next()
            }

            MenuItem {
                text: "Continue"
                shortcut: "Ctrl+g"
                onTriggered: dbg.continue()
            }
        }
    }


    SplitView {
        anchors.fill: parent
        property var asmModel: ListModel {
            id: asmModel
        }

        TableView {
            id: asmTableView
            width: 200
            headerVisible: false
            TableViewColumn{ role: "value" ; title: "" ; width: asmTableView.width - 2 }
            model: asmModel
            /*
             alternatingRowColors: false
             itemDelegate: Item {
                 Rectangle {
                     anchors.fill: parent
                     color: "#DDD"
                     Text {
                         anchors {
                             left: parent.left
                             right: parent.right
                             leftMargin: 10
                             verticalCenter: parent.verticalCenter
                         }
                         color: "#333"
                         elide: styleData.elideMode
                         text: styleData.value
                         font.pixelSize: 11
                         MouseArea {
                             acceptedButtons: Qt.LeftButton
                             anchors.fill: parent
                             onClicked: {
                                 mouse.accepted = true
                             }
                         }
                     }
                 }
             }
             */
        }

        Rectangle {
            color: "#00000000"
            anchors.left: asmTableView.right
            anchors.right: parent.right
            SplitView {
                orientation: Qt.Vertical
                anchors.fill: parent

                Rectangle {
                    color: "#00000000"
                    height: 330
                    anchors.left: parent.left
                    anchors.right: parent.right

                    TextArea {
                        id: codeEditor
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: settings.left
                        focus: true

                        /*
                         Timer {
                             id: compileTimer
                             interval: 500 ; running: true ;  repeat: true
                             onTriggered: {
                                 dbg.autoComp(codeEditor.text)
                             }
                         }
                         */
                    }

                    Column {
                        id: settings
                        spacing: 5
                        width: 300
                        height: parent.height
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom

                        Label {
                            text: "Arbitrary data"
                        }
                        TextArea {
                            id: rawDataField
                            anchors.left: parent.left
                            anchors.right: parent.right
                            height: 150
                        }

                        Label {
                            text: "Amount"
                        }
                        TextField {
                            id: txValue
                            width: 200
                            placeholderText: "Amount"
                            validator: RegExpValidator { regExp: /\d*/ }
                        }
                        Label {
                            text: "Amount of gas"
                        }
                        TextField {
                            id: txGas
                            width: 200
                            validator: RegExpValidator { regExp: /\d*/ }
                            text: "10000"
                            placeholderText: "Gas"
                        }
                        Label {
                            text: "Gas price"
                        }
                        TextField {
                            id: txGasPrice
                            width: 200
                            placeholderText: "Gas price"
                            text: "1000000000000"
                            validator: RegExpValidator { regExp: /\d*/ }
                        }
                    }
                }

                SplitView {
                    orientation: Qt.Vertical
                    id: inspectorPane
                    height: 500

                    SplitView {
                        orientation: Qt.Horizontal
                        height: 150

                        TableView {
                            id: stackTableView
                            property var stackModel: ListModel {
                                id: stackModel
                            }
                            height: parent.height
                            width: 300
                            TableViewColumn{ role: "value" ; title: "Local VM stack" ; width: stackTableView.width - 2 }
                            model: stackModel
                        }

                        TableView {
                            id: memoryTableView
                            property var memModel: ListModel {
                                id: memModel
                            }
                            height: parent.height
                            width: parent.width - stackTableView.width
                            TableViewColumn{ id:mnumColmn ; role: "num" ; title: "#" ; width: 50 }
                            TableViewColumn{ role: "value" ; title: "Memory" ; width: 650 }
                            model: memModel
                        }
                    }

                    Rectangle {
                        height: 100
                        width: parent.width
                        TableView {
                            id: storageTableView
                            property var memModel: ListModel {
                                id: storageModel
                            }
                            height: parent.height
                            width: parent.width
                            TableViewColumn{ id: key ; role: "key" ; title: "#" ; width: storageTableView.width / 2 - 1}
                            TableViewColumn{ role: "value" ; title: "Storage" ; width:  storageTableView.width / 2 - 1}
                            model: storageModel
                        }
                    }

                    Rectangle {
                        height: 200
                        width: parent.width * 0.66
                        TableView {
                            id: logTableView
                            property var logModel: ListModel {
                                id: logModel
                            }
                            height: parent.height
                            width: parent.width
                            TableViewColumn{ id: message ; role: "message" ; title: "log" ; width: logTableView.width - 2 }
                            model: logModel
                        }
                    }

                }
            }
        }
    }

    function exec() {
        dbg.execCommand(dbgCommand.text);
        dbgCommand.text = "";
    }
    statusBar: StatusBar {
        height: 30


        TextField {
            id: dbgCommand
            y: 1
            x: asmTableView.width
            width: 500
            placeholderText: "Debugger (type 'help')"
            Keys.onReturnPressed: {
                exec()
            }
        }

        RowLayout {
            anchors.left: dbgCommand.right
            anchors.leftMargin: 10
            spacing: 5
            y: parent.height / 2 - this.height / 2

            Text {
                objectName: "stackFrame"
                font.pixelSize: 10
                text: "<b>stack ptr</b>: 0"
            }

            Text {
                objectName: "stackSize"
                font.pixelSize: 10
                text: "<b>stack size</b>: 0"
            }

            Text {
                objectName: "memSize"
                font.pixelSize: 10
                text: "<b>mem size</b>: 0"
            }
        }
    }

    toolBar: ToolBar {
        height: 30
        RowLayout {
            spacing: 10

            Button {
                property var enabled: true
                id: debugStart
                onClicked: {
                    debugCurrent()
                }
                text: "Debug"
            }

            Button {
                property var enabled: true
                id: debugNextButton
                onClicked: {
                    dbg.next()
                }
                text: "Next"
            }

            Button {
                id: debugContinueButton
                onClicked: {
                    dbg.continue()
                }
                text: "Continue"
            }
        }


        ComboBox {
		visible: false
            id: snippets
            anchors.right: parent.right
            model: ListModel {
                ListElement { text: "Snippets" ; value: "" }
                ListElement { text: "Call Contract" ; value: "var[2] in = { \"arg1\", 0xdeadbeef };\nvar ret;\n\nvar success = call(0x0c542ddea93dae0c2fcb2cf175f03ad80d6be9a0, 0, 7000, in, ret)\n\nreturn ret" }
            }
            onCurrentIndexChanged: {
                if(currentIndex != 0) {
                    var code = snippets.model.get(currentIndex).value;
                    codeEditor.insert(codeEditor.cursorPosition, code)
                }
            }
        }

    }

    function debugCurrent() {
        dbg.debug(txValue.text, txGas.text, txGasPrice.text, codeEditor.text, rawDataField.text)
    }

    function setAsm(asm) {
        asmModel.append({asm: asm})
    }

    function clearAsm() {
        asmModel.clear()
    }

    function setInstruction(num) {
        asmTableView.selection.clear()
        asmTableView.selection.select(num)
        asmTableView.positionViewAtRow(num, ListView.Center)
    }

    function setMem(mem) {
        memModel.append({num: mem.num, value: mem.value})
    }
    function clearMem(){
        memModel.clear()
    }

    function setStack(stack) {
        stackModel.append({value: stack})
    }
    function addDebugMessage(message){
        debuggerLog.append({value: message})
    }

    function clearStack() {
        stackModel.clear()
    }

    function clearStorage() {
        storageModel.clear()
    }

    function setStorage(storage) {
        storageModel.append({key: storage.key, value: storage.value})
    }

    function setLog(msg) {
        // Remove first item once we've reached max log items
        if(logModel.count > 250) {
            logModel.remove(0)
        }

        if(msg.len != 0) {
            if(logTableView.flickableItem.atYEnd) {
                logModel.append({message: msg})
                logTableView.positionViewAtRow(logTableView.rowCount - 1, ListView.Contain)
            } else {
                logModel.append({message: msg})
            }
        }
    }

    function clearLog() {
        logModel.clear()
    }
}
