import QtQuick 2.0
import QtQuick.Controls 1.0;
import QtQuick.Layouts 1.0;
import QtQuick.Dialogs 1.0;
import QtQuick.Window 2.1;
import QtQuick.Controls.Styles 1.1
import Ethereum 1.0

ApplicationWindow {
    visible: false
    title: "IceCream"
    minimumWidth: 1280
    minimumHeight: 900
    width: 1290
    height: 900

	MenuBar {
		Menu {
			title: "Debugger"
			MenuItem {
				text: "Run"
				shortcut: "Ctrl+r"
				onTriggered: debugCurrent()
			}

			MenuItem {
				text: "Next"
				shortcut: "Ctrl+n"
				onTriggered: dbg.next()
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
            TableViewColumn{ role: "value" ; title: "" ; width: 100 }
            model: asmModel
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
                    height: 500
                    anchors.left: parent.left
                    anchors.right: parent.right

                    TextArea {
                        id: codeEditor
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: settings.left
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
                        height: 250

                        TableView {
                            id: stackTableView
                            property var stackModel: ListModel {
                                id: stackModel
                            }
                            height: parent.height
                            width: 300
                            TableViewColumn{ role: "value" ; title: "Stack" ; width: 200 }
                            model: stackModel
                        }

                        TableView {
                            id: memoryTableView
                            property var memModel: ListModel {
                                id: memModel
                            }
                            height: parent.height
                            width: parent.width - stackTableView.width
                            TableViewColumn{ id:mnumColmn ; role: "num" ; title: "#" ; width: 50}
                            TableViewColumn{ role: "value" ; title: "Memory" ; width: 750}
                            model: memModel
                        }
                    }

                    SplitView {
                        height: 300
                        TableView {
                            id: storageTableView
                            property var memModel: ListModel {
                                id: storageModel
                            }
                            height: parent.height
                            width: parent.width - stackTableView.width
                            TableViewColumn{ id: key ; role: "key" ; title: "#" ; width: storageTableView.width / 2}
                            TableViewColumn{ role: "value" ; title: "value" ; width:  storageTableView.width / 2}
                            model: storageModel
                        }
                    }
                }
            }
        }
    }
    toolBar: ToolBar {
        RowLayout {
            spacing: 5

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
        asmTableView.selection.select(num-1)
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
}
