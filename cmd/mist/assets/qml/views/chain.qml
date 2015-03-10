import QtQuick 2.0
import QtQuick.Controls 1.0;
import QtQuick.Layouts 1.0;
import QtQuick.Dialogs 1.0;
import QtQuick.Window 2.1;
import QtQuick.Controls.Styles 1.1
import Ethereum 1.0

Rectangle {
	id: root
	property var title: "Block Chain"
	property var menuItem

	objectName: "chainView"
	visible: false
	anchors.fill: parent

	TableView {
		id: blockTable
		width: parent.width
		anchors.top: parent.top
		anchors.bottom: parent.bottom
		TableViewColumn{ role: "number" ; title: "#" ; width: 100 }
		TableViewColumn{ role: "hash" ; title: "Hash" ; width: 560 }
		TableViewColumn{ role: "txAmount" ; title: "Tx amount" ; width: 100 }

		model: blockModel

		itemDelegate: Item {
			Text {
				anchors {
					left: parent.left
					right: parent.right
					leftMargin: 10
					verticalCenter: parent.verticalCenter
				}
				color: styleData.textColor
				elide: styleData.elideMode
				text: styleData.value
				font.pixelSize: 11
				MouseArea {
					acceptedButtons: Qt.LeftButton | Qt.RightButton
					propagateComposedEvents: true
					anchors.fill: parent
					onClicked: {
						blockTable.selection.clear()
						blockTable.selection.select(styleData.row)

						if(mouse.button == Qt.RightButton) {
							contextMenu.row = styleData.row;
							contextMenu.popup()
						}
					}

					onDoubleClicked: {
						popup.visible = true
						popup.setDetails(blockModel.get(styleData.row))
					}
				}
			}

		}

		Menu {
			id: contextMenu
			property var row
			MenuItem {
				text: "Details"
				onTriggered: {
					popup.visible = true
					popup.setDetails(blockModel.get(contextMenu.row))
				}
			}

			MenuSeparator{}

			MenuItem {
				text: "Copy"
				onTriggered: {
					copyToClipboard(blockModel.get(contextMenu.row).hash)
				}
			}

			MenuItem {
				text: "Dump State"
				onTriggered: {
					generalFileDialog.show(false, function(path) {
						var hash = blockModel.get(contextMenu.row).hash;

						gui.dumpState(hash, path);
					});
				}
			}
		}
	}



	function addBlock(block, initial) {
		if(initial == undefined){
			initial = false
		}

	       var amount = block.transactions.length;
	       var txs = [];
	       for(var i = 0; i < block.transactions.length; i++) {
		       var tx = JSON.parse(block.transactions.getAsJson(i));
		       txs.push(tx);
	       }

		if(initial){
			blockModel.append({raw: block.raw, bloom: block.bloom, size: block.size, number: block.number, name: block.name, gasLimit: block.gasLimit, gasUsed: block.gasUsed, coinbase: block.coinbase, hash: block.hash, txs: txs, txAmount: amount, time: block.time, prettyTime: convertToPretty(block.time)})
		} else {
			blockModel.insert(0, {raw: block.raw, bloom: block.bloom, size: block.size, number: block.number, name: block.name, gasLimit: block.gasLimit, gasUsed: block.gasUsed, coinbase: block.coinbase, hash: block.hash, txs: txs, txAmount: amount, time: block.time, prettyTime: convertToPretty(block.time)})
		}
	}

	Window {
		id: popup
		visible: false
		//flags: Qt.CustomizeWindowHint | Qt.Tool | Qt.WindowCloseButtonHint
		property var block
		width: root.width
		height: 300
		Component{
			id: blockDetailsDelegate
			Rectangle {
				color: "#252525"
				width: popup.width
				height: 150
				Column {
					anchors.leftMargin: 10
					anchors.topMargin: 5
					anchors.top: parent.top
					anchors.left: parent.left
					Text { text: '<h3>Block details</h3>'; color: "#F2F2F2"}
					Text { text: '<b>Block number:</b> ' + number + " (Size: " + size + ")"; color: "#F2F2F2"}
					Text { text: '<b>Hash:</b> ' + hash; color: "#F2F2F2"}
					Text { text: '<b>Bloom:</b> ' + bloom; color: "#F2F2F2"}
					Text { text: '<b>Coinbase:</b> &lt;' + name + '&gt; ' + coinbase; color: "#F2F2F2"}
					Text { text: '<b>Block found at:</b> ' + prettyTime; color: "#F2F2F2"}
					Text { text: '<b>Gas used:</b> ' + gasUsed + " / " + gasLimit; color: "#F2F2F2"}
				}
			}
		}
		ListView {
			model: singleBlock
			delegate: blockDetailsDelegate
			anchors.top: parent.top
			height: 100
			anchors.leftMargin: 20
			id: listViewThing
			Layout.maximumHeight: 40
		}
		TableView {
			id: txView
			anchors.top: listViewThing.bottom
			anchors.topMargin: 50
			width: parent.width

			TableViewColumn{width: 90; role: "value" ; title: "Value" }
			TableViewColumn{width: 200; role: "hash" ; title: "Hash" }
			TableViewColumn{width: 200; role: "sender" ; title: "Sender" }
			TableViewColumn{width: 200;role: "address" ; title: "Receiver" }
			TableViewColumn{width: 60; role: "gas" ; title: "Gas" }
			TableViewColumn{width: 60; role: "gasPrice" ; title: "Gas Price" }
			TableViewColumn{width: 60; role: "isContract" ; title: "Contract" }

			model: transactionModel
			onClicked: {
				var tx = transactionModel.get(row)
				if(tx.data) {
					popup.showContractData(tx)
				}else{
					popup.height = 440
				}
			}
		}

		function showContractData(tx) {
			if(tx.createsContract) {
				contractData.text = tx.data
				contractLabel.text = "<h4> Transaction created contract " + tx.address + "</h4>"
			}else{
				contractLabel.text = "<h4> Transaction ran contract " + tx.address + "</h4>"
				contractData.text = tx.rawData
			}
			popup.height = 540
		}

		Rectangle {
			id: txDetails
			width: popup.width
			height: 300
			anchors.left: listViewThing.left
			anchors.top: txView.bottom
			Label {
				text: "<h4>Contract data</h4>"
				anchors.top: parent.top
				anchors.left: parent.left
				id: contractLabel
				anchors.leftMargin: 10
			}
			TextArea {
				id: contractData
				text: "Contract"
				anchors.top: contractLabel.bottom
				anchors.left: parent.left
				anchors.right: parent.right
				wrapMode: Text.Wrap
				height: 80
			}
			TextArea {
				id: dumpData
				anchors.top: contractData.bottom
				anchors.left: parent.left
				anchors.right: parent.right
				anchors.bottom: parent.bottom
				height: 300
			}
		}
		property var transactionModel: ListModel {
			id: transactionModel
		}
		property var singleBlock: ListModel {
			id: singleBlock
		}
		function setDetails(bl){
			singleBlock.set(0, bl)
			popup.height = 300
			transactionModel.clear()
			if(bl.txs !== undefined){
				for(var i = 0; i < bl.txs.count; i++) {
					transactionModel.insert(0, bl.txs.get(i))
				}
				if(bl.txs.count > 0 && bl.txs.get(0).data){
					popup.showContractData(bl.txs.get(0))
				}
			}
			txView.forceActiveFocus()
			dumpData.text = bl.raw;
		}
	}
}
