#this instruction is for specific contract "Default Contract"
# contract DefaultContract{ 
#  mapping(address=>uint) account;

#  //other code
#  ....
 
#  //classify type of picture
#  function animalClassification(address input, address model){
#   //get infer result
#   var result = keccak256(infer(input, model));

#   //reward according to your choice
#   if (result == keccak256("bird"))
#    account[msg.sender] += 10
#     }
#  //other code
#  ....
# }
def Run(account,tx,CVM):
    tx["amount"] = 0
    tx["comment"] = CVM.inference("./model_bind/"+tx["model"], "./input_data/"+tx["input"])
    print tx
    if tx["comment"] == "bird":
        account[tx["from"]]+=10
        tx["amount"] = 10
