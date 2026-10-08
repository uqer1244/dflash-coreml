import Foundation
import CoreML
import Darwin
let config = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as! [String:Any]
let directory = config["directory"] as! String
let options = MLModelConfiguration(); options.computeUnits = .cpuAndNeuralEngine
let packages = config["packages"] as! [String]
var models: [MLModel] = []
for (i,path) in packages.enumerated() {
    let compiled = try MLModel.compileModel(at: URL(fileURLWithPath:path))
    let dest = URL(fileURLWithPath:directory+"/shard-\(i).mlmodelc")
    if !FileManager.default.fileExists(atPath:dest.path) { try FileManager.default.moveItem(at:compiled,to:dest) }
    models.append(try MLModel(contentsOf:dest,configuration:options))
}
var commitModels:[MLModel]=[]
for (i,path) in (config["context_packages"] as? [String] ?? []).enumerated() {
 let dest=URL(fileURLWithPath:directory+"/commit-\(i).mlmodelc")
 let compiled=try MLModel.compileModel(at:URL(fileURLWithPath:path))
 try FileManager.default.moveItem(at:compiled,to:dest)
 commitModels.append(try MLModel(contentsOf:dest,configuration:options))
}
let fd = open(config["buffer"] as! String,O_RDWR)
let length = config["buffer_bytes"] as! Int
let pointer = mmap(nil,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0)!
precondition(pointer != MAP_FAILED)
defer { munmap(pointer,length); close(fd) }
let tensors = config["tensors"] as! [String:[String:Any]]
var inputs: [String:MLFeatureValue] = [:]
for (name,item) in tensors where name != "output" {
    let shape=item["shape"] as! [Int]
    let offset=item["offset"] as! Int
    precondition(shape.count==4 && shape.allSatisfy{$0>0} && offset>=0 && offset%2==0 && offset+shape.reduce(1,*)*2<=length)
    var strides=[Int](repeating:1,count:shape.count)
    if shape.count>1 { for i in stride(from:shape.count-2,through:0,by:-1) { strides[i]=strides[i+1]*shape[i+1] } }
    let array=try MLMultiArray(dataPointer:pointer.advanced(by:item["offset"] as! Int),shape:shape.map{NSNumber(value:$0)},dataType:.float16,strides:strides.map{NSNumber(value:$0)},deallocator:nil)
    inputs[name]=MLFeatureValue(multiArray:array)
}
var states=models.map{$0.makeState()}; var nextPosition=0
func now()->UInt64 { DispatchTime.now().uptimeNanoseconds }
func send(_ obj:[String:Any]) {
    let data=try! JSONSerialization.data(withJSONObject:obj,options:[.sortedKeys])
    print(String(data:data,encoding:.utf8)!);fflush(stdout)
}
send(["ready":true,"pid":getpid(),"compute_units":"CPU_AND_NE"])
while let line=readLine() {
    let request=try JSONSerialization.jsonObject(with:Data(line.utf8)) as! [String:Any]
    let command=request["command"] as! String
    if command=="quit" { break }
    if command=="reset" {
        states=models.map{$0.makeState()};nextPosition=request["position"] as! Int
        send(["reset":true]);continue
    }
    precondition((command=="predict" || command=="commit") && request["position"] as! Int == nextPosition)
    let committing=command=="commit"
    let activeModels=committing ? commitModels : models
    precondition(activeModels.count==3)
    var feed=inputs;var provider=try MLDictionaryFeatureProvider(dictionary:feed)
    var hidden:MLMultiArray!;var shardMS:[Double]=[];var handoffMS:[Double]=[];var same=true
    let start=now()
    for (i,model) in activeModels.enumerated() {
        provider=try MLDictionaryFeatureProvider(dictionary:feed.filter { model.modelDescription.inputDescriptionsByName[$0.key] != nil })
        let t=now();let result=try model.prediction(from:provider,using:states[i])
        if !committing {hidden=result.featureValue(for:"hidden")!.multiArrayValue!}
        shardMS.append(Double(now()-t)/1e6)
        if i<models.count-1 {
            let t=now();if !committing {feed["x"]=MLFeatureValue(multiArray:hidden)}
            if let context=result.featureValue(for:"fused_context") { feed["context_input"]=context }
            provider=try MLDictionaryFeatureProvider(dictionary:feed)
            if !committing {
            let handed=provider.featureValue(for:"x")!.multiArrayValue!
            same = same && handed === hidden && handed.dataPointer == hidden.dataPointer && hidden.dataType == .float16
            }
            handoffMS.append(Double(now()-t)/1e6)
        }
    }
    let outputStart=now();var finite=true
    if !committing && (request["output"] as? Bool ?? true) {
        let outputOffset=tensors["output"]!["offset"] as! Int
        precondition(outputOffset>=0 && outputOffset%2==0 && outputOffset+30720*2<=length)
        let dest=pointer.advanced(by:outputOffset).assumingMemoryBound(to:Float16.self)
        precondition(hidden.dataType == .float16 && hidden.shape.map{$0.intValue} == [1,5120,1,6])
        let source=hidden.dataPointer.assumingMemoryBound(to:Float16.self)
        let strideC=hidden.strides[1].intValue,strideS=hidden.strides[3].intValue
        precondition(strideC>0 && strideS>0 && hidden.strides.count==4)
        for c in 0..<5120 { for s in 0..<6 {
            let index=c*strideC+s*strideS
            let value=source[index]
            finite = finite && value.isFinite;dest[c*6+s]=value
        } }
    }
    nextPosition += 1
    send(["finite":finite,"next_position":nextPosition,"same_object_handoff":same,"shard_ms":shardMS,"handoff_ms":handoffMS,"output_copy_ms":Double(now()-outputStart)/1e6,"native_total_ms":Double(now()-start)/1e6,"context_only":committing])
}
