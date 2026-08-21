import onnxruntime as ort
import json

def check_onnx():
    onnx_path = r'd:\ImageSeg\lightly-train\out\A701_AVI_wpoint_detect_ltdetrv2_l_coco\exported_models\exported_best.onnx'
    sess = ort.InferenceSession(onnx_path)
    print("Inputs:")
    for i in sess.get_inputs():
        print(f"  {i.name}: {i.shape}")
        
    print("\nOutputs:")
    for o in sess.get_outputs():
        print(f"  {o.name}: {o.shape}")

if __name__ == "__main__":
    check_onnx()
