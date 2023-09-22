# template

python export_onnx.py \
  --img_size 512 \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --dynamic


python export_onnx.py --img_size 512 --extractor_type superpoint --extractor_path weights/superpoint.onnx --lightglue_path weights/superpoint_lightglue.onnx --dynamic