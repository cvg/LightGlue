# template
python run_Lightglue_onnx.py \
  --inputdir0 assets/DSC_0410.JPG \
  --inputdir1 assets/DSC_0411.JPG \
  --img-size 512 \
  --lightglue-path weights/superpoint_lightglue.onnx \
  --extractor-type superpoint \
  --extractor-path weights/superpoint.onnx \
  --viz

python run_Lightglue_onnx.py --inputdir0 D:\\OroChiLab\\LightGlue\\data\\dir0\\DSC_0411.JPG --inputdir1 D:\\OroChiLab\\LightGlue\\data\\dir1\\DSC_0410.JPG --img-size 512 --lightglue-path D:\\OroChiLab\\LightGlue\\weights\\onnx\\superpoint_lightglue.onnx --extractor-type superpoint --extractor-path D:\\OroChiLab\\LightGlue\\weights\\onnx\\superpoint.onnx --viz